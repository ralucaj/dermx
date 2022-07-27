import torch
from time import time
from tqdm import tqdm
from torchvision.utils import make_grid
from sklearn.metrics import f1_score


def iterate_through_dataloader(
    dataloader,
    model,
    optimizer,
    criterion_dx,
    criterion_segm,
    epoch,
    n_total_epochs,
    metric_names,
    train_mode=True,
    scheduler=None,
    device="cpu",
    dx_loss_factor=1.0,
    segm_loss_factor=1.0,
    write_segm_examples_to_tb=False,
    tb_writer_segm_example_kwargs={},
):
    """
    Iterate once through a dataloader.

    Args:
    ----
    dataloader: torch.utils.data.DataLoader instance to iterate through.
    model: torch.nn.Module instance to predict or train with.
    optimizer: torch.optim instance.
    criterion_dx: Torch loss function to use for training the Dx task.
    criterion_segm: Torch loss function to use for training the segmentation
        task.
    epoch: The current epoch as an int.
    n_total_epochs: The total number of expected epochs as an int.
    metric_names: List of strings with the names of the metrics that should be
        collected.
    train_mode: bool setting whether to use training model (model.train() with
        grads) or inference (model.eval() and no grads).
    scheduler: torch.optim.lr_scheduler instance.
    device: String or torch.Device instance.
    dx_loss_factor: How much to weigh the Dx loss during training.
    segm_loss_factor: How much to weigh the segmentation loss during training.
    write_segm_examples_to_tb: Whether examples of segmentations should be
        written to torch.utils.tensorboard.SummaryWriter object passed through
        the tb_writer_segm_example_kwargs argument at the end of the epoch.
    tb_writer_segm_example_kwargs: Dict with keywords to pass to internal
        function for writing examples of segmentations. Is ignored if
        write_segm_examples_to_tb is False. Most important kwargs to pass are
        'writer' (a torch.utils.tensorboard.SummaryWriter instance), 'device',
        and 'denormalizer' (an instantiated torchvision.transform to pass the
        image returned from the dataloader through before plotting).
    Return:
    ------
    running_metrics: Losses and metrics averaged over the iteration of the
        dataloader.
    """

    def _process_batch(img_batch, label_batch, mask_batch):
        metrics = {name: 0.0 for name in metric_names}
        img_batch = img_batch.to(device)
        label_batch = label_batch.to(device)
        mask_batch = mask_batch.to(device)

        if train_mode:
            optimizer.zero_grad()

        output_dx, output_segm = model(img_batch)

        # Losses
        loss_dx = criterion_dx(output_dx, label_batch)
        loss_segm = criterion_segm(output_segm, mask_batch)
        loss = dx_loss_factor * loss_dx + segm_loss_factor * loss_segm

        if train_mode:
            loss.backward()
            optimizer.step()

        # Metrics
        metrics["loss_dx"] = loss_dx.item()
        metrics["loss_segm"] = loss_segm.item()
        metrics["loss"] = loss.item()
        output_class_dx = torch.argmax(output_dx, dim=1)
        metrics["dx_acc"] = (
            output_class_dx == label_batch
        ).sum().item() / label_batch.size(0)
        metrics["segm_iou"] = iou_metric(
            torch.sigmoid(output_segm) > 0.5, mask_batch > 0.0
        ).item()

        return output_dx, output_segm, metrics

    if write_segm_examples_to_tb and "writer" not in tb_writer_segm_example_kwargs:
        raise KeyError(
            "When 'write_segm_examples_to_tb' is True, "
            "'tb_writer_segm_example_kwargs' must at least contain a "
            "tensorboard SummaryWriter through the 'writer' key."
        )

    running_metrics = {name: 0.0 for name in metric_names}
    model.train() if train_mode else model.eval()
    iterator = (
        tqdm(dataloader, desc=f"Epoch {epoch}/{n_total_epochs}")
        if train_mode
        else dataloader
    )

    with (torch.enable_grad() if train_mode else torch.no_grad()):
        for idx_batch, batch in enumerate(iterator):
            img_batch, label_batch, mask_batch, _ = batch
            output_dx, output_segm, batch_metrics = _process_batch(
                img_batch, label_batch, mask_batch
            )
            if write_segm_examples_to_tb and (idx_batch == len(dataloader) - 1):
                model.eval()
                with torch.no_grad():
                    write_segmentation_examples_to_tb(
                        batch,
                        output_segm,
                        epoch,
                        train_mode,
                        **tb_writer_segm_example_kwargs,
                    )
            # Accumulate losses and metrics
            for name in metric_names:
                running_metrics[name] += batch_metrics[name]

    if train_mode and scheduler is not None:
        scheduler.step()

    # Get mean of losses and metrics
    for name in metric_names:
        running_metrics[name] /= len(dataloader)

    return running_metrics


def iterate_through_attn_dataloader(
    dataloader,
    model,
    optimizer,
    criterion,
    epoch,
    n_total_epochs,
    metric_names,
    train_mode=True,
    force_enable_grad=False,
    scheduler=None,
    device="cpu",
    criterion_weight=1.0,
):
    """
    Iterate once through a dataloader.

    Args:
    ----
    dataloader: torch.utils.data.DataLoader instance to iterate through.
    model: torch.nn.Module instance to predict or train with.
    optimizer: torch.optim instance.
    criterion: Dict of loss function, mapping criterion names to callables. Its
        order and length must match that of the model output.
    epoch: The current epoch as an int.
    n_total_epochs: The total number of expected epochs as an int.
    metric_names: List of strings with the names of the metrics that should be
        collected.
    train_mode: bool setting whether to use training model (model.train() with
        grads) or inference (model.eval() and no grads).
    force_enable_grad: bool forcing torch.enable_grad() when doing the forward
        pass, but without updating optimizer and learnin rate scheduler state.
    scheduler: torch.optim.lr_scheduler instance.
    device: String or torch.Device instance.
    criterion_weight: Dict mapping criterion name to loss factor to be multipled
        with the loss associated with that name. Must have the same keys as
        `criterion`.

    Return:
    ------
    running_metrics: Losses and metrics averaged over the iteration of the
        dataloader.
    """

    def _process_batch(img_batch, target_batch, criterion, criterion_weight):
        metrics = {name: 0.0 for name in metric_names}
        target_batch = tuple(t_batch.to(device) for t_batch in target_batch)
        img_batch = img_batch.to(device)

        if train_mode:
            optimizer.zero_grad()
        outputs = model(img_batch)
        if train_mode:
            optimizer.zero_grad()

        # Losses
        if not isinstance(criterion, dict):
            criterion = {criterion.__name__: criterion}
            criterion_weight = {criterion.__name__: criterion_weight}
        losses = {
            loss_name: criterion[loss_name](outputs[loss_idx], target_batch[loss_idx])
            for loss_idx, loss_name in enumerate(criterion)
        }
        loss = 0
        for loss_name in criterion:
            loss += losses[loss_name] * criterion_weight[loss_name]

        if train_mode:
            loss.backward()
            optimizer.step()

        # Report losses
        for loss_name in criterion:
            metrics[loss_name] = losses[loss_name].item()
        metrics["loss"] = loss.item()

        # Report metrics
        output_dx = outputs[0]
        target_dx = target_batch[0]
        output_class_dx = torch.argmax(output_dx, dim=1)
        metrics["dx_acc"] = (output_class_dx == target_dx).float().mean(0).item()

        output_cx = outputs[1]
        target_cx = target_batch[1]
        output_class_cx = torch.round(torch.sigmoid(output_cx))
        metrics["cx_f1"] = f1_score_from_tensors(output_class_cx, target_cx)

        output_attr = outputs[2]
        target_attr = target_batch[2]
        metrics["soft_iou"] = soft_iou_metric(output_attr, target_attr)

        return outputs, metrics

    running_metrics = {name: 0.0 for name in metric_names}
    model.train() if train_mode else model.eval()
    iterator = (
        tqdm(dataloader, desc=f"Epoch {epoch}/{n_total_epochs}")
        if train_mode
        else dataloader
    )

    with (
        torch.enable_grad() if (train_mode or force_enable_grad) else torch.no_grad()
    ):
        for idx_batch, batch in enumerate(iterator):
            input_batch = batch[0]
            target_batch = batch[1:-1]
            path_batch = batch[-1]
            outputs, batch_metrics = _process_batch(
                input_batch, target_batch, criterion, criterion_weight
            )
            # Accumulate losses and metrics
            for name in metric_names:
                running_metrics[name] += batch_metrics[name]

    if train_mode and scheduler is not None:
        scheduler.step()

    # Get mean of losses and metrics
    for name in metric_names:
        running_metrics[name] /= len(dataloader)

    return running_metrics


def write_segmentation_examples_to_tb(
    batch, output_segm, epoch, train_mode, writer, denormalizer=None, device="cpu"
):
    """Write passed batch incl. predictions to TensorBoard as images."""
    output_masks_batch = torch.sigmoid(output_segm) > 0.5
    batch += (output_masks_batch,)
    for image, label, target_masks, image_path, output_masks in zip(*batch):
        tns_to_plot = []
        if denormalizer is not None:
            tns_to_plot.append(denormalizer(image).to(device))
        else:
            tns_to_plot.append(image.to(device))
        for target_mask, output_mask in zip(target_masks, output_masks):
            overlay_mask = (torch.zeros((3,) + target_mask.size())).to(device)
            overlay_mask[0] = target_mask
            overlay_mask[2] = output_mask
            tns_to_plot.append(overlay_mask)
        grid = make_grid(tns_to_plot, nrow=3)
        mode_string = "train" if train_mode else "test"
        stemmed_image_path = image_path.rsplit("/", 1)[1]
        name = f"{mode_string}/epoch:{str(epoch).zfill(4)}_path:{stemmed_image_path}"
        writer.add_image(name, grid, epoch)


def iou_metric(
    output, target, sum_dims=(-2, -1), epsilon=1e-7, per_characteristic=False
):
    """
    Compute the IoU between two torch tensor segmentation masks.

    Input tensors should be 4D boolean tensors.
    """
    output, target = output.float(), target.float()
    intersection = output * target
    union = (output + target) - intersection
    iou = (intersection.sum(dim=sum_dims)) / (union.sum(dim=sum_dims) + epsilon)
    return iou.mean(0) if per_characteristic else iou.mean()


def masked_mean(tns, mask, dim=None):
    """Return the mean over the non-masked elements."""
    dim = dim or tuple(range(tns.ndim))
    if mask.sum(dim=dim) == 0.0:
        return torch.tensor(0.0)
    masked = torch.mul(tns, mask)
    return masked.sum(dim=dim) / mask.sum(dim=dim)


def replace_nan_inplace(tns, replace_with=0.0):
    """Replace all NaNs inplace in passed tensor."""
    tns[torch.isnan(tns)] = replace_with


def soft_iou_metric(output, target, sum_dims=(-2, -1), mask_zero_channels=True):
    """
    Compute the soft/fuzzy IoU between two torch tensors.

    Input tensors should be 4D float tensors. If `mask_zero_channels` is True,
    only channels (the second dimension) that are nonzero in both the output and
    the target will count towards the returned mean IoU. Will replace all NaNs
    with 0.0.
    """
    intersection = torch.minimum(output, target).sum(dim=sum_dims)
    union = torch.maximum(output, target).sum(dim=sum_dims)
    iou_unreduced = intersection / union
    replace_nan_inplace(iou_unreduced)
    if mask_zero_channels:
        mask = (output * target).sum(dim=sum_dims) > 0.0
        iou = masked_mean(iou_unreduced, mask)
    else:
        iou = iou_unreduced.mean()
    return iou.item()


def f1_score_from_tensors(output, target, average="macro"):
    """F1 score on tensors converted to numpy arrays."""
    target_numpy = target.cpu().detach().numpy()
    output_numpy = output.cpu().detach().numpy()
    return f1_score(target_numpy, output_numpy, average=average)


def print_history_result(history, start_time=None, prec=3):
    """Print training history dict on epoch end."""
    report_string = f"Epoch {history['epoch'][-1]}: "
    report_string += " -- ".join(
        f"{report}: {history[report][-1]:.{prec}f}"
        for report in history
        if report.lower() != "epoch"
    )
    if start_time is not None:
        end_time = time() - start_time
        report_string += f" -- completed in {end_time / 60:.1f} mins"
    print(report_string)


# Enable getting objects using strings
str_optimizer_map = {
    "sgd": torch.optim.SGD,
    "adam": torch.optim.Adam,
    "adamw": torch.optim.AdamW,
}
str_scheduler_map = {
    "multi_step_lr": torch.optim.lr_scheduler.MultiStepLR,
    "reduce_lr_on_plateau": torch.optim.lr_scheduler.ReduceLROnPlateau,
    "cyclic_lr": torch.optim.lr_scheduler.CyclicLR,
    "one_cycle_lr": torch.optim.lr_scheduler.OneCycleLR,
    "cosine_annealing_warm_restarts": torch.optim.lr_scheduler.CosineAnnealingWarmRestarts,
}
