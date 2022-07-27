import torch
from torch.nn import functional as F
from torchvision.models import segmentation
import segmentation_models_pytorch as smp
import timm
from captum.attr import LayerGradCam
import os


class DermXModelGuidedAttention(torch.nn.Module):
    """
    Dx and Cx classification model with GradCAM to be used with DermX data.
    """

    def __init__(
        self,
        num_dx_classes,
        num_cx_classes,
        target_layer,
        backbone_name="efficientnet_b0",
        pretrained=True,
        return_gradcam_attributes=True,
        interpolate_gradcam_attributes=True,
        return_cx_only=False,
        num_units_in_dx_clf=32,
        num_units_in_cx_clf=None,
        dropout=0.2,
        use_batchnorm=False,
        use_skip_connection=True,
        device="cpu",
    ):
        """
        Initialize the DermXModelGuidedAttention.

        Args:
        ----
        num_dx_classes: Number of outputs to use in the Dx head.
        num_cx_classes: Number of outputs to use in the Cx head.
        target_layer: Name of layer for which to return guided attention attributions.
        backbone_name: Name of the backbone model that should be used.
        pretrained: Whether to use backbone weights pretrained on ImageNet.
        return_gradcam_attributes: Whether to return GradCAM attributes.
        interpolate_gradcam_attributes: Whether to interpolate the GradCAM
            attributes to match the input image size.
        return_cx_only: Whether to only return the Cx head; useful for getting
            Cx GradCAMs.
        num_units_in_dx_clf: Number of features to concatenate to the Cx features vector,
            or - if `use_skip_connection` is False - number of units in an additional
            linear layer for the Dx classifier head. If None, add no extra linear layer.
        num_units_in_cx_clf: Number of units in an additional linear layer for the Cx
            classifier head. If None, add no extra linear layer.
        dropout: Dropout rate to use in the Dx and Cx heads.
        use_batchnorm: Whether to use BatchNorm in the Dx and Cx heads.
        use_skip_connection: Whether to concatenate the Cx onto the Dx head,
            and use that as input to the Dx output layer.
        device: Device to compute on.
        """
        super(DermXModelGuidedAttention, self).__init__()
        self.num_dx_classes = num_dx_classes
        self.num_cx_classes = num_cx_classes
        self.backbone_name = backbone_name
        self.pretrained = pretrained
        self.return_gradcam_attributes = return_gradcam_attributes
        self.interpolate_gradcam_attributes = interpolate_gradcam_attributes
        self.return_cx_only = return_cx_only
        self.num_units_in_dx_clf = num_units_in_dx_clf
        self.num_units_in_cx_clf = num_units_in_cx_clf
        self.dropout = dropout
        self.use_batchnorm = use_batchnorm
        self.use_skip_connection = use_skip_connection

        self.backbone = timm.create_model(
            backbone_name, pretrained=True, features_only=True,
        )
        self.flat_gap = torch.nn.Sequential(
            torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten()
        )
        self.num_enc_features = self.backbone.feature_info.channels()[-1]

        def _get_head_layers(num_outputs, num_additional_units=None):
            if num_additional_units is not None:
                layers = [
                    torch.nn.Dropout(self.dropout),
                    torch.nn.Linear(self.num_enc_features, num_additional_units),
                    torch.nn.ReLU(),
                ]
                if self.use_batchnorm:
                    layers += [torch.nn.BatchNorm1d(num_additional_units)]
                layers += [
                    torch.nn.Dropout(self.dropout),
                    torch.nn.Linear(num_additional_units, num_outputs),
                ]
            else:
                layers = [
                    torch.nn.Dropout(self.dropout),
                    torch.nn.Linear(self.num_enc_features, num_outputs),
                ]
            return layers

        # Dx head
        if self.use_skip_connection:
            dx_clf_prep_layers = [
                torch.nn.Dropout(self.dropout),
                torch.nn.Linear(self.num_enc_features, self.num_units_in_dx_clf),
                torch.nn.ReLU(),
            ]
            if self.use_batchnorm:
                dx_clf_prep_layers += [torch.nn.BatchNorm1d(self.num_units_in_dx_clf)]
            dx_clf_prep_layers += [torch.nn.Dropout(self.dropout)]
            self.dx_clf_prep = torch.nn.Sequential(*dx_clf_prep_layers)
            self.dx_clf = torch.nn.Linear(
                self.num_units_in_dx_clf + self.num_cx_classes, self.num_dx_classes
            )
        else:
            dx_clf_layers = _get_head_layers(
                self.num_dx_classes, self.num_units_in_dx_clf
            )
            self.dx_clf = torch.nn.Sequential(*dx_clf_layers)

        # Cx head
        cx_clf_layers = _get_head_layers(self.num_cx_classes, self.num_units_in_cx_clf)
        self.cx_clf = torch.nn.Sequential(*cx_clf_layers)

        # GradCAM
        if self.return_gradcam_attributes:
            self.target_layer = target_layer
            self.device = device
            self.fmap = None
            self.grad = None
            self.handlers = []

            def save_fmaps(key):
                def hook(module, fmap_in, fmap_out):
                    if isinstance(fmap_out, list):
                        fmap_out = fmap_out[-1]
                    self.fmap = fmap_out

                return hook

            def save_grads(key):
                def hook(module, grad_in, grad_out):
                    self.grad = grad_out[0]

                return hook

            target_module = dict(self.named_modules())[self.target_layer]
            self.handlers.append(
                target_module.register_forward_hook(save_fmaps(self.target_layer))
            )
            self.handlers.append(
                target_module.register_backward_hook(save_grads(self.target_layer))
            )

    def forward(self, x):
        features = self.flat_gap(self.backbone(x)[-1])
        cx = self.cx_clf(features)
        if self.return_cx_only:
            return cx
        if self.use_skip_connection:
            dx = self.dx_clf_prep(features)
            dx = torch.cat((dx, cx), 1)
            dx = self.dx_clf(dx)
        else:
            dx = self.dx_clf(features)

        # GradCAM
        if self.return_gradcam_attributes:
            attributes = tuple(
                self.attribute(cx, cx_idx) for cx_idx in range(self.num_cx_classes)
            )
            attributes = torch.cat(attributes, dim=1)
            if self.interpolate_gradcam_attributes:
                attributes = F.interpolate(
                    attributes, size=x.size()[-2:], mode="bilinear", align_corners=False
                )

            # Set all not-predicted Cx classes to zero
            cx_classes = torch.round(torch.sigmoid(cx))
            batch_idx, cx_idx = torch.nonzero(cx_classes == 0, as_tuple=True)
            attributes[batch_idx, cx_idx, ::] = 0.0

            return dx, cx, attributes
        else:
            return dx, cx

    def attribute(self, logits, class_idx):
        """The creation of logits updates fmaps, while the logits.backward call updates grads."""
        self.zero_grad()
        one_hot = F.one_hot(
            (torch.ones(logits.size(0)) * class_idx).long(), self.num_cx_classes
        ).to(self.device)
        # Two things to note about the following line:
        # 1. retrain_graph=True effectively doubles the memory footprint, see https://pytorch.org/docs/stable/generated/torch.Tensor.backward.html
        # 2. zero_grad() should also be called after the forward pass of the model (but of course before the loss.backward call), since gradients are accumulated here
        logits.backward(gradient=one_hot, retain_graph=True)
        weights = F.adaptive_avg_pool2d(self.grad, 1)
        attr = torch.mul(self.fmap, weights).sum(dim=1, keepdim=True)
        attr = F.relu(attr)

        # [0,1]-normalization - might cause NaNs if the classifier(s) are not yet well-trained
        B, C, H, W = attr.shape
        attr = attr.view(B, -1)
        attr -= attr.min(dim=1, keepdim=True)[0]
        attr /= attr.max(dim=1, keepdim=True)[0] + 1e-7
        attr = attr.view(B, C, H, W)

        return attr

    def remove_hook(self):
        """
        Remove all the forward/backward hook functions
        """
        for handle in self.handlers:
            handle.remove()
