import torch


class DiceLoss(torch.nn.Module):
    """Dice loss for semantic segmentation."""

    def __init__(self, weight=None, smooth=1.0, activation=torch.sigmoid):
        """
        Initialize the loss.

        Args:
        ----
        smooth: Smoothing factor to use in both numerator and denominator.
        activation: Torch callable that the output should first be passed
            through. Set to None to not use any activation.
        """
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.activation = activation

    def forward(self, outputs, targets):
        if self.activation is not None:
            outputs = self.activation(outputs)
        outputs = outputs.view(-1)
        targets = targets.view(-1)
        intersection = (outputs * targets).sum()
        reference = outputs.sum() + targets.sum()
        dice = (2.0 * intersection + self.smooth) / (reference + self.smooth)
        return 1 - dice


class SoftSeSpLoss(torch.nn.Module):
    """
    Soft and weighted sensitivity-specificity loss for semantic segmentation.
    """

    def __init__(
        self, weight=None, alpha=0.5, smooth=1.0, activation=torch.sigmoid,
    ):
        """
        Initialize the loss.

        Args:
        ----
        alpha: Float between 0 and 1, determining how much weight should be put
            on the sensitivity part of the loss, while the specificity part will
            be weighted with (1 - alpha).
        smooth: Smoothing factor to use in both numerator and denominator.
        activation: Torch callable that the output should first be passed
            through. Set to None to not use any activation.
        """
        super(SoftSeSpLoss, self).__init__()
        self.alpha = alpha
        self.smooth = smooth
        self.activation = activation

    def forward(self, outputs, targets):
        if self.activation is not None:
            outputs = self.activation(outputs)
        outputs = outputs.view(-1)
        targets = targets.view(-1)
        tp = torch.clamp(targets * outputs, 0, 1).sum()
        tn = torch.clamp((1 - outputs) * (1 - targets), 0, 1).sum()
        p = targets.sum()
        n = (1 - targets).sum()
        sensitivity = (tp + self.smooth) / (p + self.smooth)
        specificity = (tn + self.smooth) / (n + self.smooth)
        weighted_score = self.alpha * sensitivity + (1 - self.alpha) * specificity
        return 1 - weighted_score


class MSEOverActiveChannelsLoss(torch.nn.Module):
    """Calculate the MSE but only dividing by the # non-zero channels."""

    def __init__(self, weight=None, activation=torch.sigmoid):
        """
        Initialize the loss.

        Args:
        ----
        activation: Torch callable that the output should first be passed
            through. Set to None to not use any activation.
        """
        super(MSEOverActiveChannelsLoss, self).__init__()
        self.activation = activation

    def forward(self, outputs, targets):
        if self.activation is not None:
            outputs = self.activation(outputs)
        num_nonzero_target_channels = (targets.sum(dim=(-2, -1)) > 0.0).sum(
            dim=1
        )  # alternative to "> 0.0": torch.clamp(x, 0, 1)
        outputs = outputs.view(-1)
        targets = targets.view(-1)
        summed_sq_diff = ((outputs - targets) ** 2).sum()
        return (summed_sq_diff / num_nonzero_target_channels).float().mean()


class MSEOverChannelsLoss(torch.nn.Module):
    """Calculate the MSE dividing by the number of channels."""

    def __init__(self, weight=None, activation=torch.sigmoid):
        """
        Initialize the loss.

        Args:
        ----
        activation: Torch callable that the output should first be passed
            through. Set to None to not use any activation.
        """
        super(MSEOverChannelsLoss, self).__init__()
        self.activation = activation

    def forward(self, outputs, targets):
        if self.activation is not None:
            outputs = self.activation(outputs)
        summed_sq_diff = ((outputs - targets) ** 2).sum(dim=(-2, -1))
        return summed_sq_diff.mean(dim=1).mean()  # avg over channels, then batch


str_loss_map = {
    "ce": torch.nn.CrossEntropyLoss,
    "bce": torch.nn.BCEWithLogitsLoss,
    "dice": DiceLoss,
    "softsesp": SoftSeSpLoss,
    "mse_over_active_channels": MSEOverActiveChannelsLoss,
    "mse_over_channels": MSEOverChannelsLoss,
}
