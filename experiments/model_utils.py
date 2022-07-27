import torch


def freeze_layers(model, ignore_names=None, ignore_bn=True):
    """
    Freezes everything, and then unfreezes selected layers.

    Args:
    ----
    model: Torch model to freeze.
    ignore_names: List of layer or module names to leave unfrozen. Names can be
        obtained with model.named_modules(). Example:
        ["backbone.layer4.0.conv1"] will unfreeze only that specific layer,
        while ["backbone.layer4"] while unfreeze everything in that submodule.
    ignore_bn: Bool specifying whether Batch Norm layers should all be left
        unfrozen.
    """

    def _set_requires_grad(layer, set_to=False):
        for param in layer.parameters():
            param.requires_grad = set_to

    # Freeze everything
    for name, layer in model.named_modules():
        _set_requires_grad(layer, False)

    # Unfreeze anything batch norm related
    if ignore_bn:
        for name, layer in model.named_modules():
            if isinstance(layer, torch.nn.modules.batchnorm.BatchNorm2d):
                _set_requires_grad(layer, True)

    # Unfreeze anything in the ignore list
    if ignore_names is not None:
        for name, layer in model.named_modules():
            if name in ignore_names:
                _set_requires_grad(layer, True)
