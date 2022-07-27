import random
import sys
import os
import matplotlib.pyplot as plt
from pathlib import Path
from glob import glob
from PIL import Image

# Torch
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T


class DermXAttnDataset(Dataset):
    """Container for doing Dx, Cx and GradCAM prediction on DermX dataset."""

    def __init__(
        self,
        df_label_map,
        class_map_dx,
        class_map_char,
        image_dir="images",
        mask_dir="masks",
        mask_label_fusion_rule="any_agreement",
        mask_downsample_size=None,
        rgb_transforms=None,
        shared_transforms=None,
    ):
        """
        Initialize the DermXAttnDataset.

        Args:
        ----
        df_label_map: Assumes that the Dx label column is named `"diagnosis"`
            and that the image path column is named "current_filename".
        class_map_dx: Dict mapping Dx class string to int.
        class_map_char: Dict mapping segmentation class string to int.
        image_dir: Directory with input images.
        mask_dir: Directory with the segmentation masks in png format.
        mask_label_fusion_rule: Which label fusion rule to apply to the masks.
            Can be one of "full_agreement", "majority_agreement",
            "any_agreement", "fuzzy_fixed", or "fuzzy_relative".
        rgb_transforms: torchvision.transform, or a list of them, operating on
            tensors.
        shared_transforms: List of torchvision.transforms, operating on tensors.
        mask_downsample_size: int or tuple of ints indicating the size ground
            truth masks should be downsampled to using bilinear interpolation.
            This will be done after applying any `shared_transforms`. If None,
            the masks will not be downsampled, and will be returned as
            determined by `shared_transforms`.
        """
        # Indexing
        self.df_label_map = df_label_map
        self.class_map_dx = class_map_dx
        self.class_map_char = class_map_char
        self.inv_class_map_dx = {val: key for key, val in self.class_map_dx.items()}
        self.inv_class_map_char = {val: key for key, val in self.class_map_char.items()}
        self._filename_col = "current_filename"
        self._dx_label_col = "diagnosis"

        # IO
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.mask_label_fusion_rule = mask_label_fusion_rule
        if len(glob(os.path.join(self.image_dir, "*"))) == 0:
            raise IOError(
                f"Image directory '{self.image_dir}' is empty or does not exist."
            )
        if len(glob(os.path.join(self.mask_dir, "*.png"))) == 0:
            raise IOError(
                f"Mask directory '{self.mask_dir}' contains no pngs or does not exist."
            )

        # Transforms
        if isinstance(rgb_transforms, list):
            rgb_transforms = T.Compose(rgb_transforms)
        self.rgb_transforms = rgb_transforms
        self.shared_transforms = shared_transforms
        if isinstance(mask_downsample_size, int):
            self.mask_downsample_size = (mask_downsample_size, mask_downsample_size)
        else:
            self.mask_downsample_size = mask_downsample_size

    def __len__(self):
        return len(self.df_label_map)

    def __getitem__(self, idx):
        # This seed is created in order to apply the same transforms to both image and masks
        seed = random.randrange(sys.maxsize)

        # Image
        image_id = str(Path(self.df_label_map.loc[idx, self._filename_col]).stem)
        image_path = os.path.join(
            self.image_dir, self.df_label_map.loc[idx, self._filename_col]
        )
        image_as_pil = Image.open(image_path)
        image = T.ToTensor()(image_as_pil)
        if self.rgb_transforms is not None:
            image = self.rgb_transforms(image)
        image = self._apply_shared_transforms(image, seed, overwrite_intp_with=2)

        # Dx label
        label = self.class_map_dx[self.df_label_map.loc[idx, self._dx_label_col]]

        # Cx label and segmentation masks
        label_cx = torch.tensor([0.0] * len(self.class_map_char))
        masks = torch.zeros((len(self.class_map_char),) + image.shape[-2:])
        mask_path_glob_pattern = f"{self.mask_label_fusion_rule}_{image_id}_**.png"
        mask_paths = glob(os.path.join(self.mask_dir, mask_path_glob_pattern))
        for mask_path in mask_paths:
            char_name = str(Path(mask_path.rsplit("_", 1)[1]).stem)
            if char_name in self.class_map_char.keys():
                char_id = self.class_map_char[char_name]
                label_cx[char_id] = 1.0
                mask_as_pil = Image.open(mask_path)
                mask = T.ToTensor()(mask_as_pil)
                mask = self._apply_shared_transforms(mask, seed, overwrite_intp_with=0)
                masks[char_id] = mask

        if self.mask_downsample_size is not None:
            masks = T.Resize(self.mask_downsample_size)(masks)

        return image, label, label_cx, masks, image_path

    def _apply_shared_transforms(self, image, seed, overwrite_intp_with=None):
        """
        Apply transforms in way that makes it possible to reuse them again later.

        This function solves the problem of applying the same random (e.g. as in
        data augmentation) torchvision.transforms to mutiple images, by setting
        the torch random seed manually through an input variable.

        Args:
        ----
        image: Tensor to apply the transforms on.
        seed: Seed to feed to torch.manual_seed, fixing the transformations, so
            the exact same transformations can be applied on a later function
            call.
        overwrite_intp_with: int specifying which PIL.Image interpolation method
            to use, for transformations using interpolation (e.g. rotations).
            0 is PIL.Image.NEAREST, while 2 is PIL.Image.BILINEAR.
        """
        torch.manual_seed(seed)
        intp_dep_transf_resample = (T.RandomAffine, T.RandomRotation)
        intp_dep_transf_interpolation = (
            T.RandomPerspective,
            T.Resize,
            T.RandomResizedCrop,
        )
        for tr in self.shared_transforms:
            if overwrite_intp_with is not None:
                if isinstance(tr, intp_dep_transf_resample):
                    tr.resample = overwrite_intp_with
                elif isinstance(tr, intp_dep_transf_interpolation):
                    tr.interpolation = overwrite_intp_with
            image = tr(image)

        return image


def get_denormalize_transform(
    mean_to_denorm=[0.0, 0.0, 0.0], std_to_denorm=[1.0, 1.0, 1.0]
):
    """Reverse a standard normalization transform."""
    mean_to_denorm = torch.tensor(mean_to_denorm, dtype=torch.float32)
    std_to_denorm = torch.tensor(std_to_denorm, dtype=torch.float32)
    denormalize = T.Normalize(
        (-mean_to_denorm / std_to_denorm).tolist(), (1.0 / std_to_denorm).tolist()
    )
    return denormalize
