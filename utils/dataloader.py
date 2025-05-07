import os
import json
import torch
import numpy as np
import skimage.io as io
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import v2 as T
from torchvision.transforms import functional as F
from pycocotools import mask as maskUtils


class RandomHorizontalFlip(T.RandomHorizontalFlip):
    """
    Horizontal flip data augmentation, processes both images and instance
    segmentation annotations
    """
    def forward(self, image, target):
        """
        Horizontally flip the image and target

        Args:
            image: Input image
            target: Target dictionary containing boxes and masks

        Returns:
            Flipped image and target
        """
        if torch.rand(1) < self.p:
            image = F.hflip(image)
            if target is not None:
                _, _, width = F.get_dimensions(image)
                target["boxes"][:, [0, 2]] = width - target["boxes"][:, [2, 0]]
                target["masks"] = F.hflip(target["masks"])
        return image, target


class RandomVerticalFlip(T.RandomVerticalFlip):
    """
    Vertical flip data augmentation, processes both images and instance
    segmentation annotations
    """
    def forward(self, image, target):
        """
        Vertically flip the image and target

        Args:
            image: Input image
            target: Target dictionary containing boxes and masks

        Returns:
            Flipped image and target
        """
        if torch.rand(1) < self.p:
            image = F.vflip(image)
            if target is not None:
                _, height, _ = F.get_dimensions(image)
                target["boxes"][:, [1, 3]] = (
                    height - target["boxes"][:, [3, 1]]
                )
                target["masks"] = F.vflip(target["masks"])
        return image, target


class RandomRotation:
    """
    Random rotation data augmentation, processes both images and instance
    segmentation annotations
    """
    def __init__(self, degrees=10, p=0.5):
        self.degrees = degrees
        self.p = p

    def forward(self, image, target):
        """
        Randomly rotate the image and target

        Args:
            image: Input image
            target: Target dictionary containing boxes and masks

        Returns:
            Rotated image and target
        """
        if torch.rand(1) < self.p:
            angle = (
                torch.rand(1).item() * 2 * self.degrees - self.degrees
            )

            # Rotate image
            image = F.rotate(image, angle, expand=False)

            if target is not None:
                # Rotate masks
                target["masks"] = F.rotate(
                    target["masks"],
                    angle,
                    expand=False
                )

                # Recalculate bounding boxes from masks
                masks = target["masks"]
                boxes = []
                for mask in masks:
                    y, x = torch.where(mask > 0.5)
                    if len(y) > 0 and len(x) > 0:
                        x_min = x.min().item()
                        x_max = x.max().item()
                        y_min = y.min().item()
                        y_max = y.max().item()
                        boxes.append([x_min, y_min, x_max, y_max])
                    else:
                        # If mask is empty, use original box
                        boxes.append([0, 0, 1, 1])

                if boxes:
                    target["boxes"] = torch.tensor(
                        boxes,
                        dtype=torch.float32
                    )

        return image, target


class RandomScale:
    """
    Random scale data augmentation, processes both images and instance
    segmentation annotations
    """
    def __init__(self, scale_factors=(0.8, 1.2), p=0.5):
        self.scale_factors = scale_factors
        self.p = p

    def forward(self, image, target):
        """
        Randomly scale the image and target

        Args:
            image: Input image
            target: Target dictionary containing boxes and masks

        Returns:
            Scaled image and target
        """
        if torch.rand(1) < self.p:
            scale_factor = (
                torch.rand(1).item() *
                (self.scale_factors[1] - self.scale_factors[0]) +
                self.scale_factors[0]
            )

            # Get original dimensions
            _, height, width = F.get_dimensions(image)

            # Calculate new dimensions
            new_height = int(height * scale_factor)
            new_width = int(width * scale_factor)

            # Scale image
            image = F.resize(image, [new_height, new_width])

            if target is not None:
                # Scale masks
                target["masks"] = F.resize(
                    target["masks"],
                    [new_height, new_width]
                )

                # Scale boxes
                if "boxes" in target and len(target["boxes"]) > 0:
                    target["boxes"] = target["boxes"] * scale_factor

        return image, target


def get_transforms():
    """
    Create training and testing data transformations

    Returns:
        tuple: (train_transform, test_transform) containing training and
            testing transforms
    """
    # Training data transform: Enhanced data augmentation + tensor conversion
    train_transform = T.Compose([
        RandomHorizontalFlip(p=0.5),
        RandomVerticalFlip(p=0.5),
        RandomRotation(degrees=15, p=0.3),
        RandomScale(scale_factors=(0.8, 1.2), p=0.3),
        T.ColorJitter(
            brightness=0.3,
            contrast=0.3,
            saturation=0.3,
            hue=0.1
        ),
        T.RandomGrayscale(p=0.02),  # Occasionally convert to grayscale
        T.GaussianBlur(
            kernel_size=3,
            sigma=(0.1, 2.0)
        ),  # Add blur for robustness
        T.RandomAdjustSharpness(
            sharpness_factor=2,
            p=0.3
        ),  # Adjust sharpness randomly
        T.ToTensor(),
        T.RandomErasing(
            p=0.1,
            scale=(0.02, 0.1),
            ratio=(0.3, 3.3),
            value=0
        ),  # Random erasing for robustness
    ])

    # Test data transform: Only tensor conversion
    test_transform = T.Compose([
        T.ToTensor(),
    ])

    return train_transform, test_transform


def collate_fn(batch):
    """
    Batch collation function for handling samples of different sizes

    Args:
        batch: Data batch

    Returns:
        Collated batch tuple
    """
    return tuple(zip(*batch))


class SegmentationDataset(Dataset):
    """
    Instance segmentation dataset class

    Supports training mode and testing mode:
    - Training mode: Loads images and corresponding segmentation masks
    - Testing mode: Only loads images for prediction
    """
    def __init__(
        self,
        root_dir,
        transform=None,
        is_test=False,
        test_json=None
    ):
        """
        Initialize dataset

        Args:
            root_dir (str): Dataset root directory
            transform: Data transformation function
            is_test (bool): Whether in test mode
            test_json (str): Test data mapping file path
        """
        self.root_dir = root_dir
        self.transform = transform
        self.is_test = is_test
        self.image_dirs = [
            os.path.join(root_dir, d)
            for d in os.listdir(root_dir)
        ]

        # If in test mode, load image ID mapping
        if is_test and test_json:
            with open(test_json, "r") as f:
                self.test_data = json.load(f)
                self.image_id_map = {
                    img["file_name"]: img["id"]
                    for img in self.test_data
                }

    def __len__(self):
        """Return dataset size"""
        return len(self.image_dirs)

    def __getitem__(self, idx):
        """
        Get a sample from the dataset

        Args:
            idx (int): Sample index

        Returns:
            tuple: Returns different data based on mode:
                - Training mode: (image, target) image and target dictionary
                - Test mode: (image, image_id) image and image ID
        """
        if not self.is_test:
            return self._get_training_item(idx)
        return self._get_test_item(idx)

    def _get_training_item(self, idx):
        """
        Get training sample

        Args:
            idx (int): Sample index

        Returns:
            tuple: (image, target) image and target dictionary
        """
        image_dir = self.image_dirs[idx]
        image_path = os.path.join(image_dir, "img.png")
        mask_path = os.path.join(image_dir, "mask.png")

        # Load image and mask
        image = Image.open(image_path).convert("RGB")
        mask = read_maskfile(mask_path)

        # Create target dictionary
        target = {
            "boxes": torch.tensor(
                [[0, 0, image.size[0], image.size[1]]],
                dtype=torch.float32
            ),
            "masks": torch.tensor(mask, dtype=torch.uint8),
            "labels": torch.ones(1, dtype=torch.int64),
        }

        # Apply transformations
        if self.transform:
            image, target = self.transform(image, target)

        return image, target

    def _get_test_item(self, idx):
        """
        Get test sample

        Args:
            idx (int): Sample index

        Returns:
            tuple: (image, image_id) image and image ID
        """
        image_dir = self.image_dirs[idx]
        image_path = os.path.join(image_dir, "img.png")
        image_name = os.path.basename(image_dir)

        # Load image
        image = Image.open(image_path).convert("RGB")

        # Get image ID
        image_id = self.image_id_map.get(image_name, idx)

        # Apply transformations
        if self.transform:
            image = self.transform(image)

        return image, image_id


def encode_mask(binary_mask):
    """
    Encode binary mask to RLE format

    Args:
        binary_mask: Binary mask array

    Returns:
        RLE encoded mask
    """
    return maskUtils.encode(
        np.asfortranarray(binary_mask.astype(np.uint8))
    )


def decode_mask(rle):
    """
    Decode RLE format to binary mask

    Args:
        rle: RLE encoded mask

    Returns:
        Binary mask array
    """
    return maskUtils.decode(rle)


def read_maskfile(filepath):
    """
    Read mask file and convert to binary mask array

    Args:
        filepath: Path to mask file

    Returns:
        Binary mask array
    """
    mask = io.imread(filepath)
    return (mask > 0).astype(np.uint8)
