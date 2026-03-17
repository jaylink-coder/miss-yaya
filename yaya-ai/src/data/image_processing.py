"""Image preprocessing and augmentation for vision training.

Handles loading, resizing, normalization, and augmentation of images
for the vision encoder input pipeline.
"""

import torch
import torch.nn.functional as F
from typing import Optional, Tuple, List
from pathlib import Path

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

try:
    from torchvision import transforms
    TORCHVISION_AVAILABLE = True
except ImportError:
    TORCHVISION_AVAILABLE = False


class ImageProcessor:
    """Process images for the Yaya vision encoder.

    Handles:
    - Loading from file path
    - Resizing to target resolution
    - Normalization (ImageNet or CLIP stats)
    - Optional augmentation for training
    """

    # CLIP/OpenAI normalization stats
    CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
    CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

    # ImageNet normalization stats
    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)

    def __init__(
        self,
        image_size: int = 336,
        normalization: str = "clip",
        augment: bool = False,
    ):
        """Initialize image processor.

        Args:
            image_size: Target image size (square).
            normalization: 'clip' or 'imagenet' normalization stats.
            augment: Whether to apply training augmentations.
        """
        if not PIL_AVAILABLE:
            raise ImportError("Pillow is required for image processing: pip install Pillow")

        self.image_size = image_size
        self.augment = augment

        # Select normalization stats
        if normalization == "clip":
            mean, std = self.CLIP_MEAN, self.CLIP_STD
        else:
            mean, std = self.IMAGENET_MEAN, self.IMAGENET_STD

        if TORCHVISION_AVAILABLE:
            # Build transform pipeline
            if augment:
                self.transform = transforms.Compose([
                    transforms.RandomResizedCrop(
                        image_size, scale=(0.8, 1.0), interpolation=transforms.InterpolationMode.BICUBIC
                    ),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=mean, std=std),
                ])
            else:
                self.transform = transforms.Compose([
                    transforms.Resize(
                        image_size, interpolation=transforms.InterpolationMode.BICUBIC
                    ),
                    transforms.CenterCrop(image_size),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=mean, std=std),
                ])
        else:
            self.transform = None
            self._mean = torch.tensor(mean).view(3, 1, 1)
            self._std = torch.tensor(std).view(3, 1, 1)

    def load_image(self, path: str) -> "Image.Image":
        """Load an image from file path."""
        img = Image.open(path).convert("RGB")
        return img

    def __call__(self, image_input) -> torch.Tensor:
        """Process an image into a normalized tensor.

        Args:
            image_input: File path (str) or PIL Image.

        Returns:
            Tensor [3, image_size, image_size] normalized.
        """
        if isinstance(image_input, str):
            image = self.load_image(image_input)
        else:
            image = image_input

        if self.transform is not None:
            return self.transform(image)

        # Manual fallback without torchvision transforms
        return self._manual_process(image)

    def _manual_process(self, image: "Image.Image") -> torch.Tensor:
        """Process image without torchvision transforms."""
        # Resize
        image = image.resize((self.image_size, self.image_size), Image.BICUBIC)

        # Convert to tensor [3, H, W] in [0, 1]
        import numpy as np
        arr = np.array(image, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(arr).permute(2, 0, 1)  # HWC -> CHW

        # Normalize
        tensor = (tensor - self._mean) / self._std
        return tensor

    def batch_process(
        self,
        images: List,
        padding: bool = True,
    ) -> torch.Tensor:
        """Process a batch of images.

        Args:
            images: List of file paths or PIL Images.
            padding: Not used (all images same size after processing).

        Returns:
            Batch tensor [batch, 3, image_size, image_size].
        """
        tensors = [self(img) for img in images]
        return torch.stack(tensors)
