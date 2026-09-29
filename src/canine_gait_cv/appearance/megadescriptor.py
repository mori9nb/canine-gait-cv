from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch
from PIL import Image
from torch import Tensor, nn

from canine_gait_cv.preprocessing import BoundingBox


MEGADESCRIPTOR_MODEL_NAME = "hf-hub:BVRA/MegaDescriptor-S-224"


@dataclass(frozen=True)
class AppearanceEmbedding:
    """Appearance representation extracted from one dog crop."""

    vector: tuple[float, ...]
    crop_box: BoundingBox

    @property
    def dimension(self) -> int:
        return len(self.vector)


class MegaDescriptorExtractor:
    """Extract normalized animal ReID embeddings from RGB images."""

    def __init__(
        self,
        model: nn.Module,
        transform: Callable[[Image.Image], Tensor],
        *,
        device: str | torch.device = "cpu",
        minimum_crop_size: int = 8,
    ) -> None:
        if minimum_crop_size <= 0:
            raise ValueError("minimum_crop_size must be positive.")

        self.device = torch.device(device)
        self.minimum_crop_size = minimum_crop_size
        self.model = model.eval().to(self.device)
        self.transform = transform

    @classmethod
    def from_pretrained(
        cls,
        *,
        device: str | torch.device | None = None,
        minimum_crop_size: int = 8,
    ) -> "MegaDescriptorExtractor":
        """Load MegaDescriptor and its model-specific preprocessing."""

        import timm
        from timm.data import create_transform, resolve_model_data_config

        selected_device = torch.device(
            device
            if device is not None
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        model = timm.create_model(
            MEGADESCRIPTOR_MODEL_NAME,
            pretrained=True,
            num_classes=0,
        )

        data_config = resolve_model_data_config(model)
        transform = create_transform(**data_config, is_training=False)

        return cls(
            model=model,
            transform=transform,
            device=selected_device,
            minimum_crop_size=minimum_crop_size,
        )

    def extract(
        self,
        image: Image.Image,
        box: BoundingBox,
    ) -> AppearanceEmbedding:
        """Crop one dog and return its L2-normalized appearance embedding."""

        if not isinstance(image, Image.Image):
            raise TypeError("image must be a PIL Image.")

        clipped_box = self._clip_box(box, image.width, image.height)

        if (
            clipped_box.x_max - clipped_box.x_min < self.minimum_crop_size
            or clipped_box.y_max - clipped_box.y_min < self.minimum_crop_size
        ):
            raise ValueError("Dog crop is too small for appearance extraction.")

        crop = image.convert("RGB").crop(
            (
                clipped_box.x_min,
                clipped_box.y_min,
                clipped_box.x_max,
                clipped_box.y_max,
            )
        )

        tensor = self.transform(crop)

        if tensor.ndim != 3:
            raise ValueError("transform must return a CHW image tensor.")

        batch = tensor.unsqueeze(0).to(self.device)

        with torch.inference_mode():
            output = self.model(batch)

        if isinstance(output, (tuple, list)):
            output = output[0]

        if output.ndim > 2:
            output = output.flatten(start_dim=1)

        if output.ndim != 2 or output.shape[0] != 1:
            raise ValueError("model must return one embedding per image.")

        embedding = output[0].float()
        norm = torch.linalg.vector_norm(embedding)

        if not torch.isfinite(embedding).all() or not torch.isfinite(norm):
            raise ValueError("model returned a non-finite embedding.")

        if norm.item() == 0.0:
            raise ValueError("model returned a zero embedding.")

        embedding = embedding / norm

        return AppearanceEmbedding(
            vector=tuple(embedding.cpu().tolist()),
            crop_box=clipped_box,
        )

    @staticmethod
    def cosine_similarity(
        first: AppearanceEmbedding,
        second: AppearanceEmbedding,
    ) -> float:
        if first.dimension != second.dimension:
            raise ValueError("Embeddings must have the same dimension.")

        first_tensor = torch.tensor(first.vector, dtype=torch.float32)
        second_tensor = torch.tensor(second.vector, dtype=torch.float32)

        return float(torch.dot(first_tensor, second_tensor).item())

    @staticmethod
    def _clip_box(
        box: BoundingBox,
        image_width: int,
        image_height: int,
    ) -> BoundingBox:
        x_min = max(0, min(box.x_min, image_width))
        y_min = max(0, min(box.y_min, image_height))
        x_max = max(0, min(box.x_max, image_width))
        y_max = max(0, min(box.y_max, image_height))

        if x_max <= x_min or y_max <= y_min:
            raise ValueError("Bounding box does not intersect the image.")

        return BoundingBox(
            x_min=x_min,
            y_min=y_min,
            x_max=x_max,
            y_max=y_max,
        )