from __future__ import annotations

import pytest
import torch
from PIL import Image
from torch import Tensor, nn

from canine_gait_cv.appearance import (
    AppearanceEmbedding,
    MegaDescriptorExtractor,
)
from canine_gait_cv.preprocessing import BoundingBox


class FakeModel(nn.Module):
    def forward(self, batch: Tensor) -> Tensor:
        means = batch.mean(dim=(2, 3))
        return torch.cat((means, means), dim=1)


def fake_transform(image: Image.Image) -> Tensor:
    values = torch.tensor(
        [
            image.getchannel("R").getextrema()[1],
            image.getchannel("G").getextrema()[1],
            image.getchannel("B").getextrema()[1],
        ],
        dtype=torch.float32,
    )
    return values[:, None, None].expand(3, 4, 4) / 255.0


def test_extract_returns_normalized_embedding() -> None:
    image = Image.new("RGB", (100, 80), color=(255, 128, 64))
    extractor = MegaDescriptorExtractor(FakeModel(), fake_transform)

    result = extractor.extract(
        image,
        BoundingBox(10, 10, 70, 60),
    )

    assert result.dimension == 6
    assert result.crop_box == BoundingBox(10, 10, 70, 60)

    norm = sum(value**2 for value in result.vector) ** 0.5
    assert norm == pytest.approx(1.0)


def test_extract_clips_box_to_image_boundaries() -> None:
    image = Image.new("RGB", (100, 80), color="white")
    extractor = MegaDescriptorExtractor(FakeModel(), fake_transform)

    result = extractor.extract(
        image,
        BoundingBox(-10, -20, 120, 90),
    )

    assert result.crop_box == BoundingBox(0, 0, 100, 80)


def test_extract_rejects_tiny_crop() -> None:
    image = Image.new("RGB", (100, 80), color="white")
    extractor = MegaDescriptorExtractor(
        FakeModel(),
        fake_transform,
        minimum_crop_size=8,
    )

    with pytest.raises(ValueError, match="too small"):
        extractor.extract(
            image,
            BoundingBox(10, 10, 15, 17),
        )


def test_extract_rejects_box_outside_image() -> None:
    image = Image.new("RGB", (100, 80), color="white")
    extractor = MegaDescriptorExtractor(FakeModel(), fake_transform)

    with pytest.raises(ValueError, match="does not intersect"):
        extractor.extract(
            image,
            BoundingBox(110, 90, 130, 120),
        )


def test_cosine_similarity_for_identical_embeddings() -> None:
    embedding = AppearanceEmbedding(
        vector=(0.6, 0.8),
        crop_box=BoundingBox(0, 0, 10, 10),
    )

    similarity = MegaDescriptorExtractor.cosine_similarity(
        embedding,
        embedding,
    )

    assert similarity == pytest.approx(1.0)


def test_minimum_crop_size_must_be_positive() -> None:
    with pytest.raises(ValueError, match="must be positive"):
        MegaDescriptorExtractor(
            FakeModel(),
            fake_transform,
            minimum_crop_size=0,
        )