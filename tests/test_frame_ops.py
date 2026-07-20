import numpy as np

from canine_gait_cv.preprocessing import BoundingBox, crop_frame, resize_frame


def test_resize_frame_by_width_preserves_aspect_ratio() -> None:
    frame = np.zeros((100, 200, 3), dtype=np.uint8)

    resized = resize_frame(frame, target_width=100)

    assert resized.shape == (50, 100, 3)


def test_resize_frame_by_height_preserves_aspect_ratio() -> None:
    frame = np.zeros((100, 200, 3), dtype=np.uint8)

    resized = resize_frame(frame, target_height=50)

    assert resized.shape == (50, 100, 3)


def test_crop_frame_with_valid_box() -> None:
    frame = np.zeros((100, 200, 3), dtype=np.uint8)
    box = BoundingBox(x_min=10, y_min=20, x_max=60, y_max=80)

    cropped = crop_frame(frame, box)

    assert cropped.shape == (60, 50, 3)


def test_bounding_box_area() -> None:
    box = BoundingBox(x_min=10, y_min=20, x_max=60, y_max=80)

    assert box.width == 50
    assert box.height == 60
    assert box.area == 3000