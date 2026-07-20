import numpy as np

from canine_gait_cv.preprocessing import BoundingBox
from canine_gait_cv.visualization import draw_bounding_box


def test_draw_bounding_box_returns_same_shape() -> None:
    frame = np.zeros((100, 200, 3), dtype=np.uint8)
    box = BoundingBox(x_min=10, y_min=20, x_max=60, y_max=80)

    output = draw_bounding_box(frame, box)

    assert output.shape == frame.shape


def test_draw_bounding_box_does_not_modify_original_frame() -> None:
    frame = np.zeros((100, 200, 3), dtype=np.uint8)
    original = frame.copy()

    box = BoundingBox(x_min=10, y_min=20, x_max=60, y_max=80)

    _ = draw_bounding_box(frame, box)

    assert np.array_equal(frame, original)


def test_draw_bounding_box_changes_output_image() -> None:
    frame = np.zeros((100, 200, 3), dtype=np.uint8)
    box = BoundingBox(x_min=10, y_min=20, x_max=60, y_max=80)

    output = draw_bounding_box(frame, box)

    assert not np.array_equal(output, frame)