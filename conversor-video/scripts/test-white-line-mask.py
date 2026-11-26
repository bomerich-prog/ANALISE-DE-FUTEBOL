#!/usr/bin/env python3
"""The line mask keeps thin painted lines and drops wide sunlit grass.

Gremio x Cruzeiro, 18 s to 31 s: far, light-mown stripes in the sun passed the
white test as wide patches and swallowed the far half of the centre circle, so
the 2D map was lost. A painted line is thin and brighter than the grass beside
it; a lit stripe is wide.
"""

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "server/services/analysis/yolo"))

from run_yolo_analysis import build_grass_support_mask, build_white_line_mask  # noqa: E402


def _frame():
    frame = np.zeros((400, 800, 3), dtype=np.uint8)
    frame[:] = (40, 120, 50)  # grass, BGR
    # A wide, pale, sunlit stripe: bright and grey enough to pass the white test.
    frame[40:160, 100:400] = (140, 175, 145)
    # A painted line, 6 px wide, crossing the dark grass.
    cv2.line(frame, (450, 20), (450, 380), (235, 240, 235), 6)
    # The same line also crossing the lit stripe.
    cv2.line(frame, (100, 100), (400, 100), (250, 252, 250), 4)
    return frame


def test_line_kept_patch_dropped():
    frame = _frame()
    mask = build_white_line_mask(frame, build_grass_support_mask(frame))
    line = mask[20:380, 448:453]
    patch = mask[50:85, 120:380]
    assert (line > 0).mean() > 0.8, (line > 0).mean()
    assert (patch > 0).mean() < 0.05, (patch > 0).mean()
    stripe_line = mask[97:104, 140:360]
    assert (stripe_line > 0).mean() > 0.3, (stripe_line > 0).mean()


def main():
    test_line_kept_patch_dropped()
    print("white line mask tests passed")


if __name__ == "__main__":
    main()
