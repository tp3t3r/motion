"""Tests for StarTracker (bright-spot-on-dark centroid tracking)."""
import unittest
import numpy as np
import cv2

from motion_tracker.tracker import StarTracker, TrackResult


def make_star_frame(cx, cy, radius=4, peak=255, size=(480, 640)):
    """Dark frame with a single bright Gaussian-ish blob at (cx, cy)."""
    h, w = size
    frame = np.full((h, w, 3), 8, dtype=np.uint8)  # dark sky w/ slight bias
    yy, xx = np.ogrid[:h, :w]
    dist2 = (xx - cx) ** 2 + (yy - cy) ** 2
    blob = np.exp(-dist2 / (2.0 * radius ** 2)) * peak
    blob = np.clip(blob, 0, 255).astype(np.uint8)
    for c in range(3):
        frame[:, :, c] = np.maximum(frame[:, :, c], blob)
    return frame


class TestStarSelect(unittest.TestCase):
    def test_roi_capped_at_50(self):
        # Default and explicit oversize requests are both capped at 50.
        self.assertLessEqual(StarTracker().roi_size, 50)
        self.assertEqual(StarTracker(roi_size=200).roi_size, 50)

    def test_lock_on_bright_spot(self):
        frame = make_star_frame(320, 240)
        t = StarTracker()
        ok = t.select_target(frame, 322, 238)  # click slightly off-centre
        self.assertTrue(ok)
        self.assertTrue(t.is_active)

    def test_reject_dark_region(self):
        frame = make_star_frame(320, 240)
        t = StarTracker()
        # Click far from the star, in dark sky -> nothing to lock.
        ok = t.select_target(frame, 50, 50)
        self.assertFalse(ok)
        self.assertFalse(t.is_active)


class TestStarTrack(unittest.TestCase):
    def test_centroid_accuracy(self):
        frame = make_star_frame(320, 240)
        t = StarTracker()
        self.assertTrue(t.select_target(frame, 320, 240))
        result = t.track(frame)
        self.assertFalse(result.lost)
        # Centroid should be very close to the true centre.
        self.assertAlmostEqual(result.center[0], 320, delta=2.0)
        self.assertAlmostEqual(result.center[1], 240, delta=2.0)

    def test_tracks_translation(self):
        t = StarTracker()
        self.assertTrue(t.select_target(make_star_frame(320, 240), 320, 240))
        # Star moves by (15, -10).
        moved = make_star_frame(335, 230)
        result = t.track(moved)
        self.assertFalse(result.lost)
        self.assertAlmostEqual(result.center[0], 335, delta=3.0)
        self.assertAlmostEqual(result.center[1], 230, delta=3.0)

    def test_lost_when_star_disappears(self):
        t = StarTracker()
        self.assertTrue(t.select_target(make_star_frame(320, 240), 320, 240))
        dark = np.full((480, 640, 3), 8, dtype=np.uint8)
        result = t.track(dark)
        self.assertTrue(result.lost)

    def test_reset(self):
        t = StarTracker()
        t.select_target(make_star_frame(320, 240), 320, 240)
        t.reset()
        self.assertFalse(t.is_active)
        self.assertTrue(t.track(make_star_frame(320, 240)).lost)


if __name__ == "__main__":
    unittest.main()
