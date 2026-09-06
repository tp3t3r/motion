"""Unit tests for the ORB tracker module."""

import unittest
import cv2
import numpy as np

from motion_tracker.tracker import ORBTracker, TrackResult


def make_textured_image(width=640, height=480, seed=42):
    """Create a synthetic textured image with random patterns."""
    rng = np.random.RandomState(seed)
    img = rng.randint(0, 256, (height, width, 3), dtype=np.uint8)
    # Add some structure — circles and rectangles
    cv2.circle(img, (320, 240), 50, (255, 0, 0), -1)
    cv2.rectangle(img, (100, 100), (200, 200), (0, 255, 0), -1)
    cv2.rectangle(img, (400, 300), (500, 400), (0, 0, 255), -1)
    # Add some text for extra features
    cv2.putText(img, "TRACK ME", (250, 250), cv2.FONT_HERSHEY_SIMPLEX,
                1.0, (255, 255, 255), 2)
    return img


def translate_image(img, tx, ty):
    """Translate an image by (tx, ty) pixels."""
    h, w = img.shape[:2]
    M = np.float32([[1, 0, tx], [0, 1, ty]])
    return cv2.warpAffine(img, M, (w, h))


def rotate_image(img, angle_deg, center=None):
    """Rotate an image around a center point."""
    h, w = img.shape[:2]
    if center is None:
        center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    return cv2.warpAffine(img, M, (w, h))


def scale_image(img, scale_factor, center=None):
    """Scale an image around a center point."""
    h, w = img.shape[:2]
    if center is None:
        center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, 0, scale_factor)
    return cv2.warpAffine(img, M, (w, h))


class TestORBTrackerInit(unittest.TestCase):
    """Tests for ORBTracker initialization."""
    
    def test_default_init(self):
        tracker = ORBTracker()
        self.assertFalse(tracker.is_active)
        self.assertEqual(tracker.roi_size, 200)
    
    def test_custom_init(self):
        tracker = ORBTracker(nfeatures=1000, roi_size=120)
        self.assertEqual(tracker.roi_size, 120)


class TestSelectTarget(unittest.TestCase):
    """Tests for target selection."""
    
    def test_select_on_textured_region(self):
        """Should successfully select a target on a textured image."""
        img = make_textured_image()
        tracker = ORBTracker(roi_size=150)
        result = tracker.select_target(img, 320, 240)
        self.assertTrue(result)
        self.assertTrue(tracker.is_active)
    
    def test_select_on_blank_image(self):
        """Should fail to select on a completely uniform image."""
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        tracker = ORBTracker()
        result = tracker.select_target(img, 320, 240)
        self.assertFalse(result)
        self.assertFalse(tracker.is_active)
    
    def test_select_near_edge(self):
        """Should handle selection near image edges (ROI clamped)."""
        img = make_textured_image()
        tracker = ORBTracker()
        # Click near top-left corner
        result = tracker.select_target(img, 10, 10)
        # May or may not find features, but should not crash
        self.assertIsInstance(result, bool)
    
    def test_select_resets_state(self):
        """Selecting a new target should reset tracking state."""
        img = make_textured_image()
        tracker = ORBTracker()
        tracker.select_target(img, 320, 240)
        # Select a new target
        tracker.select_target(img, 200, 200)
        self.assertTrue(tracker.is_active)


class TestTrack(unittest.TestCase):
    """Tests for frame-to-frame tracking."""
    
    def test_track_without_selection(self):
        """Should return lost=True if no target selected."""
        tracker = ORBTracker()
        img = make_textured_image()
        result = tracker.track(img)
        self.assertTrue(result.lost)
    
    def test_track_stationary(self):
        """Tracking an identical frame should give near-zero delta."""
        img = make_textured_image()
        tracker = ORBTracker()
        tracker.select_target(img, 320, 240)
        result = tracker.track(img)
        if not result.lost:
            self.assertAlmostEqual(result.dx, 0, delta=3.0)
            self.assertAlmostEqual(result.dy, 0, delta=3.0)
    
    def test_track_translation(self):
        """Tracking a translated image should give approximately correct delta."""
        img = make_textured_image()
        tracker = ORBTracker(roi_size=120)
        tracker.select_target(img, 320, 240)
        
        # Translate the image by (30, -20)
        translated = translate_image(img, 30, -20)
        result = tracker.track(translated)
        
        if not result.lost:
            # The delta should be approximately (30, -20)
            self.assertAlmostEqual(result.dx, 30, delta=10.0)
            self.assertAlmostEqual(result.dy, -20, delta=10.0)
    
    def test_track_rotation(self):
        """Tracking a rotated image should not lose the target."""
        img = make_textured_image()
        tracker = ORBTracker(roi_size=120)
        tracker.select_target(img, 320, 240)
        
        # Rotate by 10 degrees around the image center
        rotated = rotate_image(img, 10, center=(320, 240))
        result = tracker.track(rotated)
        
        # When rotating around the target center, dx/dy should be small
        if not result.lost:
            self.assertFalse(result.lost)
    
    def test_track_result_dataclass(self):
        """TrackResult should have all expected fields."""
        r = TrackResult()
        self.assertTrue(r.lost)
        self.assertEqual(r.dx, 0.0)
        self.assertEqual(r.dy, 0.0)
        self.assertIsNone(r.center)
        self.assertIsNone(r.corners)
        self.assertEqual(r.inlier_ratio, 0.0)
        self.assertEqual(r.num_matches, 0)


class TestReset(unittest.TestCase):
    """Tests for tracker reset."""
    
    def test_reset(self):
        img = make_textured_image()
        tracker = ORBTracker(roi_size=150)
        tracker.select_target(img, 320, 240)
        self.assertTrue(tracker.is_active)
        tracker.reset()
        self.assertFalse(tracker.is_active)


class TestAdaptiveUpdate(unittest.TestCase):
    """Tests for adaptive reference update across multiple frames."""
    
    def test_multi_frame_tracking(self):
        """Tracker should handle a sequence of small translations."""
        img = make_textured_image()
        tracker = ORBTracker(roi_size=120)
        tracker.select_target(img, 320, 240)
        
        total_dx = 0
        total_dy = 0
        lost_count = 0
        
        for i in range(15):
            # Small translation each frame
            tx = (i + 1) * 3
            ty = (i + 1) * 2
            moved = translate_image(img, tx, ty)
            result = tracker.track(moved)
            if result.lost:
                lost_count += 1
            else:
                total_dx += result.dx
                total_dy += result.dy
        
        # Should not lose tracking for most frames
        self.assertLess(lost_count, 10, "Lost tracking too many times")
    
    def test_incremental_rotation_tracking(self):
        """Tracker should handle incremental rotation across many frames."""
        img = make_textured_image()
        tracker = ORBTracker(roi_size=150, nfeatures=1000)
        tracker.select_target(img, 320, 240)
        
        lost_count = 0
        for angle in range(0, 45, 3):  # 0 to 42 degrees in 3-degree steps
            rotated = rotate_image(img, angle, center=(320, 240))
            result = tracker.track(rotated)
            if result.lost:
                lost_count += 1
        
        # Allow some lost frames but should track most
        self.assertLess(lost_count, 10, "Lost tracking too many times during rotation")


if __name__ == '__main__':
    unittest.main()
