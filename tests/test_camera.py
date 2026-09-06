"""Unit tests for the camera module."""

import unittest
from unittest.mock import patch, MagicMock, PropertyMock
import queue
import time
import numpy as np

from motion_tracker.camera import CameraCapture


class TestEnumerateCameras(unittest.TestCase):
    """Tests for CameraCapture.enumerate_cameras()."""
    
    @patch('motion_tracker.camera.cv2.VideoCapture')
    def test_enumerate_finds_cameras(self, mock_vc_class):
        """Should return indices of cameras that open and read successfully."""
        def make_cap(index):
            cap = MagicMock()
            if index in (0, 2):  # cameras 0 and 2 are available
                cap.isOpened.return_value = True
                cap.read.return_value = (True, np.zeros((480, 640, 3), dtype=np.uint8))
            else:
                cap.isOpened.return_value = False
            return cap
        
        mock_vc_class.side_effect = make_cap
        result = CameraCapture.enumerate_cameras(max_index=4)
        self.assertEqual(result, [0, 2])
    
    @patch('motion_tracker.camera.cv2.VideoCapture')
    def test_enumerate_no_cameras(self, mock_vc_class):
        """Should return empty list when no cameras are found."""
        cap = MagicMock()
        cap.isOpened.return_value = False
        mock_vc_class.return_value = cap
        
        result = CameraCapture.enumerate_cameras(max_index=3)
        self.assertEqual(result, [])
    
    @patch('motion_tracker.camera.cv2.VideoCapture')
    def test_enumerate_returns_list(self, mock_vc_class):
        """Should always return a list."""
        cap = MagicMock()
        cap.isOpened.return_value = False
        mock_vc_class.return_value = cap
        
        result = CameraCapture.enumerate_cameras()
        self.assertIsInstance(result, list)


class TestCameraCapture(unittest.TestCase):
    """Tests for the CameraCapture class."""
    
    def test_init_defaults(self):
        """Should initialize with default values."""
        cam = CameraCapture()
        self.assertEqual(cam.get_exposure_ms(), 100)
        self.assertFalse(cam.is_running)
    
    def test_init_custom(self):
        """Should accept custom camera index and exposure."""
        cam = CameraCapture(camera_index=2, exposure_ms=500)
        self.assertEqual(cam.get_exposure_ms(), 500)
    
    def test_set_exposure(self):
        """set_exposure should update the stored value."""
        cam = CameraCapture()
        cam.set_exposure(250)
        self.assertEqual(cam.get_exposure_ms(), 250)
    
    def test_set_exposure_minimum(self):
        """set_exposure should clamp to minimum of 1ms."""
        cam = CameraCapture()
        cam.set_exposure(0)
        self.assertEqual(cam.get_exposure_ms(), 1)
        cam.set_exposure(-100)
        self.assertEqual(cam.get_exposure_ms(), 1)
    
    def test_get_actual_exposure_before_start(self):
        """Should return None before the camera has started."""
        cam = CameraCapture()
        self.assertIsNone(cam.get_actual_exposure())
    
    def test_get_frame_empty(self):
        """Should return None when no frames have been captured."""
        cam = CameraCapture()
        self.assertIsNone(cam.get_frame())
    
    @patch('motion_tracker.camera.cv2.VideoCapture')
    def test_start_stop(self, mock_vc_class):
        """Should start and stop without errors."""
        cap = MagicMock()
        cap.isOpened.return_value = True
        cap.read.return_value = (True, np.zeros((480, 640, 3), dtype=np.uint8))
        cap.get.return_value = 100.0
        mock_vc_class.return_value = cap
        
        cam = CameraCapture(camera_index=0)
        cam.start()
        self.assertTrue(cam.is_running)
        time.sleep(0.2)  # let the thread run briefly
        cam.stop()
        self.assertFalse(cam.is_running)
    
    @patch('motion_tracker.camera.cv2.VideoCapture')
    def test_switch_camera(self, mock_vc_class):
        """switch_camera should stop and restart with the new index."""
        cap = MagicMock()
        cap.isOpened.return_value = True
        cap.read.return_value = (True, np.zeros((480, 640, 3), dtype=np.uint8))
        cap.get.return_value = 100.0
        mock_vc_class.return_value = cap
        
        cam = CameraCapture(camera_index=0)
        cam.start()
        time.sleep(0.1)
        cam.switch_camera(1)
        time.sleep(0.1)
        self.assertTrue(cam.is_running)
        cam.stop()
    
    @patch('motion_tracker.camera.cv2.VideoCapture')
    def test_frame_capture(self, mock_vc_class):
        """Should capture and queue frames."""
        test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        test_frame[100, 100] = [255, 0, 0]  # mark to identify
        
        cap = MagicMock()
        cap.isOpened.return_value = True
        cap.read.return_value = (True, test_frame)
        cap.get.return_value = 100.0
        mock_vc_class.return_value = cap
        
        cam = CameraCapture(camera_index=0, exposure_ms=10)
        cam.start()
        time.sleep(0.3)  # let some frames be captured
        
        frame = cam.get_frame()
        cam.stop()
        
        self.assertIsNotNone(frame)
        self.assertEqual(frame.shape, (480, 640, 3))


class TestExposureSteps(unittest.TestCase):
    """Test exposure-related utility logic."""
    
    def test_snap_to_step_import(self):
        """snap_to_step should be importable from gui module."""
        from motion_tracker.gui import snap_to_step, EXPOSURE_STEPS
        self.assertEqual(snap_to_step(100), 100)
        self.assertEqual(snap_to_step(99), 100)
        self.assertEqual(snap_to_step(75), 50)
        self.assertEqual(snap_to_step(1), 1)
        self.assertEqual(snap_to_step(30000), 30000)
        self.assertIn(snap_to_step(150), EXPOSURE_STEPS)


if __name__ == '__main__':
    unittest.main()
