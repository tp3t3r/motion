import cv2
import threading
import queue
import time
import logging

logger = logging.getLogger(__name__)

class CameraCapture:
    """Captures still frames from a camera with configurable exposure.
    
    Runs a daemon thread that grabs frames and pushes them to a queue.
    Exposure changes are applied immediately by signaling the capture thread.
    """
    
    def __init__(self, camera_index=0, exposure_ms=100):
        self._camera_index = camera_index
        self._exposure_ms = exposure_ms
        self._actual_exposure = None
        self._cap = None
        self._frame_queue = queue.Queue(maxsize=1)
        self._running = False
        self._thread = None
        self._lock = threading.Lock()
        self._exposure_changed = threading.Event()
        self._stop_event = threading.Event()
    
    @staticmethod
    def enumerate_cameras(max_index=5):
        """Probe camera indices 0..max_index-1 and return list of available indices."""
        available = []
        for i in range(max_index):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                ret, _ = cap.read()
                if ret:
                    available.append(i)
                cap.release()
        return available
    
    def start(self):
        """Start the capture thread."""
        if self._running:
            return
        self._stop_event.clear()
        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
    
    def stop(self):
        """Stop the capture thread and release the camera."""
        self._running = False
        self._stop_event.set()
        self._exposure_changed.set()  # unblock any wait
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None
        with self._lock:
            if self._cap is not None:
                self._cap.release()
                self._cap = None
    
    def switch_camera(self, camera_index):
        """Switch to a different camera. Stops and restarts the capture thread."""
        was_running = self._running
        self.stop()
        self._camera_index = camera_index
        # Clear the frame queue
        while not self._frame_queue.empty():
            try:
                self._frame_queue.get_nowait()
            except queue.Empty:
                break
        if was_running:
            self.start()
    
    def set_exposure(self, exposure_ms):
        """Set the desired exposure time in milliseconds. Applied immediately."""
        self._exposure_ms = max(1, exposure_ms)
        self._exposure_changed.set()
    
    def get_exposure_ms(self):
        """Return the currently requested exposure in ms."""
        return self._exposure_ms
    
    def get_actual_exposure(self):
        """Return the actual exposure value read back from the camera, or None."""
        return self._actual_exposure
    
    def get_frame(self):
        """Get the latest captured frame, or None if no frame is available.
        
        Returns:
            numpy.ndarray (BGR) or None
        """
        try:
            return self._frame_queue.get_nowait()
        except queue.Empty:
            return None
    
    @property
    def is_running(self):
        return self._running
    
    def _open_camera(self):
        """Open the camera and apply initial exposure settings."""
        cap = cv2.VideoCapture(self._camera_index)
        if not cap.isOpened():
            logger.error(f"Failed to open camera {self._camera_index}")
            return None
        self._apply_exposure(cap)
        return cap
    
    def _apply_exposure(self, cap):
        """Apply the current exposure setting to the camera (best-effort)."""
        if cap is None:
            return
        # Try to disable auto-exposure
        # Different backends use different values:
        # 0.25 = manual mode (some backends), 1 = manual (others)
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
        
        # Set exposure - the meaning of the value is backend-dependent
        # Some backends expect milliseconds, others expect log2 values
        cap.set(cv2.CAP_PROP_EXPOSURE, self._exposure_ms)
        
        # Read back actual value
        self._actual_exposure = cap.get(cv2.CAP_PROP_EXPOSURE)
        logger.info(f"Exposure set to {self._exposure_ms}ms, actual readback: {self._actual_exposure}")
    
    def _capture_loop(self):
        """Main capture loop running in a daemon thread."""
        cap = self._open_camera()
        if cap is None:
            self._running = False
            return
        
        with self._lock:
            self._cap = cap
        
        while self._running and not self._stop_event.is_set():
            # Check if exposure changed
            if self._exposure_changed.is_set():
                self._exposure_changed.clear()
                self._apply_exposure(cap)
            
            # Capture a frame
            ret, frame = cap.read()
            if not ret:
                logger.warning(f"Failed to read frame from camera {self._camera_index}")
                # Try to recover by reopening
                cap.release()
                time.sleep(0.5)
                cap = self._open_camera()
                if cap is None:
                    break
                with self._lock:
                    self._cap = cap
                continue
            
            # Put frame in queue, dropping old frame if queue is full
            if self._frame_queue.full():
                try:
                    self._frame_queue.get_nowait()
                except queue.Empty:
                    pass
            self._frame_queue.put(frame)
            
            # Wait based on exposure time, but allow interruption
            # The wait simulates the interval between captures.
            # For very short exposures, we still add a small delay to avoid
            # spinning; for long exposures, the camera itself will block in read().
            wait_time = max(0.01, self._exposure_ms / 1000.0)
            self._exposure_changed.wait(timeout=wait_time)
            # If the event was set to change exposure, it'll be handled at top of loop
        
        cap.release()
        with self._lock:
            self._cap = None
        self._running = False
