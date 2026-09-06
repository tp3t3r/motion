import cv2
import numpy as np
import logging
import time
from dataclasses import dataclass
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


class _LowPass:
    """First-order low-pass (exponential) filter with a settable alpha."""

    def __init__(self):
        self._prev = None

    def reset(self):
        self._prev = None

    def __call__(self, value, alpha):
        if self._prev is None:
            self._prev = value
        else:
            self._prev = alpha * value + (1.0 - alpha) * self._prev
        return self._prev

    @property
    def last(self):
        return self._prev


class OneEuroFilter:
    """1-D One Euro filter (Casiez et al., 2012).

    Adaptive smoothing that trades jitter against lag based on speed:
    when the signal is nearly still it smooths hard (kills jitter), and
    when it moves fast it loosens up (reduces lag). Ideal for producing
    stable set-points for a control loop such as a stepper motor.

    Parameters:
        min_cutoff: lower -> smoother but more lag when still.
        beta:       higher -> more responsive to fast motion (less lag).
        d_cutoff:   cutoff for the derivative low-pass.
    """

    def __init__(self, min_cutoff=1.0, beta=0.02, d_cutoff=1.0):
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self._x = _LowPass()
        self._dx = _LowPass()
        self._prev_value = None

    def reset(self):
        self._x.reset()
        self._dx.reset()
        self._prev_value = None

    @staticmethod
    def _alpha(cutoff, dt):
        tau = 1.0 / (2.0 * np.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, value, dt):
        if dt <= 0:
            dt = 1e-3
        if self._prev_value is None:
            self._prev_value = value
            self._x(value, 1.0)
            return value
        # Filtered derivative (rate of change) of the signal.
        dvalue = (value - self._prev_value) / dt
        edvalue = self._dx(dvalue, self._alpha(self.d_cutoff, dt))
        # Speed-adaptive cutoff.
        cutoff = self.min_cutoff + self.beta * abs(edvalue)
        result = self._x(value, self._alpha(cutoff, dt))
        self._prev_value = value
        return result


@dataclass
class TrackResult:
    """Result of a single tracking step."""
    dx: float = 0.0
    dy: float = 0.0
    center: Optional[Tuple[float, float]] = None
    corners: Optional[np.ndarray] = None
    lost: bool = True
    inlier_ratio: float = 0.0
    num_matches: int = 0


class ORBTracker:
    """Tracks a user-selected subject using ORB features and homography.
    
    The tracker extracts ORB keypoints/descriptors from a region of interest
    (ROI) around the user's click point, then matches them frame-to-frame
    using a BFMatcher with Lowe's ratio test. A homography is computed via
    RANSAC to determine the subject's new position, yielding (dx, dy) deltas.
    
    The reference features are periodically updated to handle gradual
    transformation (rotation, scale) of the subject over time.
    """
    
    # Configuration
    MIN_GOOD_MATCHES = 4
    RATIO_TEST_THRESHOLD = 0.80
    RANSAC_REPROJ_THRESHOLD = 5.0
    REFERENCE_UPDATE_INTERVAL = 5   # update reference every N successful tracks
    REFERENCE_UPDATE_MIN_INLIER_RATIO = 0.5
    SEARCH_REGION_MULTIPLIER = 3.0
    ADAPTIVE_SEARCH_GROWTH = 1.5
    ADAPTIVE_SEARCH_SHRINK = 0.95
    MAX_SEARCH_MULTIPLIER = 6.0
    MIN_SEARCH_MULTIPLIER = 2.0
    MOTION_THRESHOLD = 20  # pixels — above this, expand search region
    
    def __init__(self, nfeatures=1000, roi_size=200):
        self._orb = cv2.ORB_create(nfeatures=nfeatures)
        self._bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        self._roi_size = roi_size
        
        # Reference data (set by select_target)
        self._ref_keypoints = None
        self._ref_descriptors = None
        self._ref_roi_kp_coords = None  # keypoint coords relative to ROI
        self._ref_center = None  # (x, y) in frame coordinates
        self._ref_corners = None  # 4 corners of the ROI in frame coords
        
        # Tracking state
        self._prev_center = None
        self._current_search_multiplier = self.SEARCH_REGION_MULTIPLIER
        self._successful_track_count = 0
        self._active = False

        # Smoothing: adaptive One Euro filter per axis, applied to the
        # reported centre so downstream consumers (display, stepper motor)
        # get a stable, low-jitter set-point.
        self._filter_x = OneEuroFilter(min_cutoff=1.0, beta=0.02)
        self._filter_y = OneEuroFilter(min_cutoff=1.0, beta=0.02)
        self._last_track_time = None
        self._smoothed_center = None
        self._prev_raw_center = None
    
    @property
    def is_active(self):
        """Whether a target has been selected and tracking is active."""
        return self._active
    
    @property
    def roi_size(self):
        return self._roi_size
    
    def select_target(self, frame, center_x, center_y, roi_size=None):
        """Select a target to track by specifying a center point on the frame.
        
        Extracts a square ROI around the point, computes ORB features, and
        stores them as the reference for subsequent tracking.
        
        Args:
            frame: BGR image (numpy array)
            center_x: X coordinate of the click point
            center_y: Y coordinate of the click point
            roi_size: Size of the square ROI (overrides default)
        
        Returns:
            True if the target was successfully selected (features found),
            False otherwise.
        """
        if roi_size is not None:
            self._roi_size = roi_size
        
        h, w = frame.shape[:2]
        half = self._roi_size // 2
        
        # Clamp ROI to frame bounds
        x1 = max(0, int(center_x) - half)
        y1 = max(0, int(center_y) - half)
        x2 = min(w, x1 + self._roi_size)
        y2 = min(h, y1 + self._roi_size)
        
        if x2 - x1 < 20 or y2 - y1 < 20:
            logger.warning("ROI too small after clamping")
            return False
        
        roi = frame[y1:y2, x1:x2]
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        keypoints, descriptors = self._orb.detectAndCompute(gray_roi, None)
        
        if descriptors is None or len(keypoints) < self.MIN_GOOD_MATCHES:
            logger.warning(f"Not enough features found in ROI: "
                           f"{len(keypoints) if keypoints else 0} "
                           f"(need at least {self.MIN_GOOD_MATCHES})")
            self._active = False
            return False
        
        # Store reference data
        # Convert keypoint coordinates to frame coordinates
        self._ref_roi_kp_coords = np.array([kp.pt for kp in keypoints], dtype=np.float32)
        self._ref_keypoints = keypoints
        self._ref_descriptors = descriptors
        
        actual_cx = (x1 + x2) / 2.0
        actual_cy = (y1 + y2) / 2.0
        self._ref_center = (actual_cx, actual_cy)
        self._prev_center = (actual_cx, actual_cy)
        
        # Store ROI corners in frame coordinates
        self._ref_corners = np.array([
            [x1, y1],
            [x2, y1],
            [x2, y2],
            [x1, y2]
        ], dtype=np.float32)
        
        # Store the ROI offset for converting keypoints to frame coords
        self._roi_offset = (x1, y1)
        
        # Reset tracking state
        self._current_search_multiplier = self.SEARCH_REGION_MULTIPLIER
        self._successful_track_count = 0
        self._active = True

        # Prime smoothing with the initial centre so the first tracked
        # frame yields a meaningful delta (measured from the click point)
        # while still being filtered thereafter.
        self._filter_x.reset()
        self._filter_y.reset()
        self._last_track_time = None
        self._smoothed_center = (actual_cx, actual_cy)
        self._prev_raw_center = (actual_cx, actual_cy)
        
        logger.info(f"Target selected at ({actual_cx:.1f}, {actual_cy:.1f}) with {len(keypoints)} features")
        return True
    
    def track(self, frame):
        """Track the subject in a new frame.
        
        Args:
            frame: BGR image (numpy array)
        
        Returns:
            TrackResult with dx, dy, center, corners, lost status, etc.
        """
        if not self._active:
            return TrackResult(lost=True)
        
        h, w = frame.shape[:2]
        
        # Define search region around last known position
        search_half = int(self._roi_size * self._current_search_multiplier / 2)
        prev_cx, prev_cy = self._prev_center
        
        sx1 = max(0, int(prev_cx) - search_half)
        sy1 = max(0, int(prev_cy) - search_half)
        sx2 = min(w, int(prev_cx) + search_half)
        sy2 = min(h, int(prev_cy) + search_half)
        
        if sx2 - sx1 < 20 or sy2 - sy1 < 20:
            return TrackResult(lost=True)
        
        # Detect ORB features in the search region
        search_roi = frame[sy1:sy2, sx1:sx2]
        gray_search = cv2.cvtColor(search_roi, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = self._orb.detectAndCompute(gray_search, None)
        
        if descriptors is None or len(keypoints) < self.MIN_GOOD_MATCHES:
            return TrackResult(lost=True, num_matches=0)
        
        # Match features
        try:
            matches = self._bf.knnMatch(self._ref_descriptors, descriptors, k=2)
        except cv2.error:
            return TrackResult(lost=True, num_matches=0)
        
        # Lowe's ratio test
        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < self.RATIO_TEST_THRESHOLD * n.distance:
                    good_matches.append(m)
        
        if len(good_matches) < self.MIN_GOOD_MATCHES:
            return TrackResult(lost=True, num_matches=len(good_matches))
        
        # Extract matched point coordinates
        src_pts = np.array(
            [self._ref_roi_kp_coords[m.queryIdx] for m in good_matches],
            dtype=np.float32
        ).reshape(-1, 1, 2)
        
        dst_pts = np.array(
            [keypoints[m.trainIdx].pt for m in good_matches],
            dtype=np.float32
        ).reshape(-1, 1, 2)
        
        # Compute homography
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,
                                      self.RANSAC_REPROJ_THRESHOLD)
        
        if H is None:
            return TrackResult(lost=True, num_matches=len(good_matches))
        
        # Compute inlier ratio
        inlier_count = int(mask.sum()) if mask is not None else 0
        inlier_ratio = inlier_count / len(good_matches)
        
        if inlier_ratio < 0.2:
            return TrackResult(lost=True, num_matches=len(good_matches),
                             inlier_ratio=inlier_ratio)
        
        # Transform reference center through homography
        # The reference center is relative to the ROI, so we use (roi_w/2, roi_h/2)
        roi_center = np.array([[
            [self._ref_center[0] - self._roi_offset[0],
             self._ref_center[1] - self._roi_offset[1]]
        ]], dtype=np.float32)
        
        new_center_in_search = cv2.perspectiveTransform(roi_center, H)
        
        # Convert from search region coordinates to frame coordinates
        new_cx = float(new_center_in_search[0][0][0]) + sx1
        new_cy = float(new_center_in_search[0][0][1]) + sy1
        
        # Sanity check — new center should be within the frame
        if not (0 <= new_cx <= w and 0 <= new_cy <= h):
            return TrackResult(lost=True, num_matches=len(good_matches),
                             inlier_ratio=inlier_ratio)

        # --- Smoothing ---------------------------------------------------
        # Adaptive One Euro filter on the raw measured centre. This removes
        # per-frame ORB/homography jitter while staying responsive to real
        # motion, producing a stable set-point for the stepper motor.
        now = time.monotonic()
        dt = (now - self._last_track_time) if self._last_track_time else 0.0
        self._last_track_time = now
        sm_cx = self._filter_x(new_cx, dt)
        sm_cy = self._filter_y(new_cy, dt)

        # Deltas are derived from the SMOOTHED centre so downstream motion
        # commands are clean; the tracker itself keeps following the raw
        # measurement for search-region/reference bookkeeping.
        if self._smoothed_center is not None:
            dx = sm_cx - self._smoothed_center[0]
            dy = sm_cy - self._smoothed_center[1]
        else:
            dx = 0.0
            dy = 0.0
        self._smoothed_center = (sm_cx, sm_cy)
        
        # Transform ROI corners through homography for visualization
        ref_corners_roi = self._ref_corners.copy()
        ref_corners_roi[:, 0] -= self._roi_offset[0]
        ref_corners_roi[:, 1] -= self._roi_offset[1]
        ref_corners_roi = ref_corners_roi.reshape(-1, 1, 2)
        
        new_corners_search = cv2.perspectiveTransform(ref_corners_roi, H)
        new_corners = new_corners_search.reshape(-1, 2)
        new_corners[:, 0] += sx1
        new_corners[:, 1] += sy1
        
        # Update state for next frame (raw measurement drives tracking)
        self._prev_center = (new_cx, new_cy)
        self._successful_track_count += 1
        
        # Adaptive search region (based on raw measured motion)
        if self._prev_raw_center is not None:
            motion = np.sqrt((new_cx - self._prev_raw_center[0])**2 +
                             (new_cy - self._prev_raw_center[1])**2)
        else:
            motion = 0.0
        self._prev_raw_center = (new_cx, new_cy)
        if motion > self.MOTION_THRESHOLD:
            self._current_search_multiplier = min(
                self.MAX_SEARCH_MULTIPLIER,
                self._current_search_multiplier * self.ADAPTIVE_SEARCH_GROWTH
            )
        else:
            self._current_search_multiplier = max(
                self.MIN_SEARCH_MULTIPLIER,
                self._current_search_multiplier * self.ADAPTIVE_SEARCH_SHRINK
            )
        
        # Adaptive reference update
        if (self._successful_track_count % self.REFERENCE_UPDATE_INTERVAL == 0
                and inlier_ratio >= self.REFERENCE_UPDATE_MIN_INLIER_RATIO):
            self._update_reference(frame, new_cx, new_cy)
        
        return TrackResult(
            dx=dx,
            dy=dy,
            center=(sm_cx, sm_cy),
            corners=new_corners,
            lost=False,
            inlier_ratio=inlier_ratio,
            num_matches=len(good_matches)
        )
    
    def _update_reference(self, frame, cx, cy):
        """Update reference features from the current tracked position."""
        h, w = frame.shape[:2]
        half = self._roi_size // 2
        
        x1 = max(0, int(cx) - half)
        y1 = max(0, int(cy) - half)
        x2 = min(w, x1 + self._roi_size)
        y2 = min(h, y1 + self._roi_size)
        
        if x2 - x1 < 20 or y2 - y1 < 20:
            return
        
        roi = frame[y1:y2, x1:x2]
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = self._orb.detectAndCompute(gray_roi, None)
        
        if descriptors is None or len(keypoints) < self.MIN_GOOD_MATCHES:
            return
        
        self._ref_roi_kp_coords = np.array([kp.pt for kp in keypoints], dtype=np.float32)
        self._ref_keypoints = keypoints
        self._ref_descriptors = descriptors
        self._ref_center = (cx, cy)
        self._ref_corners = np.array([
            [x1, y1], [x2, y1], [x2, y2], [x1, y2]
        ], dtype=np.float32)
        self._roi_offset = (x1, y1)
        
        logger.debug(f"Reference updated at ({cx:.1f}, {cy:.1f}) with {len(keypoints)} features")
    
    def reset(self):
        """Clear tracking state. User must click again to select a new target."""
        self._ref_keypoints = None
        self._ref_descriptors = None
        self._ref_roi_kp_coords = None
        self._ref_center = None
        self._ref_corners = None
        self._prev_center = None
        self._successful_track_count = 0
        self._current_search_multiplier = self.SEARCH_REGION_MULTIPLIER
        self._active = False
        self._filter_x.reset()
        self._filter_y.reset()
        self._last_track_time = None
        self._smoothed_center = None
        self._prev_raw_center = None
