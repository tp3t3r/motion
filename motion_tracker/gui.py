import tkinter as tk
import cv2
import numpy as np
import math
from PIL import Image, ImageTk
import logging

from .tracker import ORBTracker, TrackResult
from .gcode import GCodeSender, list_serial_ports

logger = logging.getLogger(__name__)

# Exposure slider steps in milliseconds (logarithmic feel)
EXPOSURE_STEPS = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 30000]


def snap_to_step(value):
    """Snap a value to the nearest exposure step."""
    idx = min(range(len(EXPOSURE_STEPS)),
             key=lambda i: abs(EXPOSURE_STEPS[i] - value))
    return EXPOSURE_STEPS[idx]


def enumerate_cameras(max_index=5):
    """Probe camera indices and return list of available ones."""
    available = []
    for i in range(max_index):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                available.append(i)
            cap.release()
    return available


class TrackerApp:
    """Main GUI application for Motion Tracker.

    Uses a Canvas widget for image display so that click coordinates
    map exactly to image pixels (no centering offset issues).
    """

    def __init__(self):
        self._root = tk.Tk()
        self._root.title("Motion Tracker")
        self._root.geometry("960x640")
        self._root.minsize(640, 480)
        self._root.configure(bg="#2e2e2e")
        self._root.protocol("WM_DELETE_WINDOW", self._on_close)

        # Camera state
        self._cap = None
        self._camera_indices = []
        self._exposure_ms = 100

        # Tracker
        self._tracker = ORBTracker()
        # CNC / G-code output (motion disabled until user connects + enables)
        self._gcode = GCodeSender()
        self._was_lost = True
        self._current_frame = None   # raw BGR frame (full resolution)
        self._photo_image = None     # prevent GC
        self._display_scale = 1.0    # frame-pixel = display-pixel / scale
        self._img_offset_x = 0       # image top-left X on canvas
        self._img_offset_y = 0       # image top-left Y on canvas
        self._display_w = 0
        self._display_h = 0
        self._ui_ready = False

        self._build_ui()
        self._ui_ready = True

        self._exposure_slider.set(EXPOSURE_STEPS.index(100))
        self._root.after(50, self._refresh_cameras)
        self._root.after(60, self._refresh_ports)

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def _build_ui(self):
        bg = "#2e2e2e"
        fg = "#e0e0e0"
        btn_bg = "#4a4a4a"
        font = ("Helvetica", 12)
        sfont = ("Helvetica", 10)

        # --- Controls ---
        ctrl = tk.Frame(self._root, bg=bg, padx=5, pady=5)
        ctrl.pack(fill=tk.X, side=tk.TOP)

        tk.Label(ctrl, text="Camera:", bg=bg, fg=fg, font=font).pack(
            side=tk.LEFT, padx=(0, 5))
        self._cam_var = tk.StringVar(value="(none)")
        self._cam_menu = tk.OptionMenu(ctrl, self._cam_var, "(none)")
        self._cam_menu.config(width=12, font=sfont, bg=btn_bg, fg=fg,
                               activebackground="#5a5a5a", highlightthickness=0)
        self._cam_menu["menu"].config(bg="#3c3c3c", fg=fg, font=sfont)
        self._cam_menu.pack(side=tk.LEFT, padx=(0, 5))
        tk.Button(ctrl, text="\u21BB", width=3, font=font, bg=btn_bg, fg=fg,
                  activebackground="#5a5a5a", highlightthickness=0,
                  command=self._refresh_cameras).pack(side=tk.LEFT, padx=(0, 15))

        tk.Label(ctrl, text="Exposure:", bg=bg, fg=fg, font=font).pack(
            side=tk.LEFT, padx=(0, 5))
        self._exposure_slider = tk.Scale(
            ctrl, from_=0, to=len(EXPOSURE_STEPS) - 1,
            orient=tk.HORIZONTAL, length=200, showvalue=False,
            bg=bg, fg=fg, troughcolor="#3c3c3c", highlightthickness=0,
            command=self._on_exposure_changed)
        self._exposure_slider.pack(side=tk.LEFT, padx=(0, 5))
        self._exp_label = tk.Label(ctrl, text="100 ms", width=10, bg=bg,
                                    fg="#00ff00", font=font)
        self._exp_label.pack(side=tk.LEFT, padx=(0, 5))
        self._actual_label = tk.Label(ctrl, text="", width=14, bg=bg,
                                       fg="#999999", font=sfont)
        self._actual_label.pack(side=tk.LEFT)
        tk.Button(ctrl, text="Stop Tracking", font=sfont, bg=btn_bg, fg=fg,
                  activebackground="#5a5a5a", highlightthickness=0,
                  command=self._on_stop_tracking).pack(side=tk.RIGHT, padx=(10, 0))

        # --- CNC / G-code control row ---
        cnc = tk.Frame(self._root, bg=bg, padx=5, pady=5)
        cnc.pack(fill=tk.X, side=tk.TOP)

        tk.Label(cnc, text="CNC Port:", bg=bg, fg=fg, font=font).pack(
            side=tk.LEFT, padx=(0, 5))
        self._port_var = tk.StringVar(value="(none)")
        self._port_menu = tk.OptionMenu(cnc, self._port_var, "(none)")
        self._port_menu.config(width=18, font=sfont, bg=btn_bg, fg=fg,
                               activebackground="#5a5a5a", highlightthickness=0)
        self._port_menu["menu"].config(bg="#3c3c3c", fg=fg, font=sfont)
        self._port_menu.pack(side=tk.LEFT, padx=(0, 5))
        tk.Button(cnc, text="\u21BB", width=3, font=font, bg=btn_bg, fg=fg,
                  activebackground="#5a5a5a", highlightthickness=0,
                  command=self._refresh_ports).pack(side=tk.LEFT, padx=(0, 8))

        self._connect_btn = tk.Button(
            cnc, text="Connect", font=sfont, bg=btn_bg, fg=fg,
            activebackground="#5a5a5a", highlightthickness=0,
            command=self._on_connect_toggle)
        self._connect_btn.pack(side=tk.LEFT, padx=(0, 12))

        tk.Label(cnc, text="mm/px:", bg=bg, fg=fg, font=sfont).pack(
            side=tk.LEFT, padx=(0, 3))
        self._scale_var = tk.StringVar(value=f"{self._gcode.mm_per_pixel:g}")
        scale_entry = tk.Entry(cnc, textvariable=self._scale_var, width=6,
                               font=sfont, bg="#3c3c3c", fg=fg,
                               insertbackground=fg, highlightthickness=0)
        scale_entry.pack(side=tk.LEFT, padx=(0, 3))
        scale_entry.bind("<Return>", self._on_scale_changed)
        scale_entry.bind("<FocusOut>", self._on_scale_changed)

        tk.Label(cnc, text="deadband px:", bg=bg, fg=fg, font=sfont).pack(
            side=tk.LEFT, padx=(8, 3))
        self._deadband_var = tk.StringVar(value=f"{self._gcode.deadband_px:g}")
        db_entry = tk.Entry(cnc, textvariable=self._deadband_var, width=5,
                            font=sfont, bg="#3c3c3c", fg=fg,
                            insertbackground=fg, highlightthickness=0)
        db_entry.pack(side=tk.LEFT, padx=(0, 3))
        db_entry.bind("<Return>", self._on_deadband_changed)
        db_entry.bind("<FocusOut>", self._on_deadband_changed)

        self._motion_var = tk.BooleanVar(value=False)
        self._motion_check = tk.Checkbutton(
            cnc, text="Enable motion", variable=self._motion_var,
            command=self._on_motion_toggle, bg=bg, fg="#ff8080",
            selectcolor="#3c3c3c", activebackground=bg,
            activeforeground="#ff8080", font=sfont, highlightthickness=0,
            state=tk.DISABLED)
        self._motion_check.pack(side=tk.LEFT, padx=(10, 0))

        self._cnc_status = tk.Label(cnc, text="Disconnected", bg=bg,
                                    fg="#999999", font=sfont)
        self._cnc_status.pack(side=tk.RIGHT)

        # --- Status bar (before canvas so it claims bottom space) ---
        sbar = tk.Frame(self._root, bg="#1e1e1e", padx=5, pady=3)
        sbar.pack(fill=tk.X, side=tk.BOTTOM)
        self._status = tk.StringVar(value="Starting\u2026")
        tk.Label(sbar, textvariable=self._status, anchor=tk.W, bg="#1e1e1e",
                 fg=fg, font=sfont).pack(side=tk.LEFT, fill=tk.X, expand=True)
        self._fps = tk.StringVar(value="")
        tk.Label(sbar, textvariable=self._fps, anchor=tk.E, width=18,
                 bg="#1e1e1e", fg="#00cccc", font=sfont).pack(side=tk.RIGHT)

        # --- Canvas for image (click coordinates = image coordinates) ---
        self._canvas = tk.Canvas(self._root, bg="black", highlightthickness=0,
                                  cursor="crosshair")
        self._canvas.pack(fill=tk.BOTH, expand=True)
        self._canvas_img_id = self._canvas.create_image(0, 0, anchor=tk.NW)
        self._canvas.bind("<Button-1>", self._on_canvas_click)
        self._canvas.bind("<Configure>", self._on_canvas_resize)

    # ------------------------------------------------------------------
    # Camera management
    # ------------------------------------------------------------------

    def _refresh_cameras(self):
        self._status.set("Scanning for cameras\u2026")
        self._root.update_idletasks()
        self._camera_indices = enumerate_cameras()
        menu = self._cam_menu["menu"]
        menu.delete(0, tk.END)
        if self._camera_indices:
            for idx in self._camera_indices:
                menu.add_command(label=f"Camera {idx}",
                                 command=lambda i=idx: self._switch_camera(i))
            self._switch_camera(self._camera_indices[0])
        else:
            self._cam_var.set("No cameras")
            self._status.set("No cameras detected. Plug one in and click \u21BB.")

    def _switch_camera(self, index):
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        self._tracker.reset()
        self._cam_var.set(f"Camera {index}")
        cap = cv2.VideoCapture(index)
        if not cap.isOpened():
            self._status.set(f"Failed to open Camera {index}")
            return
        self._cap = cap
        self._apply_exposure()
        self._status.set(f"Camera {index} active. Click on image to track.")
        self._grab()

    def _apply_exposure(self):
        """Apply the configured exposure, converting units per backend.

        Backends interpret CAP_PROP_EXPOSURE differently:
          * V4L2 (Linux): units of 100 microseconds; manual mode is
            CAP_PROP_AUTO_EXPOSURE = 1 (auto = 3).
          * DirectShow / MSMF (Windows): log2(seconds), e.g. -4 = 1/16 s;
            manual mode is CAP_PROP_AUTO_EXPOSURE = 0.25.
          * Others (e.g. AVFoundation on macOS): typically unsupported; we
            attempt best-effort and report if it doesn't take.
        """
        if self._cap is None:
            return

        backend = self._cap.getBackendName()
        ms = float(self._exposure_ms)
        seconds = ms / 1000.0

        if backend == "V4L2":
            # Manual mode = 1, auto = 3.
            self._cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
            raw = ms * 10.0  # 100us units
        elif backend in ("DSHOW", "MSMF"):
            # Manual mode magic value on these backends.
            self._cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
            # log2(seconds); clamp to a sane camera range.
            raw = round(math.log2(seconds)) if seconds > 0 else -4
            raw = max(-13, min(0, raw))
        else:
            # Unknown/unsupported backend (e.g. AVFoundation). Best effort.
            self._cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
            raw = ms

        ok = self._cap.set(cv2.CAP_PROP_EXPOSURE, raw)
        actual = self._cap.get(cv2.CAP_PROP_EXPOSURE)

        # Determine whether the setting actually took effect.
        supported = bool(ok) and actual not in (0.0, None)
        if supported:
            self._actual_label.config(text=f"(actual: {actual:.3g})",
                                      fg="#999999")
        else:
            self._actual_label.config(
                text=f"(not supported: {backend})", fg="#cc7a00")
        logger.info(
            f"Exposure {self._exposure_ms}ms -> {backend} raw={raw} "
            f"set_ok={ok} readback={actual} supported={supported}")

    # ------------------------------------------------------------------
    # Capture-display loop
    # ------------------------------------------------------------------

    def _grab(self):
        if self._cap is None or not self._cap.isOpened():
            return
        ret, frame = self._cap.read()
        if ret:
            # Invert the image on both axes (180° rotation). Done at capture
            # time so display, click coordinates, and tracking all operate in
            # the same flipped orientation.
            frame = cv2.flip(frame, -1)
            self._current_frame = frame.copy()
            display = frame.copy()
            if self._tracker.is_active:
                result = self._tracker.track(frame)
                display = self._draw_tracking(display, result)
                self._update_tracking_status(result)
                # Forward smoothed motion to the CNC controller. The sender
                # is a no-op unless connected and motion is enabled.
                if not result.lost:
                    self._gcode.submit_diff(result.dx, result.dy)
                    self._was_lost = False
                elif not self._was_lost:
                    # Tracking just lost: cancel jogging and let the steppers
                    # release (drop holding current) once GRBL goes idle.
                    self._gcode.release_steppers()
                    self._was_lost = True
            self._show_frame(display)
        self._root.after(max(30, int(self._exposure_ms)), self._grab)

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def _show_frame(self, frame):
        """Scale frame to fit the canvas, keeping aspect ratio, and display it."""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        fh, fw = rgb.shape[:2]

        cw = max(self._canvas.winfo_width(), 1)
        ch = max(self._canvas.winfo_height(), 1)

        self._display_scale = min(cw / fw, ch / fh)
        self._display_w = int(fw * self._display_scale)
        self._display_h = int(fh * self._display_scale)

        # Centre the image in the canvas
        self._img_offset_x = (cw - self._display_w) // 2
        self._img_offset_y = (ch - self._display_h) // 2

        rgb = cv2.resize(rgb, (self._display_w, self._display_h),
                         interpolation=cv2.INTER_AREA)

        pil = Image.fromarray(rgb)
        self._photo_image = ImageTk.PhotoImage(pil)
        self._canvas.itemconfig(self._canvas_img_id, image=self._photo_image)
        self._canvas.coords(self._canvas_img_id,
                            self._img_offset_x, self._img_offset_y)

    def _on_canvas_resize(self, event):
        """Redisplay on resize so the image stays centred."""
        if self._current_frame is not None:
            self._show_frame(self._current_frame)

    @staticmethod
    def _draw_dotted_line(frame, pt1, pt2, color, thickness=1,
                          dot_length=4, gap_length=6):
        """Draw a dotted straight line between pt1 and pt2 on `frame`.

        OpenCV has no native dotted line, so we step along the segment and
        draw short dashes separated by gaps. Works for any orientation, but
        is used here for purely horizontal / vertical axes.
        """
        x1, y1 = pt1
        x2, y2 = pt2
        length = int(np.hypot(x2 - x1, y2 - y1))
        if length == 0:
            return
        # Unit direction vector
        ux = (x2 - x1) / length
        uy = (y2 - y1) / length
        step = dot_length + gap_length
        pos = 0
        while pos < length:
            sx = int(round(x1 + ux * pos))
            sy = int(round(y1 + uy * pos))
            end = min(pos + dot_length, length)
            ex = int(round(x1 + ux * end))
            ey = int(round(y1 + uy * end))
            cv2.line(frame, (sx, sy), (ex, ey), color, thickness, cv2.LINE_AA)
            pos += step

    def _draw_tracking(self, frame, result):
        if result.lost:
            h, w = frame.shape[:2]
            ov = frame.copy()
            cv2.rectangle(ov, (0, 0), (w, 40), (0, 0, 180), -1)
            cv2.addWeighted(ov, 0.7, frame, 0.3, 0, frame)
            cv2.putText(frame, "TRACKING LOST - Click to re-select",
                        (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (255, 255, 255), 2, cv2.LINE_AA)
        else:
            if result.center is not None:
                cx, cy = int(result.center[0]), int(result.center[1])
                h, w = frame.shape[:2]
                # Vertical axis (full height) through the tracked centre
                self._draw_dotted_line(frame, (cx, 0), (cx, h - 1),
                                       (0, 255, 0), thickness=2)
                # Horizontal axis (full width) through the tracked centre
                self._draw_dotted_line(frame, (0, cy), (w - 1, cy),
                                       (0, 255, 0), thickness=2)
            cv2.putText(frame,
                        f"dx: {result.dx:+.1f}  dy: {result.dy:+.1f}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(frame,
                        f"matches: {result.num_matches}  "
                        f"inliers: {result.inlier_ratio:.0%}",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 200, 200), 1, cv2.LINE_AA)
        return frame

    def _update_tracking_status(self, result):
        if result.lost:
            self._status.set(
                f"LOST (matches: {result.num_matches}, "
                f"inliers: {result.inlier_ratio:.0%}) "
                f"\u2014 Click to re-select")
        else:
            self._status.set(
                f"Tracking: dx={result.dx:+.1f} dy={result.dy:+.1f}  |  "
                f"matches: {result.num_matches}  "
                f"inliers: {result.inlier_ratio:.0%}")

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    def _on_exposure_changed(self, value):
        if not self._ui_ready:
            return
        idx = max(0, min(int(round(float(value))), len(EXPOSURE_STEPS) - 1))
        self._exposure_ms = EXPOSURE_STEPS[idx]
        self._exp_label.config(text=f"{self._exposure_ms} ms")
        fps = 1000.0 / self._exposure_ms if self._exposure_ms > 0 else 0
        self._fps.set(f"~{fps:.1f} FPS")
        self._apply_exposure()

    def _on_canvas_click(self, event):
        if self._current_frame is None:
            return

        # Convert canvas pixel → frame pixel
        ix = event.x - self._img_offset_x
        iy = event.y - self._img_offset_y

        # Check within image bounds
        if ix < 0 or iy < 0 or ix >= self._display_w or iy >= self._display_h:
            return

        fx = ix / self._display_scale
        fy = iy / self._display_scale

        h, w = self._current_frame.shape[:2]
        fx = max(0.0, min(fx, w - 1))
        fy = max(0.0, min(fy, h - 1))

        ok = self._tracker.select_target(self._current_frame, fx, fy)
        if ok:
            self._status.set(
                f"Target selected at ({fx:.0f}, {fy:.0f}). Tracking\u2026")
        else:
            self._status.set(
                "No features at that location. Try a more textured area.")

    def _on_stop_tracking(self):
        self._tracker.reset()
        # Halt any commanded motion when tracking stops.
        self._gcode.stop_motion()
        self._status.set("Tracking stopped. Click on image to select target.")

    # ------------------------------------------------------------------
    # CNC / G-code callbacks
    # ------------------------------------------------------------------

    def _refresh_ports(self):
        ports = list_serial_ports()
        menu = self._port_menu["menu"]
        menu.delete(0, tk.END)
        if ports:
            for dev in ports:
                menu.add_command(
                    label=dev,
                    command=lambda d=dev: self._port_var.set(d))
            # Keep current selection if still present, else pick first.
            if self._port_var.get() not in ports:
                self._port_var.set(ports[0])
        else:
            self._port_var.set("(none)")

    def _on_connect_toggle(self):
        if self._gcode.is_connected:
            self._gcode.disconnect()
            self._motion_var.set(False)
            self._motion_check.config(state=tk.DISABLED)
            self._connect_btn.config(text="Connect")
            self._cnc_status.config(text="Disconnected", fg="#999999")
            self._status.set("CNC disconnected.")
            return

        port = self._port_var.get()
        if not port or port == "(none)":
            self._cnc_status.config(text="No port selected", fg="#ff8080")
            return
        try:
            ok = self._gcode.connect(port)
        except RuntimeError as e:
            self._cnc_status.config(text=str(e), fg="#ff8080")
            return
        if ok:
            self._connect_btn.config(text="Disconnect")
            self._motion_check.config(state=tk.NORMAL)
            self._cnc_status.config(text=f"Connected: {port}", fg="#00cc66")
            self._status.set(
                f"CNC connected on {port}. Enable motion to start jogging.")
        else:
            self._cnc_status.config(text=f"Failed to open {port}",
                                    fg="#ff8080")

    def _on_motion_toggle(self):
        enabled = bool(self._motion_var.get())
        self._gcode.motion_enabled = enabled
        if enabled:
            self._cnc_status.config(text="MOTION ENABLED", fg="#ff4040")
        else:
            self._gcode.stop_motion()
            port = self._gcode.port or ""
            self._cnc_status.config(text=f"Connected: {port}", fg="#00cc66")

    def _on_scale_changed(self, event=None):
        try:
            val = float(self._scale_var.get())
            if val > 0:
                self._gcode.mm_per_pixel = val
        except ValueError:
            self._scale_var.set(f"{self._gcode.mm_per_pixel:g}")

    def _on_deadband_changed(self, event=None):
        try:
            val = float(self._deadband_var.get())
            if val >= 0:
                self._gcode.deadband_px = val
        except ValueError:
            self._deadband_var.set(f"{self._gcode.deadband_px:g}")

    def _on_close(self):
        # Stop hardware first, then release camera and window.
        try:
            self._gcode.disconnect()
        except Exception:
            pass
        if self._cap is not None:
            self._cap.release()
        self._root.destroy()

    def run(self):
        self._root.mainloop()
