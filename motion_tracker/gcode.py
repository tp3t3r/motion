"""G-code sending for a GRBL-based CNC shield over USB serial.

Translates tracked (dx, dy) pixel diffs into GRBL relative jog commands and
streams them to the controller on a background thread so serial I/O never
blocks the GUI capture loop.

Safety model
------------
Driving physical hardware is irreversible, so motion is conservative by
design:

* Nothing moves until :meth:`GCodeSender.connect` succeeds AND
  :attr:`GCodeSender.motion_enabled` is set True (both are explicit,
  user-initiated actions in the GUI).
* Every queued move is clamped to ``max_step_mm`` per axis.
* Diffs are accumulated and emitted at a bounded rate so the GRBL planner
  buffer is never flooded.
* Disconnecting sends a jog-cancel (real-time ``0x85``) and a feed hold to
  stop motion promptly.

GRBL jog reference: ``$J=G91 G21 X<dx> Y<dy> F<feed>`` performs a relative
(G91) jog in millimetres (G21). Jog moves can be cancelled instantly with
the real-time byte ``0x85`` without a full reset.
"""

import logging
import threading
import queue
import time

try:
    import serial
    import serial.tools.list_ports as list_ports
    _SERIAL_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only without pyserial
    serial = None
    list_ports = None
    _SERIAL_AVAILABLE = False

logger = logging.getLogger(__name__)

# GRBL real-time command bytes
GRBL_JOG_CANCEL = b"\x85"
GRBL_FEED_HOLD = b"!"
GRBL_SOFT_RESET = b"\x18"  # Ctrl-X


def list_serial_ports():
    """Return a list of available serial port device names.

    On macOS these look like ``/dev/cu.usbserial-XXXX``; on Windows they
    look like ``COM3``. Bluetooth/debug pseudo-ports are filtered out.
    """
    if not _SERIAL_AVAILABLE:
        return []
    ports = []
    for p in list_ports.comports():
        dev = p.device
        low = dev.lower()
        if "bluetooth" in low or "debug-console" in low:
            continue
        ports.append(dev)
    return ports


class GCodeSender:
    """Streams GRBL jog commands derived from tracked pixel diffs.

    Args:
        mm_per_pixel: Conversion from image pixels to millimetres of travel.
        feed_rate: Jog feed rate in mm/min.
        max_step_mm: Per-command clamp (per axis) in millimetres — a hard
            safety limit on how far a single jog can command.
        min_step_mm: Deadband; accumulated motion below this (per axis) is
            not sent, preventing chatter from sub-pixel jitter.
        send_interval: Minimum seconds between emitted jog commands (rate
            limit to protect the GRBL planner buffer).
        invert_x / invert_y: Flip axis direction to match machine geometry.
    """

    def __init__(self, mm_per_pixel=0.0125, feed_rate=500.0,
                 max_step_mm=5.0, min_step_mm=0.02, send_interval=0.05,
                 deadband_px=3.0, idle_release_ms=0, invert_x=False,
                 invert_y=True):
        self.mm_per_pixel = float(mm_per_pixel)
        self.feed_rate = float(feed_rate)
        self.max_step_mm = float(max_step_mm)
        self.min_step_mm = float(min_step_mm)
        self.send_interval = float(send_interval)
        # Pixel-space dead-zone: per-frame diffs whose magnitude is below
        # this (per axis) are discarded rather than accumulated. This is the
        # primary anti-jiggle guard — it stops residual sub-pixel tracking
        # noise from slowly leaking into periodic twitches.
        self.deadband_px = float(deadband_px)
        # Steppers de-energize after this many ms of idle (GRBL $1). 0 =
        # release as soon as motion stops (no holding current / no heat).
        self.idle_release_ms = int(idle_release_ms)
        # Image Y grows downward; most machines want +Y up, so invert Y
        # by default. Adjust to match the physical rig.
        self.invert_x = bool(invert_x)
        self.invert_y = bool(invert_y)

        self.motion_enabled = False

        self._serial = None
        self._port = None
        self._lock = threading.Lock()

        # Accumulated, not-yet-sent motion in millimetres.
        self._pending_x = 0.0
        self._pending_y = 0.0
        self._cancel_requested = False

        self._tx_thread = None
        self._stop_event = threading.Event()
        self._last_send = 0.0

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------

    @property
    def is_connected(self):
        return self._serial is not None and self._serial.is_open

    @property
    def port(self):
        return self._port

    def connect(self, port, baud=115200, timeout=1.0):
        """Open the serial port and start the background sender thread.

        Returns True on success. Does not enable motion; the caller must set
        :attr:`motion_enabled` explicitly.
        """
        if not _SERIAL_AVAILABLE:
            raise RuntimeError(
                "pyserial is not installed. Run: pip install pyserial")
        self.disconnect()
        try:
            self._serial = serial.Serial(port, baud, timeout=timeout)
        except serial.SerialException as e:
            logger.error(f"Failed to open serial port {port}: {e}")
            self._serial = None
            return False

        self._port = port
        # GRBL emits a banner shortly after the port opens; give it a moment
        # and flush so our first command isn't concatenated with the banner.
        time.sleep(2.0)
        try:
            self._serial.reset_input_buffer()
        except Exception:  # pragma: no cover - defensive
            pass

        # Configure prompt idle-release of the steppers. GRBL's $1 is the
        # "step idle delay" in ms: values < 255 let the motors de-energize
        # after that many ms of idle (no holding current / no heat); $1=255
        # keeps them locked forever. We set it to `idle_release_ms` so that
        # whenever we stop commanding motion (lost tracking / disconnect),
        # the steppers release on their own. Written only if it differs, to
        # avoid needless EEPROM wear.
        self._configure_idle_release()

        self._pending_x = 0.0
        self._pending_y = 0.0
        self._cancel_requested = False
        self._stop_event.clear()
        self._tx_thread = threading.Thread(
            target=self._tx_loop, name="gcode-tx", daemon=True)
        self._tx_thread.start()
        logger.info(f"Connected to CNC controller on {port} @ {baud} baud")
        return True

    def _configure_idle_release(self):
        """Ensure GRBL's $1 step-idle-delay releases the steppers promptly.

        Reads the current $1; if it already matches the desired value we do
        nothing (avoids an EEPROM write on every connect). $1=255 would hold
        the motors indefinitely, which we explicitly avoid.
        """
        if self._serial is None:
            return
        try:
            self._serial.reset_input_buffer()
            self._serial.write(b"$$\n")
            time.sleep(0.5)
            dump = self._serial.read(self._serial.in_waiting or 1).decode(
                errors="replace")
            current = None
            for line in dump.splitlines():
                if line.startswith("$1="):
                    current = line.strip().split("=", 1)[1]
                    break
            desired = str(int(self.idle_release_ms))
            if current != desired:
                self._serial.write(f"$1={desired}\n".encode("ascii"))
                self._serial.flush()
                time.sleep(0.2)
                self._serial.reset_input_buffer()
                logger.info(f"Set GRBL $1={desired} (was {current}) for "
                            f"idle stepper release")
            else:
                logger.info(f"GRBL $1 already {current}; steppers release "
                            f"when idle")
        except Exception as e:  # pragma: no cover - defensive
            logger.warning(f"Could not configure $1 idle release: {e}")

    def disconnect(self):
        """Stop motion, stop the sender thread, and close the port."""
        self.motion_enabled = False
        self._stop_event.set()
        if self._tx_thread is not None:
            self._tx_thread.join(timeout=1.0)
            self._tx_thread = None
        if self._serial is not None:
            try:
                if self._serial.is_open:
                    # Cancel any in-progress jog and hold, then flush.
                    self._serial.write(GRBL_JOG_CANCEL)
                    self._serial.write(GRBL_FEED_HOLD)
                    self._serial.flush()
            except Exception:  # pragma: no cover - defensive
                pass
            try:
                self._serial.close()
            except Exception:  # pragma: no cover - defensive
                pass
            logger.info(f"Disconnected from {self._port}")
        self._serial = None
        self._port = None

    # ------------------------------------------------------------------
    # Motion input
    # ------------------------------------------------------------------

    def submit_diff(self, dx_px, dy_px):
        """Feed a tracked pixel diff. Accumulated and sent by the tx thread.

        No-op unless connected and motion is enabled. A per-axis pixel
        dead-zone discards sub-threshold jitter so it cannot accumulate into
        periodic twitches. Surviving motion is converted to millimetres and
        axis inversion is applied.
        """
        if not self.is_connected or not self.motion_enabled:
            return
        # Pixel-space dead-zone (anti-jiggle). Discard, do not accumulate.
        if abs(dx_px) < self.deadband_px:
            dx_px = 0.0
        if abs(dy_px) < self.deadband_px:
            dy_px = 0.0
        if dx_px == 0.0 and dy_px == 0.0:
            return
        mm_x = dx_px * self.mm_per_pixel * (-1.0 if self.invert_x else 1.0)
        mm_y = dy_px * self.mm_per_pixel * (-1.0 if self.invert_y else 1.0)
        with self._lock:
            self._pending_x += mm_x
            self._pending_y += mm_y

    def stop_motion(self):
        """Request an immediate jog-cancel and clear pending motion.

        Non-blocking: this only clears the accumulator and raises a flag.
        The background tx thread performs the actual serial jog-cancel, so
        this is safe to call from the GUI thread every frame without risking
        a stall on serial I/O.
        """
        with self._lock:
            self._pending_x = 0.0
            self._pending_y = 0.0
            self._cancel_requested = True

    def release_steppers(self):
        """Release the steppers (drop holding current) on lost track/connection.

        With GRBL, motors de-energize automatically after $1 ms of idle (we
        set $1 at connect time via idle_release_ms). So releasing == making
        the machine go idle immediately: cancel any active jog and stop
        commanding motion. Non-blocking — the actual serial cancel is issued
        by the tx thread. Once idle, GRBL drops the holding current.
        """
        self.stop_motion()

    # ------------------------------------------------------------------
    # Command construction (pure, unit-testable)
    # ------------------------------------------------------------------

    def build_jog_command(self, dx_mm, dy_mm):
        """Build a GRBL relative jog command string, or None if below deadband.

        Applies the per-axis deadband and the per-axis max-step clamp.
        Returns a command WITHOUT trailing newline, or None if nothing to do.
        """
        # Deadband: ignore sub-threshold motion on both axes.
        if abs(dx_mm) < self.min_step_mm:
            dx_mm = 0.0
        if abs(dy_mm) < self.min_step_mm:
            dy_mm = 0.0
        if dx_mm == 0.0 and dy_mm == 0.0:
            return None

        # Safety clamp per axis.
        dx_mm = max(-self.max_step_mm, min(self.max_step_mm, dx_mm))
        dy_mm = max(-self.max_step_mm, min(self.max_step_mm, dy_mm))

        parts = ["$J=G91", "G21"]
        if dx_mm != 0.0:
            parts.append(f"X{dx_mm:.3f}")
        if dy_mm != 0.0:
            parts.append(f"Y{dy_mm:.3f}")
        parts.append(f"F{self.feed_rate:.0f}")
        return " ".join(parts)

    def consume_pending(self):
        """Atomically take the accumulated motion, clamped to max_step_mm.

        Returns (dx_mm, dy_mm). Any motion beyond the clamp is retained so
        it is emitted on subsequent cycles rather than lost.
        """
        with self._lock:
            dx = self._pending_x
            dy = self._pending_y
            send_x = max(-self.max_step_mm, min(self.max_step_mm, dx))
            send_y = max(-self.max_step_mm, min(self.max_step_mm, dy))
            self._pending_x = dx - send_x
            self._pending_y = dy - send_y
        return send_x, send_y

    # ------------------------------------------------------------------
    # Background sender
    # ------------------------------------------------------------------

    def _tx_loop(self):
        while not self._stop_event.is_set():
            # Handle a pending jog-cancel first (from stop_motion / lost track).
            # Done here so all serial I/O stays on this thread.
            with self._lock:
                cancel = self._cancel_requested
                self._cancel_requested = False
            if cancel:
                self._write_bytes(GRBL_JOG_CANCEL)

            now = time.monotonic()
            if now - self._last_send < self.send_interval:
                time.sleep(0.005)
                continue
            if not self.motion_enabled:
                time.sleep(0.02)
                continue
            dx, dy = self.consume_pending()
            cmd = self.build_jog_command(dx, dy)
            if cmd is None:
                time.sleep(0.005)
                continue
            self._write_line(cmd)
            self._last_send = now

    def _write_bytes(self, data):
        if not self.is_connected:
            return
        try:
            self._serial.write(data)
            self._serial.flush()
        except Exception as e:  # pragma: no cover - hardware failure path
            logger.error(f"Serial write failed: {e}")

    def _write_line(self, line):
        if not self.is_connected:
            return
        try:
            self._serial.write((line + "\n").encode("ascii"))
            self._serial.flush()
            logger.debug(f"TX: {line}")
        except Exception as e:  # pragma: no cover - hardware failure path
            logger.error(f"Serial write failed: {e}")
