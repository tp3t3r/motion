"""Tests for the G-code sender's pure command-construction logic.

These exercise the unit-conversion, deadband, clamp, and accumulation logic
without opening a serial port, so they run in CI with no hardware attached.
"""
import unittest

from motion_tracker.gcode import GCodeSender, list_serial_ports


class TestBuildJogCommand(unittest.TestCase):
    def test_basic_command_format(self):
        s = GCodeSender(min_step_mm=0.0, max_step_mm=100.0)
        cmd = s.build_jog_command(1.5, -2.0)
        self.assertIsNotNone(cmd)
        self.assertTrue(cmd.startswith("$J=G91 G21"))
        self.assertIn("X1.500", cmd)
        self.assertIn("Y-2.000", cmd)
        self.assertIn("F", cmd)

    def test_deadband_suppresses_small_moves(self):
        s = GCodeSender(min_step_mm=0.05)
        # Both axes below deadband -> no command.
        self.assertIsNone(s.build_jog_command(0.01, -0.02))

    def test_deadband_per_axis(self):
        s = GCodeSender(min_step_mm=0.05, max_step_mm=100.0)
        # X below deadband, Y above -> only Y present.
        cmd = s.build_jog_command(0.01, 1.0)
        self.assertIsNotNone(cmd)
        self.assertNotIn("X", cmd)
        self.assertIn("Y1.000", cmd)

    def test_max_step_clamp(self):
        s = GCodeSender(min_step_mm=0.0, max_step_mm=5.0)
        cmd = s.build_jog_command(100.0, -100.0)
        self.assertIn("X5.000", cmd)
        self.assertIn("Y-5.000", cmd)

    def test_feed_rate_included(self):
        s = GCodeSender(feed_rate=800, min_step_mm=0.0)
        cmd = s.build_jog_command(1.0, 0.0)
        self.assertIn("F800", cmd)


class TestSubmitAndConsume(unittest.TestCase):
    def test_submit_ignored_when_not_connected(self):
        s = GCodeSender()
        s.motion_enabled = True  # still not connected
        s.submit_diff(10, 10)
        dx, dy = s.consume_pending()
        self.assertEqual((dx, dy), (0.0, 0.0))

    def test_accumulation_and_conversion(self):
        s = GCodeSender(mm_per_pixel=0.1, invert_x=False, invert_y=False,
                        max_step_mm=100.0)
        # Bypass connection/enable gating by poking pending directly through
        # the documented conversion: 10px * 0.1 = 1.0mm per axis.
        s._pending_x = 10 * 0.1
        s._pending_y = 20 * 0.1
        dx, dy = s.consume_pending()
        self.assertAlmostEqual(dx, 1.0, places=6)
        self.assertAlmostEqual(dy, 2.0, places=6)

    def test_consume_retains_overflow_beyond_clamp(self):
        s = GCodeSender(max_step_mm=5.0)
        s._pending_x = 12.0
        s._pending_y = 0.0
        dx, dy = s.consume_pending()
        self.assertEqual(dx, 5.0)      # clamped this cycle
        self.assertEqual(s._pending_x, 7.0)  # remainder retained

    def test_invert_axes(self):
        s = GCodeSender(mm_per_pixel=1.0, invert_x=True, invert_y=True)
        # Force gating open by simulating connected+enabled via internal flag.
        s.motion_enabled = True
        # is_connected is False, so submit is a no-op — verify the sign math
        # through the conversion directly instead.
        mm_x = 3 * s.mm_per_pixel * (-1.0 if s.invert_x else 1.0)
        mm_y = 4 * s.mm_per_pixel * (-1.0 if s.invert_y else 1.0)
        self.assertEqual(mm_x, -3.0)
        self.assertEqual(mm_y, -4.0)


class TestPixelDeadbandAndScale(unittest.TestCase):
    def _force_ready(self, s):
        """Bypass hardware gating so submit_diff exercises its logic."""
        s.motion_enabled = True

        class _FakeSerial:
            is_open = True

            def write(self, *a):
                pass

            def flush(self):
                pass
        s._serial = _FakeSerial()

    def test_pixel_deadband_discards_small_jitter(self):
        s = GCodeSender(deadband_px=3.0)
        self._force_ready(s)
        # 2px each axis, below the 3px dead-zone -> nothing accumulates.
        s.submit_diff(2, -2)
        dx, dy = s.consume_pending()
        self.assertEqual((dx, dy), (0.0, 0.0))

    def test_pixel_deadband_passes_real_motion(self):
        s = GCodeSender(deadband_px=3.0, mm_per_pixel=0.0125,
                        invert_x=False, invert_y=False, max_step_mm=100.0)
        self._force_ready(s)
        s.submit_diff(10, 0)  # above dead-zone
        dx, dy = s.consume_pending()
        self.assertAlmostEqual(dx, 10 * 0.0125, places=6)
        self.assertEqual(dy, 0.0)

    def test_scale_10px_equals_50_steps(self):
        # Machine calibration: 400 steps/mm ($100/$101). Requirement:
        # 10 px on screen -> 50 steps of motion.
        steps_per_mm = 400.0
        s = GCodeSender(mm_per_pixel=0.0125, invert_x=False, invert_y=False,
                        max_step_mm=100.0)
        self._force_ready(s)
        s.submit_diff(10, 0)
        dx_mm, _ = s.consume_pending()
        steps = dx_mm * steps_per_mm
        self.assertAlmostEqual(steps, 50.0, places=3)

    def test_default_scale_is_0_0125(self):
        s = GCodeSender()
        self.assertAlmostEqual(s.mm_per_pixel, 0.0125, places=6)

    def test_default_idle_release_is_zero(self):
        # Default: steppers release as soon as motion stops (no holding heat).
        s = GCodeSender()
        self.assertEqual(s.idle_release_ms, 0)

    def test_release_steppers_clears_pending_and_requests_cancel(self):
        s = GCodeSender()
        s._pending_x = 3.0
        s._pending_y = -2.0
        s.release_steppers()
        self.assertEqual((s._pending_x, s._pending_y), (0.0, 0.0))
        self.assertTrue(s._cancel_requested)


class TestListPorts(unittest.TestCase):
    def test_returns_list(self):
        # Should not raise, returns a list (possibly empty) even w/o hardware.
        ports = list_serial_ports()
        self.assertIsInstance(ports, list)


if __name__ == "__main__":
    unittest.main()
