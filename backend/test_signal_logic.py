import unittest

from signal_logic import (
    ALL_RED_DURATION,
    RL_MIN_GREEN_STEPS,
    RL_MAX_GREEN_STEPS,
    RL_STATE_QUEUE_MAX,
    RL_STATE_ELAPSED_MAX,
    YELLOW_DURATION,
    SIGNAL_SEQUENCE,
    normalize_rl_state,
    rl_phase_pressure,
    rl_best_alternative_pressure,
    rl_should_switch,
    next_signal,
    next_signal_rl,
    rl_action_from_weights,
    parse_signal_phase,
    get_light_states,
    can_vehicle_proceed,
    _next_phase,
)


# ---------------------------------------------------------------------------
# normalize_rl_state
# ---------------------------------------------------------------------------

class NormalizeTests(unittest.TestCase):

    def test_zero_state_normalizes_to_zeros(self):
        normed = normalize_rl_state([0, 0, 0, 0, 0, 0])
        self.assertEqual(normed, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    def test_queue_clamped_at_max(self):
        normed = normalize_rl_state([999, 999, 999, 999, 0, 0])
        for i in range(4):
            self.assertAlmostEqual(normed[i], 1.0, places=5)

    def test_queue_midpoint(self):
        normed = normalize_rl_state([RL_STATE_QUEUE_MAX / 2, 0, 0, 0, 0, 0])
        self.assertAlmostEqual(normed[0], 0.5, places=5)

    def test_phase_index_normalized(self):
        normed = normalize_rl_state([0, 0, 0, 0, 3, 0])
        self.assertAlmostEqual(normed[4], 1.0, places=5)

    def test_elapsed_normalized_and_clamped(self):
        normed = normalize_rl_state([0, 0, 0, 0, 0, RL_STATE_ELAPSED_MAX])
        self.assertAlmostEqual(normed[5], 1.0, places=5)
        normed_over = normalize_rl_state([0, 0, 0, 0, 0, RL_STATE_ELAPSED_MAX * 2])
        self.assertAlmostEqual(normed_over[5], 1.0, places=5)

    def test_returns_original_if_too_short(self):
        short = [1, 2, 3]
        result = normalize_rl_state(short)
        self.assertEqual(result, short)


# ---------------------------------------------------------------------------
# Pressure helpers
# ---------------------------------------------------------------------------

class PressureTests(unittest.TestCase):

    def test_phase_pressure_ns_straight(self):
        state = [10, 5, 3, 2, 0, 0]
        self.assertEqual(rl_phase_pressure(state, "NS_straight"), 10.0)

    def test_best_alternative_excludes_current(self):
        state = [10, 8, 6, 4, 0, 0]
        # Current is NS_straight (10), best alternative should be EW_straight (8)
        self.assertEqual(rl_best_alternative_pressure(state, "NS_straight"), 8.0)

    def test_pressure_returns_zero_for_empty_state(self):
        self.assertEqual(rl_phase_pressure(None, "NS_straight"), 0.0)
        self.assertEqual(rl_best_alternative_pressure(None, "NS_straight"), 0.0)


# ---------------------------------------------------------------------------
# rl_should_switch
# ---------------------------------------------------------------------------

class ShouldSwitchTests(unittest.TestCase):

    def test_action_zero_never_switches(self):
        self.assertFalse(rl_should_switch([20, 1, 1, 1, 0, 0], "NS_straight", 0))

    def test_heavy_current_suppresses_switch(self):
        # NS queue (18) >> best alternative (2) → should NOT switch
        state = [18, 2, 1, 1, 0, 10]
        self.assertFalse(rl_should_switch(state, "NS_straight", 1))

    def test_equal_pressure_allows_switch(self):
        # NS and EW roughly equal → should switch
        state = [5, 5, 3, 3, 0, 15]
        self.assertTrue(rl_should_switch(state, "NS_straight", 1))

    def test_empty_queue_allows_switch(self):
        state = [0, 0, 0, 0, 0, 20]
        self.assertTrue(rl_should_switch(state, "NS_straight", 1))


# ---------------------------------------------------------------------------
# next_signal_rl
# ---------------------------------------------------------------------------

class RLSignalTests(unittest.TestCase):

    def test_min_green_lock_prevents_early_switch(self):
        result = next_signal_rl(
            current_phase="NS_straight", phase_type="green",
            elapsed=2, dt=1, requested_action=1,
        )
        self.assertEqual(result["phase"], "NS_straight")
        self.assertEqual(result["phase_type"], "green")
        self.assertAlmostEqual(result["elapsed"], 3.0)

    def test_max_green_forces_switch_regardless_of_action(self):
        result = next_signal_rl(
            current_phase="NS_straight", phase_type="green",
            elapsed=61, dt=1, requested_action=0,
        )
        self.assertEqual(result["phase"], "NS_straight")
        self.assertEqual(result["phase_type"], "yellow")
        self.assertAlmostEqual(result["elapsed"], 0.0)

    def test_valid_switch_request_transitions_to_yellow(self):
        # Equal pressure — switch should be allowed
        state = [3, 4, 2, 2, 0, 15]
        result = next_signal_rl(
            current_phase="NS_straight", phase_type="green",
            elapsed=15, dt=1, requested_action=1, state=state,
        )
        self.assertEqual(result["phase_type"], "yellow")
        self.assertAlmostEqual(result["elapsed"], 0.0)

    def test_heavy_current_queue_suppresses_switch(self):
        state = [18, 2, 1, 1, 0, 20]
        result = next_signal_rl(
            current_phase="NS_straight", phase_type="green",
            elapsed=20, dt=1, requested_action=1, state=state,
        )
        self.assertEqual(result["phase"], "NS_straight")
        self.assertEqual(result["phase_type"], "green")

    def test_yellow_holds_until_duration(self):
        result = next_signal_rl(
            current_phase="NS_straight", phase_type="yellow",
            elapsed=1.0, dt=0.5, requested_action=0,
        )
        self.assertEqual(result["phase_type"], "yellow")

    def test_yellow_to_all_red(self):
        result = next_signal_rl(
            current_phase="NS_straight", phase_type="yellow",
            elapsed=YELLOW_DURATION - 0.25, dt=0.5, requested_action=0,
        )
        self.assertEqual(result["phase_type"], "all_red")
        self.assertAlmostEqual(result["elapsed"], 0.0)

    def test_all_red_holds_until_duration(self):
        result = next_signal_rl(
            current_phase="NS_straight", phase_type="all_red",
            elapsed=0.5, dt=0.5, requested_action=0,
        )
        self.assertEqual(result["phase_type"], "all_red")

    def test_all_red_to_next_green(self):
        result = next_signal_rl(
            current_phase="NS_straight", phase_type="all_red",
            elapsed=ALL_RED_DURATION - 0.25, dt=0.5, requested_action=0,
        )
        self.assertEqual(result["phase"], "NS_left")   # next in sequence
        self.assertEqual(result["phase_type"], "green")
        self.assertAlmostEqual(result["elapsed"], 0.0)

    def test_all_red_wraps_around_sequence(self):
        # Last phase in sequence → should wrap to first
        last_phase = SIGNAL_SEQUENCE[-1]["phase"]
        first_phase = SIGNAL_SEQUENCE[0]["phase"]
        result = next_signal_rl(
            current_phase=last_phase, phase_type="all_red",
            elapsed=ALL_RED_DURATION, dt=0.0, requested_action=0,
        )
        self.assertEqual(result["phase"], first_phase)
        self.assertEqual(result["phase_type"], "green")


# ---------------------------------------------------------------------------
# next_signal (fixed / adaptive / wave)
# ---------------------------------------------------------------------------

class SignalLogicTests(unittest.TestCase):

    def test_fixed_transitions_to_yellow_after_base_duration(self):
        result = next_signal(
            current_phase="NS_straight", phase_type="green",
            elapsed=7.1, strategy="fixed",
            queues_ns=0, queues_ew=0,
            approaching_ns=0, approaching_ew=0,
            dt=0.25,
        )
        self.assertEqual(result["phase_type"], "yellow")

    def test_fixed_holds_before_duration(self):
        result = next_signal(
            current_phase="NS_straight", phase_type="green",
            elapsed=3.0, strategy="fixed",
            queues_ns=0, queues_ew=0,
            approaching_ns=0, approaching_ew=0,
            dt=0.25,
        )
        self.assertEqual(result["phase_type"], "green")

    def test_adaptive_extends_duration_with_queue(self):
        # High NS queue should add bonus time
        result_high = next_signal(
            current_phase="NS_straight", phase_type="green",
            elapsed=7.5, strategy="adaptive",
            queues_ns=10, queues_ew=0,
            approaching_ns=0, approaching_ew=0,
            dt=0.0,
        )
        result_zero = next_signal(
            current_phase="NS_straight", phase_type="green",
            elapsed=7.5, strategy="fixed",
            queues_ns=10, queues_ew=0,
            approaching_ns=0, approaching_ew=0,
            dt=0.0,
        )
        # Adaptive should still be green (bonus time); fixed should be yellow
        self.assertEqual(result_high["phase_type"], "green")
        self.assertEqual(result_zero["phase_type"], "yellow")

    def test_yellow_to_all_red(self):
        result = next_signal(
            current_phase="NS_straight", phase_type="yellow",
            elapsed=YELLOW_DURATION - 0.1, strategy="fixed",
            queues_ns=0, queues_ew=0,
            approaching_ns=0, approaching_ew=0,
            dt=0.25,
        )
        self.assertEqual(result["phase_type"], "all_red")

    def test_all_red_to_next_green(self):
        result = next_signal(
            current_phase="NS_straight", phase_type="all_red",
            elapsed=ALL_RED_DURATION - 0.1, strategy="fixed",
            queues_ns=0, queues_ew=0,
            approaching_ns=0, approaching_ew=0,
            dt=0.25,
        )
        self.assertEqual(result["phase"], "NS_left")
        self.assertEqual(result["phase_type"], "green")


# ---------------------------------------------------------------------------
# get_light_states
# ---------------------------------------------------------------------------

class LightStateTests(unittest.TestCase):

    def test_ns_straight_green(self):
        lights = get_light_states("NS_straight", "green")
        self.assertTrue(lights["R1"]["green"])
        self.assertTrue(lights["R1"]["arrow_rs"])
        self.assertFalse(lights["R1"]["arrow_left"])
        self.assertFalse(lights["R3"]["green"])
        self.assertTrue(lights["R3"]["red"])

    def test_ns_left_green_lights_arrow(self):
        lights = get_light_states("NS_left", "green")
        self.assertTrue(lights["R1"]["arrow_left"])
        self.assertFalse(lights["R1"]["arrow_rs"])

    def test_all_red_all_heads_red(self):
        for phase in [s["phase"] for s in SIGNAL_SEQUENCE]:
            lights = get_light_states(phase, "all_red")
            for head in lights.values():
                self.assertTrue(head["red"])
                self.assertFalse(head["green"])
                self.assertFalse(head["yellow"])

    def test_yellow_active_axis_shows_yellow(self):
        lights = get_light_states("NS_straight", "yellow")
        self.assertTrue(lights["R1"]["yellow"])
        self.assertFalse(lights["R1"]["green"])
        self.assertTrue(lights["R3"]["red"])

    def test_r1_r2_always_identical(self):
        for phase in [s["phase"] for s in SIGNAL_SEQUENCE]:
            for pt in ("green", "yellow", "all_red"):
                lights = get_light_states(phase, pt)
                self.assertEqual(lights["R1"], lights["R2"])

    def test_r3_r4_always_identical(self):
        for phase in [s["phase"] for s in SIGNAL_SEQUENCE]:
            for pt in ("green", "yellow", "all_red"):
                lights = get_light_states(phase, pt)
                self.assertEqual(lights["R3"], lights["R4"])


# ---------------------------------------------------------------------------
# can_vehicle_proceed
# ---------------------------------------------------------------------------

class VehicleTests(unittest.TestCase):

    def test_ns_straight_allows_straight_and_right(self):
        self.assertTrue(can_vehicle_proceed("NS_straight", "green", "NS", "straight"))
        self.assertTrue(can_vehicle_proceed("NS_straight", "green", "NS", "right"))

    def test_ns_straight_blocks_left(self):
        self.assertFalse(can_vehicle_proceed("NS_straight", "green", "NS", "left"))

    def test_ns_left_allows_only_left(self):
        self.assertTrue(can_vehicle_proceed("NS_left", "green", "NS", "left"))
        self.assertFalse(can_vehicle_proceed("NS_left", "green", "NS", "straight"))

    def test_wrong_axis_blocked(self):
        self.assertFalse(can_vehicle_proceed("NS_straight", "green", "EW", "straight"))

    def test_yellow_always_blocked(self):
        self.assertFalse(can_vehicle_proceed("NS_straight", "yellow", "NS", "straight"))

    def test_all_red_always_blocked(self):
        self.assertFalse(can_vehicle_proceed("NS_straight", "all_red", "NS", "straight"))


# ---------------------------------------------------------------------------
# rl_action_from_weights — normalization effect
# ---------------------------------------------------------------------------

class RLWeightsTests(unittest.TestCase):

    def _trivial_weights(self):
        """One-layer 6→2 model that strongly prefers action 0 (hold)."""
        return {
            "layers": [
                {
                    "W": [
                        1, 0,
                        1, 0,
                        1, 0,
                        1, 0,
                        1, 0,
                        1, 0,
                    ],
                    "b": [0, -10],
                }
            ]
        }

    def test_action_from_trivial_weights(self):
        action = rl_action_from_weights(self._trivial_weights(), [1, 1, 1, 1, 1, 1])
        self.assertEqual(action, 0)

    def test_large_queue_values_still_produce_valid_action(self):
        # Without normalization, 999 queue values would saturate ReLU
        action = rl_action_from_weights(self._trivial_weights(), [999, 999, 999, 999, 3, 60])
        self.assertIn(action, (0, 1))

    def test_zero_state_produces_valid_action(self):
        action = rl_action_from_weights(self._trivial_weights(), [0, 0, 0, 0, 0, 0])
        self.assertIn(action, (0, 1))


# ---------------------------------------------------------------------------
# Phase sequence helpers
# ---------------------------------------------------------------------------

class PhaseSequenceTests(unittest.TestCase):

    def test_next_phase_sequence(self):
        phases = [s["phase"] for s in SIGNAL_SEQUENCE]
        for i, phase in enumerate(phases):
            expected = phases[(i + 1) % len(phases)]
            self.assertEqual(_next_phase(phase), expected)

    def test_next_phase_unknown_defaults_to_second(self):
        # Unknown phase maps to index 0 → next is index 1
        result = _next_phase("UNKNOWN_phase")
        self.assertEqual(result, SIGNAL_SEQUENCE[1]["phase"])


if __name__ == "__main__":
    unittest.main(verbosity=2)