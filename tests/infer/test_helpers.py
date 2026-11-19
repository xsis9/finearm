"""Unit tests for the shared executor math (clamp / align / blend)."""

from __future__ import annotations

import numpy as np
import pytest

from finearm.infer.executors import align_index, blend_transition, clamp_action


class TestClampAction:
    def test_no_limits_returns_copy(self) -> None:
        target = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
        out = clamp_action(None, target, None, None)
        np.testing.assert_array_equal(out, target)
        out[0] = 99.0
        assert target[0] == 1.0  # copy, not a view

    def test_pos_limits_clip(self) -> None:
        lo = np.zeros(7)
        hi = np.full(7, 1.0)
        out = clamp_action(None, np.full(7, 2.0), (lo, hi), None)
        np.testing.assert_array_equal(out, np.ones(7))

    def test_vel_limits_relative_to_prev(self) -> None:
        prev = np.zeros(7)
        vel = np.full(7, 0.1)
        out = clamp_action(prev, np.full(7, 5.0), None, vel)
        np.testing.assert_allclose(out, np.full(7, 0.1))
        out2 = clamp_action(prev, np.full(7, -5.0), None, vel)
        np.testing.assert_allclose(out2, np.full(7, -0.1))

    def test_pos_and_vel_combined(self) -> None:
        prev = np.array([0.0] * 7)
        lo = np.full(7, -0.05)
        hi = np.full(7, 0.05)
        # vel limit would allow 0.1, pos limit wins
        out = clamp_action(prev, np.full(7, 0.2), (lo, hi), np.full(7, 0.1))
        np.testing.assert_allclose(out, np.full(7, 0.05))


class TestAlignIndex:
    def test_finds_closest_on_first_six_joints(self) -> None:
        chunk = np.zeros((5, 7))
        chunk[:, 0] = np.arange(5)             # rows differ on joint 0 only
        chunk[:, 6] = 9.0                      # gripper differs wildly — must not matter
        ref = np.array([3.2, 0, 0, 0, 0, 0, 0])
        assert align_index(chunk, ref) == 3

    def test_gripper_excluded_from_distance(self) -> None:
        chunk = np.zeros((3, 7))
        chunk[2, :6] = 0.0
        ref = np.array([0, 0, 0, 0, 0, 0, 5.0])
        assert align_index(chunk, ref) in (0, 1, 2)  # gripper value cannot bias it


class TestBlendTransition:
    def test_blend_zero_is_hard_switch(self) -> None:
        old = np.zeros((10, 7))
        new = np.ones((10, 7))
        pending, start = blend_transition(old, 2, new, 4, 0)
        assert pending == []
        assert start == 4

    def test_blend_interpolates_and_resumes_past_window(self) -> None:
        old = np.zeros((10, 7))
        new = np.ones((10, 7))
        pending, start = blend_transition(old, 2, new, 4, 2)
        assert len(pending) == 2
        alphas = [(s + 1) / 3 for s in range(2)]
        np.testing.assert_allclose(pending[0], np.full(7, alphas[0]))
        np.testing.assert_allclose(pending[1], np.full(7, alphas[1]))
        assert start == 6  # j + blend

    def test_blend_clamped_at_chunk_end(self) -> None:
        old = np.zeros((4, 7))
        new = np.ones((6, 7))
        pending, start = blend_transition(old, 3, new, 5, 2)
        assert start == 5  # min(j+blend, len-1)
