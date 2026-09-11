"""tests/test_smoothness.py — Tests for linreg_r2_slope and signed_smoothness."""

import pytest

from technical_analysis.indicators import linreg_r2_slope, signed_smoothness


class TestLinregR2Slope:
    def test_perfect_uptrend(self):
        r2, slope = linreg_r2_slope([1.0, 2.0, 3.0, 4.0, 5.0])
        assert r2 == pytest.approx(1.0)
        assert slope == pytest.approx(1.0)

    def test_perfect_downtrend(self):
        r2, slope = linreg_r2_slope([5.0, 4.0, 3.0, 2.0, 1.0])
        assert r2 == pytest.approx(1.0)
        assert slope == pytest.approx(-1.0)

    def test_flat(self):
        # All-identical → ss_tot = 0 → returns (0, 0) by contract
        r2, slope = linreg_r2_slope([3.0, 3.0, 3.0, 3.0])
        assert r2 == 0.0
        assert slope == 0.0

    def test_pure_noise_low_r2(self):
        # Symmetric oscillation → mean-slope ≈ 0, R² ≈ 0
        r2, slope = linreg_r2_slope([1.0, 5.0, 1.0, 5.0, 1.0, 5.0])
        assert r2 < 0.15
        assert abs(slope) < 0.5

    def test_noisy_uptrend(self):
        # Up trend with jitter — high but not perfect R², positive slope
        r2, slope = linreg_r2_slope([1.0, 1.8, 3.2, 3.9, 5.1, 5.9])
        assert 0.95 <= r2 <= 1.0
        assert slope > 0

    def test_too_few_points(self):
        assert linreg_r2_slope([]) == (0.0, 0.0)
        assert linreg_r2_slope([1.0]) == (0.0, 0.0)

    def test_two_points_perfect(self):
        # Two points always fit a line perfectly
        r2, slope = linreg_r2_slope([2.0, 4.0])
        assert r2 == pytest.approx(1.0)
        assert slope == pytest.approx(2.0)


class TestSignedSmoothness:
    def test_clean_up(self):
        s = signed_smoothness([1.0, 2.0, 3.0, 4.0, 5.0])
        assert s == pytest.approx(1.0)

    def test_clean_down(self):
        s = signed_smoothness([5.0, 4.0, 3.0, 2.0, 1.0])
        assert s == pytest.approx(-1.0)

    def test_chop(self):
        s = signed_smoothness([1.0, 5.0, 1.0, 5.0, 1.0, 5.0])
        assert abs(s) < 0.1

    def test_flat(self):
        assert signed_smoothness([3.0, 3.0, 3.0]) == 0.0

    def test_clean_up_with_pullback_still_positive(self):
        # Overall up but with a midway pullback — sign positive, R² high but < 1
        s = signed_smoothness([1.0, 2.0, 1.5, 2.5, 3.0, 4.0])
        assert s > 0.6
