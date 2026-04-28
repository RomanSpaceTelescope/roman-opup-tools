#!/usr/bin/env python
"""
test_roman_attitude.py
======================
Comprehensive unit-test suite for roman_attitude.py

Run with:
    pytest test_roman_attitude.py -v
    pytest test_roman_attitude.py -v -k "not slow"    # skip JPL network tests

Markers:
    @pytest.mark.slow   — tests that hit JPL Horizons (network required)
"""

import warnings
import numpy as np
import pytest
from unittest.mock import patch, MagicMock

from astropy.time import Time
from astropy.coordinates import SkyCoord
from astropy import units as u
from scipy.spatial.transform import Rotation as R

import roman_opup_tools.roman_attitude as ra
from roman_opup_tools.roman_attitude import (
    RomanPointing,
    roman_attitude,
    quat_to_radec_pa,
    get_radec,
    get_vector,
    ecliptic_to_equatorial,
    generate_power_law_sampling,
)

# ═════════════════════════════════════════════════════════════════════════════
# FIXTURES
# ═════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def obs_date():
    """Standard observation date used by most tests."""
    return Time('2026-11-21T00:00:00.0', format='isot', scale='utc')


@pytest.fixture
def pointing(obs_date):
    """
    RomanPointing with OEM ephemeris.
    If the OEM file is absent the fixture still works (JPL fallback).
    """
    return RomanPointing(obs_date)


@pytest.fixture
def pointed(pointing):
    """RomanPointing with a target already set (RA=269°, Dec=66°)."""
    pointing.set_target_using_radec(ra=269.0, dec=66.0)
    return pointing


# ═════════════════════════════════════════════════════════════════════════════
# 1. INITIALIZATION & EPHEMERIS FALLBACK
# ═════════════════════════════════════════════════════════════════════════════

class TestInitialization:
    """Tests for RomanPointing.__init__ and ephemeris fallback."""

    def test_default_date_is_now(self):
        """If no date given, observation_date should be close to Time.now()."""
        p = RomanPointing()
        assert abs((p.observation_date - Time.now()).sec) < 5.0

    def test_explicit_date(self, obs_date):
        """Observation date matches what was passed in."""
        p = RomanPointing(obs_date)
        assert p.observation_date == obs_date

    def test_identity_attitude_before_target(self, pointing):
        """Attitude matrix is identity before any target is set."""
        np.testing.assert_array_equal(pointing.spacecraft_attitude, np.eye(3))

    def test_no_target_on_init(self, pointing):
        """target_coord should be None before set_target is called."""
        assert pointing.target_coord is None

    def test_sun_coord_populated(self, pointing):
        """Sun position should always be populated after init."""
        assert pointing.sun_coord is not None
        assert isinstance(pointing.sun_coord, SkyCoord)

    def test_sun_ra_dec_in_range(self, pointing):
        """Sun RA in [0, 360), Dec in [-90, 90]."""
        assert 0.0 <= pointing.sun_coord.ra.deg < 360.0
        assert -90.0 <= pointing.sun_coord.dec.deg <= 90.0

    def test_pitch_limits_default(self, pointing):
        """Default pitch limits are ±36°."""
        np.testing.assert_array_equal(pointing.pitch_limits.value, [-36, 36])

    def test_sun_source_attribute(self, pointing):
        """_sun_source should be either 'OEM' or 'JPL'."""
        assert pointing._sun_source in ('OEM', 'JPL')

    def test_oem_fallback_to_jpl(self, obs_date):
        """When OEM file is missing, constructor should warn and use JPL."""
        with patch.object(ra, 'ephem', '/nonexistent/path/fake.oem'):
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                p = RomanPointing(obs_date)
                # Should have emitted a warning about the missing file
                fallback_warnings = [x for x in w if 'Falling back' in str(x.message)
                                     or 'not found' in str(x.message)]
                assert len(fallback_warnings) >= 1
                assert p.ephem is None
                assert p._sun_source == 'JPL'
                # Sun should still be populated
                assert p.sun_coord is not None

    def test_oem_success_sets_source(self, pointing):
        """When OEM loads, _sun_source should be 'OEM' (unless fallback)."""
        if pointing.ephem is not None:
            assert pointing._sun_source == 'OEM'
        else:
            assert pointing._sun_source == 'JPL'


# ═════════════════════════════════════════════════════════════════════════════
# 2. TARGET SETTING
# ═════════════════════════════════════════════════════════════════════════════

class TestTargetSetting:
    """Tests for set_target_using_radec and set_target."""

    def test_set_target_using_radec(self, pointing):
        """set_target_using_radec stores the correct coordinates."""
        pointing.set_target_using_radec(ra=120.0, dec=45.0)
        np.testing.assert_allclose(pointing.target_coord.ra.deg, 120.0, atol=1e-10)
        np.testing.assert_allclose(pointing.target_coord.dec.deg, 45.0, atol=1e-10)

    def test_set_target_using_skycoord(self, pointing, obs_date):
        """set_target with SkyCoord stores the correct coordinates."""
        tgt = SkyCoord(ra=200.0 * u.deg, dec=-30.0 * u.deg,
                       frame='icrs', obstime=obs_date)
        pointing.set_target(tgt)
        np.testing.assert_allclose(pointing.target_coord.ra.deg, 200.0, atol=1e-10)
        np.testing.assert_allclose(pointing.target_coord.dec.deg, -30.0, atol=1e-10)

    def test_set_target_updates_attitude(self, pointing):
        """Attitude should no longer be identity after setting a target."""
        pointing.set_target_using_radec(ra=269.0, dec=66.0)
        assert not np.allclose(pointing.spacecraft_attitude, np.eye(3))

    def test_attitude_is_rotation_matrix(self, pointed):
        """Attitude must be a proper rotation: R^T R = I, det(R) = +1."""
        att = pointed.spacecraft_attitude
        np.testing.assert_allclose(att.T @ att, np.eye(3), atol=1e-12)
        np.testing.assert_allclose(np.linalg.det(att), 1.0, atol=1e-12)

    def test_x_axis_points_at_target(self, pointed):
        """First column of attitude matrix should point at the target."""
        tgt_vec = get_vector(pointed.target_coord)
        x_axis = pointed.spacecraft_attitude[:, 0]
        np.testing.assert_allclose(x_axis, tgt_vec, atol=1e-10)

    def test_both_set_methods_agree(self, obs_date):
        """set_target_using_radec and set_target should give same attitude."""
        p1 = RomanPointing(obs_date)
        p1.set_target_using_radec(ra=269.0, dec=66.0)

        p2 = RomanPointing(obs_date)
        tgt = SkyCoord(ra=269.0 * u.deg, dec=66.0 * u.deg,
                       frame='icrs', obstime=obs_date)
        p2.set_target(tgt)

        np.testing.assert_allclose(p1.spacecraft_attitude,
                                   p2.spacecraft_attitude, atol=1e-12)

    def test_set_target_with_hour_angle_unit(self, pointing):
        """RA specified in hour-angle units."""
        pointing.set_target_using_radec(ra=12.0, dec=45.0, unit='hour')
        np.testing.assert_allclose(pointing.target_coord.ra.deg, 180.0, atol=1e-10)

    def test_set_target_with_roll(self, pointing):
        """Setting a non-zero roll should change the attitude."""
        pointing.set_target_using_radec(ra=269.0, dec=66.0, roll=0.0)
        att0 = pointing.spacecraft_attitude.copy()

        pointing.set_target_using_radec(ra=269.0, dec=66.0, roll=5.0)
        att5 = pointing.spacecraft_attitude.copy()

        # X-axis (pointing) should be the same, but Y/Z should differ
        np.testing.assert_allclose(att0[:, 0], att5[:, 0], atol=1e-10)
        assert not np.allclose(att0[:, 1], att5[:, 1])


# ═════════════════════════════════════════════════════════════════════════════
# 3. SUN ANGLE & PITCH
# ═════════════════════════════════════════════════════════════════════════════

class TestSunAngleAndPitch:
    """Tests for get_sun_angle and get_pitch_angle."""

    def test_sun_angle_is_positive(self, pointed):
        """Sun angle should always be a positive number."""
        angle = pointed.get_sun_angle()
        assert angle > 0.0

    def test_sun_angle_in_physical_range(self, pointed):
        """Sun angle must be in [0, 180]."""
        angle = pointed.get_sun_angle()
        assert 0.0 <= angle <= 180.0

    def test_pitch_definition(self, pointed):
        """Pitch = sun_angle - 90°."""
        sun_angle = pointed.get_sun_angle()
        pitch = pointed.get_pitch_angle()
        np.testing.assert_allclose(pitch.value, sun_angle - 90.0, atol=1e-10)

    def test_sun_angle_with_explicit_target(self, pointed, obs_date):
        """get_sun_angle(target=...) should work with an external SkyCoord."""
        tgt = SkyCoord(ra=0.0 * u.deg, dec=0.0 * u.deg,
                       frame='icrs', obstime=obs_date)
        angle = pointed.get_sun_angle(target=tgt)
        assert 0.0 <= angle <= 180.0

    def test_sun_angle_none_without_target(self, pointing):
        """get_sun_angle returns None when no target is set."""
        assert pointing.get_sun_angle() is None

    def test_pitch_stored_on_object(self, pointed):
        """get_pitch_angle() with no argument stores result in self.pitch."""
        _ = pointed.get_pitch_angle()
        assert hasattr(pointed, 'pitch')
        assert pointed.pitch is not None

    def test_sun_angle_to_sun_is_zero(self, pointing):
        """If we point directly at the Sun, sun_angle ≈ 0."""
        sun_ra = pointing.sun_coord.ra.deg
        sun_dec = pointing.sun_coord.dec.deg
        pointing.set_target_using_radec(ra=sun_ra, dec=sun_dec)
        np.testing.assert_allclose(pointing.get_sun_angle(), 0.0, atol=1e-6)


# ═════════════════════════════════════════════════════════════════════════════
# 4. ATTITUDE REPRESENTATIONS
# ═════════════════════════════════════════════════════════════════════════════

class TestAttitudeRepresentations:
    """Tests for quaternion, Euler, and position angle outputs."""

    def test_quaternion_is_unit(self, pointed):
        """Quaternion must have unit norm."""
        q = pointed.get_attitude_quaternion()
        np.testing.assert_allclose(np.linalg.norm(q), 1.0, atol=1e-12)

    def test_quaternion_length(self, pointed):
        """Quaternion should have 4 components."""
        q = pointed.get_attitude_quaternion()
        assert q.shape == (4,)

    def test_quaternion_scalar_last_convention(self, pointed):
        """get_attitude_quaternion returns [x,y,z,w] matching roman_attitude()."""
        q = pointed.get_attitude_quaternion()
        ra_q, dec_q, _ = roman_attitude(q)
        np.testing.assert_allclose(ra_q, pointed.target_coord.ra.deg, atol=1e-6)
        np.testing.assert_allclose(dec_q, pointed.target_coord.dec.deg, atol=1e-6)

    def test_quaternion_round_trip_roman_attitude(self, pointed):
        """attitude → quat → roman_attitude → RA/Dec should match target."""
        q = pointed.get_attitude_quaternion()
        ra_q, dec_q, pa_q = roman_attitude(q)
        np.testing.assert_allclose(ra_q, pointed.target_coord.ra.deg, atol=1e-8)
        np.testing.assert_allclose(dec_q, pointed.target_coord.dec.deg, atol=1e-8)

    def test_quaternion_round_trip_quat_to_radec_pa(self, pointed):
        """attitude → quat → quat_to_radec_pa → RA/Dec should match target."""
        q = pointed.get_attitude_quaternion()
        ra_q, dec_q, pa_q = quat_to_radec_pa(*q)
        np.testing.assert_allclose(ra_q, pointed.target_coord.ra.deg, atol=1e-8)
        np.testing.assert_allclose(dec_q, pointed.target_coord.dec.deg, atol=1e-8)

    def test_roman_attitude_and_quat_to_radec_pa_agree(self, pointed):
        """Both quaternion converters must give identical results."""
        q = pointed.get_attitude_quaternion()
        result1 = roman_attitude(q)
        result2 = quat_to_radec_pa(*q)
        np.testing.assert_allclose(result1, result2, atol=1e-10)

    def test_quaternion_pa_matches_position_angle(self, pointed):
        """PA from quaternion should match get_position_angle()."""
        q = pointed.get_attitude_quaternion()
        _, _, pa_q = roman_attitude(q)
        pa_direct = pointed.get_position_angle().value % 360
        np.testing.assert_allclose(pa_q, pa_direct, atol=1e-6)

    def test_euler_returns_3_angles(self, pointed):
        """get_attitude_euler should return 3 Euler angles."""
        euler = pointed.get_attitude_euler()
        assert euler.shape == (3,)

    def test_euler_degrees_vs_radians(self, pointed):
        """Euler angles in degrees vs. radians should be consistent."""
        euler_deg = pointed.get_attitude_euler(degrees=True)
        euler_rad = pointed.get_attitude_euler(degrees=False)
        np.testing.assert_allclose(np.degrees(euler_rad), euler_deg, atol=1e-10)

    def test_euler_reconstructs_attitude(self, pointed):
        """Euler → rotation matrix should recover the original attitude."""
        seq = 'ZYX'
        euler = pointed.get_attitude_euler(sequence=seq, degrees=True)
        reconstructed = R.from_euler(seq.upper(), euler, degrees=True).as_matrix()
        np.testing.assert_allclose(reconstructed, pointed.spacecraft_attitude,
                                   atol=1e-10)

    def test_position_angle_in_range(self, pointed):
        """Position angle should be in [0, 360)."""
        pa = pointed.get_position_angle().value % 360
        assert 0.0 <= pa < 360.0

    def test_position_angle_with_explicit_attitude(self, pointed):
        """get_position_angle(attitude=...) should accept an external matrix."""
        att = pointed.spacecraft_attitude
        pa1 = pointed.get_position_angle()
        pa2 = pointed.get_position_angle(att)
        np.testing.assert_allclose(pa1.value, pa2.value, atol=1e-12)


# ═════════════════════════════════════════════════════════════════════════════
# 5. COORDINATE CONVERSIONS
# ═════════════════════════════════════════════════════════════════════════════

class TestCoordinateConversions:
    """Tests for ecliptic conversions and get_radec / get_vector."""

    def test_ecliptic_lon_in_range(self, pointed):
        """Ecliptic longitude in [0, 360)."""
        lon, lat = pointed.get_ecliptic_coordinates(pointed.target_coord)
        assert 0.0 <= lon < 360.0

    def test_ecliptic_lat_in_range(self, pointed):
        """Ecliptic latitude in [-90, 90]."""
        lon, lat = pointed.get_ecliptic_coordinates(pointed.target_coord)
        assert -90.0 <= lat <= 90.0

    def test_get_radec_from_identity(self):
        """Identity matrix → RA=0, Dec=0 (pointing along +x)."""
        ra_val, dec_val = get_radec(np.eye(3))
        np.testing.assert_allclose(ra_val, 0.0, atol=1e-10)
        np.testing.assert_allclose(dec_val, 0.0, atol=1e-10)

    def test_get_radec_north_pole(self):
        """Attitude with x-axis along +z → Dec=90."""
        att = np.array([[0, 0, -1],
                        [0, 1, 0],
                        [1, 0, 0]], dtype=float)
        _, dec_val = get_radec(att)
        np.testing.assert_allclose(dec_val, 90.0, atol=1e-10)

    def test_get_vector_unit_length(self, pointed):
        """get_vector should return a unit vector."""
        vec = get_vector(pointed.target_coord)
        np.testing.assert_allclose(np.linalg.norm(vec), 1.0, atol=1e-12)

    def test_get_radec_roundtrip_via_attitude(self, pointed):
        """RA/Dec extracted from attitude matrix should match the target."""
        ra_val, dec_val = get_radec(pointed.spacecraft_attitude)
        np.testing.assert_allclose(ra_val, pointed.target_coord.ra.deg, atol=1e-8)
        np.testing.assert_allclose(dec_val, pointed.target_coord.dec.deg, atol=1e-8)

    def test_ecliptic_to_equatorial_ecliptic_north(self):
        """Ecliptic north pole → equatorial coords consistent with obliquity."""
        ecl_north = np.array([0.0, 0.0, 1.0])
        eq = ecliptic_to_equatorial(ecl_north)
        eps = np.radians(23.43929111)
        expected = np.array([0.0, -np.sin(eps), np.cos(eps)])
        np.testing.assert_allclose(eq, expected, atol=1e-12)


# ═════════════════════════════════════════════════════════════════════════════
# 6. PITCH & ROLL MANEUVERS
# ═════════════════════════════════════════════════════════════════════════════

class TestPitchRollManeuvers:
    """Tests for delta_pitch_roll and get_target_new_pitch."""

    def test_zero_offset_returns_original(self, pointed):
        """Zero pitch/roll offset → same RA/Dec as the original target."""
        ra_new, dec_new, pa_new, pitch_new = pointed.delta_pitch_roll(0.0, 0.0)
        np.testing.assert_allclose(ra_new, pointed.target_coord.ra.deg, atol=1e-6)
        np.testing.assert_allclose(dec_new, pointed.target_coord.dec.deg, atol=1e-6)

    def test_pitch_changes_sun_angle(self, pointed):
        """A pitch offset should change the effective sun angle."""
        _, _, _, pitch0 = pointed.delta_pitch_roll(0.0, 0.0)
        _, _, _, pitch5 = pointed.delta_pitch_roll(5.0, 0.0)
        assert abs(pitch5 - pitch0) > 1.0  # should differ by ~5°

    def test_pitch_offset_monotonic(self, pointed):
        """Increasing pitch offset should monotonically change pitch angle."""
        pitches = []
        for dp in np.linspace(-20, 20, 9):
            _, _, _, p = pointed.delta_pitch_roll(dp, 0.0)
            pitches.append(p)
        # Should be monotonically increasing (or decreasing)
        diffs = np.diff(pitches)
        assert np.all(diffs > 0) or np.all(diffs < 0)

    def test_roll_does_not_change_pitch(self, pointed):
        """Pure roll should not change the pitch angle significantly."""
        _, _, _, pitch0 = pointed.delta_pitch_roll(0.0, 0.0)
        _, _, _, pitchR = pointed.delta_pitch_roll(0.0, 10.0)
        np.testing.assert_allclose(pitch0, pitchR, atol=0.5)

    def test_roll_changes_pa(self, pointed):
        """A roll offset should change the position angle."""
        _, _, pa0, _ = pointed.delta_pitch_roll(0.0, 0.0)
        _, _, pa5, _ = pointed.delta_pitch_roll(0.0, 5.0)
        assert abs(pa5 - pa0) > 0.1

    def test_delta_pitch_roll_returns_four(self, pointed):
        """delta_pitch_roll should return (ra, dec, pa, pitch)."""
        result = pointed.delta_pitch_roll(5.0, 3.0)
        assert len(result) == 4

    def test_delta_pitch_roll_ra_in_range(self, pointed):
        """RA from delta_pitch_roll should be in [0, 360)."""
        ra_new, _, _, _ = pointed.delta_pitch_roll(10.0, 5.0)
        assert 0.0 <= ra_new < 360.0

    def test_delta_pitch_roll_dec_in_range(self, pointed):
        """Dec from delta_pitch_roll should be in [-90, 90]."""
        _, dec_new, _, _ = pointed.delta_pitch_roll(10.0, 5.0)
        assert -90.0 <= dec_new <= 90.0

    def test_get_target_new_pitch_returns_skycoord(self, pointed):
        """get_target_new_pitch should return (SkyCoord, 3x3 array)."""
        new_tgt, new_att = pointed.get_target_new_pitch(delta_pitch=5.0)
        assert isinstance(new_tgt, SkyCoord)
        assert new_att.shape == (3, 3)

    def test_get_target_new_pitch_attitude_is_rotation(self, pointed):
        """New attitude from pitch offset must be a proper rotation."""
        _, new_att = pointed.get_target_new_pitch(delta_pitch=10.0)
        np.testing.assert_allclose(new_att.T @ new_att, np.eye(3), atol=1e-12)
        np.testing.assert_allclose(np.linalg.det(new_att), 1.0, atol=1e-12)

    def test_get_target_new_pitch_zero_is_identity_operation(self, pointed):
        """Zero delta_pitch should return the same target."""
        new_tgt, _ = pointed.get_target_new_pitch(delta_pitch=0.0)
        np.testing.assert_allclose(new_tgt.ra.deg,
                                   pointed.target_coord.ra.deg, atol=1e-8)
        np.testing.assert_allclose(new_tgt.dec.deg,
                                   pointed.target_coord.dec.deg, atol=1e-8)

    def test_symmetric_pitch_offsets(self, pointed):
        """Equal and opposite pitch offsets should bracket the original."""
        _, _, _, pitch_pos = pointed.delta_pitch_roll(10.0, 0.0)
        _, _, _, pitch_neg = pointed.delta_pitch_roll(-10.0, 0.0)
        _, _, _, pitch_ctr = pointed.delta_pitch_roll(0.0, 0.0)
        # Center should be between the two offsets
        assert min(pitch_neg, pitch_pos) <= pitch_ctr <= max(pitch_neg, pitch_pos)


# ═════════════════════════════════════════════════════════════════════════════
# 7. VISIBILITY CHECK
# ═════════════════════════════════════════════════════════════════════════════

class TestVisibility:
    """Tests for is_visible constraint checking."""

    def test_is_visible_returns_bool(self, pointed):
        """is_visible should return a boolean."""
        result = pointed.is_visible()
        assert isinstance(result, (bool, np.bool_))

    def test_target_at_sun_is_not_visible(self, pointing):
        """Pointing directly at the Sun → pitch = -90° → not visible."""
        sun_ra = pointing.sun_coord.ra.deg
        sun_dec = pointing.sun_coord.dec.deg
        pointing.set_target_using_radec(ra=sun_ra, dec=sun_dec)
        assert not pointing.is_visible()

    def test_target_opposite_sun_not_visible(self, pointing):
        """Anti-Sun direction → pitch = +90° → not visible."""
        anti_ra = (pointing.sun_coord.ra.deg + 180.0) % 360.0
        anti_dec = -pointing.sun_coord.dec.deg
        pointing.set_target_using_radec(ra=anti_ra, dec=anti_dec)
        assert not pointing.is_visible()


# ═════════════════════════════════════════════════════════════════════════════
# 8. STANDALONE QUATERNION FUNCTIONS
# ═════════════════════════════════════════════════════════════════════════════

class TestQuaternionFunctions:
    """Tests for the free-standing roman_attitude and quat_to_radec_pa."""

    def test_identity_quaternion(self):
        """Identity quaternion [0,0,0,1] → RA=0, Dec=0, PA=0."""
        ra_q, dec_q, pa_q = roman_attitude([0, 0, 0, 1])
        np.testing.assert_allclose(ra_q, 0.0, atol=1e-10)
        np.testing.assert_allclose(dec_q, 0.0, atol=1e-10)

    def test_normalised_quaternion_required(self):
        """Both functions should give consistent results for normalised q."""
        q_raw = np.array([0.1, 0.2, 0.3, 0.9])
        q = q_raw / np.linalg.norm(q_raw)
        r1 = roman_attitude(q)
        r2 = quat_to_radec_pa(*q)
        np.testing.assert_allclose(r1, r2, atol=1e-10)

    def test_ra_range(self):
        """RA output should be in [0, 360)."""
        for _ in range(20):
            q = np.random.randn(4)
            q /= np.linalg.norm(q)
            ra_q, _, _ = roman_attitude(q)
            assert 0.0 <= ra_q < 360.0

    def test_dec_range(self):
        """Dec output should be in [-90, 90]."""
        for _ in range(20):
            q = np.random.randn(4)
            q /= np.linalg.norm(q)
            _, dec_q, _ = roman_attitude(q)
            assert -90.0 <= dec_q <= 90.0

    def test_pa_range(self):
        """PA output should be in [0, 360)."""
        for _ in range(20):
            q = np.random.randn(4)
            q /= np.linalg.norm(q)
            _, _, pa_q = roman_attitude(q)
            assert 0.0 <= pa_q < 360.0

    def test_opposite_quaternions_same_attitude(self):
        """q and -q represent the same rotation → same RA/Dec/PA."""
        q = np.array([0.1, 0.2, 0.3, 0.9])
        q /= np.linalg.norm(q)
        r1 = roman_attitude(q)
        r2 = roman_attitude(-q)
        np.testing.assert_allclose(r1, r2, atol=1e-10)


# ═════════════════════════════════════════════════════════════════════════════
# 9. POWER-LAW SAMPLING
# ═════════════════════════════════════════════════════════════════════════════

class TestPowerLawSampling:
    """Tests for generate_power_law_sampling."""

    def test_correct_count(self):
        """Should return exactly n_points values."""
        for n in [10, 13, 14, 15, 20]:
            pts = generate_power_law_sampling(n_points=n)
            assert len(pts) == n

    def test_within_range(self):
        """All points should be within [-range_max, +range_max]."""
        pts = generate_power_law_sampling(n_points=14, range_max=36.0)
        assert np.all(pts >= -36.0)
        assert np.all(pts <= 36.0)

    def test_sorted(self):
        """Points should be sorted ascending."""
        pts = generate_power_law_sampling(n_points=14)
        np.testing.assert_array_equal(pts, np.sort(pts))

    def test_symmetric(self):
        """Distribution should be symmetric about zero."""
        pts = generate_power_law_sampling(n_points=14, range_max=36.0)
        np.testing.assert_allclose(pts, -pts[::-1], atol=1e-12)

    def test_edge_emphasis(self):
        """Higher power should push more points toward the edges."""
        pts_low = generate_power_law_sampling(n_points=14, power=1)
        pts_high = generate_power_law_sampling(n_points=14, power=5)
        # Median absolute value should be higher for high power
        assert np.median(np.abs(pts_high)) > np.median(np.abs(pts_low))

    def test_odd_count_includes_zero(self):
        """Odd n_points should include zero."""
        pts = generate_power_law_sampling(n_points=15)
        assert 0.0 in pts

    def test_even_count_excludes_zero(self):
        """Even n_points should not include zero."""
        pts = generate_power_law_sampling(n_points=14)
        assert 0.0 not in pts


# ═════════════════════════════════════════════════════════════════════════════
# 10. PITCH RASTER SCAN (INTEGRATION)
# ═════════════════════════════════════════════════════════════════════════════

class TestPitchRaster:
    """Integration tests for building a full pitch-raster scan."""

    def test_raster_produces_valid_coords(self, pointed):
        """Every raster point should have valid RA/Dec."""
        initial_pitch = pointed.get_pitch_angle()
        pts = generate_power_law_sampling(n_points=10, range_max=35.5, power=4)
        for target_pitch in pts:
            dp = float(target_pitch) - initial_pitch.value
            ra, dec, pa, pitch = pointed.delta_pitch_roll(dp, 0.0)
            assert 0.0 <= ra < 360.0
            assert -90.0 <= dec <= 90.0
            assert 0.0 <= pa % 360 < 360.0

    def test_raster_covers_pitch_range(self, pointed):
        """Raster scan should span most of the ±35.5° pitch range."""
        initial_pitch = pointed.get_pitch_angle()
        pts = generate_power_law_sampling(n_points=14, range_max=35.5, power=4)
        pitches = []
        for target_pitch in pts:
            dp = float(target_pitch) - initial_pitch.value
            _, _, _, pitch = pointed.delta_pitch_roll(dp, 0.0)
            pitches.append(pitch)
        pitch_range = max(pitches) - min(pitches)
        assert pitch_range > 50.0  # should span a significant portion


# ═════════════════════════════════════════════════════════════════════════════
# 11. MULTIPLE TARGETS / DIFFERENT SKY POSITIONS
# ═════════════════════════════════════════════════════════════════════════════

class TestMultipleTargets:
    """Ensure the tool works across different regions of the sky."""

    TARGETS = [
        (0.0, 0.0),         # Vernal equinox
        (180.0, 0.0),       # Autumnal equinox
        (90.0, 45.0),       # Mid-latitude
        (270.0, -45.0),     # Southern sky
        (269.0, 66.0),      # The standard test target
    ]

    @pytest.mark.parametrize("ra,dec", TARGETS)
    def test_attitude_is_valid_rotation(self, pointing, ra, dec):
        """Attitude should be a valid rotation for various targets."""
        pointing.set_target_using_radec(ra=ra, dec=dec)
        att = pointing.spacecraft_attitude
        np.testing.assert_allclose(att.T @ att, np.eye(3), atol=1e-12)
        np.testing.assert_allclose(np.linalg.det(att), 1.0, atol=1e-12)

    @pytest.mark.parametrize("ra,dec", TARGETS)
    def test_quaternion_roundtrip_various_targets(self, pointing, ra, dec):
        """Quaternion round-trip should work for any sky position."""
        pointing.set_target_using_radec(ra=ra, dec=dec)
        q = pointing.get_attitude_quaternion()
        ra_q, dec_q, _ = roman_attitude(q)
        np.testing.assert_allclose(ra_q, ra, atol=1e-6)
        np.testing.assert_allclose(dec_q, dec, atol=1e-6)

    @pytest.mark.parametrize("ra,dec", TARGETS)
    def test_sun_angle_physical_for_various_targets(self, pointing, ra, dec):
        """Sun angle should be in [0, 180] for any target."""
        pointing.set_target_using_radec(ra=ra, dec=dec)
        angle = pointing.get_sun_angle()
        assert 0.0 <= angle <= 180.0


# ═════════════════════════════════════════════════════════════════════════════
# 12. PYSIAF CROSS-CHECK
# ═════════════════════════════════════════════════════════════════════════════

class TestPysiafCrossCheck:
    """Cross-check attitude matrix against pysiaf (skipped if not installed)."""

    @pytest.fixture(autouse=True)
    def _require_pysiaf(self):
        pytest.importorskip("pysiaf")

    def test_attitude_matches_pysiaf(self, pointed):
        """Attitude matrix should match pysiaf.utils.rotations.attitude."""
        from pysiaf.utils.rotations import attitude
        boresight_ra = pointed.target_coord.ra.deg
        boresight_dec = pointed.target_coord.dec.deg
        pa_v3 = pointed.get_position_angle().value

        att_pysiaf = attitude(0, 0, boresight_ra, boresight_dec, pa_v3)
        np.testing.assert_allclose(att_pysiaf, pointed.spacecraft_attitude,
                                   rtol=1e-5, atol=1e-8)

    def test_euler_path_matches_pysiaf(self, obs_date, pointed):
        """_calculate_pointing_attitude_euler should also match pysiaf."""
        from pysiaf.utils.rotations import attitude
        boresight_ra = pointed.target_coord.ra.deg
        boresight_dec = pointed.target_coord.dec.deg

        new_pa = 150.0
        att_pysiaf = attitude(0, 0, boresight_ra, boresight_dec, new_pa)

        p2 = RomanPointing(obs_date)
        p2.set_target(pointed.target_coord)
        att_euler = p2._calculate_pointing_attitude_euler(new_pa)

        np.testing.assert_allclose(att_pysiaf, att_euler, rtol=1e-5, atol=1e-8)


# ═════════════════════════════════════════════════════════════════════════════
# 13. EDGE CASES & ROBUSTNESS
# ═════════════════════════════════════════════════════════════════════════════

class TestEdgeCases:
    """Edge cases, poles, and numerical robustness."""

    def test_target_at_north_pole(self, pointing):
        """Setting target at Dec=+90° should not crash."""
        pointing.set_target_using_radec(ra=0.0, dec=89.9999)
        att = pointing.spacecraft_attitude
        np.testing.assert_allclose(att.T @ att, np.eye(3), atol=1e-10)

    def test_target_at_south_pole(self, pointing):
        """Setting target at Dec=-90° should not crash."""
        pointing.set_target_using_radec(ra=0.0, dec=-89.9999)
        att = pointing.spacecraft_attitude
        np.testing.assert_allclose(att.T @ att, np.eye(3), atol=1e-10)

    def test_ra_wrapping_at_360(self, pointing):
        """RA=359.99 should work and be close to RA=0.01."""
        pointing.set_target_using_radec(ra=359.99, dec=0.0)
        q1 = pointing.get_attitude_quaternion()
        pointing.set_target_using_radec(ra=0.01, dec=0.0)
        q2 = pointing.get_attitude_quaternion()
        # The angular difference should be small (~0.02°)
        ra1, _, _ = roman_attitude(q1)
        ra2, _, _ = roman_attitude(q2)
        angular_diff = min(abs(ra1 - ra2), 360.0 - abs(ra1 - ra2))
        assert angular_diff < 0.1

    def test_large_pitch_offset(self, pointed):
        """A large pitch offset should not crash (may exceed limits)."""
        ra, dec, pa, pitch = pointed.delta_pitch_roll(70.0, 0.0)
        assert 0.0 <= ra < 360.0
        assert -90.0 <= dec <= 90.0

    def test_large_roll_offset(self, pointed):
        """A large roll offset should not crash."""
        ra, dec, pa, pitch = pointed.delta_pitch_roll(0.0, 30.0)
        assert 0.0 <= ra < 360.0
        assert -90.0 <= dec <= 90.0

    def test_negative_quaternion_sign(self):
        """roman_attitude should handle q and -q identically."""
        q = np.array([0.5, 0.5, 0.5, 0.5])
        r1 = roman_attitude(q)
        r2 = roman_attitude(-q)
        np.testing.assert_allclose(r1, r2, atol=1e-10)

    def test_get_radec_from_vector(self):
        """get_radec should handle a 3-element vector (not just 3x3)."""
        vec = np.array([1.0, 0.0, 0.0])
        ra_val, dec_val = get_radec(vec)
        np.testing.assert_allclose(ra_val, 0.0, atol=1e-10)
        np.testing.assert_allclose(dec_val, 0.0, atol=1e-10)

    def test_sequential_target_changes(self, pointing):
        """Setting multiple targets in sequence should not leak state."""
        pointing.set_target_using_radec(ra=10.0, dec=20.0)
        att1 = pointing.spacecraft_attitude.copy()

        pointing.set_target_using_radec(ra=200.0, dec=-40.0)
        att2 = pointing.spacecraft_attitude.copy()

        pointing.set_target_using_radec(ra=10.0, dec=20.0)
        att3 = pointing.spacecraft_attitude.copy()

        # att1 and att3 should match; att2 should differ
        np.testing.assert_allclose(att1, att3, atol=1e-12)
        assert not np.allclose(att1, att2)


# ═════════════════════════════════════════════════════════════════════════════
# 14. VISUALIZATION (smoke tests — no display)
# ═════════════════════════════════════════════════════════════════════════════

class TestVisualization:
    """Smoke tests for plotting methods (no display, just no-crash)."""

    @pytest.fixture(autouse=True)
    def _use_agg_backend(self):
        """Use non-interactive backend for CI."""
        import matplotlib
        matplotlib.use('Agg')
        yield
        import matplotlib.pyplot as plt
        plt.close('all')

    def test_visualize_pointing_no_crash(self, pointed):
        """visualize_pointing should run without error."""
        import matplotlib.pyplot as plt
        pointed.visualize_pointing(
            show_ecliptic=True,
            show_sun=True,
            show_constellation_grid=False,
        )
        plt.close('all')

    def test_visualize_pointing_with_target_list(self, pointed):
        """visualize_pointing with target_list should not crash."""
        import matplotlib.pyplot as plt
        targets = []
        for dp in [-10, 0, 10]:
            tgt, _ = pointed.get_target_new_pitch(delta_pitch=dp)
            targets.append(tgt)
        pointed.visualize_pointing(target_list=targets)
        plt.close('all')