#!/usr/bin/env python
"""
roman_attitude_demo.py
======================
A concise demonstration of the RomanPointing toolkit capabilities.

Showcases:
  1. Basic initialization and target setting
  2. Querying spacecraft attitude (quaternion, Euler, matrix)
  3. Sun angle & pitch angle calculations
  4. Coordinate system conversions (ICRS → ecliptic, galactic)
  5. Pitch & roll maneuvers — computing new RA/Dec/PA
  6. Generating a pitch-raster scan across the field of regard
  7. Quaternion-to-attitude conversion from visit files
  8. Quaternion round-trip consistency check
  9. Full 4-panel visualization (3-D sphere, sky maps, spacecraft model)
 10. Pysiaf cross-check (attitude matrix comparison)

Requirements:
  - roman_attitude.py in the Python path
  - OEM ephemeris file at the path referenced inside roman_attitude.py
  - numpy, matplotlib, astropy, scipy, pysiaf, pandas
"""

import numpy as np
import pandas as pd
from astropy.time import Time
from astropy.coordinates import SkyCoord
from astropy import units as u

from roman_attitude import (
    RomanPointing,
    roman_attitude,
    quat_to_radec_pa,
    generate_power_law_sampling,
)

# ─────────────────────────────────────────────────────────────────────────────
# Helper
# ─────────────────────────────────────────────────────────────────────────────
def section(title):
    """Pretty-print a section header."""
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


# ═══════════════════════════════════════════════════════════════════════════════
# 1.  BASIC INITIALIZATION
# ═══════════════════════════════════════════════════════════════════════════════
section("1. Initialize RomanPointing for a given observation date")

obs_date = Time('2026-11-21T00:00:00.0', format='isot', scale='utc')
pointing = RomanPointing(obs_date)

print(f"Observation date : {obs_date.iso}")
print(f"Sun position     : RA = {pointing.sun_coord.ra.deg:.4f}°, "
      f"Dec = {pointing.sun_coord.dec.deg:.4f}°")


# ═══════════════════════════════════════════════════════════════════════════════
# 2.  SET A TARGET  (two equivalent ways)
# ═══════════════════════════════════════════════════════════════════════════════
section("2. Set a celestial target")

# Method A — using RA/Dec directly
pointing.set_target_using_radec(ra=269.0, dec=66.0, roll=0.0)

# Method B — using an astropy SkyCoord (equivalent)
# tgt = SkyCoord(ra=269.0*u.deg, dec=66.0*u.deg, frame='icrs', obstime=obs_date)
# pointing.set_target(tgt, roll=0.0)

print(f"Target           : RA = {pointing.target_coord.ra.deg:.4f}°, "
      f"Dec = {pointing.target_coord.dec.deg:.4f}°")


# ═══════════════════════════════════════════════════════════════════════════════
# 3.  QUERY SUN ANGLE & PITCH
# ═══════════════════════════════════════════════════════════════════════════════
section("3. Sun angle & pitch angle")

sun_angle = pointing.get_sun_angle()
pitch     = pointing.get_pitch_angle()

print(f"Sun angle        : {sun_angle:.4f}°")
print(f"Pitch angle      : {pitch:.4f}")
print(f"Pitch limits     : {pointing.pitch_limits}")


# ═══════════════════════════════════════════════════════════════════════════════
# 4.  ATTITUDE REPRESENTATIONS
# ═══════════════════════════════════════════════════════════════════════════════
section("4. Spacecraft attitude in multiple representations")

# 3x3 rotation matrix (body → ICRS)
print("Attitude matrix (3×3):")
print(np.array2string(pointing.spacecraft_attitude, precision=6, suppress_small=True))

# Quaternion [x, y, z, w]  (scalar-last — matches roman_attitude / quat_to_radec_pa)
quat = pointing.get_attitude_quaternion()
print(f"\nQuaternion [x,y,z,w] : {quat}")

# Euler angles (ZYX sequence)
euler = pointing.get_attitude_euler(sequence='ZYX', degrees=True)
print(f"Euler ZYX (deg)      : {euler}")

# Position angle (V3PA)
pa = pointing.get_position_angle()
print(f"Position angle (V3PA): {pa:.4f}")


# ═══════════════════════════════════════════════════════════════════════════════
# 5.  COORDINATE CONVERSIONS
# ═══════════════════════════════════════════════════════════════════════════════
section("5. Ecliptic coordinate conversion")

ecl_lon, ecl_lat = pointing.get_ecliptic_coordinates(pointing.target_coord)
print(f"Target ecliptic  : lon = {ecl_lon:.4f}°, lat = {ecl_lat:.4f}°")

sun_ecl_lon, sun_ecl_lat = pointing.get_ecliptic_coordinates(pointing.sun_coord)
print(f"Sun ecliptic     : lon = {sun_ecl_lon:.4f}°, lat = {sun_ecl_lat:.4f}°")


# ═══════════════════════════════════════════════════════════════════════════════
# 6.  PITCH & ROLL MANEUVERS
# ═══════════════════════════════════════════════════════════════════════════════
section("6. Pitch & roll maneuvers")

print("Applying small pitch/roll offsets from the current target:\n")
print(f"{'dPitch':>8s}  {'dRoll':>8s}  {'RA':>10s}  {'Dec':>10s}  {'PA':>10s}  {'Pitch':>10s}")
print("-" * 64)

for dp in [-10, -5, 0, 5, 10]:
    for dr in [-5, 0, 5]:
        ra, dec, pa, pitch = pointing.delta_pitch_roll(dpitch=dp, droll=dr)
        print(f"{dp:8.1f}  {dr:8.1f}  {ra:10.4f}  {dec:10.4f}  {pa:10.4f}  {pitch:10.4f}")


# ═══════════════════════════════════════════════════════════════════════════════
# 7.  NEW TARGET AT A DIFFERENT PITCH
# ═══════════════════════════════════════════════════════════════════════════════
section("7. Compute target at a new pitch angle")

initial_pitch = pointing.get_pitch_angle()
delta = 10.0  # degrees offset from current pitch
new_target, new_attitude = pointing.get_target_new_pitch(delta_pitch=delta)

print(f"Initial pitch          : {initial_pitch:.4f}")
print(f"Requested delta_pitch  : {delta}°")
print(f"New target             : RA = {new_target.ra.deg:.4f}°, "
      f"Dec = {new_target.dec.deg:.4f}°")
print(f"New pitch angle        : {pointing.get_pitch_angle(new_target):.4f}")
print(f"New position angle     : {pointing.get_position_angle(new_attitude):.4f}")


# ═══════════════════════════════════════════════════════════════════════════════
# 8.  PITCH RASTER SCAN (field of regard sampling)
# ═══════════════════════════════════════════════════════════════════════════════
section("8. Generate pitch-raster scan with power-law sampling")

n_points  = 14
range_max = 35.5
point_list = generate_power_law_sampling(n_points, range_max=range_max, power=4)
print(f"Sampled pitch angles ({n_points} pts, power=4):")
print(f"  {np.array2string(point_list, precision=2)}\n")

# Build a table of (Name, RA, Dec, Pitch, PA) for each raster point
raster_data = []
for i, target_pitch in enumerate(point_list):
    dp = float(target_pitch) - initial_pitch.value
    ra, dec, pa, pitch = pointing.delta_pitch_roll(dp, 0.0)
    raster_data.append({
        'Name': f'Target_{i:02d}',
        'RA': round(ra, 4),
        'Dec': round(dec, 4),
        'Pitch': round(pitch, 4),
        'PA': round(pa % 360, 4),
    })

df = pd.DataFrame(raster_data)
print(df.to_string(index=False))

# Optionally save to CSV
# df.to_csv('pitch_raster_demo.csv', index=False)
# print("\nSaved to pitch_raster_demo.csv")


# ═══════════════════════════════════════════════════════════════════════════════
# 9.  QUATERNION → RA/DEC/PA  (from visit file data)
# ═══════════════════════════════════════════════════════════════════════════════
section("9. Convert a quaternion to RA/Dec/V3PA")

# Example quaternion (scalar-last: x, y, z, w)
q_example = [0.1, 0.2, 0.3, 0.9]
q_example = q_example / np.linalg.norm(q_example)  # normalise

ra_q, dec_q, pa_q = roman_attitude(q_example)
print(f"Quaternion (x,y,z,w) : {q_example}")
print(f"  → RA  = {ra_q:.4f}°")
print(f"  → Dec = {dec_q:.4f}°")
print(f"  → V3PA= {pa_q:.4f}°")

# Same result via the alternate function
ra_q2, dec_q2, pa_q2 = quat_to_radec_pa(*q_example)
print(f"\nCross-check (quat_to_radec_pa): "
      f"RA={ra_q2:.4f}°, Dec={dec_q2:.4f}°, PA={pa_q2:.4f}°")


# ═══════════════════════════════════════════════════════════════════════════════
# 10. QUATERNION ROUND-TRIP CONSISTENCY CHECK
# ═══════════════════════════════════════════════════════════════════════════════
section("10. Quaternion round-trip: attitude → quaternion → RA/Dec/PA")

# get_attitude_quaternion now returns [x, y, z, w] (scalar-last),
# matching the convention of roman_attitude() and quat_to_radec_pa().
q_rt = pointing.get_attitude_quaternion()
print(f"Quaternion from attitude matrix : {q_rt}")

# Feed directly into roman_attitude()
ra_rt1, dec_rt1, pa_rt1 = roman_attitude(q_rt)
print(f"\nroman_attitude(q):")
print(f"  RA  = {ra_rt1:.6f}°   (expected {pointing.target_coord.ra.deg:.6f}°)")
print(f"  Dec = {dec_rt1:.6f}°   (expected {pointing.target_coord.dec.deg:.6f}°)")
print(f"  PA  = {pa_rt1:.6f}°   (expected {pointing.get_position_angle().value % 360:.6f}°)")

# Feed directly into quat_to_radec_pa()
ra_rt2, dec_rt2, pa_rt2 = quat_to_radec_pa(*q_rt)
print(f"\nquat_to_radec_pa(*q):")
print(f"  RA  = {ra_rt2:.6f}°")
print(f"  Dec = {dec_rt2:.6f}°")
print(f"  PA  = {pa_rt2:.6f}°")

# Verify numerically
tol = 1e-8
try:
    np.testing.assert_allclose(ra_rt1,  pointing.target_coord.ra.deg,  atol=tol)
    np.testing.assert_allclose(dec_rt1, pointing.target_coord.dec.deg, atol=tol)
    np.testing.assert_allclose(ra_rt1,  ra_rt2,  atol=tol)
    np.testing.assert_allclose(dec_rt1, dec_rt2, atol=tol)
    np.testing.assert_allclose(pa_rt1,  pa_rt2,  atol=tol)
    print("\n✓ Round-trip passed: attitude → quat → RA/Dec/PA is self-consistent.")
    print("✓ roman_attitude() and quat_to_radec_pa() agree to within 1e-8°.")
except AssertionError as e:
    print(f"\n✗ Round-trip FAILED: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
# 11. FULL 4-PANEL VISUALIZATION
# ═══════════════════════════════════════════════════════════════════════════════
section("11. Visualize pointing (4-panel plot)")

# Build a list of raster targets to overlay on the sky maps
target_list = []
for target_pitch in point_list:
    dp = float(target_pitch) - initial_pitch.value
    new_tgt, _ = pointing.get_target_new_pitch(delta_pitch=dp)
    target_list.append(new_tgt)

# Generate the 4-panel figure:
#   Panel 1 — 3-D celestial sphere with axes, ecliptic, Sun, target
#   Panel 2 — Mollweide sky map (ICRS) with ecliptic, CVZ, and targets
#   Panel 3 — 3-D spacecraft model showing body axes & solar array
#   Panel 4 — Mollweide sky map (Galactic coordinates)
pointing.visualize_pointing(
    show_ecliptic=True,
    show_sun=True,
    show_constellation_grid=True,
    target_list=target_list,
)


# ═══════════════════════════════════════════════════════════════════════════════
# 12. PYSIAF CROSS-CHECK  (optional — requires pysiaf)
# ═══════════════════════════════════════════════════════════════════════════════
section("12. Verify attitude matrix against pysiaf")

try:
    import pysiaf
    from pysiaf.utils.rotations import attitude

    boresight_ra  = pointing.target_coord.ra.deg
    boresight_dec = pointing.target_coord.dec.deg
    pa_v3         = pointing.get_position_angle().value

    att_pysiaf = attitude(0, 0, boresight_ra, boresight_dec, pa_v3)
    np.testing.assert_allclose(att_pysiaf, pointing.spacecraft_attitude,
                               rtol=1e-5, atol=1e-8)
    print("✓ Attitude matrix matches pysiaf to rtol=1e-5.")

    # Also check the Euler-angle path at a different PA
    new_pa = 150.0
    att_pysiaf2 = attitude(0, 0, boresight_ra, boresight_dec, new_pa)

    p2 = RomanPointing(obs_date)
    p2.set_target(pointing.target_coord)
    att_euler = p2._calculate_pointing_attitude_euler(new_pa)

    np.testing.assert_allclose(att_pysiaf2, att_euler, rtol=1e-5, atol=1e-8)
    print("✓ Euler-angle attitude path also matches pysiaf.")

except ImportError:
    print("pysiaf not installed — skipping cross-check.")
except Exception as e:
    print(f"pysiaf check failed: {e}")


print("\n" + "="*70)
print("  Demo complete!")
print("="*70)