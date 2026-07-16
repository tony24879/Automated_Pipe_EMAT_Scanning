"""Interactive calibration for horizontal-cylinder scans with robust 3D cylinder fitting."""

import json
import math
import random
import argparse
from pathlib import Path

from config.robot_config import ROBOT_IP
from robot.connection import RobotConnection
from robot.lite6 import Lite6
from robot.setup import RobotSetup


def _fit_circle_2d(points):
    """Least-squares algebraic circle fit to a list of 2D points."""
    n = len(points)
    if n < 3:
        raise ValueError("At least 3 points required for a circle fit")

    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    r2s = [x ** 2 + y ** 2 for x, y in zip(xs, ys)]

    sxx = sum(x ** 2 for x in xs)
    syy = sum(y ** 2 for y in ys)
    sxy = sum(x * y for x, y in zip(xs, ys))
    sx = sum(xs)
    sy = sum(ys)
    sxr = sum(x * r for x, r in zip(xs, r2s))
    syr = sum(y * r for y, r in zip(ys, r2s))
    sr = sum(r2s)

    mat = [
        [sxx, sxy, sx, -sxr],
        [sxy, syy, sy, -syr],
        [sx, sy, n, -sr],
    ]

    for col in range(3):
        max_row = max(range(col, 3), key=lambda r: abs(mat[r][col]))
        mat[col], mat[max_row] = mat[max_row], mat[col]
        if abs(mat[col][col]) < 1e-12:
            raise ValueError("Circle fit failed - points may be collinear or too close together")
        for row in range(col + 1, 3):
            factor = mat[row][col] / mat[col][col]
            for k in range(col, 4):
                mat[row][k] -= factor * mat[col][k]

    sol = [0.0, 0.0, 0.0]
    for row in range(2, -1, -1):
        sol[row] = mat[row][3]
        for k in range(row + 1, 3):
            sol[row] -= mat[row][k] * sol[k]
        sol[row] /= mat[row][row]

    a_coeff, b_coeff, c_coeff = sol
    cx = -a_coeff / 2.0
    cy = -b_coeff / 2.0
    radius = math.sqrt(max(0.0, cx ** 2 + cy ** 2 - c_coeff))
    return cx, cy, radius


def _fit_circle_2d_weighted(points, weights):
    """Weighted least-squares algebraic circle fit to a list of 2D points."""
    n = len(points)
    if n < 3:
        raise ValueError("At least 3 points required for a circle fit")
    if len(weights) != n:
        raise ValueError("weights length must match points length")

    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    r2s = [x ** 2 + y ** 2 for x, y in zip(xs, ys)]

    sw = sum(weights)
    if sw <= 0.0:
        raise ValueError("sum of weights must be positive")

    sxx = sum(w * (x ** 2) for w, x in zip(weights, xs))
    syy = sum(w * (y ** 2) for w, y in zip(weights, ys))
    sxy = sum(w * x * y for w, x, y in zip(weights, xs, ys))
    sx = sum(w * x for w, x in zip(weights, xs))
    sy = sum(w * y for w, y in zip(weights, ys))
    sxr = sum(w * x * r for w, x, r in zip(weights, xs, r2s))
    syr = sum(w * y * r for w, y, r in zip(weights, ys, r2s))
    sr = sum(w * r for w, r in zip(weights, r2s))

    mat = [
        [sxx, sxy, sx, -sxr],
        [sxy, syy, sy, -syr],
        [sx, sy, sw, -sr],
    ]

    for col in range(3):
        max_row = max(range(col, 3), key=lambda r: abs(mat[r][col]))
        mat[col], mat[max_row] = mat[max_row], mat[col]
        if abs(mat[col][col]) < 1e-12:
            raise ValueError("Circle fit failed - points may be collinear or too close together")
        for row in range(col + 1, 3):
            factor = mat[row][col] / mat[col][col]
            for k in range(col, 4):
                mat[row][k] -= factor * mat[col][k]

    sol = [0.0, 0.0, 0.0]
    for row in range(2, -1, -1):
        sol[row] = mat[row][3]
        for k in range(row + 1, 3):
            sol[row] -= mat[row][k] * sol[k]
        sol[row] /= mat[row][row]

    a_coeff, b_coeff, c_coeff = sol
    cx = -a_coeff / 2.0
    cy = -b_coeff / 2.0
    radius = math.sqrt(max(0.0, cx ** 2 + cy ** 2 - c_coeff))
    return cx, cy, radius


def _dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _sub(a, b):
    return [a[0] - b[0], a[1] - b[1], a[2] - b[2]]


def _add(a, b):
    return [a[0] + b[0], a[1] + b[1], a[2] + b[2]]


def _scale(a, s):
    return [a[0] * s, a[1] * s, a[2] * s]


def _norm(a):
    return math.sqrt(_dot(a, a))


def _normalize(a):
    n = _norm(a)
    if n < 1e-12:
        return [1.0, 0.0, 0.0]
    return [a[0] / n, a[1] / n, a[2] / n]


def _cross(a, b):
    return [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]


def _weighted_centroid(points, weights):
    sw = sum(weights)
    if sw <= 0.0:
        raise ValueError("sum of weights must be positive")
    c = [0.0, 0.0, 0.0]
    for p, w in zip(points, weights):
        c[0] += w * p[0]
        c[1] += w * p[1]
        c[2] += w * p[2]
    return [c[0] / sw, c[1] / sw, c[2] / sw]


def _dominant_eigenvector_sym3(cov, iters=30):
    """Power iteration for dominant eigenvector of a symmetric 3x3 matrix."""
    v = [1.0, 0.3, 0.2]
    v = _normalize(v)
    for _ in range(iters):
        w = [
            cov[0][0] * v[0] + cov[0][1] * v[1] + cov[0][2] * v[2],
            cov[1][0] * v[0] + cov[1][1] * v[1] + cov[1][2] * v[2],
            cov[2][0] * v[0] + cov[2][1] * v[1] + cov[2][2] * v[2],
        ]
        v = _normalize(w)
    return v


def _axis_basis(axis_direction):
    """Build orthonormal (v, w) spanning the plane normal to axis_direction."""
    u = _normalize(axis_direction)
    ref = [1.0, 0.0, 0.0] if abs(u[0]) < 0.9 else [0.0, 1.0, 0.0]
    v = _cross(u, ref)
    if _norm(v) < 1e-12:
        ref = [0.0, 0.0, 1.0]
        v = _cross(u, ref)
    v = _normalize(v)
    w = _normalize(_cross(u, v))
    return v, w


def _fit_cylinder_once(points, weights):
    """Single weighted cylinder fit estimate from 3D surface contact points."""
    centroid = _weighted_centroid(points, weights)

    cov = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
    sw = sum(weights)
    for p, wt in zip(points, weights):
        d = _sub(p, centroid)
        cov[0][0] += wt * d[0] * d[0]
        cov[0][1] += wt * d[0] * d[1]
        cov[0][2] += wt * d[0] * d[2]
        cov[1][1] += wt * d[1] * d[1]
        cov[1][2] += wt * d[1] * d[2]
        cov[2][2] += wt * d[2] * d[2]
    cov[1][0] = cov[0][1]
    cov[2][0] = cov[0][2]
    cov[2][1] = cov[1][2]
    for i in range(3):
        for j in range(3):
            cov[i][j] /= max(sw, 1e-12)

    axis_direction = _dominant_eigenvector_sym3(cov)
    if axis_direction[0] < 0.0:
        axis_direction = _scale(axis_direction, -1.0)

    v, w = _axis_basis(axis_direction)
    proj_points = []
    for p in points:
        d = _sub(p, centroid)
        proj_points.append((_dot(d, v), _dot(d, w)))

    cx2d, cy2d, radius = _fit_circle_2d_weighted(proj_points, weights)
    axis_point = _add(centroid, _add(_scale(v, cx2d), _scale(w, cy2d)))

    residuals = []
    for p in points:
        ap = _sub(p, axis_point)
        t = _dot(ap, axis_direction)
        closest = _add(axis_point, _scale(axis_direction, t))
        radial = _norm(_sub(p, closest))
        residuals.append(radial - radius)

    return {
        "axis_point": axis_point,
        "axis_direction": axis_direction,
        "radius": radius,
        "residuals": residuals,
    }


def _median(values):
    if not values:
        return 0.0
    sorted_values = sorted(values)
    n = len(sorted_values)
    mid = n // 2
    if n % 2 == 1:
        return sorted_values[mid]
    return 0.5 * (sorted_values[mid - 1] + sorted_values[mid])


def _percentile(values, q):
    if not values:
        return 0.0
    s = sorted(values)
    if len(s) == 1:
        return s[0]
    q = min(1.0, max(0.0, float(q)))
    idx = q * (len(s) - 1)
    lo = int(math.floor(idx))
    hi = int(math.ceil(idx))
    if lo == hi:
        return s[lo]
    t = idx - lo
    return s[lo] * (1.0 - t) + s[hi] * t


def _default_acceptance_thresholds():
    """Default calibration acceptance criteria for post-fit gating."""
    return {
        "max_rms_radial_error_mm": 1.0,
        "max_p95_radial_error_mm": 2.0,
        "min_inlier_ratio": 0.85,
        "min_quality_score": 70.0,
        "max_radius_ci95_width_mm": 1.0,
        "max_axis_tilt_ci95_width_deg": 1.0,
    }


def _evaluate_acceptance_gate(
    rms_radial_error_mm,
    p95_radial_error_mm,
    inlier_ratio,
    quality_score,
    bootstrap,
    thresholds,
):
    """Check fit quality against acceptance thresholds."""
    failures = []
    if rms_radial_error_mm > thresholds["max_rms_radial_error_mm"]:
        failures.append(
            f"RMS radial error {rms_radial_error_mm:.3f} mm exceeds {thresholds['max_rms_radial_error_mm']:.3f} mm"
        )
    if p95_radial_error_mm > thresholds["max_p95_radial_error_mm"]:
        failures.append(
            f"P95 radial error {p95_radial_error_mm:.3f} mm exceeds {thresholds['max_p95_radial_error_mm']:.3f} mm"
        )
    if inlier_ratio < thresholds["min_inlier_ratio"]:
        failures.append(
            f"Inlier ratio {inlier_ratio:.3f} is below {thresholds['min_inlier_ratio']:.3f}"
        )
    if quality_score < thresholds["min_quality_score"]:
        failures.append(
            f"Quality score {quality_score:.1f} is below {thresholds['min_quality_score']:.1f}"
        )

    if bootstrap:
        radius_ci = bootstrap["radius_ci95_mm"]
        tilt_ci = bootstrap["axis_tilt_ci95_deg"]
        radius_ci_width = float(radius_ci[1] - radius_ci[0])
        tilt_ci_width = float(tilt_ci[1] - tilt_ci[0])
        if radius_ci_width > thresholds["max_radius_ci95_width_mm"]:
            failures.append(
                f"Radius CI width {radius_ci_width:.3f} mm exceeds {thresholds['max_radius_ci95_width_mm']:.3f} mm"
            )
        if tilt_ci_width > thresholds["max_axis_tilt_ci95_width_deg"]:
            failures.append(
                f"Axis-tilt CI width {tilt_ci_width:.4f} deg exceeds "
                f"{thresholds['max_axis_tilt_ci95_width_deg']:.4f} deg"
            )

    return failures


def _fit_cylinder_3d(points, max_iters=20):
    """Robust IRLS cylinder fit in 3D using radial residuals."""
    if len(points) < 8:
        raise ValueError("At least 8 taught surface points are recommended for robust 3D cylinder fitting")

    weights = [1.0 for _ in points]
    fit = None
    prev_rms = None

    for _ in range(max_iters):
        fit = _fit_cylinder_once(points, weights)
        residuals = fit["residuals"]
        abs_res = [abs(r) for r in residuals]
        sigma = 1.4826 * _median(abs_res)
        sigma = max(sigma, 1e-6)

        huber_k = 1.5
        c = huber_k * sigma
        new_weights = []
        for r in residuals:
            ar = abs(r)
            if ar <= c:
                new_weights.append(1.0)
            else:
                new_weights.append(c / ar)

        rms = math.sqrt(sum(r * r for r in residuals) / len(residuals))
        if prev_rms is not None and abs(prev_rms - rms) < 1e-5:
            weights = new_weights
            break

        prev_rms = rms
        weights = new_weights

    fit = _fit_cylinder_once(points, weights)
    residuals = fit["residuals"]
    abs_res = [abs(r) for r in residuals]
    sigma = max(1e-6, 1.4826 * _median(abs_res))
    inlier_thresh = 2.5 * sigma
    inlier_count = sum(1 for a in abs_res if a <= inlier_thresh)

    fit["weights"] = weights
    fit["sigma"] = sigma
    fit["inlier_count"] = inlier_count
    fit["total_count"] = len(points)
    fit["rms_radial_error_mm"] = math.sqrt(sum(r * r for r in residuals) / len(residuals))
    fit["p95_radial_error_mm"] = _percentile(abs_res, 0.95)
    return fit


def _bootstrap_cylinder_confidence(points, bootstrap_samples=200):
    """Estimate confidence intervals for radius and axis tilt via bootstrap."""
    if len(points) < 8:
        return None

    radii = []
    tilts = []
    n = len(points)
    x_axis = [1.0, 0.0, 0.0]

    for _ in range(bootstrap_samples):
        sample = [points[random.randint(0, n - 1)] for _ in range(n)]
        try:
            fit = _fit_cylinder_3d(sample, max_iters=10)
        except ValueError:
            continue

        axis_direction = _normalize(fit["axis_direction"])
        cosang = max(-1.0, min(1.0, abs(_dot(axis_direction, x_axis))))
        tilt = math.degrees(math.acos(cosang))
        radii.append(float(fit["radius"]))
        tilts.append(float(tilt))

    if len(radii) < 20:
        return None

    return {
        "radius_ci95_mm": [_percentile(radii, 0.025), _percentile(radii, 0.975)],
        "axis_tilt_ci95_deg": [_percentile(tilts, 0.025), _percentile(tilts, 0.975)],
        "bootstrap_count": len(radii),
    }


def _closest_equivalent_angle(target_deg, reference_deg):
    """Choose equivalent target angle nearest to reference angle."""
    candidates = [target_deg - 720.0, target_deg - 360.0, target_deg, target_deg + 360.0, target_deg + 720.0]
    return min(candidates, key=lambda c: (abs(c - reference_deg), abs(c)))


def _theta_yz_deg(point, cy, cz):
    """Compute yz-plane polar angle (deg) for a point around horizontal cylinder axis."""
    return math.degrees(math.atan2(point[2] - cz, point[1] - cy))


def save_calibration(
    filename,
    centre,
    radius,
    x_start,
    x_end,
    theta_limit_a_deg,
    theta_limit_b_deg,
    raw_points,
    extras=None,
):
    """Persist fitted horizontal-cylinder geometry and taught points to JSON."""
    calibration = {
        "centre": [float(v) for v in centre],
        "radius": float(radius),
        "x_start": float(x_start),
        "x_end": float(x_end),
        "theta_limit_a_deg": float(theta_limit_a_deg),
        "theta_limit_b_deg": float(theta_limit_b_deg),
        "raw_points": [[float(v) for v in p] for p in raw_points],
    }
    if extras:
        calibration.update(extras)
    Path(filename).write_text(json.dumps(calibration, indent=2))


def main(surface_points=None):
    """Guide operator through horizontal-cylinder calibration and save geometry."""
    calibration_file = Path("data/raw/cylinder_calibration_horizontal.json")
    calibration_file.parent.mkdir(parents=True, exist_ok=True)

    conn = RobotConnection(ROBOT_IP)
    arm = conn.connect()
    robot = Lite6(arm)
    setup = RobotSetup(arm)
    setup.configure()

    try:
        print("Switching robot to teach mode for manual touch movement...")
        arm.motion_enable(True)
        arm.set_mode(2)
        arm.set_state(0)

        print("\n--- PHASE 1: 3D cylinder surface teaching ---")
        print("Touch multiple surface points spread across both circumference and x range.")
        print("Important: point 1 and point 2 will become your scan theta limits.")
        print("For top-surface-only scans, keep these two limits within about 180 degrees span.")

        default_count = 18
        if surface_points is None:
            taught_count_text = input(
                f"How many surface points to teach? (recommended >= 18, default {default_count}): "
            ).strip()
            taught_count = default_count if not taught_count_text else int(taught_count_text)
        else:
            taught_count = int(surface_points)
            print(f"Using surface point count from CLI: {taught_count}")
        taught_count = max(8, taught_count)

        circ_points = []
        for i in range(taught_count):
            input(f"\nSurface point {i + 1}/{taught_count} - move to surface, then press ENTER")
            pose = robot.get_pose()
            circ_points.append(pose)
            print(f"  Recorded: x={pose[0]:.2f}  y={pose[1]:.2f}  z={pose[2]:.2f}")

        print("\n--- PHASE 2: Axis range (X limits) ---")
        input("\nMove to one end of the scan range along cylinder axis (x-start), then press ENTER")
        x_start_pose = robot.get_pose()
        print(f"  Axis start x: {x_start_pose[0]:.2f} mm")

        input("\nMove to the opposite end of the scan range along cylinder axis (x-end), then press ENTER")
        x_end_pose = robot.get_pose()
        print(f"  Axis end x: {x_end_pose[0]:.2f} mm")

        print("Restoring normal robot mode...")
        arm.set_mode(0)
        arm.set_state(0)
    finally:
        conn.disconnect()

    fit_points = [[float(p[0]), float(p[1]), float(p[2])] for p in circ_points]
    fit = _fit_cylinder_3d(fit_points)
    axis_point = fit["axis_point"]
    axis_direction = _normalize(fit["axis_direction"])
    radius = float(fit["radius"])
    residuals = fit["residuals"]

    # Keep legacy planner compatibility by deriving YZ centre at the taught x midpoint.
    x_start = min(float(x_start_pose[0]), float(x_end_pose[0]))
    x_end = max(float(x_start_pose[0]), float(x_end_pose[0]))
    centre_x = 0.5 * (x_start + x_end)

    if abs(axis_direction[0]) > 1e-8:
        t_mid = (centre_x - axis_point[0]) / axis_direction[0]
        axis_mid = _add(axis_point, _scale(axis_direction, t_mid))
    else:
        axis_mid = axis_point[:]

    cy = axis_mid[1]
    cz = axis_mid[2]
    centre = [centre_x, cy, cz]

    theta_a = _theta_yz_deg(circ_points[0], cy, cz)
    theta_b_raw = _theta_yz_deg(circ_points[1], cy, cz)
    theta_b = _closest_equivalent_angle(theta_b_raw, theta_a)
    theta_span = abs(theta_b - theta_a)

    x_axis = [1.0, 0.0, 0.0]
    tilt_cos = max(-1.0, min(1.0, abs(_dot(axis_direction, x_axis))))
    axis_tilt_deg = math.degrees(math.acos(tilt_cos))

    print(f"\nFitted axis point:     x={axis_point[0]:.2f}  y={axis_point[1]:.2f}  z={axis_point[2]:.2f}")
    print(
        "Fitted axis direction: "
        f"[{axis_direction[0]:+.5f}, {axis_direction[1]:+.5f}, {axis_direction[2]:+.5f}]"
    )
    print(f"Axis tilt from +x:     {axis_tilt_deg:.3f} deg")
    print(f"Compatibility centre:  y={cy:.2f}  z={cz:.2f}")
    print(f"Fitted radius:         {radius:.2f} mm")
    print(f"Axis range (x):        {x_start:.2f} mm  ->  {x_end:.2f} mm")
    print(f"Theta limits:          {theta_a:.2f} deg  ->  {theta_b:.2f} deg")
    print(f"Theta span:            {theta_span:.2f} deg")

    if theta_span > 180.0 + 1e-6:
        print("Warning: taught theta span exceeds 180 degrees; top-surface scan intent may be violated.")

    rms_radial_error_mm = float(fit["rms_radial_error_mm"])
    p95_radial_error_mm = float(fit["p95_radial_error_mm"])
    inlier_count = int(fit["inlier_count"])
    total_count = int(fit["total_count"])
    outlier_ratio = 1.0 - (inlier_count / max(total_count, 1))
    inlier_ratio = 1.0 - outlier_ratio

    quality_score = 100.0
    quality_score -= min(60.0, 12.0 * rms_radial_error_mm)
    quality_score -= min(20.0, 4.0 * p95_radial_error_mm)
    quality_score -= min(20.0, 100.0 * outlier_ratio)
    quality_score = max(0.0, min(100.0, quality_score))

    print(f"RMS radial residual:   {rms_radial_error_mm:.3f} mm")
    print(f"P95 radial residual:   {p95_radial_error_mm:.3f} mm")
    print(f"Inliers:               {inlier_count}/{total_count}")
    print(f"Quality score:         {quality_score:.1f} / 100")

    print("\nComputing bootstrap confidence intervals (this may take a few seconds)...")
    bootstrap = _bootstrap_cylinder_confidence(fit_points, bootstrap_samples=200)
    if bootstrap:
        radius_ci = bootstrap["radius_ci95_mm"]
        tilt_ci = bootstrap["axis_tilt_ci95_deg"]
        print(f"Radius 95% CI:         [{radius_ci[0]:.3f}, {radius_ci[1]:.3f}] mm")
        print(f"Axis tilt 95% CI:      [{tilt_ci[0]:.4f}, {tilt_ci[1]:.4f}] deg")
    else:
        print("Bootstrap confidence unavailable (insufficient stable bootstrap samples)")

    print("\nPer-point radial residuals in 3D fit (ideal: +/-1 mm):")
    for i, residual in enumerate(residuals):
        print(f"  Point {i + 1}: {residual:+.2f} mm")

    thresholds = _default_acceptance_thresholds()
    failures = _evaluate_acceptance_gate(
        rms_radial_error_mm=rms_radial_error_mm,
        p95_radial_error_mm=p95_radial_error_mm,
        inlier_ratio=inlier_ratio,
        quality_score=quality_score,
        bootstrap=bootstrap,
        thresholds=thresholds,
    )

    gate_passed = len(failures) == 0
    if gate_passed:
        print("\nAcceptance gate: PASS")
    else:
        print("\nAcceptance gate: FAIL")
        for failure in failures:
            print(f"  - {failure}")

        force_save = input("Calibration failed acceptance gate. Save anyway? [y/N]: ").strip().lower()
        if force_save not in ("y", "yes"):
            print("Calibration not saved. Please re-teach points and rerun calibration.")
            return

    all_points = circ_points + [x_start_pose, x_end_pose]
    extras = {
        "fit_version": "horizontal_cylinder_3d_v1",
        "acceptance_gate": {
            "passed": gate_passed,
            "failures": failures,
            "thresholds": {k: float(v) for k, v in thresholds.items()},
        },
        "axis_point": [float(v) for v in axis_point],
        "axis_direction": [float(v) for v in axis_direction],
        "fit_stats": {
            "rms_radial_error_mm": rms_radial_error_mm,
            "p95_radial_error_mm": p95_radial_error_mm,
            "inlier_count": inlier_count,
            "total_count": total_count,
            "axis_tilt_deg_from_x": float(axis_tilt_deg),
            "quality_score": float(quality_score),
            "robust_sigma_mm": float(fit["sigma"]),
        },
    }
    if bootstrap:
        extras["confidence"] = {
            "radius_ci95_mm": [float(v) for v in bootstrap["radius_ci95_mm"]],
            "axis_tilt_ci95_deg": [float(v) for v in bootstrap["axis_tilt_ci95_deg"]],
            "bootstrap_count": int(bootstrap["bootstrap_count"]),
        }

    save_calibration(
        calibration_file,
        centre,
        radius,
        x_start,
        x_end,
        theta_a,
        theta_b,
        all_points,
        extras=extras,
    )
    print(f"\nHorizontal calibration saved to {calibration_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run horizontal cylinder calibration")
    parser.add_argument(
        "--surface-points",
        type=int,
        default=None,
        help="Number of surface points to teach before fitting",
    )
    args = parser.parse_args()
    main(surface_points=args.surface_points)
