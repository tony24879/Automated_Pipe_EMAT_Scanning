"""GUI launcher for calibration and scan scripts."""

from __future__ import annotations

import math
import subprocess
import sys
import tkinter as tk
import os
from pathlib import Path
from tkinter import messagebox


PROJECT_ROOT = Path(__file__).resolve().parent


def _format_duration(seconds: float) -> str:
	"""Format a duration as h:mm:ss or m:ss for the scan estimate display."""
	total_seconds = max(0, int(round(seconds)))
	hours, remainder = divmod(total_seconds, 3600)
	minutes, secs = divmod(remainder, 60)
	if hours:
		return f"{hours}h {minutes:02d}m {secs:02d}s"
	if minutes:
		return f"{minutes}m {secs:02d}s"
	return f"{secs}s"


def _estimate_horizontal_scan_time(theta_steps: int, axis_steps: int) -> tuple[float, float, float, int, int]:
	"""Estimate horizontal scan duration using the generated motion path."""
	from scans.cylindrical_scan_horizontal import HorizontalCylindricalScanPlanner, _read_horizontal_calibration_geometry

	calibration = _read_horizontal_calibration_geometry("cylinder_calibration_horizontal.json")
	if calibration is not None:
		centre = calibration["centre"]
		radius = calibration["radius"]
		length = calibration["length"]
		x_start = calibration["x_start"]
		x_end = calibration["x_end"]
		theta_limit_a_deg = calibration["theta_limit_a_deg"]
		theta_limit_b_deg = calibration["theta_limit_b_deg"]
	else:
		centre = [250.0, 0.0, 150.0]
		radius = 50.0
		length = 150.0
		x_start = centre[0]
		x_end = centre[0] + length
		theta_limit_a_deg = 0.0
		theta_limit_b_deg = 180.0

	planner = HorizontalCylindricalScanPlanner(
		centre=centre,
		radius=radius,
		length=length,
		lift_off=0.0,
		outer_offset_mm=0.0,
		x_start=x_start,
		x_end=x_end,
		theta_limit_a_deg=theta_limit_a_deg,
		theta_limit_b_deg=theta_limit_b_deg,
		scan_points_per_x=theta_steps,
		x_scans=axis_steps,
	)

	points = planner.generate()
	travel_distance_mm = sum(math.dist(start[:3], end[:3]) for start, end in zip(points, points[1:]))
	motion_point_count = len(points)
	speed = 40.0
	travel_seconds = (travel_distance_mm / speed) + (0.2 * motion_point_count)
	capture_count = sum(1 for *_, capture in points if capture)
	emat_averages = 1000
	block_seconds = emat_averages / 1000.0
	dwell = 0.5
	a = math.ceil(dwell / block_seconds)
	capture_seconds = capture_count * ((a * block_seconds) + (a * 0.1))
	total_seconds = travel_seconds + capture_seconds
	return total_seconds, travel_seconds, capture_seconds, motion_point_count, capture_count


def _launch_script(script_relative_path: str, args: list[str]) -> None:
	"""Launch a script in a separate terminal so interactive prompts remain usable."""
	script_path = PROJECT_ROOT / script_relative_path
	if not script_path.exists():
		raise FileNotFoundError(f"Script not found: {script_path}")

	command = [sys.executable, str(script_path), *args]
	env = dict(os.environ)
	existing_pythonpath = env.get("PYTHONPATH", "")
	env["PYTHONPATH"] = (
		f"{PROJECT_ROOT}{os.pathsep}{existing_pythonpath}"
		if existing_pythonpath
		else str(PROJECT_ROOT)
	)
	creationflags = 0
	if sys.platform.startswith("win"):
		creationflags = subprocess.CREATE_NEW_CONSOLE

	subprocess.Popen(
		command,
		cwd=str(PROJECT_ROOT),
		env=env,
		creationflags=creationflags,
	)


def _launch_module(module_name: str, args: list[str]) -> None:
	"""Launch a Python module in a separate terminal."""
	command = [sys.executable, "-m", module_name, *args]
	env = dict(os.environ)
	existing_pythonpath = env.get("PYTHONPATH", "")
	env["PYTHONPATH"] = (
		f"{PROJECT_ROOT}{os.pathsep}{existing_pythonpath}"
		if existing_pythonpath
		else str(PROJECT_ROOT)
	)
	creationflags = 0
	if sys.platform.startswith("win"):
		creationflags = subprocess.CREATE_NEW_CONSOLE

	subprocess.Popen(
		command,
		cwd=str(PROJECT_ROOT),
		env=env,
		creationflags=creationflags,
	)


def _show_calibrate_dialog(parent: tk.Tk) -> None:
	dialog = tk.Toplevel(parent)
	dialog.title("Calibrate")
	dialog.resizable(False, False)
	dialog.transient(parent)
	dialog.grab_set()

	frame = tk.Frame(dialog, padx=14, pady=12)
	frame.pack(fill="both", expand=True)

	tk.Label(frame, text="Number of calibration points:").grid(row=0, column=0, sticky="w")
	points_var = tk.StringVar(value="18")
	points_entry = tk.Entry(frame, textvariable=points_var, width=14)
	points_entry.grid(row=1, column=0, sticky="we", pady=(4, 10))
	points_entry.focus_set()

	buttons = tk.Frame(frame)
	buttons.grid(row=2, column=0, sticky="e")

	def on_calibrate() -> None:
		raw_value = points_var.get().strip()
		try:
			points = int(raw_value)
		except ValueError:
			messagebox.showerror("Invalid Input", "Calibration points must be an integer.", parent=dialog)
			return
		if points <= 0:
			messagebox.showerror("Invalid Input", "Calibration points must be greater than 0.", parent=dialog)
			return

		try:
			_launch_script("scripts/calibrate_cylinder_horizontal.py", ["--surface-points", str(points)])
		except Exception as exc:
			messagebox.showerror("Launch Error", f"Unable to start calibration:\n{exc}", parent=dialog)
			return
		dialog.destroy()

	tk.Button(buttons, text="Cancel", width=12, command=dialog.destroy).pack(side="left", padx=(0, 8))
	tk.Button(buttons, text="Calibrate", width=12, command=on_calibrate).pack(side="left")

	dialog.bind("<Return>", lambda _event: on_calibrate())
	dialog.bind("<Escape>", lambda _event: dialog.destroy())


def _show_scan_dialog(parent: tk.Tk) -> None:
	dialog = tk.Toplevel(parent)
	dialog.title("Run Scan")
	dialog.resizable(False, False)
	dialog.transient(parent)
	dialog.grab_set()

	frame = tk.Frame(dialog, padx=14, pady=12)
	frame.pack(fill="both", expand=True)

	tk.Label(frame, text="Number of theta steps:").grid(row=0, column=0, sticky="w")
	theta_var = tk.StringVar(value="12")
	theta_entry = tk.Entry(frame, textvariable=theta_var, width=14)
	theta_entry.grid(row=1, column=0, sticky="we", pady=(4, 10))

	tk.Label(frame, text="Number of axis steps:").grid(row=2, column=0, sticky="w")
	axis_var = tk.StringVar(value="5")
	axis_entry = tk.Entry(frame, textvariable=axis_var, width=14)
	axis_entry.grid(row=3, column=0, sticky="we", pady=(4, 10))
	theta_entry.focus_set()

	estimate_var = tk.StringVar(value="Estimated scan time: not calculated yet.")
	tk.Label(frame, textvariable=estimate_var, justify="left", wraplength=260).grid(row=4, column=0, sticky="w", pady=(0, 10))

	buttons = tk.Frame(frame)
	buttons.grid(row=5, column=0, sticky="e")

	def on_estimate_scan_time() -> None:
		try:
			theta_steps = int(theta_var.get().strip())
			axis_steps = int(axis_var.get().strip())
		except ValueError:
			messagebox.showerror("Invalid Input", "Theta and axis steps must be integers.", parent=dialog)
			return
		if theta_steps <= 0 or axis_steps <= 0:
			messagebox.showerror("Invalid Input", "Theta and axis steps must be greater than 0.", parent=dialog)
			return

		try:
			total_seconds, travel_seconds, capture_seconds, motion_points, capture_points = _estimate_horizontal_scan_time(theta_steps, axis_steps)
		except Exception as exc:
			messagebox.showerror("Estimate Error", f"Unable to estimate scan time:\n{exc}", parent=dialog)
			return

		estimate_var.set(
			"Estimated scan time: "
			f"{_format_duration(total_seconds)} "
			f"(travel {_format_duration(travel_seconds)}, captures {_format_duration(capture_seconds)}, "
			f"{motion_points} motion points, {capture_points} capture points)"
		)

	def on_run_scan() -> None:
		try:
			theta_steps = int(theta_var.get().strip())
			axis_steps = int(axis_var.get().strip())
		except ValueError:
			messagebox.showerror("Invalid Input", "Theta and axis steps must be integers.", parent=dialog)
			return
		if theta_steps <= 0 or axis_steps <= 0:
			messagebox.showerror("Invalid Input", "Theta and axis steps must be greater than 0.", parent=dialog)
			return

		try:
			_launch_module(
				"scans.cylindrical_scan_horizontal",
				[
					"--calibration-file",
					"cylinder_calibration_horizontal.json",
					"--view-3d-backend",
					"mesh",
					"--robot-mesh-dir",
					"3Dview/meshes/lite6",
					"--scan-points-per-x",
					str(theta_steps),
					"--x-scans",
					str(axis_steps),
				],
			)
		except Exception as exc:
			messagebox.showerror("Launch Error", f"Unable to start scan:\n{exc}", parent=dialog)
			return
		dialog.destroy()

	tk.Button(buttons, text="Cancel", width=12, command=dialog.destroy).pack(side="left", padx=(0, 8))
	tk.Button(buttons, text="Estimate Scan Time", width=18, command=on_estimate_scan_time).pack(side="left", padx=(0, 8))
	tk.Button(buttons, text="Run Scan", width=12, command=on_run_scan).pack(side="left")

	dialog.bind("<Control-Return>", lambda _event: on_estimate_scan_time())
	dialog.bind("<Return>", lambda _event: on_run_scan())
	dialog.bind("<Escape>", lambda _event: dialog.destroy())


def main() -> None:
	root = tk.Tk()
	root.title("Scan Runner")
	root.resizable(False, False)

	frame = tk.Frame(root, padx=18, pady=16)
	frame.pack(fill="both", expand=True)

	tk.Label(frame, text="Choose an action", font=("Segoe UI", 12, "bold")).pack(anchor="w", pady=(0, 10))
	tk.Button(frame, text="Calibrate", width=22, command=lambda: _show_calibrate_dialog(root)).pack(pady=(0, 8))
	tk.Button(frame, text="Run Scan", width=22, command=lambda: _show_scan_dialog(root)).pack(pady=(0, 8))
	tk.Button(frame, text="Exit", width=22, command=root.destroy).pack()

	root.mainloop()


if __name__ == "__main__":
	main()
