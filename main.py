"""GUI launcher for calibration and scan scripts."""

from __future__ import annotations

import subprocess
import sys
import tkinter as tk
import os
from pathlib import Path
from tkinter import messagebox


PROJECT_ROOT = Path(__file__).resolve().parent


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

	buttons = tk.Frame(frame)
	buttons.grid(row=4, column=0, sticky="e")

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
	tk.Button(buttons, text="Run Scan", width=12, command=on_run_scan).pack(side="left")

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
