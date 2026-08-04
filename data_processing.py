"""GUI launcher for data-processing scripts."""

from __future__ import annotations

import subprocess
import sys
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox

PROJECT_ROOT = Path(__file__).resolve().parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


def _launch_script(script_relative_path: str, args: list[str]) -> None:
    """Launch a script in a separate terminal window."""
    script_path = PROJECT_ROOT / script_relative_path
    if not script_path.exists():
        raise FileNotFoundError(f"Script not found: {script_path}")

    command = [sys.executable, str(script_path), *args]
    creationflags = 0
    if sys.platform.startswith("win"):
        # Separate console keeps interactive prompts and matplotlib windows
        # independent from the launcher UI process.
        creationflags = subprocess.CREATE_NEW_CONSOLE

    subprocess.Popen(command, cwd=str(PROJECT_ROOT), creationflags=creationflags)


def _add_single_csv_input(parent: tk.Widget, row: int, label: str, default: str = "") -> tk.StringVar:
    tk.Label(parent, text=label).grid(row=row, column=0, sticky="w")
    var = tk.StringVar(value=default)
    entry = tk.Entry(parent, textvariable=var, width=62)
    entry.grid(row=row + 1, column=0, sticky="we", padx=(0, 8), pady=(4, 10))

    def browse() -> None:
        path = filedialog.askopenfilename(
            title=label,
            initialdir=str(RAW_DIR if RAW_DIR.exists() else PROJECT_ROOT),
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
        if path:
            var.set(path)

    tk.Button(parent, text="Browse...", command=browse, width=12).grid(row=row + 1, column=1, sticky="e")
    return var


def _add_output_folder_input(parent: tk.Widget, row: int, label: str, default: str = "") -> tk.StringVar:
    tk.Label(parent, text=label).grid(row=row, column=0, sticky="w")
    var = tk.StringVar(value=default)
    entry = tk.Entry(parent, textvariable=var, width=62)
    entry.grid(row=row + 1, column=0, sticky="we", padx=(0, 8), pady=(4, 10))

    def browse() -> None:
        path = filedialog.askdirectory(
            title=label,
            initialdir=str(PROCESSED_DIR if PROCESSED_DIR.exists() else PROJECT_ROOT),
        )
        if path:
            var.set(path)

    tk.Button(parent, text="Browse...", command=browse, width=12).grid(row=row + 1, column=1, sticky="e")
    return var


def _add_output_file_input(parent: tk.Widget, row: int, label: str, default: str = "") -> tk.StringVar:
    tk.Label(parent, text=label).grid(row=row, column=0, sticky="w")
    var = tk.StringVar(value=default)
    entry = tk.Entry(parent, textvariable=var, width=62)
    entry.grid(row=row + 1, column=0, sticky="we", padx=(0, 8), pady=(4, 10))

    def browse() -> None:
        path = filedialog.asksaveasfilename(
            title=label,
            initialdir=str(PROCESSED_DIR if PROCESSED_DIR.exists() else PROJECT_ROOT),
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
        if path:
            var.set(path)

    tk.Button(parent, text="Browse...", command=browse, width=12).grid(row=row + 1, column=1, sticky="e")
    return var


def _open_dialog(parent: tk.Tk, title: str) -> tuple[tk.Toplevel, tk.Frame]:
    """Create a modal child dialog with consistent sizing and focus behavior."""
    dialog = tk.Toplevel(parent)
    dialog.title(title)
    dialog.resizable(False, False)
    dialog.transient(parent)
    dialog.grab_set()

    frame = tk.Frame(dialog, padx=14, pady=12)
    frame.pack(fill="both", expand=True)
    return dialog, frame


def _show_apply_hilbert(parent: tk.Tk) -> None:
    dialog, frame = _open_dialog(parent, "Apply Hilbert Filter")
    csv_var = _add_single_csv_input(frame, 0, "Input CSV file:")
    out_var = _add_output_folder_input(frame, 2, "Output folder location:", default=str(PROCESSED_DIR))

    buttons = tk.Frame(frame)
    buttons.grid(row=4, column=0, columnspan=2, sticky="e", pady=(4, 0))

    def run() -> None:
        input_csv = csv_var.get().strip()
        output_folder = out_var.get().strip()
        if not input_csv:
            messagebox.showerror("Missing Input", "Please select an input CSV file.", parent=dialog)
            return
        if not output_folder:
            messagebox.showerror("Missing Output", "Please select an output folder.", parent=dialog)
            return

        try:
            _launch_script(
                "data/apply_hilbert_to_raw_scan.py",
                [input_csv, "--output-folder", output_folder],
            )
        except Exception as exc:  # noqa: BLE001 - propagate any launcher failure to the GUI error dialog.
            messagebox.showerror("Launch Error", f"Unable to run script:\n{exc}", parent=dialog)
            return
        dialog.destroy()

    tk.Button(buttons, text="Cancel", width=12, command=dialog.destroy).pack(side="left", padx=(0, 8))
    tk.Button(buttons, text="Run", width=12, command=run).pack(side="left")


def _show_first_peak_tof(parent: tk.Tk) -> None:
    dialog, frame = _open_dialog(parent, "Calculate First Peak Time of Flight")
    csv_var = _add_single_csv_input(frame, 0, "Input CSV file:")
    out_var = _add_output_folder_input(frame, 2, "Output folder location:", default=str(PROCESSED_DIR))

    buttons = tk.Frame(frame)
    buttons.grid(row=4, column=0, columnspan=2, sticky="e", pady=(4, 0))

    def run() -> None:
        input_csv = csv_var.get().strip()
        output_folder = out_var.get().strip()
        if not input_csv:
            messagebox.showerror("Missing Input", "Please select an input CSV file.", parent=dialog)
            return
        if not output_folder:
            messagebox.showerror("Missing Output", "Please select an output folder.", parent=dialog)
            return

        try:
            _launch_script(
                "data/calculate_first_peak_tof.py",
                [input_csv, "--output-folder", output_folder],
            )
        except Exception as exc:  # noqa: BLE001 - propagate any launcher failure to the GUI error dialog.
            messagebox.showerror("Launch Error", f"Unable to run script:\n{exc}", parent=dialog)
            return
        dialog.destroy()

    tk.Button(buttons, text="Cancel", width=12, command=dialog.destroy).pack(side="left", padx=(0, 8))
    tk.Button(buttons, text="Run", width=12, command=run).pack(side="left")


def _show_average_tof(parent: tk.Tk) -> None:
    dialog, frame = _open_dialog(parent, "Calculate Average ToF")
    csv_var = _add_single_csv_input(frame, 0, "Input CSV file:")
    default_output = str(PROCESSED_DIR / "scan_avg_tof_3.csv")
    out_file_var = _add_output_file_input(frame, 2, "Output CSV file location:", default=default_output)

    tk.Label(frame, text="Number of peaks:").grid(row=4, column=0, sticky="w")
    num_peaks_var = tk.StringVar(value="3")
    tk.Entry(frame, textvariable=num_peaks_var, width=20).grid(row=5, column=0, sticky="w", pady=(4, 10))

    output_manually_edited = False

    def _suggest_output_path() -> str:
        input_csv = csv_var.get().strip()
        stem = "scan"
        if input_csv:
            stem = Path(input_csv).stem or "scan"

        peaks_text = num_peaks_var.get().strip()
        peak_suffix = peaks_text if peaks_text else "3"
        return str(PROCESSED_DIR / f"{stem}_avg_tof_{peak_suffix}.csv")

    def _handle_input_change(*_: object) -> None:
        nonlocal output_manually_edited
        if output_manually_edited:
            return
        out_file_var.set(_suggest_output_path())

    def _handle_output_edit(*_: object) -> None:
        nonlocal output_manually_edited
        suggested = _suggest_output_path()
        output_manually_edited = out_file_var.get().strip() != suggested

    csv_var.trace_add("write", _handle_input_change)
    num_peaks_var.trace_add("write", _handle_input_change)
    out_file_var.trace_add("write", _handle_output_edit)

    out_file_var.set(_suggest_output_path())

    buttons = tk.Frame(frame)
    buttons.grid(row=6, column=0, columnspan=2, sticky="e", pady=(4, 0))

    def run() -> None:
        input_csv = csv_var.get().strip()
        output_csv = out_file_var.get().strip()
        if not input_csv:
            messagebox.showerror("Missing Input", "Please select an input CSV file.", parent=dialog)
            return
        if not output_csv:
            messagebox.showerror("Missing Output", "Please select an output CSV file location.", parent=dialog)
            return

        try:
            num_peaks = int(num_peaks_var.get().strip())
        except ValueError:
            messagebox.showerror("Invalid Input", "Number of peaks must be an integer.", parent=dialog)
            return

        if num_peaks < 2:
            messagebox.showerror("Invalid Input", "Number of peaks must be at least 2.", parent=dialog)
            return

        try:
            _launch_script(
                "data/calculate_average_pairwise_tof.py",
                [input_csv, "--num-peaks", str(num_peaks), "--output-file", output_csv],
            )
        except Exception as exc:  # noqa: BLE001 - propagate any launcher failure to the GUI error dialog.
            messagebox.showerror("Launch Error", f"Unable to run script:\n{exc}", parent=dialog)
            return
        dialog.destroy()

    tk.Button(buttons, text="Exit", width=12, command=dialog.destroy).pack(side="left", padx=(0, 8))
    tk.Button(buttons, text="Run", width=12, command=run).pack(side="left")


def _show_compute_errors(parent: tk.Tk) -> None:
    dialog, frame = _open_dialog(parent, "Calculate Errors")
    first_var = _add_single_csv_input(frame, 0, "First input CSV file:")
    second_var = _add_single_csv_input(frame, 2, "Second input CSV file:")
    out_var = _add_output_folder_input(frame, 4, "Output folder location:", default=str(PROCESSED_DIR))

    buttons = tk.Frame(frame)
    buttons.grid(row=6, column=0, columnspan=2, sticky="e", pady=(4, 0))

    def run() -> None:
        first_csv = first_var.get().strip()
        second_csv = second_var.get().strip()
        output_folder = out_var.get().strip()
        if not first_csv or not second_csv:
            messagebox.showerror("Missing Input", "Please select both input CSV files.", parent=dialog)
            return
        if not output_folder:
            messagebox.showerror("Missing Output", "Please select an output folder.", parent=dialog)
            return

        try:
            _launch_script(
                "data/compute_column10_errors.py",
                [first_csv, second_csv, "--output-folder", output_folder],
            )
        except Exception as exc:  # noqa: BLE001 - propagate any launcher failure to the GUI error dialog.
            messagebox.showerror("Launch Error", f"Unable to run script:\n{exc}", parent=dialog)
            return
        dialog.destroy()

    tk.Button(buttons, text="Cancel", width=12, command=dialog.destroy).pack(side="left", padx=(0, 8))
    tk.Button(buttons, text="Run", width=12, command=run).pack(side="left")


def _show_histogram(parent: tk.Tk) -> None:
    dialog, frame = _open_dialog(parent, "Plot Histogram")

    tk.Label(frame, text="Input CSV files (multiple):").grid(row=0, column=0, sticky="w")
    files_preview = tk.Label(frame, text="No files selected", anchor="w", justify="left", wraplength=500)
    files_preview.grid(row=1, column=0, sticky="we", padx=(0, 8), pady=(4, 10))

    selected_files: list[str] = []

    def browse() -> None:
        paths = filedialog.askopenfilenames(
            title="Input CSV files",
            initialdir=str(RAW_DIR if RAW_DIR.exists() else PROJECT_ROOT),
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
        if not paths:
            return
        selected_files.clear()
        selected_files.extend(paths)
        # Keep preview compact even for large multi-file selections.
        files_preview.config(text=f"{len(selected_files)} file(s) selected")

    tk.Button(frame, text="Browse...", command=browse, width=12).grid(row=1, column=1, sticky="e")

    out_var = _add_output_folder_input(frame, 2, "Output folder location:", default=str(PROCESSED_DIR / "plots"))

    buttons = tk.Frame(frame)
    buttons.grid(row=4, column=0, columnspan=2, sticky="e", pady=(4, 0))

    def run() -> None:
        if not selected_files:
            messagebox.showerror("Missing Input", "Please select at least one input CSV file.", parent=dialog)
            return
        output_folder = out_var.get().strip()
        if not output_folder:
            messagebox.showerror("Missing Output", "Please select an output folder.", parent=dialog)
            return

        try:
            _launch_script("data/plot_tof_histogram.py", [*selected_files, "--save", output_folder])
        except Exception as exc:  # noqa: BLE001 - propagate any launcher failure to the GUI error dialog.
            messagebox.showerror("Launch Error", f"Unable to run script:\n{exc}", parent=dialog)
            return
        dialog.destroy()

    tk.Button(buttons, text="Cancel", width=12, command=dialog.destroy).pack(side="left", padx=(0, 8))
    tk.Button(buttons, text="Run", width=12, command=run).pack(side="left")


def _show_heatmap(parent: tk.Tk) -> None:
    dialog, frame = _open_dialog(parent, "Plot Heatmap")

    tk.Label(frame, text="Input CSV files (multiple):").grid(row=0, column=0, sticky="w")
    files_preview = tk.Label(frame, text="No files selected", anchor="w", justify="left", wraplength=500)
    files_preview.grid(row=1, column=0, sticky="we", padx=(0, 8), pady=(4, 10))

    selected_files: list[str] = []

    def browse() -> None:
        paths = filedialog.askopenfilenames(
            title="Input CSV files",
            initialdir=str(RAW_DIR if RAW_DIR.exists() else PROJECT_ROOT),
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
        if not paths:
            return
        selected_files.clear()
        selected_files.extend(paths)
        files_preview.config(text=f"{len(selected_files)} file(s) selected")

    tk.Button(frame, text="Browse...", command=browse, width=12).grid(row=1, column=1, sticky="e")

    out_var = _add_output_folder_input(frame, 2, "Output folder location:", default=str(PROCESSED_DIR / "plots"))

    tk.Label(frame, text="Plot title prefix:").grid(row=4, column=0, sticky="w")
    title_var = tk.StringVar(value="ToF Heatmap")
    tk.Entry(frame, textvariable=title_var, width=62).grid(row=5, column=0, sticky="we", padx=(0, 8), pady=(4, 10))

    tk.Label(frame, text="Color bar label:").grid(row=6, column=0, sticky="w")
    cbar_var = tk.StringVar(value="ToF (s)")
    tk.Entry(frame, textvariable=cbar_var, width=62).grid(row=7, column=0, sticky="we", padx=(0, 8), pady=(4, 10))

    interpolation_var = tk.BooleanVar(value=True)
    tk.Checkbutton(
        frame,
        text="Enable interpolation",
        variable=interpolation_var,
        onvalue=True,
        offvalue=False,
    ).grid(row=8, column=0, columnspan=2, sticky="w", pady=(0, 8))

    buttons = tk.Frame(frame)
    buttons.grid(row=9, column=0, columnspan=2, sticky="e", pady=(4, 0))

    def run() -> None:
        if not selected_files:
            messagebox.showerror("Missing Input", "Please select at least one input CSV file.", parent=dialog)
            return
        output_folder = out_var.get().strip()
        if not output_folder:
            messagebox.showerror("Missing Output", "Please select an output folder.", parent=dialog)
            return

        title_prefix = title_var.get().strip()
        cbar_label = cbar_var.get().strip()

        interpolation = "on" if interpolation_var.get() else "off"
        args = [
            *selected_files,
            "--save",
            output_folder,
            "--interpolation",
            interpolation,
            "--title-prefix",
            title_prefix,
            "--cbar-label",
            cbar_label,
        ]

        try:
            _launch_script("data/plot_tof_heatmap.py", args)
        except Exception as exc:  # noqa: BLE001 - propagate any launcher failure to the GUI error dialog.
            messagebox.showerror("Launch Error", f"Unable to run script:\n{exc}", parent=dialog)
            return
        dialog.destroy()

    tk.Button(buttons, text="Cancel", width=12, command=dialog.destroy).pack(side="left", padx=(0, 8))
    tk.Button(buttons, text="Run", width=12, command=run).pack(side="left")


def _show_thickness_or_speed(parent: tk.Tk) -> None:
    dialog, frame = _open_dialog(parent, "Calculate Thickness/Speed")

    csv_var = _add_single_csv_input(frame, 0, "Input CSV file:")

    tk.Label(frame, text="Mode:").grid(row=2, column=0, sticky="w")
    mode_var = tk.StringVar(value="thickness")
    mode_frame = tk.Frame(frame)
    mode_frame.grid(row=3, column=0, columnspan=2, sticky="w", pady=(4, 10))
    tk.Radiobutton(mode_frame, text="Thickness", value="thickness", variable=mode_var).pack(side="left", padx=(0, 14))
    tk.Radiobutton(mode_frame, text="Speed", value="speed", variable=mode_var).pack(side="left")

    tk.Label(frame, text="Other Parameter Estimate:").grid(row=4, column=0, sticky="w")
    estimate_var = tk.StringVar(value="1.0")
    tk.Entry(frame, textvariable=estimate_var, width=20).grid(row=5, column=0, sticky="w", pady=(4, 10))

    out_var = _add_output_folder_input(frame, 6, "Output folder location:", default=str(PROCESSED_DIR))

    buttons = tk.Frame(frame)
    buttons.grid(row=8, column=0, columnspan=2, sticky="e", pady=(4, 0))

    def run() -> None:
        input_csv = csv_var.get().strip()
        output_folder = out_var.get().strip()
        if not input_csv:
            messagebox.showerror("Missing Input", "Please select an input CSV file.", parent=dialog)
            return
        if not output_folder:
            messagebox.showerror("Missing Output", "Please select an output folder.", parent=dialog)
            return

        try:
            estimate = float(estimate_var.get().strip())
        except ValueError:
            messagebox.showerror("Invalid Input", "Other Parameter Estimate must be numeric.", parent=dialog)
            return

        args = [
            "--mode",
            mode_var.get(),
            "--estimate",
            str(estimate),
            "--input-csv",
            input_csv,
            "--output-folder",
            output_folder,
        ]

        try:
            _launch_script("data/calculate_thickness_or_speed.py", args)
        except Exception as exc:  # noqa: BLE001 - propagate any launcher failure to the GUI error dialog.
            messagebox.showerror("Launch Error", f"Unable to run script:\n{exc}", parent=dialog)
            return
        dialog.destroy()

    tk.Button(buttons, text="Cancel", width=12, command=dialog.destroy).pack(side="left", padx=(0, 8))
    tk.Button(buttons, text="Run", width=12, command=run).pack(side="left")


def main() -> None:
    root = tk.Tk()
    root.title("Data Processing")
    root.resizable(False, False)

    frame = tk.Frame(root, padx=18, pady=16)
    frame.pack(fill="both", expand=True)

    tk.Label(frame, text="Data Processing Actions", font=("Segoe UI", 12, "bold")).pack(anchor="w", pady=(0, 10))
    tk.Button(frame, text="Apply Hilbert Filter", width=34, command=lambda: _show_apply_hilbert(root)).pack(pady=(0, 8))
    tk.Button(
        frame,
        text="Calculate First Peak Time of Flight",
        width=34,
        command=lambda: _show_first_peak_tof(root),
    ).pack(pady=(0, 8))
    tk.Button(frame, text="Calculate Average ToF", width=34, command=lambda: _show_average_tof(root)).pack(pady=(0, 8))
    tk.Button(frame, text="Calculate Errors", width=34, command=lambda: _show_compute_errors(root)).pack(pady=(0, 8))
    tk.Button(frame, text="Plot Histogram", width=34, command=lambda: _show_histogram(root)).pack(pady=(0, 8))
    tk.Button(frame, text="Plot Heatmap", width=34, command=lambda: _show_heatmap(root)).pack(pady=(0, 8))
    tk.Button(
        frame,
        text="Calculate Thickness/Speed",
        width=34,
        command=lambda: _show_thickness_or_speed(root),
    ).pack(pady=(0, 8))
    tk.Button(frame, text="Exit", width=34, command=root.destroy).pack()

    root.mainloop()


if __name__ == "__main__":
    main()
