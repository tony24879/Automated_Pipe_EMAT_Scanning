"""GUI launcher for data-processing scripts."""

from __future__ import annotations

import ast
import re
import shutil
import subprocess
import sys
import tempfile
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


def _build_csv_paths_section(
    parent: tk.Widget,
    input_csv_var: tk.StringVar,
    working_csv_var: tk.StringVar,
    output_csv_var: tk.StringVar,
) -> None:
    section = tk.LabelFrame(parent, text="CSV Paths", padx=12, pady=10)
    section.pack(fill="x", pady=(0, 14))

    tk.Label(section, text="Input CSV file:").grid(row=0, column=0, sticky="w")
    tk.Entry(section, textvariable=input_csv_var, width=62).grid(
        row=1, column=0, sticky="we", padx=(0, 8), pady=(4, 10)
    )

    def browse_input() -> None:
        path = filedialog.askopenfilename(
            title="Input CSV file",
            initialdir=str(RAW_DIR if RAW_DIR.exists() else PROJECT_ROOT),
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
        if path:
            input_csv_var.set(path)

    tk.Button(section, text="Browse...", command=browse_input, width=12).grid(row=1, column=1, sticky="e")

    tk.Label(section, text="Output CSV file (working copy):").grid(row=2, column=0, sticky="w")
    tk.Entry(section, textvariable=output_csv_var, width=62).grid(
        row=3, column=0, sticky="we", padx=(0, 8), pady=(4, 10)
    )

    def browse_output() -> None:
        path = filedialog.asksaveasfilename(
            title="Output CSV file (working copy):",
            initialdir=str(PROCESSED_DIR if PROCESSED_DIR.exists() else PROJECT_ROOT),
            initialfile=Path(output_csv_var.get()).name if output_csv_var.get() else "",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
        if path:
            output_csv_var.set(path)

    tk.Button(section, text="Browse...", command=browse_output, width=12).grid(row=3, column=1, sticky="e")

    output_manually_edited = False

    def _suggest_output_path() -> str:
        input_csv = input_csv_var.get().strip()
        if not input_csv:
            return ""
        return str(PROCESSED_DIR / f"{Path(input_csv).stem}_processed.csv")

    def _handle_input_change(*_: object) -> None:
        nonlocal output_manually_edited
        if output_manually_edited:
            return
        output_csv_var.set(_suggest_output_path())

    def _handle_output_edit(*_: object) -> None:
        nonlocal output_manually_edited
        output_manually_edited = output_csv_var.get().strip() != _suggest_output_path()

    input_csv_var.trace_add("write", _handle_input_change)
    output_csv_var.trace_add("write", _handle_output_edit)

    status_var = tk.StringVar(value="No working CSV set.")
    tk.Label(section, textvariable=status_var, fg="gray", anchor="w", wraplength=520, justify="left").grid(
        row=4, column=0, columnspan=2, sticky="we", pady=(0, 8)
    )

    def set_working_csv() -> None:
        input_csv = input_csv_var.get().strip()
        output_csv = output_csv_var.get().strip()
        if not input_csv:
            messagebox.showerror("Missing Input", "Please select an input CSV file.")
            return
        if not output_csv:
            messagebox.showerror("Missing Output", "Please choose an output CSV file name and location.")
            return

        input_path = Path(input_csv)
        if not input_path.exists():
            messagebox.showerror("Input Not Found", f"Input CSV file does not exist:\n{input_path}")
            return

        output_path = Path(output_csv)
        if output_path.exists():
            overwrite = messagebox.askyesno(
                "Overwrite File?", f"Output file already exists:\n{output_path}\n\nOverwrite it?"
            )
            if not overwrite:
                return

        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(input_path, output_path)
        except OSError as exc:
            messagebox.showerror("Copy Error", f"Unable to create working CSV:\n{exc}")
            return

        working_csv_var.set(str(output_path))
        status_var.set(f"Working CSV: {output_path}")

    tk.Button(section, text="Set Working CSV", width=18, command=set_working_csv).grid(
        row=5, column=0, sticky="w"
    )


def _build_filters_section(
    parent: tk.Widget,
    input_csv_var: tk.StringVar,
    working_csv_var: tk.StringVar,
    output_csv_var: tk.StringVar,
) -> None:
    section = tk.LabelFrame(parent, text="Filters", padx=12, pady=10)
    section.pack(fill="x", pady=(0, 14))

    tk.Label(section, text="Filter mode:").grid(row=0, column=0, sticky="w")
    filter_var = tk.StringVar(value="none")
    tk.OptionMenu(section, filter_var, "none", "hilbert").grid(row=0, column=1, sticky="w", padx=(8, 0))

    def apply_filter() -> None:
        input_csv = input_csv_var.get().strip()
        if not input_csv:
            messagebox.showerror("Missing Input", "Please select an input CSV in the CSV Paths section first.")
            return

        output_csv = working_csv_var.get().strip() or output_csv_var.get().strip()
        if not output_csv:
            output_csv = str(PROCESSED_DIR / f"{Path(input_csv).stem}_processed.csv")
            output_csv_var.set(output_csv)

        mode = filter_var.get().strip().lower()
        try:
            _launch_script(
                "data/apply_hilbert_to_raw_scan.py",
                [input_csv, "--mode", mode, "--output-file", output_csv],
            )
        except Exception as exc:  # noqa: BLE001 - propagate any launcher failure to the GUI error dialog.
            messagebox.showerror("Launch Error", f"Unable to run script:\n{exc}")
            return

        working_csv_var.set(output_csv)
        output_csv_var.set(output_csv)

    tk.Button(section, text="Apply Filter", width=16, command=apply_filter).grid(row=1, column=0, sticky="w", pady=(10, 0))


def _build_signal_plotting_section(parent: tk.Widget, working_csv_var: tk.StringVar) -> None:
    section = tk.LabelFrame(parent, text="Signal Plotting", padx=12, pady=10)
    section.pack(fill="x", pady=(0, 14))

    def _require_working_csv() -> str | None:
        working_csv = working_csv_var.get().strip()
        if not working_csv:
            messagebox.showerror("No Working CSV", "Please set a working CSV in the CSV Paths section first.")
            return None
        return working_csv

    output_dir_var = tk.StringVar(value=str(PROCESSED_DIR / "plots"))

    def browse_output_dir() -> None:
        folder = filedialog.askdirectory(
            title="Select output folder",
            initialdir=str(PROCESSED_DIR / "plots"),
        )
        if folder:
            output_dir_var.set(folder)

    tk.Label(section, text="Output folder (blank = no save):").grid(row=0, column=0, sticky="w")
    tk.Entry(section, textvariable=output_dir_var, width=48).grid(row=1, column=0, sticky="we", padx=(0, 8), pady=(4, 8))
    tk.Button(section, text="Browse...", command=browse_output_dir, width=12).grid(row=1, column=1, sticky="e")

    tk.Label(section, text="Row number:").grid(row=2, column=0, sticky="w")
    row_var = tk.StringVar()
    tk.Entry(section, textvariable=row_var, width=10).grid(row=2, column=1, sticky="w", padx=(6, 8))

    def plot_row() -> None:
        working_csv = _require_working_csv()
        if working_csv is None:
            return

        row_text = row_var.get().strip()
        if not row_text.lstrip("-").isdigit() or int(row_text) < 1:
            messagebox.showerror("Invalid Input", "Row number must be a positive integer.")
            return

        output_dir = output_dir_var.get().strip()
        plot_args = [working_csv, "--row", row_text, "--peaks", "on" if show_peaks_var.get() else "off"]
        if output_dir:
            stem = Path(working_csv).stem
            save_path = str(Path(output_dir) / f"{stem}_plot_row_{row_text}.png")
            plot_args.extend(["--save", save_path])

        try:
            _launch_script("data/plot_row_peaks.py", plot_args)
        except Exception as exc:  # noqa: BLE001 - propagate any launcher failure to the GUI error dialog.
            messagebox.showerror("Launch Error", f"Unable to run script:\n{exc}")

    tk.Button(section, text="Plot", width=10, command=plot_row).grid(row=2, column=2, sticky="w")

    tk.Label(section, text="Start row:").grid(row=3, column=0, sticky="w", pady=(12, 0))
    start_row_var = tk.StringVar()
    tk.Entry(section, textvariable=start_row_var, width=10).grid(row=3, column=1, sticky="w", padx=(6, 8), pady=(12, 0))

    tk.Label(section, text="End row:").grid(row=4, column=0, sticky="w")
    end_row_var = tk.StringVar()
    tk.Entry(section, textvariable=end_row_var, width=10).grid(row=4, column=1, sticky="w", padx=(6, 8))

    speed_var = tk.StringVar(value="120")
    tk.Label(section, text="Animation speed (ms):").grid(row=5, column=0, sticky="w", pady=(10, 0))
    tk.Entry(section, textvariable=speed_var, width=10).grid(row=5, column=1, sticky="w", padx=(6, 8), pady=(10, 0))

    show_peaks_var = tk.BooleanVar(value=True)
    tk.Checkbutton(section, text="Show peak markers", variable=show_peaks_var).grid(
        row=6, column=0, columnspan=2, sticky="w", pady=(10, 0)
    )

    def animate_rows() -> None:
        working_csv = _require_working_csv()
        if working_csv is None:
            return

        start_text = start_row_var.get().strip()
        end_text = end_row_var.get().strip()
        if not start_text.isdigit() or not end_text.isdigit():
            messagebox.showerror("Invalid Input", "Start and end row must be positive integers.")
            return
        if int(end_text) < int(start_text):
            messagebox.showerror("Invalid Input", "End row must be greater than or equal to start row.")
            return

        speed_text = speed_var.get().strip()
        try:
            speed_ms = int(speed_text)
        except ValueError:
            messagebox.showerror("Invalid Input", "Animation speed must be an integer number of milliseconds.")
            return
        if speed_ms <= 0:
            messagebox.showerror("Invalid Input", "Animation speed must be greater than zero.")
            return

        output_dir = output_dir_var.get().strip()
        args = [
            working_csv,
            "--start-row",
            start_text,
            "--end-row",
            end_text,
            "--interval-ms",
            str(speed_ms),
            "--peaks",
            "on" if show_peaks_var.get() else "off",
        ]
        if output_dir:
            stem = Path(working_csv).stem
            save_path = str(Path(output_dir) / f"{stem}_animate_rows_{start_text}_{end_text}.gif")
            args.extend(["--save", save_path])

        try:
            _launch_script("data/animate_row_scroll.py", args)
        except Exception as exc:  # noqa: BLE001 - propagate any launcher failure to the GUI error dialog.
            messagebox.showerror("Launch Error", f"Unable to run script:\n{exc}")

    tk.Button(section, text="Animate", width=10, command=animate_rows).grid(row=4, column=2, sticky="w")


def _build_plots_section(parent: tk.Widget, working_csv_var: tk.StringVar) -> None:
    section = tk.LabelFrame(parent, text="Plots", padx=12, pady=10)
    section.pack(fill="x", pady=(0, 14))

    def _require_working_csv() -> str | None:
        working_csv = working_csv_var.get().strip()
        if not working_csv:
            messagebox.showerror("No Working CSV", "Please set a working CSV in the CSV Paths section first.")
            return None
        return working_csv

    def _browse_extra_csvs() -> list[str]:
        paths = filedialog.askopenfilenames(
            title="Additional CSV files",
            initialdir=str(RAW_DIR if RAW_DIR.exists() else PROJECT_ROOT),
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
        return list(paths)

    def _browse_output_file(var: tk.StringVar, working_csv_var: tk.StringVar, suffix: str) -> None:
        output_dir = PROCESSED_DIR / "plots"
        output_dir.mkdir(parents=True, exist_ok=True)
        folder = filedialog.askdirectory(
            title="Select output folder",
            initialdir=str(output_dir),
        )
        if not folder:
            return

        working_csv = working_csv_var.get().strip()
        stem = Path(working_csv).stem if working_csv else "plot"
        var.set(str(Path(folder) / f"{stem}{suffix}.png"))

    def _plot_output_default(working_csv: str, suffix: str) -> str:
        if not working_csv:
            return ""
        stem = Path(working_csv).stem
        return str(PROCESSED_DIR / "plots" / f"{stem}{suffix}.png")

    def _parse_numeric_list(stdout_text: str) -> list[float]:
        for match in reversed(re.findall(r"\[[^\]]+\]", stdout_text)):
            try:
                parsed = ast.literal_eval(match)
            except (ValueError, SyntaxError):
                continue
            if isinstance(parsed, (list, tuple)):
                return [float(value) for value in parsed]
        return []

    def _write_temp_override_file(values: list[float]) -> str:
        fh = tempfile.NamedTemporaryFile("w", prefix="plot_override_", suffix=".csv", delete=False, encoding="utf-8")
        with fh:
            for value in values:
                fh.write(f"{float(value)}\n")
        return fh.name

    def _build_plot_value_override(mode: str, working_csv: str) -> str | None:
        if mode == "ToF":
            return None

        if mode in {"Thickness", "Speed"}:
            calc_mode = "thickness" if mode == "Thickness" else "speed"
            estimate_var = speed_estimate_var if mode == "Thickness" else thickness_estimate_var
            try:
                estimate = float(estimate_var.get().strip())
            except ValueError:
                messagebox.showerror("Invalid Input", f"{ 'Speed Estimate' if mode == 'Thickness' else 'Thickness Estimate' } must be numeric.")
                return ""

            args = [
                sys.executable,
                str(PROJECT_ROOT / "data" / "calculate_thickness_or_speed.py"),
                "--mode",
                calc_mode,
                "--estimate",
                str(estimate),
                "--input-csv",
                working_csv,
            ]
            result = subprocess.run(args, capture_output=True, text=True, cwd=str(PROJECT_ROOT), check=False)
            if result.returncode != 0:
                error_text = (result.stderr or result.stdout or "Calculation failed.").strip()
                messagebox.showerror("Calculation Error", error_text)
                return ""
            values = _parse_numeric_list(result.stdout)
            if not values:
                messagebox.showerror("Calculation Error", "The calculated values list was empty.")
                return ""
            return _write_temp_override_file(values)

        if mode == "Errors":
            second_csv = second_csv_var.get().strip()
            if not second_csv:
                messagebox.showerror("Missing CSV", "Please select the second input CSV for the Errors plot.")
                return ""
            args = [
                sys.executable,
                str(PROJECT_ROOT / "data" / "compute_column10_errors.py"),
                working_csv,
                second_csv,
            ]
            result = subprocess.run(args, capture_output=True, text=True, cwd=str(PROJECT_ROOT), check=False)
            if result.returncode != 0:
                error_text = (result.stderr or result.stdout or "Calculation failed.").strip()
                messagebox.showerror("Calculation Error", error_text)
                return ""
            values = _parse_numeric_list(result.stdout)
            if not values:
                messagebox.showerror("Calculation Error", "The error values list was empty.")
                return ""
            return _write_temp_override_file(values)

        return None

    plot_kind_var = tk.StringVar(value="histogram")
    tk.Label(section, text="Select plot:").grid(row=0, column=0, sticky="w")
    tk.OptionMenu(section, plot_kind_var, "histogram", "heatmap", "3d", "boxplot").grid(row=0, column=1, sticky="w", padx=(8, 0))

    plot_value_mode_var = tk.StringVar(value="ToF")
    speed_estimate_var = tk.StringVar(value="1500")
    thickness_estimate_var = tk.StringVar(value="0.01")
    second_csv_var = tk.StringVar(value="")

    tk.Label(section, text="Plotted Values:").grid(row=1, column=0, sticky="w", pady=(10, 4))
    tk.OptionMenu(section, plot_value_mode_var, "ToF", "Thickness", "Speed", "Errors").grid(
        row=1, column=1, sticky="w", padx=(8, 0), pady=(10, 4)
    )

    speed_estimate_label = tk.Label(section, text="Speed Estimate (m/s):")
    speed_estimate_entry = tk.Entry(section, textvariable=speed_estimate_var, width=20)
    thickness_estimate_label = tk.Label(section, text="Thickness Estimate (m):")
    thickness_estimate_entry = tk.Entry(section, textvariable=thickness_estimate_var, width=20)
    second_csv_label = tk.Label(section, text="Second Input CSV:")
    second_csv_entry = tk.Entry(section, textvariable=second_csv_var, width=40)
    second_csv_browse = tk.Button(section, text="Browse...", width=12)

    def _toggle_value_inputs(*_: object) -> None:
        kind = plot_value_mode_var.get()
        speed_estimate_label.grid_remove()
        speed_estimate_entry.grid_remove()
        thickness_estimate_label.grid_remove()
        thickness_estimate_entry.grid_remove()
        second_csv_label.grid_remove()
        second_csv_entry.grid_remove()
        second_csv_browse.grid_remove()

        if kind == "Thickness":
            speed_estimate_label.grid(row=2, column=0, sticky="w", pady=(0, 6))
            speed_estimate_entry.grid(row=2, column=1, sticky="w", padx=(8, 0), pady=(0, 6))
        elif kind == "Speed":
            thickness_estimate_label.grid(row=2, column=0, sticky="w", pady=(0, 6))
            thickness_estimate_entry.grid(row=2, column=1, sticky="w", padx=(8, 0), pady=(0, 6))
        elif kind == "Errors":
            second_csv_label.grid(row=2, column=0, sticky="w", pady=(0, 6))
            second_csv_entry.grid(row=2, column=1, sticky="w", padx=(8, 0), pady=(0, 6))
            second_csv_browse.grid(row=2, column=2, sticky="w", padx=(8, 0), pady=(0, 6))

    def browse_second_csv() -> None:
        path = filedialog.askopenfilename(
            title="Second input CSV for error plot",
            initialdir=str(PROCESSED_DIR if PROCESSED_DIR.exists() else PROJECT_ROOT),
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
        if path:
            second_csv_var.set(path)

    second_csv_browse.configure(command=browse_second_csv)
    plot_value_mode_var.trace_add("write", _toggle_value_inputs)
    _toggle_value_inputs()

    panel_container = tk.Frame(section)
    panel_container.grid(row=3, column=0, columnspan=3, sticky="we", pady=(10, 0))

    def _panel_for(kind: str) -> tk.Frame:
        panel = tk.Frame(panel_container, padx=6, pady=4)
        panel.grid_columnconfigure(0, weight=1)
        return panel

    hist_panel = _panel_for("histogram")
    heat_panel = _panel_for("heatmap")
    plot3_panel = _panel_for("3d")
    box_panel = _panel_for("boxplot")
    panels = {"histogram": hist_panel, "heatmap": heat_panel, "3d": plot3_panel, "boxplot": box_panel}

    hist_files_var = tk.StringVar(value="")
    hist_bins_var = tk.StringVar(value="50")
    hist_out_var = tk.StringVar(value="")
    hist_title_var = tk.StringVar(value="Time of Flight Histogram")
    hist_xlabel_var = tk.StringVar(value="Time of Flight (s)")
    hist_group_by_repeat_point_var = tk.BooleanVar(value=False)

    hist_output_manually_edited = False

    def _sync_hist_output(*_: object) -> None:
        nonlocal hist_output_manually_edited
        if hist_output_manually_edited:
            return
        hist_out_var.set(_plot_output_default(working_csv_var.get().strip(), "_histogram"))

    def _mark_hist_output_manual(*_: object) -> None:
        nonlocal hist_output_manually_edited
        hist_output_manually_edited = hist_out_var.get().strip() != _plot_output_default(working_csv_var.get().strip(), "_histogram")

    working_csv_var.trace_add("write", _sync_hist_output)
    hist_out_var.trace_add("write", _mark_hist_output_manual)

    def browse_hist_files() -> None:
        paths = _browse_extra_csvs()
        if not paths:
            return
        hist_files_var.set("; ".join(paths))

    tk.Label(hist_panel, text="Histogram", font=("Segoe UI", 10, "bold")).grid(row=0, column=0, sticky="w")
    tk.Label(hist_panel, text="Additional CSVs:").grid(row=1, column=0, sticky="w")
    tk.Entry(hist_panel, textvariable=hist_files_var, width=56).grid(row=2, column=0, sticky="we", padx=(0, 8), pady=(4, 8))
    tk.Button(hist_panel, text="Browse...", width=12, command=browse_hist_files).grid(row=2, column=1, sticky="e")
    tk.Label(hist_panel, text="Bins:").grid(row=3, column=0, sticky="w")
    tk.Entry(hist_panel, textvariable=hist_bins_var, width=18).grid(row=4, column=0, sticky="w", pady=(4, 8))
    tk.Label(hist_panel, text="Output path (blank = no save):").grid(row=5, column=0, sticky="w")
    tk.Entry(hist_panel, textvariable=hist_out_var, width=56).grid(row=6, column=0, sticky="we", padx=(0, 8), pady=(4, 8))
    tk.Button(hist_panel, text="Browse...", command=lambda: _browse_output_file(hist_out_var, working_csv_var, "_histogram"), width=12).grid(row=6, column=1, sticky="e")
    tk.Label(hist_panel, text="Title:").grid(row=7, column=0, sticky="w")
    tk.Entry(hist_panel, textvariable=hist_title_var, width=56).grid(row=8, column=0, sticky="we", padx=(0, 8), pady=(4, 8))
    tk.Label(hist_panel, text="X-axis label:").grid(row=9, column=0, sticky="w")
    tk.Entry(hist_panel, textvariable=hist_xlabel_var, width=56).grid(row=10, column=0, sticky="we", padx=(0, 8), pady=(4, 8))
    tk.Checkbutton(
        hist_panel,
        text="Group by repeated point (new group when col 8/9 changes)",
        variable=hist_group_by_repeat_point_var,
    ).grid(row=11, column=0, columnspan=2, sticky="w", pady=(0, 8))

    def plot_histogram() -> None:
        working_csv = _require_working_csv()
        if working_csv is None:
            return
        csv_paths = [working_csv]
        extra_text = hist_files_var.get().strip()
        if extra_text:
            csv_paths.extend(part.strip() for part in extra_text.split(";") if part.strip())
        try:
            bins = int(hist_bins_var.get().strip())
        except ValueError:
            messagebox.showerror("Invalid Input", "Number of bins must be an integer.")
            return
        if bins < 1:
            messagebox.showerror("Invalid Input", "Number of bins must be at least 1.")
            return

        override_path = _build_plot_value_override(plot_value_mode_var.get(), working_csv)
        if override_path == "":
            return

        output_path = hist_out_var.get().strip()
        output_arg: list[str] = []
        if output_path:
            output_arg = ["--save", output_path]

        title = hist_title_var.get().strip() or "Time of Flight Histogram"
        x_label = hist_xlabel_var.get().strip() or "Time of Flight (s)"
        args = [*csv_paths, "--bins", str(bins), *output_arg, "--title", title, "--x-label", x_label]
        if hist_group_by_repeat_point_var.get():
            args.append("--group-by-repeat-point")
        if override_path:
            args.extend(["--override-tof-file", override_path])
        try:
            _launch_script("data/plot_tof_histogram.py", args)
        except Exception as exc:  # noqa: BLE001 - propagate any launcher failure to the GUI error dialog.
            messagebox.showerror("Launch Error", f"Unable to run script:\n{exc}")

    tk.Button(hist_panel, text="Plot", width=12, command=plot_histogram).grid(row=12, column=0, sticky="w", pady=(8, 10))

    box_files_var = tk.StringVar(value="")
    box_out_var = tk.StringVar(value="")
    box_title_var = tk.StringVar(value="Time of Flight Box Plot")
    box_ylabel_var = tk.StringVar(value="Time of Flight (s)")
    box_group_by_repeat_point_var = tk.BooleanVar(value=False)

    box_output_manually_edited = False

    def _sync_box_output(*_: object) -> None:
        nonlocal box_output_manually_edited
        if box_output_manually_edited:
            return
        box_out_var.set(_plot_output_default(working_csv_var.get().strip(), "_boxplot"))

    def _mark_box_output_manual(*_: object) -> None:
        nonlocal box_output_manually_edited
        box_output_manually_edited = box_out_var.get().strip() != _plot_output_default(working_csv_var.get().strip(), "_boxplot")

    working_csv_var.trace_add("write", _sync_box_output)
    box_out_var.trace_add("write", _mark_box_output_manual)

    def browse_box_files() -> None:
        paths = _browse_extra_csvs()
        if not paths:
            return
        box_files_var.set("; ".join(paths))

    tk.Label(box_panel, text="Box Plot", font=("Segoe UI", 10, "bold")).grid(row=0, column=0, sticky="w")
    tk.Label(box_panel, text="Additional CSVs:").grid(row=1, column=0, sticky="w")
    tk.Entry(box_panel, textvariable=box_files_var, width=56).grid(row=2, column=0, sticky="we", padx=(0, 8), pady=(4, 8))
    tk.Button(box_panel, text="Browse...", width=12, command=browse_box_files).grid(row=2, column=1, sticky="e")
    tk.Label(box_panel, text="Output path (blank = no save):").grid(row=3, column=0, sticky="w")
    tk.Entry(box_panel, textvariable=box_out_var, width=56).grid(row=4, column=0, sticky="we", padx=(0, 8), pady=(4, 8))
    tk.Button(box_panel, text="Browse...", command=lambda: _browse_output_file(box_out_var, working_csv_var, "_boxplot"), width=12).grid(row=4, column=1, sticky="e")
    tk.Label(box_panel, text="Title:").grid(row=5, column=0, sticky="w")
    tk.Entry(box_panel, textvariable=box_title_var, width=56).grid(row=6, column=0, sticky="we", padx=(0, 8), pady=(4, 8))
    tk.Label(box_panel, text="Y-axis label:").grid(row=7, column=0, sticky="w")
    tk.Entry(box_panel, textvariable=box_ylabel_var, width=56).grid(row=8, column=0, sticky="we", padx=(0, 8), pady=(4, 8))
    tk.Checkbutton(
        box_panel,
        text="Group by repeated point (new group when col 8/9 changes)",
        variable=box_group_by_repeat_point_var,
    ).grid(row=9, column=0, columnspan=2, sticky="w", pady=(0, 8))

    def plot_boxplot() -> None:
        working_csv = _require_working_csv()
        if working_csv is None:
            return
        csv_paths = [working_csv]
        extra_text = box_files_var.get().strip()
        if extra_text:
            csv_paths.extend(part.strip() for part in extra_text.split(";") if part.strip())

        override_path = _build_plot_value_override(plot_value_mode_var.get(), working_csv)
        if override_path == "":
            return

        output_path = box_out_var.get().strip()
        output_arg: list[str] = []
        if output_path:
            output_arg = ["--save", output_path]

        title = box_title_var.get().strip() or "Time of Flight Box Plot"
        y_label = box_ylabel_var.get().strip() or "Time of Flight (s)"
        args = [*csv_paths, *output_arg, "--title", title, "--y-label", y_label]
        if box_group_by_repeat_point_var.get():
            args.append("--group-by-repeat-point")
        if override_path:
            args.extend(["--override-tof-file", override_path])
        try:
            _launch_script("data/plot_tof_boxplot.py", args)
        except Exception as exc:  # noqa: BLE001 - propagate any launcher failure to the GUI error dialog.
            messagebox.showerror("Launch Error", f"Unable to run script:\n{exc}")

    tk.Button(box_panel, text="Plot", width=12, command=plot_boxplot).grid(row=10, column=0, sticky="w", pady=(8, 10))

    heat_files_var = tk.StringVar(value="")
    heat_interpolation_var = tk.BooleanVar(value=True)
    heat_out_var = tk.StringVar(value="")
    heat_title_var = tk.StringVar(value="ToF Heatmap")
    heat_cbar_var = tk.StringVar(value="ToF (s)")

    heat_output_manually_edited = False

    def _sync_heat_output(*_: object) -> None:
        nonlocal heat_output_manually_edited
        if heat_output_manually_edited:
            return
        heat_out_var.set(_plot_output_default(working_csv_var.get().strip(), "_heatmap"))

    def _mark_heat_output_manual(*_: object) -> None:
        nonlocal heat_output_manually_edited
        heat_output_manually_edited = heat_out_var.get().strip() != _plot_output_default(working_csv_var.get().strip(), "_heatmap")

    working_csv_var.trace_add("write", _sync_heat_output)
    heat_out_var.trace_add("write", _mark_heat_output_manual)

    def browse_heat_files() -> None:
        paths = _browse_extra_csvs()
        if not paths:
            return
        heat_files_var.set("; ".join(paths))

    tk.Label(heat_panel, text="Heatmap", font=("Segoe UI", 10, "bold")).grid(row=0, column=0, sticky="w")
    tk.Label(heat_panel, text="Additional CSVs:").grid(row=1, column=0, sticky="w")
    tk.Entry(heat_panel, textvariable=heat_files_var, width=56).grid(row=2, column=0, sticky="we", padx=(0, 8), pady=(4, 8))
    tk.Button(heat_panel, text="Browse...", width=12, command=browse_heat_files).grid(row=2, column=1, sticky="e")
    tk.Checkbutton(heat_panel, text="Interpolation enabled", variable=heat_interpolation_var).grid(row=3, column=0, sticky="w", pady=(0, 8))
    tk.Label(heat_panel, text="Output path (blank = no save):").grid(row=4, column=0, sticky="w")
    tk.Entry(heat_panel, textvariable=heat_out_var, width=56).grid(row=5, column=0, sticky="we", padx=(0, 8), pady=(4, 8))
    tk.Button(heat_panel, text="Browse...", command=lambda: _browse_output_file(heat_out_var, working_csv_var, "_heatmap"), width=12).grid(row=5, column=1, sticky="e")
    tk.Label(heat_panel, text="Title:").grid(row=6, column=0, sticky="w")
    tk.Entry(heat_panel, textvariable=heat_title_var, width=56).grid(row=7, column=0, sticky="we", padx=(0, 8), pady=(4, 8))
    tk.Label(heat_panel, text="Color bar / z-axis label:").grid(row=8, column=0, sticky="w")
    tk.Entry(heat_panel, textvariable=heat_cbar_var, width=56).grid(row=9, column=0, sticky="we", padx=(0, 8), pady=(4, 8))

    def plot_heatmap() -> None:
        working_csv = _require_working_csv()
        if working_csv is None:
            return
        csv_paths = [working_csv]
        extra_text = heat_files_var.get().strip()
        if extra_text:
            csv_paths.extend(part.strip() for part in extra_text.split(";") if part.strip())

        override_path = _build_plot_value_override(plot_value_mode_var.get(), working_csv)
        if override_path == "":
            return

        output_path = heat_out_var.get().strip()
        output_arg: list[str] = []
        if output_path:
            output_arg = ["--save", output_path]

        args = [
            *csv_paths,
            "--interpolation",
            "on" if heat_interpolation_var.get() else "off",
            *output_arg,
            "--title-prefix",
            heat_title_var.get().strip() or "ToF Heatmap",
            "--cbar-label",
            heat_cbar_var.get().strip() or "ToF (s)",
        ]
        if override_path:
            args.extend(["--override-tof-file", override_path])
        try:
            _launch_script("data/plot_tof_heatmap.py", args)
        except Exception as exc:  # noqa: BLE001 - propagate any launcher failure to the GUI error dialog.
            messagebox.showerror("Launch Error", f"Unable to run script:\n{exc}")

    tk.Button(heat_panel, text="Plot", width=12, command=plot_heatmap).grid(row=10, column=0, sticky="w", pady=(8, 10))

    plot3_files_var = tk.StringVar(value="")
    plot3_out_var = tk.StringVar(value="")
    plot3_title_var = tk.StringVar(value="ToF 3D Plot")
    plot3_cbar_var = tk.StringVar(value="Time of Flight (s)")
    plot3_type_var = tk.StringVar(value="scatter")

    plot3_output_manually_edited = False

    def _sync_plot3_output(*_: object) -> None:
        nonlocal plot3_output_manually_edited
        if plot3_output_manually_edited:
            return
        plot3_out_var.set(_plot_output_default(working_csv_var.get().strip(), "_3d"))

    def _mark_plot3_output_manual(*_: object) -> None:
        nonlocal plot3_output_manually_edited
        plot3_output_manually_edited = plot3_out_var.get().strip() != _plot_output_default(working_csv_var.get().strip(), "_3d")

    working_csv_var.trace_add("write", _sync_plot3_output)
    plot3_out_var.trace_add("write", _mark_plot3_output_manual)

    def browse_plot3_files() -> None:
        paths = _browse_extra_csvs()
        if not paths:
            return
        plot3_files_var.set("; ".join(paths))

    tk.Label(plot3_panel, text="3D Plot", font=("Segoe UI", 10, "bold")).grid(row=0, column=0, sticky="w")
    tk.Label(plot3_panel, text="Additional CSVs:").grid(row=1, column=0, sticky="w")
    tk.Entry(plot3_panel, textvariable=plot3_files_var, width=56).grid(row=2, column=0, sticky="we", padx=(0, 8), pady=(4, 8))
    tk.Button(plot3_panel, text="Browse...", width=12, command=browse_plot3_files).grid(row=2, column=1, sticky="e")
    tk.Label(plot3_panel, text="Plot type:").grid(row=3, column=0, sticky="w")
    tk.OptionMenu(plot3_panel, plot3_type_var, "scatter", "surface").grid(row=3, column=1, sticky="w", padx=(8, 0))
    tk.Label(plot3_panel, text="Output path (blank = no save):").grid(row=5, column=0, sticky="w")
    tk.Entry(plot3_panel, textvariable=plot3_out_var, width=56).grid(row=6, column=0, sticky="we", padx=(0, 8), pady=(4, 8))
    tk.Button(plot3_panel, text="Browse...", command=lambda: _browse_output_file(plot3_out_var, working_csv_var, "_3d"), width=12).grid(row=6, column=1, sticky="e")
    tk.Label(plot3_panel, text="Title:").grid(row=7, column=0, sticky="w")
    tk.Entry(plot3_panel, textvariable=plot3_title_var, width=56).grid(row=8, column=0, sticky="we", padx=(0, 8), pady=(4, 8))
    tk.Label(plot3_panel, text="Color bar / z-axis label:").grid(row=9, column=0, sticky="w")
    tk.Entry(plot3_panel, textvariable=plot3_cbar_var, width=56).grid(row=10, column=0, sticky="we", padx=(0, 8), pady=(4, 8))

    def plot_3d() -> None:
        working_csv = _require_working_csv()
        if working_csv is None:
            return
        csv_paths = [working_csv]
        extra_text = plot3_files_var.get().strip()
        if extra_text:
            csv_paths.extend(part.strip() for part in extra_text.split(";") if part.strip())

        override_path = _build_plot_value_override(plot_value_mode_var.get(), working_csv)
        if override_path == "":
            return

        output_path = plot3_out_var.get().strip()
        output_arg: list[str] = []
        if output_path:
            output_arg = ["--save", output_path]

        args = [
            *csv_paths,
            "--plot-type",
            plot3_type_var.get(),
            *output_arg,
            "--title-prefix",
            plot3_title_var.get().strip() or "ToF 3D Plot",
            "--cbar-label",
            plot3_cbar_var.get().strip() or "Time of Flight (s)",
            "--z-label",
            plot3_cbar_var.get().strip() or "Time of Flight (s)",
        ]
        if override_path:
            args.extend(["--override-tof-file", override_path])
        try:
            _launch_script("data/plot_tof_3d.py", args)
        except Exception as exc:  # noqa: BLE001 - propagate any launcher failure to the GUI error dialog.
            messagebox.showerror("Launch Error", f"Unable to run script:\n{exc}")

    tk.Button(plot3_panel, text="Plot", width=12, command=plot_3d).grid(row=11, column=0, sticky="w", pady=(8, 10))

    def show_panel(kind: str) -> None:
        for name, panel in panels.items():
            if name == kind:
                panel.grid(row=0, column=0, sticky="we")
            else:
                panel.grid_remove()

    show_panel(plot_kind_var.get())
    plot_kind_var.trace_add("write", lambda *_: show_panel(plot_kind_var.get()))


def _build_tof_calculation_methods_section(parent: tk.Widget, working_csv_var: tk.StringVar) -> None:
    section = tk.LabelFrame(parent, text="ToF Calculation Methods", padx=12, pady=10)
    section.pack(fill="x", pady=(0, 14))

    def _require_working_csv() -> str | None:
        working_csv = working_csv_var.get().strip()
        if not working_csv:
            messagebox.showerror("No Working CSV", "Please set a working CSV in the CSV Paths section first.")
            return None
        return working_csv

    def _set_status(method_name: str, working_csv: str) -> None:
        status_var.set(f"Last run: {method_name} on {Path(working_csv).name}")

    method_var = tk.StringVar(value="Two Echoes")
    tk.Label(section, text="Method:").grid(row=0, column=0, sticky="w")
    tk.OptionMenu(section, method_var, "Two Echoes", "One Echo", "Multiple Echoes").grid(
        row=0, column=1, sticky="w", padx=(8, 0)
    )

    status_var = tk.StringVar(value="No calculation run yet.")
    tk.Label(section, textvariable=status_var, fg="gray", anchor="w", justify="left", wraplength=320).grid(
        row=1, column=0, columnspan=2, sticky="we", pady=(10, 0)
    )

    panel_container = tk.Frame(section)
    panel_container.grid(row=2, column=0, columnspan=2, sticky="we", pady=(10, 0))

    def _panel() -> tk.Frame:
        panel = tk.Frame(panel_container, padx=6, pady=4)
        panel.grid_columnconfigure(0, weight=1)
        return panel

    two_echo_panel = _panel()
    one_echo_panel = _panel()
    multi_echo_panel = _panel()
    panels = {
        "Two Echoes": two_echo_panel,
        "One Echo": one_echo_panel,
        "Multiple Echoes": multi_echo_panel,
    }

    tk.Label(two_echo_panel, text="Two Echoes", font=("Segoe UI", 10, "bold")).grid(row=0, column=0, sticky="w")

    def calculate_two_echoes() -> None:
        working_csv = _require_working_csv()
        if working_csv is None:
            return

        _set_status("Two Echoes", working_csv)
        try:
            _launch_script("data/recalculate_tof_column10.py", [working_csv])
        except Exception as exc:  # noqa: BLE001 - propagate any launcher failure to the GUI error dialog.
            messagebox.showerror("Launch Error", f"Unable to run script:\n{exc}")

    tk.Button(two_echo_panel, text="Calculate", width=12, command=calculate_two_echoes).grid(
        row=1, column=0, sticky="w", pady=(8, 10)
    )

    tk.Label(one_echo_panel, text="One Echo", font=("Segoe UI", 10, "bold")).grid(row=0, column=0, sticky="w")

    one_echo_delay_var = tk.StringVar(
        value=str(
            _read_peak_setting_from_script(
                PROJECT_ROOT / "data" / "calculate_first_peak_tof.py",
                "PEAK_SAMPLE_OFFSET",
                default=34,
            )
        )
    )
    tk.Label(one_echo_panel, text="Index Zero Delay:").grid(row=1, column=0, sticky="w")
    tk.Entry(one_echo_panel, textvariable=one_echo_delay_var, width=12).grid(
        row=1, column=1, sticky="w", padx=(8, 0), pady=(4, 8)
    )

    def calculate_one_echo() -> None:
        working_csv = _require_working_csv()
        if working_csv is None:
            return

        try:
            peak_sample_offset = int(one_echo_delay_var.get().strip())
            if peak_sample_offset < 0:
                raise ValueError("Index Zero Delay must be non-negative.")
            _update_script_constant_in_file(
                PROJECT_ROOT / "data" / "calculate_first_peak_tof.py",
                "PEAK_SAMPLE_OFFSET",
                peak_sample_offset,
            )
        except ValueError as exc:
            messagebox.showerror("Invalid Input", str(exc))
            return
        except Exception as exc:  # noqa: BLE001 - propagate any launcher failure to the GUI error dialog.
            messagebox.showerror("Update Error", f"Unable to update PEAK_SAMPLE_OFFSET:\n{exc}")
            return

        _set_status("One Echo", working_csv)
        try:
            _launch_script("data/calculate_first_peak_tof.py", [working_csv])
        except Exception as exc:  # noqa: BLE001 - propagate any launcher failure to the GUI error dialog.
            messagebox.showerror("Launch Error", f"Unable to run script:\n{exc}")

    tk.Button(one_echo_panel, text="Calculate", width=12, command=calculate_one_echo).grid(
        row=2, column=0, sticky="w", pady=(8, 10)
    )

    tk.Label(multi_echo_panel, text="Multiple Echoes", font=("Segoe UI", 10, "bold")).grid(row=0, column=0, sticky="w")
    tk.Label(multi_echo_panel, text="Number of Echoes:").grid(row=1, column=0, sticky="w")
    num_echoes_var = tk.StringVar(value="3")
    tk.Entry(multi_echo_panel, textvariable=num_echoes_var, width=12).grid(row=2, column=0, sticky="w", pady=(4, 8))

    def calculate_multiple_echoes() -> None:
        working_csv = _require_working_csv()
        if working_csv is None:
            return

        try:
            num_echoes = int(num_echoes_var.get().strip())
        except ValueError:
            messagebox.showerror("Invalid Input", "Number of Echoes must be an integer.")
            return

        if num_echoes < 2:
            messagebox.showerror("Invalid Input", "Number of Echoes must be at least 2.")
            return

        _set_status("Multiple Echoes", working_csv)
        try:
            _launch_script(
                "data/calculate_average_pairwise_tof.py",
                [working_csv, "--num-peaks", str(num_echoes)],
            )
        except Exception as exc:  # noqa: BLE001 - propagate any launcher failure to the GUI error dialog.
            messagebox.showerror("Launch Error", f"Unable to run script:\n{exc}")

    tk.Button(multi_echo_panel, text="Calculate", width=12, command=calculate_multiple_echoes).grid(
        row=3, column=0, sticky="w", pady=(4, 10)
    )

    def show_panel(kind: str) -> None:
        for name, panel in panels.items():
            if name == kind:
                panel.grid(row=0, column=0, sticky="we")
            else:
                panel.grid_remove()

    show_panel(method_var.get())
    method_var.trace_add("write", lambda *_: show_panel(method_var.get()))


def _read_peak_setting_from_script(script_path: Path, setting_name: str, default: int | None = None) -> int:
    text = script_path.read_text(encoding="utf-8")
    match = re.search(
        rf"^\s*{re.escape(setting_name)}\s*(?::\s*[^=\n]+)?\s*=\s*([-+]?\d+(?:\.\d+)?)\b",
        text,
        flags=re.MULTILINE,
    )
    if match is None:
        if default is not None:
            return default
        raise ValueError(f"Could not find {setting_name} in {script_path}")
    return int(float(match.group(1)))


def _update_script_constant_in_file(script_path: Path, setting_name: str, value: int) -> None:
    text = script_path.read_text(encoding="utf-8")
    lines = text.splitlines(keepends=True)
    pattern = re.compile(rf"^\s*{re.escape(setting_name)}\s*(?::\s*[^=\n]+)?\s*=\s*[-+]?\d+(?:\.\d+)?\b")
    new_lines: list[str] = []
    seen = False

    for line in lines:
        if pattern.match(line):
            if not seen:
                new_lines.append(f"{setting_name} = {value}\n")
                seen = True
            continue
        new_lines.append(line)

    if not seen:
        anchor_pattern = re.compile(r"^TOF_SCALE_SECONDS\s*=\s*.*$", re.MULTILINE)
        anchor_match = anchor_pattern.search(text)
        if anchor_match is None:
            raise ValueError(f"Could not find a safe insertion point for {setting_name} in {script_path}")
        insert_at = anchor_match.end()
        updated = text[:insert_at] + "\n" + f"{setting_name} = {value}" + text[insert_at:]
    else:
        updated = "".join(new_lines)

    script_path.write_text(updated, encoding="utf-8")


def _load_peak_settings_defaults() -> dict[str, int]:
    base_path = PROJECT_ROOT / "data" / "recalculate_tof_column10.py"
    return {
        "SKIP_SAMPLES": _read_peak_setting_from_script(base_path, "SKIP_SAMPLES"),
        "MIN_PEAK_DISTANCE": _read_peak_setting_from_script(base_path, "MIN_PEAK_DISTANCE"),
        "MIN_PROMINENCE": _read_peak_setting_from_script(base_path, "MIN_PROMINENCE"),
    }


def _update_peak_settings_in_scripts(values: dict[str, int]) -> None:
    script_paths = [
        PROJECT_ROOT / "data" / "calculate_average_pairwise_tof.py",
        PROJECT_ROOT / "data" / "calculate_first_peak_tof.py",
        PROJECT_ROOT / "data" / "plot_row_peaks.py",
        PROJECT_ROOT / "data" / "recalculate_tof_column10.py",
    ]
    for script_path in script_paths:
        text = script_path.read_text(encoding="utf-8")
        updated = text
        for setting_name, setting_value in values.items():
            pattern = rf"(?m)^\s*{re.escape(setting_name)}\s*(?::\s*[^=\n]+)?\s*=\s*[-+]?\d+(?:\.\d+)?\b"
            updated = re.sub(pattern, f"{setting_name} = {setting_value}", updated)
        if updated != text:
            script_path.write_text(updated, encoding="utf-8")


def _build_peak_finding_variables_section(parent: tk.Widget) -> None:
    section = tk.LabelFrame(parent, text="Peak Finding Variables", padx=12, pady=10)
    section.pack(fill="x", pady=(0, 14))

    defaults = _load_peak_settings_defaults()
    skip_var = tk.StringVar(value=str(defaults["SKIP_SAMPLES"]))
    distance_var = tk.StringVar(value=str(defaults["MIN_PEAK_DISTANCE"]))
    prominence_var = tk.StringVar(value=str(defaults["MIN_PROMINENCE"]))

    status_var = tk.StringVar(value="No peak-setting changes yet.")

    tk.Label(section, text="Skip Samples:").grid(row=0, column=0, sticky="w")
    tk.Entry(section, textvariable=skip_var, width=16).grid(row=0, column=1, sticky="w", padx=(8, 0), pady=(4, 8))

    tk.Label(section, text="Min Peak Distance:").grid(row=1, column=0, sticky="w")
    tk.Entry(section, textvariable=distance_var, width=16).grid(row=1, column=1, sticky="w", padx=(8, 0), pady=(4, 8))

    tk.Label(section, text="Min Prominence:").grid(row=2, column=0, sticky="w")
    tk.Entry(section, textvariable=prominence_var, width=16).grid(row=2, column=1, sticky="w", padx=(8, 0), pady=(4, 8))

    tk.Label(section, textvariable=status_var, fg="gray", anchor="w", justify="left", wraplength=300).grid(
        row=3, column=0, columnspan=2, sticky="we", pady=(4, 10)
    )

    def change_peak_settings() -> None:
        try:
            values = {
                "SKIP_SAMPLES": int(skip_var.get().strip()),
                "MIN_PEAK_DISTANCE": int(distance_var.get().strip()),
                "MIN_PROMINENCE": int(prominence_var.get().strip()),
            }
        except ValueError:
            messagebox.showerror("Invalid Input", "All peak-finding values must be integers.")
            return

        if values["SKIP_SAMPLES"] < 0 or values["MIN_PEAK_DISTANCE"] < 0 or values["MIN_PROMINENCE"] < 0:
            messagebox.showerror("Invalid Input", "Peak-finding values must all be non-negative.")
            return

        try:
            _update_peak_settings_in_scripts(values)
        except Exception as exc:  # noqa: BLE001 - propagate any launcher failure to the GUI error dialog.
            messagebox.showerror("Update Error", f"Unable to update peak-setting constants:\n{exc}")
            return

        status_var.set(
            "Last updated: SKIP_SAMPLES="
            f"{values['SKIP_SAMPLES']}, MIN_PEAK_DISTANCE={values['MIN_PEAK_DISTANCE']}, "
            f"MIN_PROMINENCE={values['MIN_PROMINENCE']}"
        )

    tk.Button(section, text="Change", width=12, command=change_peak_settings).grid(
        row=4, column=0, sticky="w", pady=(0, 4)
    )


def _build_repeatability_errors_section(parent: tk.Widget, working_csv_var: tk.StringVar) -> None:
    section = tk.LabelFrame(parent, text="Repeatability Errors", padx=12, pady=10)
    section.pack(fill="x", pady=(0, 14))

    def _require_working_csv() -> str | None:
        working_csv = working_csv_var.get().strip()
        if not working_csv:
            messagebox.showerror("No Working CSV", "Please set a working CSV in the CSV Paths section first.")
            return None
        return working_csv

    def _resolve_source_estimate() -> float | None:
        source_mode = source_var.get()
        if source_mode == "ToF":
            return None
        if source_mode == "Thickness":
            try:
                return float(speed_estimate_var.get().strip())
            except ValueError as exc:
                raise ValueError("Thickness estimate must be numeric.") from exc
        if source_mode == "Speed":
            try:
                return float(thickness_estimate_var.get().strip())
            except ValueError as exc:
                raise ValueError("Speed estimate must be numeric.") from exc
        return None

    method_var = tk.StringVar(value="range")
    source_var = tk.StringVar(value="ToF")
    speed_estimate_var = tk.StringVar(value="1500")
    thickness_estimate_var = tk.StringVar(value="0.01")
    second_csv_var = tk.StringVar(value="")

    tk.Label(section, text="Method:").grid(row=0, column=0, sticky="w")
    tk.OptionMenu(section, method_var, "range", "std").grid(row=0, column=1, sticky="w", padx=(8, 0))

    tk.Label(section, text="Value Source:").grid(row=1, column=0, sticky="w", pady=(10, 4))
    tk.OptionMenu(section, source_var, "ToF", "Thickness", "Speed", "Errors").grid(
        row=1, column=1, sticky="w", padx=(8, 0), pady=(10, 4)
    )

    speed_estimate_label = tk.Label(section, text="Speed Estimate (m/s):")
    speed_estimate_entry = tk.Entry(section, textvariable=speed_estimate_var, width=18)
    thickness_estimate_label = tk.Label(section, text="Thickness Estimate (m):")
    thickness_estimate_entry = tk.Entry(section, textvariable=thickness_estimate_var, width=18)
    second_csv_label = tk.Label(section, text="Second Input CSV:")
    second_csv_entry = tk.Entry(section, textvariable=second_csv_var, width=27)
    second_csv_browse = tk.Button(section, text="Browse...", width=12)

    def _toggle_value_inputs(*_: object) -> None:
        speed_estimate_label.grid_remove()
        speed_estimate_entry.grid_remove()
        thickness_estimate_label.grid_remove()
        thickness_estimate_entry.grid_remove()
        second_csv_label.grid_remove()
        second_csv_entry.grid_remove()
        second_csv_browse.grid_remove()

        if source_var.get() == "Thickness":
            speed_estimate_label.grid(row=2, column=0, sticky="w", pady=(0, 6))
            speed_estimate_entry.grid(row=2, column=1, sticky="w", padx=(8, 0), pady=(0, 6))
        elif source_var.get() == "Speed":
            thickness_estimate_label.grid(row=2, column=0, sticky="w", pady=(0, 6))
            thickness_estimate_entry.grid(row=2, column=1, sticky="w", padx=(8, 0), pady=(0, 6))
        elif source_var.get() == "Errors":
            second_csv_label.grid(row=2, column=0, sticky="w", pady=(0, 6))
            second_csv_entry.grid(row=2, column=1, sticky="w", padx=(8, 0), pady=(0, 6))
            second_csv_browse.grid(row=2, column=2, sticky="w", padx=(8, 0), pady=(0, 6))

    def browse_second_csv() -> None:
        path = filedialog.askopenfilename(
            title="Second input CSV for repeatability errors",
            initialdir=str(RAW_DIR if RAW_DIR.exists() else PROJECT_ROOT),
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
        if path:
            second_csv_var.set(path)

    second_csv_browse.configure(command=browse_second_csv)
    source_var.trace_add("write", _toggle_value_inputs)
    _toggle_value_inputs()

    tk.Label(
        section,
        text="Grouping uses repeated-point changes: a new group starts when column 8 or 9 changes.",
        fg="gray",
        justify="left",
        wraplength=430,
    ).grid(row=3, column=0, columnspan=3, sticky="w", pady=(10, 10))

    def _show_results_text(output_text: str) -> None:
        dialog = tk.Toplevel()
        dialog.title("Repeatability Error Results")
        dialog.resizable(True, True)
        dialog.transient(parent.winfo_toplevel())
        dialog.grab_set()

        text_widget = tk.Text(dialog, wrap="word", height=20, width=80)
        text_widget.insert("1.0", output_text)
        text_widget.config(state="disabled")
        text_widget.pack(fill="both", expand=True, padx=12, pady=12)

        tk.Button(dialog, text="Close", command=dialog.destroy, width=12).pack(pady=(0, 12))
        dialog.focus_set()

    def calculate_repeatability() -> None:
        working_csv = _require_working_csv()
        if working_csv is None:
            return

        command = [
            sys.executable,
            str(PROJECT_ROOT / "data" / "calculate_repeatability_errors.py"),
            working_csv,
            "--method",
            method_var.get(),
            "--source-mode",
            source_var.get(),
        ]

        if source_var.get() in {"Thickness", "Speed"}:
            try:
                estimate = float(_resolve_source_estimate())
            except ValueError as exc:
                messagebox.showerror("Invalid Input", str(exc))
                return
            command.extend(["--estimate", str(estimate)])

        if source_var.get() == "Errors":
            second_csv = second_csv_var.get().strip()
            if not second_csv:
                messagebox.showerror("Missing CSV", "Please select the second input CSV for the Errors mode.")
                return
            command.extend(["--second-csv", second_csv])

        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            cwd=str(PROJECT_ROOT),
            check=False,
        )

        if result.returncode != 0:
            error_text = (result.stderr or result.stdout or "Calculation failed.").strip()
            messagebox.showerror("Repeatability Error", error_text)
            return

        output_text = (result.stdout or "No output returned.").strip()
        _show_results_text(output_text)

    tk.Button(section, text="Calculate", width=12, command=calculate_repeatability).grid(
        row=4, column=0, sticky="w"
    )


def _build_zero_delay_section(parent: tk.Widget, working_csv_var: tk.StringVar) -> None:
    section = tk.LabelFrame(parent, text="Zero Delay", padx=12, pady=10)
    section.pack(fill="x", pady=(0, 14))

    def _require_working_csv() -> str | None:
        working_csv = working_csv_var.get().strip()
        if not working_csv:
            messagebox.showerror("No Working CSV", "Please set a working CSV in the CSV Paths section first.")
            return None
        return working_csv

    def _show_average_result(output_text: str) -> None:
        dialog = tk.Toplevel()
        dialog.title("Zero Delay Result")
        dialog.resizable(False, False)
        dialog.transient(parent.winfo_toplevel())
        dialog.grab_set()

        message = tk.Label(dialog, text=output_text, justify="left", wraplength=420, padx=16, pady=16)
        message.pack()

        tk.Button(dialog, text="Close", command=dialog.destroy, width=12).pack(pady=(0, 12))
        dialog.focus_set()

    def calculate_zero_delay() -> None:
        working_csv = _require_working_csv()
        if working_csv is None:
            return

        result = subprocess.run(
            [
                sys.executable,
                str(PROJECT_ROOT / "data" / "calculate_zero_delay.py"),
                working_csv,
            ],
            capture_output=True,
            text=True,
            cwd=str(PROJECT_ROOT),
            check=False,
        )

        if result.returncode != 0:
            error_text = (result.stderr or result.stdout or "Calculation failed.").strip()
            messagebox.showerror("Zero Delay Error", error_text)
            return

        output_text = (result.stdout or "No output returned.").strip()
        _show_average_result(output_text)

    tk.Button(section, text="Calculate", width=12, command=calculate_zero_delay).grid(row=0, column=0, sticky="w")


def main() -> None:
    root = tk.Tk()
    root.title("Data Processing")
    root.resizable(False, False)

    outer = tk.Frame(root, padx=18, pady=16)
    outer.pack(fill="both", expand=True)

    tk.Label(outer, text="Data Processing", font=("Segoe UI", 14, "bold")).pack(anchor="w", pady=(0, 12))

    content = tk.Frame(outer)
    content.pack(fill="both", expand=True)

    left_panel = tk.Frame(content)
    left_panel.pack(side="left", fill="y", anchor="n")

    right_panel = tk.Frame(content)
    right_panel.pack(side="left", fill="y", padx=(18, 0), anchor="n")

    tof_panel = tk.Frame(content)
    tof_panel.pack(side="left", fill="y", padx=(18, 0), anchor="n")

    input_csv_var = tk.StringVar(value="")
    working_csv_var = tk.StringVar(value="")
    output_csv_var = tk.StringVar(value="")

    _build_csv_paths_section(left_panel, input_csv_var, working_csv_var, output_csv_var)
    _build_filters_section(left_panel, input_csv_var, working_csv_var, output_csv_var)
    _build_signal_plotting_section(left_panel, working_csv_var)
    _build_plots_section(right_panel, working_csv_var)
    _build_tof_calculation_methods_section(tof_panel, working_csv_var)
    _build_peak_finding_variables_section(tof_panel)
    _build_repeatability_errors_section(tof_panel, working_csv_var)
    _build_zero_delay_section(tof_panel, working_csv_var)

    tk.Button(outer, text="Exit", width=34, command=root.destroy).pack(pady=(4, 0))

    root.mainloop()


if __name__ == "__main__":
    main()
