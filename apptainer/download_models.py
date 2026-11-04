#!/usr/bin/env python
"""Run shiva_download in an existing container using only the standard library."""
import argparse
import os
from pathlib import Path
import subprocess


def pick_output_dir():
    try:
        import tkinter as tk
        from tkinter import filedialog
    except ImportError as exc:
        raise RuntimeError("The folder picker requires tkinter; install it or provide -o.") from exc

    try:
        root = tk.Tk()
        try:
            root.withdraw()
            return filedialog.askdirectory(parent=root, title="Download SHiVAi models to")
        finally:
            root.destroy()
    except tk.TclError as exc:
        raise RuntimeError(f"Cannot open the folder picker; provide -o instead: {exc}") from exc


def build_exec_command(args, output_dir):
    # Preserve host paths in auto_config.yml wherever Linux bind targets allow it.
    target = str(output_dir) if os.name == 'posix' else '/mnt/model'
    mount = f"{output_dir}:{target}:rw"
    if args.docker:
        command = ['docker', 'run', '--rm']
        if os.name == 'posix':
            command += ['--user', f'{os.getuid()}:{os.getgid()}']
        command += ['-e', f'HOME={target}', '-w', target, '-v', mount, args.docker]
    else:
        command = ['apptainer', 'exec', '--bind', mount, '--pwd', target, str(args.apptainer)]
    command += ['shiva_download', '--models', *args.models, '--output-dir', target]
    if args.version:
        command += ['--version', args.version]
    return command


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Download SHiVAi models using an existing Apptainer or Docker image."
    )
    runtime = parser.add_mutually_exclusive_group(required=True)
    runtime.add_argument('--apptainer', type=Path, metavar='IMAGE',
                         help="Path to a local SHiVAi Apptainer image (.sif)")
    runtime.add_argument('--docker', metavar='IMAGE',
                         help="SHiVAi Docker image name, optionally including a tag")
    parser.add_argument('--models', '-m', nargs='+', required=True,
                        choices=['PVS', 'PVS2', 'WMH', 'CMB', 'LAC', 'brainmask', 'all'],
                        help="Models to download")
    parser.add_argument('--output-dir', '-o', type=Path,
                        help="Host output directory (if omitted, open a host folder picker)")
    parser.add_argument('--version', '-v', choices=['V0', 'V2', 'V4', 'V5'],
                        help="Model version; multiple models use their latest versions")
    args = parser.parse_args(argv)

    if args.apptainer:
        if os.name != 'posix':
            parser.error("Apptainer requires a Linux environment; run this script in WSL.")
        args.apptainer = args.apptainer.expanduser().resolve()
        if not args.apptainer.is_file():
            parser.error(f"Apptainer image does not exist: {args.apptainer}")

    try:
        if args.output_dir is None:
            selected_dir = pick_output_dir()
            if not selected_dir:
                print("Download cancelled: no output directory selected.")
                return 0
            args.output_dir = Path(selected_dir)
        output_dir = args.output_dir.expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Models will be downloaded to: {output_dir}", flush=True)
        return subprocess.run(build_exec_command(args, output_dir)).returncode
    except (OSError, RuntimeError) as exc:
        parser.exit(1, f"{parser.prog}: error: {exc}\n")


if __name__ == '__main__':
    raise SystemExit(main())
