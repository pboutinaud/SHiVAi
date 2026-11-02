"""Download model ZIPs after establishing a Synology sharing session."""
import argparse
import shutil
import requests
from pathlib import Path
import tkinter as tk
from tkinter import filedialog 
import zipfile
from tqdm import tqdm

LATEST_MODELS = {
    'PVS': {'model': 'https://cloud.efixia.com/fsdownload/Zvbj1CpkM/T1-PVS.zip',  # V4
            'descriptor': ''},
    'WMH': {'model': 'https://cloud.efixia.com/fsdownload/Iae0B1old/T1.FLAIR-WMH.zip',  # V2
            'descriptor': ''},
    'CMB': {'model': 'https://cloud.efixia.com/fsdownload/qWYAqMeGX/SWI-CMB.zip',  # V2
            'descriptor': ''},
    'LAC': {'model': 'https://cloud.efixia.com/fsdownload/imqU23Nvl/T1.FLAIR-LAC.zip',  # V0
            'descriptor': ''},
    'brainmask': {'model': 'https://cloud.efixia.com/fsdownload/SXtROgtI7/brainmask.zip',
                  'descriptor': ''}
}

def main():
    parser = argparse.ArgumentParser(description="Download models for Shivai")
    parser.add_argument(
        "--models",
        choices=list(LATEST_MODELS.keys()) + ['all'],
        nargs='+',
        required=True,
        help="Name of the model to download"
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Directory to save the downloaded models (if not specified, a dialog will prompt for it)",
        required=False
    )
    args = parser.parse_args()

    if args.output_dir is None:
        root = tk.Tk()
        root.withdraw()
        try:
            if root.tk.call('tk', 'windowingsystem') == 'x11':
                for option, color in {
                    '*foreground': '#202020',
                    '*background': '#eeeeee',
                    '*activeForeground': '#202020',
                    '*activeBackground': '#dddddd',
                    '*selectForeground': '#ffffff',
                    '*selectBackground': '#315a86',
                    '*disabledForeground': '#666666',
                }.items():
                    root.option_add(option, color, 80)
            tkres = filedialog.askdirectory(parent=root)
        finally:
            root.destroy()
        if not tkres:
            return
        args.output_dir = Path(tkres)
    if not args.output_dir.exists():
        args.output_dir.mkdir(parents=True)
    print(f"Models will be downloaded to: {args.output_dir}")

    if 'all' in args.models:
        args.models = list(LATEST_MODELS.keys())
    for model_name in args.models:
        print(f"Downloading model: {model_name}")
        if model_name in LATEST_MODELS:
            url = LATEST_MODELS[model_name]['model']
            filename = url.split('/')[-1]
            output_path = args.output_dir / filename
            sharing_url = url.rsplit('/', 1)[0].replace('/fsdownload/', '/sharing/')
            with requests.Session() as session:
                with session.get(sharing_url, timeout=(10, 60)) as page:
                    page.raise_for_status()
                with session.get(url, stream=True, timeout=(10, 120)) as response:
                    response.raise_for_status()
                    if 'text/html' in response.headers.get('Content-Type', '').lower():
                        raise RuntimeError(
                            f"Received an HTML sharing page instead of the ZIP for {model_name}"
                        )
                    total_size = int(response.headers.get('Content-Length', 0)) or None
                    with open(output_path, "wb") as f, tqdm(
                        total=total_size,
                        desc=filename,
                        unit='B',
                        unit_scale=True,
                        unit_divisor=1024,
                    ) as progress:
                        for chunk in response.iter_content(chunk_size=10 * 1024):
                            if chunk:
                                f.write(chunk)
                                progress.update(len(chunk))
            print(f"Finished downloading {model_name} model to {output_path}")
            print(f"Extracting {model_name} model...")
            with zipfile.ZipFile(output_path, 'r') as zip_ref:
                zip_ref.extractall(args.output_dir)
            output_path.unlink()  # Remove the ZIP file after extraction
            print(f"Extracted {model_name} model to {args.output_dir}")
        else:
            print(f"Model {model_name} not found in LATEST_MODELS")
    

if __name__ == "__main__":
    main()