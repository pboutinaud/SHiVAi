"""Download model ZIPs after establishing a Synology sharing session."""
import argparse
import requests
from pathlib import Path
import tkinter as tk
from tkinter import filedialog 
import zipfile
from tqdm import tqdm

LATEST_MODELS = {
    'PVS': 'https://cloud.efixia.com/fsdownload/Zvbj1CpkM/T1-PVS.zip',  # V4
    'WMH': 'https://cloud.efixia.com/fsdownload/Iae0B1old/T1.FLAIR-WMH.zip',  # V2
    'CMB': 'https://cloud.efixia.com/fsdownload/qWYAqMeGX/SWI-CMB.zip',  # V2
    'LAC': 'https://cloud.efixia.com/fsdownload/imqU23Nvl/T1.FLAIR-LAC.zip',  # V0
    'brainmask': 'https://cloud.efixia.com/fsdownload/SXtROgtI7/brainmask.zip'
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
        args.output_dir = Path(filedialog.askdirectory())
    if not args.output_dir.exists():
        args.output_dir.mkdir(parents=True)
    print(f"Models will be downloaded to: {args.output_dir}")

    if 'all' in args.models:
        args.models = list(LATEST_MODELS.keys())
    for model_name in args.models:
        print(f"Downloading model: {model_name}")
        if model_name in LATEST_MODELS:
            url = LATEST_MODELS[model_name]
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
            print(f"Downloaded {model_name} to {output_path}")
            with zipfile.ZipFile(output_path, 'r') as zip_ref:
                zip_ref.extractall(args.output_dir)
            print(f"Extracted {model_name} to {args.output_dir}")
        else:
            print(f"Model {model_name} not found in LATEST_MODELS")
    

if __name__ == "__main__":
    main()