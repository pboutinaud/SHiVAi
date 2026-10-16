
import argparse
import requests
from pathlib import Path
import tkinter as tk
from tkinter import filedialog 
import zipfile

LATEST_MODELS = {
    'PVS': 'https://cloud.efixia.com/sharing/Zvbj1CpkM/T1-PVS.zip', # V4
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
            response = requests.get(url, stream=True)
            response.raise_for_status()
            with open(output_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=10 * 1024):
                    f.write(chunk)
            print(f"Downloaded {model_name} to {output_path}")
            with zipfile.ZipFile(output_path, 'r') as zip_ref:
                zip_ref.extractall(args.output_dir)
            print(f"Extracted {model_name} to {args.output_dir}")
        else:
            print(f"Model {model_name} not found in LATEST_MODELS")
    

if __name__ == "__main__":
    main()