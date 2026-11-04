"""Download model ZIPs after establishing a Synology sharing session."""
import argparse
import yaml
import requests
from pathlib import Path
import tkinter as tk
from tkinter import filedialog
import zipfile
from tqdm import tqdm

MODELS = {
    'PVS': {'V4':
                {'model': 'https://cloud.efixia.com/fsdownload/Zvbj1CpkM/T1-PVS.zip',
                 'descriptor': 'https://raw.githubusercontent.com/pboutinaud/SHIVA_PVS/refs/heads/main/model_info_t1-pvs-v4.json',
                 'alias': 'T1-PVS'},
            'V5': 
                {'model': 'https://cloud.efixia.com/fsdownload/TDbSoPRmm/T1-PVS.zip',
                 'descriptor': 'https://raw.githubusercontent.com/pboutinaud/SHIVA_PVS/refs/heads/main/model_info_t1-pvs-v5.json',
                 'alias': 'T1-PVS'}
            },
    'PVS2': {'V4':
                {'model': 'https://cloud.efixia.com/fsdownload/7r5t1hLTt/T1.FLAIR-PVS.zip',
                'descriptor': 'https://raw.githubusercontent.com/pboutinaud/SHIVA_PVS/refs/heads/main/model_info_t1-flair-pvs-v4.json',
                'alias': 'T1.FLAIR-PVS'}},
    'WMH': {'V2':
                {'model': 'https://cloud.efixia.com/fsdownload/Iae0B1old/T1.FLAIR-WMH.zip',
                'descriptor': 'https://raw.githubusercontent.com/pboutinaud/SHIVA_WMH/refs/heads/main/model_info_t1-flair-wmh-v2.json',
                'alias': 'T1.FLAIR-WMH'},
            'V5': 
                {'model': 'https://cloud.efixia.com/fsdownload/RiRbjLTam/T1.FLAIR-WMH.zip',
                 'descriptor': 'https://raw.githubusercontent.com/pboutinaud/SHIVA_WMH/refs/heads/main/model_info_t1-flair-wmh-v5.json',
                 'alias': 'T1.FLAIR-WMH'}
            },
    'CMB': {'V2':
                {'model': 'https://cloud.efixia.com/fsdownload/qWYAqMeGX/SWI-CMB.zip',
                'descriptor': 'https://raw.githubusercontent.com/pboutinaud/SHIVA_CMB/refs/heads/main/model_info_swi-cmb-v2.json',
                'alias': 'SWI-CMB'},
            'V5': 
                {'model': 'https://cloud.efixia.com/fsdownload/0A63lc9bQ/SWI-CMB.zip',
                 'descriptor': 'https://raw.githubusercontent.com/pboutinaud/SHIVA_CMB/refs/heads/main/model_info_swi-cmb-v5.json',
                 'alias': 'SWI-CMB'}
            },
    'LAC': {'V0':
                {'model': 'https://cloud.efixia.com/fsdownload/imqU23Nvl/T1.FLAIR-LAC.zip',
                'descriptor': 'https://raw.githubusercontent.com/pboutinaud/SHIVA_LAC/refs/heads/main/model_info_t1-flair-lac-v0.json',
                'alias': 'T1.FLAIR-LAC'}},
    'brainmask': {'V0': {'model': 'https://cloud.efixia.com/fsdownload/SXtROgtI7/brainmask.zip',
                         'alias': 'brainmask'}}
}

YAML_EX_URL = "https://raw.githubusercontent.com/pboutinaud/SHiVAi/refs/heads/master/apptainer/config_example.yml"


def main():
    parser = argparse.ArgumentParser(description="Download models for Shivai")
    parser.add_argument(
        "--models", "-m",
        choices=list(MODELS.keys()) + ['all'],
        nargs='+',
        required=True,
        help="Name of the model to download"
    )

    parser.add_argument(
        "--output-dir", "-o",
        type=Path,
        help="Directory to save the downloaded models (if not specified, a dialog will prompt for it)",
        required=False
    )

    parser.add_argument(
        "--version", "-v",
        choices=['V0', 'V2', 'V4', 'V5'],
        help=("Version of the model to download. Only applicable when downloading models one at a time. "
              "Downloading multiple models will ignore this option and download the latest versions.")
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
        args.models = list(MODELS.keys())
    if args.version and len(args.models) == 1:
        version = args.version
    else:
        args.version = -1
    for model_name in args.models:
        if args.version < 0:
            version = list(MODELS[model_name].keys())[-1]
        else:
            if version not in MODELS[model_name]:
                raise ValueError(f"Version {version} not available for model {model_name}. Available versions: {list(MODELS[model_name].keys())}")
        if model_name in MODELS:
            alias = MODELS[model_name][version].get('alias', model_name)
            url = MODELS[model_name][version]['model']
            filename = url.split('/')[-1]
            model_outdir = args.output_dir / alias / version
            zip_path = model_outdir / filename
            if not model_outdir.exists():
                model_outdir.mkdir(parents=True)
            if any(model_outdir.iterdir()):
                print(f"Model output directory {model_outdir} is not empty. Skipping download.")
                continue
            print(f"Downloading model: {model_name} ({version})")
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
                    with open(zip_path, "wb") as f, tqdm(
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
            print(f"Finished downloading {model_name} model to {zip_path}")
            print(f"Extracting {model_name} model...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(model_outdir)
            zip_path.unlink()  # Remove the ZIP file after extraction
            # If the zip contained a single folder, move its contents up one level
            extracted_items = list(model_outdir.iterdir())
            if len(extracted_items) == 1 and extracted_items[0].is_dir():
                for item in extracted_items[0].iterdir():
                    item.rename(model_outdir / item.name)
                extracted_items[0].rmdir()
            print(f"Extracted {model_name} model to {model_outdir}")
            if 'descriptor' in MODELS[model_name][version]:
                descriptor_url = MODELS[model_name][version]['descriptor']
                descriptor_filename = descriptor_url.split("/")[-1]
                with requests.get(descriptor_url) as response:
                    response.raise_for_status()
                    descriptor_path = model_outdir / descriptor_filename
                    with open(descriptor_path, "wb") as f:
                        f.write(response.content)
                print(
                    f"Downloaded descriptor for {model_name} to {descriptor_path}")
        else:
            print(f"Model {model_name} not found in MODELS")
    prep_yaml(args.output_dir)


def prep_yaml(output_dir: Path):
    descriptors = find_latest_descriptors(output_dir)
    mapping = {f"{model}_descriptor": descriptors.get(model, ("", None))[1] for model in MODELS}
    mapping["model_path"] = str(output_dir.absolute())
    TO_COMMENT = [
        "container_runtime",
        "apptainer_image",
        "synthseg_image"
    ]
    yaml_path = output_dir / "auto_config.yml"
    if not yaml_path.exists():
        print("Generating a minimal configuration file")
        with requests.get(YAML_EX_URL) as response:
            response.raise_for_status()
            with open(yaml_path, "wb") as f:
                f.write(response.content)
        print(f"Downloaded YAML example to {yaml_path}")
        # Read the YAML file line by line, replacing placeholders with actual paths and commenting usused entries
        with open(yaml_path, "r") as f:
            lines = f.readlines()
        with open(yaml_path, "w") as f:
            for line in lines:
                for keyname, value in mapping.items():
                    if keyname in line:
                        startIndex = line.find(":") + 1
                        # Overwrite example placeholder with actual value if available, and delete the placeholder if no value is provided
                        line = line[:startIndex] + (f' "{value}"' if value else '') + "\n"
                if any(to_comment in line for to_comment in TO_COMMENT) and not line.strip().startswith("#"):
                    line = f"# {line}"
                f.write(line)
    else:
        print(f"Configuration file already exists at {yaml_path}. Only updating the empty fields with the new values.")
        with open(yaml_path, "r") as f:
            lines = f.readlines()
        with open(yaml_path, "w") as f:
            for line in lines:
                for keyname, value in mapping.items():
                    if keyname in line:
                        startIndex = line.find(":") + 1
                        if value and len(line[startIndex:].strip()) == 0:
                            line = line[:startIndex] + f' "{value}"\n'
                f.write(line)


def find_latest_descriptors(output_dir: Path) -> dict[str, tuple[str, str]]:
    # Find the latest descriptor for each model in the output directory
    # Return a dictionary of form {"PVS": ("V4", "path/to/descriptor.json")}
    descriptors = {}
    for model_name in MODELS:
        for version in MODELS[model_name]:
            alias = MODELS[model_name][version].get('alias', model_name)
            model_outdir = output_dir / alias / version
            descriptor_url = MODELS[model_name][version].get('descriptor')
            if not descriptor_url and model_name == "brainmask":
                # The descriptor file is in the ziped folder so it's always the same name for brainmask
                descriptor_filename = "model_info.json"
            elif descriptor_url:
                descriptor_filename = MODELS[model_name][version]['descriptor'].split("/")[-1]
            else:
                continue  # Skip if no descriptor is available for this model/version
            descriptor_path = model_outdir / descriptor_filename
            if descriptor_path.exists():
                # Overwrite if this version is newer than the currently stored one
                descriptors[model_name] = (version, str(descriptor_path.absolute()))
    return descriptors


if __name__ == "__main__":
    main()
