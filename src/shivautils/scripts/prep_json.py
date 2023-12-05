#!/usr/bin/env python

import hashlib
import argparse
import os
import glob
import json


def md5(fname):
    hash_md5 = hashlib.md5()
    with open(fname, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def my_parser():
    DESCRIPTION = """Prepare the JSON files necessary for shivautils(SHIV-AI) as they
    store the information about the deep-learning models being used."""

    parser = argparse.ArgumentParser(description=DESCRIPTION)

    parser.add_argument('--folder', '-f',
                        help='Path to the folder containing the deep-learning model (e.g. /home/ReferenceModels/T1-PVS)',
                        required=True)
    parser.add_argument('--target', '-t',
                        help='Type of structure segmented (if not given, use the input folder name to guess them)',
                        choices=['PVS', 'WMH', 'CMB', 'brain_mask'],
                        required=False)
    parser.add_argument('--modalities', '-m',
                        help='Acquisition modalities (if not given, use the input folder name to guess them)',
                        choices=['t1', 'flair', 'swi', 't2', 't2s'],
                        nargs='+',
                        required=False)
    parser.add_argument('--version', '-v',
                        help='Model version (e.g. "V0")',
                        default='')
    parser.add_argument('--date', '-d',
                        help='Release date (e.g. "2023/07/17")',
                        default='')
    parser.add_argument('--author', '-a',
                        help='Model author',
                        default='')
    parser.add_argument('--extension', '-x',
                        help='Extension of the model files (without the dot, e.g.: "h5")',
                        default='h5')
    return parser


def main():
    parser = my_parser()
    args = parser.parse_args()
    model_dict = {}

    modalities = ['t1', 'flair', 'swi', 't2', 't2s']
    targets = ['PVS', 'WMH', 'CMB', 'brain_mask']

    in_dir = args.folder
    ext = args.extension
    dirname = os.path.basename(in_dir)
    if not dirname:  # When indir end with a separator
        dirname = os.path.basename(dirname)
    if args.target is not None:
        model_dict['target'] = args.target
    else:
        targ = [t for t in targets if t in dirname]
        if 'brainmask' in dirname:
            targ.append('brain_mask')
        if len(targ) != 1:
            raise ValueError(f'There should only be one target name in the folder name, but {len(targ)} were found: {targ}')
        else:
            model_dict['target'] = targ[0]
    if args.modalities is not None:
        model_dict['modalities'] = args.modalities
    else:
        mod = [m for m in modalities if m.upper() in dirname]
        if len(mod) == 0:
            raise ValueError('No correct modality was found in the folder name')
        else:
            model_dict['modalities'] = mod

    model_files = glob.glob(os.path.join(in_dir, '*', f'*.{ext}'))
    if not model_files:
        model_files = glob.glob(os.path.join(in_dir, f'*.{ext}'))
    base_dir = os.path.dirname(in_dir)
    model_files = [f[len(base_dir) + len(os.sep):] for f in model_files]  # removing the base_dir part
    model_dirs = list(set([os.path.dirname(f) for f in model_files]))
    if len(model_dirs) > 1:
        if not args.version:
            raise ValueError('Multiple sub-folders were detected in the model folder (assuming it is different versions) '
                             'but no version label was specified. Please, enter the version in the arguments')
        else:
            model_dir = [d for d in model_dirs if args.version in d.split(os.sep)][0]
        if len(model_dir) == 0:
            raise ValueError(f'No folder path containing the specified version {args.version} (as a sub-folders).')
    else:
        model_dir = model_dirs[0]
    model_files = [f for f in model_files if model_dir in f]
    modelfiles_dicts = [{'name': f, 'md5': md5(os.path.join(base_dir, f))} for f in model_files]
    model_dict['files'] = modelfiles_dicts
    model_dict['version'] = args.version
    model_dict['release date'] = args.date
    model_dict['author'] = args.author
    json_file = os.path.join(base_dir, model_dir, 'model_info.json')
    with open(json_file, 'w') as fp:
        json.dump(model_dict, fp, indent=2)


if __name__ == "__main__":
    main()
