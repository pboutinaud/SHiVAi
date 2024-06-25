#!/usr/bin/env python

import argparse
import os
import glob
import json
from shivai.utils.misc import md5


def my_parser(targs, mods):
    DESCRIPTION = (
        """
        Prepares the JSON files necessary for shivai(SHIV-AI) as they store the 
        information about the deep-learning models being used. If the name of the folder
        containing the data also indicates the necessary modalities (T1, FLAIR, etc.) and
        the target of the segmentation (PVS, WMH, etc.), the program can automatically fill
        these information in the json (so you don't need to specify --target or --modalities).
        """
    )

    parser = argparse.ArgumentParser(description=DESCRIPTION)

    parser.add_argument('--folder', '-f',
                        help=(
                            'Path to the folder containing the deep-learning model or the sub-folders '
                            'for the versions (e.g. /home/ReferenceModels/T1-PVS)'
                        ),
                        required=True)
    parser.add_argument('--target', '-t',
                        help='Type of structure segmented (if not given, use the input folder name to guess them)',
                        choices=targs,
                        required=False)
    parser.add_argument('--modalities', '-m',
                        help='Acquisition modalities (if not given, use the input folder name to guess them)',
                        choices=mods,
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
    parser.add_argument('--data_type', '-dt',
                        help=('Type of data ("folder" or "file") in which the model is saved. '
                              'Default is "file".\nWhen using "folder", all the folders in --folder '
                              'will be considered as model saved data ,so it is important to manually '
                              'specify the version with --version if there is an additional version '
                              'subfolder in --folder. Also, the --extension argument will be ignored.'),
                        choices=['file', 'folder'],
                        default='file')
    return parser


def json_maker(target, modalities, root_dir, model_files, version='', date='', author=''):
    """Creates a json file used as a 'model descriptor' in Shivai

    Args:
        target (str): Type of segmentation done by the model ('PVS', 'WMH', 'CMB', 'LAC', 'brain_mask')
        modalities (list): Imaging modalities used by the model (e.g. 't1', 'flair', 'swi', 't2', 't2s')
        root_dir (str): Directory containing the model file
        model_files (list): List of the model files (or directory), one per training fold (relative to 'root_dir')
        version (str): (opt.) Version number of the model (usually also in the directory path)
        date (str): (opt.) date of creation of the model
        author (str): (opt.) Author of the model
    """

    model_dict = {}

    # Preparing model_dict
    model_dict['version'] = version
    model_dict['release date'] = date
    model_dict['author'] = author
    model_dict['target'] = target
    model_dict['modalities'] = modalities
    model_files = [f[:-1] if f[-1] == os.sep else f for f in model_files]  # removing the "/" at the end if it's there
    model_dir = os.path.dirname(model_files[0])
    modelfiles_dicts = [{'name': f, 'md5': md5(os.path.join(root_dir, f))} for f in model_files]
    model_dict['files'] = modelfiles_dicts
    json_file = os.path.join(root_dir, model_dir, 'model_info.json')
    with open(json_file, 'w') as fp:
        json.dump(model_dict, fp, indent=2)


def main():

    targets = ['PVS', 'WMH', 'CMB', 'LAC', 'brain_mask']
    modalities = ['t1', 'flair', 'swi', 't2', 't2s']

    parser = my_parser(targets, modalities)
    args = parser.parse_args()

    in_dir = os.path.abspath(args.folder)
    ext = args.extension

    # Auto detect target and modalities
    root_dir, dirname = os.path.split(in_dir)  # Name (not path) of the folder i.g. "T1.FLAIR-PVS")
    if args.target is not None:
        target = args.target
    else:
        targ = [t for t in targets if t in dirname]
        if 'brainmask' in dirname:
            targ.append('brain_mask')
        if len(targ) != 1:
            raise ValueError(f'There should only be one target name in the folder name, but {len(targ)} were found: {targ}')
        else:
            target = targ[0]
    if args.modalities is not None:
        modals = args.modalities
    else:
        mod = [m for m in modalities if m.upper() in dirname]
        if len(mod) == 0:
            raise ValueError('No correct modality was found in the folder name')
        else:
            modals = mod

    # Looking for the model files/folders
    if args.data_type == 'file':
        model_files = glob.glob(os.path.join(in_dir, '*', f'*.{ext}'))
        if not model_files:
            model_files = glob.glob(os.path.join(in_dir, f'*.{ext}'))
        model_files = [os.path.relpath(f, root_dir) for f in model_files]  # removing the dirname part
        model_dirs = list(set([os.path.dirname(f) for f in model_files]))
        if len(model_dirs) > 1:
            if not args.version:
                raise ValueError('Multiple sub-folders were detected in the model folder (assuming it is different versions) '
                                 'but no version label was specified. Please, enter the version in the arguments')
            else:
                model_dir = [d for d in model_dirs if args.version in d.split(os.sep)][0]
            if len(model_dir) == 0:
                raise ValueError(f'No folder path containing the specified version {args.version} (as a sub-folder).')
        else:
            model_dir = model_dirs[0]
        model_files = [f for f in model_files if model_dir in f]  # Keeping only the files inside model_dir (necessary when multiple available versions)
    else:  # data_type = 'folder'
        model_dir = os.path.join(dirname, args.version)
        model_files = os.listdir(os.path.join(root_dir, model_dir))
        model_files = [os.path.join(model_dir, f) for f in model_files]
        model_files = [f for f in model_files if os.path.isdir(os.path.join(root_dir, f))]  # keeping only the folders
    if not model_files:
        raise FileNotFoundError('No file/folder detected')

    json_maker(target, modals, root_dir, model_files, args.version, args.date, args.author)


if __name__ == "__main__":
    main()
