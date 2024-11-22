#!/usr/bin/env python

# # predict something from one multi-modal nifti images
# Tested with Python 3.7, Tensorflow 2.7
# @author : Philippe Boutinaud - Fealinx

import gc
import os
import json

import argparse
from pathlib import Path


import numpy as np
import nibabel as nib
import tensorflow as tf
from shivai.utils.misc import md5


# Script parameters
def predict_parser():
    parser = argparse.ArgumentParser(
        description="Run inference with tensorflow models(s) on images that may be built from several modalities"
    )

    parser.add_argument(
        "--subjects",
        type=str,
        help="List of the subjects name of the files given to --img1_files (must be in the same order)",
        nargs='+',
        required=True)

    parser.add_argument(
        "--img1_files",
        type=Path,
        help="List of the primary image files used by the model (separated by a space)",
        nargs='+',
        required=True)

    parser.add_argument(
        "--img2_files",
        type=Path,
        help="List of the secondary image files used by the model (separated by a space). Only used in multi-modal predictions",
        nargs='*',
        required=False)

    parser.add_argument(
        "--mask_files",
        type=Path,
        help="List of the brain mask files (optional)",
        nargs='*',
        required=False)

    parser.add_argument(
        "--out_dir",
        type=Path,
        help="path for the output file")

    parser.add_argument(
        "--foutname",
        default='{sub}_segmentation.nii.gz',
        help=('Output names of the files to be formated by replacing "{sub}" by the subect name '
              '(e.g. "{sub}_segmentation.nii.gz")')
    )

    parser.add_argument(
        "--model_dir",
        type=Path,
        help="Folder containing the AI models",
        required=True)

    parser.add_argument(
        "--descriptor",
        type=Path,
        required=True,
        help="JSON file describing info about the models")

    # parser.add_argument(
    #     "--acq_types",
    #     choices=['t1', 't2', 'flair', 'swi', 't2gre'],
    #     nargs='+',
    #     help='List of the acquisition modalities used for the prediction (in lower case)')

    parser.add_argument(
        '--batch_size',
        default=20,
        type=int,
        help='Number of images to load in the RAM at the same time(upstream of the GPU)'
    )

    parser.add_argument(
        "--input_size",
        default=(160, 214, 176),
        type=lambda strtuple: tuple([int(i) for i in strtuple.split('x')]),
        help=('Expected input (and output) size of the volumes accepted by the model, '
              'given as a string separated by "x" (e.g. "160x214x176")')
    )

    parser.add_argument(
        '--use_cpu',
        default=0,
        type=int,
        required=False,
        help=('If other than 0, will run the model on CPUs (limiting usage by the given number). '
              'Note however that some model may not be compatible with CPUs, which will lead to a crash.')
    )

    return parser


def main():
    pred_parser = predict_parser()
    args = pred_parser.parse_args()
    if args.use_cpu:
        tf.config.set_visible_devices([], 'GPU')
        tf.config.threading.set_intra_op_parallelism_threads(args.use_cpu)
        tf.config.threading.set_inter_op_parallelism_threads(args.use_cpu)
    # Obtaining the absolute path to all the model files
    model_files = []
    with open(args.descriptor) as f:
        meta_data = json.load(f)
    for mfile in meta_data['files']:
        mdirname = os.path.basename(args.model_dir)
        mfilename = mfile['name']
        if mfilename[:len(mdirname)] == mdirname:  # model dir is in both args.model and mfile['name']
            mfilename = os.path.join(*mfilename.split(os.sep)[1:])
        model_file = os.path.join(args.model_dir, mfilename)
        model_files.append(model_file)

    notfound = []  # Doing it this way to found all the missing files in one pass
    badmd5 = []
    for model_file, file_data in zip(model_files, meta_data['files']):
        if not os.path.exists(model_file):
            notfound.append(model_file)
        else:
            hashmd5 = md5(model_file)
            if file_data["md5"] != hashmd5:
                badmd5.append(model_file)
    if notfound:
        raise FileNotFoundError('Some (or all) model files/folders were missing.\n'
                                'Please supply or mount a folder '
                                'containing the model files/folders with model weights.\n'
                                'Current problematic paths:\n\t' +
                                '\n\t'.join(notfound))
    if badmd5:
        raise ValueError("Mismatch between expected file from the model descriptor and the actual model file.\n"
                         "Files in question:\n\t" +
                         "\n\t".join(badmd5))

    # for modality in args.acq_types:
    #     if modality not in meta_data['modalities']:
    #         raise ValueError(f"ERROR : the prediction model does not use {modality} modality according to json descriptor file")

    # iterating over the model files and the input image files
    img1_files = args.img1_files
    img2_files = args.img2_files
    sub_list = args.subjects
    modality_num = 2 if img2_files is not None else 1
    if len(img1_files) != len(sub_list):
        raise ValueError(f'Missmatch between the list of subects and of main image')
    if img2_files is not None and len(img1_files) != len(img2_files):
        raise ValueError('Missmatch between the number of main and secondary files')
    step = len(sub_list)//args.batch_size + int(bool(len(sub_list) % args.batch_size))
    affine_dict = {}
    for fold, mfile in enumerate(model_files):
        # Load the model
        tf.keras.backend.clear_session()
        gc.collect()
        print(f"Loading model file: {mfile}")
        model = tf.keras.models.load_model(
            mfile,
            compile=False,
            custom_objects={"tf": tf})
        for i in range(step):
            curr_slice = slice(i*args.batch_size, (i+1)*args.batch_size)
            input_images = np.zeros((len(sub_list[curr_slice]), *args.input_size, modality_num))
            for j, (sub, in_file1) in enumerate(zip(sub_list[curr_slice], img1_files[curr_slice])):
                inIm = nib.load(in_file1)
                affine_dict[sub] = inIm.affine
                input_images[j, ..., 0] = inIm.get_fdata(dtype=np.float32)
                if img2_files is not None:
                    inIm = nib.load(img2_files[sub_list.index(sub)])
                    input_images[j, ..., 1] = inIm.get_fdata(dtype=np.float32)
            predictions = model.predict(
                input_images,
                batch_size=1
            )
            # Save temp results for the current fold model
            for j, sub in enumerate(sub_list[curr_slice]):
                sub_pred = predictions[j].squeeze()
                sub_pred[sub_pred < 0.001] = 0  # Threshold to remove near-zero voxels
                subpred_im = nib.Nifti1Image(sub_pred.astype('float32'), affine=affine_dict[sub])
                nib.save(subpred_im, f'tmp_{sub}_fold{fold}.nii.gz')

    # Taking each fold's results and averaging them
    print('Averaging the results of each model (done for each subject)...')
    for i, sub in enumerate(sub_list):
        pred_list = [nib.load(f'tmp_{sub}_fold{fold}.nii.gz').get_fdata(dtype='float32') for fold in range(len(model_files))]
        mean_pred = np.mean(pred_list, axis=0)
        if args.mask_files is not None:
            brainmask = nib.load(args.mask_files[sub_list.index(sub)]).get_fdata().astype(bool)
            mean_pred *= brainmask
        mean_pred_im = nib.Nifti1Image(mean_pred.astype('float32'),  affine=affine_dict[sub])
        outname = args.foutname.format(sub=sub)
        if args.out_dir:
            outname = os.path.join(args.out_dir, outname)
        nib.save(mean_pred_im, outname)
        for fold in range(len(model_files)):
            os.remove(f'tmp_{sub}_fold{fold}.nii.gz')


if __name__ == "__main__":
    main()
