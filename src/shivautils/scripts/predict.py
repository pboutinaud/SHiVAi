#!/usr/bin/env python

# # predict something from one multi-modal nifti images
# Tested with Python 3.7, Tensorflow 2.7
# @author : Philippe Boutinaud - Fealinx

import gc
import os
import time
import json

import argparse
from pathlib import Path
import hashlib


import numpy as np
import nibabel
import tensorflow as tf
from shivautils.utils.misc import md5


def _load_image(filename):
    data_nii = nibabel.load(filename)
    # load file and add dimension for the modality
    image = data_nii.get_fdata(dtype=np.float32)[..., np.newaxis]
    return image, data_nii.affine


# Script parameters
def predict_parser():
    parser = argparse.ArgumentParser(
        description="Run inference with tensorflow models(s) on an image that may be built from several modalities"
    )
    parser.add_argument(
        "--t1",
        type=Path,
        action='store',
        help="T1W image",
        required=False)

    parser.add_argument(
        "--flair",
        type=Path,
        action='store',
        help="FLAIR image",
        required=False)

    parser.add_argument(
        "--swi",
        type=Path,
        action='store',
        help="SWI image",
        required=False)

    parser.add_argument(
        "--t2",
        type=Path,
        action='store',
        help="T2W image",
        required=False)

    parser.add_argument(
        "-m", "--model",
        type=Path,
        action='store',
        help="(multiple) input modality",
        required=True)

    parser.add_argument(
        "-descriptor",
        type=Path,
        required=True,
        help="File info json about model")

    parser.add_argument(
        "-b", "--braimask",
        type=Path,
        help="brain mask image")

    parser.add_argument(
        "-o", "--output",
        type=Path,
        help="path for the output file (output of the inference from tensorflow model)")

    parser.add_argument(
        "-g", "--gpu",
        type=int,
        help="GPU card ID; for CPU use -1")

    parser.add_argument(
        "--verbose",
        help="increase output verbosity",
        action="store_true")

    return parser


def main():
    pred_parser = predict_parser()
    args = pred_parser.parse_args()

    _VERBOSE = args.verbose

    # Set GPU
    if args.gpu is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
        if _VERBOSE:
            if args.gpu >= 0:
                print(f"Trying to run inference on GPU {args.gpu}")
            else:
                print("Trying to run inference on CPU")
    else:
        print(f"Trying to run inference on GPU {os.getenv('CUDA_VISIBLE_DEVICES')}")

    # The tf model files for the predictors, the prediction will be averaged
    predictor_files = []
    path = args.descriptor
    with open(path) as f:
        meta_data = json.load(f)

    for i in meta_data['files']:
        predictor_files.append(os.path.join(args.model, i['name']))

    if len(predictor_files) == 0:
        raise FileNotFoundError('Found no model files, '
                                'please supply or mount a folder '
                                'containing h5 files with model weights.')

    modalities = []
    for modality in meta_data['modalities']:
        if args.t1 and "t1" == modality:
            modalities.append(args.t1)
        if args.flair and "flair" == modality:
            modalities.append(args.flair)
        if args.swi and "swi" == modality:
            modalities.append(args.swi)
        if args.t2 and "t2" == modality:
            modalities.append(args.swi)
    if args.t1 and "t1" not in meta_data['modalities']:
        raise ValueError("ERROR : the prediction task doesn't require t1 modality according to json descriptor file metadata")
    if args.flair and "flair" not in meta_data['modalities']:
        raise ValueError("ERROR : the prediction task doesn't require flair modality according to json descriptor file metadata")
    if args.swi and "swi" not in meta_data['modalities']:
        raise ValueError("ERROR : the prediction task doesn't require swi modality according to json descriptor file metadata")
    if args.t2 and "t2" not in meta_data['modalities']:
        raise ValueError("ERROR : the prediction task doesn't require t2 modality according to json descriptor file metadata")

    for file in meta_data["files"]:
        print(args.model)
        path_file = os.path.join(args.model, file["name"])
        hashmd5 = md5(path_file)
        if file["md5"] != hashmd5:
            raise ValueError("Mismatch between expected file from the model descriptor and the actual model file")

    brainmask = args.braimask
    output_path = args.output

    affine = None
    image_shape = None
    # Load brainmask if given (and get the affine & shape from it)
    if brainmask is not None:
        brainmask, aff = _load_image(brainmask)
        image_shape = brainmask.shape
        if affine is None:
            affine = aff

    # Load and/or build image from modalities
    images = []
    for modality in modalities:
        image, aff = _load_image(modality)
        if affine is None:
            affine = aff
        if image_shape is None:
            image_shape = image.shape
        else:
            if image.shape != image_shape:
                raise ValueError(
                    f'Images have different shape {image_shape} vs {image.shape} in {modality}'  # noqa: E501
                )
        if brainmask is not None:
            image *= brainmask
        images.append(image)
    # Concat all modalities
    images = np.concatenate(images, axis=-1)
    # Add a dimension for a batch of one image
    images = np.reshape(images, (1,) + images.shape)

    chrono0 = time.time()
    # Load models & predict
    predictions = []
    for predictor_file in predictor_files:
        print(f"Loading predictor file: {predictor_file}")
        tf.keras.backend.clear_session()
        gc.collect()
        try:
            model = tf.keras.models.load_model(
                predictor_file,
                compile=False,
                custom_objects={"tf": tf})
        except Exception as err:
            print(f'\n\tWARNING : Exception loading model : {predictor_file}\n{err}')
            continue
        if hasattr(predictor_file, 'stem'):
            print('INFO : Predicting fold :', predictor_file.stem)
        prediction = model.predict(
            images,
            batch_size=1
        )
        if brainmask is not None:
            prediction *= brainmask
        predictions.append(prediction)

    # Average all predictions
    predictions = np.mean(predictions, axis=0)

    chrono1 = (time.time() - chrono0) / 60.
    if _VERBOSE:
        print(f'Inference time : {chrono1} sec.')

    # Threshold to remove near-zero voxels
    pred = predictions[0]
    pred[pred < 0.001] = 0

    # Save prediction
    nifti = nibabel.Nifti1Image(pred.astype('float32'), affine=affine)
    nibabel.save(nifti, output_path)

    if _VERBOSE:
        print(f'\nINFO : Done with predictions -> {output_path}\n')


if __name__ == "__main__":
    main()
