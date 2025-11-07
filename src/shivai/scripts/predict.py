#!/usr/bin/env python

# # predict something from one multi-modal nifti images
# Tested with Python 3.7, Tensorflow 2.7
# @author : Philippe Boutinaud - Fealinx

import gc
import os
import time
import json
import importlib
import sys
import inspect

import argparse
from pathlib import Path


import numpy as np
import nibabel
import tensorflow as tf
import keras
from shivai.utils.misc import md5


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
        "-o", "--out_dir",
        type=Path,
        help="path for the output file (output of the inference from tensorflow model)")

    parser.add_argument(
        "-g", "--gpu",
        type=int,
        help="GPU card ID; for CPU use -1")

    parser.add_argument(
        '--use_cpu',
        default=0,
        type=int,
        required=False,
        help=('If other than 0, will run the model on CPUs (limiting usage by the given number). '
              'Note however that some model may not be compatible with CPUs, which will lead to a crash.')
    )

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
    if args.use_cpu:
        tf.config.set_visible_devices([], 'GPU')
        tf.config.threading.set_intra_op_parallelism_threads(args.use_cpu)
        tf.config.threading.set_inter_op_parallelism_threads(args.use_cpu)
        if _VERBOSE:
            print("Trying to run inference on CPU")
    else:
        if args.gpu is not None:
            if args.gpu >= 0:
                os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
                if _VERBOSE:
                    print(f"Trying to run inference on GPU {args.gpu}")
            else:
                raise ValueError("Trying to run the inference on CPU (--gpu is negative) but --use_cpu is 0 or not set")
        else:
            if _VERBOSE:
                print(f"Trying to run inference on available GPU(s)")

    # The tf model files for the predictors, the prediction will be averaged
    model_files = []  # type: list[Path]
    path = args.descriptor
    with open(path) as f:
        meta_data = json.load(f)

    model_dir = Path(args.model)

    notfound = []

    keras_model = None
    if 'script' in meta_data:
        keras_model = model_dir / meta_data['script']['name']
        if not keras_model.exists():
            notfound.append(keras_model)
        else:
            k_md5 = meta_data['script']['md5']
            k_hashmd5 = md5(keras_model)
            if k_hashmd5 != k_md5:
                raise ValueError("Mismatch between expected file from the model descriptor and the actual model script")

    savedModel = False  # Whether the model is a keras savedModel or an h5 file
    for mfile in meta_data['files']:
        mfilename = Path(mfile['name'])
        if not (model_dir / mfilename).exists():
            if mfilename.parts[0] == model_dir.parts[-1]:
                # model dir is in both model_dir and mfilename
                model_dir = model_dir.parent
        model_file = model_dir / mfilename
        if not model_file.exists():
            raise ValueError(f'Model file {model_file} was not found.')
        hashmd5 = md5(model_file)
        if mfile["md5"] != hashmd5:
            raise ValueError("Mismatch between expected file from the model descriptor and the actual model file")
        savedModel = model_file.is_dir()
        model_files.append(model_file)

    if len(model_files) == 0:
        raise ValueError('Found no model files, '
                         'please supply or mount a folder '
                         'containing h5 files with model weights.')
    for model_file in model_files:
        if not os.path.exists(model_file):
            notfound.append(model_file)
    if notfound:
        raise ValueError('Some (or all) model files/folders were missing.\n'
                         'Please supply or mount a folder '
                         'containing the model files/folders with model weights.\n'
                         'Current problematic paths:\n\t' +
                         '\n\t'.join(notfound))

    if keras_model:
        # Execute keras_model to have access to its classes
        # with open(keras_model) as kf:
        #     exec(kf.read())  # doesn't work when calling the script...
        spec = importlib.util.spec_from_file_location('kmodel', keras_model)
        kmodel = importlib.util.module_from_spec(spec)
        sys.modules['kmodel'] = kmodel
        globals()['kmodel'] = kmodel
        spec.loader.exec_module(kmodel)
        detected_classes = [c for c in dir(kmodel) if inspect.isclass(eval(f'kmodel.{c}'))]
        for modl_class in detected_classes:
            exec(f'{modl_class} = kmodel.{modl_class}')

    modalities = []
    for modality in meta_data['modalities']:
        if args.t1 and "t1" == modality:
            modalities.append(args.t1)
        if args.flair and "flair" == modality:
            modalities.append(args.flair)
        if args.swi and "swi" == modality:
            modalities.append(args.swi)
        if args.t2 and "t2" == modality:
            modalities.append(args.t2)
    if args.t1 and "t1" not in meta_data['modalities']:
        raise ValueError("ERROR : the prediction task doesn't require t1 modality according to json descriptor file metadata")
    if args.flair and "flair" not in meta_data['modalities']:
        raise ValueError("ERROR : the prediction task doesn't require flair modality according to json descriptor file metadata")
    if args.swi and "swi" not in meta_data['modalities']:
        raise ValueError("ERROR : the prediction task doesn't require swi modality according to json descriptor file metadata")
    if args.t2 and "t2" not in meta_data['modalities']:
        raise ValueError("ERROR : the prediction task doesn't require t2 modality according to json descriptor file metadata")

    brainmask = args.braimask
    output_path = args.out_dir

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
    for model_file in model_files:
        print(f"Loading predictor file: {model_file}")
        tf.keras.backend.clear_session()
        gc.collect()
        try:
            if keras_model:
                model = keras.saving.load_model(model_file, custom_objects=None, compile=False)
            elif savedModel:
                model = tf.saved_model.load(model_file)
                infer = model.signatures["serving_default"]
                input_names = list(infer.structured_input_signature[1].keys())
                input_name = input_names[0]
            else:
                model = tf.keras.models.load_model(
                    model_file,
                    compile=False,
                    custom_objects={"tf": tf})
        except Exception as err:
            print(f'\n\tWARNING : Exception loading model : {model_file}\n{err}')
            continue
        if hasattr(model_file, 'stem'):
            print('INFO : Predicting fold :', model_file.stem)

        if savedModel:
            result = infer(**{input_name: tf.constant(images, dtype=tf.float32)})
            output_names = list(result.keys())
            predictions = result[output_names[0]].numpy()
        else:
            prediction = model.predict(
                images,
                batch_size=1
            )
        # prediction = model(images, training=False)  # slightly (?) faster alternative
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
