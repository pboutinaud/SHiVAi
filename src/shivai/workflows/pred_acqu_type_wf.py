import json

from nipype.pipeline.engine import Workflow, Node, JoinNode
from nipype.interfaces.utility import IdentityInterface, Function
from shivai.interfaces.shiva import Predict_Multi
from nipype.interfaces.io import DataSink
from shivai.interfaces.image import (Threshold, Normalization, CorrectAffine,
                                     Conform, Crop, Resample_from_to)
from pathlib import Path
import nibabel as nib
import numpy as np
import argparse


def parse_filename(filename):
    filename = Path(filename)
    sub = filename.parent.parent.parent.name
    sub_id = sub.split('-')[1]
    session = filename.parent.parent.name
    session = session.split(sub_id)[-1].strip()
    modality = filename.parent.name
    return f'{sub}_{session}_{modality}'


def res_to_dict(orig_files, in_res):
    from pathlib import Path

    def parse_filename(filename):
        filename = Path(filename)
        sub = filename.parent.parent.parent.name
        sub_id = sub.split('-')[1]
        session = filename.parent.parent.name
        session = session.split(sub_id)[-1].strip()
        modality = filename.parent.name
        return f'{sub}_{session}_{modality}'

    dict_files = {parse_filename(f): r for f, r in zip(orig_files, in_res)}
    return dict_files


def dict_to_res(orig_file, files_dict):
    from pathlib import Path

    def parse_filename(filename):
        filename = Path(filename)
        sub = filename.parent.parent.parent.name
        sub_id = sub.split('-')[1]
        session = filename.parent.parent.name
        session = session.split(sub_id)[-1].strip()
        modality = filename.parent.name
        return f'{sub}_{session}_{modality}'

    return files_dict[parse_filename(orig_file)]


def final_pred_gen(res_dict: dict[str, str], outname: str):
    import json
    final_dict = {}
    for acqu, res_dict_f in res_dict.items():
        with open(res_dict_f) as res_dict_fo:
            res_dict = json.load(res_dict_fo)
        pred = max(res_dict, key=res_dict.get)
        final_dict[acqu] = pred
    with open(outname, 'w') as f:
        json.dump(final_dict, f, indent=4)
    return outname


def filter_files(files, limit=None):
    valid_files = []
    print(f'Filtering through {len(files)} files to find valid acquisitions...')
    for i, f in enumerate(files, start=1):
        if i % 20 == 0:
            print(f'Checked {i} files so far...')
        im = nib.load(f)
        if not np.issubdtype(im.get_data_dtype(), np.number):
            continue
        if np.isnan(nib.orientations.io_orientation(im.affine)).any():
            continue
        if len(im.shape) == 3 and im.shape[-1] > 10:
            valid_files.append(f)
        elif len(im.shape) == 4 and im.shape[-1] == 1:
            valid_files.append(f)
        if limit and len(valid_files) >= limit:
            break
    return valid_files


def main():
    

    parser = argparse.ArgumentParser(
        description='Predict acquisition type from input images for SHiVAi.')
    parser.add_argument('--input', type=Path, default='/input/',
                        dest='indir',
                        help='Input directory with all files (**/*/*.nii.gz).')
    parser.add_argument('--output', type=Path, default='/output/',
                        dest='outdir',
                        help='Output directory for results.')
    parser.add_argument('--model_dir', type=Path, required=True,
                        help='Directory containing the model and model_info.json')
    args = parser.parse_args()
    out_dir = args.outdir
    indir = args.indir
    pred_in_shape = (160, 214, 176)

    all_files = list(indir.glob('**/*/*.nii.gz'))

    valid_files = filter_files(all_files, 10)
    all_subjs = set([f.parent.parent.parent.name for f in valid_files])

    wf = Workflow(name='test_wf')
    wf.base_dir = out_dir
    file_iterator = Node(
        IdentityInterface(
            fields=['image_file'],
            mandatory_inputs=True),
        name="file_iterator")
    file_iterator.iterables = ('image_file', valid_files)

    # data_grabber = Node(DataGrabber(), name='data_grabber')

    conform = Node(Conform(), name='conform')
    conform.inputs.dimensions = pred_in_shape
    conform.inputs.voxel_size = (1, 1, 1)

    wf.connect(file_iterator, 'image_file', conform, 'img')

    norm = Node(Normalization(), name='normalization')
    norm.inputs.percentile = 0.99

    wf.connect(conform, 'resampled', norm, 'input_image')

    join_preproc = JoinNode(Function(input_names=['orig_files', 'in_res'],
                                     output_names=['files_dict'],
                                     function=res_to_dict),
                            joinsource=file_iterator,
                            joinfield=['orig_files', 'in_res'],
                            name='preproc_joiner')

    wf.connect(norm, 'intensity_normalized', join_preproc, 'in_res')
    wf.connect(file_iterator, 'image_file', join_preproc, 'orig_files')

    pred = Node(Predict_Multi(), name='predict_multi')
    pred.inputs.model_dir = args.model_dir
    pred.inputs.descriptor = args.model_dir / 'model_info.json'
    pred.inputs.foutname = '{sub}_pred.json'
    pred.plugin_args = {'sbatch_args': '--nodes 1 --cpus-per-task 8 --gpus 1'}

    wf.connect(join_preproc, 'files_dict', pred, 'primary_image_file')

    compile_pred = Node(Function(input_names=['res_dict', 'outname'],
                                 output_names=['final_pred_file'],
                                 function=final_pred_gen),
                        name='compile_pred')
    compile_pred.inputs.outname = out_dir / 'final_predictions.json'
    wf.connect(pred, 'segmentations', compile_pred, 'res_dict')

    wf.run(plugin="SLURM", plugin_args={'sbatch_args': '--nodes 1 --cpus-per-task 8 '})


if __name__ == '__main__':
    main()
