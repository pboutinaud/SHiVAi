import argparse
import os
from pathlib import Path
from shivai.workflows.shiva_postproc_wf import genWorkflow
from shivai.utils.parsing import parse_LUT, parse_sub_list, parse_plugin_args


def main():
    _DESCRIPTION = """
    Shiva post-processing pipeline, used to compute statistics on cSVD biomarkers similar to the Shivai pipeline.
    Expected input structure in the "indir" folder (the nifti files can be .nii or .nii.gz, and their name in not important):
    .
    ├── sub-21
    │   ├── pred
    │   │   └── sub-21_pvs_map.nii.gz
    │   └── seg
    │       └── sub-21_brainparc.nii.gz
    ├── sub-51
    │   ├── pred
    │   │   └── sub-51_pvs_map.nii.gz
    │   └── seg
    ·       └── sub-51_brainparc.nii.gz
    
    """
    
    usage = """
    Usage examples:
    
    shiva_postproc --indir /path/to/input --outdir /path/to/output --segtype synthseg --pred PVS
    shiva_postproc --indir /path/to/input --outdir /path/to/output --segtype synthseg --pred WMH -cvt 0.5 -cst 10 
    shiva_postproc --indir /path/to/input --outdir /path/to/output --segtype custom --custom_lut /path/to/custom_lut.txt --pred CMB
    """

    parser = argparse.ArgumentParser(description=_DESCRIPTION, epilog=usage, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--indir', '-i', dest='in_dir', type=Path, required=True, help='Path to the input directory')
    parser.add_argument('--outdir', '-o', type=Path, required=True, help='Path to the output directory')
    parser.add_argument('--segtype', '-s', type=str, choices=['synthseg', 'freesurfer', 'brain_mask', 'custom'], required=True, help='Type of segmentation used')
    parser.add_argument('--pred', '-p', type=str, choices=['PVS', 'WMH', 'CMB', 'LAC'], required=True, help='Type of prediction to process')
    sub_lists_args = parser.add_mutually_exclusive_group()
    sub_lists_args.add_argument('--sub_list',
                                type=str,
                                required=False,
                                help=('Text file containing the list of participant IDs to be processed. The IDs must be '
                                      'the same as the ones given in the input folder. In the file, the IDs can be separated '
                                      'by a whitespace, a new line, or any of the following characters [ "," ";" "|" ] '
                                      '(or a combination of those). If none of --sub_list, --sub_names, or --exclusion_list '
                                      'are used, all the participants in the input folder will be processed'))

    sub_lists_args.add_argument('--sub_names',
                                nargs='+',
                                required=False,
                                help=('List of participant IDs to be processed. With this option, the IDs are given directly '
                                      'in the command line, separated by a white-space, and must be the same as the ones given '
                                      'in the input folder. If none of --sub_list, --sub_names, or --exclusion_list '
                                      'are used, all the participants in the input folder will be processed'))

    sub_lists_args.add_argument('--exclusion_list',
                                type=str,
                                required=False,
                                help=('Text file containing the list of participant IDs to NOT be processed. This option can be '
                                      'used when processing all the data in the input folder except for a few (because they have '
                                      'faulty data for exemple).\n'
                                      'In the file, the syntax is the same as for --sub_list\n.'
                                      'If none of --sub_list, --sub_names, or --exclusion_list '
                                      'are used, all the participants in the input folder will be processed'))

    parser.add_argument('--custom_lut', '-lut', type=Path, required=False, help='Path to the custom LUT file (required if segtype is custom)')
    parser.add_argument('--cluster_val_thr', '-cvt', type=float, default=0.5, help='Threshold value for cluster labelling (default: 0.5)')
    parser.add_argument('--cluster_size_thr', '-cst', type=int, default=1, help='Minimum cluster size (in voxels, in the cluster image space) for labelling (default: 1)')
    parser.add_argument('--run_plugin', '-rp', type=str, default='MultiProc', help='Nipype plugin to use for running the workflow (default: MultiProc, can be set to "SLURM" for cluster execution)')
    parser.add_argument('--run_plugin_args', 
                        type=str,
                        help=('Configuration file (.yml) for the plugin used by Nipype to run the workflow.\n'
                              'It will be imported as a dictionary and given plugin_args '
                              '(see https://nipype.readthedocs.io/en/0.11.0/users/plugins.html '
                              'for more details ). Alternatively to a .yml file, you can also give '
                              'a json string directly in the command line (e.g. \'{"arg1": val1, "arg2": val2}\').'))
    parser.add_argument('--debug', action='store_true', help='If set, the workflow will be run in debug mode (stop on first crash)')
    args = parser.parse_args()
    args = parse_sub_list(parser, args)

    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    kwargs = {
        'DATA_DIR': str(args.in_dir.resolve()),
        'BASE_DIR': str(outdir.resolve()),
        'BRAIN_SEG': args.segtype,
        'PREDICTION': args.pred,
        'SUBJECT_LIST': args.sub_list,
        'THRESHOLD': args.cluster_val_thr,
        'MIN_SIZE': args.cluster_size_thr,
    }
    if args.segtype == 'custom':
        if not args.custom_lut:
            raise parser.error('Using the "custom" segmentation with a LUT but no LUT file was given. Please provide a LUT file with --custom_lut')
        args.custom_lut = args.custom_lut.resolve()
        if not args.custom_lut.exists():
            raise parser.error(f'Using the "custom" segmentation with a LUT but the file given with '
                               f'--custom_lut was not found: {args.custom_lut}')
        kwargs['CUSTOM_LUT'] = parse_LUT(args.custom_lut)

    if args.run_plugin == 'MultiProc':
        run_plugin_args = {'n_procs': 8}
    elif args.run_plugin == 'SLURM':
        run_plugin_args = {'sbatch_args': '--cpus-per-task=8'}
    else:
        run_plugin_args = {}
    
    if args.run_plugin_args:
        args.run_plugin_args = parse_plugin_args(args.run_plugin_args)
        run_plugin_args = args.run_plugin_args

    workflow = genWorkflow(**kwargs)
    if args.debug:
        workflow.config['execution']['stop_on_first_crash'] = 'True'
    
    workflow.run(plugin=args.run_plugin, plugin_args=run_plugin_args)

if __name__ == '__main__':
    main()