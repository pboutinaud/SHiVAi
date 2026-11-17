import argparse
import csv
from pathlib import Path

import pytest
from nipype.interfaces.io import DataGrabber
from nipype.interfaces.utility import Function, IdentityInterface
from nipype.pipeline.engine import Node, Workflow

from shivai.interfaces.datasink import DataSink_CSV_and_PDF_safe, GirderSink
from shivai.utils.parsing import parse_sub_list
from shivai.workflows.main_workflow import (
    generate_main_wf,
    generate_main_wf_grab_preproc,
    generate_main_wf_rerun_postproc,
    update_wf_grabber,
)


@pytest.fixture
def bids_tree(tmp_path):
    (tmp_path / 'dataset_description.json').write_text('{}')
    (tmp_path / 'participants.tsv').touch()
    (tmp_path / 'derivatives' / 'anat').mkdir(parents=True)
    for session in ('ses-AB123', 'ses-CD456_adhoc'):
        (tmp_path / 'sub-01' / session / 'anat').mkdir(parents=True)
    return tmp_path


@pytest.mark.parametrize('selection', ['all', 'subject', 'session', 'file', 'exclude'])
def test_bids_session_selection_preserves_folder_names(bids_tree, selection):
    args = argparse.Namespace(
        in_dir=str(bids_tree), input_type='BIDS',
        sub_list=None, sub_names=None, exclusion_list=None,
    )
    expected = ['sub-01/ses-AB123', 'sub-01/ses-CD456_adhoc']
    if selection == 'subject':
        args.sub_names = ['sub-01']
    elif selection == 'session':
        args.sub_names = ['sub-01/ses-CD456_adhoc']
        expected = ['sub-01/ses-CD456_adhoc']
    elif selection == 'file':
        subjects = bids_tree / 'subjects.txt'
        subjects.write_text('sub-01/ses-CD456_adhoc')
        args.sub_list = str(subjects)
        expected = ['sub-01/ses-CD456_adhoc']
    elif selection == 'exclude':
        exclusions = bids_tree / 'exclude.txt'
        exclusions.write_text('sub-01/ses-CD456_adhoc')
        args.exclusion_list = str(exclusions)
        expected = ['sub-01/ses-AB123']

    assert parse_sub_list(argparse.ArgumentParser(), args).sub_list == expected


def test_bids_without_participants_reports_parser_error(tmp_path, capsys):
    (tmp_path / 'dataset_description.json').write_text('{}')
    args = argparse.Namespace(
        in_dir=str(tmp_path), input_type='BIDS',
        sub_list=None, sub_names=None, exclusion_list=None,
    )

    with pytest.raises(SystemExit) as exc:
        parse_sub_list(argparse.ArgumentParser(), args)

    assert exc.value.code == 2
    assert 'No participants' in capsys.readouterr().err


@pytest.mark.parametrize('session', [None, 'ses-01', 'ses-AB123', 'ses-CD456_adhoc'])
@pytest.mark.parametrize('grabber_name', ['datagrabber', 'preproc_grabber', 'prev_res_grabber'])
def test_bids_grabber_resolves_only_selected_session(tmp_path, session, grabber_name):
    subject_id = 'sub-01' if session is None else f'sub-01/{session}'
    prefix = subject_id.replace('/', '_')
    anat = tmp_path / subject_id / 'anat'
    anat.mkdir(parents=True)
    expected = {}
    for field, modality in [('img1', 'T1w'), ('img2', 'FLAIR'), ('img3', 'SWI'), ('seg', 'seg')]:
        image = anat / f'{prefix}_{modality}.nii.gz'
        image.touch()
        expected[field] = str(image)
    other_anat = tmp_path / 'sub-01' / 'ses-other' / 'anat'
    other_anat.mkdir(parents=True)
    (other_anat / 'sub-01_ses-other_T1w.nii.gz').touch()

    wf = Workflow(name='test_bids_grabber')
    grabber = Node(
        DataGrabber(infields=['subject_id'], outfields=list(expected)),
        name=grabber_name,
    )
    grabber.inputs.base_directory = str(tmp_path)
    grabber.inputs.template = '*'
    grabber.inputs.sort_filelist = True
    grabber.inputs.subject_id = subject_id
    wf.add_nodes([grabber])
    kwargs = {'PREP_SETTINGS': {'input_type': 'BIDS'}, 'BRAIN_SEG': 'custom'}

    update_wf_grabber(
        wf, [('img1', 't1'), ('img2', 'flair'), ('img3', 'swi')], 'nifti', kwargs,
        grabber_name=grabber_name,
    )

    outputs = grabber.interface.run().outputs
    for field, image in expected.items():
        assert Path(getattr(outputs, field)) == Path(image)


def test_datasinks_keep_sessions_separate_and_match_girder_metadata(tmp_path):
    results = tmp_path / 'results'
    destinations = []
    for subject_id in ('sub-02', 'sub-01/ses-AB123', 'sub-01/ses-CD456_adhoc'):
        source = tmp_path / 'work' / subject_id / 'brainmask_cropped.nii.gz'
        source.parent.mkdir(parents=True)
        source.write_text(subject_id)
        sink = DataSink_CSV_and_PDF_safe(
            base_directory=str(results), subject_path=subject_id,
            substitutions=[('_subject_id_', '')],
        )
        setattr(sink.inputs, 'shiva_preproc.t1_preproc', str(source))

        destination = Path(sink.run().outputs.out_file[0])

        expected = Path('shiva_preproc') / 't1_preproc' / subject_id / source.name
        assert destination == results / expected
        destinations.append(destination)
        remote = GirderSink(
            base_directory=str(results), subject_id=subject_id, subject_path=subject_id,
        )
        assert Path(remote._local_relative_path('shiva_preproc.t1_preproc', str(source))) == expected

    assert [path.read_text() for path in destinations] == [
        'sub-02', 'sub-01/ses-AB123', 'sub-01/ses-CD456_adhoc',
    ]


def test_summary_sink_does_not_add_routing_subject_to_local_path(tmp_path):
    source = tmp_path / 'metrics.csv'
    source.write_text('sub_id,value\nsub-01/ses-AB123,1\n')
    sink = DataSink_CSV_and_PDF_safe(
        base_directory=str(tmp_path / 'results'), container='results_summary',
    )
    setattr(sink.inputs, 'segmentations.pvs_metrics', str(source))
    destination = Path(sink.run().outputs.out_file[0])
    remote = GirderSink(
        base_directory=sink.inputs.base_directory, container='results_summary',
        subject_id='results_summary',
    )

    assert Path(remote._local_relative_path('segmentations.pvs_metrics', str(source))) == \
        destination.relative_to(tmp_path / 'results')
    assert destination.read_text() == source.read_text()


def test_csv_append_is_confined_to_the_selected_session(tmp_path):
    results = tmp_path / 'results'
    for run, session in enumerate(('ses-AB123', 'ses-CD456_adhoc', 'ses-AB123')):
        source = tmp_path / f'run-{run}' / 'metrics.csv'
        source.parent.mkdir()
        source.write_text(f'value\n{run}\n')
        sink = DataSink_CSV_and_PDF_safe(
            base_directory=str(results), subject_path=f'sub-01/{session}',
        )
        setattr(sink.inputs, 'segmentations.pvs_segmentation', str(source))
        sink.run()

    for session, values in [('ses-AB123', ['0', '2']), ('ses-CD456_adhoc', ['1'])]:
        output = results / 'segmentations' / 'pvs_segmentation' / 'sub-01' / session / 'metrics.csv'
        with output.open() as stream:
            assert [row['value'] for row in csv.DictReader(stream)] == values


def write_session_result(subject_id):
    from pathlib import Path
    result = Path('brainmask_cropped.nii.gz')
    result.write_text(subject_id)
    return str(result.resolve())


def test_iterated_workflow_saves_both_sessions_without_overwriting(tmp_path):
    pytest.importorskip('pwd', reason="Nipype's workflow plugin loader requires POSIX pwd")
    subjects = ['sub-01/ses-AB123', 'sub-01/ses-CD456_adhoc']
    workflow = Workflow(name='session_outputs', base_dir=str(tmp_path / 'work'))
    iterator = Node(IdentityInterface(fields=['subject_id']), name='subjects')
    iterator.iterables = ('subject_id', subjects)
    writer = Node(Function(
        input_names=['subject_id'], output_names=['out_file'], function=write_session_result,
    ), name='writer')
    results = tmp_path / 'results'
    sink = Node(DataSink_CSV_and_PDF_safe(base_directory=str(results)), name='sink')
    workflow.connect([
        (iterator, writer, [('subject_id', 'subject_id')]),
        (iterator, sink, [('subject_id', 'subject_path')]),
        (writer, sink, [('out_file', 'shiva_preproc.t1_preproc')]),
    ])

    workflow.run(plugin='Linear')

    for subject_id in subjects:
        output = results / 'shiva_preproc' / 't1_preproc' / subject_id / 'brainmask_cropped.nii.gz'
        assert output.read_text() == subject_id


@pytest.fixture
def session_results(tmp_path):
    subjects = ['sub-01/ses-AB123', 'sub-01/ses-CD456_adhoc', 'sub-02']
    rawdata = tmp_path / 'rawdata'
    results = tmp_path / 'previous' / 'results'
    for subject_id in subjects:
        anat = rawdata / subject_id / 'anat'
        anat.mkdir(parents=True)
        (anat / f"{subject_id.replace('/', '_')}_T1w.nii.gz").touch()
        (anat / 'aparc+aseg.nii.gz').touch()
        source = tmp_path / 'work' / subject_id
        source.mkdir(parents=True)
        sink = DataSink_CSV_and_PDF_safe(
            base_directory=str(results), subject_path=subject_id,
        )
        for field, name in [
            ('shiva_preproc.t1_preproc', 't1_cropped_intensity_normed.nii.gz'),
            ('shiva_preproc.t1_preproc.@mask', 'brainmask_cropped.nii.gz'),
            ('shiva_preproc.qc_metrics', 'qc_metrics.csv'),
            ('shiva_preproc.freesurfer', 'derived_parc.nii.gz'),
            ('shiva_preproc.freesurfer.@mask', 'brainmask_no_csf.nii.gz'),
            ('shiva_preproc.synthseg', 'derived_parc.nii.gz'),
            ('shiva_preproc.synthseg.@mask', 'brainmask_no_csf.nii.gz'),
            ('segmentations.pvs_segmentation', f'{Path(subject_id).name}_pvs_map.nii.gz'),
            ('segmentations.pvs_segmentation.@foldwise', f'{Path(subject_id).name}_pvs_map_by_fold.nii.gz'),
        ]:
            image = source / name
            image.write_text('value\n1\n' if image.suffix == '.csv' else '')
            setattr(sink.inputs, field, str(image))
        sink.run()

    descriptor = tmp_path / 'descriptor.json'
    descriptor.write_text('{}')
    output = tmp_path / 'output'
    output.mkdir()
    kwargs = {
        'BASE_DIR': str(output), 'DATA_DIR': str(rawdata), 'SUBJECT_LIST': subjects,
        'PREP_SETTINGS': {
            'input_type': 'BIDS', 'file_type': 'nifti', 'prereg_flair': False,
            'preproc_res': str(results / 'shiva_preproc'), 'prev_res': str(results),
            'prev_qc': None, 'preproc_only': False, 'affine_reset': False,
        },
        'PREDICTION': ['PVS'], 'BRAIN_SEG': 'premasked', 'CUSTOM_LUT': None,
        'ACQUISITIONS': {'t1-like': None, 'flair-like': None, 'swi-like': None},
        'PVS_DESCRIPTOR': str(descriptor), 'MODELS_PATH': str(tmp_path),
        'CONTAINERIZE_NODES': False, 'IMAGE_SIZE': (16, 16, 16),
        'RESOLUTION': (1., 1., 1.), 'TOLERANCE': (0., 0., 0.), 'ORIENTATION': 'RAS',
        'AFFINE_CORREC_THRESHOLD': 0.005, 'BATCH_SIZE': 1, 'PRED_PLUGIN_ARGS': {},
        'GPU': None, 'ANONYMIZED': False, 'PERCENTILE': 99., 'THRESHOLD': 0.5,
        'BCG_RATIO': 0.25, 'DB': None, 'SAVE_GRAPH': False,
        'GIRDER_ENABLED': True, 'SHIVAI_VERSION': 'test', 'GIRDER_MAPPING': {},
        'GIRDER_HOST': 'https://girder.example.com/api/v1', 'GIRDER_AUTH_METHOD': 'api_key',
    }
    for pred in ('PVS', 'WMH', 'CMB', 'LAC'):
        kwargs[f'THRESHOLD_{pred}'] = 0.5
        kwargs[f'MIN_{pred}_SIZE'] = 1
    return kwargs


@pytest.mark.parametrize('factory,grabber_name', [
    (generate_main_wf_grab_preproc, 'preproc_grabber'),
    (generate_main_wf_rerun_postproc, 'prev_res_grabber'),
])
@pytest.mark.parametrize('brain_seg', ['premasked', 'fs_precomp', 'synthseg_precomp'])
def test_session_results_can_be_read_back(session_results, factory, grabber_name, brain_seg):
    session_results['BRAIN_SEG'] = brain_seg
    workflow = factory(**session_results)
    grabber = workflow.get_node(grabber_name)
    results = Path(session_results['PREP_SETTINGS']['prev_res'])

    for subject_id in session_results['SUBJECT_LIST']:
        grabber.inputs.subject_id = subject_id
        outputs = grabber.interface.run().outputs
        assert Path(outputs.brain_mask) == \
            results / 'shiva_preproc' / 't1_preproc' / subject_id / 'brainmask_cropped.nii.gz'
        if brain_seg != 'premasked':
            kind = 'freesurfer' if brain_seg == 'fs_precomp' else 'synthseg'
            assert Path(outputs.brain_seg) == \
                results / 'shiva_preproc' / kind / subject_id / 'derived_parc.nii.gz'
        if grabber_name == 'prev_res_grabber':
            for field, suffix in [('pvs_segmentation', ''), ('pvs_prediction_foldwise', '_by_fold')]:
                assert Path(getattr(outputs, field)) == \
                    results / 'segmentations' / 'pvs_segmentation' / subject_id / \
                    f'{Path(subject_id).name}_pvs_map{suffix}.nii.gz'


@pytest.mark.parametrize('factory', [
    generate_main_wf, generate_main_wf_grab_preproc, generate_main_wf_rerun_postproc,
])
def test_all_workflow_modes_wire_explicit_subject_paths(session_results, factory):
    workflow = factory(**session_results)
    iterator = workflow.get_node('subject_iterator')
    for name in ('sink_node_subjects', 'girder_sink_node_subjects'):
        sink = workflow.get_node(name)
        connections = workflow._graph.get_edge_data(iterator, sink)['connect']
        assert ('subject_id', 'subject_path') in connections
    for name in ('sink_node_all', 'girder_sink_node_all'):
        sink = workflow.get_node(name)
        assert Path(sink.inputs.base_directory) == Path(session_results['BASE_DIR']) / 'results'
        assert sink.inputs.container == 'results_summary'


@pytest.mark.parametrize('factory', [generate_main_wf_grab_preproc, generate_main_wf_rerun_postproc])
def test_readback_reports_a_missing_session_even_if_subject_exists(session_results, factory):
    missing = 'sub-01/ses-missing'
    session_results['SUBJECT_LIST'].append(missing)

    with pytest.raises(ValueError) as exc:
        factory(**session_results)

    if factory is generate_main_wf_grab_preproc:
        missing_file = Path(session_results['BASE_DIR']) / 'missing_datasets.txt'
        assert missing_file.read_text().splitlines() == [missing]
    else:
        assert missing in str(exc.value)
