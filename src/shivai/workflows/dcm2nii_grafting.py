
from nipype.pipeline.engine import Node, Workflow
from nipype.interfaces.dcm2nii import Dcm2niix
from shivai.interfaces.shiva import Dcm2niix_Singularity
from shivai.utils.misc import file_selector


def graft_dcm2nii(workflow: Workflow, **kwargs):
    """This function will interpose a dcm2nii node between the datagrabber
    and all the relevent connected nodes (mutate the workflow)

    Args:
        workflow (Workflow): Preprocessing workflow with a databrabber called "databrabber"

    Returns:
        Workflow: input workflow with the dcm2nii node added
    """
    # file selector
    datagrabber = workflow.get_node('datagrabber')
    # List storing the reconnections to avoid doing it while iterating on the graph edges
    reconnections = []
    dmc2nii_nodes = {}
    for _, connected_node, connection_dict in workflow._graph.out_edges(datagrabber, data=True):
        out_and_in = connection_dict['connect']
        for grab_out, node_in in out_and_in:
            if grab_out in ['img1', 'img2', 'img3']:  # Does not interpose dcm2nii for the "seg" output
                reconnections.append((grab_out, connected_node, node_in))
                if grab_out not in dmc2nii_nodes.keys():
                    if kwargs['CONTAINERIZE_NODES']:
                        dcm2nii = Node(Dcm2niix_Singularity(), f'dicom2nifti_{grab_out}')
                        dcm2nii.inputs.snglrt_bind = [
                            ('`pwd`', '`pwd`', 'rw'),
                            (kwargs['DATA_DIR'], kwargs['DATA_DIR'], 'rw'),
                        ]
                        dcm2nii.inputs.snglrt_image = kwargs['CONTAINER_IMAGE']
                    else:
                        dcm2nii = Node(Dcm2niix(), name=f'dicom2nifti_{grab_out}')
                    dcm2nii.inputs.anon_bids = True
                    dcm2nii.inputs.out_filename = 'converted_%p'
                    dmc2nii_nodes[grab_out] = dcm2nii

    # dmc2nii_nodes specific loop because multiple node_in can use the same dmc2nii
    for grab_out, dcm2nii in dmc2nii_nodes.items():
        workflow.connect(datagrabber, grab_out,
                         dcm2nii, 'source_dir')

    for grab_out, connected_node, node_in in reconnections:
        workflow.disconnect(datagrabber, grab_out,
                            connected_node, node_in)
        workflow.connect(dmc2nii_nodes[grab_out], ('converted_files', file_selector, kwargs['SWI_FILE_NUM']),
                         connected_node, node_in)
