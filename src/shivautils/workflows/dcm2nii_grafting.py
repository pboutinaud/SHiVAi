
from nipype.pipeline.engine import Node, Workflow
from nipype.interfaces.dcm2nii import Dcm2niix
from shivautils.interfaces.shiva import Dcm2niix_Singularity

def graft_dcm2nii(workflow: Workflow, **kwargs) -> Workflow:
    """This function will interpose a dcm2nii node between the datagrabber
    and all the relevent connected nodes

    Args:
        workflow (Workflow): Preprocessing workflow with a databrabber called "databrabber"

    Returns:
        Workflow: input workflow with the dcm2nii node added
    """
    # file selector
    datagrabber = workflow.get_node('datagrabber')
    # List storing the reconnections to avoid doing it while iterating on the graph edges
    reconnections = []
    for _, connected_node, connection_dict in workflow._graph.out_edges(datagrabber, data=True):
        out_and_in = connection_dict['connect']
        for grab_out, node_in in out_and_in:
            if grab_out in ['img1', 'img2', 'img3']:  # Does not interpose dcm2nii for the "seg" output
                if kwargs['CONTAINERIZE_NODES']:
                    dcm2nii = Node(Dcm2niix_Singularity(), f'dicom2nifti_{grab_out}')
                    dcm2nii.inputs.snglrt_bind = [
                        (kwargs['BASE_DIR'], kwargs['BASE_DIR'], 'rw'),
                        ('`pwd`', '`pwd`', 'rw'),]
                    dcm2nii.inputs.snglrt_image = kwargs['CONTAINER_IMAGE']
                else:
                    dcm2nii = Node(Dcm2niix(), name=f'dicom2nifti_{grab_out}')
                
                dcm2nii.inputs.anon_bids = True
                dcm2nii.inputs.out_filename = 'converted_%p'
                reconnections.append((grab_out, dcm2nii, connected_node, node_in))

    for grab_out, dcm2nii, connected_node, node_in in reconnections:
        workflow.disconnect(datagrabber, grab_out, connected_node, node_in)
        workflow.connect(datagrabber, grab_out,
                         dcm2nii, 'source_dir')
        workflow.connect(dcm2nii, 'converted_files',
                         connected_node, node_in)
    
    return workflow