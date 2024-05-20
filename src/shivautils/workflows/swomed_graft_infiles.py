from nipype.pipeline.engine import Node, Workflow
from nipype.interfaces.utility import IdentityInterface


def graft_swomed_infiles(workflow: Workflow) -> Workflow:
    '''
    Changes the datagrabber from a processing workflow to and identity interface that gets the
    file paths directly givent to the pipeline in SWOMed
    Only to be used for non-synthseg wf, as synthseg has its own thing
    '''
    datagrabber = workflow.get_node('datagrabber')
    reconnections = []
    for _, connected_node, connection_dict in workflow._graph.out_edges(datagrabber, data=True):
        out_and_in = connection_dict['connect']
        for grab_out, node_in in out_and_in:
            reconnections.append((grab_out, connected_node, node_in))

    # Removing unused nodes
    workflow.remove_nodes([datagrabber])

    # Datagrabber replacement with swomed input
    files_plug = Node(
        IdentityInterface(
            fields=['subject_id', 'img1', 'img2', 'img3', 'seg'],  # subject_id is just a dummy argument
            mandatory_inputs=False),
        name='datagrabber')

    for grabber_out, connected_node, node_in in reconnections:
        workflow.connect(files_plug, grabber_out,
                         connected_node, node_in)

    return workflow
