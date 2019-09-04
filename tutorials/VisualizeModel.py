####################################################
# Visialization work JUPITER NOTEBOOK in CHROME ONLY
####################################################
# tvm, relay
import tvm
from tvm import relay

# os and numpy
import numpy as np
import os.path

# Tensorflow imports
import tensorflow as tf

# Tensorflow utility functions
import tvm.relay.testing.tf as tf_testing

from IPython.display import clear_output, Image, display, HTML

########## Model Files #############
img_path = 'C:\\Users\\SVT\\.tvm_test_data\\data\\elephant-299.jpg'
model_path = "c:\\Users\SVT\\.tvm_test_data\\tf\\InceptionV1\\classify_image_graph_def-with_shapes.pb"
map_proto_path = "C:\\Users\\SVT\\.tvm_test_data\\data\\imagenet_2012_challenge_label_map_proto.pbtxt"
label_path = "C:\\Users\\SVT\\.tvm_test_data\\data\\imagenet_synset_to_human_label_map.txt"

######################################################################
print(" =!=!=!= Import model")
# ------------
print(" ==== Creates tensorflow graph definition from protobuf file.")
xgraph_def = tf.compat.v1.GraphDef()

with tf.io.gfile.GFile(model_path, 'rb') as f:
    graph_def = tf.compat.v1.GraphDef()
    graph_def.ParseFromString(f.read())
    graph = tf.import_graph_def(graph_def, name='')
    print(" ==== Call the utility to import the graph definition into default graph.")
    graph_def = tf_testing.ProcessGraphDefParam(graph_def)
    print(" ==== Add shapes to the graph.")
    with tf.compat.v1.Session() as sess:
        graph_def = tf_testing.AddShapesToGraphDef(sess, 'softmax')


node_count = 0
for node in graph_def.node:
    node_count += 1


print("nodes: {}".format(node_count) )
##############################################################################################################
def strip_consts(graph_def, max_const_size=32):
    """Strip large constant values from graph_def."""
    strip_def = tf.GraphDef()
    for n0 in graph_def.node:
        n = strip_def.node.add() 
        n.MergeFrom(n0)
        if n.op == 'Const':
            tensor = n.attr['value'].tensor
            size = len(tensor.tensor_content)
            if size > max_const_size:
                tensor.tensor_content = bytes("<stripped %d bytes>"%size, "utf-8")
    return strip_def

def show_graph(graph_def, max_const_size=32):
    """Visualize TensorFlow graph."""
    if hasattr(graph_def, 'as_graph_def'):
        graph_def = graph_def.as_graph_def()
    strip_def = strip_consts(graph_def, max_const_size=max_const_size)
    code = """
        <script>
          function load() {{
            document.getElementById("{id}").pbtxt = {data};
          }}
        </script>
        <link rel="import" href="https://tensorboard.appspot.com/tf-graph-basic.build.html" onload=load()>
        <div style="height:600px">
          <tf-graph-basic id="{id}"></tf-graph-basic>
        </div>
    """.format(data=repr(str(strip_def)), id='graph'+str(np.random.rand()))

    iframe = """
        <iframe seamless style="width:1200px;height:620px;border:0" srcdoc="{}"></iframe>
    """.format(code.replace('"', '&quot;'))
    display(HTML(iframe))


show_graph(tf.get_default_graph().as_graph_def())
##======================================================================
import pydot
from itertools import chain
def tf_graph_to_dot(in_graph):
    dot = pydot.Dot()
    dot.set('rankdir', 'LR')
    dot.set('concentrate', True)
    dot.set_node_defaults(shape='record')
    all_ops = in_graph.get_operations()
    all_tens_dict = {k: i for i,k in enumerate(set(chain(*[c_op.outputs for c_op in all_ops])))}
    for c_node in all_tens_dict.keys():
        node = pydot.Node(c_node.name)#, label=label)
        dot.add_node(node)
    for c_op in all_ops:
        for c_output in c_op.outputs:
            for c_input in c_op.inputs:
                dot.add_edge(pydot.Edge(c_input.name, c_output.name))
    return dot


from IPython.display import SVG
# Define model
# tf_graph_to_dot(tf.get_default_graph()).write_svg('simple_tf.svg')
# SVG('simple_tf.svg')
