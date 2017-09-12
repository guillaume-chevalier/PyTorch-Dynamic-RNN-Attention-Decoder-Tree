
from html_tools import nn_tree_to_html

import torch

import numpy as np

import random


def concat(*x):
    """
    Batch-flatten (flatten all except batch dimension and neurons dimension)
    and then concatenate *x on neuron dimensions.
    """
    return *x

def fc_layer(x):
    """Basic Fully Connected (FC) layer with an activation function."""
    return x

def concat_fc(*x):
    """Input all *x to a FC layer."""
    return fc_layer(concat(*x))

def rnn(rnn_instance, attention_in, state):
    """Call the RNN for advancing of 1 time step."""
    # in: attention
    # state: state
    do_continue = random.random()

    # call rnn instance on attention input and state to generate outputs,
    # continue, and recurse flags

    return do_continue, will_recurse, out, new_state

# First parent_state: an FC from the cnn_z
# Second parent state and others: different FC from parent node's output
def generate_run_rnn(rnn_instance, cnn_z, parent_state, max_length, remaining_depth):
    """
    This call is recursive as it generates a tree from an RNN that decodes the "parent_state".
    """

    do_continue = 1.0
    remaining_length = max_length

    outputs = []
    states = []
    recurses = []

    # This recursively contains the 3 previous lists and itself for childrens:
    childs_tree = []

    # Loop forward pass RNN
    while do_continue > 0.5 and remaining_length > 0:
        attention = soft_attention(cnn_z, prev_state, parent_state)
        do_continue, will_recurse, output, state = rnn(
            rnn_instance, attention, state
        )

        # Call children recurse
        if will_recurse > 0.5 and remaining_depth > 0:
            # The following line may be replaced by an RNN
            # as it theorically unfolds through depth of the tree:
            child_context = concat_fc(parent_state, attention, output, state)

            child = generate_run_rnn(
                rnn_instance, cnn_z, parent_state, max_length, remaining_depth - 1
            )
            childs_tree.append(childs)

        outputs.append(output)
        states.append(state)
        recurses.append(will_recurse)

        remaining_length -= 1

    return [
        outputs
        states
        recurses
        childs_tree
    ]

# The previous method will also need static versions depending on the
# training data itself so as to build a valid loss function or error metric:
# def train_run_rnn(...)
# def test_run_rnn(...)

def run_rnn_tree(cnn_z):
    """
    From cnn_z (CNN feature map as encoded image), generate HTML code
    with an RNN Decoder Tree. We also need the train-time and test-time
    version of that function, which are not generative, but tied to the test
    data for having a valid loss function for supervised learning.
    """
    rnn_instance = torch.rnn()

    # Note: first (parent) state is computed from cnn_z (feature map).
    childs_tree = generate_run_rnn(
        rnn_instance, cnn_z, fc_layer(cnn_z), max_length=7, remaining_depth=4
    )

    return nn_tree_to_html(childs_tree)


# Call this after the CNN input:
run_rnn_tree(cnn_z)
