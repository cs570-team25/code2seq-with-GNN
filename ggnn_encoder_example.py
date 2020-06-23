
import argparse
import numpy as np
import networkx as nx
import tensorflow as tf
import opengnn as ognn

import pickle

BATCH_SIZE = 1
NODE_FEATURE_SIZE = 256

"""
Example of embedding one method graph using GGNN encoder
"""


def main(args):
    method_embeddings = generate_embeddings(args.target)


def generate_embeddings(filename):
    with open(filename, 'rb') as f:
        method_graphs = pickle.load(f)

    def aggregate(node_embeddings):
        # TODO: aggregate set of node embeddings to the single graph embedding
        return node_embeddings

    method_embeddings = dict()
    for name in method_graphs:
        tf.reset_default_graph()
        method_graph = method_graphs[name]
        edges, node_features, graph_sizes = process_one_method(method_graph)

        encoder = ognn.encoders.GGNNEncoder(BATCH_SIZE, NODE_FEATURE_SIZE)
        output, state = encoder(edges, node_features, graph_sizes)

        print(output)

        output = aggregate(output)
        method_embeddings[name] = output

    return method_embeddings


def process_one_method(method_graph):
    num_nodes = len(method_graph.nodes)

    # step 1: construct sparse tensor denoting all edges

    edge_labels = dict()
    for e in method_graph.edges:
        label = method_graph.edges[e]['label']
        if label not in edge_labels:
            edge_labels[label] = [e]
        else:
            edge_labels[label].append(e)

    num_edge_types = len(edge_labels.keys())

    indices = []
    label_index = 0
    for label in edge_labels:
        edges = edge_labels[label]

        indices.extend([[label_index, 0, e[0], e[1]] for e in edges])

        label_index += 1

    print('# of nodes: ', num_nodes)
    print('# of edge types: ', num_edge_types)
    print('# of edges: ', len(indices))

    values = [1] * len(indices)
    dense_shape = [0, num_edge_types, num_nodes, num_nodes]
    edges = tf.SparseTensor(
        indices=indices,
        values=values,
        dense_shape=dense_shape)

    # step 2: construct node_features with random initialization

    node_features = tf.random_uniform(
        (BATCH_SIZE, num_nodes, NODE_FEATURE_SIZE))
    graph_sizes = [BATCH_SIZE, num_edge_types, num_nodes]

    return edges, node_features, graph_sizes


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Digest Graphs for GNNs.')
    parser.add_argument('--target', type=str,
                        help='Pickled dictionary containing method graphs')
    args = parser.parse_args()

    main(args)
