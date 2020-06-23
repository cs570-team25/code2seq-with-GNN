import networkx as nx
import graph_pb2
import argparse

import pickle

def read_proto(file_path):
    '''Takes proto file path and returns networkx MultiDiGraph.'''
    # read graph from proto file
    input_graph = graph_pb2.Graph()
    with open(file_path, 'rb') as f:
        input_graph.ParseFromString(f.read())
    edge_dict = {v:k for k, v in input_graph.edge[0].EdgeType.items()}
    
    # convert custom graph to networkx graph
    nx_graph = nx.MultiDiGraph()
    for node in input_graph.node:
        nx_graph.add_node(node.id, label=node.contents)
    for edge in input_graph.edge:
        nx_graph.add_edge(edge.sourceId, edge.destinationId, label=edge_dict[edge.type])
    
    return nx_graph

def extract_methods(nx_graph, reverse_edges=True):
    '''Extract separate method graphs from networkx graph.
    When reverse=True, adds reverse edges, as the GGNN paper did.'''
    method_nodes = [x for x, y in nx_graph.nodes(data=True) 
                    if y['label'] == 'METHOD']
    allowed_edges = ['AST_CHILD', 'ASSOCIATED_TOKEN']
    method_graphs = {}
    
    for root_node in method_nodes:
        ## phase 0. find name of method
        name = None
        for r_e in nx_graph.out_edges(root_node, keys=True):
            dest_node = r_e[1]
            if nx_graph.nodes[dest_node]['label'] == 'NAME':
                name_out_edges = list(nx_graph.out_edges(dest_node, keys=True))
                assert len(name_out_edges) == 1
                name_edge = name_out_edges[0]
                name_node = name_edge[1] # destination node of name edge
                name = nx_graph.nodes[name_node]['label']
                break
        assert name is not None
        
        ## phase 1. identifying nodes that are part of this method
        # find direct descendants
        included_nodes = set([root_node])
        search_queue = [root_node]
        while len(search_queue) != 0:
            search_node = search_queue.pop(0)
            outgoing_edges = nx_graph.out_edges(search_node, keys=True)

            for e in outgoing_edges:
                edge_dict = nx_graph.edges[e]
                dest_node = e[1]
                if edge_dict['label'] in allowed_edges and dest_node not in included_nodes:
                    search_queue.append(dest_node)
                    included_nodes.add(dest_node)

        # find nodes that are only connected to method-nodes
        related_nodes = set()
        for node in nx_graph.nodes:
            if node in included_nodes:
                continue
            in_conn = list(map(lambda x: x[1], nx_graph.in_edges(node)))
            out_conn = list(map(lambda x: x[1], nx_graph.out_edges(node)))
            all_conn = in_conn + out_conn
            if all(map(lambda x: x in included_nodes, all_conn)):
                related_nodes.add(node)

        all_method_nodes = related_nodes | included_nodes
        
        ## phase 2. Construct graph without edges to external nodes
        method_graph = nx.MultiDiGraph()
        for node in all_method_nodes: # first add all nodes, to migrate information
            method_graph.add_node(node, label=nx_graph.nodes[node]['label'])

        for node in all_method_nodes:
            outgoing_edges = nx_graph.out_edges(node, keys=True)
            for e in outgoing_edges:
                dest_node = e[1]
                if dest_node in all_method_nodes:
                    method_graph.add_edge(e[0], e[1], label=nx_graph.edges[e]['label'])
                    if reverse_edges:
                        method_graph.add_edge(e[1], e[0], label=nx_graph.edges[e]['label']+'_REVERSE')

        method_graphs[name] = method_graph
    return method_graphs

def main(args):
    nx_graph = read_proto(args.proto_file)
    method2graph = extract_methods(nx_graph)
    with open('method2graph.pkl', 'wb+') as f:
        pickle.dump(method2graph, f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Digest Graphs for GNNs.')
    parser.add_argument('--proto_file', type=str, help='Proto file to process.')
    args = parser.parse_args()
    main(args)
    
