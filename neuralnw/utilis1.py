import pandas as pd
import networkx as nx
import json

def read_graph(input_path):
    """
    The function reads the graph from input path.
    :param input_path: Graph read into memory.
    :return G: Networkx graph.
    """
    edges = pd.read_csv(input_path)
    edge_list=edges.values.tolist()

    """
    remove self loops
    """
    for e in edge_list:
        if len(set(e))==len(e):
            continue
        else:
            edge_list.remove(e)
        
    G = nx.from_edgelist(edge_list)
    return G

def json_read(path):
    """
    Function to read json file of seed and activation sequences.

    :param path: Path for dumping the JSON.
    """
    with open(path, 'r') as myfile:
          data=myfile.read()
    obj = json.loads(data)
    return obj

def json_dumper(data, path):
    """
    Function to save a JSON to disk.
    :param data: Dictionary of seed set and corresponding activations.
    :param path: Path for dumping the JSON.
    """
    with open(path, 'a') as outfile:
        json.dump(data, outfile)
        outfile.write(',')
        outfile.write('\n')
