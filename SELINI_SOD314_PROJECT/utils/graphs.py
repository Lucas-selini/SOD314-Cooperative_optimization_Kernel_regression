"""
We use the networkx python library to generate the graphs. 
The graphs are then converted to numpy arrays and normalized. 
The normalized graph is then returned. 
The following types of graphs are generated:
1. Fully connected graph
2. Line graph
3. Small world graph
4. Cycle graph
"""
import networkx as nx
import numpy as np

def graph_normalization(graph):
    """
    Normalize the given graph by row and column sums.

    Parameters:
    graph (numpy.ndarray): The input graph as a 2D numpy array.

    Returns:
    numpy.ndarray: The normalized graph as a 2D numpy array.
    """
    row_sums = np.sum(graph, axis=1)
    graph = graph / row_sums[:, np.newaxis]
    col_sums = np.sum(graph, axis=0)
    graph = graph / col_sums[np.newaxis, :]
    return graph

def generate_graph_fully_connected(a):
    """
    Generates a fully connected graph with 'a' nodes.

    Parameters:
    a (int): The number of nodes in the graph.

    Returns:
    numpy.ndarray: The adjacency matrix representing the fully connected graph.
    """
    return graph_normalization(np.ones((a, a)) / a)

def generate_graph_line(a):
    """
    Generates a line graph with 'a' number of nodes.

    Parameters:
    a (int): The number of nodes in the line graph.

    Returns:
    numpy.ndarray: The generated line graph.
    """
    line_graph = np.zeros((a, a))
    
    for i in range(a - 1):
        line_graph[i, i + 1] = 1
        line_graph[i + 1, i] = 1
        
    np.fill_diagonal(line_graph, 1)
    
    return graph_normalization(line_graph)

def generate_graph_small_world(a, p=0.1):
    """
    Generates a small-world graph using the Watts-Strogatz model.

    Parameters:
    - a (int): The number of nodes in the graph.
    - p (float): The probability of rewiring each edge.

    Returns:
    - numpy.ndarray: The adjacency matrix of the generated small-world graph.
    """
    small_world_graph = nx.watts_strogatz_graph(a, 2, p)
    small_world_graph = nx.to_numpy_array(small_world_graph)
    
    np.fill_diagonal(small_world_graph, 1)
    
    return graph_normalization(small_world_graph)