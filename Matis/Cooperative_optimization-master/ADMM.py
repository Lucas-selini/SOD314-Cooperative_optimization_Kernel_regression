import numpy as np
import matplotlib.pyplot as plt
import pickle
import networkx as nx
from utils import kernel_matrix, kernel_im, compute_alpha, get_agents_from_pickle, grad_alpha, grad_alpha2, grad_alpha_v3


# ADMM Distributed Algorithm

def ADMM(mu, sigma, adjacency_matrix, y, x, selected_points, selected_points_agent, K, lr,Beta):
    # A is the constraint matrix of communication between nodes. (yij = yji)
    A = 
    B = 