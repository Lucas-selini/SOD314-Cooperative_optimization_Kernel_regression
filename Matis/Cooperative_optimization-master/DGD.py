import numpy as np
import matplotlib.pyplot as plt
import pickle
import networkx as nx
from utils import kernel_matrix, kernel_im, compute_alpha, get_agents_from_pickle, grad_alpha, grad_alpha2, grad_alpha_v3


def DGD_revisited(mu, sigma, a, adjacency_matrix, y_agent, x_agent, x_selected, alpha_opt, lr):
    # stacked points
    alpha = np.zeros((a, len(x_selected)))
    # initial alpha
    alpha = alpha.reshape(a*len(x_selected), 1)
    n = len(x_selected)
    W = 1.0/(a)*(adjacency_matrix)  # define the weight matrix
    # define the kronecker product of the weight matrix
    W_bar = np.kron(W, np.eye(n))
    j = 0
    optimal_gap = []
    while j < 50000:
        j += 1
        alpha = W_bar @ alpha - lr * \
            np.array(grad_alpha(sigma, mu, y_agent, x_agent,
                                x_selected, alpha.reshape(a, n))).reshape(a*n, 1)
        optimal_gap.append(np.linalg.norm(
            alpha.reshape(a, n).mean(axis=0)-alpha_opt))

    return optimal_gap


def DGD_revisited_v2(mu, sigma, adjacency_matrix, y, x, selected_points, selected_points_agent, K, lr):
    # stacked points
    a = len(selected_points_agent)
    n = len(selected_points)
    alpha = np.zeros((a, n))
    # initial alpha
    alpha = alpha.reshape(a*n, 1)
    W = 1/(a)*(adjacency_matrix)  # define the weight matrix
    # define the kronecker product of the weight matrix
    W_bar = np.kron(W, np.eye(n))
    j = 0
    while j < 5000:
        j += 1
        g = grad_alpha_v3(sigma, mu, x, y, alpha.reshape(
            a, n), K, selected_points, selected_points_agent)
        alpha = W_bar @ alpha - lr * np.array(g).reshape(a*n, 1)
    return alpha


if __name__ == "__main__":
    with open('first_database.pkl', 'rb') as f:
        x, y = pickle.load(f)
    # Generate the data
    a = 5
    n = 100
    m = 10
    agent_x, agent_y, x_selected, y_selected, selected_points, selected_points_agents, K, x, y = get_agents_from_pickle(
        'first_database.pkl', 5, 100, 10)
    print(x_selected.shape)
    print(x_selected)
    mu = 1
    sigma = 0.5
    lr = 0.001
    Kmm = kernel_matrix(x_selected, x_selected)
    Knm = kernel_matrix(x[0:n], x_selected)
    # alpha_optim using cvxpy
    # Define the optimization variable
    alpha_exact = np.linalg.inv(
        sigma**2*Kmm + np.eye(m) + np.transpose(Knm) @ Knm) @ np.transpose(Knm) @ y[0:n]
    print("the alpha exact is :", alpha_exact)
    # define the graph connection of the agents (a undirected star graph for now):
    # Gx = nx.complete_graph(a).to_undirected()
    # nx.draw(Gx, with_labels=True)
    # plt.show()
    Adj = np.ones((5, 5))
    print(Adj)
    alpha_dgd = DGD_revisited(
        mu, sigma, a, Adj, agent_y, agent_x, x_selected, alpha_exact, lr)
    print(alpha_dgd.reshape(a, m).mean(axis=0))
