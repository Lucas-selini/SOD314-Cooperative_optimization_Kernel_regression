from utils.fct import compute_kernel_matrix
import numpy as np

def dual_decomposition(X, X_selected, Y, sigma2, nu, a, n_epochs, step_size, alpha_star, W, lamb0=0):
    """
    Performs dual decomposition optimization algorithm.

    Args:
        X (numpy.ndarray): The input data matrix of shape (n, d).
        X_selected (numpy.ndarray): The selected data matrix of shape (m, d).
        Y (numpy.ndarray): The target values of shape (n,).
        sigma2 (float): The noise variance.
        nu (float): The regularization parameter.
        a (int): The number of agents.
        n_epochs (int): The number of epochs.
        step_size (float): The step size for updating lambda_ij.
        alpha_star (numpy.ndarray): The optimal alpha values of shape (a, m).
        W (numpy.ndarray): The weight matrix of shape (m, m).
        lamb0 (float, optional): The initial value for lambda_ij. Defaults to 0.

    Returns:
        tuple: A tuple containing:
            - optimality_gaps (list): A list of lists containing the optimality gaps for each agent at each epoch.
            - alpha_optim (numpy.ndarray): The final optimized alpha values of shape (a, m).
    """
    graph = 1 * (W>0)
    n = X.shape[0]
    m = X_selected.shape[0]
    agents_data_idx = np.array_split(np.random.permutation(n), a)

    for agent_idx1 in range(a):
        graph[agent_idx1, agent_idx1] = 0

    lambda_ij = lamb0*np.ones((a, a, m, 1))

    alphas = []
    optimality_gaps = [[] for _ in range(a)]

    for _ in range(n_epochs):
        alpha_optim = np.zeros((a,m))
        alpha_optim = alpha_dual(X, X_selected, Y, sigma2, nu, a, graph, lambda_ij, agents_data_idx)

        for agent_idx1 in range(a):
            for agent_idx2 in range(agent_idx1):
                lambda_ij[agent_idx1, agent_idx2, : ] += step_size * (alpha_optim[agent_idx1, :] - alpha_optim[agent_idx2, :])
        alphas.append(alpha_optim)

        for agent_idx in range(a):
            alpha_agent = alpha_optim[agent_idx, :].reshape(-1, 1)
            optimality_gap = np.linalg.norm(alpha_agent - alpha_star)
            optimality_gaps[agent_idx].append(optimality_gap)

    return optimality_gaps, alphas

def alpha_dual(X, X_selected, Y, sigma2, nu, a, adj_matrix, lamb, agents_data_idx):
    """
    Compute the dual variables alpha for each agent in a distributed learning setting.

    Parameters:
    X (ndarray): The input data matrix of shape (n, d), where n is the number of samples and d is the number of features.
    X_selected (ndarray): The selected input data matrix of shape (m, d), where m is the number of selected samples and d is the number of features.
    Y (ndarray): The target values of shape (n,).
    sigma2 (float): The regularization parameter for the kernel matrix.
    nu (float): The regularization parameter for the identity matrix.
    a (int): The number of agents.
    adj_matrix (ndarray): The adjacency matrix of shape (a, a) representing the communication structure between agents.
    lamb (ndarray): The regularization parameter for the dual variables of shape (a, a, m).
    agents_data_idx (list): The list of indices representing the data assigned to each agent.

    Returns:
    ndarray: The dual variables alpha for each agent, of shape (a, m, 1).
    """
    m = X_selected.shape[0]
    K_mm = compute_kernel_matrix(X_selected, X_selected)
    alpha = []

    for agent_idx, data_idx in enumerate(agents_data_idx):

        X_local = X[data_idx]
        y_local = Y[data_idx].reshape(-1, 1)
        K_im = compute_kernel_matrix(X_local, X_selected)

        A = sigma2 * K_mm + np.eye(m)*nu + np.transpose(K_im) @ K_im
        b = np.transpose(K_im) @ y_local

        for j in range(a):

            if adj_matrix[agent_idx, j] != 0:

                if agent_idx > j:
                    b-= lamb[agent_idx, j, :]

                else:
                    b+= lamb[j, agent_idx, :]

        alpha.append(np.linalg.solve(A, b))

    return np.array(alpha)