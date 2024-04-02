import numpy as np
from utils.fct import compute_kernel_matrix, compute_local_gradient

def decentralized_gradient_descent(X, X_selected, Y, sigma2, nu, a, n_epochs, step_size, alpha_star, W):
    """
    Performs decentralized gradient descent optimization algorithm.

    Args:
        X (numpy.ndarray): The input data matrix of shape (num_total_data_points, num_features).
        X_selected (numpy.ndarray): The selected data matrix of shape (num_selected_data_points, num_features).
        Y (numpy.ndarray): The target values of shape (num_total_data_points, 1).
        sigma2 (float): The value of the variance parameter.
        nu (float): The value of the regularization parameter.
        a (int): The number of agents.
        n_epochs (int): The number of epochs for the optimization algorithm.
        step_size (float): The step size for the optimization algorithm.
        alpha_star (numpy.ndarray): The optimal alpha values of shape (num_selected_data_points * num_agents, 1).
        W (numpy.ndarray): The communication matrix of shape (num_agents, num_agents).

    Returns:
        tuple: A tuple containing two lists:
            - optimality_gaps (list): A list of lists containing the optimality gaps for each agent at each iteration.
            - alphas (list): A list of alpha values at each iteration.
    """
    m = X_selected.shape[0]
    n = X.shape[0]
    agents_data_idx = np.array_split(np.random.permutation(n), a)
    W = np.kron(W, np.eye(m))
    K_mm = compute_kernel_matrix(X_selected, X_selected)
    alpha = np.ones((m * a, 1))
    optimality_gaps = [[] for _ in range(a)]
    alphas = []
    
    for _ in range(n_epochs):
        alphas.append(alpha)
        gradients = np.zeros((m * a, 1))

        for agent_idx, data_idx in enumerate(agents_data_idx):
            X_local = X[data_idx]
            y_local = Y[data_idx].reshape(-1, 1)
            K_im = compute_kernel_matrix(X_local, X_selected)
            local_gradient = compute_local_gradient(alpha[m * agent_idx: m * (agent_idx + 1)], sigma2, K_mm, y_local, K_im, nu, a)
            gradients[m * agent_idx: m * (agent_idx + 1)] = local_gradient
        
        alpha = W @ alpha - step_size * gradients
        
        for agent_idx in range(a):
            alpha_agent = alpha[m * agent_idx: m * (agent_idx + 1)].reshape(-1, 1)
            optimality_gap = np.linalg.norm(alpha_agent - alpha_star.reshape(-1, 1))
            optimality_gaps[agent_idx].append(optimality_gap)
        
    return optimality_gaps, alphas
