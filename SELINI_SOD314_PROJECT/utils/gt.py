import numpy as np
from utils.fct import compute_kernel_matrix, compute_local_gradient

def gradient_tracking(X, X_selected, Y, sigma2, nu, a, n_epochs, step_size, alpha_star, W):
    """
    Perform gradient tracking optimization algorithm.

    Args:
        X (numpy.ndarray): The input data matrix of shape (n, d), where n is the number of samples and d is the number of features.
        X_selected (numpy.ndarray): The selected data matrix of shape (m, d), where m is the number of selected samples and d is the number of features.
        Y (numpy.ndarray): The target values of shape (n,).
        sigma2 (float): The kernel parameter.
        nu (float): The regularization parameter.
        a (int): The number of agents.
        n_epochs (int): The number of epochs.
        step_size (float): The step size for updating the alpha values.
        alpha_star (numpy.ndarray): The optimal alpha values of shape (m * a, 1).
        W (numpy.ndarray): The weight matrix of shape (m * a, m * a).

    Returns:
        tuple: A tuple containing two lists:
            - optimality_gaps: A list of length a, where each element is a list of optimality gaps for each agent at each epoch.
            - alphas: A list of alpha values at each epoch.
    """
    m = X_selected.shape[0]
    n = X.shape[0]
    agents_data_idx = np.array_split(np.random.permutation(n), a)
    K_mm = compute_kernel_matrix(X_selected, X_selected)

    W = np.kron(W, np.eye(m))
    
    # Initialize alpha
    alpha_old = np.random.normal(0, 100, (m * a, 1))

    # Initialize gradients
    gradients = np.zeros((m * a, 1))
    for agent_idx1, data_idx in enumerate(agents_data_idx):
        X_local = X[data_idx]
        y_local = Y[data_idx].reshape(-1, 1)
        K_im = compute_kernel_matrix(X_local, X_selected)
        local_gradient = compute_local_gradient(alpha_old[m * agent_idx1: m * (agent_idx1 + 1)], sigma2, K_mm, y_local, K_im, nu, a)
        gradients[m * agent_idx1: m * (agent_idx1 + 1)] = local_gradient
    
    g_old = np.zeros((m * a, 1))
    g_new = np.zeros((m * a, 1))

    alphas = []
    optimality_gaps = [[] for _ in range(a)]
    
    for _ in range(n_epochs):
        
        alpha_new = W @ alpha_old - step_size * gradients
        
        for agent_idx2, data_idx in enumerate(agents_data_idx):
            
            X_local = X[data_idx]
            y_local = Y[data_idx].reshape(-1, 1)
            K_im = compute_kernel_matrix(X_local, X_selected)
            
            local_gradient_old = compute_local_gradient(alpha_old[m * agent_idx2: m * (agent_idx2 + 1)], sigma2, K_mm, y_local, K_im, nu, a)
            local_gradient_new = compute_local_gradient(alpha_new[m * agent_idx2: m * (agent_idx2 + 1)], sigma2, K_mm, y_local, K_im, nu, a)
            
            gradients[m * agent_idx2: m * (agent_idx2 + 1)] = local_gradient_old
            g_new[m * agent_idx2: m * (agent_idx2 + 1)] = local_gradient_new
        
        gradients = W @ gradients
        gradients += (g_new - g_old)
        
        alpha_old = alpha_new
        alphas.append(alpha_old)
        
        for agent_idx3 in range(a):
            alpha_agent = alpha_new[m * agent_idx3: m * (agent_idx3 + 1)].reshape(-1, 1)
            optimality_gap = np.linalg.norm(alpha_agent - alpha_star.reshape(-1, 1))
            optimality_gaps[agent_idx3].append(optimality_gap)

    return optimality_gaps, alphas