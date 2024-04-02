import numpy as np
from utils.fct import compute_kernel_matrix, compute_local_gradient

# def get_Kij(index_i, index_j, K):
#     Kij = K[np.array(index_i), :]
#     Kij = Kij[:, np.array(index_j)]
#     return Kij

# def grad_alpha_v3(sigma, mu, x, y, alpha, K, selected_points, selected_points_agents):
#     Kmm = get_Kij(selected_points, selected_points, K)
#     a = len(selected_points_agents)
#     grad = [0 for i in range(a)]
#     for i in range(a):
#         big_kernel_im = get_Kij(selected_points_agents[i], selected_points, K)
#         big_kernel_im_transpose = np.transpose(big_kernel_im)
#         grad[i] = (1/a) * (sigma**2 * Kmm + mu * np.eye(len(selected_points))) @ alpha[i] + \
#             big_kernel_im_transpose @ (big_kernel_im @ alpha[i] - y[selected_points_agents[i]])
#     return np.array(grad).reshape(a, len(selected_points))

def gradient_tracking(X, X_selected, Y, sigma2, nu, a, n_epochs, step_size, alpha_star, W):
    """
    This function implements the gradient tracking algorithm.
    Parameters
    ----------
    x : list of numpy array
        The x coordinates of the data points 
    y : list of numpy array
        The y coordinates of the data points 
    """
    m = X_selected.shape[0]
    n = X.shape[0]
    agents_data_idx = np.array_split(np.random.permutation(n), a)
    K_mm = compute_kernel_matrix(X_selected, X_selected)

    W = np.kron(W, np.eye(m))
    
    # Initialize alpha
    alpha_old = np.random.normal(0, 100, (a*m, 1))

    # gradient = grad_alpha_v3(
    #     sigma, mu, X, Y, alpha_old.reshape(a, m),
    #     K, selected_points, selected_points_agent).reshape(a*m, 1)
    
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
    print(f'After initialization, len(optimality_gaps): {len(optimality_gaps)}')
    
    for _ in range(n_epochs):
        
        alpha_new = W @ alpha_old - step_size * gradients
        
        # IMPORTANT : in grad_alpha alpha should be a 2D array
        # g_new = grad_alpha_v3(
        #     sigma, mu, X, Y, alpha_new.reshape(a, m),
        #     K, selected_points, selected_points_agent).reshape(a*m, 1)
        # g_old = grad_alpha_v3(
        #     sigma, mu, X, Y, alpha_old.reshape(a, m),
        #     K, selected_points, selected_points_agent).reshape(a*m, 1)
        
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
        #alpha_mean = np.mean(alpha_old.reshape(a, m), axis=0)
        alphas.append(alpha_old)
        
        for agent_idx3 in range(a):
            alpha_agent = alpha_new[m * agent_idx3: m * (agent_idx3 + 1)].reshape(-1, 1)
            optimality_gap = np.linalg.norm(alpha_agent - alpha_star.reshape(-1, 1))
            print(f'Before appending, len(optimality_gaps): {len(optimality_gaps)}')
            optimality_gaps[agent_idx3].append(optimality_gap)  # Assuming this is how you're appending
            print(f'After appending, len(optimality_gaps): {len(optimality_gaps)}')
            
    print(f'At end of function, len(optimality_gaps): {len(optimality_gaps)}')
    # alpha_optim = alpha_new.reshape(a, m)
    # alpha_optim = np.mean(alpha_optim, axis=0)
    return optimality_gaps, alphas #, alpha_optim