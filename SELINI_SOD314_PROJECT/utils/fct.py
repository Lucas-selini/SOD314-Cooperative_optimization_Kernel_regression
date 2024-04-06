import numpy as np
import pickle
import matplotlib.pyplot as plt

def load_plot_data(database, n, m, a):
    # Load data
    with open(f'./database/{database}.pkl', 'rb') as f:
        x, y = pickle.load(f)

    # Generate data
    x_n=x[:n] 
    y_n=y[:n]
    sel = [i for i in range(n)]
    ind = np.random.choice(sel, m, replace=False)
    x_selected = np.array([x[i] for i in ind])
    # Create a plot to visualize the data
    plt.figure(figsize=(8, 6))
    plt.scatter(x, y, color='blue',s=0.2, alpha=0.5, label='Data Points')
    plt.scatter(x_n, y_n, color='red',s=10, alpha=0.5, label='Training Points')
    plt.scatter(x_selected, [y[i] for i in ind], color='black',s=10, alpha=0.5, label='Selected Points')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Plot of Data')
    plt.legend()
    plt.grid(True)
    plt.show()
    return x, y, x_n, y_n, x_selected, ind

def euclidean_kernel(x, xi):
    """
    Compute the Euclidean kernel between two vectors.

    Parameters:
    x (numpy.ndarray): The first vector.
    xi (numpy.ndarray): The second vector.

    Returns:
    float: The Euclidean kernel value.
    """
    return np.exp(-np.linalg.norm(x - xi)**2)

def compute_kernel_matrix(X, Y):
    """
    Compute the kernel matrix between two sets of vectors.

    Parameters:
    X (numpy.ndarray): The first set of vectors.
    Y (numpy.ndarray): The second set of vectors.

    Returns:
    numpy.ndarray: The kernel matrix.
    """
    a = X.shape[0]
    b = Y.shape[0]
    K = np.zeros((a, b))
    for i in range(a):
        for j in range(b):
            K[i, j] = euclidean_kernel(X[i], Y[j])
            
    return K

def compute_alpha_star(Kmm, Knm, y, sigma2, nu):
    """
    Compute the alpha_star vector.

    Parameters:
    Kmm (numpy.ndarray): The covariance matrix.
    Knm (numpy.ndarray): The covariance matrix.
    y (numpy.ndarray): The response vector.
    sigma2 (float): The variance parameter.
    nu (float): The regularization parameter.

    Returns:
    numpy.ndarray: The alpha_star vector.
    """
    m = Kmm.shape[0]
    A = sigma2 * Kmm + np.dot(Knm.T, Knm) + nu * np.eye(m)
    b = np.dot(Knm.T, y)
    alpha_star = np.linalg.solve(A, b)
    
    return alpha_star

def local_objective_function(alpha, sigma2, K_mm, y_local, K_im, nu, a):
    """
    Calculate the value of the local objective function.

    Parameters:
    alpha (numpy.ndarray): The vector of coefficients.
    sigma2 (float): The variance parameter.
    K_mm (numpy.ndarray): The covariance matrix.
    y_local (numpy.ndarray): The local response vector.
    K_im (numpy.ndarray): The covariance matrix.
    nu (float): The regularization parameter.
    a (float): The scaling parameter.

    Returns:
    float: The value of the local objective function.
    """
    A = (sigma2 / a) * (1 / 2) * np.dot(np.dot(alpha.T, K_mm), alpha)
    B = (1 / 2) * np.sum((y_local - K_im @ alpha)**2)
    C = (nu / (2 * a)) * np.linalg.norm(alpha)**2
    
    return A + B + C

def compute_local_gradient(alpha, sigma2, K_mm, y_local, K_im, nu, a):
    """
    Compute the local gradient.

    Parameters:
    alpha (numpy.ndarray): The vector of coefficients.
    sigma2 (float): The variance parameter.
    K_mm (numpy.ndarray): The covariance matrix.
    y_local (numpy.ndarray): The local response vector.
    K_im (numpy.ndarray): The covariance matrix.
    nu (float): The regularization parameter.
    a (float): The scaling parameter.

    Returns:
    numpy.ndarray: The local gradient.
    """
    grad = sigma2 * K_mm @ alpha / a
    grad += K_im.T @ (K_im @ alpha - y_local)
    grad += (nu / a) * alpha
    
    return grad

def nystrom_approx(alpha, X_selected, X):
    """
    Compute the Nystrom approximation.

    Parameters:
    alpha (numpy.ndarray): The vector of coefficients.
    X_selected (numpy.ndarray): The selected set of vectors.
    X (numpy.ndarray): The original set of vectors.

    Returns:
    numpy.ndarray: The Nystrom approximation.
    """
    K1m = compute_kernel_matrix(X, X_selected)
    return K1m @ alpha

def plot_optimality_gaps(algroithm_name, graph_type, step_size, n_iter, optimality_gaps):
    for agent_idx, optimality_gap in enumerate(optimality_gaps):
        plt.plot(range(n_iter), optimality_gap, label=f"Agent {agent_idx + 1}")

    plt.xlabel('Number of iterations')
    plt.ylabel('Optimality Gap')
    plt.title(f'{algroithm_name} Convergence, Step size : {step_size}, {graph_type} Graph')
    plt.legend()
    plt.yscale('log')
    plt.xscale('log')
    plt.grid(True)
    plt.show()

def plot_function(algroithm_name, graph_type, step_size, n_iter, alphas, x_selected, x_n, y_n, m, alpha_star):
    nt=250
    x_prime=np.linspace(-1,1,nt)

    ag_idx = 0 

    alpha_agent_line100 = alphas[100][ag_idx*m:(ag_idx+1)*m]
    alpha_agent_line4000 = alphas[4000][ag_idx*m:(ag_idx+1)*m]
    alpha_agent_linefinal = alphas[n_iter-1][ag_idx*m:(ag_idx+1)*m]

    reconstruction_line100=nystrom_approx(alpha_agent_line100,x_selected,x_prime)
    reconstruction_line4000=nystrom_approx(alpha_agent_line4000,x_selected,x_prime)
    reconstruction_linefinal=nystrom_approx(alpha_agent_linefinal,x_selected,x_prime)
    reconstruction_alphastar = nystrom_approx(alpha_star,x_selected,x_prime)

    # Plot data points used for DGD with squares
    plt.scatter(x_n, y_n, color='blue', marker='s', label='Training Data')

    # Plot reconstructions with thicker lines
    plt.plot(x_prime, reconstruction_alphastar, color='green', linewidth=2, label=r'Reconstruction Optimal $\alpha^*$')
    plt.plot(x_prime, reconstruction_line100, color='yellow',linewidth=2, label=f'Reconstruction Agent {ag_idx+1} after 100 iterations', linestyle='dashed')
    plt.plot(x_prime, reconstruction_line4000, color='orange', linewidth=2, label=f'Reconstruction Agent {ag_idx+1} after 4 000 iterations', linestyle='dashed')
    plt.plot(x_prime, reconstruction_linefinal, color='red', linewidth=2, label=f'Reconstruction Agent {ag_idx+1} after {n_iter} iterations', linestyle='dashed')



    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title(f'{algroithm_name} Obtained Reconstruction with a {graph_type} graph and step size : {step_size}')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_function_dual(algroithm_name, graph_type, step_size, n_iter, alphas, x_selected, x_n, y_n, m, alpha_star):
    nt=250
    x_prime=np.linspace(-1,1,nt)

    ag_idx = 0 

    alpha_agent_line100 = alphas[100][ag_idx, :]
    alpha_agent_line4000 = alphas[4000][ag_idx, :]
    alpha_agent_linefinal = alphas[n_iter-1][ag_idx, :]

    reconstruction_line100=nystrom_approx(alpha_agent_line100,x_selected,x_prime)
    reconstruction_line4000=nystrom_approx(alpha_agent_line4000,x_selected,x_prime)
    reconstruction_linefinal=nystrom_approx(alpha_agent_linefinal,x_selected,x_prime)
    reconstruction_alphastar = nystrom_approx(alpha_star,x_selected,x_prime)

    # Plot data points used for DGD with squares
    plt.scatter(x_n, y_n, color='blue', marker='s', label='Training Data')

    # Plot reconstructions with thicker lines
    plt.plot(x_prime, reconstruction_alphastar, color='green', linewidth=2, label=r'Reconstruction Optimal $\alpha^*$')
    plt.plot(x_prime, reconstruction_line100, color='yellow',linewidth=2, label=f'Reconstruction Agent {ag_idx+1} after 100 iterations', linestyle='dashed')
    plt.plot(x_prime, reconstruction_line4000, color='orange', linewidth=2, label=f'Reconstruction Agent {ag_idx+1} after 4 000 iterations', linestyle='dashed')
    plt.plot(x_prime, reconstruction_linefinal, color='red', linewidth=2, label=f'Reconstruction Agent {ag_idx+1} after {n_iter} iterations', linestyle='dashed')



    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title(f'{algroithm_name} Obtained Reconstruction with a {graph_type} graph and step size : {step_size}')
    plt.legend()
    plt.grid(True)
    plt.show()