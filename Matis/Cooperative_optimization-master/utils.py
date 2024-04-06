import numpy as np
import matplotlib.pyplot as plt
import pickle
import networkx as nx
from sinkhorn_knopp import sinkhorn_knopp as skp
from tqdm import tqdm
import time


# defining kernel function
def kernel_matrix(x, y):
    # Euclidean Kernel
    # x and y are numpy arrays
    n = len(x)
    m = len(y)
    K = np.zeros((n, m))
    for i in range(n):
        for j in range(m):
            K[i, j] = np.exp(-np.linalg.norm(x[i]-y[j])**2)
    return K


def kernel_im(xi, xm):
    # Euclidean Kernel
    # xi is a point and xm is a numpy array of points
    n = len(xm)
    K = np.zeros((1, n))
    for j in range(n):
        K[0][j] = np.exp(-np.linalg.norm(xi-xm[j])**2)
    # K should be of shape (1,n)
    return K


def get_Kij(index_i, index_j, K):
    Kij = K[np.array(index_i), :]
    Kij = Kij[:, np.array(index_j)]
    return Kij


def compute_alpha(x, y, x_selected, sigma, mu):
    n = len(x)
    m = len(x_selected)
    Kmm = kernel_matrix(x_selected, x_selected)
    Knm = kernel_matrix(x[0:n], x_selected)
    alpha_exact = np.linalg.inv(
        sigma**2*Kmm + mu*np.eye(m) + np.transpose(Knm) @ Knm) @ np.transpose(Knm) @ y[0:n]
    return alpha_exact


def get_agents_from_pickle(pickle_name, a, n, m, plot=False):
    # summary :
    # Inputs :
    # a : number of agents
    # n : number of data points
    # m : number of selected points
    # pickle_name : name of the pickle file

    # Load the data
    with open(pickle_name, 'rb') as f:
        x, y = pickle.load(f)
    agent_x = []
    agent_y = []
    # Randomly select m points
    # the points should be shared between the different agents
    selected_points = np.random.choice(np.array(range(n)), m, replace=False)
    x_selected = x[selected_points]
    y_selected = y[selected_points]
    selected_points_agents = np.array(range(n))
    np.random.shuffle(selected_points_agents)
    for j in range(a):
        agent_x.append(x[selected_points_agents[j*20:j*20+20]])
        agent_y.append(y[selected_points_agents[j*20:j*20+20]])
    K = kernel_matrix(x[0:n], x[0:n])
    # Data visualization
    if plot:
        for j in range(a):
            plt.plot(agent_x[j], agent_y[j], 'o', label='Data')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.show()
    selected_points_agents = selected_points_agents.reshape((int(a), int(n/a)))
    return agent_x, agent_y, x_selected, y_selected, selected_points, selected_points_agents, K, x[0:n], y[0:n]


def grad_alpha(sigma, mu, y_agent, x_agent, x_selected, alpha):
    Kmm = kernel_matrix(x_selected, x_selected)
    a = len(x_agent)
    grad = [0 for i in range(a)]  # list of numpy arrays
    for i in range(a):
        big_kernel_im = kernel_matrix(x_agent[i], x_selected)
        big_kernel_im_transpose = np.transpose(big_kernel_im)

        grad[i] = (1/a) * (sigma**2 * Kmm + mu * np.eye(len(x_selected))) @ alpha[i] + \
            big_kernel_im_transpose @ (big_kernel_im @ alpha[i] - y_agent[i])
    return np.array(grad).reshape(a, len(x_selected))


def grad_alpha_v3(sigma, mu, x, y, alpha, K, selected_points, selected_points_agents):
    Kmm = get_Kij(selected_points, selected_points, K)
    a = len(selected_points_agents)
    grad = [0 for i in range(a)]
    for i in range(a):
        big_kernel_im = get_Kij(selected_points_agents[i], selected_points, K)
        big_kernel_im_transpose = np.transpose(big_kernel_im)
        grad[i] = (1/a) * (sigma**2 * Kmm + mu * np.eye(len(selected_points))) @ alpha[i] + \
            big_kernel_im_transpose @ (big_kernel_im @
                                       alpha[i] - y[selected_points_agents[i]])
    return np.array(grad).reshape(a, len(selected_points))


def grad_alpha2(sigma, mu, y_agent, x_agent, x_selected, alpha):
    Kmm = kernel_matrix(x_selected, x_selected)
    a = len(x_agent)
    grad = [0 for i in range(a)]  # list of numpy arrays
    for i in range(a):
        grad[i] = (sigma**2*Kmm @ alpha[i] + mu*alpha[i]) / a
        for j in range(len(x_agent[i])):
            term_toadd = - y_agent[i][j]*np.transpose(kernel_im(x_agent[i][j], x_selected)) + \
                np.transpose(kernel_im(
                    x_agent[i][j], x_selected)) * (kernel_im(x_agent[i][j], x_selected) @ alpha[i])
            grad[i] += term_toadd.squeeze()
    return np.array(grad).reshape(a, len(x_selected))


def is_double_sto(A):
    """
    Vérifie si A est doublement stochastique
    """
    if A.shape[0] != A.shape[1]:
        print("Mauvaise forme")
        return False
    reponse = True
    ligne = np.zeros(A.shape[0])
    col = np.zeros(A.shape[0])
    for i in range(A.shape[0]):
        ligne[i] = A[i, :].sum()
        if not np.isclose(A[i, :].sum(), 1):
            reponse = False
        col[i] = A[:, i].sum()
        if not np.isclose(A[:, i].sum(), 1, rtol=0.01):
            reponse = False
    if reponse == False:
        print("ligne : ", ligne)
        print("Colonne : ", col)

    return reponse


def create_W(liste, taille, auto=True):
    """
    liste des arrêtes
    taille du graphe
    auto : bool : diagonale mise à 1
    """
    sk = skp.SinkhornKnopp()
    if auto == True:
        res = np.eye(taille)
    else:
        res = np.zeros((taille, taille))
    for i, j in liste:
        res[i, j] = 1
        res[j, i] = 1
    return sk.fit(res)


def visual_graph(liste_indice):
    G = nx.Graph()
    for i, j in liste_indice:
        G.add_edge(i+1, j+1)
    # explicitly set positions
    pos = {1: (-1, 0), 2: (np.cos(3*np.pi/5), np.sin(3*np.pi/5)), 3: (np.cos(np.pi/5), np.sin(np.pi/5)),
           4: (np.cos(-np.pi/5), np.sin(-np.pi/5)), 5: (np.cos(-3*np.pi/5), np.sin(-3*np.pi/5))}
    options = {
        "font_size": 36,
        "node_size": 3000,
        "node_color": "white",
        "edgecolors": "black",
        "linewidths": 1,
        "width": 1,
    }
    nx.draw_networkx(G, pos, **options)
    # Set margins for the axes so that nodes aren't clipped
    ax = plt.gca()
    ax.margins(0.1)
    plt.axis("off")
    plt.show()
    return 0


# main
if __name__ == "__main__":
    # Test kernel matrix :
    x = np.array([1, 1, 1])
    y = np.array([1, 1, 1, 1])
    test = kernel_matrix(x, y)
    print(test.shape == (3, 4))
    print(test == np.array([[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]]))
    # Test kernel_im
    xi = 1
    xm = np.array([1, 1, 1, 1])
    test = kernel_im(xi, xm)
    print(test.shape == (1, 4))
    print(test)
    # test kernel matrix with x of size 1
    x = np.array([1])
    y = np.array([1, 1, 1, 1])
    test = kernel_matrix(x, y)
    print(test.shape == (1, 4))
    print(test)

    x_agent, y_agent, x_selected, y_selected, selected_points, selected_points_agent, K_test, x, y = get_agents_from_pickle(
        'first_database.pkl', a=5, n=100, m=10, plot=False)

    # test if get_Kij is good
    K1 = kernel_matrix(x_selected, x_selected)
    K2 = get_Kij(selected_points, selected_points, K_test)
    print("TEST get_Kij : ", K1 == K2)

    # test grad_alpha vs grad_alpha_v3
    sigma = 0.5
    mu = 1
    alpha = [np.zeros(10) for i in range(5)]
    grad = grad_alpha(sigma, mu, y_agent, x_agent, x_selected, alpha)
    grad = np.round(grad, 1)
    gradv3 = grad_alpha_v3(
        sigma, mu, x, y, alpha, K_test, selected_points, selected_points_agent)
    gradv3 = np.round(gradv3, 1)
    print(gradv3)
    print(f'TEST : {grad==gradv3}')  # GOOOOOOD !!!!

    """
    # test grad_alpha2
    grad2 = grad_alpha2(sigma, mu, y_agent, x_agent, x_selected, alpha)
    # test grad_alpha and grad_alpha2 with 10e-1 precision
    grad[0] = np.round(grad[0], 1)
    grad2[0] = np.round(grad2[0], 1)
    grad[1] = np.round(grad[1], 1)
    grad2[1] = np.round(grad2[1], 1)
    print(f'TEST : {grad[0] == grad2[0]}')

    # time the two functions
    import time
    start = time.time()
    grad = grad_alpha(sigma, mu, y_agent, x_agent, x_selected, alpha)
    end = time.time()
    print(f'grad_alpha time : {end - start}')
    start = time.time()
    grad2 = grad_alpha2(sigma, mu, y_agent, x_agent, x_selected, alpha)
    end = time.time()
    print(f'grad_alpha2 time : {end - start}')
    """

    # Test is_double_sto
    ind = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)]
    Wtest = create_W(ind, 5, auto=False)
    print("TEST is_double_sto : ", is_double_sto(Wtest))
    print(Wtest)
    # Test visual_graph
    visual_graph(ind)
