from utils import *
import numpy as np

    
def solve_alpha_dualdec(x, y, selected_points, selected_points_agent, sigma, mu, K, adj_matrix, lamb):
    # print("lamb shape : ", lamb.shape)
    n = len(x)
    a = len(selected_points_agent)
    m = len(selected_points)
    Kmm = get_Kij(selected_points, selected_points, K)
    alpha = []
    for i in range(a):
        Kim = get_Kij(selected_points_agent[i], selected_points, K)
        A = sigma**2 * Kmm + np.eye(m)*mu + np.transpose(Kim) @ Kim
        b = np.transpose(Kim) @ y[selected_points_agent[i]]
        for j in range(a):
            if adj_matrix[i, j] != 0:
                if i > j:
                    b-= lamb[i, j, :]
                else:
                    b+= lamb[j, i, :]
        alpha.append(np.linalg.solve(A, b))
    return np.array(alpha)

def dualDec(x, y, selected_points, selected_points_agent, K, sigma, mu, lr, W, max_iter=1000, lamb0=0):
    # graph is the adjacency matrix
    # W is the weight matrix
    graph = 1 * (W>0)
    m = len(selected_points)
    a = len(selected_points_agent)
    for i in range(a):
        graph[i, i] = 0
    lambda_ij = lamb0*np.ones((a, a, m)) # should be shape number of edges in communication graph
    alpha_mean_list = []
    alpha_list_agent = []
    for n_iter in tqdm(range(max_iter)):
        # Calcul de x_i_star pour tous les noeuds
        alpha_optim = np.zeros((a,m))
        # for agent in range(a): 
        alpha_optim = solve_alpha_dualdec(
            x, y, selected_points, selected_points_agent, sigma, mu,
            K, graph, lambda_ij)
        for i in range(a):
            for j in range(i):
                lambda_ij[i, j, : ] += lr * (alpha_optim[i, :] - alpha_optim[j, :])
        alpha_mean_list.append(alpha_optim.mean(axis=0))
        alpha_list_agent.append(alpha_optim)
    
    alpha_optim = alpha_optim.reshape(a, m)
    alpha_optim = np.mean(alpha_optim, axis=0)

    return alpha_optim, alpha_list_agent, alpha_mean_list


# main
if __name__ == "__main__":
    with open('first_database.pkl', 'rb') as f:
        x, y = pickle.load(f)
    # # Data visualization
    # plt.plot(x, y, 'o', label='Data')
    # plt.xlabel('x')
    # plt.ylabel('y')
    # plt.show()
    # Generate the data
    a = 5
    n = 100
    m = 10
    agent_x, agent_y, x_selected, y_selected, selected_points, selected_points_agents, K, x, y = get_agents_from_pickle(
        'first_database.pkl', a, n, m)
    print(f'Nb agents : {a}')
    print(f'Nb data points : {n}')
    print(f'Nb selected points : {m}')
    print(f'Points per agent : {n/a}\n')

    sigma=0.5
    mu=0.1

    # Compute the alpha optimal
    print("Compute the alpha optimal....")
    start = time.time()
    alpha_optim = compute_alpha(x, y, x_selected, sigma, mu)
    end = time.time()
    print(f'Time to compute alpha optimal : {end - start}\n')
    # # Export alpha optimal to a file
    # with open('alpha_optim.pkl', 'wb') as f:
    #     pickle.dump(alpha_optim, f)
    print(f'alpha optimal : {alpha_optim}\n')

    # create the weight matrix
    ind = [(0,1), (1,2), (2,3), (3,4), (4,0)]
    W = create_W(ind, 5, auto=False)
    print(W)
    visual_graph(ind)
    # W = np.array([[1/3, 1/3, 0, 0, 1/3],
    #               [1/3, 1/3, 1/3, 0, 0],
    #               [0, 1/3, 1/3, 1/3, 0],
    #               [0, 0, 1/3, 1/3, 1/3],
    #               [1/3, 0, 0, 1/3, 1/3]])
    print("TEST MATRICE DOUBLE STO : ", is_double_sto(W))
    # Compute the alpha optimal with the dual decomposition algorithm
    start = time.time()
    alpha_optim, alpha_list, alpha_mean_list = dualDec(
        x, y, selected_points, selected_points_agents,
        K, sigma, mu, 0.01, W, max_iter=1000, lamb0=0.
    )
    end = time.time()
    print(f'alpha optimal with dual decomposition : {alpha_optim}')
    print(
        f'Time to compute alpha optimal with dual decomposition : {end - start}')
    # print(f'Total iterations : {tot_ite}\n')

    # Data visualization
    Y = np.linalg.norm(alpha_list - alpha_optim, axis=1)
    # unpack the list of alpha to get for each agent the evolution of alpha
    agent_1 = np.linalg.norm(np.array(
        [alpha_list[i][0] for i in range(len(alpha_list))]) - alpha_optim, axis=1)
    agent_2 = np.linalg.norm(np.array(
        [alpha_list[i][1] for i in range(len(alpha_list))]) - alpha_optim, axis=1)
    agent_3 = np.linalg.norm(np.array(
        [alpha_list[i][2] for i in range(len(alpha_list))]) - alpha_optim, axis=1)
    agent_4 = np.linalg.norm(np.array(
        [alpha_list[i][3] for i in range(len(alpha_list))]) - alpha_optim, axis=1)
    agent_5 = np.linalg.norm(np.array(
        [alpha_list[i][4] for i in range(len(alpha_list))]) - alpha_optim, axis=1)

    plt.plot(agent_1, label='Agent 1', color='blue')
    plt.plot(agent_2, label='Agent 2', color='red')
    plt.plot(agent_3, label='Agent 3', color='green')
    plt.plot(agent_4, label='Agent 4', color='orange')
    plt.plot(agent_5, label='Agent 5', color='purple')
    plt.xlabel('Iterations')
    plt.ylabel('Optimality gap (norm)')
    plt.xscale("log")
    plt.yscale("log")
    plt.grid()
    plt.show()
    # Plot selected points and the prediction of the model with the alpha optimal 
    plt.figure(0)
    for i in range(a):
        plt.plot(agent_x[i], agent_y[i], 'o', label=f'Agent {i+1}')
    # plt.plot(x[0:n], y[0:n], 'o', label='Data')
    x_predict = np.linspace(-1, 1, 250)
    K_f = kernel_matrix(x_predict, x_selected)
    # fx_predict = get_Kij(range(n), selected_points, K) @ alpha_optim_gt
    fx_predict = K_f @ alpha_optim
    plt.plot(x_predict, fx_predict, label='Prediction')
    plt.grid()
    plt.legend()
    plt.show()