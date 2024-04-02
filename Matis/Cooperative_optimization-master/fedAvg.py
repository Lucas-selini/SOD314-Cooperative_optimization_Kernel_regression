from utils import *

# Consider back the case of n = 100, m = 10, and a = 5. Consider the second database and assign each
# groups of 20 points to the different agents.
# with open(’second_database.pkl’, ’rb’) as f:
# X, Y = pickle.load(f)
# In particular X[i] and Y[i] correspond to the data available to agent i.
# As you may have noticed, the points m do not have to be necessarily points for which we have labels.
# Here it is convenient to use
# x_m_points=np.linspace(-1,1,m)

def grad_alpha_fedavg(sigma, mu, y, alpha, Kmm, Kim, a, m):
    # 
    # Kmm = get_Kij(selected_points, selected_points, K)
    # a = len(selected_points_agents)
    # m = len(selected_points)
    grad = [0 for i in range(a)]
    for i in range(a):
        big_kernel_im = Kim[i]
        # big_kernel_im = get_Kij(selected_points_agents[i], selected_points, K)
        big_kernel_im_transpose = np.transpose(big_kernel_im)
        grad[i] = (1/a) * (sigma**2 * Kmm + mu * np.eye(m)) @ alpha[i] + \
            big_kernel_im_transpose @ (big_kernel_im @
                                       alpha[i] - y[i])
    return np.array(grad).reshape(a, m)

def fedAvg(X, Y, x_m_points, T, E, K, Kim, sigma, mu, lr):
    # init alpha for server
    m = len(x_m_points)
    a = len(X)
    alpha_server = np.zeros((1, m))
    # repeat for t=1, .., T
    for t in range(T):
        # send alpha_server to all agents
        # Client update
        # init alpha_agents 
        alpha_agents = np.zeros((a, m))
        for epoch in range(E):
            grad = grad_alpha_fedavg(sigma, mu, Y, alpha_agents, K, Kim, a, m)
            # for each agent i=1, .., a
            for i in range(a):
                alpha_agents[i] = alpha_agents[i] - lr * grad[i]
        # Server update
        for i in range(a):
            # normalizing 
            alpha_agents[i] = alpha_agents[i] *(len(alpha_agents[i])/alpha_agents.size)
        # Mixing
        alpha_server = np.sum(alpha_agents, axis=0)

    return alpha_server, alpha_agents


if __name__=="__main__":   
    # Generate the data
    a = 5
    n = 100
    m = 10
    with open('second_database.pkl', 'rb') as f:
        X, Y = pickle.load(f)
    X = np.array(X)
    Y = np.array(Y)
    print(X)
    print(Y)
    print("X shape : ", X.shape)
    print("Y shape : ", Y.shape)
    x_m_points=np.linspace(-1,1,m)

    K = kernel_matrix(x_m_points, x_m_points)
    Kim = []
    for i in range(a):
        Kim.append(kernel_matrix(X[i], x_m_points))
    print("K shape : ", K.shape)   
    print("Kim shape : ", Kim[0].shape)

    T = 100
    E = 50
    sigma = 0.5
    mu = 0.1
    lr = 0.001
    alpha_server, alpha_agents = fedAvg(X, Y, x_m_points, T, E, K, Kim, sigma, mu, lr)
    print("Alpha server : ", alpha_server)
    print("Alpha agents : ", alpha_agents)

    # Data visualization
    # Y = np.linalg.norm(alpha_list - alpha_optim, axis=1)
    # unpack the list of alpha to get for each agent the evolution of alpha
    # agent_1 = np.linalg.norm(np.array(
    #     [alpha_list[i][0] for i in range(len(alpha_list))]) - alpha_optim, axis=1)
    # agent_2 = np.linalg.norm(np.array(
    #     [alpha_list[i][1] for i in range(len(alpha_list))]) - alpha_optim, axis=1)
    # agent_3 = np.linalg.norm(np.array(
    #     [alpha_list[i][2] for i in range(len(alpha_list))]) - alpha_optim, axis=1)
    # agent_4 = np.linalg.norm(np.array(
    #     [alpha_list[i][3] for i in range(len(alpha_list))]) - alpha_optim, axis=1)
    # agent_5 = np.linalg.norm(np.array(
    #     [alpha_list[i][4] for i in range(len(alpha_list))]) - alpha_optim, axis=1)

    # plt.plot(agent_1, label='Agent 1', color='blue')
    # plt.plot(agent_2, label='Agent 2', color='red')
    # plt.plot(agent_3, label='Agent 3', color='green')
    # plt.plot(agent_4, label='Agent 4', color='orange')
    # plt.plot(agent_5, label='Agent 5', color='purple')
    # plt.xlabel('Iterations')
    # plt.ylabel('Optimality gap (norm)')
    # plt.xscale("log")
    # plt.yscale("log")
    # plt.grid()
    # plt.show()
    # # Plot selected points and the prediction of the model with the alpha optimal 
    # plt.figure(0)
    # for i in range(a):
    #     plt.plot(agent_x[i], agent_y[i], 'o', label=f'Agent {i+1}')
    # # plt.plot(x[0:n], y[0:n], 'o', label='Data')
    # x_predict = np.linspace(-1, 1, 250)
    # K_f = kernel_matrix(x_predict, x_selected)
    # # fx_predict = get_Kij(range(n), selected_points, K) @ alpha_optim_gt
    # fx_predict = K_f @ alpha_optim_gt
    # plt.plot(x_predict, fx_predict, label='Prediction')
    # plt.grid()
    # plt.legend()
    # plt.show()

    # agent_x, agent_y, x_selected, y_selected, selected_points, selected_points_agents, K, x, y = get_agents_from_pickle(
    #     'second_database.pkl', a, n, m)
    # print(f'Nb agents : {a}')
    # print(f'Nb data points : {n}')
    # print(f'Nb selected points : {m}')
    # print(f'Points per agent : {n/a}\n')

    # # Compute the alpha optimal
    # print("Compute the alpha optimal....")
    # sigma = 0.5
    # start = time.time()
    # alpha_optim = compute_alpha(x, y, x_selected, sigma)
    # end = time.time()
    # print(f'Time to compute alpha optimal : {end - start}\n')
    # # Export alpha optimal to a file
    # with open('alpha_optim.pkl', 'wb') as f:
    #     pickle.dump(alpha_optim, f)
    # print(f'alpha optimal : {alpha_optim}\n')

