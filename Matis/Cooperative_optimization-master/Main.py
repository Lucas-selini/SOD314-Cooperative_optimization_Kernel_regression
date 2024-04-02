import numpy as np
import matplotlib.pyplot as plt
import pickle
import networkx as nx
from utils import kernel_matrix, kernel_im

# Load the data
with open('first_database.pkl', 'rb') as f:
    x, y = pickle.load(f)

# Data visualization

plt.plot(x, y, 'o', label='Data')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

# Point selection
n, m = 100, 10
sigma = 0.5
mu = 1
a = 5  # number of agents to be selected
agent_x = []
agent_y = []
# Randomly select m points
# the points should be shared between the different agents
selected_points = np.random.choice(np.array(range(n)), m, replace=False)
x_selected = x[selected_points]
y_selected = y[selected_points]
for j in range(a):
    agent_x.append(x[j*20:j*20+20])
    agent_y.append(y[j*20:j*20+20])
# Data visualization
for j in range(a):
    plt.plot(agent_x[j], agent_y[j], 'o', label='Data')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

# define kernel matrix
Kmm = kernel_matrix(x_selected, x_selected)
Knm = kernel_matrix(x[0:n], x_selected)
# alpha_optim using cvxpy
# Define the optimization variable
sigma = 0.5
alpha_exact = np.linalg.inv(
    sigma**2*Kmm + np.eye(m) + np.transpose(Knm) @ Knm) @ np.transpose(Knm) @ y[0:n]
print("the alpha exact is :", alpha_exact)
# define the graph connection of the agents (a undirected star graph for now):
Gx = nx.star_graph(a-1).to_undirected()
nx.draw(Gx, with_labels=True)
plt.show()
Adj = nx.adjacency_matrix(Gx).todense()
print(Adj)
# define the decentralized gradient descent :
# define the step size
