from nnetwork import NeuralNetwork
from funcs import ActivationFunctions, CostFunctions
import numpy as np

myInputs = np.array([[5], [6]])
nn = NeuralNetwork([2, 2], ActivationFunctions.sigmoid, CostFunctions.mse_node_cost)
res = nn._get_cost(myInputs, np.array([[1.0], [0.0]], dtype=float))

print(res)