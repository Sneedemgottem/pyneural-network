import math

class ActivationFunctions:
    @staticmethod
    def sigmoid(x) -> float:
        return 1 / (1 + math.exp(-x))
    
    @staticmethod
    def sigmoid_derivative(x) -> float:
        ActivationFunctions.sigmoid(x) * (1 - ActivationFunctions.sigmoid(x))
    
    @staticmethod
    def reLu(x) -> float:
        return max(0, x)

class CostFunctions:
    # utility function
    @staticmethod
    def mse_node_cost(outputActivation: float, expectedOutput) -> float:
        error = outputActivation - expectedOutput
        return error * error