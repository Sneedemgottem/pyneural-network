import math

class ActivationFunctions:
    @staticmethod
    def sigmoid(x) -> float:
        return 1 / (1 + math.exp(-x))
    
    @staticmethod
    def reLu(x) -> float:
        return max(0, x)