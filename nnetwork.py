import numpy as np

class Layer:
    def __init__(self, nodesIn: int, nodesOut: int) -> None:
        self.nodesIn = nodesIn
        self.nodesOut = nodesOut

        self.weights = np.random.uniform(-10, 10, size=(nodesOut, nodesIn))
        self.biases = np.random.uniform(-10, 10, size=(nodesOut, 1))
        self.inputs = None
    
    def send_input(self, vector): # use this method to set x's 
        self.inputs = vector

    def calculate_output(self, activationFunction):
        res = self.weights.dot(self.inputs)
        res = res + self.biases

        for element in res:
            element[0] = activationFunction(element[0])
        
        return res
    
    def __str__(self):
        return f"({self.nodesIn}, {self.nodesOut})"


class NeuralNetwork:
    def __init__(self, layers: list[int], activationFunction, costFunction):
        self.layers = layers # just gives size. for example, [2, 3, 2] describes 2 inputs, 3 hidden nodes and 2 outputs. 3 layers total
        self._layers = self._create_layers() # this holds the actual Layer objects
        self._activationFunction = activationFunction
        self._costFunction = costFunction
    
    """
    Realistically, these layers represent the bridges between the neurons.
    For this reason, the len(_layers) is 1 less than the len(layers)
    """
    def _create_layers(self) -> list[Layer]:
        res = []
        for i in range(len(self.layers) - 1):
            res.append(Layer(self.layers[i], self.layers[i + 1]))
        return res
    
    def forward(self, input): # send an input through the network. This is the forward propogation function
        self._layers[0].send_input(input)
        for i in range(1, len(self._layers)):
            res = self._layers[i - 1]
            self._layers[i].send_input(res.calculate_output(self._activationFunction))
        
        return self._layers[-1].calculate_output(self._activationFunction)

    def _get_cost(self, input, expected) -> float:
        output = self.forward(input)
        cost = 0.0

        for i in range(len(output)):
            cost += self._costFunction(output[i][0], expected[i][0])
        
        return cost

    
    # Debug
    def print_layers(self):
        for layer in self._layers:
            print(layer)