import sys # for sys.argv
import numpy # NumPy math library


# Sigmoid function (S-curve), and its derivative
def sigmoid(x, deriv):
    if(deriv == True):
        return x * (1 - x)
    return 1 / (1 + numpy.exp(-x))

# Implementation of a simple neural network with configurable input size, output size, number of layers, and transfer function
class NeuralNet:
    # NeuralNet(): instantiate a NeuralNet object
    # Note: transferFunction must take the form of: f(x, bool derivative)
    def __init__(self, inputSize, outputSize, numLayers, dataSize, transferFunction = sigmoid):
        self.inputSize = inputSize
        self.outputSize = outputSize
        self.numLayers = numLayers
        self.dataSize = dataSize
        self.transferFunction = transferFunction

        # initialize weights with random values in the range [-1, 1]
        self.weights = [None] * self.numLayers
        self.weights[0] = 2 * numpy.random.random((self.inputSize, self.dataSize)) - 1 # initialize input weights
        for layer in range(1, self.numLayers - 1):
            self.weights[layer] = 2 * numpy.random.random((self.dataSize, self.dataSize)) - 1 # initialize hidden weights
        self.weights[self.numLayers - 1] = 2 * numpy.random.random((self.dataSize, self.outputSize)) - 1 # initialize output weights

    def __repr__(self):
        return str(self.weights)
    
    # run(): propagate data forwards through the network
    def run(self, inputData):
        self.layers = [None] * (self.numLayers + 1)
        self.layers[0] = inputData # apply data to input layer
        for layer in range(1, self.numLayers + 1):
            self.layers[layer] = self.transferFunction(numpy.dot(self.layers[layer - 1], self.weights[layer - 1]), False) # propagate through hidden layers
        return self.layers[self.numLayers] # return output layer

    # train(): propagate an input data set forwards, then compare against expected results and propagate error backwards to update network weights
    def train(self, inputData, expectedResults):
        output = self.run(inputData) # forward propagation (i.e. run the neural net!)

        # calculate error at output
        layer_errors = [None] * (self.numLayers + 1)
        layer_errors[self.numLayers] = expectedResults - output

        # calculate deltas based on confidence
        layer_deltas = [None] * (self.numLayers + 1)
        layer_deltas[self.numLayers] = layer_errors[self.numLayers] * self.transferFunction(self.layers[self.numLayers], True)
        for layer in reversed(range(1, self.numLayers)):
            # how much did each layers[n] value contrinute to the layers[n+1] error (according to weights)?
            layer_errors[layer] = layer_deltas[layer + 1].dot(self.weights[layer].T) # .T transposes the array (flips it sideways)

            # calculate deltas based on confidence
            layer_deltas[layer] = layer_errors[layer] * self.transferFunction(self.layers[layer], True)

        # update weights
        for layer in range(self.numLayers):
            self.weights[layer] += self.layers[layer].T.dot(layer_deltas[layer + 1])


def main(argv):
    # input dataset (each row is a training example, each column is one of 3 input nodes)
    input_data = numpy.array([[0,0,1],
                              [0,1,1],
                              [1,0,1],
                              [1,1,1]])
        
    # expected output dataset            
    expected_output_data = numpy.array([[0],
                                        [1],
                                        [1],
                                        [0]])

    numpy.random.seed(1) # seed random numbers to make calculation deterministic
    net = NeuralNet(inputSize = 3, outputSize = 1, numLayers = 3, dataSize = 4, transferFunction = sigmoid)
    for i in range(10000):
        net.train(input_data, expected_output_data)
    print("Final Output:")
    print(net.run(input_data))
    pass

if __name__ == "__main__":
    main(sys.argv)
