import numpy # NumPy math library


# Implementation of a simple 2-layer neural net with 3 inputs and 1 output
class NeuralNet:

    # NeuralNet( transferFunction(x, deriv = False) )
    def __init__(self, _transferFunction):
        self.transferFunction = _transferFunction

        # initialize weights randomly with mean of 0
        self.weights = [None] * 2
        self.weights[0] = 2 * numpy.random.random((3,4)) - 1
        self.weights[1] = 2 * numpy.random.random((4,1)) - 1

    def __repr__(self):
        return str(self.layers[2])

    def run(self, numIterations = 10000):
        for attempt in range(numIterations):

            # forward propagation (i.e. run the neural net!)
            self.layers = [None] * 3
            self.layers[0] = input_data
            self.layers[1] = self.transferFunction(numpy.dot(self.layers[0], self.weights[0]))
            self.layers[2] = self.transferFunction(numpy.dot(self.layers[1], self.weights[1]))

            # how much did we miss?
            layer_errors = [None] * 3
            layer_errors[2] = expected_output_data - self.layers[2]
            #if (attempt % 1000) == 0:
            #    print("Error: " + str(numpy.mean(numpy.abs(layer_errors[2]))))
    
            # in what direction is the target value?
            # were we really sure? if so, don't change too much.
            layer_deltas = [None] * 3
            layer_deltas[2] = layer_errors[2] * self.transferFunction(self.layers[2], deriv=True)
    
            # how much did each layers[1] value contrinute to the layers[2] error (according to weights)?
            layer_errors[1] = layer_deltas[2].dot(self.weights[1].T) # .T transposes the array (flips it sideways)

            # in what direction is the target layers[1]?
            # were we really sure? if so, don't change too much.
            layer_deltas[1] = layer_errors[1] * self.transferFunction(self.layers[1], deriv=True)

            # update weights
            self.weights[1] += self.layers[1].T.dot(layer_deltas[2])
            self.weights[0] += self.layers[0].T.dot(layer_deltas[1])


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

# sigmoid function (S-curve)
def sigmoid(x, deriv = False):
    if(deriv == True):
        return x * (1 - x)
    return 1 / (1 + numpy.exp(-x))

# seed random numbers to make calculation deterministic
numpy.random.seed(1)
net = NeuralNet(sigmoid)
net.run(10000)
print("Output After Training:\n")
print(net)
