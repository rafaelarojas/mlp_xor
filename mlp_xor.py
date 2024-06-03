import numpy as np

class MLP:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.1):
        self.weights_input_hidden = np.random.rand(input_size, hidden_size)
        self.weights_hidden_output = np.random.rand(hidden_size, output_size)
        self.bias_hidden = np.random.rand(hidden_size)
        self.bias_output = np.random.rand(output_size)
        self.learning_rate = learning_rate

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def _sigmoid_derivative(self, x):
        return x * (1 - x)

    def _forward_pass(self, inputs):
        self.hidden_input = np.dot(inputs, self.weights_input_hidden) + self.bias_hidden
        self.hidden_output = self._sigmoid(self.hidden_input)
        
        self.final_input = np.dot(self.hidden_output, self.weights_hidden_output) + self.bias_output
        self.final_output = self._sigmoid(self.final_input)
        
        return self.final_output

    def _backward_pass(self, inputs, targets):
        output_error = targets - self.final_output
        output_delta = output_error * self._sigmoid_derivative(self.final_output)
        
        hidden_error = np.dot(output_delta, self.weights_hidden_output.T)
        hidden_delta = hidden_error * self._sigmoid_derivative(self.hidden_output)
        
        self.weights_hidden_output += self.learning_rate * np.dot(self.hidden_output.reshape(-1, 1), output_delta.reshape(1, -1))
        self.bias_output += self.learning_rate * output_delta
        
        self.weights_input_hidden += self.learning_rate * np.dot(inputs.reshape(-1, 1), hidden_delta.reshape(1, -1))
        self.bias_hidden += self.learning_rate * hidden_delta

    def train(self, inputs, targets, epochs=10000):
        for _ in range(epochs):
            for input_data, target in zip(inputs, targets):
                self._forward_pass(input_data)
                self._backward_pass(input_data, target)

    def predict(self, inputs):
        return self._forward_pass(inputs)

inputs = np.array([[0, 0],
                   [0, 1],
                   [1, 0],
                   [1, 1]])

targets = np.array([[0],
                    [1],
                    [1],
                    [0]])

mlp = MLP(input_size=2, hidden_size=2, output_size=1, learning_rate=0.1)

mlp.train(inputs, targets, epochs=10000)

for input_data in inputs:
    output = mlp.predict(input_data)
    print(f"Input: {input_data}")
    print(f"Output: {output}")
    print(f"---------------------")
