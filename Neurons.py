from numpy import dot, power, array, exp, zeros, clip, log, sum as sum_total, max as max_value
from numpy.random import uniform as random_float
from datasets import load_dataset

# Load MNIST
mnist = load_dataset('mnist')
one_item = mnist['train'][0]
answer = one_item['label']
print(one_item)

# Network setup
def preprocess_image(image):
    return array(image).reshape(1, 784) / 255.0


inputs = preprocess_image(one_item['image'])
nodes = [784, 128, 64, 10]




class Layer:
    def __init__(self, input_size, output_size):
        self.weights = random_float(-1, 1, (input_size, output_size))
        self.biases = random_float(-1, 1, (1, output_size)) 
        self.input = None
        self.output = None
        self.grad_w = None
        self.grad_b = None

        
    def forward(self, inputs):
        self.input = inputs
        self.output = dot(inputs, self.weights) + self.biases
        return self.output

    def update(self, learning_rate): 
        self.weights -= learning_rate * self.grad_w 
        self.biases -= learning_rate * self.grad_b

# Create layers
layers = []
for i in range(len(nodes) - 1):
    layers.append(Layer(nodes[i], nodes[i+1]))

# Activation function (optional)
def Activation_Function(x):
    return 0.2 * power(x, 4) + 0.1 * power(x, 3) - power(x, 2) + 2

def softmax(x):
    e_x = exp(x - max_value(x, axis=1, keepdims=True))  # prevent overflow
    return e_x / sum_total(e_x, axis=1, keepdims=True)

def relu(x):
    return x * (x > 0)

def relu_derivative(x):
    return (x > 0).astype(float)

# Forward pass
def calculate_output(inputs):
    inputs = array(inputs)
    for i, layer in enumerate(layers):
        inputs = layer.forward(inputs)
        if i < len(layers) - 1:
            inputs = relu(inputs)  # hidden layers
        else:
            inputs = softmax(inputs)  # final layer
    return inputs

def one_hot(label, num_classes=10):
    one_hot_vector = zeros((1, num_classes))
    one_hot_vector[0][label] = 1
    return one_hot_vector

def cross_entropy_loss(predictions, targets):
    epsilon = 1e-12  # prevent log(0)
    predictions = clip(predictions, epsilon, 1. - epsilon)
    loss = -sum_total(targets * log(predictions)) / predictions.shape[0]
    return loss.item()


learning_rate = 0.01
epochs = 5

for epoch in range(epochs):
    total_loss = 0
    for example in mnist['train']:
        x = preprocess_image(example['image'])        # Shape (1, 784)
        y = one_hot(example['label'])                 # Shape (1, 10)

        # --- Forward pass ---
        activations = [x]
        for i, layer in enumerate(layers):
            x = layer.forward(x)
            if i < len(layers) - 1:
                x = relu(x)
                layer.output = x  # Store post-activation for backprop
            else:
                x = softmax(x)
            activations.append(x)

        prediction = activations[-1]
        loss = cross_entropy_loss(prediction, y)
        total_loss += loss

        # --- Backward pass ---
        grad = prediction - y  # Gradient of loss w.r.t. output

        for i in reversed(range(len(layers))):
            layer = layers[i]
            if i == len(layers) - 1:
                # Output layer (no activation derivative, already applied)
                grad_input = dot(grad, layer.weights.T)
                layer.grad_w = dot(activations[i].T, grad)
                layer.grad_b = sum_total(grad, axis=0)

            else:
                grad = grad_input * relu_derivative(layer.output)
                layer.grad_w = dot(activations[i].T, grad)
                layer.grad_b = sum_total(grad, axis=0)
                grad_input = dot(grad, layer.weights.T)

        # --- Update weights ---
        for layer in layers:
            layer.update(learning_rate)

    print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}")


