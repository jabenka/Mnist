import numpy as np


class Layer:
    def __init__(self, neurons, weights, biases):
        self.neurons = neurons
        self.weights = weights
        self.biases = biases


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def get_init_layer(img):
    a = np.asarray(img).flatten() / 255.0
    i_layer = Layer(np.array(a), None, None)
    return i_layer


def initialize_layers(layer_sizes):
    layers = [Layer(None, None, None)]
    for i in range(1, len(layer_sizes)):
        in_count = layer_sizes[i - 1]
        out_count = layer_sizes[i]
        w = np.random.randn(in_count, out_count) * 0.01
        biases = np.zeros(out_count)
        layers.append(Layer(None, w, biases))
    return layers


def forward(layers, i_layer):
    layers[0] = i_layer
    for i in range(1, len(layers)):
        z = layers[i - 1].neurons @ layers[i].weights + layers[i].biases
        if i == len(layers) - 1:
            layers[i].neurons = softmax(z)
        else:
            layers[i].neurons = sigmoid(z)
    return layers


def backward(layers, label, learning_rate=0.1):
    out_size = layers[-1].neurons.shape[0]
    true_vector = np.zeros(out_size)
    true_vector[label] = 1

    dZ = layers[-1].neurons - true_vector

    gradients = [None] * len(layers)

    for i in range(len(layers) - 1, 0, -1):
        prev_activation = layers[i - 1].neurons
        dW = np.outer(prev_activation, dZ)
        db = dZ
        gradients[i] = (dW, db)
        if i > 1:
            dA_prev = dZ @ layers[i].weights.T
            dZ = dA_prev * prev_activation * (1-prev_activation)


    for i in range(1,len(layers)):
        dW,db = gradients[i]
        layers[i].weights -= learning_rate * dW
        layers[i].biases -= learning_rate * db

    return layers

def cost(out_activations, label):
    return -np.log(out_activations[label] + 1e-9)

def training(layers,train_df, learning_rate=0.1,shuffle=True):
    df = train_df.sample(frac=1).reset_index(drop=True) if shuffle else train_df
    n = len(df)
    total_loss = 0
    for i in range(n):
        image = df.iloc[i]['image']
        label = df.iloc[i]['label']
        i_layer = get_init_layer(image)
        forward(layers, i_layer)
        loss = cost(layers[-1].neurons, label)
        total_loss += loss
        backward(layers, label, learning_rate)
        if i % 5000 == 0:
            print(f"Example {i}/{n}, loss: {loss:.4f}")
    avg_loss = total_loss / n
    print(f"Average loss: {avg_loss:.4f}")
    return avg_loss

def train_epochs(epoch_count,layers,train_df,learning_rate=0.1,shuffle=True):
    for epoch in range(epoch_count):
        print(f"--- Epoch {epoch + 1}/{epoch_count} ---")
        training(layers, train_df, learning_rate=learning_rate,shuffle = shuffle)
    return layers

def predict(image,layers):
    i_layer = get_init_layer(image)
    forward(layers,i_layer)
    return layers[-1].neurons