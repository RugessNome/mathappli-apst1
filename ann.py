

import numpy as np

class Layer(object):
    pass

def create_layer(w, b, activation, jacobi):
    l = Layer()
    l.w = w
    l.b = b
    l.activation = activation
    l.J = jacobi
    l.z = np.zeros(b.shape)
    l.a = np.zeros(b.shape)
    return l

def input_layer(n):
    activation = lambda x: x
    jacobi = lambda x,y: np.eye(n)
    return create_layer(np.eye(n), np.zeros(n), activation, jacobi)

def softmax(x):
    x = np.exp(x)
    return x / x.sum()

def jacobi_softmax(x, y):
    j = -np.outer(y, y)
    for i in range(len(x)):
        j[i][i] = y[i] * (1 - y[i])
    return j

def softmax_layer(n_input, n_output):
    return create_layer(np.random.randn(n_output, n_input), np.random.randn(n_output), softmax, jacobi_softmax)

def relu(x):
    return (x > 0) * x

def jacobi_relu(x, y):
    return np.diag((x > 0) * 1)

def relu_layer(n_input, n_output):
    return create_layer(np.random.randn(n_output, n_input), np.random.randn(n_output), relu, jacobi_relu)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def jacobi_sigmoid(x, y):
    sx = sigmoid(x)
    return np.diag(sx * (1 - sx))

def sigmoid_layer(n_input, n_output):
    return create_layer(np.random.randn(n_output, n_input), np.random.randn(n_output), sigmoid, jacobi_sigmoid)

def feed_forward(layers, x):
    for l in layers:
        x = np.dot(l.w, x) + l.b
        l.z = x
        l.a = l.activation(l.z)
    return layers[-1].a

def backprop(layers, cost_derivative):
    # Processing last layer
    l = layers[-1]
    l.delta = np.dot(l.J(l.z, l.a).T, cost_derivative)
    l.nabla_b = l.delta
    l.nabla_w = np.outer(l.delta, layers[-2].a)
    # Iterating over the other layers
    for i in range(2, len(layers)):
        l = layers[-i]
        prev_l = layers[-(i-1)]
        next_l = layers[-(i+1)]
        nabla_a = np.dot(prev_l.w.T, prev_l.delta)
        l.delta = np.dot(l.J(l.z, l.a).T, nabla_a)
        l.nabla_b = l.delta
        l.nabla_w = np.outer(l.delta, next_l.a)

def categorical_cross_entropy(x, y):
    z = 0.0
    for i in range(len(x)):
        if x[i] == 0:
            pass
        else:
            z = z + x[i] * np.log(y)
    return -z

def batch_backprop(layers, batch, eta):
    n = len(batch)
    for x, y in batch:
        feed_forward(layers, x)
        cost_derivative = np.zeros(len(layers[-1].a))
        cost_derivative[y] = -1 / layers[-1].a[y]
        backprop(layers, cost_derivative)
        for i in range(len(layers)-1):
            l = layers[-(1+i)]
            l.w = l.w - eta * l.nabla_w / n
            l.b = l.b - eta * l.nabla_b / n

def classify(layers, x):
    return np.argmax(feed_forward(layers, x))

def evaluate_classifier(layers, test_data):
    n = 0
    for x, y in test_data:
        prediction = np.argmax(feed_forward(layers, x))
        if prediction == y:
            n = n + 1
    return n

def fit(layers, training_data, epochs, batch_size, eta):
    n = len(training_data)
    for j in range(epochs):
        import random
        random.shuffle(training_data)
        batches = [training_data[k:k+batch_size] for k in range(0, n, batch_size)]
        for batch in batches:
            batch_backprop(layers, batch, eta)
        print("Epoch {0} : {1} / {2}".format(j, evaluate_classifier(layers, training_data), n))
        print("Epoch {0} complete".format(j))


# Example

def get_class(x):
    if x[1] <= -0.25:
        return 0
    elif x[0] <= 0:
        return 1
    return 2

def gen_data(n):
    ret = []
    for i in range(n):
        x = np.random.uniform(-1, 1, size=2)
        y = get_class(x)
        ret.append((x,y))
    return ret

def plot_points(points):
    import matplotlib.pyplot as plt
    X, Y, Z = [], [], []
    for (x, y), z in points:
        X.append(x)
        Y.append(y)
        Z.append(z)
    fig = plt.figure()
    plt.axis([-1, 1, -1, 1])
    plt.scatter(X, Y, c=Z)
    plt.show()


def main():
    data = gen_data(10000)
    plot_points(data)

    layers = []
    layers.append(input_layer(2))
    layers.append(relu_layer(2, 4))
    layers.append(softmax_layer(4, 3))
    
    predictions = []
    for x, _ in data:
        predictions.append((x, classify(layers, x)))

    plot_points(predictions)

    fit(layers, data, 20, 500, 1)

    predictions = []
    for x, _ in data:
        predictions.append((x, classify(layers, x)))

    plot_points(predictions)

if __name__ == '__main__':
    main()