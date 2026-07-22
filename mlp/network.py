import numpy as np
import math


class Dense:
    def __init__(self, n_in, n_out, rng: np.random.Generator, dtype=np.float32):
        self.b = np.zeros(n_out, dtype)

        # He initialization for weights
        self.W = rng.standard_normal((n_in, n_out), dtype) * math.sqrt(2 / n_in)

    def forward(self, X: np.ndarray):
        self.X = X
        return X @ self.W + self.b

    def backward(self, dZ: np.ndarray):
        self.dW = self.X.T @ dZ
        self.db = dZ.sum(axis=0)

        assert self.dW.shape == self.W.shape
        assert self.db.shape == self.b.shape

        return dZ @ self.W.T

    def step(self, lr):
        self.W -= lr * self.dW
        self.b -= lr * self.db


class ReLU:
    def forward(self, z: np.ndarray):
        self.mask = z > 0
        return z * self.mask

    def backward(self, dA: np.ndarray):
        return dA * self.mask

    def step(self, lr):
        pass


def softmax_cross_entropy(logits: np.ndarray, y: np.ndarray):
    # softmax logits to probabilities
    z = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(z)
    probs = exp / exp.sum(axis=1, keepdims=True)

    # then calculate cross-entropy loss
    B = len(y)
    correct = probs[np.arange(B), y]
    loss = -np.log(correct).mean()

    # calculate dZ for backprop
    dZ = probs.copy()
    dZ[np.arange(B), y] -= 1
    dZ /= B

    return loss, dZ


class MLP:
    def __init__(
        self, n_in, n_hidden, n_out, seed=0, dtype: type[np.floating] = np.float32
    ):
        rng = np.random.default_rng(seed)

        self.layers = [
            Dense(n_in, n_hidden, rng, dtype),
            ReLU(),
            Dense(n_hidden, n_out, rng, dtype),
        ]

    def forward(self, X: np.ndarray):
        for layer in self.layers:
            X = layer.forward(X)
        return X

    def backward(self, dZ: np.ndarray):
        for layer in reversed(self.layers):
            dZ = layer.backward(dZ)

    def step(self, lr):
        for layer in self.layers:
            layer.step(lr)


def grad_check(net: MLP, param, grad, X, y, num_samples=20, eps=1e-5, layer=""):
    loss_only = lambda: softmax_cross_entropy(net.forward(X), y)[0]

    rng = np.random.default_rng(seed=123)
    idx = rng.integers(0, param.size, num_samples)

    for i in idx:
        orig = param.flat[i]

        param.flat[i] = orig + eps
        L_plus = loss_only()

        param.flat[i] = orig - eps
        L_minus = loss_only()

        param.flat[i] = orig

        numerical = (L_plus - L_minus) / (2 * eps)
        analytic = grad.flat[i]
        rel = abs(analytic - numerical) / max(abs(analytic) + abs(numerical), 1e-12)

        if rel > 1e-7:
            print(f"{layer}: rel {rel} of ({i}) exceeds 1e-7")


if __name__ == "__main__":
    net = MLP(4, 3, 2, seed=123, dtype=np.float64)
    B = 5

    rng = np.random.default_rng()
    X = rng.random((B, 4), dtype=np.float64)
    y = rng.integers(0, 2, size=B)

    logits = net.forward(X)
    loss, dZ = softmax_cross_entropy(logits, y)
    net.backward(dZ)

    grad_check(net, net.layers[0].W, net.layers[0].dW, X, y, layer="W of 0")
    grad_check(net, net.layers[0].b, net.layers[0].db, X, y, layer="b of 0")
    grad_check(net, net.layers[2].W, net.layers[2].dW, X, y, layer="W of 2")
    grad_check(net, net.layers[2].b, net.layers[2].db, X, y, layer="b of 2")
