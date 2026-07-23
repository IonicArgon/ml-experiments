from dataset import load_mnist, Loader
from network import MLP, softmax_cross_entropy, softmax
import matplotlib.pyplot as plt
import numpy as np


def overfit_test(train: Loader, net: MLP):
    # just grabbing 10 images to overfit
    train_X_10, train_y_10 = next(iter(train))

    epochs = 500
    for epoch in range(epochs):
        logits = net.forward(train_X_10)
        loss, dZ = softmax_cross_entropy(logits, train_y_10)
        net.backward(dZ)
        net.step(0.01)

        print(f"epoch {epoch} | overfit test loss: {loss}")

    # visual inspection
    logits = net.forward(train_X_10)
    probs = softmax(logits)

    rows, cols = 2, 10
    _, axs = plt.subplots(
        rows, cols, figsize=(cols * 3, rows * 3), constrained_layout=True
    )

    for i in range(cols):
        axs[0, i].imshow(
            train_X_10[i].reshape(28, 28), cmap="gray_r", interpolation="nearest"
        )
        axs[0, i].set_title(f"true label: {train_y_10[i]}")
        axs[0, i].set_yticks([])
        axs[0, i].set_xticks([])

        axs[1, i].bar(range(10), probs[i])
        axs[1, i].set_xticks(range(10))
        axs[1, i].set_title(f"predicted labels for image {i}")

    plt.show()


def actual_training(train: Loader, validate: Loader, test: Loader, net: MLP):
    epochs = 15
    lr = 0.1

    for epoch in range(epochs):
        for X, y in train:
            logits = net.forward(X)
            _, dZ = softmax_cross_entropy(logits, y)
            net.backward(dZ)
            net.step(lr)

        total_hits = 0
        running_loss = 0
        total = 0
        for X, y in validate:
            logits = net.forward(X)
            predictions = logits.argmax(axis=1)
            loss, _ = softmax_cross_entropy(logits, y)

            total_hits += (predictions == y).sum()
            running_loss += loss * len(y)
            total += len(y)

        epoch_accuracy = total_hits / total
        epoch_validation_loss = running_loss / total
        print(
            f"epoch {epoch} | acc: {epoch_accuracy * 100:.2f}%, avg loss: {epoch_validation_loss}"
        )

    # and then testing
    total_hits = 0
    total = 0
    for X, y in test:
        logits = net.forward(X)
        predictions = logits.argmax(axis=1)
        total_hits += (predictions == y).sum()
        total += len(y)

    accuracy =  total_hits / total
    print(
        f"final accuracy: {accuracy * 100:.2f}%"
    )


if __name__ == "__main__":
    net = MLP(28 * 28, 392, 10, seed=123)
    train, validate, test = load_mnist("./data", 64, seed=123)

    actual_training(train, validate, test, net)
