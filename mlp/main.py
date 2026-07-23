from dataset import load_mnist
from network import MLP, softmax_cross_entropy, softmax
import matplotlib.pyplot as plt


def overfit_test(train):
    net = MLP(28 * 28, 392, 10, seed=123)
    
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
    fig, axs = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3), constrained_layout=True)

    for i in range(cols):
        axs[0, i].imshow(train_X_10[i].reshape(28, 28), cmap="gray_r",  interpolation="nearest")
        axs[0, i].set_title(f"true label: {train_y_10[i]}")
        axs[0, i].set_yticks([])
        axs[0, i].set_xticks([])

        axs[1, i].bar(range(10), probs[i])
        axs[1, i].set_xticks(range(10))
        axs[1, i].set_title(f"predicted labels for image {i}")
    
    plt.show()

if __name__ == "__main__":
    train, validate, test = load_mnist("./data", 10)

    overfit_test(train)
