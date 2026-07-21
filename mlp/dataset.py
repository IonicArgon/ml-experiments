import numpy as np
import matplotlib.pyplot as plt


def idx_to_nparray(file):
    """return a numpy array from an IDX file"""

    byte_to_dtype = {
        0x08: "B",
        0x09: "b",
        0x0B: ">i2",
        0x0C: ">i4",
        0x0D: ">f4",
        0x0E: ">f8",
    }

    raw_array = np.fromfile(file, dtype=">B")

    data_type = byte_to_dtype[(raw_array[2]).item()]
    num_dims = (raw_array[3]).item()
    dims = raw_array[4 : 4 + num_dims * 4].view(">u4")

    data = (raw_array[4 + num_dims * 4 :]).view(data_type).reshape(dims)

    return data


def to_ascii(image: np.typing.NDArray):
    """for in-training visualization"""

    image = image.reshape((-1, 28))
    ramp = " .:-=+*#%@"

    vmax = image.max()
    index_array = (image / vmax * (len(ramp) - 1)).astype(int)
    chars_array = np.array(list(ramp))[index_array]
    double_chars = np.repeat(chars_array, 2, axis=1)

    print("\n".join("".join(row) for row in double_chars))


class Loader:
    """just a fancy class to wrap batches in an iterator"""

    def __init__(
        self,
        X: np.typing.NDArray,
        y: np.typing.NDArray,
        batch_size: int,
        shuffle=False,
        seed=None,
    ):
        self.X: np.typing.NDArray = X
        self.y: np.typing.NDArray = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.rng = np.random.default_rng(seed)

    def __iter__(self):
        n = len(self.X)

        if self.shuffle:
            perm = self.rng.permutation(n)
        else:
            perm = np.arange(n)

        for start in range(0, n, self.batch_size):
            idx = perm[start : start + self.batch_size]
            yield self.X[idx], self.y[idx]

    def __len__(self):
        return len(range(0, len(self.X), self.batch_size))


def load_mnist(data_dir, batch_size, val_size=5000, seed=0):
    """
    read the MNIST data and return three loaders:
    - `train`: for the training loop
    - `validate`: for validation during training
    - `test`: for evaluation of the model
    
    all loaders use the same Loader class, but `train` shuffles on new epochs
    """

    rng = np.random.default_rng(seed)

    raw_train_images = idx_to_nparray(f"{data_dir}/train-images-idx3-ubyte")
    raw_train_labels = idx_to_nparray(f"{data_dir}/train-labels-idx1-ubyte").astype(
        np.int64
    )
    raw_test_images = idx_to_nparray(f"{data_dir}/t10k-images-idx3-ubyte")
    test_labels = idx_to_nparray(f"{data_dir}/t10k-labels-idx1-ubyte").astype(np.int64)

    # have to flatten and normalize the images for 784-input into mlp
    train_n = len(raw_train_labels)
    test_n = len(test_labels)

    train_img_flat = raw_train_images.reshape((train_n, 784)).astype(np.float32) / 255
    test = raw_test_images.reshape((test_n, 784)).astype(np.float32) / 255

    perm = rng.permutation(train_n)
    train_images_perm = train_img_flat[perm]
    train_labels_perm = raw_train_labels[perm]

    # create train/validate split
    train, validate = np.split(train_images_perm, [train_n - val_size])
    train_labels, validate_labels = np.split(train_labels_perm, [train_n - val_size])

    loader_train = Loader(train, train_labels, batch_size, True, seed)
    loader_validate = Loader(validate, validate_labels, batch_size)
    loader_test = Loader(test, test_labels, batch_size)

    return loader_train, loader_validate, loader_test


if __name__ == "__main__":
    data_path = "./data"
    train, validate, test = load_mnist(data_path, 64)

    # just a bunch of shape and data assertions
    assert train.X.shape == (55000, 784)
    assert train.X.max() == 1.0 and train.X.min() == 0.0
    assert train.X.dtype == np.float32
    assert train.y.dtype == np.int64
    assert len(train) == 860

    assert validate.X.shape == (5000, 784)
    assert validate.X.max() == 1.0 and validate.X.min() == 0.0
    assert validate.X.dtype == np.float32
    assert validate.y.dtype == np.int64
    assert len(validate) == 79

    assert test.X.shape == (10000, 784)
    assert test.X.max() == 1.0 and test.X.min() == 0.0
    assert test.X.dtype == np.float32
    assert test.y.dtype == np.int64
    assert len(test) == 157

    # asserting that the first batch of each epoch does not equal the first
    # batch of a previous epoch when shuffle = True
    epoch_1_X, epoch_1_y = next(iter(train))
    epoch_2_X, epoch_2_y = next(iter(train))
    assert not np.array_equal(epoch_1_X, epoch_2_X) and not np.array_equal(
        epoch_1_y, epoch_2_y
    )

    # inverse test on validate because it's not supposed to shuffle
    epoch_1_X, epoch_1_y = next(iter(validate))
    epoch_2_X, epoch_2_y = next(iter(validate))
    assert np.array_equal(epoch_1_X, epoch_2_X) and np.array_equal(epoch_1_y, epoch_2_y)

    # and assert that after a whole epoch, the bincounts align
    acc = np.array([], dtype=np.int64)
    for X, y in train:
        acc = np.concat((acc, y), axis=None)
    assert np.array_equal(np.bincount(train.y), np.bincount(acc))

    # visual confirmation that labels and images align
    fig, axs = plt.subplots(5, 3)

    for i, (X, y) in enumerate(train):
        if i >= 5:
            break

        images = X.reshape((-1, 28, 28))[:3]
        labels = y[:3]
        for j, image in enumerate(images):
            axs[i, j].imshow(image, cmap="gray_r", interpolation="nearest")
            axs[i, j].set_title(f"Label: {labels[j]}")
            axs[i, j].set_yticks([])
            axs[i, j].set_xticks([])

            if j == 0:
                axs[i, j].set_ylabel(f"batch {i}")

    plt.tight_layout()
    plt.show()
