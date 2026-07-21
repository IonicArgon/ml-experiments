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
    image = image.reshape((-1, 28))
    ramp = " .:-=+*#%@"

    vmax = image.max()
    index_array = (image / vmax * (len(ramp) - 1)).astype(int)
    chars_array = np.array(list(ramp))[index_array]
    double_chars = np.repeat(chars_array, 2, axis=1)

    print("\n".join("".join(row) for row in double_chars))


class Loader:
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
    rng = np.random.default_rng(seed)

    raw_train_images = idx_to_nparray(f"{data_dir}/train-images-idx3-ubyte")
    raw_train_labels = idx_to_nparray(f"{data_dir}/train-labels-idx1-ubyte").astype(np.int64)
    raw_test_images = idx_to_nparray(f"{data_dir}/t10k-images-idx3-ubyte")
    test_labels = idx_to_nparray(f"{data_dir}/t10k-labels-idx1-ubyte").astype(np.int64)

    train_n = len(raw_train_labels)
    test_n = len(test_labels)

    train_img_flat = raw_train_images.reshape((train_n, 784)).astype(np.float32) / 255
    test = raw_test_images.reshape((test_n, 784)).astype(np.float32) / 255

    perm = rng.permutation(train_n)
    train_images_perm = train_img_flat[perm]
    train_labels_perm = raw_train_labels[perm]

    train, validate = np.split(train_images_perm, [train_n - val_size])
    train_labels, validate_labels = np.split(train_labels_perm, [train_n - val_size])

    loader_train = Loader(train, train_labels, batch_size, True, seed)
    loader_validate = Loader(validate, validate_labels, batch_size)
    loader_test = Loader(test, test_labels, batch_size)

    return loader_train, loader_validate, loader_test


if __name__ == "__main__":
    data_path = "./data"
    train, validate, test = load_mnist(data_path, 64)

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

    # dir_images = "./data/t10k-images-idx3-ubyte"
    # dir_labels = "./data/t10k-labels-idx1-ubyte"

    # images = idx_to_nparray(dir_images)
    # labels = idx_to_nparray(dir_labels)

    # fig, axs = plt.subplots(3, 3)

    # for row in range(3):
    #     for col in range(3):
    #         axs[row, col].imshow(
    #             images[row * 3 + col], cmap="gray_r", interpolation="nearest"
    #         )
    #         axs[row, col].set_title(labels[row * 3 + col])
    #         axs[row, col].set_axis_off()
    #         to_ascii(images[row * 3 + col])

    # plt.tight_layout()
    # plt.show()
