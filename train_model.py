import gzip
import matplotlib.pyplot as plt
import os
from sklearn.datasets import fetch_openml
from sklearn.neural_network import MLPClassifier
import numpy as np
from joblib import dump, load

def extract_data(filename, image_shape, image_number):
    with gzip.open(filename) as bytestream:
        bytestream.read(16)
        buf = bytestream.read(np.prod(image_shape) * image_number)
        data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
        data = data.reshape(image_number, image_shape[0], image_shape[1])
    return data


def extract_labels(filename, image_number):
    with gzip.open(filename) as bytestream:
        bytestream.read(8)
        buf = bytestream.read(1 * image_number)
        labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
    return labels

def train_and_save_model():
    data_base_path = os.path.join(os.pardir, 'data')
    data_folder = 'lab-03-data'

    image_shape = (28, 28)
    train_set_size = 60000
    test_set_size = 10000

    data_part2_folder = os.path.join(data_base_path, data_folder, 'part2')

    train_images_path = os.path.join(data_part2_folder, 'train-images-idx3-ubyte.gz')
    train_labels_path = os.path.join(data_part2_folder, 'train-labels-idx1-ubyte.gz')
    test_images_path = os.path.join(data_part2_folder, 't10k-images-idx3-ubyte.gz')
    test_labels_path = os.path.join(data_part2_folder, 't10k-labels-idx1-ubyte.gz')

    train_images_3d = extract_data(train_images_path, image_shape, train_set_size)
    test_images_3d = extract_data(test_images_path, image_shape, test_set_size)
    train_labels = extract_labels(train_labels_path, train_set_size)
    test_labels = extract_labels(test_labels_path, test_set_size)


    nsamples1, nx1, ny1 = train_images_3d.shape
    train_images = train_images_3d.reshape((nsamples1,nx1*ny1))

    nsamples2, nx2, ny2 = test_images_3d.shape
    test_images = test_images_3d.reshape((nsamples2,nx2*ny2))

    # initialization of the Multi-Layer Perceptron
    mlp = MLPClassifier(solver='sgd', hidden_layer_sizes=(100,), random_state=1)


    N_TRAIN_SAMPLES = train_images.shape[0]
    N_EPOCHS = 30
    N_BATCH = 256
    N_CLASSES = np.unique(train_images)

    scores_train = []
    scores_test = []

    # EPOCH
    epoch = 0
    while epoch < N_EPOCHS:
        print('epoch: ', epoch)
        # SHUFFLING
        random_perm = np.random.permutation(train_images.shape[0])
        mini_batch_index = 0
        while True:
            # MINI-BATCH
            indices = random_perm[mini_batch_index:mini_batch_index + N_BATCH]
            mlp.partial_fit(train_images[indices], train_labels[indices], classes=N_CLASSES)
            mini_batch_index += N_BATCH

            if mini_batch_index >= N_TRAIN_SAMPLES:
                break

        # SCORE TRAIN
        scores_train.append(mlp.score(train_images, train_labels))

        # SCORE TEST
        scores_test.append(mlp.score(test_images, test_labels))

        epoch += 1

        print("train accuracy: %s" % mlp.score(train_images, train_labels))

        print("test accuracy: %s" % mlp.score(test_images, test_labels))

    """ Plot """
    fig, ax = plt.subplots(2, sharex=True, sharey=True)
    ax[0].plot(scores_train)
    ax[0].set_title('Train')
    ax[1].plot(scores_test)
    ax[1].set_title('Test')
    fig.suptitle("Accuracy over epochs", fontsize=14)
    plt.show()

    # to save the model
    dump(mlp, 'model_mlp.joblib')