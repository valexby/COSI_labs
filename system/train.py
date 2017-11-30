#!/usr/bin/env python3
import click, os, theano, sys
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn import metrics
from sklearn.model_selection import train_test_split
from neupy import environment
from neupy import algorithms, layers


IMAGES_PATH = "./r/"
IMAGES_COUNT = 5
SAMPLE_LENGTH = 24
NOISE_LEVELS = [1, 2, 3, 4, 5]
EXPECTED_ERROR = 0.10
INIT_ALPHA = 0.30
NUMBER_PATH = './control/'

@click.group()
def cli():
    pass

def read_image(path):
    """Загрузка изображения и приведение к бинарному виду"""
    origin = mpimg.imread(path)
    image = np.ndarray(origin.shape[:2])
    for i in range (image.shape[0]):
        for j in range(image.shape[1]):
            if origin[i][j] > 70:
                image[i][j] = 1
            else:
                image[i][j] = 0
    return image

def negative(image):
    """Инвертирование всех битов изображения"""
    for i in np.nditer(image, op_flags=['readwrite']):
        i[...] = int(not i)

def make_noise(origin, noise_level):
    """Зашумление изображения рандомными шумами"""
    image = origin.copy()
    direction = noise_level < (origin.size / 2)
    if not direction:
        noise_level = (image.size // 2) - noise_level
        negative(image)

    while noise_level:
        x = np.random.randint(0, image.shape[0])
        y = np.random.randint(0, image.shape[1])
        if (direction and image[x][y] != origin[x][y]) or \
           (not direction and image[x][y] == origin[x][y]):
            continue
        image[x][y] = int(not image[x][y])
        noise_level -= 1

    return image

def nonlin(x, deriv=False):
    """Сигмоидная функция. Индус зачем-то сделал тут их две.
    Они как бы обе нужны, но зачем пихать все в один вызов с ключем"""
    if(deriv==True):
        return (x*(1-x))

    return 1/(1+np.exp(-x))

def train(X, Y):

    environment.reproducible()
    x_train, x_test, y_train, y_test = train_test_split(
        X.astype(np.float32),
        Y.astype(np.float32),
        train_size=(6. / 7)
    )
    theano.config.floatX = 'float32'
    img_size = X.shape[1]
    network = algorithms.Momentum(
        [
            layers.Input(img_size),
            layers.Relu(100),
            layers.Softmax(Y.shape[1]),
        ],
        error='categorical_crossentropy',
        step=0.01,
        verbose=True,
        shuffle_data=True,
        momentum=0.9,
        nesterov=True,
    )
    network.architecture()
    network.train(x_train, y_train, x_test, y_test, epochs=20)
    y_predicted = network.predict(x_test).argmax(axis=1)
    y_test = np.asarray(y_test.argmax(axis=1)).reshape(len(y_test))
    print(metrics.classification_report(y_test, y_predicted))
    return network

def get_number(im_number, symbol_dict, network, origins):
    grid = gridspec.GridSpec(im_number.shape[0], 2, top=1., bottom=0., right=1., left=0., hspace=0.,
                           wspace=0.)
    grid = [plt.subplot(cell) for cell in grid]
    for ax in grid:
        ax.set_xticks([])
        ax.set_yticks([])
    positions = []
    for i in range(im_number.shape[0]):
        out = network.predict(im_number[i])
        positions.append(out.argmax())
        grid[i*2].imshow(origins[i])
        grid[(i*2)+1].text(0.5, 0.5, symbol_dict[positions[-1]])
    plt.savefig('output.png')
    out = [symbol_dict[pos] for pos in positions]
    return "".join(out)

def main():
    image_names = os.listdir(IMAGES_PATH)
    origins = []
    max_number = 0
    known_clusters = {}
    symbol_dict = {}
    target = []
    for name in image_names:
        origins.append(mpimg.imread(IMAGES_PATH + name).flatten())
        if name[0] not in known_clusters:
            known_clusters[name[0]] = max_number
            symbol_dict[max_number] = name[0]
            max_number += 1
        target.append(known_clusters[name[0]])
    origins = np.array(origins)
    origins = origins / 255
    origins = origins - origins.mean(axis=0)
    positions = np.zeros((origins.shape[0], max(target) + 1))

    for i in range(positions.shape[0]):
        positions[i][target[i]] = 1

    network = train(origins, positions)
    number_path = os.listdir(sys.argv[1])
    number_path.sort()
    origins = []
    for symbol_path in number_path:
        origins.append(mpimg.imread(sys.argv[1] + symbol_path))
    inputs = np.array([origin.flatten() for origin in origins])
    origins = np.array(origins)
    inputs = inputs / 255
    inputs = inputs - inputs.mean(axis=0)
    print(get_number(inputs, symbol_dict, network, origins))


if __name__ == "__main__":
    main()
