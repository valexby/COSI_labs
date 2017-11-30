#!/usr/bin/env python3
import click, os, theano, sys
import numpy as np
import matplotlib
matplotlib.rcParams['backend'] = "Qt4Agg"
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
    network.train(X, Y, epochs=20)
    return network

def get_number(origins, symbol_dict, network):
    grid = gridspec.GridSpec(4, len(origins), top=1., bottom=0., right=1., left=0., hspace=0.,
                           wspace=0.)
    grid = np.array([plt.subplot(cell) for cell in grid]).reshape((4, len(origins)))
    for row in grid:
        for ax in row:
            ax.set_xticks([])
            ax.set_yticks([])
    images = np.array([origin.flatten() for origin in origins])
    images = images / 255
    images = images - images.mean(axis=0)
    positions = []
    commullative = 1
    for i in range(images.shape[0]):
        out = network.predict(images[i])[0]
        out = np.around((out / out.sum()) * 100)
        commullative *= (out.max()) / 100
        positions.append(out.argmax())
        grid[0][i].imshow(origins[i])
        for j in range(1, 4):
            pos = out.argmax()
            grid[j][i].text(0.2, 0.5, "{} - {}%".format(symbol_dict[pos], str(out[pos])))
            out[pos] = 0
    plt.savefig('output.png', fmt='png')
    plt.show()
    print(commullative)
    return "".join([symbol_dict[pos] for pos in positions])

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
    print(get_number(origins, symbol_dict, network))


if __name__ == "__main__":
    main()
