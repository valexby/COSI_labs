#!/usr/bin/env python3
import json
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import click

IMAGES_PATH = "./cosi/signs10x10/"
IMAGES_COUNT = 5
SAMPLE_LENGTH = 20
NOISE_LEVELS = [1, 2, 3, 4, 5]
EXPECTED_ERROR = 0.10

@click.group()
def cli():
    pass

def read_image(path):
    origin = mpimg.imread(path)
    image = np.ndarray(origin.shape[:2])
    for i in range (image.shape[0]):
        for j in range(image.shape[1]):
            if origin[i][j][0]:
                image[i][j] = 1
            else:
                image[i][j] = 0
    return image

def negative(image):
    for i in np.nditer(image, op_flags=['readwrite']):
        i[...] = int(not i)

def make_noise(origin, noise_level):
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

def nonlin(x, deriv=False):  # Note: there is a typo on this line in the video
    if(deriv==True):
        return (x*(1-x))

    return 1/(1+np.exp(-x))  # Note: there is a typo on this line in the video

def train(origins):
    Y = []
    X = []
    for noise in NOISE_LEVELS:
        for i in range(len(origins)):
            for _ in range(SAMPLE_LENGTH):
                X.append(make_noise(origins[i], noise).flatten())
                Y.append(np.zeros((IMAGES_COUNT,)))
                Y[-1][i] = 1

    np.random.seed()
    H = int((len(X)/len(X[0])) ** 0.5)
    V = 2*np.random.random((len(X[0]), H)) - 1
    W = 2*np.random.random((H, IMAGES_COUNT)) - 1
    Q = 2*np.random.random((H,)) - 1
    T = 2*np.random.random((IMAGES_COUNT,)) - 1

    max_error = -1
    alpha = 0.15
    while max_error == -1 or max_error > EXPECTED_ERROR:
        for x, y in zip(X, Y):

            l0 = x

            l1 = nonlin(np.dot(l0, V) + Q)
            l2 = nonlin(np.dot(l1, W) + T)

            l2_error = y - l2
            if np.abs(l2_error).max() > max_error:
                max_error = l2_error.max()
            l2_delta = l2_error*nonlin(l2, deriv=True)*alpha
            l1_error = (l2_error*nonlin(l2, deriv=True)).dot(W.T)
            l1_delta = l1_error * nonlin(l1, deriv=True)*alpha

            W += l1[np.newaxis].T * l2_delta
            T += l2_delta
            V += l0[np.newaxis].T * l1_delta
            Q += l1_delta
        print(max_error)
        if alpha > max_error and alpha > 0.05:
            alpha = max_error
            print("Max Error: {}".format(max_error))
    print("Error mean : {}".format(np.abs(l2_error).mean()))
    return [W, V, T, Q]

def clust(origins, network, noise):
    samples = [make_noise(origin, noise) for origin in origins]

    W, V, T, Q = network

    grid = gridspec.GridSpec(IMAGES_COUNT+1, IMAGES_COUNT+1, top=1., bottom=0., right=1., left=0., hspace=0.,
                           wspace=0.)
    grid = [plt.subplot(cell) for cell in grid]
    for ax in grid:
        ax.set_xticks([])
        ax.set_yticks([])
    clusters = {}
    np.set_printoptions(precision=2)
    for origin in origins:
        l0 = origin.flatten()
        l1 = nonlin(np.dot(l0, V) + Q)
        l2 = nonlin(np.dot(l1, W) + T)
        clusters[l2.tolist().index(l2.max())] = origin
    if (len(clusters) != IMAGES_COUNT):
        print("NN doesn't work!")
        return
    for i in range(IMAGES_COUNT):
        grid[i+1].imshow(clusters[i])

    for i in range(IMAGES_COUNT):
        pos = (i + 1) * (IMAGES_COUNT + 1)
        grid[pos].imshow(samples[i])
        l0 = samples[i].flatten()
        l1 = nonlin(np.dot(l0, V) + Q)
        l2 = nonlin(np.dot(l1, W) + T)
        l2 /= l2.sum()
        l2 *= 100
        l2 = np.around(l2)
        for j in range(IMAGES_COUNT):
            grid[pos + j + 1].text(0.5, 0.5, str(l2[j]))

    plt.savefig('output.png')

@cli.command(help='Neural network clusterisation process')
@click.argument('noise', required=True, type=click.INT)
def main(noise):
    origins = [read_image(IMAGES_PATH + "sign_{}.png".format(x)) for x in range(1, IMAGES_COUNT + 1)]
    network = train(origins)
    clust(origins, network, noise)

if __name__ == "__main__":
    cli()
