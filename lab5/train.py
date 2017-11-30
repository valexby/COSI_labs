#!/usr/bin/env python3
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
INIT_ALPHA = 0.15

@click.group()
def cli():
    pass

def read_image(path):
    """Загрузка изображения и приведение к бинарному виду"""
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

def train(origins):
    """Функция обучния.
    Названы переменные в соответствии с обозначениями в методе,
    кроме того, что у меня скрытый слой - H, а у них G"""
    Y = []
    X = []

    """Закоментил здесь кусок с генерацией зашумленной обучающей выборкой"""
    # for noise in NOISE_LEVELS:
    #     """Генерация обучающей выборки. Генерирую SAMPLE_LENGTH зашумленных изображений
    #     для каждой картинки для каждого уровня шума, заданного в NOISE_LEVELS"""
    #     for i in range(len(origins)):
    #         for _ in range(SAMPLE_LENGTH):
    #             X.append(make_noise(origins[i], noise).flatten()) #Рандомно зашумил изображение. flatten развернет матрицу в 1-D массив - вектор
    #             Y.append(np.zeros((IMAGES_COUNT,)))#Y - это правильные ответы,  по которым будет обучаться сеть. Создал вектор нулей
    #             Y[-1][i] = 1 # и одному нейрону с номером текущего изображения присваиваю 1,

    """Здесь я просто пять оригинальных картинок в X загоняю"""
    for i in range(len(origins)):
        X.append(origins[i].flatten())
        Y.append(np.zeros((IMAGES_COUNT,)))#Y - это правильные ответы,  по которым будет обучаться сеть. Создал вектор нулей
        Y[-1][i] = 1 # и одному нейрону с номером текущего изображения присваиваю 1,


    np.random.seed()
    H = int((len(X)/len(X[0])) ** 0.5) # Размер скрытого слоя - корень квадратный из отношения размера обучающей выбокри к количеству нейронов во входном слое
    H = 8 # !!!!! А вот тут вот хак, как заставить сеть обучаться на пятикартинок. Просто задал размер скрытого слоя, без всяких отношений.
    V = 2*np.random.random((len(X[0]), H)) - 1
    W = 2*np.random.random((H, IMAGES_COUNT)) - 1
    Q = 2*np.random.random((H,)) - 1
    T = 2*np.random.random((IMAGES_COUNT,)) - 1

    max_error = -1
    alpha = INIT_ALPHA #Начальная скорость обучения, можно смело править. Но у меня и с этим сходилось всегда
    while max_error == -1 or max_error > EXPECTED_ERROR: # Продолжаем обучать, пока максимальный размер ошибки не удовлетворит
        max_error = -1
        for x, y in zip(X, Y): # Это просто итерирование по двум массивам сразу, входные векторы и выходные

            l0 = x # Значения нейронов на первом слое равны входному вектору
            # dot - это произведение матриц по правилам матана. Если увидишь просто умножение, то это будет просто перемножение членов одной матрицы на члены другой. <Матрица>.T - это транспонирование матрицы
            l1 = nonlin(np.dot(l0, V) + Q) # На скрытом слое это произведение входного слоя на матрицу весов V + порог Q
            l2 = nonlin(np.dot(l1, W) + T) # То же самое для второго слоя

            l2_error = y - l2 # Ошибка второго слоя как разность ожидаемого выхода второго слоя и полученного
            if np.abs(l2_error).max() > max_error: # Находим максимальный модуль ошибки за весь цикл обучения
                max_error = np.abs(l2_error).max()
            l2_delta = l2_error*nonlin(l2, deriv=True)*alpha # Получаем дельту, на которую двигаем веса между скрытым и входным слоем
            l1_error = (l2_error*nonlin(l2, deriv=True)).dot(W.T) # Находим ошибку второго слоя через посчитанный умными людми градиент
            l1_delta = l1_error * nonlin(l1, deriv=True)*alpha # И дельта, на которую двигать веса между входным и скрытым слоем

            # Собственно перемещение весов
            W += l1[np.newaxis].T * l2_delta
            T += l2_delta
            V += l0[np.newaxis].T * l1_delta
            Q += l1_delta

        print(max_error)

        if alpha > max_error and alpha > 0.05:
            alpha = alpha ** 2 # Изменяем скорость обучения по ходу обучения, чтобы сеть сходилась
            print("Max Error: {}".format(max_error))
    print("Error mean : {}".format(np.abs(l2_error).mean())) # Вывожу среднюю ошибку
    return [W, V, T, Q] # Возвращаю параметры нейронки

def clust(origins, network, noise):
    """Процесс кластеризации. Тут в основном рисование таблицы.
    Принцип такой, на нейронку, обученную ранее, подаются идеальные изображения. Значения на выходе нейронки считаются так же, как и при обучении. На каком нейроне больше выход получился, к такому по счету кластеру изображение и отнесем.
    Таким образом каждое идеальное изображение получает по кластеру так сказать.
    Затем я генерирую ещё пять зашумленных изображений, для каждого идеального по одной. И подаю их на входы нейронки. По выходам уже смотрю, какое изображение в какой кластер попало, нахожу вероятности. В шестой лабе нормализовать и находит вероятности ни к чему по сути, но только для дебага может, чтобы прикинуть как оно вообще различает ли разные буквы, или рандом полный. По идее должно хватить просто выбирания нейрона с самым большим выходом """
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
