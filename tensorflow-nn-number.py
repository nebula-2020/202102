#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Demo项目，神经网络拟合28x28包含0~9数字字样的图片。
"""
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import *
from tensorflow.keras.layers import *
from threading import *
from util import beep
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)


def read_mnist(data, convert_to_tensor=True):
    SOURCE_DATA_WIDTH = 784
    TARGET_DATA_WIDTH = 28
    data_size = data.num_examples
    x = [None]*data_size
    y = [None]*data_size
    names = [None]*data_size
    for i in range(data_size):
        read = data.images[i]
        label = data.labels[i]
        img = []
        for line_h in range(0, SOURCE_DATA_WIDTH, TARGET_DATA_WIDTH):
            line = []
            for e in read[line_h:line_h+TARGET_DATA_WIDTH]:
                line.append([e])
            img.append(line)
        x[i] = img
        y[i] = label
        names[i] = '%s-%s' % (hex(i), str(label.tolist().index(1.)))
    if convert_to_tensor:
        x = tf.convert_to_tensor(x)
        y = tf.convert_to_tensor(y)
        names = tf.convert_to_tensor(names)
    return x, y, names


np.set_printoptions(threshold=np.inf)
np.set_printoptions(suppress=True)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
DATA_ROOT = './data/letnet5/'
OUTPUT_SCALE = 10
X = 'x'
Y = 'y'
NAME = 'name'
t_loss = 0.1


def find_max(v: list):
    if len(v) == 0:
        return None, None
    m = v[0]
    mi = 0
    for i in range(len(v)):
        if v[i] >= m:
            m = v[i]
            mi = i
    return mi, m


x, y, names = read_mnist(mnist.train)

x_t, y_t, name_t = read_mnist(
    mnist.validation, convert_to_tensor=False)

x_t = tf.convert_to_tensor(x_t[30:])


class haltCallback(callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        global t_loss, x_t, y_t, name_t
        result = model.predict(x_t)
        for row_i in range(len(result)):
            row = result[row_i]
            print('[ ', end='')
            l = len(row)
            mi, m = find_max(row)
            tar_i, tar = find_max(y_t[row_i])
            if m <= .1:
                mi = -1
            for index in range(l):
                color = index == mi
                if color:
                    if tar_i == index:
                        print('\033[32m', end='')
                    else:
                        print('\033[31m', end='')
                print('%.2f' % row[index], end='')
                if color:
                    print('\033[0m', end='')
                if index == l-1:
                    if index % 2 == 1:
                        end = '\n'
                    else:
                        end = ' \t '
                    print(' ] ', tar_i, ' ', name_t[row_i], end=end)
                else:
                    print(', ', end='')
        print('')
        lo = logs.get('loss')
        print('\033[34m'+'LOST: '+str(lo) + '\033[0m' + '\n')
        if(lo <= t_loss):
            self.model.stop_training = True


def create_opt():
    t = beep.beep()
    while True:
        try:
            print('LEARNING RATE: ', end='')
            lr = float(input())
            break
        except BaseException as e:
            print('\033[31m', 'ERROR: ', e, '\033[0m')
            pass
    while True:
        try:
            print('MOMENTUM: ', end='')
            mom = float(input())
            break
        except BaseException as e:
            print('\033[31m', 'ERROR: ', e, '\033[0m')
            pass
    while True:
        try:
            print('FINAL LOSS: ', end='')
            global t_loss
            t_loss = float(input())
            break
        except BaseException as e:
            print('\033[31m', 'ERROR: ', e, '\033[0m')
            pass
    t.join()
    return tf.optimizers.SGD(lr=lr, momentum=mom)


def out(name: str, path: str, file_name: str):
    global x_t, name_t
    m = Model(inputs=model.input, outputs=model.get_layer(name).output)
    output = m.predict(x_t)
    try:
        length, height, width, count = output.shape
        for i in range(length):
            for index in range(count):
                bitmap = []
                for y in range(height):
                    line = []
                    for x in range(width):
                        v = output[i, y, x, index]
                        pixel = [v]*3
                        line.append(pixel)
                    bitmap.append(line)
                bitmap = tf.convert_to_tensor(bitmap)
                tmp = np.floor(tf.multiply(bitmap, 255).numpy())
                img = tf.convert_to_tensor(tmp, dtype='uint8')
                img = tf.image.encode_png(img, compression=3)
                if not os.path.exists(path):
                    os.makedirs(path)
                fp = os.path.join(path, file_name+str(i)+'-'+str(index)+'.png')
                with tf.io.gfile.GFile(fp, 'wb') as file:
                    file.write(img.numpy())
    except:
        length, width = output.shape
        for i in range(length):
            bitmap = []
            line = []
            for x in range(len(output[i])):
                pixel = [output[i, x]]*3
                line.append(pixel)
            bitmap.append(line)
            bitmap = tf.convert_to_tensor(bitmap)
            tmp = np.floor(tf.multiply(bitmap, 255).numpy())
            img = tf.convert_to_tensor(tmp, dtype='uint8')
            img = tf.image.encode_png(img, compression=3)
            fp = path+file_name+str(i)+'.png'
            with tf.io.gfile.GFile(fp, 'wb') as file:
                file.write(img.numpy())


# LeNet结构
layers = [Conv2D(filters=20, kernel_size=(5, 5), strides=1, activation='relu', padding='same', name="Conv_1"),
          AvgPool2D(data_format='channels_last', strides=2,
                    pool_size=(2, 2), name="MaxPool_1"),
          Conv2D(filters=50, kernel_size=(5, 5), strides=1,
                 activation='relu', padding='same', name="Conv_2"),
          AvgPool2D(data_format='channels_last', strides=2,
                    pool_size=(2, 2), name="MaxPool_2"),
          layers.Flatten(),
          Dense(units=500, activation='relu', name="Dense_1"),
          Dense(units=10, activation='softmax', name="Dense_2")]

while True:
    model = Sequential(layers)

    opt = create_opt()

    model.compile(optimizer=opt, loss='binary_crossentropy')
    history = model.fit(x, y, epochs=500, callbacks=[haltCallback()])
    print('\033[34m')
    print(model.summary(), '\033[0m', '\n')

    out('Conv_1', os.path.join(DATA_ROOT, 'conv_1'), '')
    out('Conv_2',  os.path.join(DATA_ROOT, 'conv_2'), '')
    out('MaxPool_1', os.path.join(DATA_ROOT, 'mp_1'), '')
    out('MaxPool_2', os.path.join(DATA_ROOT, 'mp_2'), '')
    out('Dense_1',  os.path.join(DATA_ROOT, 'dense_1'), '')
    out('Dense_2',  os.path.join(DATA_ROOT, 'dense_2'), '')

    t = beep.beep()
    print('CONTINUE? (y/n) ', end='')
    if str.lower(input()) == 'n':
        break
    t.join()
print('END.')
