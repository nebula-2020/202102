#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Demo项目，神经网络拟合224x224的1000个常用汉字字样的图片。
"""
import random
import time
import os
import traceback
from threading import *

import matplotlib.pyplot as pyp
import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw, ImageFont
from tensorflow.keras import *
from tensorflow.keras.layers import *

from util import beep
from util import wordch, network
pyp.figure(figsize=(8, 8))
pyp.rcParams['font.sans-serif'] = ['simhei']
np.set_printoptions(threshold=np.inf)


def print_img(show=False):
    if not os.path.exists(IMG_PATH):
        os.makedirs(IMG_PATH)
    imgs = []
    labels = []
    LEN = 12
    x_t, y_t, n_t = create_data(random.sample(
        wordch.WORDS, LEN), wordch.FONTS, times=1)
    for i in range(int(LEN)):
        test = tf.convert_to_tensor([x_t[i]])
        o = model.predict(test)
        imgs.append(o)
        labels.append(n_t[i])
        imgs.append(y_t[i])
        labels.append(n_t[i])
    for i in range(LEN*2):
        pyp.subplot(4, int(LEN/2), i+1)
        pyp.imshow(tf.reshape([imgs[i]], (14, 14)))
        pyp.xticks([])
        pyp.yticks([])
        pyp.title(labels[i])
        pyp.axis('off')
    pyp.savefig(
        os.path.join(
            IMG_PATH,
            '%s_%s_%s.png' % (
                NETWORK_NAME,
                str(time.strftime(r"%Y-%m-%d-%H-%M-%S", time.localtime())),
                str(hex(random.randint(0, 255))[2:])
            )
        )
    )
    if show:
        beep.beep()
        pyp.show()


def create_img(string, font_name: str = 'simsun.ttc', width=224, height=224, font_size=300, random_location=True):
    min_persent = .9
    max_persent = 1.15
    offset = max(1, int(width*.05))
    safe = 5
    while True:
        try:
            font = ImageFont.truetype(font_name, font_size, encoding="unic")
            w, h = font.getsize(string)
            if random_location:
                if h < w:
                    if w < width*min_persent:
                        font_size += 1
                    elif w > width*max_persent:
                        font_size -= 1
                    else:
                        break
                else:
                    if h < height*min_persent:
                        font_size += 1
                    elif h > height*max_persent:
                        font_size -= 1
                    else:
                        break
                font_size += random.randint(-int(offset/2), int(offset/2))
            else:
                break
        except:
            if safe > 0:
                safe -= 1
                time.sleep(1)
            else:
                print(font_name)
                traceback.print_exc()
                return None
    bg = Image.new('RGB', (width, height), color=(0, 0, 0))
    draw = ImageDraw.Draw(bg)
    if random_location:
        loc = ((width-w)/2 + random.randint(-offset, offset),
               (height-h)/2+random.randint(-offset, offset))
    else:
        loc = ((width-w)/2,  (height-h)/2)
    draw.text(loc, string, fill="#ffffff", font=font)
    if random_location:
        bg = bg.rotate(random.randint(-7, 7))
    bg = bg.convert('L')
    return bg


def flist2str(l, lvl=0):
    if type(l) is list:
        tmp = [flist2str(e, lvl=lvl+1) for e in l]
        ret = ''.join(
            ['\n ']*lvl+['[', ', '.join(['%s']*len(l)), ''if lvl > 0 else '\n', ']']) % tuple(tmp)
    else:
        ret = '%.4f' % l
    return ret


def list_simmilarity(y: list, y_hat: list):
    l1 = len(y)
    l2 = len(y_hat)
    if l1 != l2:
        raise ValueError(
            'The lists must have the same length: %d != %d' % (l1, l2)
        )
    ret = 0
    total = 0.
    for i in range(l1):
        sel1 = float(y[i])
        sel2 = float(y_hat[i])
        total += 1 if sel1 >= 1 else .05
        if abs(sel1 - sel2) <= .05:
            ret += 1 if sel1 >= 1 else .05
    return ret/total


class ImgThread (Thread):

    def __init__(self, word: str, fonts: list = ['simsun.ttc'], times=5, start=0):
        Thread.__init__(self)
        self.word = word
        self.__fonts = fonts
        self.x = []
        self.y = []
        self.__times = times
        self.size = 0
        self.__startIndex = start
        self.__font_count = len(self.__fonts)

    def run(self):
        print("THREAD START: "+self.word)
        img = create_img(self.word, width=14, height=14,
                         font_size=14, random_location=False)
        if img is not None:
            img = np.matrix(img, dtype='float')/255
            self.y.append(np.array(img).flatten())
            try:
                for index in range(self.__font_count):
                    i = (index+self.__startIndex) % self.__font_count
                    for _ in range(self.__times):
                        t_img = create_img(self.word, width=112, height=112,
                                           font_size=72,
                                           font_name=self.__fonts[i])
                        if t_img is not None:
                            t_img = np.matrix(t_img, dtype='float')/255
                            self.x.append(tf.convert_to_tensor(t_img))
            except:
                traceback.print_exc()
        self.size = len(self.x)
        print('THREAD FINISH: '+self.word)
        pass


def create_data(words: list, fonts: list, times=5, parallel=50):
    threads = []
    for word_i in range(len(words)):
        threads.append(
            ImgThread(words[word_i], fonts=fonts, times=times, start=word_i))
    thread_len = len(threads)
    for i in range(0, thread_len, parallel):
        for j in range(parallel):
            if i+j < thread_len:
                threads[i+j].start()
        for j in range(parallel):
            if i+j < thread_len:
                threads[i+j].join()
    x, y, n = [], [], []
    for t in threads:
        for i in range(t.size):
            x.append(t.x[i])
            y.append(t.y[0])
            n.append(t.word)
    x = tf.convert_to_tensor(x)
    y = tf.convert_to_tensor(y)
    x = tf.expand_dims(x, -1)  # 解决维度不匹配的问题
    return x, y, n


NETWORK = [
    Input(shape=(112, 112, 1)),
    Conv2D(filters=64, kernel_size=(3, 3), padding='SAME', strides=1,
           activation=tf.nn.relu,
           name='Conv2D-1-1'),
    Conv2D(filters=64, kernel_size=(3, 3), padding='SAME', strides=1,
           activation=tf.nn.relu,
           name='Conv2D-1-2'),
    MaxPool2D(pool_size=(2, 2), strides=2, padding='SAME', name='Pool-1'),
    Conv2D(filters=128, kernel_size=(3, 3), padding='SAME', strides=1,
           activation=tf.nn.relu, name='Conv2D-2-1'),
    Conv2D(filters=128, kernel_size=(1, 1), padding='SAME', strides=1,
           activation=tf.nn.relu, name='Conv2D-2-2'),
    MaxPool2D(pool_size=(2, 2), strides=2, padding='SAME', name='Pool-2'),
    Conv2D(filters=256, kernel_size=(3, 3), padding='SAME', strides=1,
           activation=tf.nn.relu,
           name='Conv2D-3-1'),
    Conv2D(filters=256, kernel_size=(3, 3), padding='SAME', strides=1,
           activation=tf.nn.relu,
           name='Conv2D-3-2'),
    Conv2D(filters=256, kernel_size=(1, 1), padding='SAME', strides=1,
           activation=tf.nn.relu,
           name='Conv2D-3-3'),
    MaxPool2D(pool_size=(2, 2), strides=2, padding='SAME', name='Pool-3'),
    Conv2D(filters=256, kernel_size=(3, 3), padding='SAME', strides=1,
           activation=tf.nn.relu,
           name='Conv2D-4-1'),
    Conv2D(filters=256, kernel_size=(3, 3), padding='SAME', strides=1,
           activation=tf.nn.relu,
           name='Conv2D-4-2'),
    Conv2D(filters=256, kernel_size=(1, 1), padding='SAME', strides=1,
           activation=tf.nn.relu,
           name='Conv2D-4-3'),
    MaxPool2D(pool_size=(2, 2), strides=2, padding='SAME', name='Pool-4'),
    Flatten(),
    Dense(units=196, activation=tf.nn.sigmoid, name='Dense-1'),
    # 1x1x196
]
model = Sequential(NETWORK)
NETWORK_NAME = 'chinese'
MODEL_NAME = os.path.join('data', NETWORK_NAME)
IMG_PATH = os.path.join(MODEL_NAME, 'img')
PRINT_DELAY = 100
TAR_ACC = 0.9
TRAIN_SCALE = 12
x, y, _ = create_data(random.sample(wordch.WORDS, min(
    TRAIN_SCALE, len(wordch.WORDS))), wordch.FONTS)


class haltCallback(callbacks.Callback):

    def on_epoch_end(self, epoch, logs={}):
        global x_t, y_t, n_t, count
        acc = logs.get('accuracy')
        if epoch % PRINT_DELAY == 0:
            print_img()
            lo = logs.get('loss')
            print('\033[34mLOST: %.15f, ACCURACY: %.15f\033[0m' % (lo, acc))
        if(acc >= TAR_ACC):
            self.model.stop_training = True


print('NOW START TRAINING.')
opt = tf.optimizers.Adadelta(learning_rate=.001)
try:
    model = tf.keras.models.load_model(MODEL_NAME)
except:
    # opt = tf.optimizers.SGD(lr=.1, momentum=.25)
    model.compile(
        optimizer=opt,
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    model.build(input_shape=(1, 112, 112, 1))
    model.summary()
history = model.fit(x, y, epochs=5000, verbose=1, callbacks=[haltCallback()])
model.save(MODEL_NAME)
model.summary()
lost = history.history['loss']
acc = history.history['accuracy']
pyp.plot(lost, color='red', label='loss')
pyp.plot(acc, color='blue', label='accuracy')
print_img(True)  # 显示图表
