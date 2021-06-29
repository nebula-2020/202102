#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Demo项目，神经网络拟合224x224的1000个常用汉字字样的图片。
"""
import json
import os
import random
import time
import traceback
from threading import *

import matplotlib.pyplot as pyp
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from tensorflow import (
    convert_to_tensor, expand_dims, nn, optimizers, reshape
)
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.layers import (
    AvgPool2D, Conv2D, Dense, Dropout, Flatten, Input, MaxPool2D, Reshape
)
from tensorflow.keras.models import load_model

from util import beep, network, wordch

pyp.figure(figsize=(8, 8))
pyp.rcParams['font.sans-serif'] = ['simhei']
np.set_printoptions(threshold=np.inf)

TEST_WORDS = [
    '已', '己', '乙', '巳', '未', '末', '徽', '微', '人', '入',
    '八', '曰', '日', '的', '一', '是', '了', '我', '多', '然',
    '心', '高', '　', '感', '苯', '泉', '圈', '术', '剑', '忖'
]


class Consts():
    IMG_SHAPE = (112, 112, 1)
    TAR_SHAPE = (14, 14, 1)
    NETWORK_NAME = 'chinese'
    SAVE_PATH = os.path.join('data', NETWORK_NAME)
    IMG_PATH = os.path.join(SAVE_PATH, 'img')
    JSON_PATH = os.path.join(SAVE_PATH, 'json.json')
    TAR_ACC = 0.9
    X_SCALE = 96
    TAG_LOSS = 'loss'
    TAG_ACC = 'acc'
    EPOCHS = 250


def print_img(show=False, name=str(hex(random.randint(0, 255))[2:])):
    if not os.path.exists(Consts.IMG_PATH):
        os.makedirs(Consts.IMG_PATH)
    imgs = []
    labels = []
    LEN = 40
    x_t, y_t, n_t = create_data(
        random.sample(TEST_WORDS, LEN),
        random.sample(wordch.FONTS, 1),
        times=1
    )
    for i in range(int(LEN)):
        test = convert_to_tensor([x_t[i]])
        o = model.predict(test)
        imgs.append(o)
        labels.append(n_t[i])
        imgs.append(y_t[i])
        labels.append(n_t[i])
    for i in range(LEN*2):
        pyp.subplot(5, int(LEN/2), i+1)
        pyp.imshow(reshape([imgs[i]], (14, 14)))
        pyp.xticks([])
        pyp.yticks([])
        pyp.title(labels[i])
        pyp.axis('off')
    pyp.savefig(
        os.path.join(
            Consts.IMG_PATH,
            '%s_%s_%s.png' % (
                Consts.NETWORK_NAME,
                str(time.strftime(r"%Y-%m-%d-%H-%M-%S", time.localtime())),
                name
            )
        )
    )
    if show:
        beep.beep()
        pyp.show()


def create_img(string, font_name: str = 'simsun.ttc', width=224, height=224, font_size=150, random_location=True):
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
            ['\n ']*lvl +
            ['[', ', '.join(['%s']*len(l)), ''if lvl > 0 else '\n', ']']
        ) % tuple(tmp)
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
        img = create_img(
            self.word, width=Consts.TAR_SHAPE[-3],
            height=Consts.TAR_SHAPE[-2],
            font_size=14, random_location=False
        )
        if img is not None:
            img = np.matrix(img, dtype='float')/255
            self.y.append(np.array(img).flatten())
            try:
                for index in range(self.__font_count):
                    i = (index+self.__startIndex) % self.__font_count
                    for _ in range(self.__times):
                        t_img = create_img(
                            self.word, width=Consts.IMG_SHAPE[-3],
                            height=Consts.IMG_SHAPE[-2],
                            font_size=94, font_name=self.__fonts[i]
                        )
                        if t_img is not None:
                            t_img = np.matrix(t_img, dtype='float')/255
                            self.x.append(convert_to_tensor(t_img))
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
    x = convert_to_tensor(x)
    y = convert_to_tensor(y)
    x = expand_dims(x, -1)  # 解决维度不匹配的问题
    return x, y, n


def create_set():
    x, y, _ = create_data(
        random.sample(wordch.WORDS, min(Consts.X_SCALE, len(wordch.WORDS))),
        wordch.FONTS
    )
    x = x*np.random.normal(loc=1.0, scale=0.15, size=Consts.IMG_SHAPE)
    x = x + (np.random.normal(size=Consts.IMG_SHAPE)*0.15)
    # y = y*([0.9]*196)
    return x, y


NETWORK = [
    Input(shape=Consts.IMG_SHAPE, name='input'),
    Conv2D(filters=64, kernel_size=(7, 7), padding='SAME', strides=2,
           activation=nn.relu,
           name='Conv2D-1'),
    Conv2D(filters=64, kernel_size=(3, 3), padding='SAME', strides=1,
           activation=nn.relu,
           name='Conv2D-2'),
    MaxPool2D(pool_size=(3, 3), strides=2, padding='SAME', name='Pool-1'),
    network.Inception(
        col_11=64, col_33_r=96, col_33=128, col_55_r=16,
        col_55=32, col_pool=32, name='Inception-1-1'
    ),
    network.Inception(
        col_11=128, col_33_r=128, col_33=192, col_55_r=32,
        col_55=96,  col_pool=64, name='Inception-1-2'
    ),
    MaxPool2D(pool_size=(3, 3), strides=2, padding='SAME', name='Pool-2'),
    network.Inception(
        col_11=192, col_33_r=96, col_33=208, col_55_r=16,
        col_55=48,  col_pool=64, name='Inception-2-1'
    ),
    network.Inception(
        col_11=160, col_33_r=112, col_33=224, col_55_r=24,
        col_55=64,  col_pool=64, name='Inception-2-2'
    ),
    network.Inception(
        col_11=128, col_33_r=128, col_33=256, col_55_r=24,
        col_55=64,  col_pool=64, name='Inception-2-3'
    ),
    network.Inception(
        col_11=112, col_33_r=144, col_33=288, col_55_r=32,
        col_55=64,  col_pool=64, name='Inception-2-4'
    ),
    network.Inception(
        col_11=256, col_33_r=160, col_33=320, col_55_r=32,
        col_55=128,  col_pool=128, name='Inception-2-5'
    ),
    MaxPool2D(pool_size=(3, 3), strides=2, padding='SAME', name='Pool-3'),
    network.Inception(
        col_11=256, col_33_r=160, col_33=320, col_55_r=32,
        col_55=128,  col_pool=128, name='Inception-3-1'
    ),
    network.Inception(
        col_11=384, col_33_r=192, col_33=384, col_55_r=48,
        col_55=128,  col_pool=128, name='Inception-3-2'
    ),
    AvgPool2D(pool_size=(7, 7)),
    Flatten(),
    Dropout(0.4),
    Dense(units=196, activation=nn.sigmoid, name='Dense-1'),
    # 1x1x196
]

model = network.network(NETWORK)
x, y = create_set()

try:
    with open(Consts.JSON_PATH, 'r') as f:
        data = json.loads(f.read())
except:
    data = {
        'loss': [],
        'accuracy': []
    }


class Callback_(Callback):
    PRINT_DELAY = 50
    SAVE_DELAY = 200

    def on_epoch_end(self, epoch, logs={}):
        global x, y, data
        acc = logs.get('accuracy')
        data['loss'].append(logs.get('loss'))
        data['accuracy'].append(acc)
        if epoch % Callback_.PRINT_DELAY == 0:
            print_img()
            lo = logs.get('loss')
            print('\033[34mLOST: %.15f, ACCURACY: %.15f\033[0m' % (lo, acc))
        if epoch % Callback_.SAVE_DELAY == 0:
            model.save(Consts.SAVE_PATH)
            with open(Consts.JSON_PATH, 'w') as f:
                f.write(json.dumps(data))
            x, y = create_set()
        if(acc >= Consts.TAR_ACC):
            self.model.stop_training = True


print('NOW START TRAINING.')
opt = optimizers.Nadam(learning_rate=.0001)
try:
    print('LOADED.')
    model = load_model(Consts.SAVE_PATH)
except:
    model.compile(
        optimizer=opt,
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    model.build(input_shape=(1,)+Consts.IMG_SHAPE)
    model.summary()
print_img()
history = model.fit(
    x, y, epochs=Consts.EPOCHS, verbose=1, callbacks=[Callback_()]
)
lost = list(history.history['loss'])
acc = list(history.history['accuracy'])
model.save(Consts.SAVE_PATH)
model.summary()
# pyp.plot(lost, color='red', label='loss')
# pyp.plot(acc, color='blue', label='accuracy')
print_img(True)
