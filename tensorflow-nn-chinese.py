#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Demo项目，神经网络拟合224x224的1000个常用汉字字样的图片。
"""
import random
import time
import traceback
from threading import *

import matplotlib.pyplot as pyp
import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw, ImageFont
from tensorflow.keras import *
from tensorflow.keras.layers import *

from util import beep
from util import wordch
from util.cnn import NETWORK

np.set_printoptions(threshold=np.inf)


def create_img(string, font_name: str = 'simsun.ttc', width=224, height=224, font_size=140, random_location=True):
    min_persent = .75
    max_persent = .85
    offset = max(1, int(width*.1))
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
        raise ValueError('The lists must have the same length.')
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


NETWORK_NAME = 'resnet'
MODEL_NAME = '/'.join(['data', NETWORK_NAME])
COUNT_MAX = 25
TAR_ACC = 0.9
TRAIN_SCALE = 300
TEST_SCALE = 7
x, y, _ = create_data(random.sample(wordch.WORDS, TRAIN_SCALE), wordch.FONTS)
x_t, y_t, n_t = create_data(random.sample(
    wordch.WORDS, TEST_SCALE), wordch.FONTS, times=1)

network = NETWORK[NETWORK_NAME]
model = Sequential(network)
count = -1


class haltCallback(callbacks.Callback):

    def on_epoch_end(self, epoch, logs={}):
        global TAR_LOSS, x_t, y_t, n_t, count
        acc = logs.get('accuracy')
        if count > COUNT_MAX or count < 0:
            count = 0
            result = model.predict(x_t)
            print(flist2str(np.resize(result[0], (14, 14)).tolist()))
            for row_i in range(len(result)):
                row = result[row_i]
                for yi in range(len(y_t)):
                    sim = list_simmilarity(y_t[yi], row)
                    if yi == row_i:
                        if sim > .9:
                            s = '\033[32m%s: %.7f\033[0m'
                        else:
                            s = '\033[31m%s: %.7f\033[0m'
                    else:
                        s = '%s: %.7f'
                    s = s % (n_t[yi], sim)
                    print(s, end='\t')  # n_t[row_i]
                print('')
            lo = logs.get('loss')
            print('\033[34mLOST: %.15f, ACCURACY: %.15f\033[0m' % (lo, acc))
        else:
            count += 1
        if(acc >= TAR_ACC):
            self.model.stop_training = True


print('NOW START TRAINING.')
try:
    model = tf.keras.models.load_model(MODEL_NAME)
except:
    # opt = tf.optimizers.SGD(lr=.1, momentum=.25)
    opt = tf.optimizers.Adadelta(learning_rate=.05)
    model.compile(optimizer=opt, loss='binary_crossentropy',
                  metrics=['accuracy'])
history = model.fit(x, y, epochs=100, verbose=1, callbacks=[haltCallback()])
model.save(MODEL_NAME)
model.summary()
lost = history.history['loss']
acc = history.history['accuracy']
pyp.plot(lost, color='red', label='loss')
pyp.plot(acc, color='blue', label='accuracy')
pyp.legend()
beep()
pyp.show()  # 显示图表
