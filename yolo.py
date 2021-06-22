#!/usr/bin/env python
# -*- coding: utf-8 -*-
from tensorflow.keras.layers import *
from tensorflow.keras import *
import tensorflow as tf
import math
import random
import traceback
from time import strftime, localtime
import os
import numpy as np
import util.beep as beep
import util.network as network
from PIL import Image, ImageDraw
import matplotlib.pyplot as pyp
import xml.dom.minidom as minidom
SRC = 'E:\\data\\UA-DETRAC\\train\\UA-DETRAC\\DETRAC-train-data'
TRAIN_FOLDER_NAME = 'Insight-MVT_Annotation_Train'
TEST_FOLDER_NAME = 'Insight-MVT_Annotation_Test'
LABEL_FOLDER_NAME = 'DETRAC-Train-Annotations-XML'
IMG_SHAPE = (224, 224, 3)
TAR_SHAPE = (7, 7, 5)
MODEL_NAME = 'yolo_v2'
IMG_SAVE_PATH = os.path.join('data', MODEL_NAME, 'imgs')
MODEL_SAVE_PATH = os.path.join('data', MODEL_NAME)


def read_data(img_only=False):
    X_PATH = os.path.join(
        SRC, TEST_FOLDER_NAME if img_only else TRAIN_FOLDER_NAME
    )
    x = []
    TAG_X, TAG_Y, TAG_W, TAG_H = ('x', 'y', 'w', 'h')

    folders = []
    for root, dirs, _ in os.walk(X_PATH, topdown=False):
        for name in dirs:
            folders.append(os.path.join(root, name))
    folder_name = random.sample(folders, 1)[0]  # 随机挑选一个文件夹并取文件夹名
    folder_name = folder_name.split('\\')[-1]

    if not img_only:
        L_PATH = os.path.join(SRC, LABEL_FOLDER_NAME)
        CELL_W = IMG_SHAPE[0] / TAR_SHAPE[0]
        CELL_H = IMG_SHAPE[1] / TAR_SHAPE[1]
        y = []
        ig_rects = []
        labels = []
        with minidom.parse(os.path.join(L_PATH, '%s.xml' % folder_name)) as dom:
            # 解析与文件夹同名xml
            sel = dom.getElementsByTagName('ignored_region')[0]
            for box in sel.getElementsByTagName('box'):
                ele = {
                    TAG_X: float(box.getAttribute('left')),
                    TAG_Y: float(box.getAttribute('top')),
                    TAG_W: float(box.getAttribute('width')),
                    TAG_H: float(box.getAttribute('height'))
                }
                ig_rects.append(ele)
            sel = dom.getElementsByTagName('frame')
            for frame in sel:
                frame_rects = []
                for tar_list in frame.getElementsByTagName('target_list'):
                    for tar in tar_list.getElementsByTagName('target'):
                        info = tar.getElementsByTagName('box')[0]
                        ele = {
                            TAG_X: float(info.getAttribute('left')),
                            TAG_Y: float(info.getAttribute('top')),
                            TAG_W: float(info.getAttribute('width')),
                            TAG_H: float(info.getAttribute('height'))
                        }
                        frame_rects.append(ele)
                labels.append(frame_rects)  # 所在序号表示第几张图，元素为每个矩形的位置大小
        for root, dirs, files in os.walk(
            os.path.join(X_PATH, folder_name), topdown=False
        ):
            for i in range(len(files)):
                name = files[i]
                img = Image.open(os.path.join(root, name))
                d = ImageDraw.Draw(img)
                for r in ig_rects:  # 预处理打马赛克
                    d.rectangle(
                        [
                            (r[TAG_X], r[TAG_Y]),
                            (r[TAG_X]+r[TAG_W], r[TAG_Y]+r[TAG_H])
                        ],
                        fill=(0, 0, 0)
                    )
                w, h = img.size
                padding_top = 0 if h > w else (w-h)/2
                padding_left = 0 if w > h else (h-w)/2
                scale = max(w, h)
                img = img.crop(
                    (
                        -padding_left,
                        -padding_top,
                        w + (0 if w > h else (h - w) / 2),
                        h + (0 if h > w else (w - h) / 2),
                    )
                )
                sh = IMG_SHAPE[:-1]
                img = img.resize(sh, Image.ANTIALIAS)
                label = labels[i]
                mat = tf.zeros(shape=TAR_SHAPE).numpy()
                for ri in range(len(label)):  # 计算目标位置
                    label[ri][TAG_X] = (label[ri][TAG_X] + padding_left)
                    label[ri][TAG_X] = label[ri][TAG_X] / scale * IMG_SHAPE[0]
                    label[ri][TAG_Y] = (label[ri][TAG_Y] + padding_top)
                    label[ri][TAG_Y] = label[ri][TAG_Y] / scale * IMG_SHAPE[1]
                    label[ri][TAG_W] = label[ri][TAG_W] / scale * IMG_SHAPE[0]
                    label[ri][TAG_H] = label[ri][TAG_H] / scale * IMG_SHAPE[1]
                for ri in range(len(label)):
                    rect = label[ri]
                    # 矩形中心位置
                    rect_cx = rect[TAG_X]+rect[TAG_W]/2
                    rect_cy = rect[TAG_Y]+rect[TAG_H]/2
                    # 计算cell序号
                    cell_xi = math.floor(rect_cx/CELL_W)
                    cell_yi = math.floor(rect_cy/CELL_H)
                    x_, y_, w_, h_, c = (
                        (rect_cx-cell_xi * CELL_W) / CELL_W,
                        (rect_cy-cell_yi * CELL_H) / CELL_H,
                        rect[TAG_W] / CELL_W,
                        rect[TAG_H] / CELL_H,
                        1
                    )
                    rect = [x_, y_, w_, h_, c]  # 只有1个bbox
                    for i_ in range(len(rect)):
                        mat[cell_yi][cell_xi][i_] = rect[i_]
                x.append(np.array(img))
                y.append(mat)
        return x, y
    else:
        for root, dirs, files in os.walk(
            os.path.join(X_PATH, folder_name), topdown=False
        ):
            for i in range(len(files)):
                name = files[i]
                img = Image.open(os.path.join(root, name))
                x.append(np.array(img))
        return x


def model_draw(model: Model, x, name=''):
    y = model.predict(tf.convert_to_tensor([x]))
    draw_(x, y[0], name)


def draw_(x, y, name=''):
    if not os.path.exists(IMG_SAVE_PATH):
        os.makedirs(IMG_SAVE_PATH)
    CELL_W = IMG_SHAPE[0] / TAR_SHAPE[0]
    CELL_H = IMG_SHAPE[1] / TAR_SHAPE[1]
    sh = tf.convert_to_tensor(y).shape
    img = Image.fromarray(np.array(x).astype('uint8')).convert('RGB')
    d = ImageDraw.Draw(img)
    for yi in range(sh[0]):
        for xi in range(sh[1]):
            box = y[yi][xi]
            # if box[4] > .1:
            l = float((xi+box[0]-box[2]/2)*CELL_W)
            t = float((yi+box[1]-box[3]/2)*CELL_H)
            w = min(float(box[2]*CELL_W), 1.0)
            h = min(float(box[3]*CELL_H), 1.0)
            d.polygon(
                [(l, t), (l+w, t), (l+w, t+h), (l, t+h), (l, t)],
                None,
                'red' if box[4] > .5 else 'yellow'
            )
    img.save(
        os.path.join(
            IMG_SAVE_PATH,
            '%s_%s_%s_%s.png' % (
                MODEL_NAME,
                str(strftime(r"%Y-%m-%d-%H-%M-%S", localtime())),
                str(hex(random.randint(0, 255))[2:]),
                name
            )
        )
    )


NETWORK = [
    Input(shape=(224, 224, 3), name='input'),
    Conv2D(filters=32, kernel_size=(3, 3), strides=1, padding='same'),
    MaxPool2D(),
    Conv2D(filters=64, kernel_size=(3, 3), strides=1, padding='same'),
    MaxPool2D(),
    network.SandglassConv2D(filters_io=128, filters_mid=64),
    MaxPool2D(),
    network.SandglassConv2D(filters_io=256, filters_mid=128),
    MaxPool2D(),
    network.SandglassConv2D(filters_io=512, filters_mid=256, layers=5),
    MaxPool2D(),
    network.SandglassConv2D(filters_io=1024, filters_mid=512, layers=5),
    Conv2D(filters=1024, kernel_size=(3, 3), padding='same'),
    Conv2D(filters=TAR_SHAPE[-1], kernel_size=(1, 1), padding='valid'),
]

if __name__ == '__main__':
    try:
        model = network.network(NETWORK)
        model.summary()
        opt = optimizers.Nadam(learning_rate=.0001)
        model.compile(
            optimizer=opt,
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        x, y = read_data()
        x_t = [x[0]]
        model_draw(model, random.sample(x_t, 1))

        class Callback_(callbacks.Callback):
            def on_epoch_end(self, epoch, logs={}):
                global x_t, model
                model_draw(model, random.sample(x_t, 1), name=str(epoch))
                #
        history = model.fit(
            tf.convert_to_tensor(x[1:]),
            tf.convert_to_tensor(y[1:]),
            epochs=50,
            callbacks=Callback_
        )
        model.save(MODEL_SAVE_PATH)
        model_draw(model, random.sample(x_t, 1))
        lost = list(history.history['loss'])
        acc = list(history.history['accuracy'])
        pyp.plot(lost, color='red', label='loss')
        pyp.plot(acc, color='blue', label='accuracy')
        beep.beep()
        pyp.show()
    except:
        traceback.print_exc()
        beep.beep(times=5, delay=0.1, duration=0.25)
