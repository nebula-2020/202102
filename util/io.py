#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
常用io类
"""
import os
import re
from threading import *
import tensorflow as tf
import traceback
from tensorflow.examples.tutorials import mnist


def read_mnist(size=None):
    data = mnist.input_data.read_data_sets('MNIST_data', one_hot=True).train
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
    if size is not None:
        x = tf.convert_to_tensor(x[:size])
        y = tf.convert_to_tensor(y[:size])
        names = tf.convert_to_tensor(names[:size])
    else:
        x = tf.convert_to_tensor(x)
        y = tf.convert_to_tensor(y)
        names = tf.convert_to_tensor(names)
    return x, y, names


class Reader (Thread):
    def __init__(self, name: str = '', folder: str = '', file: str = '', suffix: list = [], lazy: bool = False, in_char: bool = False):
        Thread.__init__(self)
        self.data = []
        self.data_size = 0
        folder_r = folder.replace('/', '\\').strip('\\')
        file_r = file.replace('/', '\\').strip('\\')
        self.__suffix = suffix
        self.__path = os.path.join(os.getcwd(), folder_r, file_r)
        self.__name = name
        self.__rc = in_char
        self.__lazy = lazy

    def run(self):
        print("THREAD START: "+self.__name)
        self.data = []
        self.data_size = 0
        file_paths = []
        reg = None
        if self.__suffix != []:
            reg = '|'.join(self.__suffix)
            reg.replace('.', '\.')
            reg = '.+('+reg+')$'
        if os.path.exists(self.__path):
            if os.path.isdir(self.__path):
                for root, dirs, files in os.walk(self.__path):
                    for f in files:
                        file_path = os.path.join(root, f)
                        if reg is not None and re.match(reg, file_path, re.IGNORECASE):
                            ele = os.path.join(root, file_path)
                            file_paths.append(ele)  # 后缀匹配成功，加入结果集
                    if self.__lazy:  # 只遍历表层
                        break
            elif os.path.isfile(self.__path):
                file_paths.append(self.__path)
            if self.__rc:  # 是否以字符读取，遍历结果集
                for file_path in file_paths:
                    try:
                        with tf.io.gfile.GFile(file_path, 'r') as file:
                            self.data.append(file.read())
                    except:
                        traceback.print_exc()
            else:
                for file_path in file_paths:
                    try:
                        with tf.io.gfile.GFile(file_path, 'rb') as file:
                            self.data.append(file.read())
                    except:
                        traceback.print_exc()
        self.data_size = len(self.data)
        print('THREAD FINISH: '+self.__name)
        pass


class Writer (Thread):
    def __init__(self, file: str, data,  name: str = '', folder: str = '', in_char: bool = False):
        Thread.__init__(self)
        self.__data = data
        folder_r = folder.replace('/', '\\').strip('\\')
        self.__folder = os.path.join(os.getcwd(), folder_r)
        file_r = file.replace('/', '\\').strip('\\')
        self.__path = os.path.join(self.__folder, file_r)
        self.__name = name
        self.__wc = in_char

    def run(self):
        print("THREAD START: "+self.__name)
        try:
            if not os.path.exists(self.__folder):
                os.makedirs(self.__folder)
            with tf.io.gfile.GFile(self.__path, 'w'if self.__wc else 'wb') as file:
                file.write(self.__data)
        except:
            traceback.print_exc()
        print('THREAD FINISH: '+self.__name)
        pass
