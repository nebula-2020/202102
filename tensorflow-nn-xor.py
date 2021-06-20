#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
神经网络拟合异或。
Python 3.8.5
tensorflow 2.3.0
"""
import tensorflow as tf
from tensorflow.keras import *
import matplotlib.pyplot as pyp

model = Sequential()
model.add(layers.Dense(units=2, input_shape=[2], activation='sigmoid'))
model.add(layers.Dense(units=3, input_shape=[2], activation='sigmoid'))
model.add(layers.Dense(units=1, input_shape=[3], activation='sigmoid'))
opt = tf.optimizers.SGD(lr=0.5, decay=0.0001, momentum=0.2)  # 定义学习率，学习衰减率和学习动量

# 定义优化器和损失函数名字，优化器也可以写成名字而不是对象，现损失函数为交叉熵
model.compile(optimizer=opt, loss='binary_crossentropy')


class haltCallback(callbacks.Callback):  # 自定义回调，给fit当作参数，没有回调也行
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('loss') <= 0.01):  # 当损失充分小的时候停止学习
            self.model.stop_training = True


x = tf.convert_to_tensor([[1., 1.], [1., 0.], [0., 0.], [0., 1.]])
y = tf.convert_to_tensor([[0.], [1.], [0.], [1.]])
print('\033[34m')  # 控制台换成蓝色
# 指定训练集进行10000次拟合，回调每次拟合都会调用on_epoch_end()，stop_training = True时就跳出循环提前结束拟合
history = model.fit(x, y, epochs=10000, verbose=0, callbacks=[haltCallback()])
lost = history.history['loss']  # 获取损失
print(model.summary(), '\033[0m')  # 打印网络，可以看到刚才插入的层
print('\033[32m', model.predict(x), '\033[0m')
pyp.plot(lost)  # 绘制图表
pyp.show()  # 显示图表
