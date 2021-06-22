#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
工具，创建更加随意的网络结构。
"""
from tensorflow import nn
from tensorflow.keras import Model
from tensorflow.keras.layers import (
    concatenate, Conv2D, MaxPool2D, Input, AvgPool2D, Flatten, Softmax
)


def network(layers: list) -> Model:
    """新建模型。

    example
    ----------
    由
    ```
    model = network([
        tf.keras.layers.Input(shape=(7, 7, 1), name='input'),
        (
            Conv2D(filters=4, kernel_size=(3, 3), name='conv-1-1'),
            [
                Conv2D(
                    filters=2, kernel_size=(3, 3), name='conv-2-1'
                ),
                Conv2D(
                    filters=2, kernel_size=(1, 1), name='conv-2-2'
                ),
            ]
        ),
        Flatten(name='flatten'),
        Dense(units=16, name='dense-1'),
        Dense(units=2, name='dense-2')
    ])
    ```
    …
    ```
    model.build(input_shape=(7, 7, 1))
    model.summary()
    ```
    构造的模型：

    Model: "model"

    |Layer (type)|Output Shape|Param #|Connected to|
    |:--|:-:|:-:|:--|
    |input (InputLayer)       |[(None, 7, 7, 1)]|0
    |conv-2-1 (Conv2D)        |(None, 5, 5, 2)  |20  |input[0][0]
    |conv-1-1 (Conv2D)        |(None, 5, 5, 4)  |40  |input[0][0]
    |conv-2-2 (Conv2D)        |(None, 5, 5, 2)  |6   |conv-2-1[0][0]
    |concatenate (Concatenate)|(None, 5, 5, 6)  |0   |conv-1-1[0][0]&emsp;conv-2-2[0][0]
    |flatten (Flatten)        |(None, 150)      |0   |concatenate[0][0]
    |dense-1 (Dense)          |(None, 16)       |2416|flatten[0][0]
    |dense-2 (Dense)          |(None, 2)        |34  |dense-1[0][0]

    Total params: 2,516
    Trainable params: 2,516
    Non-trainable params: 0

    Parameters
    ----------
    layers : list
        设置网络层，`list`构造串联层，`tuple`构造并联层

    Returns
    -------
    tensorflow.keras.Model
        构造的模型。
    """

    def build(obj, input_layer):
        if obj is None:
            return input_layer
        elif type(obj) == list:
            ret = input_layer
            for e in obj:
                ret = build(e, ret)
            return ret
        elif type(obj) == tuple:
            li = []
            for e in obj:
                li.append(build(e, input_layer))
            return concatenate(inputs=li, axis=-1)
        else:
            return obj(input_layer)
    output_layer = build(layers[1:], layers[0])
    return Model(layers[0], output_layer)


def Inception(
    col_33_r: int = 96, col_33: int = 128, col_55_r: int = 16,
    col_55: int = 32, col_11: int = 64, col_pool: int = 32,
    activation=nn.relu, name: str = None
) -> tuple:
    """Googlenet块。

    Parameters
    ----------
    col_33_r : int, optional
        3x3列1x1卷积通道数, by default 96
    col_33 : int, optional
        3x3列卷积通道数, by default 128
    col_55_r : int, optional
        5x5列1x1卷积通道数, by default 16
    col_55 : int, optional
        5x5列卷积通道数, by default 32
    col_11 : int, optional
        1x1列卷积通道数, by default 64
    col_pool : int, optional
        池化层列1x1卷积通道数, by default 32
    activation : optional
        激活函数, by default tensorflow.nn.relu
    name : str, optional
        网络名称, by default None

    Returns
    -------
    tuple
        Googlenet块。
    """
    return tuple([
        [
            Conv2D(
                filters=col_33_r, kernel_size=(1, 1), padding='SAME',
                activation=activation, strides=1,
                name=None if name is None else ('%s-east-1' % name)
            ),
            Conv2D(
                filters=col_33, kernel_size=(3, 3), padding='SAME',
                activation=activation, strides=1,
                name=None if name is None else ('%s-east-2' % name)
            )
        ],
        [
            Conv2D(
                filters=col_55_r, kernel_size=(1, 1), padding='SAME',
                activation=activation, strides=1,
                name=None if name is None else ('%s-south-1' % name)
            ),
            Conv2D(
                filters=col_55, kernel_size=(5, 5), padding='SAME',
                activation=activation, strides=1,
                name=None if name is None else ('%s-south-2' % name)
            )
        ],
        [
            MaxPool2D(
                pool_size=(3, 3), strides=1, padding='SAME',
                name=None if name is None else ('%s-west-1' % name)
            ),
            Conv2D(
                filters=col_pool, kernel_size=(1, 1), padding='SAME',
                activation=activation, strides=1,
                name=None if name is None else ('%s-west-2' % name)
            )
        ],
        Conv2D(
            filters=col_11, kernel_size=(1, 1), padding='SAME',
            activation=activation, strides=1,
            name=None if name is None else ('%s-north' % name)
        )
    ])


def SandglassConv2D(filters_io: int, filters_mid: int, layers: int = 3, name: str = None):
    ret = []
    for i in range(max(3, layers if layers % 2 == 1 else (layers+1))):
        if i % 2 == 0:
            ret.append(
                Conv2D(
                    filters=filters_io, kernel_size=(3, 3), strides=1,
                    padding='same',
                    name=None if name is None else ('%s-%d' % (name, i))
                )
            )
        else:
            ret.append(
                Conv2D(
                    filters=filters_mid, kernel_size=(1, 1), strides=1,
                    padding='valid',
                    name=None if name is None else ('%s-%d' % (name, i))
                )
            )
    return ret


def Darknet_19() -> list:
    """Darknet-19

    输入尺寸`(224,224,3)`，输出尺寸`(1000,)`。

    Returns
    -------
    list
        层次列表。
    """
    return [
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
        Conv2D(filters=1000, kernel_size=(1, 1), padding='same'),
        AvgPool2D(pool_size=(7, 7)),
        Flatten(),
        Softmax()
    ]
