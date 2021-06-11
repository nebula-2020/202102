#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
工具，创建更加随意的网络结构。
"""
from tensorflow.keras import Model
from tensorflow.keras.layers import concatenate


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
    |:--|:--|:--|:--|
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
        if type(obj) == list:
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
