import tensorflow as tf
from tensorflow.keras import *
from tensorflow.keras.layers import *


def conv2D(filters, kernel_size: tuple = (3, 3), **kwargs) -> Conv2D:
    return Conv2D(filters=filters, kernel_size=kernel_size, strides=1, activation='relu', padding='same', **kwargs)


def pool2D(pool_size=(2, 2), **kwargs) -> MaxPool2D:
    return MaxPool2D(pool_size=pool_size, **kwargs)


class ResNet(layers.Layer):
    def __init__(self, sequential: Sequential, **kwargs):
        super(ResNet, self).__init__(**kwargs)
        self.network = sequential

    def call(self, inputs, **kwargs):
        x = self.network(inputs)
        ret = layers.concatenate([inputs, x])
        return ret


class Inception(layers.Layer):
    def __init__(self, filters: tuple = (64, 96, 128, 16, 32, 32), **kwargs):
        if len(filters) != 6:
            raise ValueError('Filters must be a six-elements tuple.')
        super(Inception, self).__init__(**kwargs)
        self.network = [
            Conv2D(filters=filters[0], kernel_size=(1, 1), padding='VALID',
                   activation=tf.nn.relu,  strides=1),  # 1
            Sequential(
                [
                    Conv2D(filters=filters[1], kernel_size=(1, 1), padding='VALID',
                           activation=tf.nn.relu, strides=1),  # 2
                    Conv2D(filters=filters[2], kernel_size=(3, 3), padding='SAME',
                           activation=tf.nn.relu, strides=1),  # 2
                ]
            ),
            Sequential(
                [
                    Conv2D(filters=filters[3], kernel_size=(1, 1), padding='VALID',
                           activation=tf.nn.relu, strides=1),  # 3
                    Conv2D(filters=filters[4], kernel_size=(5, 5), padding='SAME',
                           activation=tf.nn.relu, strides=1),  # 3
                ]
            ),
            Sequential(
                [
                    MaxPool2D(pool_size=(3, 3), strides=1,
                              padding='SAME'),  # 4
                    Conv2D(filters=filters[5], kernel_size=(1, 1), padding='VALID',
                           activation=tf.nn.relu, strides=1),  # 4
                ]
            )
        ]

    def call(self, inputs, **kwargs):
        res = []
        inp = tf.nn.lrn(inputs)
        for e in self.network:
            res.append(e(inp))
        ret = layers.concatenate(res)
        return ret


NETWORK = {
    'dense': [
        # 1x112x112
        Conv2D(filters=64, kernel_size=(5, 5), padding='SAME', strides=1,
               name='Conv2D-1'),
        # 64x112x112
        MaxPool2D(pool_size=(3, 3), strides=2, padding='SAME',  name='Pool-1'),
        # 64x56x56
        Conv2D(filters=128, kernel_size=(3, 3),  padding='SAME',
               activation=tf.nn.relu, strides=1, name='Conv2D-2',
               bias_initializer=tf.constant_initializer(.5)),
        # 128x56x56
        MaxPool2D(pool_size=(3, 3), strides=2, padding='SAME', name='Pool-2'),
        # 128x28x28
        Inception(filters=(64, 96, 128, 16, 32, 32),  name='Inception-1'),
        # 256x28x28
        MaxPool2D(pool_size=(3, 3), strides=2, padding='SAME', name='Pool'),
        # 256x14x14
        Flatten(),
        Dense(units=196, activation=tf.nn.softmax,
              bias_initializer=tf.constant_initializer(.5), name='Dense')
        # 1x1x196
    ],
    'dinception': [
        Conv2D(filters=64, kernel_size=(5, 5), padding='SAME', strides=1,
               name='Conv2D-1'),
        MaxPool2D(pool_size=(3, 3), strides=2, padding='SAME',  name='Pool-1'),
        Conv2D(filters=128, kernel_size=(3, 3),  padding='SAME',
               activation=tf.nn.relu, strides=1, name='Conv2D-2',
               bias_initializer=tf.constant_initializer(.5)),
        MaxPool2D(pool_size=(3, 3), strides=2, padding='SAME', name='Pool-2'),
        Inception(filters=(64, 96, 128, 16, 32, 32),  name='Inception-1'),
        MaxPool2D(pool_size=(3, 3), strides=2, padding='SAME', name='Pool'),
        Inception(filters=(128, 128, 192, 32, 96, 64),  name='Inception-2'),
        Flatten(),
        Dense(units=196, activation=tf.nn.softmax,
              bias_initializer=tf.constant_initializer(.5), name='Dense')
        # 1x1x196
    ],
    'nodense': [
        Conv2D(filters=64, kernel_size=(5, 5), padding='SAME', strides=1,
               name='Conv2D-1'),
        Conv2D(filters=128, kernel_size=(3, 3),  padding='SAME',
               activation=tf.nn.relu, strides=1, name='Conv2D-2',
               bias_initializer=tf.constant_initializer(.5)),
        MaxPool2D(pool_size=(3, 3), strides=2, padding='SAME', name='Pool-1'),
        Inception(filters=(64, 96, 128, 16, 32, 32),  name='Inception-1'),
        MaxPool2D(pool_size=(3, 3), strides=2, padding='SAME', name='Pool'),
        Conv2D(filters=128, kernel_size=(1, 1), padding='SAME', strides=1,
               name='Conv2D-3'),
        MaxPool2D(pool_size=(2, 2), strides=2, padding='SAME',  name='Pool-2'),
        Conv2D(filters=1, kernel_size=(1, 1), padding='SAME', strides=1,
               bias_initializer=tf.constant_initializer(.5),
               activation=tf.nn.softmax, name='Conv2D-4'),
        Flatten(),
    ],
    'resnet': [
        Conv2D(filters=64, kernel_size=(3, 3), padding='SAME', strides=1,
               name='Conv2D-11'),
        Conv2D(filters=128, kernel_size=(3, 3),  padding='SAME',
               activation=tf.nn.relu, strides=1, name='Conv2D-12',
               bias_initializer=tf.constant_initializer(.2)),
        MaxPool2D(pool_size=(3, 3), strides=2, padding='SAME', name='Pool-1'),
        ResNet(
            Sequential(
                [
                    Conv2D(filters=192, kernel_size=(3, 3), padding='SAME', strides=1,
                           activation=tf.nn.relu,  name='Conv2D-21'),
                    Conv2D(filters=256, kernel_size=(3, 3),  padding='SAME',
                           activation=tf.nn.relu, strides=1, name='Conv2D-22'),
                ]
            )
        ),
        MaxPool2D(pool_size=(3, 3), strides=2, padding='SAME', name='Pool-2'),
        ResNet(
            Sequential(
                [
                    Conv2D(filters=192, kernel_size=(3, 3), padding='SAME', strides=1,
                           activation=tf.nn.relu,  name='Conv2D-31'),
                    Conv2D(filters=256, kernel_size=(3, 3),  padding='SAME',
                           activation=tf.nn.relu, strides=1, name='Conv2D-32'),
                ]
            )
        ),
        MaxPool2D(pool_size=(3, 3), strides=2, padding='SAME', name='Pool-3'),
        ResNet(
            Sequential(
                [
                    Conv2D(filters=480, kernel_size=(3, 3), padding='SAME', strides=1,
                           activation=tf.nn.relu,  name='Conv2D-31'),
                    Conv2D(filters=512, kernel_size=(3, 3),  padding='SAME',
                           activation=tf.nn.relu, strides=1, name='Conv2D-32'),
                ]
            )
        ),
        MaxPool2D(pool_size=(3, 3), strides=2, padding='SAME', name='Pool'),
        Flatten(),
        Dense(units=196, activation=tf.nn.softmax,
              bias_initializer=tf.constant_initializer(.5), name='Dense')
    ],
    'vgg': [
        conv2D(64, name='c_64_1'),
        conv2D(64, name='c_64_2'),
        pool2D(name='p_1'),
        conv2D(128, name='c_128_1'),
        conv2D(128, name='c_128_2'),
        pool2D(name='p_2'),
        conv2D(256, name='c_256_1'),
        conv2D(256, name='c_256_2'),
        conv2D(256, name='c_256_3'),
        pool2D(name='p_3'),
        conv2D(512, name='c_512_1'),
        conv2D(512, name='c_512_2'),
        conv2D(512, name='c_512_3'),
        pool2D(name='p_4'),
        conv2D(512, name='c_512_1'),
        conv2D(512, name='c_512_2'),
        conv2D(512, name='c_512_3'),
        pool2D(name='p_5'),
        Dense(units=4096, activation='relu', name="d_1"),
        Dense(units=4096, activation='relu', name="d_2"),
        Dense(units=1000, activation='softmax', name="d_3"),
    ]
}
