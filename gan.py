#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import random
from util.io import read_mnist
from util.cnn import Inception
import util.beep as beep
import tensorflow as tf
import time
from tensorflow.keras import *
from tensorflow.keras.layers import *
import matplotlib.pyplot as pyp
import traceback


@tf.function
def train_step(imgs, noises):
    noise = [tf.random.normal(shape=NOISE_SHAPE[1:]) for _ in range(noises)]
    noise = tf.convert_to_tensor(noise)
    gen_lost = None
    disc_lost = None
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:

        gen_imgs = gen(noise, training=True)  # 训练生成网络

        y_hat_disc_p = disc(imgs, training=True)  # 正样本训练判别网络
        y_hat_disc_n = disc(gen_imgs, training=True)  # 负样本训练判别网络

        y_gen = tf.ones_like(y_hat_disc_n)
        gen_lost = loss(y_gen, y_hat_disc_n)  # 生成网络损失

        y_disc_p = tf.convert_to_tensor([[.9, .1]]*noises)
        disc_lost_p = loss(y_disc_p, y_hat_disc_p)  # 判别网络正样本损失

        y_disc_n = tf.convert_to_tensor([[.1, .9]]*noises)
        disc_lost_n = loss(y_disc_n, y_hat_disc_n)  # 判别网络负样本损失

        disc_lost = disc_lost_p + disc_lost_n  # 判别网络损失

    gen_grad = gen_tape.gradient(gen_lost, gen.trainable_variables)
    disc_grad = disc_tape.gradient(disc_lost, disc.trainable_variables)

    gen_opt.apply_gradients(zip(gen_grad, gen.trainable_variables))
    disc_opt.apply_gradients(zip(disc_grad, disc.trainable_variables))

    imgs = None


def read_x(size):
    x, _, _ = read_mnist()
    x = x.numpy().tolist()
    x = random.sample(x, min(len(x), size))
    return x


def fit(x=None, size=128, epochs=100, show=100, read=5000):
    sign = False
    if x is None:
        sign = True
        x = read_x(size*10)
    bar_size = 100

    pyp.imshow(x[0])
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
    for e in range(epochs):
        if e % bar_size == 0:
            print('epochs %d~%d ' % (int(e), int(e)+bar_size))
        s = min(size+min(int(size*e/epochs), int(size)), len(x))
        inputs = tf.convert_to_tensor(random.sample(x, s))
        train_step(inputs, noises=s)

        if e % show == 0:
            print_img()
        if sign and e % read == 0 and e > 0:
            x = read_x(size*10)


def print_img(show=False):
    imgs = []
    labels = []
    for i in range(3*5):
        o = gen(tf.random.normal(shape=NOISE_SHAPE), training=False)
        imgs.append(o)
        labels.append(disc(o, training=False))
    for i in range(len(imgs)):
        pyp.subplot(3, 5, i+1)
        pyp.imshow(imgs[i].numpy()[0])
        pyp.xticks([])
        pyp.yticks([])
        pyp.title(
            '[%.2f,%.2f]' % tuple(labels[i][0].numpy().tolist())
        )
        pyp.axis('off')
    if not os.path.exists(IMG_PATH):
        os.makedirs(IMG_PATH)
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


NETWORK_NAME = 'gan'
IMG_PATH = os.path.join('data', NETWORK_NAME, 'imgs')
MODEL_NAME = os.path.join('data', NETWORK_NAME)
DISC_NAME = 'disc'
GEN_NAME = 'gen'
NOISE_SHAPE = (1, 100)
IMG_SHAPE = (1, 28, 28, 1)
DATA_SIZE = 32

GEN = [
    Dense(units=128, bias_initializer='zeros', activation=tf.nn.relu),
    Dense(units=784, activation=tf.nn.tanh),
    Reshape(target_shape=IMG_SHAPE[1:]),
]

DISC = [
    Flatten(),
    Dense(units=784, bias_initializer='zeros', activation=tf.nn.relu),
    Dense(units=2, bias_initializer='zeros', activation=tf.nn.sigmoid),
]

loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
gen_opt = optimizers.Adam(learning_rate=0.00011, epsilon=1e-8)
disc_opt = optimizers.Adam(learning_rate=0.0001)
try:
    gen = tf.keras.models.load_model(os.path.join(MODEL_NAME, GEN_NAME))
    print('gen loaded.')
except:
    print('gen rebuilt.')
    gen = Sequential(GEN)
    disc = Sequential(DISC)

    gen.compile(optimizer=gen_opt, loss=loss)
    gen.build(input_shape=NOISE_SHAPE)
    gen.summary()
try:
    disc = tf.keras.models.load_model(os.path.join(MODEL_NAME, DISC_NAME))
    print('disc loaded.')
except:
    print('disc rebuilt.')
    disc = Sequential(DISC)

    disc.compile(optimizer=disc_opt, loss=loss)
    disc.build(input_shape=IMG_SHAPE)
    disc.summary()
try:
    fit(size=DATA_SIZE, epochs=8000, show=500, read=9000)
    disc.save(os.path.join(MODEL_NAME, DISC_NAME))
    gen.save(os.path.join(MODEL_NAME, GEN_NAME))
    print_img(True)
except:
    traceback.print_exc()
    beep.beep(times=5, delay=0.1, duration=0.25)
