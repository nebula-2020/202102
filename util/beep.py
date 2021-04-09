#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
声音提示。
"""
import winsound
import time
from threading import *


def beeps(times: int = 3, frequency: int = 5000, duration: float = 1., delay: float = .6):
    for _ in range(abs(times)):
        winsound.Beep(max(37, min(32767, abs(frequency))),
                      int(abs(duration)*1000))
        time.sleep(abs(delay))


def beep(times: int = 3, frequency: int = 5000, duration: float = 1., delay: float = .6):
    t = Thread(target=beeps, args=(times, frequency, duration, delay))
    t.start()
    return t
