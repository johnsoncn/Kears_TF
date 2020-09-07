# -*- coding: utf-8 -*-
""" 
@Time    : 2020/9/4 16:46
@Author  : HCF
@FileName: test.py
@SoftWare: PyCharm
"""
from keras.models import Sequential,load_model

from keras.utils.vis_utils import plot_model

model = load_model('mnist.h5')

plot_model(model,to_file="model_1.png",show_shapes=True,show_layer_names=True,rankdir="TB")
