#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
    @version: V-17-4-13
    @author: Linlifang
    @file: config.py
    @time: 17-4-13.下午3:02
"""
import os

path = os.getcwd()
# train

train_path = os.path.join(path, 'data/知产训练文件.txt')
test_path = os.path.join(path, 'data/知产测试文件.txt')
model_path = os.path.join(path, 'model/model')
output_path = os.path.join(path, 'output/output.data')
val_path = None
# emb_path = None
emb_path = os.path.join(path, 'data/WordVec4GuideTagUTF')
num_epochs = 100   # 迭代次数
num_steps = 200    # 句子长度
cpu_config = '/cpu:0'
gpu_config = '/gpu:0'
