#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
    @version: V-17-4-13
    @author: zhangyuo
    @file: config.py
    @time: 17-4-13.下午3:02
"""

train_path = u'../data/知产训练文件.txt'
test_path = u'../data/知产测试文件.txt'
model_path = u'../model/'
output_path = u'../output/output.data'
val_path = None
emb_path = u'../model/WordVec4GuideTagUTF'
# 词与id对应关系的路径
CHAR_ID_PATH = u'../data/char_id'
# label与id对应关系的路径
LABEL_ID_PATH = u'../data/label_id'

num_epochs = 100  # 迭代次数
SENT_LENGTH = 200  # 句子长度

cpu_config = '/cpu:0'
gpu_config = '/gpu:0'
