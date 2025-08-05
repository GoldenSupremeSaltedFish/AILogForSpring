#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Intel Arc GPU 模型模块
"""

from .textcnn import TextCNN
from .fasttext import FastTextModel
from .model_factory import ModelFactory

__all__ = ['TextCNN', 'FastTextModel', 'ModelFactory'] 