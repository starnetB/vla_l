'''
Author: 小d 2102690391@qq.com
Date: 2025-07-30 14:07:25
LastEditors: 小d 2102690391@qq.com
LastEditTime: 2025-07-30 14:16:50
FilePath: /nlp-tutorial-master/VLA/RT1/robotic_transformer_pytorch.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
from __future__ import annotations 

import torch 
from torch.nn import Module,ModuleList 
import torch.nn.functional as F 
from torch import nn, einsum, Tensor 

from typing import Callable 
from beartype import beartype 

from einops import pack,unpack,repeat,reduce,rearrange 
from einops.layers.torch import Rearrange,Reduce 

from functools import partial 

from classifier_free_guidance_pytorch import TextConditioner, AttentionTextConditioner, classifier_free_guidance 

def exists(val):
    return val is not None 

def default(val,d):
    return val if exists(val) else d 

def cast_tuple(val,length = 1):
    return val if isinstance(val,tuple) else ((val,)*length)

def pack_one(x,pattern):
    return pack([x],pattern) 

