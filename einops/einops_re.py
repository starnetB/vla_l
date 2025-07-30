'''
Author: 小d 2102690391@qq.com
Date: 2025-07-30 14:20:35
LastEditors: 小d 2102690391@qq.com
LastEditTime: 2025-07-30 17:49:57
FilePath: /nlp-tutorial-master/VLA/einops/einops_re.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
from einops import rearrange,repeat
import numpy as np 
import torch 


input_tensor = torch.tensor(np.array([1,2,3]))
print("raw shape:")
print(input_tensor.shape)
print(input_tensor)
input_tensor_2 = repeat(input_tensor,"h -> h w",w=2)
print("repeat 1:")
print(input_tensor.shape)
print(input_tensor)
input_tensor = repeat(input_tensor,"h -> h w c",w=2,c=3)
print("repeat 2:")
print(input_tensor.shape)
print(input_tensor)

# reduce 
# reduce 能够同时实现重排和压缩
from einops import reduce 
output_tensor = reduce(input_tensor,'b c (h h2) (w w2) -> b h w c','mean',h2=2,w2=2)

# 全局平均池化操作
y = reduce(input_tensor,'b c h w -> b c',reduction='mean')

# 2x2 最大池化操作
y = reduce(input_tensor, 'b c (h h1) (w w1) -> b c h w', reduction='max', h1=2, w1=2)
# you can skip names for reduced axes
y = reduce(input_tensor, 'b c (h 2) (w 2) -> b c h w', reduction='max')
# 通道级的均值归一化 
y = input_tensor - reduce(input_tensor, 'b c h w ->1 c 1 1','mean')

# rearranage
from einops import rearrange 
output_tensor = rearrange(input_tensor, 'b c h w -> b h w c')

# 增加维度 
input_tensor = rearrange(input_tensor, 'b h w c -> b 1 h w 1 c')

# Flatten操作
input_tensor = rearrange(input_tensor, 'b c h w -> b (c h w)')

# PixelShuffle中的空间重排操作
input_tensor = rearrange(input_tensor, 'b (h1 w1 c) h w -> b c (h h1) (w w1)', h1=2, w1=2)

# ShuffleNet中的通道打乱操作
input_tensor = rearrange(input_tensor, 'b (g1 g2 c) h w-> b (g2 g1 c) h w', g1=4, g2=4)

# 拆分张量 
y1,y2 = rearrange(input_tensor,'b (split c) h w -> split b c h w',split=2)

# 2.相关层，Rearrgen,Reduce EinMix 
import torch.nn as nn 
from einops.layers.torch import Rearrange 

model = nn.Sequential(
    nn.Conv2(3,6,kernel_size=5),
    nn.MaxPool2d(kernel_size=2),
    nn.Conv2d(6,16,kernel_size=2),
    Rearrange('b c h w -> b (c h w)'),
    nn.Linear(16*5*5,120),
    nn.ReLU(),
    nn.Linear(120,10),
)

#3 torch.einsum多维先行表达式的方法
import torch 

a = torch.randn((1,1,3,2))
b = torch.randn((1,1,1,2))

# 或 torch.einsum('b h i d, b h j d -> b h i j', [a,b])
# 相当于 torch.matmul(a,b.transpose(2,3))
c = torch.einsum('b h i d, b h j d -> b h i j', a, b)
print(c.shape)

# 5 pack 用法 

x = torch.randn(1,6,1,2) #形状(1,6,1,2

