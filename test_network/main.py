import torch
import torchvision

import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

trans = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
])

ds_mnist = torchvision.datasets.MNIST(".", transform=trans)

print(ds_mnist[0][0].shape)
print(ds_mnist[0][0].squeeze().shape)
plt.imshow(ds_mnist[0][0].squeeze())
plt.show()
print(ds_mnist[0][1])
