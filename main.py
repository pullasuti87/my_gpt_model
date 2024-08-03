# pytorch more familiar, haven't been using tensorflow
import torch


data = open("dataset.txt", "r")
dataset = data.read()
print(dataset)
print(len(dataset))
