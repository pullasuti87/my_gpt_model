# pytorch more familiar, haven't been using tensorflow
import torch


data = open("dataset.txt", "r")
dataset = data.read()
# print(dataset)
# print(len(dataset))

chars = sorted(set(dataset))
# print(chars)

# chars to int
str_to_int = {}
for i, c in enumerate(chars):
    str_to_int[c] = i


# int to char
int_to_str = {}
for i, c in enumerate(chars):
    int_to_str[i] = c


print(str_to_int, end="\n\n")
print(int_to_str)
