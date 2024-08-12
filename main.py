# pytorch more familiar, haven't been using tensorflow
# import torch


data = open("dataset.txt", "r")
dataset = data.read()
# print(dataset)
# print(len(dataset))

chars = sorted(set(dataset))
# print(chars)

# TOKENIZATION

# str to int
str_to_int = {}
# int to str
int_to_str = {}
for i, c in enumerate(chars):
    str_to_int[c] = i
    int_to_str[i] = c


def encode_chars(s):
    t = []
    for c in s:
        t.append(str_to_int[c])
    return t


def decode_ints(ints):
    s = ""
    for i in ints:
        s += int_to_str[i]
    return s


s = "my name is slim Shady"
print(s)
test = encode_chars(s)
print(test)

print(decode_ints(test))

# print(str_to_int, end="\n\n")
# print(int_to_str)
