import torch
import torch.nn as nn
from torch.nn import functional as F
import tiktoken

# tokenizer -> https://github.com/google/sentencepiece
# import sentencepiece as sp

# configuration
sequences_per_batch = 32  # sequences processed in parallel
context_length = 64  # max length of predictions
max_training_steps = 5000
evaluation_frequence = 500
evaluation_steps = 100
learning_rate = 3e-4  # adam optimizer
device = "cpu"  # cuda
# might need more, not sure

# registry number of known starship
torch.manual_seed(1701)


text = open("dataset.txt", "r", encoding="utf-8")
dataset = text.read()

# used by openai:
# gpt-4-turbo, gpt-4, gpt-3.5-turbo, text-embedding-ada-002,
# text-embedding-3-small, text-embedding-3-large
enc = tiktoken.get_encoding("cl100k_base")
s = "test"
encode = enc.encode(s)
decode = enc.decode(encode)
assert s == decode, "tiktoken fail"

tensor_data = torch.tensor(enc.encode(dataset))
# tensor([  937,   964, 84839,  ..., 14200, 26577,   382])

# 80% training, 20% validation
point = int(0.8 * len(tensor_data))
# datasets
training_ds = tensor_data[:point]
validation_ds = tensor_data[point:]


def training_batch(ds):
    if ds == "train":
        current_ds = training_ds
    else:
        current_ds = validation_ds

    # randint wants tuple -> (sequences_per_batch,)
    random_pos = torch.randint(len(current_ds) - context_length, (sequences_per_batch,))
    input_seq = torch.stack([current_ds[i : i + context_length] for i in random_pos])
    # input [a,b,c,d], target [b,c,d,e]
    target_seq = torch.stack(
        [current_ds[i + 1 : i + context_length + 1] for i in random_pos]
    )
    input_seq = input_seq.to(device)
    target_seq = target_seq.to(device)

    return input_seq, target_seq


# WORKS
# x_input, y_target = training_batch(training_ds)
# print("\n\n", "input:", x_input, "\n\n", "target:", y_target)
# input: tensor([[14727, 10126,  5169,  ...,   307, 15492, 51055],
#        [   76,  2357,    13,  ...,   297,  1543, 18757],
#        [ 2357,   321,   344,  ..., 69560,    11, 13080],
#        ...,
#        [50291,  2357,  3900,  ...,  1295,   616, 94702],
#        [ 1980, 25554, 56047,  ...,   380,  2357,  4415],
#        [53756, 12778,   658,  ...,  6870,    11, 34065]])
#
# target: tensor([[10126,  5169,    11,  ..., 15492, 51055,   484],
#        [ 2357,    13,   735,  ...,  1543, 18757,   383],
#        [  321,   344, 14360,  ...,    11, 13080,   258],
#        ...,
#        [ 2357,  3900,    76,  ...,   616, 94702,    11],
#        [25554, 56047,    13,  ...,  2357,  4415,   680],
#        [12778,   658, 67997,  ...,    11, 34065,   664]])


def loss():
    results = {}
    # dropout, normalization off
    model.eval()

    for ds in ["train", "val"]:
        losses = torch.zeros(eval_iters)

        for i in range(eval_iters):
            x_input, y_target = training_batch(ds)
            targets, loss = model(x_input, y_target)
            losses[i] = loss.item()

        results[ds] = losses.mean()
    # dropo.., normal.. on
    model.train()

    return results


def main():
    pass


if __name__ == "__main__":
    main()
