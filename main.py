import torch
import torch.nn as nn
from torch.nn import functional as F
import tiktoken

# tokenizer -> https://github.com/google/sentencepiece
# import sentencepiece as sp

# configuration
sequences_per_batch = 0  # sequences processed in parallel
context_length = 0  # max length of predictions
max_training_steps = 0
evaluation_frequence = 0
evaluation_steps = 0
learning_rate = 0
device = "cpu"
# might need more, not sure

# registry number of known starship
torch.manual_seed(1701)


text = open("dataset.txt", "r", encoding="utf-8")
dataset = text.read()

# same as chatgpt
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
    current_ds = training_ds if ds == "train" else validation_ds

    # randint wants tuple -> (sequences_per_batch,)
    random_pos = torch.randint(len(current_ds) - context_length, (sequences_per_batch))
    input_seq = torch.stack([current_ds[i : i + context_length] for i in random_pos])
    # input [a,b,c,d], target [b,c,d,e]
    target_seq = torch.stack(
        [current_ds[i + 1 : i + context_length + 1] for i in random_ps]
    )
    input_seq = input_seq.to(device)
    target_seq = target_seq.to(device)

    return input_seq, target_seq


def main():
    pass


if __name__ == "__main__":
    main()
