# this is the actual model

import torch
import torch.nn as nn
from torch.nn import functional as F

# tokenizer -> https://github.com/google/sentencepiece
import sentencepiece as sp

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

if __name__ == "__main__":
    print("not operational!")
    print(dataset)
