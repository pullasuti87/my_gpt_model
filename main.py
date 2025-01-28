import torch
import torch.nn as nn
from torch.nn import functional as F
import tiktoken

# tokenizer -> https://github.com/google/sentencepiece
# import sentencepiece as sp

# configuration
batch_size = 4  # seq processed in parallel
context_length = 8  # max length of predictions
max_training_steps = 5000
evaluation_frequency = 500
evaluation_iterations = 100  # iterations for evaluation metrics
learning_rate = 3e-4  # Adam optimizer
device = "cpu"  # "cuda" if available

# hyperparameters
embedding_dimension = 128  # dimensionality of token embeddings
n_attention_heads = 8  # n of attention heads
n_transformer_layers = 6  # n of layers
dropout_rate = 0.1  # rate for regularization -> 0.0 no dropout

# registery number of specific starship
torch.manual_seed(1701)

with open("data.txt", "r", encoding="utf-8") as file:
    dataset = file.read()

# used by openai:
# gpt-4-turbo, gpt-4, gpt-3.5-turbo, text-embedding-ada-002,
# text-embedding-3-small, text-embedding-3-large
enc = tiktoken.get_encoding("cl100k_base")

test_string = "test"
encoded = enc.encode(test_string)
decoded = enc.decode(encoded)
assert test_string == decoded, "tokenizer test failed"

# dataset to tensor
tensor_data = torch.tensor(enc.encode(dataset))

# 80% training, 20% validation
split = int(0.8 * len(tensor_data))
train_data = tensor_data[:split]
val_data = tensor_data[split:]


def get_batch(split):
    """batch of data."""

    data = train_data if split == "train" else val_data
    random_indices = torch.randint(len(data) - context_length, (batch_size,))

    inputs = torch.stack([data[i : i + context_length] for i in random_indices])

    targets = torch.stack(
        [data[i + 1 : i + context_length + 1] for i in random_indices]
    )

    inputs, targets = inputs.to(device), targets.to(device)

    return inputs, targets


# test batch generation
# inputs, targets = get_batch("train")
# print("inputs shapes:", inputs.shape)
# print("targets:", targets.shape)
# print("inputs:", inputs)
# print("targets:", targets)

# inputs: tensor([[16820,    11,  5568,  2357,   308, 16820,  1980, 99682],
#         [12203, 60166,   597,  2319, 31824,  7643, 14635,   477],
#         [   71, 15492, 51890, 16373, 14360,  2357, 93464, 22243],
#         [ 1441,  2357,   267,  2357, 39004, 12949,    11, 73730]])
# targets: tensor([[   11,  5568,  2357,   308, 16820,  1980, 99682,  1937],
#         [60166,   597,  2319, 31824,  7643, 14635,   477,   300],
#         [15492, 51890, 16373, 14360,  2357, 93464, 22243, 15492],
#         [ 2357,   267,  2357, 39004, 12949,    11, 73730, 10248]])


class AttentionHead(nn.Module):
    """self-attention head with causal masking."""

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(embedding_dimension, head_size, bias=False)
        self.query = nn.Linear(embedding_dimension, head_size, bias=False)
        self.value = nn.Linear(embedding_dimension, head_size, bias=False)
        self.register_buffer(
            "causal_mask", torch.tril(torch.ones(context_length, context_length))
        )
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        batch_size, seq_length, _ = x.shape

        # (batch_size, seq_length, head_size)
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)

        # attention scores
        scores = q @ k.transpose(-2, -1) * (embedding_dimension**-0.5)
        # apply causal mask
        scores = scores.masked_fill(
            self.causal_mask[:seq_length, :seq_length] == 0, float("-inf")
        )

        weights = F.softmax(scores, dim=-1)  # (batch_size, seq_length, seq_length)
        weights = self.dropout(weights)

        # weighted values
        output = weights @ v  # (batch_size, seq_length, head_size)
        return output


dummy = torch.randn(batch_size, context_length, embedding_dimension)
head = AttentionHead(head_size=16)
output = head(dummy)
print("attentionhead:", output.shape)
print("attentionhead:", output)


def main():
    pass


if __name__ == "__main__":
    main()
