# check tensorflow
import torch
import tiktoken

data = open("dataset.txt", "r")
dataset = data.read()
# print(dataset)
# print(len(dataset))

chars = sorted(set(dataset))
# print(chars)

# TOKENIZATION
# simple char to int tokenization
# others to try sentencepiece, tiktoken

"""
str_to_int = {}
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
"""

# s = "my name is slim Shady"
# print(s)
# test = encode_chars(s)
# print(test)
#
# print(decode_ints(test))
#
#
# s2 = "sauna"
# print(s2)
# test2 = encode_chars(s2)
# print(test2)
#
# print(decode_ints(test2))

# print(str_to_int, end="\n\n")
# print(int_to_str)

# tensor_data = torch.tensor(encode_chars(dataset))
# print(tensor_data)
# print(tensor_data.shape)
# print(tensor_data[:500])


""" trying tiktoken instead """
enc = tiktoken.get_encoding("cl100k_base")
tensor_data = torch.tensor(enc.encode(dataset))

# training
percent = int(0.85 * len(tensor_data))
training_data = tensor_data[:percent]

# validation data
val_data = tensor_data[percent:]

# print("training", training_data.shape)
# print("val", val_data.shape)

segment_size = 10
group_size = 5

# print(training_data[: segment_size + 1])


def print_sequences():
    print("\n")
    for t in range(1, segment_size + 1):

        sequence = training_data[:t]
        prediction = training_data[t]

        print("sequence:", sequence, "prediction:", prediction)


# make sure that random number stays same
torch.manual_seed(1987)


# using to make random_numbers torch.randint(low, high, size)
# SIZE needs to be tuble
def get_groups(data):
    # make sure there is enough tokens
    groups = torch.randint(len(data) - group_size, (group_size,))

    input_seq = []
    for i in groups:
        # slice operation
        input_seq.append(data[i : i + segment_size])
    input_group = torch.stack(input_seq)

    prediction_seq = []
    for i in groups:
        # slice operation
        prediction_seq.append(data[i + 1 : i + segment_size + 1])
    prediction_group = torch.stack(prediction_seq)

    print("\n")
    print("input:", input_group, "\n")
    print("prediction:", prediction_seq, "\n")
    return input_group, prediction_group


print_sequences()
get_groups(training_data)


# bigram relationships, nn.Module -> base class for neural network module
class BigramModel(torch.nn.Module):
    def __init__(self, unique_token_size):
        super().__init__()
        # maps tokens their vector numbers -> reads next token from table
        self.token_table = torch.nn.Embedding(unique_token_size, unique_token_size)
        print(self.token_table)


print(len(chars))
unique_token_size = len(chars)
BigramModel(unique_token_size)
