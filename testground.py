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

segment_size = 4
group_size = 8

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
    # check enough tokens
    groups = torch.randint(len(data) - group_size, (segment_size,))

    input_seq = []
    for i in groups:
        input_seq.append(data[i : i + group_size])
    input_group = torch.stack(input_seq)

    prediction_seq = []
    for i in groups:
        prediction_seq.append(data[i + 1 : i + group_size + 1])
    prediction_group = torch.stack(prediction_seq)

    #  print("\n")
    #  print("input:", input_group, "\n")
    #  print("prediction:", prediction_seq, "\n")
    return input_group, prediction_group


# print_sequences()
x_input, y_prediction = get_groups(training_data)
print(x_input.shape)
print(y_prediction.shape)


# bigram relationships, nn.Module -> base class for neural network module
class BigramModel(torch.nn.Module):
    def __init__(self, unique_token_size, embed_size=32):
        super().__init__()
        # maps tokens their vector numbers -> reads next token from table
        self.token_table = torch.nn.Embedding(unique_token_size, embed_size)
        # takes embedding size and projects it to uniqu token size
        # fix memory error
        self.output = torch.nn.Linear(embed_size, unique_token_size)

    def forward(self, input_seq):
        # raw predictions from embedding table
        embeds = self.token_table(input_seq)
        raw_predictions = self.output(embeds)

        return raw_predictions


# print(len(chars))
# get unique size from tiktoken
unique_token_size = enc.n_vocab
model = BigramModel(unique_token_size)
model_output = model(x_input)
print(model_output)
print(model_output.shape)
