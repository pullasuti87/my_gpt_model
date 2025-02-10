# my notes

## setup

```python
import torch
import torch.nn as nn
from torch.nn import functional as F
import tiktoken
```

- `torch`: deep learning
- `torch.nn`: neural network layers and components
- `torch.nn.functional`: activation functions and other operations
- `tiktoken`: openai tokenizer

## configuration

- batch_size
>- how many sequences are processed simultaneously during training
>- bigger spped up training, require more training

- context_length
>- max length of input sequences model can process
>- how much context model use to make predictions

- max_training_steps
>- number of training iterations
>- each step, model makes predictions and updates its weights

evaluation_frequency
evaluation_iterations
learning_rate
device = "cpu"

embedding_dimension
n_attention_heads
n_transformer_layers
dropout_rate
