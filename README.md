# my_chatgpt (simple transformerLM)

my attempt to understand language models a little better,
relies heavily on the _Building Transformer Models with Attention_ book
and Andrej Karpathy's transformer implementations

- self-attention
- multi-head attention
- position embeddings
- tiktoken
- training data -> Aleksis Kiven seitsemän veljestä

## needs

- python3
- pytorch

## classes

## parameters

```python
sequences_per_batch = 0  # sequences processed in parallel
context_length = 0  # max length of predictions
max_training_steps = 0
evaluation_frequence = 0
evaluation_steps = 0
learning_rate = 0 # for optimizer
device = "cpu" # or cuda
```


