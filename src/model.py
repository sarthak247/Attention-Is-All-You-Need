'''
What we need here:
1. Embedding layer: Kinda a dictionary between a number and a vector
2. Positional Encoding: The goat of the original paper... I might supplement this later with additional ones like RoPE etc but that is for later


'''

# Import libraries:
import torch
import torch.nn as nn
import math


# Building blocks: Embedding Layer first
class InputEmbeddings(nn.Module):
    '''
    Input: Vocab_size -> how many words we have in the vocabulary
            d_model -> Model dimension (as per transformers paper)
    Output: Embedding layer
    Explanation: Just a mapping between a number and a vector (done by embedding layer):
                 Each time we pass a number, we get that same embedding vector
                 This is used to convert the input IDs to embeddings
    '''
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model) # Create a mapping/dictionary between vocab size and d_model vector

    def forward(self, x):
        # in original paper, the authors scale by sqrt(d_model) so we do it too hence why embeddings get multiplied by that
        return self.embedding(x) * math.sqrt(self.d_model)
    




