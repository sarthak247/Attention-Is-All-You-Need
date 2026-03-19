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
    Note: The authors scale the embeddings by sqrt(d_model) for some reason, I don't know why
          Also, these embeddings server as learnable parameters and are subject to adjustment during training to optimize the 
          model's loss function.
    '''
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model) # Create a mapping/dictionary between vocab size and d_model vector

    def forward(self, x):
        # in original paper, the authors scale by sqrt(d_model) so we do it too hence why embeddings get multiplied by that
        return self.embedding(x) * math.sqrt(self.d_model)
    

# Moving forward we build the positional encoding... keeping this vanilla for now
class PositionalEncoding(nn.Module):
    '''
    Input: d_model -> Model dimension (need to be same to the embeddings so that we can concatenate them later)
            seq_len -> Length of the sequence (number of tokens in the sequence)
            dropout -> Dropout rate to prevent overfitting
    Output: Positional encoding matrix of shape which will then be concatenated back to the input embedding vectors
    Explanation: Once we have the input embeddings, we somehow need to add some positional information to those embeddings as well.
                 Like, where each word comes in a sequence etc.. This is what this PE does.
    '''
    def __init__(self, d_model: int, seq_len: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)

        # this will be mostly copy pasted from original implementations as I have no fucking intention of writing trignometry again
        
        # Create a matrix of shape (seq_len, d_model)
        # Why seq_len x d_model... because we need vectors of size d_model for each token in the sequence
        pe = torch.zeros(seq_len, d_model) # Empty vector
        
        # Create a vector of shape (seq_len, 1)
        position = torch.arange(0, seq_len, dtype = torch.float).unsqueeze(1) # (seq_len, 1) to be used as the numerator in the sine and cosine functions
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        # Apply the sin to the even positions:
        pe[:, 0::2] = torch.sin(position * div_term) # [0, 2, etc etc]
        # Apply the cos to the odd positions:
        pe[:, 1::2]  = torch.cos(position * div_term) # [1, 3, etc etc])

        pe = pe.unsqueeze(0) # (1, seq_len, d_model) to be broadcasted to the input of the model

        self.register_buffer('pe', pe) # so that we save this tensor whenever we save the model and it is not part of the model's parameters.

    def forward(self, x):
        # Add this positional encoding we just created to every word inside the sentence
        x = x + (self.pe[:, x.shape[1], :]).requires_grad_(False) # because we do not want these PE to be learned.
        # and then we apply the dropout
        return self.dropout(x)


# Next we start building the blocks of the transformer.. Let's start with layernorm
class LayerNormalization(nn.Module):
    '''
    Input: eps -> to prevent division by zero
    Output: Normalized vector
    Explanation: Consider a batch of 3 items.. We calculate the mean and std for each of the 3 items and then normalize them.
    Note: We have two additonal parameters gamma and beta which are learnable parameters.
          The network learns to adjust these parameters as needed to accomodate for fluctuations in the data when needed as it can be
          too restrictive to keep the values in just [0,1] range
    '''
    def __init__(self, eps: float = 10**-6) -> None:
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1)) # Multiplicative learnable parameter
        self.bias = nn.Parameter(torch.zeros(1)) # Additive learnable parameter

    def forward(self, x):
        # calculate the mean and std of the last dimension
        mean = x.mean(dim=-1, keepdim=True) # why keeydim? Because we want to keep the dimensions of the tensor same as the input tensor.
        std = x.std(dim=-1, keepdim=True)
        # normalize the tensor
        return self.alpha * (x - mean) / (std + self.eps) + self.bias
    
# Next, we build the FeedForwardBlock (essentially two linear matrices which serve as expansion)
# Simple linear layer expansion blocks and nothing special here... For the sake of this paper we keep it 512 -> 2048 as original paper
# but industry has moved to other standards over the years (like the classical 2.67x expansion)
class FeedForwardBlock(nn.Module):
    '''
    Input: d_model -> Dimensions of the model
           d_ff -> Dimensions of the feed forward expansion
           dropout -> Dropout
    Output: Feedforward block
    Explanation: This block is used to perform a linear transformation of the input tensor.
                 We will go from (batch_size, seq_len, d_model) -> (batch, seq_len, d_ff) -> (batch, seq_len, d_model)
    '''
    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        self.dropout = nn.Dropout(dropout)
        self.linear1 = nn.Linear(d_model, d_ff) # 512 -> 2048 expansion
        self.linear2 = nn.Linear(d_ff, d_model) # 2048 -> 512 reduction

    def forward(self, x):
        # (batch, seq_le, d_model) -> (batch, seq_len, d_ff) using linear 1 -> (batch, seq_len, d_model) using linear 2
        return self.linear2(self.dropout(torch.relu(self.linear1(x))))
    

# Next, we build the Multi-Head Attention Block: The most important block over here







