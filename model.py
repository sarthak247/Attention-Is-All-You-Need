import torch
import torch.nn as nn
import math


# Step 1: We create the Input Embeddings
# The input embeddings are just a mapping between a token and it's embedding.'
class InputEmbeddings(nn.Module):
    """
    Input: d_model: Size of the embeddings: 512 by default as per paper
            vocab_size: The size of the vocabulary. Eg. 30K words in our vocab. Then this will be used in our embedding layer
                        when we initialize it as (vocab_size, d_model) or 30K embeddings of size 512 each
    Output: (x) converted to embedding(x) or vector of size (512)
    """

    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)  # Our embedding layer which does exactly what we want, ie. convert from vocab_size to d_model

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)  # Since in the paper we multiply the embeddings with the square root of d_model
    

# Step 2: We create the positional embeddings
# The positional embeddings are added to the input embeddings as input for our model
class PositionalEnconding(nn.Module):
    """
    Input: d_model: Model size
            seq_len: Maximum length of a sentence/sequence
    Output: Positional Embedding for that position in the sequence
    """
    def __init__(self, d_model: int, seq_len: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = dropout

        # Create matrix of shape (seq_len, d_model) eg (6, 512) for a sequence with 6 words and each word represented as an embedding of size 512
        pe = torch.zeros(seq_len, d_model)

        # Create a vector of shape (seq_len, 1)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        # Apply the sin to even positions and cos to odd positions
        pe[:, 0::2] = torch.sin(position * div_term) # starting from 0 and every 2nd term from there
        pe[:, 1::2] = torch.cos(position * div_term) # starting from 1 and every 2nd term from there

        # Add a batch dimension to these (as we will be working with batches of sentences)
        pe = pe.unsqueeze(0) # (seq_len, d_model) -> (1, seq_len, d_model)

        # Will save the pe whenever we save the model
        self.register_buffer('pe', pe)

    def forward(self, x):
        # we add pe to the input embedding x and also make sure they are not learned as they are computed only once
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False) # Requires grad false means not learned 
        return self.dropout(x)
    
# Step 3: Next, we begin building encoder modules and we start off with the simplest one being Layer Normalization
class LayerNormalization(nn.Module):
    """
    Input: x (what we need to normalize)
    Output: Layer Normalized x
    """
    def __init__(self, eps: float = 10**-6) -> None:
        super().__init__()
        self.eps = eps # to prevent division by zero
        self.alpha = nn.Parameter(torch.ones(1)) # Multiplicative term
        self.bias = nn.Parameter(torch.zeros(1)) # Additive term

    def forward(self, x):
        mean = x.mean(dim = -1, keepdim = True) # Calculate mean across the last dimension (i.e. everything after batch dim)
        std = x.std(dim = -1, keepdim = True) # Again similar like mean
        return self.alpha * (x - mean) / (std + self.eps) + self.bias
    
# Step 4: Next, we build the feedforward block. This will be used both in encoder and decoder
class FeedForwardBlock(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        """
        Input: d_model: dimensions of model: 512 in this case
                d_ff: Dimensions of hidden layer: 2048 in this case
                dropout: Dropout
        Output: Linear2(dropout(relu(Linear1(x)))) -> x*linear1 then relu and dropout and then linaer2 to get to the original dimension
        """
        self.d_model = d_model
        self.d_ff = d_ff
        self.linear_1 = nn.Linear(d_model, d_ff) # W1 and b1 (512, 2048)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model) # W2 and b2 (2048, 512)

    def forward(self, x):
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))
    
# Step 5: Multi-Head Attention: The most important part
"""
Input: d_model and number of heads
Forward method input: Q, K, V and the mask (which will be used in the decoder)
"""
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, h: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.h = h # Number of heads
        self.dropout = nn.Dropout(dropout)

        # Assetion check.. Make sure d_model is divisible by h to get same number of dimensions in our heads
        assert d_model % h == 0, "d_model is not divisible by h"

        self.d_k = d_model // h # dimension of each head

        # create parameter matrices
        self.w_q = nn.Linear(d_model, d_model) # W_Q (512, 512)
        self.w_k = nn.Linear(d_model, d_model) # W_K (512, 512)
        self.w_v = nn.Linear(d_model, d_model) # W_V (512, 512)
        self.w_o = nn.Linear(d_model, d_model) # W_O (512, 512)

    # The reason we use staticmethod is so that we can call this anything without creating an instance first.
    # A simple MultiHeadAttention.attention() will do the job
    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):
        d_k = query.shape[-1] # get the shape of d_k from (batch, h, seq_len, d_k)

        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k) # (batch, h, seq_len, d_k) @ (batch, h, d_k, seq_len) -> (batch, h, seq_len, seq_len)
        if mask is not None:
            attention_scores.masked_fill_(mask == 0, -1e9) # replace with very small value wherver mask == 0 (the values which we need to hide/mask)
        if dropout is not None:
            attention_scores = dropout(attention_scores)

        return (attention_scores @ value), attention_scores

    def forward(self, q, k, v, mask):
        query = self.w_q(q) # q * w_q
        key = self.w_k(k) # k * w_k
        value = self.w_v(v) # v * w_v

        # Split into heads
        # (batch, seq_len, d_model) -> (batch, seq_len, h, d_k) -> Transpose -> (batch, h, seq_len, d_k) 
        # We transpose so that each heads gets to the partial embeddings but of the entire sequence
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).tranpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).tranpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1, 2)

        x, self.attention_scores = MultiHeadAttention.attention(query, key, value, mask, self.dropout)

        # Finally we concat the heads and then multiply by w_0
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.d_k) # (batch, h, seq_len, d_k) -> (batch, seq_len, h, d_k) -> (batch, seq_len, d_model)

        # (batch, seq_len, d_model) -> (batch, seq_len, d_model)
        return self.w_o(x)
    

# Step 6. Next, we build the residual connection to all of our components together
"""
Input: dropout
Forward input: x, sublayer where sublayer is the previous layer where we want to add the output
Output: x + drop(sublayer(norm(x))
"""
class ResidualConnection(nn.Module):
    def __init__(self, dropout: float) -> None:
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization()

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x))) # x + drop(sub(norm(x)))

# Step 7. Now that we have everything in place, we define our EncoderBlock. We will stack multiple of these togther to build our encoder
"""
Input: Self Attention Block: Essentially a multiheadattention block but since all the values will be within itself we call this self attention
        Feed Forward Block: Feed forward block
        It also needs two residual connections which we will define later as a modulelist
"""
class EncoderBlock(nn.Module):
    def __init__(self, self_attention_block: MultiHeadAttention, feed_forward_block: FeedForwardBlock, dropout: float) -> None:
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(2)]) # will give us two residual blocks

    # here we need a src_mask in order to mask the padding words as we do not want them to interact with the rest of the tokens
    def forward(self, x, src_mask):
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, src_mask)) # First residual connection between input x and output from self attention block
        x - self.residual_connections[1](x, self.feed_forward_block)
        return x
    
# Step 8: Now we need to build our encoder given our encoderblocks
class Encoder(nn.Module):
    def __init__(self, layers: nn.ModuleList) -> None:
        self.layers = layers
        self.norm = LayerNormalization

    def forward(self, x, mask):
        for layer in self.layers: # apply one after the another
            x = layer(x, mask)
        return self.norm(x)  # return with norm applied
    
# Step 9. Next, we define the decoder block. We already have everything built for us which the decoder needs
# Masked Multi Head Attention (self attention block), Cross Multi Head Attention and a FeedForwardBlock and 3 skip layers
class DecoderBlock(nn.Module):
    def __init__(self, self_attention_block: MultiHeadAttention, cross_attention_block: MultiHeadAttention, feed_forward_block: FeedForwardBlock, dropout: float) -> None:
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = FeedForwardBlock
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(3)])

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, tgt_mask)) # self attention with the target mask applied
        x = self.residual_connections[1](x, lambda x: self.cross_attention_block(x, encoder_output, encoder_output, src_mask)) # cross attention with the src_mask applied
        x = self.residual_connections[2](x, self.feed_forward_block)

        return x
    
# Step 10: Now that we have the decoderblock, we can build the decoder too similarly to how we built the encoder
class Decoder(nn.Module):
    def __init__(self, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        
        return self.norm(x)
    
# Step 11. Now that we have everything, we build our projectionlayer which converts back from d_model to vocab_size
class ProjectionLayer(nn.Module):
    def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        # (batch, seq_len, d_model) -> (batch, seq_len, vocab_size)
        return torch.log_softmax(self.proj(x), dim = -1) # Apply log softmax to the last layer of projection
    
# Step 12. Transformer block which will combine everything together
"""
Input: Encoder, Decoder,
    Src_embed: Source language embeddings
    tgt_embed: Target language embeddings
    src_pos: Source lanuage positional encodings
    tgt_pos: Target language positional encodings
    projection_layer: Projection layer
"""
class Transformer(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder, src_embed: InputEmbeddings, tgt_embed: InputEmbeddings, src_pos: PositionalEnconding, tgt_pos: PositionalEnconding, projection_layer: ProjectionLayer) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.projection_layer = projection_layer

    # Next, instead of defining a single forward function, we define individual encode, decode, project functions
    # This is because we do not wish to recompute the encoder output again and again and can use it multiple times
    def encode(self, src, src_mask):
        # takes in input source sentence and source mask
        src = self.src_embed(src) # Get embeddings for source sentence
        src = self.src_pos(src) # Add positional encodings for source sentence tokens
        return self.encoder(src, src_mask)
    
    def decode(self, encoder_output, src_mask, tgt, tgt_mask):
        # Takes in tgt and tgt_mask but also takes in encoder_output and src_mask (for the cross attention part)
        tgt = self.tgt_embed(tgt) # Similar as before
        tgt = self.tgt_pos(tgt)
        return self.decoder(tgt, encoder_output, src_mask, tgt_mask)

    def project(self, x):
        return self.projection_layer(x)
    
