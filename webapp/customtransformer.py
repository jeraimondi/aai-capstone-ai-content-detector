import joblib # serialization
import math # mathematical functions
import torch # machine learning framework
from torch import Tensor # multi-dimensional matrix
import torch.nn as nn # neural networks
import torchtext # data processing utilities
from torchtext.data import get_tokenizer # tokenizer function


# define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# define variables
model_weights = 'custom_model_weights.pt' # model weights file
vocab = 'vocab.pkl' # model vocabulary file
tokenizer = get_tokenizer('basic_english') # tokenizer function


# define class for positional encoding
# NOTE* - Class code retrieved and modified from:
# https://pytorch.org/tutorials/beginner/transformer_tutorial.html
class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(p=dropout)


    # forward pass
    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[batch_size, seq_length, embedding_dim]``
        """
        # obtain shape of input tensor
        batch_size, seq_length, embedding_dim = x.size()

        # calculate positional encoding, results in [1, seq_length, 1] shape
        position = torch.arange(seq_length).unsqueeze(0).unsqueeze(-1)

        # calculate divisor term for sine and cosine functions
        div_term = torch.exp(torch.arange(0, self.d_model, 2) * (-math.log(10000.0) / self.d_model))

        # initialize tensor of 0s
        pe = torch.zeros(batch_size, seq_length, self.d_model)

        # calculate sine for even indices and cosine for odd
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)

        # add positional encoding to input tensor, set to same device
        x = x + pe.to(device)

        # apply dropout and return
        return self.dropout(x)


# define class to build custom transformer model
class TransformerModel(nn.Module):
    def __init__(
        self,
        num_embeddings, # size of vocab
        embedding_dim, # embedding dimensions
        d_model, # number of expected input features
        nhead, # number of heads for multi-attention
        dim_feedforward, # dimension of feedforward network
        dropout, # dropout value
        activation, # intermediate layer activation function
        num_layers # number of transformer layers
    ):
        super().__init__()

        # embedding layer
        self.embedding_layer = nn.Embedding(num_embeddings, embedding_dim)

        # positional encoding module
        # NOTE* - Module use obtained from:
        # https://pytorch.org/tutorials/beginner/transformer_tutorial.html
        self.pos_encoder = PositionalEncoding(d_model, dropout)

        # transformer layers
        self.transformer_layers = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model = d_model, # number of expected input features
                nhead = nhead, # number of heads for multi-attention
                dim_feedforward=dim_feedforward, # dimension of feedforward network
                dropout=dropout, # dropout value
                activation=activation, # intermediate layer activation function
                batch_first=True # (batch, seq, feature) format

            ),
            num_layers=num_layers # number of transformer layers
        )

        # predictive layer, 1 output for binary classification
        self.predictive_layer = nn.Linear(embedding_dim, 1)

        # sigmoid activation function for output between 0 and 1
        self.sigmoid_activation = nn.Sigmoid()


    # function for forward pass
    def forward(self, x):
        x = self.embedding_layer(x)
        x = self.pos_encoder(x)
        x = self.transformer_layers(x)
        x = torch.mean(x, dim=1) # mean pooling
        x = self.predictive_layer(x)
        x = self.sigmoid_activation(x)
        return x


# function to predict
def predict_with_transformer(model, text, tokenizer, vocab):
    # tokenize the text
    tokenized_text = tokenizer(text)

    # generate input_ids using vocab
    input_ids = [vocab[token] if token in vocab else 0 for token in tokenized_text]

    # convert to tensor, set to device
    input_tensor = torch.tensor([input_ids]).to(device)

    # generate and return prediction
    prediction = model(input_tensor)
    return prediction.item()


# function to load custom transformer model
def load_model():
    custom_model = TransformerModel(
        num_embeddings=208251, # size of vocab
        embedding_dim=768, # embedding dimensions
        d_model=768, # number of expected input features
        nhead=12, # number of heads for multi-attention
        dim_feedforward=1024, # dimension of feedforward network
        dropout=0.1, # dropout value
        activation='gelu', # intermediate layer activation function
        num_layers=2 # number of transformer layers
    )

    #load model weights
    custom_model.load_state_dict(torch.load(model_weights, map_location=torch.device(device)))

    # set model to eval mode
    custom_model.eval()

    return custom_model


# function to load vocabulary from file
def load_vocab():
    return joblib.load(vocab)


# load vocabulary
vocab = load_vocab()

# load model
custom_model = load_model()