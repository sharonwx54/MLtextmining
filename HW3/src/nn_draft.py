from utils import *
from torch import nn


def mod_pad_tensor_shift(tensor, length, padding_index=DEFAULT_PADDING_INDEX):
    """ Pad a ``tensor`` to ``length`` with ``padding_index``.
    This is a modified version of pad_tensor, but end up NOT being used
    Args:
        tensor (torch.Tensor [n, ...]): Tensor to pad.
        length (int): Pad the ``tensor`` up to ``length``.
        padding_index (int, optional): Index to pad tensor with.
    Returns
        (torch.Tensor [length, ...]) Padded Tensor.
    """
    # since the pre-process of data using StaticEncoder allocate 1-4 to other char, we start w 5 for the first word
    # here we shift the tensor embedding by 4 to put the most frequent word at position 1
    tensor = tr.sub(tensor, 4)
    # for all words that's not appearing, we drop it
    tensor = tensor[tensor > 0]
    n_padding = length - tensor.shape[0]
    if n_padding == 0:
        return tensor
    if n_padding < 0:
        return tensor[0:length]
    padding = tensor.new(n_padding, *tensor.shape[1:]).fill_(padding_index)
    return tr.cat((tensor, padding), dim=0)


class RNN_Model(nn.Module):
    # Originally using Pytorch for NN model, but switch to tensorflow and keras after reading documents
    # this is the UNUSED pytorch version of model
    def __init__(self, hidden_size, lin_size, layer_size, embed_mtx):

        # Initialize some parameters for your model
        self.hidden_size = hidden_size
        self.layer_size = layer_size
        drp = 0.2

        # Layer 1: Word2Vec Embeddings.
        self.embedding = nn.Embedding(TOP_WORD_INDEX, EMBED_VEC_SIZE)
        self.embedding.weight = nn.Parameter(tr.tensor(embed_mtx, dtype=tr.float32))
        self.embedding.weight.requires_grad = False

        # Layer 2: Dropout1D(0.2)
        self.embedding_dropout = nn.Dropout2d(drp)

        # Layer 3: One- directional LSTM, default is tanh function for nonlinearity
        self.lstm = nn.LSTM(EMBED_VEC_SIZE, hidden_size, # nonlinearity="tanh",
                            bidirectional=False, batch_first=True, dropout=drp)

        # Layer 4: Dense Output layer
        self.output = nn.Linear(lin_size, 1)

    def forward(self, x):
        batch_size = x.size(0)
        hidden = self.init_hidden(batch_size)

        h_embedding = self.embedding(x[0])
        h_embedding = tr.squeeze(self.embedding_dropout(tr.unsqueeze(h_embedding, 0)))

        h_lstm, _ = self.lstm(h_embedding)
        output = self.output(h_lstm)

        return output, hidden

    def init_hidden(self, batch_size):
        # This method generates the first hidden state of zeros which we'll use in the forward pass
        # We'll send the tensor holding the hidden state to the device we specified earlier as well
        hidden = tr.zeros(self.n_layers, batch_size, self.hidden_size)
        return hidden