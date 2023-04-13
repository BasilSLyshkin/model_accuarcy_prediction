import torch
from typing import Type
from torch import nn
from scripts.dataset import TextDataset
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from torch.distributions.categorical import Categorical


class LanguageModel(nn.Module):
    def __init__(self, dataset: TextDataset, embed_size: int = 256, hidden_size: int = 256,
                 rnn_type: Type = nn.RNN, rnn_layers: int = 1):
        """
        Model for text generation
        :param dataset: text data dataset (to extract vocab_size and max_length)
        :param embed_size: dimensionality of embeddings
        :param hidden_size: dimensionality of hidden state
        :param rnn_type: type of RNN layer (nn.RNN or nn.LSTM)
        :param rnn_layers: number of layers in RNN
        """
        super(LanguageModel, self).__init__()

        self.embedding = nn.Embedding(
            num_embeddings=dataset.vocab_size,
            embedding_dim=embed_size,
            padding_idx=dataset.pad_id
        )
        self.rnn = rnn_type(
            input_size=embed_size,
            hidden_size=hidden_size, 
            num_layers=rnn_layers,
            batch_first=True
        )
        self.linear = nn.Linear(
            in_features=hidden_size,
            out_features=2
        )

        
    def forward(self, indices: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """
        Compute forward pass through the model and
        return logits for the next token probabilities
        :param indices: LongTensor of encoded tokens of size (batch_size, length)
        :param lengths: LongTensor of lengths of size (batch_size, )
        :return: FloatTensor of logits of shape (batch_size, length, vocab_size)
        """

        embeds = self.embedding(indices)
        
        packed_embeds = pack_padded_sequence(
            input=embeds, 
            lengths=lengths,
            batch_first=True,
            enforce_sorted=False,
        )

        packed_output, _ = self.rnn(packed_embeds)

        output, _ = pad_packed_sequence(
            sequence=packed_output, 
            batch_first=True
        )
        
        logits = self.linear(output.mean(1))
        return logits