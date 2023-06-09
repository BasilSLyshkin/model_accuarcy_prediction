import os
import torch
from typing import Union, List, Tuple
from sentencepiece import SentencePieceTrainer, SentencePieceProcessor
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
import numpy as np


class TextDataset(Dataset):
    TRAIN_VAL_RANDOM_SEED = 42
    VAL_RATIO = 0.05

    def __init__(self, data, split: str, data_file: str = 'datasets/data.txt', sp_model_prefix: str = 'tokenizer_models/',
                 vocab_size: int = 5000, normalization_rule_name: str = 'nmt_nfkc_cf',
                 model_type: str = 'bpe', max_length: int = 128):
        """
        Dataset with texts, supporting BPE tokenizer
        :param data_file: txt file containing texts
        :param train: whether to use train or validation split
        :param sp_model_prefix: path prefix to save tokenizer model
        :param vocab_size: sentencepiece tokenizer vocabulary size
        :param normalization_rule_name: sentencepiece tokenizer normalization rule
        :param model_type: sentencepiece tokenizer model type
        :param max_length: maximal length of text in tokens
        """
        if not os.path.isfile(sp_model_prefix +str(model_type)+ str(vocab_size)+'.model'):
            # train tokenizer if not trained yet
            SentencePieceTrainer.train(
                input=data_file, vocab_size=vocab_size,
                model_type=model_type, model_prefix=sp_model_prefix +str(model_type) + str(vocab_size),
                normalization_rule_name=normalization_rule_name,
                pad_id = 0, bos_id = 1, eos_id = 2, unk_id = 3
            )
        # load tokenizer from file
        self.sp_model = SentencePieceProcessor(model_file=sp_model_prefix+str(model_type)+str(vocab_size)+ '.model')
        if split == 'train':
            self.dataset = data.filter(lambda example: example['sample'] == 'train').map(remove_columns = ['sample'])
        elif split == 'validation':
            self.dataset = data.filter(lambda example: example['sample'] == 'validation').map(remove_columns = ['sample'])
        elif split == 'test':
            self.dataset = data.filter(lambda example: example['sample'] == 'test').map(remove_columns = ['sample'])
        elif split == 'all': 
            self.dataset = data.filter(lambda example: example['sample'] != 'test').map(remove_columns = ['sample'])

        self.pad_id, self.unk_id, self.bos_id, self.eos_id = \
            self.sp_model.pad_id(), self.sp_model.unk_id(), \
            self.sp_model.bos_id(), self.sp_model.eos_id()
        self.max_length = max_length
        self.vocab_size = self.sp_model.vocab_size()

    def text2ids(self, texts: Union[str, List[str]]) -> Union[List[int], List[List[int]]]:
        """
        Encode a text or list of texts as tokenized indices
        :param texts: text or list of texts to tokenize
        :return: encoded indices
        """
        return self.sp_model.encode(texts)

    def ids2text(self, ids: Union[torch.Tensor, List[int], List[List[int]]]) -> Union[str, List[str]]:
        """
        Decode indices as a text or list of tokens
        :param ids: 1D or 2D list (or torch.Tensor) of indices to decode
        :return: decoded texts
        """
        if torch.is_tensor(ids):
            assert len(ids.shape) <= 2, 'Expected tensor of shape (length, ) or (batch_size, length)'
            ids = ids.cpu().tolist()

        return self.sp_model.decode(ids)

    def __len__(self):
        """
        Size of the dataset
        :return: number of texts in the dataset
        """
        return len(self.dataset)

    def __getitem__(self, item: int) -> Tuple[torch.Tensor, int]:
        """
        Add specials to the index array and pad to maximal length
        :param item: text id
        :return: encoded text indices and its actual length (including BOS and EOS specials)
        """
        
        encoded = [self.bos_id] + self.sp_model.encode(self.dataset[item]['review']) + [self.eos_id]
        padded = torch.full((self.max_length,), fill_value = self.pad_id)
        padded[:len(encoded)] = torch.tensor(encoded[:self.max_length])
        
        length = min(len(encoded), self.max_length)
        target = self.dataset[item]['star']
        return self.dataset[item]['date'], padded, length, target