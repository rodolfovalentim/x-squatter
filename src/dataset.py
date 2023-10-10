import os
import torch
import torchaudio
import numpy as np
import pandas as pd
import pytorch_lightning as pl
from typing import Optional
from torch.utils.data import Dataset, DataLoader, random_split
import itertools
from typing import Dict, List
import math
import re
import ipatok

from .symbols import Symbols
import panphon

class Tokenizer(object):

    _num_reserved_tokens = 6
    _pad_idx = 0
    _sos_idx = 1
    _eos_idx = 2
    _unk_idx = 3
    _mlt_idx = 4
    _sep_idx = 5
    
    _ft = None

    def __init__(self, 
                 splitter=None, 
                 vocab=None, 
                 vocab_dict=None, 
                 seq_len=25,
                 characters_to_remove=None, 
                 phon_vector=False, 
                 data_type="grapheme"):
        
        self.vocab = vocab
        self.vocab_dict = vocab_dict
        self.phon_vector = phon_vector
        self.__reversed_vocab_dict = None      
        self.__seq_len = seq_len        
        self.__splitter = splitter
        self.data_type= data_type
        
        if self.phon_vector:
            self._ft = panphon.FeatureTable()
            self.__splitter = (self._ft.ipa_segs, {})
            
            self.special_tokens = {
              '<PAD>': self._pad_idx,
              '<SOS>': self._sos_idx, 
              '<EOS>': self._eos_idx, 
              '<UNK>': self._unk_idx,
              '<MLT>': self._mlt_idx,
              '<SEP>': self._sep_idx
            }
            
            self.special_token_vector = {}

            for token, value in self.special_tokens.items():
                vector = np.zeros((1, len(self._ft.names)))
                vector[:, :] = value
                self.special_token_vector[token] = vector
                self.special_token_vector[value] = vector

        self.characters_to_remove = characters_to_remove
        if characters_to_remove is not None:
            self.characters_to_remove = "[" + characters_to_remove + "]"
    
    def get_feature_size(self):
        return len(self._ft.names)
       
    def normalize(self, text):
        if self.data_type == "phoneme":
            text = "".join(ipatok.tokenise(text, replace=True))
        return text
    
    def split(self, text):
        text = self.normalize(text)
        
        if self.__splitter is not None:
            splitted = self.__splitter[0](str(text), **self.__splitter[1])
        else:
            if self.characters_to_remove is not None:
                splitted = list(str(re.sub(self.characters_to_remove, "", text)))
            else:
                splitted = list(str(text))
        return splitted
    
    def load_predefined_vocab(self):
        if self.data_type == 'phoneme':
            self.vocab = sorted(set(Symbols.ALL_GRAPHEME_CHARACTERS + Symbols.ALL_PHONEME_SYMBOLS))
        elif self.data_type == 'grapheme':
            self.vocab = sorted(set(Symbols.ALL_GRAPHEME_CHARACTERS))   
        
        self.vocab_dict = { char: idx for idx, char in enumerate(self.vocab, start=self._num_reserved_tokens) }
        self.vocab_dict.update({'<PAD>': self._pad_idx, '<SOS>': self._sos_idx, '<EOS>': self._eos_idx, '<UNK>': self._unk_idx})
        self.vocab_dict.update({'<MLT>': self._mlt_idx, '<SEP>': self._sep_idx }) # multilanguage special token
        self.vocab = self.vocab_dict.keys()

    def build_vocab(self, inputs):
        if self.characters_to_remove is not None:
            self.vocab = [ self.split(re.sub(self.characters_to_remove, "", text)) for text in inputs ]
        else:
            self.vocab = [ self.split(text) for text in inputs ]
                
        self.vocab = set(list(itertools.chain.from_iterable(self.vocab)))
        self.vocab_dict = { char: idx for idx, char in enumerate(self.vocab, start=self._num_reserved_tokens) }
        self.vocab_dict.update({'<PAD>': self._pad_idx, '<SOS>': self._sos_idx, '<EOS>': self._eos_idx, '<UNK>': self._unk_idx})
        self.vocab_dict.update({'<MLT>': self._mlt_idx,'<SEP>': self._sep_idx }) # multilanguage special token
        self.vocab = self.vocab_dict.keys()
            
    @property
    def reversed_vocab(self):
        if self.__reversed_vocab_dict is None:
            self.__reversed_vocab_dict = { v: k for k, v in self.vocab_dict.items() } 
        return self.__reversed_vocab_dict
    
    def __len__(self):
        return len(self.vocab_dict.keys())
    
    def split_sentence(self, sentence, add_sos_marker=False, add_eos_marker=False):
        words = sentence.split(" ")
        new_sentence = []
        for word in words:
            new_sentence += self.split(word) + ['<SEP>']
        
        new_sentence = new_sentence[:-1]
        if add_sos_marker:
            new_sentence = ['<SOS>'] + new_sentence 

        if add_eos_marker:
            new_sentence = new_sentence + ['<EOS>']

        new_sentence = (new_sentence + self.__seq_len * ['<PAD>'])[:self.__seq_len]        
        return new_sentence
      
    def encode_sentences_to_digits(self, inputs, add_sos_marker=False, add_eos_marker=False, return_attentions=False):
        new_inputs = []
        attentions = []
        
        for idx, line in enumerate(inputs): 
            sentence = self.split_sentence(line, add_sos_marker=add_sos_marker, add_eos_marker=add_eos_marker)
            new_sentence = [ self.vocab_dict.get(token, self._unk_idx) for token in sentence ]
            new_sentence = np.array(new_sentence)
            attention = new_sentence == self._pad_idx

            new_inputs.append(new_sentence)
            attentions.append(attention)

        new_inputs = np.array(new_inputs)
        attentions = np.array(attentions)

        if return_attentions:
            return new_inputs, attentions
        else:
            return new_inputs
    
    def encode_sentences_to_vectors(self, inputs, add_sos_marker=False, add_eos_marker=False, return_attentions=True):
        new_inputs = []
        attentions = []

        for idx, sentence in enumerate(inputs):
            words = self.split_sentence(sentence, add_sos_marker=add_sos_marker, add_eos_marker=add_eos_marker)
            
            new_line = []
            for word in words:
                if word in self.special_tokens.keys():
                    new_line.append(self.special_token_vector.get(word))
                else:
                    new_line.append(self._ft.word_to_vector_list(word, numeric=True))
                                    
            attention = [ token == '<PAD>' for token in words]
            attention = np.array(attention, dtype=bool)
           
            new_line = np.array(new_line, dtype=np.float32).squeeze(1)
            new_inputs.append(new_line)
            attentions.append(attention)
        
        new_inputs = np.array(new_inputs)
        attentions = np.array(attentions)

        if return_attentions:
            return new_inputs, attentions
        else:
            return new_inputs
        
    def encode_sentences(self, inputs, add_sos_marker=False, add_eos_marker=False, return_attentions=False, return_digits=True):
        # TODO return object
        
        if not self.phon_vector:
            return self.encode_sentences_to_digits(inputs,  add_sos_marker, add_eos_marker, return_attentions)
        elif not return_digits:
            return self.encode_sentences_to_vectors(inputs, add_sos_marker, add_eos_marker, return_attentions)
        else:
            if return_attentions:
                vectors, attentions = self.encode_sentences_to_vectors(inputs, add_sos_marker, add_eos_marker, return_attentions)
                digits = self.encode_sentences_to_digits(inputs,  add_sos_marker, add_eos_marker, return_attentions=False)
                return vectors, attentions, digits
            else:
                vectors = self.encode_sentences_to_vectors(inputs, add_sos_marker, add_eos_marker, return_attentions)
                return vectors
            
    def decode_sentences(self, inputs, add_sentence_marker=False):
        new_inputs = []
        
        for line in inputs:
            new_line = [ self.reversed_vocab.get(token, '<UNK>') for token in line ]
            new_inputs.append(new_line)
            
        return new_inputs
    
    def save(self, path):
        np.save(path, { 'vocab_dict': self.vocab_dict, 'seq_len': self.__seq_len }, allow_pickle=True)
        
    def load(self, path):
        loaded_data = np.load(path, allow_pickle=True).item()
        self.vocab_dict = loaded_data['vocab_dict']
        self.__seq_len = loaded_data['seq_len']
        self.vocab = list(self.vocab_dict.keys())
        
    def get_vocab_size(self):
        return len(self.vocab)
        
class TextDataset(Dataset):
    def __init__(self, inputs, attentions, labels) -> None:
      
        super().__init__()
        
        self.inputs = inputs
        self.attentions = attentions
        self.labels = labels

    def __len__(self):
        return len(self.inputs)
      
    def __getitem__(self, index):
        return {'input_vectors': self.inputs[index], 
                'labels': self.labels[index], 
                'attention_mask': self.attentions[index] }

class MultiLanguageTextDataModule(pl.LightningDataModule):
    def __init__(self, 
                 dataset,
                 batch_size: int = 32,
                 num_workers: int = 2,
                 validation_split: float = 0.1):
        
        super().__init__()
        
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        val_test_size = int(validation_split * len(self.dataset))
        train_size = len(self.dataset) - (2 * val_test_size)
        
        self.train_data, self.val_data, self.test_data = random_split(self.dataset, 
                                                                      [train_size, 
                                                                       val_test_size, 
                                                                       val_test_size])
        
    def train_dataloader(self):
        return DataLoader(self.train_data, num_workers=self.num_workers, batch_size=self.batch_size, shuffle=True, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_data, num_workers=self.num_workers, batch_size=self.batch_size, shuffle=False, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.test_data, num_workers=self.num_workers, batch_size=self.batch_size, shuffle=False, pin_memory=True)