import pickle
import pathlib
from typing import Dict, List, Any, Union

import torch
import yaml
import glob
import numpy as np
import pandas as pd

from .dataset import Tokenizer
from .transformer import LitTransformer

from typing import List
import warnings
import epitran
import espeakng
import ipatok
from dataclasses import dataclass
from itertools import product
from tqdm.auto import trange, tqdm

import json
import re 

warnings.filterwarnings('ignore')

def read_config(path: str) -> Dict[str, Any]:
    with open(path, 'r') as stream:
        config = yaml.load(stream, Loader=yaml.FullLoader)
    return config


def save_config(config: Dict[str, Any], path: str) -> None:
    with open(path, 'w+', encoding='utf-8') as stream:
        yaml.dump(config, stream, default_flow_style=False)

@dataclass
class Generation:
    phoneme : str = None
    pronunciation_language : str = None
    domain : str = None
    homophone_language: str = None
    homophones : List[str] = None
    
    def __str__(self):
        return f'''Candidates for: {self.domain}
        read in: {self.pronunciation_language} 
        pronounced as: {self.phoneme}
        written in: {self.homophone_language} 
        are: [ { ", ".join(self.homophones)} ]'''

        
class Pipeline(object):
    
    release_name = "Current Version"
    
    __str = """Submitted to: TBD
Published at: TBD
Note: Soundsquatter uses seed in the output to conditionate the generation to a specific language. 
      Uses a multiple linear layer to encode the panphon feature representation. """
    
    def __str__(self):
        return self.__str
    
    @property
    def release(self):
        return self.release_name
    
    def __init__(self, model_path, checkpoint=None):
        
        model_path = pathlib.Path(model_path)
        
        self.voices = {
                 'fr-fr': epitran.Epitran('fra-Latn'),
                 'en-us': espeakng.ESpeakNG(voice='en-us'),
                 'en-gb': espeakng.ESpeakNG(voice='en-gb'),
                 'it': epitran.Epitran('ita-Latn'),
                 'pt-br': epitran.Epitran('por-Latn'), 
                 'es-es': epitran.Epitran('spa-Latn'),
        }
        
        with open(model_path / 'commandline_args.txt', 'r') as fp:
            training_params = json.load(fp)
    
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.phoneme = Tokenizer(phon_vector=True, seq_len=training_params['seq_len'], data_type='phoneme')
        self.phoneme.load(model_path / 'phoneme_vocab.npy')
        vocab_size_src = self.phoneme.get_feature_size()
        vocab_size = len(self.phoneme.vocab)
        
        self.grapheme = Tokenizer(phon_vector=False, seq_len=training_params['seq_len'], data_type='grapheme')
        self.grapheme.load(model_path / 'grapheme_vocab.npy')
        # self.grapheme.load_predefined_vocab(data_type='grapheme')
        vocab_size_tgt = len(self.grapheme.vocab)

        if checkpoint is not None:
            self.model = LitTransformer.load_from_checkpoint(model_path / 'checkpoints' / checkpoint,
                                                         vocab_size_src=vocab_size_src,
                                                         vocab_size_tgt=vocab_size_tgt, 
                                                         max_len=training_params['seq_len']).to(self.device)
    
        else:
            self.model = LitTransformer.load_from_checkpoint(model_path / 'last_model.ckpt',
                                                         vocab_size_src=vocab_size_src,
                                                         vocab_size_tgt=vocab_size_tgt, 
                                                         max_len=training_params['seq_len']).to(self.device)    
    
    def normalize_phoneme(self, phoneme):
        return "".join(ipatok.tokenize(phoneme, replace=True))
    
    def generate_homophones(self,
                            sentence, 
                            pronunciation_language='en-us', 
                            target_language=None, 
                            number_candidates=15, 
                            max_decoded_sentence_length=50,
                            p=0.85, t=1, 
                            word_replace=None, 
                            validate_sentence=False,
                            disable_progress_bar=False, 
                            from_phoneme=False):

        if not from_phoneme:
            phoneme = self.transliterate(sentence, voice=self.voices[pronunciation_language], word_replace=word_replace)
        else:
            phoneme = sentence

        phoneme = self.normalize_phoneme(phoneme)
        
        encoded_input, attentions = self.phoneme.encode_sentences([phoneme], return_attentions=True, return_digits=False)
                
        max_decoded_sentence_length = len(sentence) + 3

        sentence_started = ''
        if target_language is not None:
            sentence_started = '[' + target_language + '] '
        
        encoded_start = self.grapheme.encode_sentences([sentence_started], add_sos_marker=True, add_eos_marker=False)
        # print(encoded_start)
        
        src = torch.tensor(encoded_input)        
        tgt = torch.tensor(encoded_start)
        
        # print(self.grapheme.decode_sentences(encoded_start))
        
        attentions = torch.tensor(attentions)
        start_index = len(sentence_started)
        
        words, probs = self.beam_search(src, tgt, attentions, start_index=start_index, p=p, t=t,
                                  number_candidates=number_candidates, 
                                  max_decoded_sentence_length=max_decoded_sentence_length, 
                                  disable_progress_bar=disable_progress_bar)

        return self.process_output(sentence, words, probs, pronunciation_language, target_language, validate_sentence=validate_sentence, phoneme=phoneme)
             
    def transliterate(self, grapheme, voice=None, word_replace=None):
        phoneme = ''

        if word_replace is not None:
            for key, value in word_replace.items():
                grapheme = grapheme.replace(key, value)
        
        if type(voice)  == espeakng.ESpeakNG:
            phoneme = voice.g2p(grapheme, ipa=2)
        elif type(voice)  == epitran._epitran.Epitran: 
            phoneme = voice.transliterate(grapheme)

        return phoneme

    def clean_sentence(self, sentence):
        return "".join(sentence).replace('<PAD>', "").replace('<SOS>', "").replace('<SEP>', " ").replace("<UNK>", "").split("<EOS>", 1)[0]

    def validate_sentence(self, sentence):
        pattern = re.compile('\[.*\] [\w+]+')

        if pattern.fullmatch(sentence.strip()):
            result = re.sub('\[.*\] ', '', sentence)
            return result
        else:
            return ""

    def process_output(self, target, words, probs, pronunciation_language, target_language, validate_sentence=True, phoneme=None):
        candidates = [self.clean_sentence(sentence) for sentence in self.grapheme.decode_sentences(words)]

        if validate_sentence:
            candidates = [ self.validate_sentence(sentence)  for sentence in candidates ]
            candidates = [ sentence for sentence in candidates if sentence != '' ]
            
        sorted_candidates = list(reversed(sorted(zip(candidates, probs), key=lambda x: x[1])))
        df_sorted_candidates = pd.DataFrame(sorted_candidates, columns=['squatting', 'probability'])
        generation = Generation(
                    phoneme=phoneme,
                    pronunciation_language=pronunciation_language,
                    domain=target,
                    homophone_language=target_language,
                    homophones=df_sorted_candidates['squatting'].unique())
        return generation

    def beam_search(self, tokenized_input, start_token, attentions,
                    start_index=1, number_candidates=100, p=0.01, t=1, 
                    max_decoded_sentence_length=30, disable_progress_bar=True):
        
        self.model.eval()

        curr_state = start_token.repeat(1, 1)
        prob = np.ones((1))

        for i in trange(start_index, max_decoded_sentence_length + start_index - 1, disable=disable_progress_bar, leave=False):
            if not torch.is_tensor(curr_state):
                curr_state = torch.tensor(curr_state)

            tokenized_input = tokenized_input.to(self.device)
            curr_state = curr_state.to(self.device)
            attentions = attentions.to(self.device)
    
            output = self.model(tokenized_input, curr_state, attentions)
                        
            pred = output['output']
            pred = pred.permute(1, 0, 2).detach().cpu().numpy()

            sampled_token_probs_t = torch.nn.functional.softmax(torch.tensor(pred[:, i, :])/t, dim=1).numpy()

            sampled_token_for_stack = []
            sampled_prob_for_stack = []
            input_for_stack = []
            attention_for_stack = []
            
            curr_for_stack = []

            for token_input, curr_input_token, token_softmax, attention in zip(tokenized_input, 
                                                                               curr_state,
                                                                               sampled_token_probs_t,
                                                                               attentions):
                if self.device == 'cuda':
                    token_input = token_input.cpu()
                    curr_input_token = curr_input_token.cpu()
                    attention = attention.cpu()
                    
                args_sorted_than_p = np.argsort(-token_softmax)
                cumsum_tokens_prob = np.cumsum(token_softmax[args_sorted_than_p])
                threshold_arg = np.argmax(cumsum_tokens_prob > p)

                probs_bigger_than_p = token_softmax[args_sorted_than_p][:threshold_arg + 1]
                args_bigger_than_p = args_sorted_than_p[:threshold_arg + 1]
                actual_k = len(args_bigger_than_p)
                
                input_for_stack.append(np.tile(token_input[None, :, :], (actual_k, 1, 1)))
                curr_for_stack.append(np.tile(curr_input_token, (actual_k, 1)))
                attention_for_stack.append(np.tile(attention, (actual_k, 1)))
                
                sampled_prob_for_stack.append(probs_bigger_than_p)
                sampled_token_for_stack.append(args_bigger_than_p)
                
            token_stack = tuple( k[:, np.newaxis] for k in sampled_token_for_stack )           
            token_stack = np.vstack(token_stack).squeeze(axis=1)
        
            tokenized_input = torch.tensor(np.vstack(input_for_stack))
            attentions = torch.tensor(np.vstack(attention_for_stack))
        
            curr_state = np.vstack(curr_for_stack)
            curr_state[:, i+1] = token_stack
            
            new_prob = prob[:, np.newaxis] * sampled_prob_for_stack
            
            # Check if the matrix is ragged
            is_ragged = len(set(len(row) for row in sampled_prob_for_stack)) > 1
            if is_ragged:
                new_prob = np.diag(new_prob)    
            prob = np.concatenate(new_prob)                

            curr_state = np.array(curr_state)
            threshold = np.argsort(-prob)[:number_candidates]
            mask = threshold

            tokenized_input = tokenized_input[mask]
            curr_state = curr_state[mask]
            prob = prob[mask]
            attentions = attentions[mask]
            
            if np.all(np.any(np.isin(curr_state, [2]), axis=1)):
                return curr_state, prob            
                        
        return curr_state, prob
    
    def compare_homophones(self, generation: Generation, known_homophones: List[str]):
        gen = set(generation.homophones)
        hom = set(known_homophones)

        return { 'pronunciation_language': generation.pronunciation_language, 'pronunciation': generation.phoneme, 
                 'homophone_language': generation.homophone_language, 'homophones': list(generation.homophones), 'known_homophones' : list(hom),
                 'found': list(gen.intersection(hom)), 'missing':  list(hom - gen), 'extra': list(gen-hom) } 

    def grid_search(self, data, pronunciation_language, target_language, probabilities=[], temperatures=[], number_candidates=100):
        results = []
        
        combinations = list(product(probabilities, temperatures))
        for p, t in tqdm(combinations, total=len(combinations), leave=True):
            for targets in tqdm(data, total=len(data), leave=False):
                targets = [ target.lower() for target in targets ]
                
                word = targets[0]
                generation = self.generate_homophones(word, pronunciation_language=pronunciation_language,
                                                      target_language=target_language, 
                                                      max_decoded_sentence_length=50,
                                                      number_candidates=number_candidates,
                                                      p=p,
                                                      t=t, 
                                                      disable_progress_bar=True, 
                                                      validate_sentence=True)

                result = self.compare_homophones(generation, targets)
                result['p'] = p
                result['t'] = t
                results.append(result)
                    
        grid = [ {'p': ele['p'], 't': ele['t'], 'ratio': len(ele['found'])/len(ele['known_homophones'])} for ele in results ]

        return grid, results
    
