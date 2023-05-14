"""Utility functions.
"""

import os
import sys
import pdb
import read_cmn
import numpy as np

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import torch
import torch.nn.functional as F
from torch.autograd import Variable


def string_to_index_list(s, char_to_index, end_token):
    """Converts a sentence into a list of indexes (for each character).
    """
    return [char_to_index[char] for char in s] + [end_token]  # Adds the end token to each index list


def translate_sentence(sentence, encoder, decoder, idx_dict, opts):
    """Translates a sentence from English to Pig-Latin, by splitting the sentence into
    words (whitespace-separated), running the encoder-decoder model to translate each
    word independently, and then stitching the words back together with spaces between them.
    """
    return ' '.join([translate(word, encoder, decoder, idx_dict, opts) for word in sentence.split()])


def translate(input_string, word_to_index,index_to_word, encoder, decoder, idx_dict, opts):
    """Translates a given string from English to Pig-Latin.
    """

    #char_to_index = idx_dict['char_to_index']
    #index_to_char = idx_dict['index_to_char']
    #start_token = idx_dict['start_token']
    end_token = '_'

    max_generated_chars = 20
    gen_string = ''
    #indexes = string_to_index_list(input_string, char_to_index, end_token)
    
    #to_be_append = []
    tokenized_sentence = read_cmn.sentence_to_index_one_sentence(input_string, word_to_index, False)
    # sentence_to_index_one_sentence()
    # read_cmn.sentencetoindex([input_string])
    # for word in input_string:
    #     to_be_append.append(word_to_index[word.lower()])

    indexes = to_var(torch.LongTensor(tokenized_sentence).unsqueeze(0), opts)  # Unsqueeze to make it like BS = 1
    #indexes = to_var(torch.LongTensor(tokenized_sentence), opts)  # Unsqueeze to make it like BS = 1

    encoder_annotations, encoder_last_hidden = encoder(indexes)

    decoder_hidden = encoder_last_hidden

    decoder_input = to_var(torch.LongTensor([[word_to_index['_']]]), opts)  # For BS = 1

    for i in range(max_generated_chars):
        decoder_output, decoder_hidden, attention_weights = decoder(decoder_input, decoder_hidden, encoder_annotations)
        ni = F.softmax(decoder_output, dim=1).data.max(1)[1]  # LongTensor of size 1
        print(ni)
        ni = int(ni[0])
        
        if ni == end_token:
            break
        else:
            gen_string += index_to_word[ni]
            decoder_input = to_var(torch.LongTensor([[ni]]), opts)

    return gen_string


def visualize_attention(input_string, encoder, decoder, idx_dict, opts, save='save.pdf'):
    """Generates a heatmap to show where attention is focused in each decoder step.
    """

    char_to_index = idx_dict['char_to_index']
    index_to_char = idx_dict['index_to_char']
    start_token = idx_dict['start_token']
    end_token = idx_dict['end_token']

    max_generated_chars = 20
    gen_string = ''

    all_attention_weights = []

    indexes = string_to_index_list(input_string, char_to_index, end_token)
    indexes = to_var(torch.LongTensor(indexes).unsqueeze(0), opts)  # Unsqueeze to make it like BS = 1

    encoder_annotations, encoder_hidden = encoder(indexes)

    decoder_hidden = encoder_hidden
    decoder_input = to_var(torch.LongTensor([[start_token]]), opts)  # For BS = 1

    produced_end_token = False

    for i in range(max_generated_chars):
        decoder_output, decoder_hidden, attention_weights = decoder(decoder_input, decoder_hidden, encoder_annotations)

        ni = F.softmax(decoder_output, dim=1).data.max(1)[1]  # LongTensor of size 1
        ni = ni[0]

        all_attention_weights.append(attention_weights.squeeze().data.cpu().numpy())

        if ni == end_token:
            produced_end_token = True
            break
        else:
            gen_string += index_to_char[ni]
            decoder_input = to_var(torch.LongTensor([[ni]]), opts)

    attention_weights_matrix = np.stack(all_attention_weights)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(attention_weights_matrix.T, cmap='bone')
    fig.colorbar(cax)

    # Set up axes
    ax.set_yticklabels([''] + list(input_string) + ['EOS'], rotation=90)
    ax.set_xticklabels([''] + list(gen_string) + (['EOS'] if produced_end_token else []))

    # Show label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.tight_layout()
    plt.savefig(save)

    plt.close(fig)

    return gen_string


def to_var(tensor, opts):
    """Wraps a Tensor in a Variable, optionally placing it on the GPU.

        Arguments:
            tensor: A Tensor object.
            cuda: A boolean flag indicating whether to use the GPU.

        Returns:
            A Variable object, on the GPU if cuda==True.
    """
    if opts.cuda:
        return Variable(tensor.cuda())
    elif opts.mps:
        return Variable(tensor.to("mps"))
    else:
        return Variable(tensor)


def create_dir_if_not_exists(directory):
    """Creates a directory if it doesn't already exist.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)


### chinese reader
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Usage:
    vocab.py --train-src=<file> --train-tgt=<file> [options] VOCAB_FILE

Options:
    -h --help                  Show this screen.
    --train-src=<file>         File of training source sentences
    --train-tgt=<file>         File of training target sentences
    --size=<int>               vocab size [default: 50000]
    --freq-cutoff=<int>        frequency cutoff [default: 2]
"""

from collections import Counter
from docopt import docopt
from itertools import chain
import json
import torch
from typing import List
import sentencepiece as spm


class VocabEntry(object):
    """ Vocabulary Entry, i.e. structure containing either
    src or tgt language terms.
    """
    def __init__(self, word2id=None):
        """ Init VocabEntry Instance.
        @param word2id (dict): dictionary mapping words 2 indices
        """
        if word2id:
            self.word2id = word2id
        else:
            self.word2id = dict()
            self.word2id['<pad>'] = 0   # Pad Token
            self.word2id['<s>'] = 1     # Start Token
            self.word2id['</s>'] = 2    # End Token
            self.word2id['<unk>'] = 3   # Unknown Token
        self.unk_id = self.word2id['<unk>']
        self.id2word = {v: k for k, v in self.word2id.items()}

    def __getitem__(self, word):
        """ Retrieve word's index. Return the index for the unk
        token if the word is out of vocabulary.
        @param word (str): word to look up.
        @returns index (int): index of word 
        """
        return self.word2id.get(word, self.unk_id)

    def __contains__(self, word):
        """ Check if word is captured by VocabEntry.
        @param word (str): word to look up
        @returns contains (bool): whether word is contained    
        """
        return word in self.word2id

    def __len__(self):
        """ Compute number of words in VocabEntry.
        @returns len (int): number of words in VocabEntry
        """
        return len(self.word2id)

    def __repr__(self):
        """ Representation of VocabEntry to be used
        when printing the object.
        """
        return 'Vocabulary[size=%d]' % len(self)

    def id2word(self, wid):
        """ Return mapping of index to word.
        @param wid (int): word index
        @returns word (str): word corresponding to index
        """
        return self.id2word[wid]

    def add(self, word):
        """ Add word to VocabEntry, if it is previously unseen.
        @param word (str): word to add to VocabEntry
        @return index (int): index that the word has been assigned
        """
        if word not in self:
            wid = self.word2id[word] = len(self)
            self.id2word[wid] = word
            return wid
        else:
            return self[word]

    def words2indices(self, sents):
        """ Convert list of words or list of sentences of words
        into list or list of list of indices.
        @param sents (list[str] or list[list[str]]): sentence(s) in words
        @return word_ids (list[int] or list[list[int]]): sentence(s) in indices
        """
        if type(sents[0]) == list:
            return [[self[w] for w in s] for s in sents]
        else:
            return [self[w] for w in sents]

    def indices2words(self, word_ids):
        """ Convert list of indices into words.
        @param word_ids (list[int]): list of word ids
        @return sents (list[str]): list of words
        """
        return [self.id2word[w_id] for w_id in word_ids]

    def to_input_tensor(self, sents: List[List[str]], device: torch.device) -> torch.Tensor:
        """ Convert list of sentences (words) into tensor with necessary padding for 
        shorter sentences.

        @param sents (List[List[str]]): list of sentences (words)
        @param device: device on which to load the tesnor, i.e. CPU or GPU

        @returns sents_var: tensor of (max_sentence_length, batch_size)
        """
        word_ids = self.words2indices(sents)
        sents_t = pad_sents(word_ids, self['<pad>'])
        sents_var = torch.tensor(sents_t, dtype=torch.long, device=device)
        return torch.t(sents_var)

    @staticmethod
    def from_corpus(corpus, size, freq_cutoff=2):
        """ Given a corpus construct a Vocab Entry.
        @param corpus (list[str]): corpus of text produced by read_corpus function
        @param size (int): # of words in vocabulary
        @param freq_cutoff (int): if word occurs n < freq_cutoff times, drop the word
        @returns vocab_entry (VocabEntry): VocabEntry instance produced from provided corpus
        """
        vocab_entry = VocabEntry()
        word_freq = Counter(chain(*corpus))
        valid_words = [w for w, v in word_freq.items() if v >= freq_cutoff]
        print('number of word types: {}, number of word types w/ frequency >= {}: {}'
              .format(len(word_freq), freq_cutoff, len(valid_words)))
        top_k_words = sorted(valid_words, key=lambda w: word_freq[w], reverse=True)[:size]
        for word in top_k_words:
            vocab_entry.add(word)
        return vocab_entry
    
    @staticmethod
    def from_subword_list(subword_list):
        vocab_entry = VocabEntry()
        for subword in subword_list:
            vocab_entry.add(subword)
        return vocab_entry


class Vocab(object):
    """ Vocab encapsulating src and target langauges.
    """
    def __init__(self, src_vocab: VocabEntry, tgt_vocab: VocabEntry):
        """ Init Vocab.
        @param src_vocab (VocabEntry): VocabEntry for source language
        @param tgt_vocab (VocabEntry): VocabEntry for target language
        """
        self.src = src_vocab
        self.tgt = tgt_vocab

    @staticmethod
    def build(src_sents, tgt_sents) -> 'Vocab':
        """ Build Vocabulary.
        @param src_sents (list[str]): Source subwords provided by SentencePiece
        @param tgt_sents (list[str]): Target subwords provided by SentencePiece
        """

        print('initialize source vocabulary ..')
        src = VocabEntry.from_subword_list(src_sents)

        print('initialize target vocabulary ..')
        tgt = VocabEntry.from_subword_list(tgt_sents)

        return Vocab(src, tgt)

    def save(self, file_path):
        """ Save Vocab to file as JSON dump.
        @param file_path (str): file path to vocab file
        """
        with open(file_path, 'w') as f:
            json.dump(dict(src_word2id=self.src.word2id, tgt_word2id=self.tgt.word2id), f, indent=2)

    @staticmethod
    def load(file_path):
        """ Load vocabulary from JSON dump.
        @param file_path (str): file path to vocab file
        @returns Vocab object loaded from JSON dump
        """
        entry = json.load(open(file_path, 'r'))
        src_word2id = entry['src_word2id']
        tgt_word2id = entry['tgt_word2id']
        return Vocab(VocabEntry(src_word2id), VocabEntry(tgt_word2id))

    def __repr__(self):
        """ Representation of Vocab to be used
        when printing the object.
        """
        return 'Vocab(source %d words, target %d words)' % (len(self.src), len(self.tgt))


def get_vocab_list(file_path, source, vocab_size):
    """ Use SentencePiece to tokenize and acquire list of unique subwords.
    @param file_path (str): file path to corpus
    @param source (str): tgt or src
    @param vocab_size: desired vocabulary size
    """ 
    spm.SentencePieceTrainer.train(input=file_path, model_prefix=source, vocab_size=vocab_size)     # train the spm model
    sp = spm.SentencePieceProcessor()                                                               # create an instance; this saves .model and .vocab files 
    sp.load('{}.model'.format(source))                                                              # loads tgt.model or src.model
    sp_list = [sp.id_to_piece(piece_id) for piece_id in range(sp.get_piece_size())]                 # this is the list of subwords
    return sp_list 



# if __name__ == '__main__':
#     args = docopt(__doc__)

#     print('read in source sentences: %s' % args['--train-src'])
#     print('read in target sentences: %s' % args['--train-tgt'])

#     src_sents = get_vocab_list(args['--train-src'], source='src', vocab_size=21000)         
#     tgt_sents = get_vocab_list(args['--train-tgt'], source='tgt', vocab_size=8000)
#     vocab = Vocab.build(src_sents, tgt_sents)
#     print('generated vocabulary, source %d words, target %d words' % (len(src_sents), len(tgt_sents)))

#     vocab.save(args['VOCAB_FILE'])
#     print('vocabulary saved to %s' % args['VOCAB_FILE'])
