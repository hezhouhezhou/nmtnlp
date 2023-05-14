import os
import pdb
import argparse
import pickle as pkl

from collections import defaultdict

import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import jieba

# Local imports
import read_data
import read_cmn
import model
import utils


# TEST_WORD_ATTN = 'street'
# TEST_SENTENCE = 'the air conditioning is working'


# def read_lines(filename):
#     """Read a file and split it into lines.
#     """
#     lines = open(filename).read().strip().lower().split('\n')
#     return lines


# def read_pairs(filename):
#     """Reads lines that consist of two words, separated by a space.

#     Returns:
#         source_words: A list of the first word in each line of the file.
#         target_words: A list of the second word in each line of the file.
#     """
#     lines = read_lines(filename)
#     source_words, target_words = [], []
#     for line in lines:
#         line = line.strip()
#         if line:
#             source, target = line.split()
#             source_words.append(source)
#             target_words.append(target)
#     return source_words, target_words

# def read_lines_jieba(filename):
    

def all_alpha_or_dash(s):
    """Helper function to check whether a string is alphabetic, allowing dashes '-'.
    """
    return all(c.isalpha() or c == '-' for c in s)


def filter_lines(lines):
    """Filters lines to consist of only alphabetic characters or dashes "-".
    """
    return [line for line in lines if all_alpha_or_dash(line)]


def create_dict(pairs):
    """Creates a mapping { (source_length, target_length): [list of (source, target) pairs]
    This is used to make batches: each batch consists of two parallel tensors, one containing
    all source indexes and the other containing all corresponding target indexes.
    Within a batch, all the source words are the same length, and all the target words are
    the same length.
    """
    unique_pairs = list(set(pairs))  # Find all unique (source, target) pairs

    d = defaultdict(list)
    for (s,t) in unique_pairs:
        d[(len(s), len(t))].append((s,t))

    return d


def save_loss_plot(train_losses, val_losses, opts):
    """Saves a plot of the training and validation loss curves.
    """
    plt.figure()
    plt.plot(range(len(train_losses)), train_losses)
    plt.plot(range(len(val_losses)), val_losses)
    plt.title('BS={}, nhid={}'.format(opts.batch_size, opts.hidden_size), fontsize=20)
    plt.xlabel('Epochs', fontsize=16)
    plt.ylabel('Loss', fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(opts.checkpoint_path, 'loss_plot.pdf'))
    plt.close()


def checkpoint(encoder, decoder, idx_dict, opts):
    """Saves the current encoder and decoder models, along with idx_dict, which
    contains the char_to_index and index_to_char mappings, and the start_token
    and end_token values.
    """
    with open(os.path.join(opts.checkpoint_path, 'encoder.pt'), 'wb') as f:
        torch.save(encoder, f)

    with open(os.path.join(opts.checkpoint_path, 'decoder.pt'), 'wb') as f:
        torch.save(decoder, f)

    with open(os.path.join(opts.checkpoint_path, 'idx_dict.pkl'), 'wb') as f:
        pkl.dump(idx_dict, f)


def evaluate(tokenized_dataset_index_en_test,tokenized_dataset_index_zh_test, encoder, decoder, idx_dict, criterion, opts):
    """Evaluates the model on a held-out validation or test set.

    Arguments:
        data_dict: The validation/test word pairs, organized by source and target lengths.
        encoder: An encoder model to produce annotations for each step of the input sequence.
        decoder: A decoder model (with or without attention) to generate output tokens.
        idx_dict: Contains char-to-index and index-to-char mappings, and start & end token indexes.
        criterion: Used to compute the CrossEntropyLoss for each decoder output.
        opts: The command-line arguments.

    Returns:
        mean_loss: The average loss over all batches from data_dict.
    """

    # start_token = idx_dict['start_token']
    # end_token = idx_dict['end_token']
    # char_to_index = idx_dict['char_to_index']

    losses = []
    input_tensors = [torch.LongTensor(w) for w in tokenized_dataset_index_en_test]
    target_tensors = [torch.LongTensor(w) for w in tokenized_dataset_index_zh_test]
    
    num_tensors = len(input_tensors)
    num_batches = int(np.ceil(num_tensors / float(opts.batch_size)))

    for i in range(num_batches):
        start = i * opts.batch_size
        #print("batch size "+ str(opts.batch_size))
        adder = min(opts.batch_size, num_tensors-start)
        end = start + adder
        maxlen = len(max(input_tensors[start:end],key=len))
        maxlentarget = len(max(target_tensors[start:end],key=len))
        
        for j in range(adder):
            #print(start+j)
            sentence = input_tensors[start+j]
            out_sentence = target_tensors[start+j]
            padding_length = maxlen - sentence.size()[0]
            padding_length_out = maxlentarget - out_sentence.size()[0]

            if padding_length != 0:
                padding_array = np.full(
                    shape=padding_length,
                    fill_value=word_to_index['_'],
                    dtype=np.int
                    )
                #print(padding_array)
                new_sentence = torch.cat((sentence, torch.from_numpy(padding_array)), dim=0)
                input_tensors[start+j] = new_sentence
            if padding_length_out != 0:
                padding_array = np.full(
                    shape=padding_length_out,
                    fill_value=word_to_index_zh['_'],
                    dtype=np.int
                    )
                #print(padding_array)
                new_sentence = torch.cat((out_sentence, torch.from_numpy(padding_array)), dim=0)
                target_tensors[start+j] = new_sentence
        inputs = utils.to_var(torch.stack(input_tensors[start:end]), opts)
        targets = utils.to_var(torch.stack(target_tensors[start:end]), opts)
        BS = inputs.size(0)
        encoder_annotations, encoder_hidden = encoder(inputs)
        decoder_hidden = encoder_hidden
        start_vector = torch.ones(BS).long().unsqueeze(1) # BS x 1
        decoder_input = utils.to_var(start_vector, opts)  # BS x 1

        loss = 0.0

        seq_len = targets.size(1)  # Gets seq_len from BS x seq_len

        for i in range(seq_len):
            decoder_output, decoder_hidden, attention_weights = decoder(decoder_input, decoder_hidden, encoder_annotations)

            current_target = targets[:,i]
            loss += criterion(decoder_output, current_target)  # cross entropy between the decoder distribution and GT

            ni = F.softmax(decoder_output, dim=1).data.max(1)[1]

            decoder_input = targets[:,i].unsqueeze(1)

        loss /= float(seq_len)
        losses.append(loss.item())

    mean_loss = np.mean(losses)

    # for key in data_dict:

    #     input_strings, target_strings = zip(*data_dict[key])
    #     input_tensors = [torch.LongTensor(utils.string_to_index_list(s, char_to_index, end_token)) for s in input_strings]
    #     target_tensors = [torch.LongTensor(utils.string_to_index_list(s, char_to_index, end_token)) for s in target_strings]

    #     num_tensors = len(input_tensors)
    #     num_batches = int(np.ceil(num_tensors / float(opts.batch_size)))

    #     for i in range(num_batches):

    #         start = i * opts.batch_size
    #         end = start + opts.batch_size

    #         inputs = utils.to_var(torch.stack(input_tensors[start:end]), opts)
    #         targets = utils.to_var(torch.stack(target_tensors[start:end]), opts)

    #         # The batch size may be different in each epoch
    #         BS = inputs.size(0)

    #         encoder_annotations, encoder_hidden = encoder(inputs)

    #         # The final hidden state of the encoder becomes the initial hidden state of the decoder
    #         decoder_hidden = encoder_hidden
    #         start_vector = torch.ones(BS).long().unsqueeze(1) * start_token  # BS x 1
    #         decoder_input = utils.to_var(start_vector, opts)  # BS x 1

    #         loss = 0.0

    #         seq_len = targets.size(1)  # Gets seq_len from BS x seq_len

    #         for i in range(seq_len):
    #             decoder_output, decoder_hidden, attention_weights = decoder(decoder_input, decoder_hidden, encoder_annotations)

    #             current_target = targets[:,i]
    #             loss += criterion(decoder_output, current_target)  # cross entropy between the decoder distribution and GT

    #             ni = F.softmax(decoder_output, dim=1).data.max(1)[1]

    #             decoder_input = targets[:,i].unsqueeze(1)

    #         loss /= float(seq_len)
    #         losses.append(loss.data[0])

    # mean_loss = np.mean(losses)

    return mean_loss

def padding(length, sentences, isEN):
    count = 0
    if isEN:
        for sentence in sentences:
            padding_length = length - sentence.size()[0]
            if padding_length == 0:
                continue
            padding_array = np.full(
                shape=padding_length,
                fill_value=word_to_index['_'],
                dtype=np.int
                )
            #print(padding_array)
            new_sentence = torch.cat((sentence, torch.from_numpy(padding_array)), dim=0)
            sentences[count] = new_sentence
            count+=1
    else:

        for sentence in sentences:
            padding_length = length - sentence.size()[0]
            #print(padding_length)
            if padding_length == 0:
                continue
            padding_array = torch.from_numpy(np.full(
                shape=padding_length,
                fill_value=word_to_index_zh['_'],
                dtype=np.int
                ))
            new_sentence = torch.cat((sentence,padding_array), dim=0)
            sentences[count] = new_sentence
            count +=1

def training_loop(tokenized_dataset_index_en_train, tokenized_dataset_index_zh_train, tokenized_dataset_index_en_test, tokenized_dataset_index_zh_test,idx_dict, encoder, decoder, criterion, optimizer, opts):
    """Runs the main training loop; evaluates the model on the val set every epoch.
        * Prints training and val loss each epoch.
        * Prints qualitative translation results each epoch using TEST_SENTENCE
        * Saves an attention map for TEST_WORD_ATTN each epoch

    Arguments:
        train_dict: The training word pairs, organized by source and target lengths.
        val_dict: The validation word pairs, organized by source and target lengths.
        idx_dict: Contains char-to-index and index-to-char mappings, and start & end token indexes.
        encoder: An encoder model to produce annotations for each step of the input sequence.
        decoder: A decoder model (with or without attention) to generate output tokens.
        criterion: Used to compute the CrossEntropyLoss for each decoder output.
        optimizer: Implements a step rule to update the parameters of the encoder and decoder.
        opts: The command-line arguments.
    """

    # start_token = idx_dict['start_token']
    # end_token = idx_dict['end_token']
    # char_to_index = idx_dict['char_to_index']

    loss_log = open(os.path.join(opts.checkpoint_path, 'loss_log.txt'), 'w')

    best_val_loss = 1e6
    train_losses = []
    val_losses = []

    for epoch in range(opts.nepochs):
        print("epoch: " + str(epoch))
        optimizer.param_groups[0]['lr'] *= opts.lr_decay

        epoch_losses = []

        input_tensors = [torch.LongTensor(w) for w in tokenized_dataset_index_en_train]
        target_tensors = [torch.LongTensor(w) for w in tokenized_dataset_index_zh_train]
        num_tensors = len(input_tensors)
        #print(f'num_tensors '+str(num_tensors))
        num_batches = int(np.ceil(num_tensors / float(opts.batch_size)))

        for i in range(num_batches):
            #print("batch " + str(i))
            start = i * opts.batch_size
            end = start + opts.batch_size
            maxlen = len(max(input_tensors[start:end],key=len))
            maxlentarget = len(max(target_tensors[start:end],key=len))
            #print(input_tensors)
            # print('\n')
            #padding(maxlen, input_tensors[start:end], True)
            padding(maxlentarget, target_tensors[start:end], False)
            count = 0
            #print(input_tensors)
            for j in range(opts.batch_size):
                sentence = input_tensors[start+j]
                out_sentence = target_tensors[start+j]
                padding_length = maxlen - sentence.size()[0]
                padding_length_out = maxlentarget - out_sentence.size()[0]
                if padding_length != 0:
                    padding_array = np.full(
                        shape=padding_length,
                        fill_value=word_to_index['_'],
                        dtype=np.int
                        )
                #print(padding_array)
                    new_sentence = torch.cat((sentence, torch.from_numpy(padding_array)), dim=0)
                    input_tensors[start+j] = new_sentence
                if padding_length_out != 0:
                    padding_array = np.full(
                        shape=padding_length_out,
                        fill_value=word_to_index_zh['_'],
                        dtype=np.int
                        )
                #print(padding_array)
                    new_sentence = torch.cat((out_sentence, torch.from_numpy(padding_array)), dim=0)
                    target_tensors[start+j] = new_sentence
            #print(input_tensors)
            inputs = utils.to_var(torch.stack(input_tensors[start:end]), opts)
            targets = utils.to_var(torch.stack(target_tensors[start:end]), opts)

            BS = inputs.size(0)
            encoder_annotations, encoder_hidden = encoder(inputs)
            decoder_hidden = encoder_hidden

            start_vector = torch.ones(BS).long().unsqueeze(1)  # BS x 1 --> 16x1  CHECKED
            decoder_input = utils.to_var(start_vector, opts)  # BS x 1 --> 16x1  CHECKED
            loss = 0.0

            seq_len = targets.size(1)  # Gets seq_len from BS x seq_len

            use_teacher_forcing = np.random.rand() < opts.teacher_forcing_ratio
            for i in range(seq_len):
                decoder_output, decoder_hidden, attention_weights = decoder(decoder_input, decoder_hidden, encoder_annotations)

                current_target = targets[:,i]
                loss += criterion(decoder_output, current_target)  # cross entropy between the decoder distribution and GT
                ni = F.softmax(decoder_output, dim=1).data.max(1)[1]

                if use_teacher_forcing:
                    # With teacher forcing, use the ground-truth token to condition the next step
                    decoder_input = targets[:,i].unsqueeze(1)
                else:
                    # Without teacher forcing, use the model's own predictions to condition the next step
                    decoder_input = utils.to_var(ni.unsqueeze(1), opts)
            #print(loss.item())
            #loss /= float(seq_len)
            epoch_losses.append(loss.item())

            # Zero gradients
            optimizer.zero_grad()

            # Compute gradients
            loss.backward()

            # Update the parameters of the encoder and decoder
            optimizer.step()

        train_loss = np.mean(epoch_losses)
        #print(len(tokenized_dataset_index_en_test))
        val_loss = evaluate(tokenized_dataset_index_en_test,tokenized_dataset_index_zh_test, encoder, decoder, idx_dict, criterion, opts)
        if val_loss < best_val_loss:
            checkpoint(encoder, decoder, idx_dict, opts)

        #if not opts.no_attention:
            # Save attention maps for the fixed word TEST_WORD_ATTN throughout training
            # utils.visualize_attention(TEST_WORD_ATTN,
            #                           encoder,
            #                           decoder,
            #                           idx_dict,
            #                           opts,
            #                           save=os.path.join(opts.checkpoint_path, 'train_attns/attn-epoch-{}.png'.format(epoch)))
        #print(word_to_index)
        gen_string = utils.translate("I am your dad", word_to_index,index_to_zh, encoder, decoder, idx_dict, opts)
        #gen_string = utils.translate_sentence("I am your dad", encoder, decoder, idx_dict, opts)
        print("Epoch: {:3d} | Train loss: {:.3f} | Val loss: {:.3f} | Gen: {:20s}".format(epoch, train_loss, val_loss, gen_string))

        loss_log.write('{} {} {}\n'.format(epoch, train_loss, val_loss))
        loss_log.flush()

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        save_loss_plot(train_losses, val_losses, opts)

        # if val_loss < best_val_loss:
        #     checkpoint(encoder, decoder, idx_dict, opts)

        # for key in train_dict:

        #     input_strings, target_strings = zip(*train_dict[key])
        #     input_tensors = [torch.LongTensor(utils.string_to_index_list(s, char_to_index, end_token)) for s in input_strings]
        #     target_tensors = [torch.LongTensor(utils.string_to_index_list(s, char_to_index, end_token)) for s in target_strings]

        #     num_tensors = len(input_tensors)
        #     num_batches = int(np.ceil(num_tensors / float(opts.batch_size)))

        #     for i in range(num_batches):

        #         start = i * opts.batch_size
        #         end = start + opts.batch_size

        #         inputs = utils.to_var(torch.stack(input_tensors[start:end]), opts)
        #         targets = utils.to_var(torch.stack(target_tensors[start:end]), opts)

        #         # The batch size may be different in each epoch
        #         BS = inputs.size(0)

        #         encoder_annotations, encoder_hidden = encoder(inputs)

        #         # The last hidden state of the encoder becomes the first hidden state of the decoder
        #         decoder_hidden = encoder_hidden

        #         start_vector = torch.ones(BS).long().unsqueeze(1) * start_token  # BS x 1 --> 16x1  CHECKED
        #         decoder_input = utils.to_var(start_vector, opts)  # BS x 1 --> 16x1  CHECKED

        #         loss = 0.0

        #         seq_len = targets.size(1)  # Gets seq_len from BS x seq_len

        #         use_teacher_forcing = np.random.rand() < opts.teacher_forcing_ratio

        #         for i in range(seq_len):
        #             decoder_output, decoder_hidden, attention_weights = decoder(decoder_input, decoder_hidden, encoder_annotations)

        #             current_target = targets[:,i]
        #             loss += criterion(decoder_output, current_target)  # cross entropy between the decoder distribution and GT
        #             ni = F.softmax(decoder_output, dim=1).data.max(1)[1]

        #             if use_teacher_forcing:
        #                 # With teacher forcing, use the ground-truth token to condition the next step
        #                 decoder_input = targets[:,i].unsqueeze(1)
        #             else:
        #                 # Without teacher forcing, use the model's own predictions to condition the next step
        #                 decoder_input = utils.to_var(ni.unsqueeze(1), opts)

        #         loss /= float(seq_len)
        #         epoch_losses.append(loss.data[0])

        #         # Zero gradients
        #         optimizer.zero_grad()

        #         # Compute gradients
        #         loss.backward()

        #         # Update the parameters of the encoder and decoder
        #         optimizer.step()

        # train_loss = np.mean(epoch_losses)
        # val_loss = evaluate(val_dict, encoder, decoder, idx_dict, criterion, opts)

        # if val_loss < best_val_loss:
        #     checkpoint(encoder, decoder, idx_dict, opts)

        # if not opts.no_attention:
        #     # Save attention maps for the fixed word TEST_WORD_ATTN throughout training
        #     utils.visualize_attention(TEST_WORD_ATTN,
        #                               encoder,
        #                               decoder,
        #                               idx_dict,
        #                               opts,
        #                               save=os.path.join(opts.checkpoint_path, 'train_attns/attn-epoch-{}.png'.format(epoch)))

        # gen_string = utils.translate_sentence(TEST_SENTENCE, encoder, decoder, idx_dict, opts)
        # print("Epoch: {:3d} | Train loss: {:.3f} | Val loss: {:.3f} | Gen: {:20s}".format(epoch, train_loss, val_loss, gen_string))

        # loss_log.write('{} {} {}\n'.format(epoch, train_loss, val_loss))
        # loss_log.flush()

        # train_losses.append(train_loss)
        # val_losses.append(val_loss)

        # save_loss_plot(train_losses, val_losses, opts)


def print_data_stats(line_pairs, vocab_size, idx_dict):
    """Prints example word pairs, the number of data points, and the vocabulary.
    """
    print('=' * 80)
    print('Data Stats'.center(80))
    print('-' * 80)
    for pair in line_pairs[:5]:
        print(pair)
    print('Num unique word pairs: {}'.format(len(line_pairs)))
    print('Vocabulary: {}'.format(idx_dict['char_to_index'].keys()))
    print('Vocab size: {}'.format(vocab_size))
    print('=' * 80)


def main(opts):
    
    
    # Split the line pairs into an 80% train and 20% val split
    num_lines = len(tokenized_dataset)
    num_train = int(0.8 * num_lines)
    train_en, train_cn= tokenized_dataset[:num_train], tokenized_dataset_zh[:num_train]
    target_en, target_cn = tokenized_dataset[num_train:], tokenized_dataset_zh[num_train:]
    train_pairs, val_pairs = tokenized_dataset[:num_train], tokenized_dataset[num_train:]

    # Group the data by the lengths of the source and target words, to form batches
    #train_dict = create_dict(train_pairs)

    # val_dict = create_dict(val_pairs)

    tokenized_dataset_index_en_train, tokenized_dataset_index_en_test = tokenized_dataset_index_en[:num_train],tokenized_dataset_index_en[num_train:]
    tokenized_dataset_index_zh_train, tokenized_dataset_index_zh_test = tokenized_dataset_index_zh[:num_train],tokenized_dataset_index_zh[num_train:]
    #print(opts.mps)
    ##########################################################################
    ### Setup: Create Encoder, Decoder, Learning Criterion, and Optimizers ###
    ##########################################################################
    encoder = model.GRUEncoder(vocab_size=len(vocabulary_zh), hidden_size=opts.hidden_size, opts=opts)

    if opts.no_attention:
        decoder = model.NoAttentionDecoder(vocab_size=len(vocabulary_zh), hidden_size=opts.hidden_size)
    else:
        decoder = model.AttentionDecoder(vocab_size=len(vocabulary_zh), hidden_size=opts.hidden_size)

    if opts.cuda:
        encoder.cuda()
        decoder.cuda()
        print("Moved models to GPU!")
    elif opts.mps:
        encoder.to("mps")
        decoder.to("mps")
        print("Moved models APPLE!")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=opts.learning_rate)

    try:
        training_loop(tokenized_dataset_index_en_train, tokenized_dataset_index_zh_train, tokenized_dataset_index_en_test, tokenized_dataset_index_zh_test, idx_dict, encoder, decoder, criterion, optimizer, opts)
    except KeyboardInterrupt:
        print('Exiting early from training.')


def create_parser():
    """Creates a parser for command-line arguments.
    """
    parser = argparse.ArgumentParser()

    # Training hyper-parameters
    parser.add_argument('--nepochs', type=int, default=100,
                        help='The max number of epochs.')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='The number of examples in a batch.')
    parser.add_argument('--learning_rate', type=float, default=0.01,
                        help='The learning rate (default 0.01)')
    parser.add_argument('--lr_decay', type=float, default=0.99,
                        help='Set the learning rate decay factor.')
    parser.add_argument('--hidden_size', type=int, default=10,
                        help='The size of the GRU hidden state.')
    parser.add_argument('--teacher_forcing_ratio', type=float, default=0.5,
                        help='The proportion of the time teacher forcing is used.')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                        help='Set the directry to store the best model checkpoints.')
    parser.add_argument('--no_attention', action='store_true', default=False,
                        help='Use the NoAttentionDecoder model.')
    parser.add_argument('--cuda', action='store_true', default=False,
                        help='Choose whether to use GPU.')
    parser.add_argument('--mps', action='store_true', default=False, 
                        help='choose whether to use mps')

    return parser


def print_opts(opts):
    """Prints the values of all command-line arguments.
    """
    print('=' * 80)
    print('Opts'.center(80))
    print('-' * 80)
    for key in opts.__dict__:
        print('{:>30}: {:<30}'.format(key, opts.__dict__[key]).center(80))
    print('=' * 80)


if __name__ == '__main__':

    parser = create_parser()
    opts = parser.parse_args()

    print_opts(opts)

    model_name = 'h{}-bs{}'.format(opts.hidden_size, opts.batch_size)
    opts.checkpoint_path = os.path.join(opts.checkpoint_dir, model_name)

    #utils.create_dir_if_not_exists(opts.checkpoint_path)
    #utils.create_dir_if_not_exists(os.path.join(opts.checkpoint_path, 'train_attns'))
    vocabulary_zh, vocabulary_en, tokenized_dataset,tokenized_dataset_zh = read_cmn.read_data_from_path('cmn.txt')
    vocabulary_en.append('_')
    # create 2 mappings, 1 from word_to_index, 1 from index_to_word
    word_to_index = { word: index for (index, word) in enumerate(sorted(list(vocabulary_en))) }
    index_to_word = { index: word for (word, index) in word_to_index.items() }

    vocabulary_zh.append('_')
    word_to_index_zh =  { word: index for (index, word) in enumerate(sorted(list(vocabulary_zh))) }
    index_to_zh = { index: word for (word, index) in word_to_index_zh.items() }
    tokenized_dataset_index_en, tokenized_dataset_index_zh = read_cmn.sentencetoindex(tokenized_dataset,word_to_index, False),read_cmn.sentencetoindex(tokenized_dataset_zh,word_to_index_zh, True)
    #print(tokenized_dataset_index_en)
    idx_dict = { 'char_to_index': word_to_index,
                'index_to_char': index_to_word,
                'word_to_index_zh':word_to_index_zh,
                'index_to_zh': index_to_zh}
    main(opts)
