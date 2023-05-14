import torch


import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse

# Local imports
import collections

class MyGRUCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(MyGRUCell, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.wir=nn.Linear(input_size, hidden_size,bias=False)
        self.whr=nn.Linear(hidden_size, hidden_size)
        self.wiz=nn.Linear(input_size, hidden_size,bias=False)
        self.whz=nn.Linear(hidden_size, hidden_size)
        self.win=nn.Linear(input_size, hidden_size,bias=False)
        self.whn=nn.Linear(hidden_size, hidden_size)


    def forward(self, x, h_prev):
        """Forward pass of the GRU computation for one time step.

        Arguments
            x: batch_size x input_size
            h_prev: batch_size x hidden_size

        Returns:
            h_new: batch_size x hidden_size
        """

        z = F.sigmoid(self.wiz(x)+self.whz(h_prev))#update gate vector
        r = F.sigmoid(self.wir(x)+self.whr(h_prev))#reset gate vector
        g = F.tanh(self.win(x)+r*self.whn(h_prev))#new information
        h_new = (1-z)*g+z*h_prev#output vector
        return h_new


class GRUEncoder(nn.Module):
    def __init__(self, vocab_size, hidden_size, opts):
        super(GRUEncoder, self).__init__()

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.opts = opts
        #generate embeddings   
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.gru = MyGRUCell(hidden_size, hidden_size).to("mps")

    def forward(self, inputs):
        """Forward pass of the encoder RNN.

        Arguments:
            inputs: Input token indexes across a batch for all time steps in the sequence. (batch_size x seq_len)

        Returns:
            annotations: The hidden states computed at each step of the input sequence. (batch_size x seq_len x hidden_size)
            hidden: The final hidden state of the encoder, for each sequence in a batch. (batch_size x hidden_size)
        """

        batch_size, seq_len = inputs.size()
        hidden = self.init_hidden(batch_size)

        encoded = self.embedding(inputs)  # batch_size x seq_len x hidden_size
        annotations = []

        for i in range(seq_len):
            x = encoded[:,i,:]  # Get the current time step, across the whole batch
            hidden = self.gru(x, hidden)
            annotations.append(hidden)

        annotations = torch.stack(annotations, dim=1)
        return annotations, hidden

    def init_hidden(self, bs):
        """Creates a tensor of zeros to represent the initial hidden states
        of a batch of sequences.

        Arguments:
            bs: The batch size for the initial hidden state.

        Returns:
            hidden: An initial hidden state of all zeros. (batch_size x hidden_size)
        """
        return torch.zeros(bs, self.hidden_size).to('mps')


class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()

        self.hidden_size = hidden_size

        # ------------
        # FILL THIS IN
        # ------------

        # Create a two layer fully-connected network. Hint: Use nn.Sequential
        # hidden_size*2 --> hidden_size, ReLU, hidden_size --> 1

        # self.attention_network = ...

        self.attention_network=nn.Sequential(collections.OrderedDict([
            ('W1',nn.Linear(2*hidden_size,hidden_size)),
            ('ReLU',nn.ReLU()),
            ('W2',nn.Linear(hidden_size,1))
        ]))

        self.softmax = nn.Softmax(dim=1)

    def forward(self, hidden, annotations):
        """The forward pass of the attention mechanism.

        Arguments:
            hidden: The current decoder hidden state. (batch_size x hidden_size)
            annotations: The encoder hidden states for each step of the input sequence. (batch_size x seq_len x hidden_size)

        Returns:p
            output: Normalized attention weights for each encoder hidden state. (batch_size x seq_len x 1)

            The output must be a softmax weighting over the seq_len annotations.
        """

#         >>> a =torch.tensor([[1,2],[3,4],[5,6]])
# >>> a =torch.Tensor([[1,2],[3,4],[5,6]])
# >>> a.size()
# torch.Size([3, 2])
# >>> a.unsqueeze(1)
# tensor([[[1., 2.]],

#         [[3., 4.]],

#         [[5., 6.]]])
# >>> b = a.unsqueeze(1)
# >>> b.size()
# torch.Size([3, 1, 2])
# >>> c =torch.tensor([[[1,2],[3,1]],[[3,4],[1,4]],[[7,8],[0,0]]])
# >>> c.size()
# torch.Size([3, 2, 2])
# >>> b.expand_as(c)
# tensor([[[1., 2.],
#          [1., 2.]],

#         [[3., 4.],
#          [3., 4.]],

#         [[5., 6.],
#          [5., 6.]]])
        # 在sequence length那一维度上 复制所有的batch * hidden的平面,meaning 对于encoder种所有的step进行相同的操作
        batch_size, seq_len, hid_size = annotations.size()
        expanded_hidden = hidden.unsqueeze(1).expand_as(annotations)

       #annotations=annotations.view(-1,hid_size).unsqueeze(0)
        # 将上一个hidden state和这个annotation 在concat在hidden size那一维上，因此
        # annotations 变成了batch_size x seq_len x 2 *hidden_size
        concat=torch.cat((expanded_hidden, annotations),dim=2)

        #将所有batch中词进行平铺，得到 (batch_size * seq_len) x （2 * hidden_size）的tensor
        reshaped_for_attention_net=concat.view(-1,2*hid_size)

        # 第一步先得到一个(batch_size * seq_len) x （hidden_size）的matrix
        # nn.Linear(2*hidden_size,hidden_size)
        # 执行 relu
        # 得到一个 (batch_size * seq_len) x 1 的matrix, 即为attention_net_output
        # nn.Linear(batch_size * seq_len)
        attention_net_output = self.attention_network(reshaped_for_attention_net)
        # matrix扩大为三维，即对于每一个batch每一个位置都有一个数, 一遍进行 softmax
        unnormalized_attention=attention_net_output.view(batch_size,seq_len,1)
        return self.softmax(unnormalized_attention)

class AttentionDecoder(nn.Module):
    def __init__(self, vocab_size, hidden_size):
        super(AttentionDecoder, self).__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(vocab_size, hidden_size)

        self.rnn = MyGRUCell(input_size=hidden_size*2, hidden_size=hidden_size)
        self.attention = Attention(hidden_size=hidden_size)
        self.out = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, h_prev, annotations):
        """Forward pass of the attention-based decoder RNN.

        Arguments:
            x: Input token indexes across a batch for a single time step. (batch_size x 1)
            h_prev: The hidden states from the previous step, across a batch. (batch_size x hidden_size)
            annotations: The encoder hidden states for each step of the input.
                         sequence. (batch_size x seq_len x hidden_size)

        Returns:
            output: Un-normalized scores for each token in the vocabulary, across a batch. (batch_size x vocab_size)
            h_new: The new hidden states, across a batch. (batch_size x hidden_size)
            attention_weights: The weights applied to the encoder annotations, across a batch. (batch_size x encoder_seq_len x 1)
        """
        embed = self.embedding(x)    # batch_size x 1 x hidden_size
        embed = embed.squeeze(1)     # batch_size x hidden_size

        attention_weights = self.attention(hidden=h_prev, annotations=annotations)
        context = torch.sum(attention_weights*annotations,1)

        embed = embed.expand_as(context)
        embed_and_context =torch.cat([embed, context],dim=1)            
        h_new = self.rnn(embed_and_context, h_prev)
        output = self.out(h_new)
        return output, h_new, attention_weights
