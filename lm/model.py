import torch

import torch.nn as nn

from embed_regularize import embedded_dropout
from locked_dropout import LockedDropout
from weight_drop import WeightDrop


class RNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, rnn_type, vocab, ninp, nhid, nlayers, dropout=0.5, dropouth=0.5, dropouti=0.5, dropoute=0.1, wdrop=0, tie_weights=False, locked=True, padding_idx=None):
        super(RNNModel, self).__init__()

        self.lockdrop = LockedDropout()
        self.drop = self._lock_drop if locked else self._nonlock_drop

        self.encoder = nn.Embedding(vocab.size, ninp, padding_idx=padding_idx)
        assert rnn_type in ['LSTM', 'QRNN', 'GRU'], 'RNN type is not supported'
        if rnn_type == 'LSTM':
            self.rnns = [torch.nn.LSTM(ninp if l == 0 else nhid, nhid if l != nlayers - 1 else (ninp if tie_weights else nhid), 1, dropout=0, batch_first=True) for l in range(nlayers)]
            if wdrop:
                self.rnns = [WeightDrop(rnn, ['weight_hh_l0'], dropout=wdrop) for rnn in self.rnns]
        if rnn_type == 'GRU':
            self.rnns = [torch.nn.GRU(ninp if l == 0 else nhid, nhid if l != nlayers - 1 else ninp, 1, dropout=0, batch_first=True) for l in range(nlayers)]
            if wdrop:
                self.rnns = [WeightDrop(rnn, ['weight_hh_l0'], dropout=wdrop) for rnn in self.rnns]
        elif rnn_type == 'QRNN':
            from torchqrnn import QRNNLayer
            self.rnns = [QRNNLayer(input_size=ninp if l == 0 else nhid, hidden_size=nhid if l != nlayers - 1 else (ninp if tie_weights else nhid), save_prev_x=True, zoneout=0, window=2 if l == 0 else 1, output_gate=True) for l in range(nlayers)]
            for rnn in self.rnns:
                rnn.linear = WeightDrop(rnn.linear, ['weight'], dropout=wdrop)

        self.rnns = torch.nn.ModuleList(self.rnns)
        self.decoder = nn.Linear(nhid, vocab.size)

        # Optionally tie weights as in:
        # "Using the Output Embedding to Improve Language Models" (Press & Wolf 2016)
        # https://arxiv.org/abs/1608.05859
        # and
        # "Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling" (Inan et al. 2016)
        # https://arxiv.org/abs/1611.01462
        if tie_weights:
            #if nhid != ninp:
            #    raise ValueError('When using the tied flag, nhid must be equal to emsize')
            self.decoder.weight = self.encoder.weight

        self.init_weights()

        self.rnn_type = rnn_type
        self.vocab = vocab
        self.ninp = ninp
        self.nhid = nhid
        self.nlayers = nlayers
        self.dropout = dropout
        self.dropouti = dropouti
        self.dropouth = dropouth
        self.dropoute = dropoute
        self.tie_weights = tie_weights

    def _lock_drop(self, w, ratio):
        return self.lockdrop(w, ratio)

    def _nonlock_drop(self, w, ratio):
        d = nn.Dropout(ratio)
        d(w)
        return w

    def reset(self):
        if self.rnn_type == 'QRNN': [r.reset() for r in self.rnns]

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, input, hidden=None, return_h=False, input_lengths=None):
        """
        Apply a multi-layer RNN to an input sequence.

        If input_lengths is given, variable-length inputs are assumed. The input to
        RNN is packed in this case. This method does not sort the input by length
        internally, so input_var should be sorted beforehand.
        """
        emb = embedded_dropout(
            self.encoder, input, dropout=self.dropoute if self.training else 0)

        #emb = self.lockdrop(emb, self.dropouti)
        emb = self.drop(emb, self.dropouti)
        raw_output = emb

        if not hidden:
            hidden = self.init_hidden(input.size(0)) # batch_first

        new_hidden = []
        #raw_output, hidden = self.rnn(emb, hidden)
        raw_outputs = []
        outputs = []
        # enumerate layers
        for l, rnn in enumerate(self.rnns):
            if input_lengths is not None:
                raw_output = nn.utils.rnn.pack_padded_sequence(
                    raw_output, input_lengths, batch_first=True)
            raw_output, new_h = rnn(raw_output, hidden[l])
            new_hidden.append(new_h)
            if input_lengths is not None:
                raw_output, _ = nn.utils.rnn.pad_packed_sequence(
                    raw_output, batch_first=True)
            raw_outputs.append(raw_output)
            if l != self.nlayers - 1:
                # self.hdrop(raw_output)
                raw_output = self.drop(raw_output, self.dropouth)
                # raw_output = self.lockdrop(raw_output, self.dropouth)
                outputs.append(raw_output)
        hidden = new_hidden

        # output = self.lockdrop(raw_output, self.dropout)
        output = self.drop(raw_output, self.dropout)
        outputs.append(output)

        result = output.contiguous().view(
            output.size(0)*output.size(1), output.size(2))
        if return_h:
            return result, hidden, raw_outputs, outputs
        return result, hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        if self.rnn_type == 'LSTM':
            return [(weight.new(1, bsz, self.nhid if l != self.nlayers - 1 else (self.ninp if self.tie_weights else self.nhid)).zero_(),
                     weight.new(1, bsz, self.nhid if l != self.nlayers - 1 else (self.ninp if self.tie_weights else self.nhid)).zero_())
                    for l in range(self.nlayers)]
        elif self.rnn_type == 'QRNN' or self.rnn_type == 'GRU':
            return [weight.new(1, bsz, self.nhid if l != self.nlayers - 1 else (self.ninp if self.tie_weights else self.nhid)).zero_()
                    for l in range(self.nlayers)]
