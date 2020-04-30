
import torch
import torch.nn as nn

import data
import logging
import splitcross

logger = logging.getLogger('train_lm')

class RNNModelWrapper(nn.Module):
    """This module will be saved/loaded.

    Abstracts additional network mechanisms (for handling multi-tasking) apart from
    the base RNN.
    """
    def __init__(self, rnn, do_binary_classification=False):
        super().__init__()

        self.rnn = rnn
        vocab = rnn.vocab

        self.criterion = self._mk_criterion(self.rnn.ninp, vocab.size, vocab.index(data.PAD))
        if do_binary_classification:
            self.bc_linear = nn.Linear(self.rnn.decoder.weight.size(1), 1)
            self.bc_criterion = nn.BCEWithLogitsLoss(reduction='none')

    @property
    def vocab(self):
        return self.rnn.vocab

    def _mk_criterion(self, emsize, vocab_size, pad_id):
        splits = []
        if vocab_size > 500000:
            # One Billion
            # This produces fairly even matrix mults for the buckets:
            # 0: 11723136, 1: 10854630, 2: 11270961, 3: 11219422
            splits = [4200, 35000, 180000]
        elif vocab_size > 75000:
            # WikiText-103
            splits = [2800, 20000, 76000]
        logger.info('Using {}'.format(splits))

        return splitcross.SplitCrossEntropyLoss(
            emsize, splits=splits, verbose=False, ignore_index=pad_id)

    def loss(self, output, targets, reduce=True):
        d = self.rnn.decoder
        return self.criterion(d.weight, d.bias, output, targets, reduce=reduce)

    def bc_loss(self, output, targets, reduce=True):
        logit = self.bc_linear(output).squeeze(1)
        loss = self.bc_criterion(logit, targets)
        if reduce == True:
            return loss.mean()
        else:
            return loss

    def entropy(self, hiddens):
        d = self.rnn.decoder
        return self.criterion.entropy(d.weight, d.bias, hiddens)

    def log_dist(self, hidden):
        d = self.rnn.decoder
        return self.criterion.log_dist(d.weight, d.bias, hidden.unsqueeze(0))[0]

class PartiallySharedMultiTaskModel(RNNModelWrapper):
    def __init__(self, rnn, tag_vocab):
        super().__init__(rnn)

        self.tag_vocab = tag_vocab
        hidden_size = self.rnn.ninp if self.rnn.tie_weights else self.rnn.nhid
        self.tag_decoder = nn.Linear(hidden_size, self.tag_vocab.size)

        self.init_weights()

        # TODO: maybe this ninp should also be conditioned on tie_weights value.
        self.tag_criterion = self._mk_criterion(self.rnn.ninp, tag_vocab.size, tag_vocab.index(data.PAD))

    def init_weights(self):
        initrange = 0.1
        self.tag_decoder.bias.data.fill_(0)
        self.tag_decoder.weight.data.uniform_(-initrange, initrange)

    def tag_loss(self, output, targets, reduce=True):
        return self.tag_criterion(
            self.tag_decoder.weight, self.tag_decoder.bias, output, targets, reduce=reduce)


class FullySharedMultiTaskModel(RNNModelWrapper):
    """Fully shared network does not introduce additional parameters.

    Just use the same network to predict tags.
    """

    def __init__(self, rnn):
        super().__init__(rnn)

    def tag_loss(self, output, targets, reduce=True):
        return self.loss(output, targets, reduce)
