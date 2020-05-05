
import logging
import numpy as np
import math
import time
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ExponentialLR, ReduceLROnPlateau

import model_wrapper
import utils

try:
    from apex import amp
except ModuleNotFoundError:
    pass

logger = logging.getLogger('train_lm')

class TrainConfig(object):
    def __init__(self, lr, wdecay, when_lr, optimizer, nonmono, clip=None, alpha=None, beta=None, start_lr_decay=None, plateau_lr_decay=False, start_plateau_lr_decay=10, plateau_patience=False, lr_decay_gamma=None, non_average=False, negative_loss_alpha=1.0, amp=False, do_sample_average_before=False, within_batch_neg_prob=-1, normalize_negative=False, sample_average_jointly=False):
        self.lr = lr
        self.wdecay = wdecay
        self.when_lr = when_lr
        self.optimizer = optimizer
        self.nonmono = nonmono
        self.clip = clip
        self.alpha = alpha
        self.beta = beta
        self.start_lr_decay = start_lr_decay
        self.plateau_lr_decay = plateau_lr_decay
        self.start_plateau_lr_decay = start_plateau_lr_decay
        self.plateau_patience = plateau_patience
        self.lr_decay_gamma = lr_decay_gamma
        self.non_average = non_average
        self.negative_loss_alpha = negative_loss_alpha
        self.amp = amp
        self.do_sample_average_before = do_sample_average_before
        # this is only used with token neg loss calculators.
        self.within_batch_neg_prob = within_batch_neg_prob
        self.normalize_negative = normalize_negative
        self.sample_average_jointly = sample_average_jointly


class LMTrainer(object):
    def __init__(self, model, validator, config):
        assert isinstance(model, model_wrapper.RNNModelWrapper)
        self.model = model

        self.validator = validator
        self.config = config

        self.params = list(model.parameters())

        # optimizer will be preserved in this class, not given externally.
        if config.optimizer == 'sgd':
            self.optimizer = torch.optim.SGD(self.params, lr=config.lr, weight_decay=config.wdecay)
        elif config.optimizer == 'adam':
            self.optimizer = torch.optim.Add(self.params, lr=config.lr, weight_decay=config.wdecay)
        else:
            raise ValueError('--config should be either sgd or adam.')

        if config.amp:
            logger.info('Automatic mixed-precision training (AMP): ON')
            self.model, self.optimizer = amp.initialize(
                self.model, self.optimizer, opt_level='O1')

        total_params = sum(x.size()[0] * x.size()[1] if len(x.size()) > 1 else x.size()[0] for x in self.params if x.size())
        logger.info('Model initialization done.')
        logger.info('Model total parameters: {}'.format(total_params))

        self.batches_processed = 0

        self.scheduler = None
        if self.config.plateau_lr_decay:
            self.scheduler = ReduceLROnPlateau(self.optimizer,
                                               factor=self.config.lr_decay_gamma,
                                               patience=self.config.plateau_patience,
                                               verbose=True)

    def may_setup_lr_decay(self, epoch):
        if (self.config.start_lr_decay >= 0
            and epoch >= self.config.start_lr_decay
            and self.scheduler is None):
            self.scheduler = ExponentialLR(self.optimizer, self.config.lr_decay_gamma)

    def run_epoch(self, batch_gen, val_gen, log_interval, validate_batches, epoch):
        self.may_setup_lr_decay(epoch)

        total_loss = 0
        start_time = time.time()
        hidden = self.model.rnn.init_hidden(batch_gen.batch_size)

        total_batchs =  batch_gen.total_batches()
        for batch_i, (batch, is_tag, loss_alpha) in enumerate(batch_gen()):
            # This batch might be a word batch, or a tag batch, distinguished by is_tag.
            # loss_alpha is a multiplier (lambda) for multi-task leraning loss.

            self.batches_processed += 1
            raw_loss, hidden = self.run_batch(batch, hidden, is_tag, loss_alpha)
            total_loss += raw_loss.item()

            if batch_i % log_interval == 0 and batch_i > 0 or \
               total_batchs < log_interval and batch_i == total_batchs:
                # print the status at the last batch when # total_batches is small

                cur_loss = total_loss / log_interval

                elapsed = time.time() - start_time
                logger.info('| epoch {:3d} | {:5d}/{:5d} batches | lr {:05.5f} | ms/batch {:5.2f} | '
                            'loss {:5.2f} | ppl {:8.2f} | bpc {:8.3f}'.format(
                        epoch, batch_i, total_batchs, self.optimizer.param_groups[0]['lr'],
                elapsed * 1000 / log_interval, cur_loss, math.exp(cur_loss), cur_loss / math.log(2)))
                total_loss = 0
                start_time = time.time()

            if validate_batches > 0 and self.batches_processed % validate_batches == 0:
                self.validate_and_save(val_gen, epoch)
                self.batches_processed = 0

        if validate_batches == 0:
            # validation step after each epoch
            self.validate_and_save(val_gen, epoch)

        if self.config.start_lr_decay >= 0 and self.scheduler is not None:
            self.scheduler.step()

    def run_batch(self, batch, hidden, is_tag, loss_alpha):
        orig_lr = self.optimizer.param_groups[0]['lr']
        self._pre_batch(batch)

        self.model.rnn.train()
        self.optimizer.zero_grad()

        raw_loss, loss, hidden = self._batch_loss(batch, is_tag, hidden)
        if loss_alpha != 1.0: loss *= loss_alpha

        if self.config.amp:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()

        if self.config.clip:
            nn.utils.clip_grad_norm_(self.params, self.config.clip)
        self.optimizer.step()

        self.optimizer.param_groups[0]['lr'] = orig_lr

        return raw_loss, hidden

    def validate_and_save(self, val_gen, epoch):
        validator, conf = self.validator, self.config
        val_loss = validator.validate(
            self.model, self.optimizer, val_gen, epoch)

        if conf.optimizer == 'sgd' and not conf.non_average and \
           't0' not in self.optimizer.param_groups[0] and \
           (len(validator.best_val_loss) > conf.nonmono and \
            val_loss > min(validator.best_val_loss[:-conf.nonmono])):

            self.optimizer = torch.optim.ASGD(self.model.parameters(), lr=conf.lr, t0=0, lambd=0., weight_decay=conf.wdecay)

        if self.config.plateau_lr_decay and epoch >= self.config.start_plateau_lr_decay:
            self.scheduler.step(val_loss)

        if epoch in conf.when_lr:
            logger.info('Saving model before learning rate decreased')
            self.validator.save_epoch(self.model, self.optimizer, epoch)
            logger.info('Dividing learning rate by 10')
            self.optimizer.param_groups[0]['lr'] /= 10.

    def _pre_batch(self, batch):
        """Process before doing a batch"""
        pass

    def _batch_loss(self, batch, is_tag, hidden=None):
        """Return raw_loss and loss. loss is a total of raw_loss and additional losses
        for reguralization."""
        pass


class SentenceLMTrainer(LMTrainer):

    def _batch_loss(self, batch, is_tag, hidden=None):
        return self._sent_batch_loss(batch, is_tag)

    def _sent_batch_loss(self, batch, is_tag):
        device = next(self.model.parameters()).device
        sources, lengths, targets = batch
        sources = sources.to(device)
        lengths = lengths.to(device)
        targets = targets.to(device)

        output, hidden, rnn_hs, dropped_rnn_hs = self.model.rnn(
            sources, input_lengths = lengths, return_h = True)

        if is_tag:
            raw_loss = self.model.tag_loss(output, targets)
        else:
            raw_loss = self.model.loss(output, targets)
        loss = raw_loss
        loss += self._reguralize(dropped_rnn_hs[-1], lengths.sum())

        # Returnning hidden here is not meaningful. Just None should be ok.
        return raw_loss, loss, hidden

    def _reguralize(self, last_rnn_h, ntokens):
        alpha, beta = self.config.alpha, self.config.beta
        reg = 0.0
        if alpha:
            non_zero = last_rnn_h.size(0) * ntokens * last_rnn_h.size(2)
            reg += alpha * last_rnn_h.pow(2).sum() / non_zero.item()
        if beta:
            diffs = last_rnn_h[:,1:] - last_rnn_h[:,:-1]
            bs = last_rnn_h.size(0)
            non_zero = bs * (ntokens - bs) * last_rnn_h.size(2)
            reg += diffs.pow(2).sum() / non_zero.item()
        return reg


class SentenceNegLossCalculator(object):
    def __init__(self):
        pass

    def _to_sent_probs(self, word_probs, lengths):
        sent_probs = word_probs.new_zeros(lengths.size(0))
        offset = 0
        for i, l in enumerate(lengths):
            sent_probs[i] = word_probs[offset:offset+l].sum()
            offset += l
        assert offset == lengths.sum()
        return sent_probs

    def _last_token_indexes(self, lengths, sent_len):
        last_idxes = torch.zeros_like(lengths)
        last_idxes[1:] = sent_len  # padded sequence length
        last_idxes = last_idxes.cumsum(0) - 1 + lengths
        return last_idxes

class SentenceMarginLossCalculator(SentenceNegLossCalculator):
    def __init__(self, margin=1.0):
        self.margin_criterion = torch.nn.MultiMarginLoss(margin=margin)

    def loss(self, model, batch):
        device = next(model.parameters()).device
        gold_srcs, lengths, gold_tgts, wrong_srcs, wrong_tgts = batch
        gold_srcs = gold_srcs.to(device)
        lengths = lengths.to(device)
        gold_tgts = gold_tgts.to(device)
        wrong_srcs = wrong_srcs.to(device)
        wrong_tgts = wrong_tgts.to(device)

        gold_outputs, _ = model.rnn(gold_srcs, input_lengths = lengths, return_h = False)
        wrong_outputs, _ = model.rnn(wrong_srcs, input_lengths = lengths, return_h = False)

        gold_probs = model.loss(gold_outputs, gold_tgts, reduce=False) * -1.0 # (total_length, 1)
        wrong_probs = model.loss(wrong_outputs, wrong_tgts, reduce=False) * -1.0

        # We first decompose *_probs into sentences using `lengths`.
        gold_sent_probs = self._to_sent_probs(gold_probs, lengths).unsqueeze(1)
        wrong_sent_probs = self._to_sent_probs(wrong_probs, lengths).unsqueeze(1)
        probs = torch.cat((gold_sent_probs, wrong_sent_probs), 1)
        targets = gold_sent_probs.new_zeros(gold_sent_probs.size(0), dtype=torch.long)

        # Fix unintended division by dimention number by the library.
        # https://github.com/pytorch/pytorch/blob/v1.3.1/aten/src/THNN/generic/MultiMarginCriterion.c#L106
        loss = self.margin_criterion(probs, targets) * 2.0
        return loss, loss, None

class SentenceUnlikelihoodLossCalculator(SentenceNegLossCalculator):
    def loss(self, model, batch):
        # Try to decrease p(wrong_tgt) by increasing 1 - p(wrong_tgt).
        # Loss: -log(1 - p(wrong_tgt)).
        device = next(model.parameters()).device
        gold_srcs, lengths, gold_tgts, wrong_srcs, wrong_tgts = batch
        lengths = lengths.to(device)
        wrong_srcs = wrong_srcs.to(device)
        wrong_tgts = wrong_tgts.to(device)

        wrong_outputs, _ = model.rnn(wrong_srcs, input_lengths = lengths, return_h = False)
        wrong_probs = model.loss(wrong_outputs, wrong_tgts, reduce=False) * -1.0

        wrong_sent_probs = self._to_sent_probs(wrong_probs, lengths)

        loss = utils.log1mexp(wrong_sent_probs)
        loss = loss.mean() * -1.0
        return loss, loss, None


class SentenceWrongLikelihoodLossCalculator(SentenceNegLossCalculator):
    def loss(self, model, batch):
        # Loss: log(p(wrong_tgt))
        # Just reverse of the original loss; decrease the likelihood of wrong_tgt.
        device = next(self.model.parameters()).device
        gold_srcs, lengths, gold_tgts, wrong_srcs, wrong_tgts = batch
        lengths = lengths.to(device)
        wrong_srcs = wrong_srcs.to(device)
        wrong_tgts = wrong_tgts.to(device)

        wrong_outputs, _ = model.rnn(wrong_srcs, input_lengths = lengths, return_h = False)
        loss = model.loss(wrong_outputs, wrong_tgts) * -1.0
        return loss, loss, None


class SentenceLastTokenUnlikelihoodLossCalculator(SentenceNegLossCalculator):
    def loss(self, model, batch):
        # Loss: -log(1-p(wrong_tgt[-1]))
        # This should be used with `--margin-prefix-only`
        device = next(model.parameters()).device
        gold_srcs, lengths, gold_tgts, wrong_srcs, wrong_tgts = batch
        lengths = lengths.to(device)
        wrong_srcs = wrong_srcs.to(device)
        wrong_tgts = wrong_tgts.to(device)

        wrong_outputs, _ = model.rnn(wrong_srcs, input_lengths = lengths, return_h = False)
        last_idxes = self._last_token_indexes(lengths, wrong_srcs.size(1))
        last_outputs = wrong_outputs[last_idxes]
        last_wrong_tgts = wrong_tgts[last_idxes]

        wrong_probs = model.loss(last_outputs, last_wrong_tgts, reduce=False) * -1.0
        loss = utils.log1mexp(wrong_probs) * -1.0
        loss = loss.mean()
        return loss, loss, None


class SentenceLastTokenBinaryPredictionLossCalculator(SentenceNegLossCalculator):
    def loss(self, model, batch):
        # Loss: -log(1-p(number))
        # We assume the last tokens in gold_tgts are transformed to 0/1, the correct
        # outputs for binary prediction task.
        device = next(model.parameters()).device
        gold_srcs, lengths, gold_tgts, wrong_srcs, wrong_tgts = batch
        lengths = lengths.to(device)
        srcs = gold_srcs.to(device)
        tgts = gold_tgts.to(device)

        outputs, _ = model.rnn(srcs, input_lengths = lengths, return_h = False)
        last_idxes = self._last_token_indexes(lengths, gold_srcs.size(1))
        bc_outputs = outputs[last_idxes]
        bc_tgts = tgts[last_idxes].float()

        loss = model.bc_loss(bc_outputs, bc_tgts)
        return loss, loss, None


class SentenceWithAgreementSentenceLossTrainer(SentenceLMTrainer):
    def __init__(self, model, validator, config, sentence_neg_loss_calculator):
        super().__init__(model, validator, config)
        self.sentence_neg_loss_calculator = sentence_neg_loss_calculator

    def _batch_loss(self, batch, is_tag, hidden=None):
        if len(batch) == 3:
            return self._sent_batch_loss(batch, is_tag)
        else:
            return self.sentence_neg_loss_calculator.loss(self.model, batch)


class TokenNegLossCalculator(object):

    def __init__(self):
        self.sentloss_calc = SentenceNegLossCalculator()

    def loss(self, model, output, targets, negatives, sent_len, batch):
        pass

    def _neg_idx_targets(self, negatives, sent_len):
        neg_idx = []
        neg_targets = []
        for i in range(len(negatives)):
            sent_neg = negatives[i]
            if len(sent_neg) == 0:
                continue
            for idx, neg_tokens in sent_neg:
                for neg_token in neg_tokens:
                    neg_idx.append(i*sent_len + idx)
                    neg_targets.append(neg_token)
        return neg_idx, neg_targets

    def _neg_sent_probs(self, model, negatives, gold_srcs, lengths, gold_tgts):
        # create wrong_srcs, wrong_tgts
        # the batch size equals len(gold_sent_probs_copy)
        # copy gold_srcs and gold_tgts appropriately;
        # then, modify elements of them
        gold_tgts = gold_tgts.view(gold_srcs.size(0), gold_srcs.size(1))
        wrong_srcs = []
        wrong_tgts = []
        wrong_lengths = []
        for i in range(len(negatives)):
            for idx, neg_tokens in negatives[i]:
                for neg_token in neg_tokens:
                    src = gold_srcs[i].clone()
                    tgt = gold_tgts[i].clone()
                    src[idx+1] = neg_token
                    tgt[idx] = neg_token
                    wrong_srcs.append(src)
                    wrong_tgts.append(tgt)
                    wrong_lengths.append(lengths[i])

        # batch_len = wrong_lengths[0]
        wrong_srcs = [src.unsqueeze(0) for src in wrong_srcs]
        wrong_tgts = [tgt.unsqueeze(0) for tgt in wrong_tgts]
        wrong_srcs = torch.cat(wrong_srcs, 0)
        wrong_tgts = torch.cat(wrong_tgts, 0)
        wrong_lengths = lengths.new_tensor(wrong_lengths)

        mask = self._mask_for_sample_neg_sents(gold_srcs, wrong_srcs)
        wrong_srcs = wrong_srcs[mask]
        wrong_tgts = wrong_tgts[mask]
        wrong_lengths = wrong_lengths[mask]

        batch_len = wrong_lengths[0]
        wrong_srcs = wrong_srcs[:,:batch_len]
        wrong_tgts = wrong_tgts[:,:batch_len]
        wrong_tgts = wrong_tgts.contiguous().view(-1)

        wrong_outputs, _ = model.rnn(wrong_srcs, input_lengths = wrong_lengths, return_h = False)
        wrong_probs = model.loss(wrong_outputs, wrong_tgts, reduce=False) * -1.0

        wrong_sent_probs = self.sentloss_calc._to_sent_probs(wrong_probs, wrong_lengths)

        return wrong_sent_probs, mask

    def _mask_for_sample_neg_sents(self, gold_srcs, wrong_srcs):
        idxs = wrong_srcs.new_tensor(np.random.permutation(range(wrong_srcs.size(0))), dtype=torch.long)
        gold_tokens = gold_srcs.size(0) * gold_srcs.size(1)

        i = 1
        for i in range(1, len(idxs)):
            if i * wrong_srcs.size(1) > gold_tokens // 2:
                break
        wrong_num_sents = i
        mask = gold_srcs.new_zeros(wrong_srcs.size(0), dtype=torch.bool)
        mask[idxs[:wrong_num_sents]] = True
        return mask

class TokenUnlikelihoodLossCalculator(TokenNegLossCalculator):

    def loss(self, model, output, targets, negatives, sent_len, batch):
        neg_idx, neg_targets = self._neg_idx_targets(negatives, sent_len)
        if len(neg_idx) == 0:
            return output.new_tensor([])
        masked_output = output[neg_idx]
        neg_targets = output.new_tensor(neg_targets, dtype=torch.long)

        neg_probs = model.loss(masked_output, neg_targets, reduce=False) * -1.0
        neg_loss = utils.log1mexp(neg_probs) * -1.0
        return neg_loss


class TokenMarginLossCalculator(TokenNegLossCalculator):

    def __init__(self, margin=1.0):
        super(TokenNegLossCalculator, self).__init__()
        self.margin_criterion = torch.nn.MultiMarginLoss(
            margin=margin, reduction='none')

    def loss(self, model, output, targets, negatives, sent_len, batch):
        neg_idx, neg_targets = self._neg_idx_targets(negatives, sent_len)
        if len(neg_idx) == 0:
            return output.new_tensor([])
        masked_output = output[neg_idx]
        neg_targets = output.new_tensor(neg_targets, dtype=torch.long)
        targets = targets[neg_idx]
        assert neg_targets.size() == targets.size()

        # loss: margin between log_p(targets[i]) and log_p(neg_targets[i]) from
        # a state masked_output[i].
        gold_probs = model.loss(masked_output, targets, reduce=False) * -1.0
        wrong_probs = model.loss(masked_output, neg_targets, reduce=False) * -1.0
        probs = torch.cat((gold_probs, wrong_probs), 1)
        margin_targets = targets.new_zeros(probs.size(0))
        # Fix unintended division by dimention number by the library.
        # https://github.com/pytorch/pytorch/blob/v1.3.1/aten/src/THNN/generic/MultiMarginCriterion.c#L106
        neg_loss = self.margin_criterion(probs, margin_targets) * 2.0

        return neg_loss


class TokenBinaryPredictionLossCalculator(TokenNegLossCalculator):

    def loss(self, model, output, targets, negatives, sent_len, batch):
        """For binary prediction, `negatives` has a different meaning.

        Instead of keeping the negative token ids, each value is 0 or 1; 0 means the
        output is singular and 1 means the output is plural.
        """
        bc_idx, bc_targets = self._neg_idx_targets(negatives, sent_len)
        if len(bc_idx) == 0:
            return output.new_tensor([])
        masked_output = output[bc_idx]
        bc_targets = output.new_tensor(bc_targets)

        bc_loss = model.bc_loss(masked_output, bc_targets, reduce=False)
        return bc_loss


class WithinBatchSentenceMarginLossCalculator(TokenNegLossCalculator):

    def __init__(self, margin=1.0):
        super(WithinBatchSentenceMarginLossCalculator, self).__init__()
        self.margin_criterion = torch.nn.MultiMarginLoss(
            margin=margin, reduction='none')

    def loss(self, model, output, targets, negatives, sent_len, batch):
        gold_srcs, lengths, gold_tgts = batch

        gold_probs = model.loss(output, targets, reduce=False) * -1.0
        gold_sent_probs = self.sentloss_calc._to_sent_probs(gold_probs, lengths)
        gold_sent_probs_copy = []

        for i in range(len(negatives)):
            for idx, neg_tokens in negatives[i]:
                for neg_token in neg_tokens:
                    gold_sent_probs_copy.append(gold_sent_probs[i].unsqueeze(0))

        if len(gold_sent_probs_copy) == 0:
            return output.new_tensor([])

        gold_sent_probs_copy = torch.cat(gold_sent_probs_copy)
        wrong_sent_probs, mask = self._neg_sent_probs(
            model, negatives, gold_srcs, lengths, gold_tgts)
        gold_sent_probs_copy = gold_sent_probs_copy[mask]

        probs = torch.cat(
            (gold_sent_probs_copy.unsqueeze(1), wrong_sent_probs.unsqueeze(1)), 1)
        margin_targets = gold_sent_probs.new_zeros(gold_sent_probs_copy.size(0),
                                                   dtype=torch.long)

        # Fix unintended division by dimention number by the library.
        # https://github.com/pytorch/pytorch/blob/v1.3.1/aten/src/THNN/generic/MultiMarginCriterion.c#L106
        neg_loss = self.margin_criterion(probs, margin_targets) * 2.0

        return neg_loss


class WithinBatchSentenceUnlikelihoodLossCalculator(TokenNegLossCalculator):

    def loss(self, model, output, targets, negatives, sent_len, batch):
        gold_srcs, lengths, gold_tgts = batch

        if all(len(neg_sent) == 0 for neg_sent in negatives):
            return output.new_tensor([])

        wrong_sent_probs, _ = self._neg_sent_probs(
            model, negatives, gold_srcs, lengths, gold_tgts)
        neg_loss = utils.log1mexp(wrong_sent_probs) * -1.0

        return neg_loss


class SentenceWithAgreementTokenLossTrainer(SentenceLMTrainer):

    def __init__(self, model, validator, config, token_neg_loss_calculator):
        super().__init__(model, validator, config)
        self.token_neg_loss_calculator = token_neg_loss_calculator

    def _batch_loss(self, batch, is_tag, hidden=None):
        return self._batch_loss_with_neg_tokens(batch, hidden)

    def _batch_loss_with_neg_tokens(self, batch, hidden):
        device = next(self.model.parameters()).device
        sources, lengths, targets, negatives = batch
        sources = sources.to(device)
        lengths = lengths.to(device)
        targets = targets.to(device)
        batch = (sources, lengths, targets)

        if self.config.within_batch_neg_prob > 0:
            # Do not sample negative for some probability.
            # Intended to be used with sentence-level margin loss within a batch.
            if np.random.rand() > self.config.within_batch_neg_prob:
                negatives = []

        output, hidden, rnn_hs, dropped_rnn_hs = self.model.rnn(
            sources, input_lengths = lengths, return_h = True)

        raw_loss = self.model.loss(output, targets, reduce=False)

        unlikelihood_loss = self.token_neg_loss_calculator.loss(
            self.model, output, targets, negatives, sources.size(1), batch)

        ntokens = lengths.sum()
        if self.config.do_sample_average_before:
            if self.config.normalize_negative:
                x = 1.0 / (1.0 + self.config.negative_loss_alpha)
                y = self.config.negative_loss_alpha / (1.0 + self.config.negative_loss_alpha)
                raw_loss = raw_loss.mean() * x + unlikelihood_loss.mean() * y
            else:
                raw_loss = raw_loss.mean()
                raw_loss += unlikelihood_loss.mean() * self.config.negative_loss_alpha
        elif self.config.sample_average_jointly:
            unlikelihood_loss *= self.config.negative_loss_alpha
            raw_loss = (raw_loss.sum() + unlikelihood_loss.sum()) / (ntokens + unlikelihood_loss.size(0))
        else:
            unlikelihood_loss *= self.config.negative_loss_alpha
            raw_loss = (raw_loss.sum() + unlikelihood_loss.sum()) / ntokens

        loss = raw_loss
        loss += self._reguralize(dropped_rnn_hs[-1], lengths.sum())

        # Returnning hidden here is not meaningful. Just None should be ok.
        return raw_loss, loss, hidden


class DocumentLMTrainer(LMTrainer):

    def __init__(self, model, validator, config, bptt):
        super().__init__(model, validator, config)
        self.bptt = bptt

    def _pre_batch(self, batch):
        lr2 = self.optimizer.param_groups[0]['lr']
        seq_len = batch[0].size(1)
        self.optimizer.param_groups[0]['lr'] = lr2 * seq_len / self.bptt

    def _batch_loss(self, batch, is_tag, hidden):
        assert is_tag == False # we don't support multi-task for document models
        device = next(self.model.parameters()).device
        sources, targets = batch
        sources = sources.to(device)
        targets = targets.to(device)

        # Starting each batch, we detach the hidden state from how it was previously
        # produced. If we didn't, the model would try backpropagating all the way
        # to start of the dataset.
        hidden = utils.repackage_hidden(hidden)
        output, hidden, rnn_hs, dropped_rnn_hs = self.model.rnn(
            sources, hidden, return_h=True)

        if is_tag:
            raw_loss = self.model.tag_loss(output, targets)
        else:
            raw_loss = self.model.loss(output, targets)
        loss = raw_loss

        last_rnn_h = dropped_rnn_hs[-1]
        alpha, beta = self.config.alpha, self.config.beta
        if alpha:
            loss += alpha * last_rnn_h.pow(2).mean()
        if beta:
            loss += beta * (last_rnn_h[:,1:] - last_rnn_h[:,:-1]).pow(2).mean()

        return raw_loss, loss, hidden
