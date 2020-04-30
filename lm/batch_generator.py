
import numpy as np
import torch

import data
import utils

class BatchGenerator(object):
    def __call__(self):
        pass

    def for_eval(self):
        pass

    def total_batches(self):
        pass

class DocumentBatchGenerator(BatchGenerator):
    """A batch generator that only generates word batches.

    Each seq in a batch is a sequence of tokens in the input document, not necessaliry a
    sentence. (This means the training is done via truncated back-prop.)

    SentenceBatchGenerator, on the other hand, handles each sentence independently.

    Each __call__ method returns a tuple of (batch, is_tag, tag_alpha). is_tag indicates
    whether this batch is a batch of tags, rather than words. For this class, this value is
    fixed to the value given to the initializer. tag_alpha is also fixed to 1.0, which weights
    the loss computation for this batch.

    These values will be changed for different batches in `SentenceAndTagBatchGenerator` class,
    whether each batch may be a batch of word sequences, or a batch of tag sequences.

    """

    def __init__(self, tensors, batch_size, bptt, shuffle=False, is_tag=False):
        self.tensors = tensors
        self.data = utils.batchify(tensors, batch_size, batch_first=True)
        self.batch_size = batch_size
        self.bptt = bptt
        self.shuffle=shuffle
        self.is_tag = False

    def __call__(self):
        if self.shuffle:
            # I know shuffling the sentences in the document mode is non-sensical. This is just
            # experimental to see the importance of sentence worder for document-LM.
            np.random.shuffle(self.tensors)
            self.data = utils.batchify(self.tensors, self.batch_size, batch_first=True)
        i = 0
        while i < self.data.size(1) - 1 - 1:
            bptt = self.bptt if np.random.random() < 0.95 else self.bptt / 2.
            # Prevent excessively small or negative sequence lengths
            seq_len = max(5, int(np.random.normal(self.bptt, 5)))
            # There's a very small chance that it could select a very long sequence
            # length resulting in OOM
            # seq_len = min(seq_len, args.bptt + 10)
            sources, targets = utils.get_batch(
                self.data, i, bptt, seq_len=seq_len, batch_first=True)
            yield (sources, targets), self.is_tag, 1.0
            i += seq_len

    def for_eval(self):
        for i in range(0, self.data.size(1) - 1, self.bptt):
            sources, targets = utils.get_batch(
                self.data, i, self.bptt, evaluation=True, batch_first=True)
            yield sources, targets

    def total_batches(self):
        return self.data.size(1) // self.bptt

class SentenceBatchGenerator(BatchGenerator):

    def __init__(self, tensors, batch_size, pad_id, shuffle=False, length_bucket=False):
        self.tensors = tensors
        self.batch_size = batch_size
        self.pad_id = pad_id
        self.shuffle = shuffle
        self.length_bucket = length_bucket

    def __call__(self):
        if self.shuffle and self.length_bucket:
            sents_sorted = sorted(self.tensors, key=lambda x: len(x))
            batches = [b for b in data.batches_in_buckets(self.tensors, self.batch_size)]
            np.random.shuffle(batches)
            for batch in batches:
                sources, lengths, targets = utils.get_sorted_sentences_batch(
                    batch, 0, self.batch_size, self.pad_id)
                yield (sources, lengths, targets), False, 1.0
        else:
           if self.shuffle:
               np.random.shuffle(self.tensors)
           for i in range(0, len(self.tensors), self.batch_size):
               # Create a batch matrix with padding.
               sources, lengths, targets = utils.get_sorted_sentences_batch(
                   self.tensors, i, self.batch_size, self.pad_id)
               yield (sources, lengths, targets), False, 1.0

    def for_eval(self):
        for i in range(0, len(self.tensors), self.batch_size):
            sources, lengths, targets = utils.get_sorted_sentences_batch(
                self.tensors, i, self.batch_size, self.pad_id)
            yield sources, lengths, targets

    def total_batches(self):
        return len(self.tensors) // self.batch_size


def mk_pairs_batch(batch_pairs, pad_id):
    batch_pairs = sorted(batch_pairs, key=lambda x: len(x[0]), reverse=True)
    assert len(batch_pairs[0]) == 3

    golds = [p[0] for p in batch_pairs]
    wrongs = [p[1] for p in batch_pairs]
    obj_rels = [p[2] for p in batch_pairs]

    gold_sources, lengths, gold_targets = utils.get_sorted_sentences_batch(
        golds, 0, len(golds), pad_id, sorted=True)
    wrong_sources, wrong_lengths, wrong_targets = utils.get_sorted_sentences_batch(
        wrongs, 0, len(wrongs), pad_id, sorted=True)

    assert lengths.eq(wrong_lengths).all()
    return (gold_sources, lengths, gold_targets, wrong_sources, wrong_targets)


def mk_batches_list(seq, batch_size, shuffle, length_bucket,
                    len_fun = lambda x: len(x)):
    if shuffle and length_bucket:
        seq_sorted = sorted(seq, key=len_fun)
        batches = [b for b in data.batches_in_buckets(seq_sorted, batch_size)]
        np.random.shuffle(batches)
    else:
        if shuffle:
            np.random.shuffle(seq)
        batches = [seq[j:j+batch_size] for j in range(0, len(seq), batch_size)]
    return batches


class AgreementPairBatchGenerator(BatchGenerator):

    def __init__(self, tensor_pairs, batch_size, pad_id, shuffle=False):
        self.tensor_pairs = tensor_pairs
        self.batch_size = batch_size
        self.pad_id = pad_id
        self.shuffle = shuffle

    def __call__(self):
        """A batch for this generator is a quintuple
        (gold_sources, lengths, gold_targets, wrong_sources, wrong_targets).

        gold_sources and gold_targets are sources and targets sequences for calculating
        probabilities of gold sequences. wrong_sources and wrong_targets are the ones
        for distracted (ungrammatical) sequences.

        lengths are shared between two types of sequences, because we assume a distracted
        sentence has the same length as the original one (with a minimal change).

        """

        if self.shuffle:
            np.random.shuffle(self.tensor_pairs)
        return self._gen()

    def _gen(self):
        for i in range(0, len(self.tensor_pairs), self.batch_size):
            batch_pairs = self.tensor_pairs[i:i+self.batch_size]
            yield mk_pairs_batch(batch_pairs, self.pad_id), False, 1.0

    def for_eval(self):
        return self._gen()

    def total_batches(self):
        return len(self.tensor_pairs) // self.batch_size


class AgreementPairBatchSampler(object):
    def __init__(self, tensor_pairs, batch_size, one_pair_per_sent=False):
        self.tensor_pairs = tensor_pairs
        self.batch_size = batch_size
        self.batches = self.init_batches()
        self.idx = 0
        self.one_pair_per_sent = one_pair_per_sent

        if not one_pair_per_sent:
            from itertools import chain
            self.flatten = list(chain.from_iterable(self.tensor_pairs))

    def init_batches(self):
        if self.one_pair_per_sent:
            def sample(sent_pairs):
                return sent_pairs[np.random.randint(len(sent_pairs))]
            examples = [sample(sent_pairs) for sent_pairs in self.tensor_pairs]
        else:
            examples = self.flatten
        # np.random.shuffle(self.tensor_pairs)
        # t = self.tensor_pairs
        t = examples
        b = self.batch_size
        return [t[j:j+b] for j in range(0, len(t), b)]

    def next(self):
        if self.idx < len(self.batches):
            self.idx += 1
            return self.batches[self.idx - 1]
        else:
            assert self.idx == len(self.batches)
            self.batches = self.init_batches()
            self.idx = 0
            return self.next()


class AgreementBatchGeneratorBase(BatchGenerator):

    # We want to make this class equip all required methods manipulating tensor_pairs.
    # Some methods may be specialized (and not relevant) in some child class.
    def __init__(self,
                 tensor_pairs,
                 batch_size,
                 pad_id,
                 shuffle,
                 upsample_agreement,
                 agreement_loss_alpha,
                 half_agreement_batch,
                 one_pair_per_sent,
                 agreement_sample_ratio,
                 prefer_obj_rel):

        assert isinstance(tensor_pairs, list)
        if len(tensor_pairs) > 0: assert isinstance(tensor_pairs[0], list)
        if len(tensor_pairs[0]) > 0:
            assert isinstance(tensor_pairs[0][0], tuple)
            assert len(tensor_pairs[0][0]) == 3
            assert isinstance(tensor_pairs[0][0][0], torch.Tensor)
            assert isinstance(tensor_pairs[0][0][1], torch.Tensor)
            assert isinstance(tensor_pairs[0][0][2], bool)

        self.tensor_pairs = tensor_pairs

        self.batch_size = batch_size
        self.ag_batch_size = self.batch_size // 2 if half_agreement_batch \
                             else self.batch_size

        self.pad_id = pad_id
        self.shuffle = shuffle
        self.upsample_agreement = upsample_agreement
        self.agreement_loss_alpha = agreement_loss_alpha
        self.half_agreement_batch = half_agreement_batch
        self.one_pair_per_sent = one_pair_per_sent
        self.agreement_sample_ratio = agreement_sample_ratio
        self.prefer_obj_rel = prefer_obj_rel

        self.pair_batch_sampler = None
        if self.upsample_agreement:
            self.pair_batch_sampler = AgreementPairBatchSampler(
                self.tensor_pairs, self.ag_batch_size)

        if not self.upsample_agreement and not self.one_pair_per_sent:
            from itertools import chain
            self.flatten_pairs = list(chain.from_iterable(self.tensor_pairs))

    def __call__(self):
        """A batch for this generator is either a sentence (or document) batch, or a pair
        batch (for optimizing sensitivity to agreement errors). Generally, # training
        instances for agreement is smaller; the aim for `upsample` is to reduce this bias.
        If this value is True, batches for sentence pairs are upsampled, and two types
        of batches are generated alternatively.

        """

        if self.upsample_agreement:
            return self.alternate()
        else:
            return self.visit_once()

    def alternate(self):
        for main_batch in self._gen_main_batches():
            yield main_batch
            yield self._sample_pairs_batch()

    def visit_once(self):
        def maybe_iter_to_list(gen):
            if not isinstance(gen, list):
                gen = [a for a in gen]
            return gen

        main_batches = self._gen_main_batches()
        # main_batches = maybe_iter_to_list(main_batches)

        pair_examples = self._gen_cand_pairs()
        pair_examples = maybe_iter_to_list(pair_examples)

        if self.prefer_obj_rel:  # always contains obj rel cases.
            obj_rel_examples = [p for p in pair_examples if p[2]]
            non_obj_rel_examples = [p for p in pair_examples if not p[2]]

            preferred_pair_batches = mk_batches_list(
                obj_rel_examples, self.ag_batch_size, self.shuffle, False)
            pair_batches = mk_batches_list(
                non_obj_rel_examples, self.ag_batch_size, self.shuffle, False)
        else:
            preferred_pair_batches = []
            pair_batches = mk_batches_list(
                pair_examples, self.ag_batch_size, self.shuffle, False)

        main_batches = [(b, False) for b in main_batches]
        preferred_pair_batches = [(b, True) for b in preferred_pair_batches]
        pair_batches = [(b, True) for b in pair_batches]

        if self.agreement_sample_ratio > 0:
            def get_sample_recur(n_remain, current_batches):
                if n_remain <= 0:
                    return current_batches
                if n_remain < len(pair_batches):
                    np.random.shuffle(pair_batches)
                    return current_batches + pair_batches[:n_remain]
                else:
                    combined = current_batches + pair_batches
                    return get_sample_recur(n_remain - len(pair_batches), combined)

            r = self.agreement_sample_ratio
            n = int(len(main_batches) * r)

            print('size of preferred: {}'.format(len(preferred_pair_batches)))
            print('size of others: {}'.format(len(pair_batches)))
            pair_batches = get_sample_recur(n - len(preferred_pair_batches),
                                            preferred_pair_batches)
            print('total batches for agreement: {}'.format(len(pair_batches)))
        else:
            # preferred_pair_bathces is not meaningful when agreement_sample_ratio is inactive.
            if len(preferred_pair_batches) > 0:
                # recover the original, non segmented pair_batches
                pair_batches += preferred_pair_batches

        batches = main_batches + pair_batches
        if self.shuffle:
            np.random.shuffle(batches)
        for batch, is_pair in batches:
            if is_pair:
                batch = mk_pairs_batch(batch, self.pad_id)
                yield batch, False, self.agreement_loss_alpha
            else:
                yield batch
                # sources, lengths, targets = utils.get_sorted_sentences_batch(
                #     batch, 0, self.batch_size, self.pad_id)
                # yield (sources, lengths, targets), False, 1.0

    def _gen_main_batches(self):
        pass

    def _gen_cand_pairs(self):
        if self.one_pair_per_sent:
            def sample(sent_pairs):
                return sent_pairs[np.random.randint(len(sent_pairs))]
            return [sample(sent_pairs) for sent_pairs in self.tensor_pairs]
        else:
            return self.flatten_pairs

    def _sample_pairs_batch(self):
        assert self.upsample_agreement
        batch = self.pair_batch_sampler.next()
        batch = mk_pairs_batch(batch, self.pad_id)
        return batch, False, self.agreement_loss_alpha


class SentenceAndAgreementBatchGenerator(AgreementBatchGeneratorBase):

    def __init__(self,
                 tensors,
                 tensor_pairs,
                 batch_size,
                 pad_id,
                 shuffle=False,
                 length_bucket=False,
                 upsample_agreement=False,
                 agreement_loss_alpha=1.0,
                 half_agreement_batch=False,
                 one_pair_per_sent=False,
                 agreement_sample_ratio=0.0,
                 prefer_obj_rel=False):

        super(SentenceAndAgreementBatchGenerator, self).__init__(
            tensor_pairs,
            batch_size,
            pad_id,
            shuffle,
            upsample_agreement,
            agreement_loss_alpha,
            half_agreement_batch,
            one_pair_per_sent,
            agreement_sample_ratio,
            prefer_obj_rel)

        assert isinstance(tensors, list)
        if tensors: assert isinstance(tensors[0], torch.Tensor)

        self.tensors = tensors

        self.length_bucket = length_bucket

    def _gen_main_batches(self):
        def sent_batch_to_ready(batch):
            sources, lengths, targets = utils.get_sorted_sentences_batch(
                batch, 0, self.batch_size, self.pad_id)
            return (sources, lengths, targets), False, 1.0

        sent_batches = mk_batches_list(
            self.tensors, self.batch_size, self.shuffle, self.length_bucket)

        sent_batches = [sent_batch_to_ready(b) for b in sent_batches]

        return sent_batches

    def for_eval(self):
        for i in range(0, len(self.tensors), self.batch_size):
            sources, lengths, targets = utils.get_sorted_sentences_batch(
                self.tensors, i, self.batch_size, self.pad_id)
            yield sources, lengths, targets

        pair_batches = mk_batches_list(
            self.flatten_pairs, self.ag_batch_size, False, False)

        for batch in pair_batches:
            yield mk_pairs_batch(batch, self.pad_id)

    def total_batches(self):
        if self.pair_batch_sampler:
            return (len(self.tensors) // self.batch_size) * 2
        else:
            sent_batches = len(self.tensors) // self.batch_size
            if self.one_pair_per_sent:
                ag_batches = len(self.tensor_pairs) // self.ag_batch_size
            elif self.agreement_sample_ratio > 0:
                ag_batches = int(sent_batches * self.agreement_sample_ratio)
            else:
                ag_batches = len(self.flatten_pairs) // self.ag_batch_size
            return sent_batches + ag_batches


class SentenceWithAgreementBatchGenerator(BatchGenerator):

    def __init__(self,
                 tensors,
                 neg_vectors,
                 batch_size,
                 pad_id,
                 shuffle=False,
                 length_bucket=False):
                 # is_binary_prediction=False):

        assert len(tensors) == len(neg_vectors)

        self.tensors = list(zip(tensors, neg_vectors))
        self.neg_vectors = neg_vectors
        self.batch_size = batch_size
        self.pad_id = pad_id
        self.shuffle = shuffle
        self.length_bucket = length_bucket
        # self.is_binary_prediction = is_binary_prediction

    def __call__(self):
        if self.shuffle and self.length_bucket:
            sents_sorted = sorted(self.tensors, key=lambda x: len(x[0]))
            batches = [b for b in data.batches_in_buckets(self.tensors, self.batch_size)]
            np.random.shuffle(batches)
            for batch in batches:
                sources, lengths, targets, agreements = (
                    utils.get_sorted_sentences_with_agreement_batch(
                        batch, 0, self.batch_size, self.pad_id))
                yield (sources, lengths, targets, agreements), False, 1.0
        else:
           if self.shuffle:
               np.random.shuffle(self.tensors)
           for i in range(0, len(self.tensors), self.batch_size):
               # Create a batch matrix with padding.
               sources, lengths, targets, agreements = (
                   utils.get_sorted_sentences_with_agreement_batch(
                       self.tensors, i, self.batch_size, self.pad_id))
               yield (sources, lengths, targets, agreements), False, 1.0

    def for_eval(self):
        for i in range(0, len(self.tensors), self.batch_size):
            sources, lengths, targets, agreements = (
                utils.get_sorted_sentences_with_agreement_batch(
                    self.tensors, i, self.batch_size, self.pad_id))
            yield sources, lengths, targets, agreements

    def total_batches(self):
        return len(self.tensors) // self.batch_size


class DocumentAndAgreementBatchGenerator(BatchGenerator):
    def __init__(self, tensors,
                 tensor_pairs,
                 batch_size,
                 bptt,
                 shuffle=False,
                 upsample_agreement=False,
                 agreement_loss_alpha=1.0,
                 half_agreement_batch=False,
                 agreement_sample_ratio=0.0,
                 prefer_obj_rel=False):

        self.tensors = tensors
        self.tensor_pairs = tensor_pairs
        self.batch_size = batch_size
        self.ag_batch_size = self.batch_size // 2 if half_agreement_batch \
                             else self.batch_size
        self.pad_id = pad_id
        self.shuffle = shuffle
        self.length_bucket = length_bucket
        self.upsample_agreement = upsample_agreement
        self.agreement_loss_alpha = agreement_loss_alpha
        self.agreement_sample_ratio = agreement_sample_ratio

        self.pair_batch_sampler = None
        self.one_pair_per_sent = one_pair_per_sent
        self.prefer_obj_rel = prefer_obj_rel
        if self.upsample_agreement:
            self.pair_batch_sampler = AgreementPairBatchSampler(
                self.tensor_pairs, self.ag_batch_size)
        else:
            if not one_pair_per_sent:
                from itertools import chain
                self.flatten_pairs = list(chain.from_iterable(self.tensor_pairs))



class TagBatchSampler(object):
    def __init__(self, tensors, batch_size):
        self.tensors = tensors
        self.batch_size = batch_size
        self.batches = self.init_batches()
        self.idx = 0

    def init_batches(self):
        np.random.shuffle(self.tensors)
        t = self.tensors
        b = self.batch_size
        return [t[j:j+b] for j in range(0, len(t), b)]

    def next(self):
        if self.idx < len(self.batches):
            self.idx += 1
            return self.batches[self.idx - 1]
        else:
            assert self.idx == len(self.batches)
            self.batches = self.init_batches()
            self.idx = 0
            return self.next()


class SentenceAndTagBatchGenerator(BatchGenerator):

    def __init__(
            self,
            sent_tensors,
            tag_tensors, # [(sent1, tags1), (sent2, tags2), ...]
            batch_size,
            pad_id,
            tag_pad_id,
            shuffle=False,
            length_bucket=False,
            upsample_tags=True,
            tag_loss_alpha=1.0):

        assert isinstance(tag_tensors, list)
        assert isinstance(tag_tensors[0], tuple)

        self.sent_tensors = sent_tensors
        self.tag_tensors = tag_tensors
        self.batch_size = batch_size
        self.pad_id = pad_id
        self.tag_pad_id = tag_pad_id
        self.shuffle = shuffle
        self.length_bucket = length_bucket
        self.upsample_tags = upsample_tags
        self.tag_loss_alpha = tag_loss_alpha

        self.tag_batch_sampler = None
        if self.upsample_tags:
            self.tag_batch_sampler = TagBatchSampler(self.tag_tensors, self.batch_size)

    def __call__(self):
        if self.upsample_tags:
            return self.alternate()
        else:
            return self.visit_once()

    def alternate(self):
        def to_sent_batch(batch):
            sources, lengths, targets = utils.get_sorted_sentences_batch(
                batch, 0, self.batch_size, self.pad_id)
            return (sources, lengths, targets), False, 1.0

        def sample_tag_batch():
            batch = self.tag_batch_sampler.next()
            sources, lengths, targets = utils.get_sorted_tagged_sentences_batch(
                batch, 0, self.batch_size, self.tag_pad_id)
            return (sources, lengths, targets), True, self.tag_loss_alpha

        # alternate between sent batch and tag batch.
        sent_bathces = mk_batches_list(
            self.sent_tensors, self.batch_size, self.shuffle, self.length_bucket)
        for batch in sent_bathces:
            yield to_sent_batch(batch)
            yield sample_tag_batch()

    def visit_once(self):
        sent_bathces = mk_batches_list(
            self.sent_tensors, self.batch_size, self.shuffle, self.length_bucket)
        tag_batches = mk_batches_list(
            self.tag_tensors, self.batch_size, self.shuffle, False)
        sent_bathces = [(b, False) for b in sent_bathces]
        tag_batches = [(b, True) for b in tag_batches]
        batches = sent_bathces + tag_batches
        if self.shuffle:
            np.random.shuffle(batches)
        for batch, is_tag in batches:
            if is_tag:
                sources, lengths, targets = utils.get_sorted_tagged_sentences_batch(
                    batch, 0, self.batch_size, self.tag_pad_id)
                yield (sources, lengths, targets), True, self.tag_loss_alpha
            else:
                sources, lengths, targets = utils.get_sorted_sentences_batch(
                    batch, 0, self.batch_size, self.pad_id)
                yield (sources, lengths, targets), False, 1.0

    # def sent_bathces(self):
    #     if self.shuffle and self.length_bucket:
    #         sents_sorted = sorted(self.sent_tensors, key=lambda x: len(x))
    #         batches = [b for b in data.batches_in_buckets(self.sent_tensors, self.batch_size)]
    #         np.random.shuffle(batches)
    #     else:
    #         if self.shuffle:
    #             np.random.shuffle(self.sent_tensors)
    #         t = self.sent_tensors
    #         b = self.batch_size
    #         batches = [t[j:j+b] for j in range(0, len(t), b)]
    #     return batches

    # def tag_batches(self):
    #     if self.shuffle:
    #         np.random.shuffle(self.tag_tensors)
    #     t = self.tag_tensors
    #     b = self.batch_size
    #     batches = [t[j:j+b] for j in range(0, len(t), b)]
    #     return batches

    def total_batches(self):
        if self.tag_batch_sampler:
            return (len(self.sent_tensors) // self.batch_size) * 2
        else:
            return (len(self.sent_tensors) + len(self.tag_tensors)) // self.batch_size

