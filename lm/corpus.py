
import inflect
import numpy as np
import os

import data
import logging
import unkifier
import utils

logger = logging.getLogger('train_lm')

class Corpus(object):

    def __init__(self,
                 mode,
                 add_document_start,
                 data_path,
                 start_symbol,
                 min_count,
                 max_vocab_size,
                 unkify,
                 preproc,
                 reverse,
                 min_length,
                 max_unk_ratio):

        self.data_path = data_path

        is_tbptt = mode == 'document'
        def read_sentences(path):
            return data.read_sentences(path, is_tbptt, add_document_start, start_symbol)

        self.train_sentences = read_sentences(os.path.join(data_path, 'train.txt'))
        self.val_sentences = read_sentences(os.path.join(data_path, 'valid.txt'))
        self.test_sentences = read_sentences(os.path.join(data_path, 'test.txt'))

        if reverse:
            self._reverse_sents(self.train_sentences)
            self._reverse_sents(self.val_sentences)
            self._reverse_sents(self.test_sentences)

        self.vocab = data.build_vocab(
            self.train_sentences,
            min_count,
            max_vocab_size,
            utils.mk_unkifier(unkify),
            utils.mk_preproc(preproc),
            start_symbol)

        logger.info('Vocab size: {}'.format(self.vocab.size))

        # We first filter train_sentences, and then convert them into ids.
        self.ignore_idx = []
        if min_length > 0 or max_unk_ratio != 0.0:
            len_offset = 1 if is_tbptt else 2
            unk_ok = data.mk_filter_by_unk_ratio(self.vocab, max_unk_ratio, is_tbptt)
            self.ignore_idx = [i for i, s in enumerate(self.train_sentences)
                               if not unk_ok(s) and (len(s) - len_offset) < min_length]
        if self.ignore_idx:
            self.train_sentences = self._ignore_by_idx(self.train_sentences, self.ignore_idx)

        self.train_tensors = data.to_tensors(self.train_sentences, self.vocab)
        self.val_tensors = data.to_tensors(self.val_sentences, self.vocab)
        self.test_tensors = data.to_tensors(self.test_sentences, self.vocab)

    def mk_agreement_pairs(self, ignore_simple_agreement, target_syntax, prefix_only=False):
        def make_and_combine_pairs(files, is_train):
            agreement_pairs = []
            for f in files:
                # Consider prefix_only only when is_train is true.
                if 'agreements' in f and is_train:
                    obj_rel_fn = 'obj_rel_agreements.train.txt.gz'
                else:
                    obj_rel_fn = None
                agreement_pairs = agreement_pairs + self._mk_agreement_pairs_from(
                    ignore_simple_agreement, f, is_train, prefix_only,
                    obj_rel_fn)
            return data.to_tensor_pairs(agreement_pairs, self.vocab)

        train_paths, val_paths = self._train_valid_paths(target_syntax)
        train_pairs = make_and_combine_pairs(train_paths, True)
        val_pairs = make_and_combine_pairs(val_paths, False)

        print('Example pairs for agreement:')
        for pairs_for_sent in train_pairs[:3]:
            pair = pairs_for_sent[0]  # the first pair for one sentence
            print(' '.join(self.vocab.value(w) for w in pair[0]))
            print(' '.join(self.vocab.value(w) for w in pair[1]))
        print('Example pairs for agreement (valid):')
        for pairs_for_sent in val_pairs[:3]:
            pair = pairs_for_sent[0]  # the first pair for one sentence
            print(' '.join(self.vocab.value(w) for w in pair[0]))
            print(' '.join(self.vocab.value(w) for w in pair[1]))
        return train_pairs, val_pairs

    def mk_agreement_tokens(self,
                            ignore_simple_agreement,
                            target_syntax,
                            exclude_targets=[],
                            only_targets=[]):
        """Read the files of negative examples, output the sparse vectors preserving
        the relevant negative tokens.

        Example:

        When the file content looks like below,

        0
        1
        2 1 are True
        3 1 are True

        output a list [[], [], [(1, [31])], [(1, [31])]].

        We assume 31 is an id of "are" in the dictionary. Note that as in [31], each
        negative token can be a list, that is, can be multiple tokens. This is for
        handling cases where negative examples are not unique, e.g., "himself" and
        "herself" as negative examples of "themselves".

        When is_binary_prediction .

        """

        def mk_sparse_vectors(files, sents, is_train):
            def isunk(w):
                return (w[0] == '<' and w[-1] == '>' and w.find('unk') > 0) or \
                    not self.vocab.contains(w)

            def untarget(w):
                return isunk(w) or (w in exclude_targets)

            negative_examples = None
            for path in files:
                _negative_examples = data.read_negative_examples(
                    os.path.join(self.data_path, path))
                if ignore_simple_agreement:
                    _negative_examples = [[neg_token for neg_token in neg_sent \
                                           if not neg_token[2]] \
                                          for neg_sent in _negative_examples]
                if is_train:
                    _negative_examples = self._ignore_by_idx(_negative_examples,
                                                             self.ignore_idx)
                assert len(_negative_examples) == len(sents)

                if len(only_targets) > 0 and 'agreement' in path:
                    def accept(w1, w2):
                        return w1 in only_targets or w2 in only_targets
                else:
                    def accept(w1, w2):
                        return True

                _negative_examples = [[(i, [w]) for (i, w, _) in neg_sent \
                                       if ((not untarget(w) and
                                            not untarget(sents[sent_i][i+1]))
                                           and accept(w, sents[sent_i][i+1]))] \
                                      for sent_i, neg_sent in enumerate(_negative_examples)]

                if negative_examples is None:
                    negative_examples = _negative_examples
                else:
                    negative_examples = self._union_negatives(negative_examples,
                                                              _negative_examples)
            return data.neg_examples_to_ids(negative_examples, self.vocab)

        train_paths, val_paths = self._train_valid_paths(target_syntax)

        train_neg_vectors = mk_sparse_vectors(train_paths, self.train_sentences, True)
        val_neg_vectors = mk_sparse_vectors(val_paths, self.val_sentences, False)

        return (train_neg_vectors, val_neg_vectors)

    def mk_agreement_tokens_for_binary_prediction(self,
                                                  ignore_simple_agreement,
                                                  target_syntax):
        """Similar to `mk_agreement_tokens`, but output the list keeping information
        for binary prediction of singular/plural of each target token.

        Given the example in `mk_agreement_tokens`, output a list
        [[], [], [(1, [1])], [(1, [1])]], on which (1, [1]) means there will be binary
        prediction for 1-th token and the correct output is 1 (plural).

        Each token index (e.g., [1]) is a list, although it would never have a size
        greater than 1. This is for compatibility with the output format of the
        `mk_agreement_token`.
        """


        infl = inflect.engine()

        def neg_vectors_to_bc_targets(neg_vectors, sentences):
            infl = inflect.engine()
            assert len(neg_vectors) == len(sentences)
            for neg_vector, sentence in zip(neg_vectors, sentences):
                if len(neg_vector) == 0: continue
                for i in range(len(neg_vector)):
                    idx = neg_vector[i][0]
                    negatives = neg_vector[i][1]
                    orig = sentence[idx+1]  # skip BOS
                    is_plural = orig == 'themselves' or (infl.plural_verb(orig) == orig)
                    neg_vector[i] = (idx, [1]) if is_plural else (idx, [0])
            return neg_vectors

        # We first obtain the outputs of `mk_agreemnt_tokens` and modify them.
        train_neg_vectors, val_neg_vectors = self.mk_agreement_tokens(
            ignore_simple_agreement, target_syntax)

        train_bc_targets = neg_vectors_to_bc_targets(train_neg_vectors,
                                                     self.train_sentences)
        val_bc_targets = neg_vectors_to_bc_targets(val_neg_vectors, self.val_sentences)
        return (train_bc_targets, val_bc_targets)

    def _train_valid_paths(self, target_syntax):
        train_paths, val_paths = [], []
        if 'agreement' in target_syntax:
            train_paths.append('negative_agreements.train.txt.gz')
            val_paths.append('negative_agreements.valid.txt.gz')
        if 'reflexive' in target_syntax:
            train_paths.append('negative_reflexives.train.txt.gz')
            val_paths.append('negative_reflexives.valid.txt.gz')
        return (train_paths, val_paths)

    def _mk_agreement_pairs_from(self, ignore_simple_agreement, fn, is_train,
                                 prefix_only, obj_rel_fn = None):
        if is_train:
            sentences = self.train_sentences
            ignore_idx = self.ignore_idx
        else:
            sentences = self.val_sentences
            ignore_idx = []
        negative_examples = data.read_negative_examples(os.path.join(self.data_path, fn))

        obj_rel_examples = [[] for _ in range(len(negative_examples))]
        if obj_rel_fn != None:
            obj_rel_path = os.path.join(self.data_path, obj_rel_fn)
            if os.path.exists(obj_rel_path):
                obj_rel_examples = data.read_agreement_indexes(obj_rel_path,
                                                               negative_examples)
        if ignore_idx:
            negative_examples = self._ignore_by_idx(negative_examples, ignore_idx)
            obj_rel_examples = self._ignore_by_idx(obj_rel_examples, ignore_idx)
        # assert len(self.train_sentences) == len(negative_examples)
        agreement_pairs = data.mk_agreement_pairs(
            sentences, negative_examples, ignore_simple_agreement, True, prefix_only,
            obj_rel_examples)
        return agreement_pairs

    def _ignore_by_idx(self, items, ignore_idx):
        mask = np.ones(len(items), np.bool)
        mask[ignore_idx] = 0
        return np.array(items)[mask].tolist()

    def _union_negatives(self, negative1, negative2):
        assert len(negative1) == len(negative2)
        for i in range(len(negative2)):
            l1 = negative1[i]  # maybe [(1, [30]), (3, [21])]
            l1 = dict(l1)  # {1: [30], 3: [21]}
            for k, v in negative2[i]:  # negative2[i] might be [(3, [10])]
                if k in l1:
                    l1[k].extend(v)
                else:
                    l1[k] = v
            negative1[i] = sorted(l1.items())  # [(1, [30], (3, [21, 10]))]
        return negative1

    def _reverse_sents(self, sents):
        for s in sents: s.reverse()


class TagCorpus(object):

    def __init__(self,
                 data_path,
                 start_symbol,
                 word_vocab, # unkify and preproc are extracted from here
                 min_count,
                 fix_word_vocab = False,
                 build_tag_vocab = False,
                 reverse = False):

        """Parameters:

        @param fix_word_vocab bool If True, do not expand the word vocab anymore.
        @param build_tag_vocab bool If False, vocab for tags is treated separately from the
                                    word vocab.
        """

        start_tag = data.BOSTAG if start_symbol == data.BOS else data.EOSTAG

        def read_tagged_sentences(path):
            return data.read_tagged_sentences(path, start_symbol, start_tag)

        self.train_sentences = read_tagged_sentences(os.path.join(data_path, 'train.txt'))
        self.val_sentences = read_tagged_sentences(os.path.join(data_path, 'valid.txt'))
        self.test_sentences = read_tagged_sentences(os.path.join(data_path, 'test.txt'))

        if reverse:
            self._reverse_sents(self.train_sentences)
            self._reverse_sents(self.val_sentences)
            self._reverse_sents(self.test_sentences)

        token_seqs, tag_seqs = zip(*self.train_sentences)

        if not fix_word_vocab: # expand word_vocab
            word_vocab.unfreeze()
            additional_vocab = data.build_vocab(
                token_seqs, min_count, word_vocab.unkifier, word_vocab.proc,
                start_symbol=start_symbol)
            for w in additional_vocab.values: word_vocab.index(w)
            word_vocab.freeze()

        if build_tag_vocab:
            unkifier_ = unkifier.ConstUnkifier(data.UNKTAG)
            self.tag_vocab = data.build_vocab(tag_seqs, 2, unkifier_, start_symbol=start_tag)
            self.tag_vocab.freeze()
        else:
            word_vocab.unfreeze()
            for tags in tag_seqs:
                for tag in tags:
                    word_vocab.index(tag)
            word_vocab.freeze()
            self.tag_vocab = word_vocab

        self.word_vocab = word_vocab

        self.train_tensors = data.to_tagged_tensors(self.train_sentences, self.word_vocab, self.tag_vocab)
        self.val_tensors = data.to_tagged_tensors(self.val_sentences, self.word_vocab, self.tag_vocab)
        self.test_tensors = data.to_tagged_tensors(self.test_sentences, self.word_vocab, self.tag_vocab)

        logger.info('Updated vocab size: {}'.format(self.word_vocab.size))
        logger.info('Tag vocab size: {}'.format(self.tag_vocab.size))

        def _reverse_sents(self, sents):
            for s, t in sents:
                s.reverse()
                t.reverse()
