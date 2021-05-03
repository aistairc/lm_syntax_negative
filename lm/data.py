import collections
import gzip
import numpy as np
import os
import torch

from collections import Counter

from unkifier import Unkifier
import utils
import word_processor

BOS = '<bos>'
EOS = '<eos>'
PAD = '<pad>'
UNK = '<unk>'

BOSTAG = '<bostag>'
EOSTAG = '<eostag>'
UNKTAG = '<unktag>'

class Vocabulary(object):
    def __init__(self, proc = None, unkifier = None, start_symbol = None):
        if proc: assert isinstance(proc, word_processor.WordProcessor)
        if unkifier: assert isinstance(unkifier, Unkifier)
        self.frozen = False
        self.values = []
        self.indices = {}
        self.counts = collections.defaultdict(int)

        self.proc = proc
        self.unkifier = unkifier
        self.start_symbol = start_symbol

    @property
    def size(self):
        return len(self.values)

    def value(self, index):
        assert 0 <= index < len(self.values)
        return self.values[index]

    def index(self, value):
        if self.proc: value = self.proc(value)
        if not self.frozen:
            self.counts[value] += 1

        if value in self.indices:
            return self.indices[value]

        elif not self.frozen:
            self.values.append(value)
            self.indices[value] = len(self.values) - 1
            return self.indices[value]

        else:
            raise ValueError("Unknown value: {}".format(value))

    def index_unked(self, value, additional=None):
        """Return index after appropriate unkifying.

        `additoinal` is currently not used. `unkifier.unkify` can accept a pair of
        word and POS, but currently, POS inputs are not assumed. `additional` may provide
        such tags in future.
        """
        if self.proc: value = self.proc(value)
        if self.contains(value):
            return self.index(value)
        else:
            unk = self.unkifier.unkify(value, 'pos') if self.unkifier else UNK
            return self.index(unk)

    def contains(self, value):
        return value in self.counts

    def count(self, value):
        return self.counts[value]

    def freeze(self):
        self.frozen = True

    def unfreeze(self):
        self.frozen = False

def add_symbols(sentences, is_tbptt, add_document_start, start_symbol):
    """Add <eos> sentences appropriately."""
    if is_tbptt:
        def deco(seq): return seq + [EOS]
    else:
        def deco(seq): return [start_symbol] + seq + [EOS]
    decorated = [deco(s) for s in sentences]
    if add_document_start and is_tbptt:
        decorated[0] = [start_symbol] + decorated[0]
    return decorated


def read_sentences(path, is_tbptt = True, add_document_start = False, start_symbol = BOS,
                   sentence_piece_model = None):
    """Return list of tokenized sentences.

    If `is_tbptt=False`, each sentence is later treated independently, and an <bos>
    system is appended at the begin of each sentence.

    `add_document_start` is only valid when `is_tbptt=True`. When `add_document_start=True`,
    a <bos> or <eos> symbol is appended at the beginning of the first sentence (of the
    document). This symbol is assumed to be a start symbol of the entire document.
    """
    if not os.path.exists(path): path += '.gz'
    assert os.path.exists(path)
    sentences = []

    sp = None
    if sentence_piece_model is not None:
        import sentencepiece as spm
        sp = spm.SentencePieceProcessor(model_file=sentence_piece_model)
    if path.endswith('.gz'):
        def openf(p): return gzip.open(p, 'rt')
        # def decode(line): return line.decode('utf-8')
        def decode(line): return line
    else:
        def openf(p): return open(p)
        def decode(line): return line
    with openf(path) as f:
        for line in f:
            sent = decode(line).strip()
            if sp is not None:
                sent = sp.encode(sent, out_type=str)
            else:
                sent = sent.split()
            sentences.append(sent)
    return add_symbols(sentences, is_tbptt, add_document_start, start_symbol)

def read_negative_examples(path):
    if path.endswith('.gz'):
        def openf(p): return gzip.open(p, 'rt')
        # def decode(line): return line.decode('utf-8')
        def decode(line): return line
    else:
        def openf(p): return open(p)
        def decode(line): return line

    examples = []
    with openf(path) as f:
        for line in f:
            line = line.strip()
            # Each line has format: line_number idx word simple?[TAB]idx word simple?[TAB]...
            # We first ignore the first line_number column.
            s = line.find(' ')
            if s == -1:
                examples.append([])
            else:
                items = line[s+1:].split('\t')
                items = [item.split(' ') for item in items]
                items = [(int(item[0]), item[1], item[2] == 'True') for item in items]
                examples.append(items)
    return examples

def read_agreement_indexes(path, negative_examples):
    if path.endswith('.gz'):
        def openf(p): return gzip.open(p, 'rt')
        # def decode(line): return line.decode('utf-8')
        def decode(line): return line
    else:
        def openf(p): return open(p)
        def decode(line): return line

    examples = []
    with openf(path) as f:
        for i, line in enumerate(f):
            neg_idx = [n for (n, _, _) in negative_examples[i]]
            items = [int(i) for i in line[:-1].split()]
            items = [i for i in items if i in neg_idx]
            examples.append(items)
    return examples

def read_tagged_sentences(path, start_symbol = BOS, start_tag = BOSTAG):
    if not os.path.exists(path): path += '.gz'
    assert os.path.exists(path)

    if path.endswith('.gz'):
        def openf(p): return gzip.open(p, 'r')
        def decode(line): return line.decode('utf-8')
    else:
        def openf(p): return open(p)
        def decode(line): return line

    sentences = []
    with openf(path) as f:
        sent = []
        for line in f:
            line = line[:-1]
            if line:
                sent.append(line.split('\t'))
            else:
                tokens = [t[0] for t in sent]
                tokens = [start_symbol] + tokens + [EOS]
                tags = [t[1]+'-tag' for t in sent]
                tags = [start_tag] + tags + [EOSTAG]

                sentences.append((tokens, tags))
                sent = []
    return sentences

def mk_agreement_pairs(sentences, examples, ignore_simple=True, ignore_unk=True,
                       prefix_only=False, obj_rel_examples = None):
    def isunk(w):
        # This check is adhoc but currently sufficient. All preprocessing was done
        # with simple <unk> or with Berkeley's rules (that produces <unking>, <unked>, etc.).
        # Possibly this function may receive a function (or a regular expression) to judge
        # unk or not.
        return w[0] == '<' and w[-1] == '>' and w.find('unk') > 0
    assert len(sentences) == len(examples)
    pairs = [] # grouped by a sent
    for s, e, orc in zip(sentences, examples, obj_rel_examples):
        # s contains BOS, which shifts idx for 1 in e
        def position_to_pair(position):
            idx = position[0]
            wrong = position[1]
            simple = position[2]

            if ignore_simple and simple: return None

            orig = s
            orig_word = orig[idx + 1]  # 1 is for BOS

            if ignore_unk and (isunk(orig_word) or isunk(wrong)): return None

            changed = orig.copy()
            changed[idx + 1] = wrong

            obj_rel = idx in orc

            if prefix_only:
                return (orig[:idx+2], changed[:idx+2], obj_rel)
            else:
                return (orig, changed, obj_rel)
        if not e: continue
        sent_pairs = [position_to_pair(p) for p in e]
        sent_pairs = [p for p in sent_pairs if p]
        if len(sent_pairs) > 0:
            pairs.append(sent_pairs)
    return pairs

def build_vocab(sentences, min_occurs = 1, max_vocab_size = -1, unkifier=None, preproc=None, start_symbol=BOS):
    """Building vocabulary from a set of sentences, perhaps training sentences."""
    vocab = Vocabulary(preproc)
    # vocab.index(EOS)

    # common procedure: 1) count words (preproc will be done in vocab)
    for sent in sentences:
        for w in sent: vocab.index(w)
    # 2) only preserve words occuring >= min_occurs (and guarantee that vocab size does not exceed some max)
    if max_vocab_size > 0:
        word_set = sorted([(w, c) for w, c in vocab.counts.items() if c >= min_occurs],
                          key=lambda x: x[1], reverse=True)[:max_vocab_size]
        word_set = set(w for w, c in word_set)
    else:
        word_set = set(w for w, c in vocab.counts.items() if c >= min_occurs)
    # 3) crate new vocab reading all sentences again with unkifying.
    vocab = Vocabulary(preproc, unkifier, start_symbol)

    # Guarantee that the padding token gets index 0
    # We always add `PAD` in the vocabulary, for interchaning at test time
    vocab.index(PAD)
    # Always add <unk>; added 2021/1/8 to fix a problem that occured when
    # processing subword-tokenized sentences. Since there is no unk in training
    # data, there is a (small) possibility that some token in val (or test)
    # sentences are unknown. To deal with this case, instead of defining vocab by
    # reading sentencepiece model file, unk token is added here.
    vocab.index(UNK)

    for sent in sentences:
        for w in sent:
            proced = preproc(w) if preproc else w
            if proced in word_set:
                # A bit tricky, but `proproc` is internally done in vocab, so we
                # give the original form here.
                vocab.index(w)
            else:
                vocab.index_unked(w)

    vocab.freeze()
    return vocab


def to_tensors(sentences, vocab):
    def to_id(sent):
        return torch.tensor([vocab.index_unked(w) for w in sent], dtype=torch.long)
    return [to_id(sent) for sent in sentences]


def mk_filter_by_unk_ratio(vocab, max_unk_ratio=0.0, is_tbptt=False):
    """Return a filter function that takes a sentence, and returns True if a sentence
    meet a criterion (unk ratio is below max).

    NOTE: each sentence contains a bos and an eos, which are excluded for calculating unk ratio.
    If is_tbptt is True, offset is changed to 1 (since each sentence only contains eos).
    """
    if max_unk_ratio == 0.0: return lambda sent: True
    offset = 1 if is_tbptt else 2

    def filter(sent):
        cnt = len([w for w in sent if not vocab.contains(w)])
        return cnt / (len(sent) - offset) < max_unk_ratio
    return filter

def to_tensors_with_filtering(sentences, vocab, max_unk_ratio=0.0):
    """Return a pair of lists:

    1) a list of sentences (each sentence is a seq of token ids)
    2) a list of idxs, which are removed from the original sentences
    """
    def unkify_sent(sent):
        ids = torch.zeros(len(sent), dtype=torch.long)
        cnt = 0
        for i in range(len(sent)):
            w = sent[i]
            if not vocab.contains(w):
                ids[i] = vocab.index_unked(w)
                cnt += 1
            else:
                ids[i] = vocab.index(w)
        return (ids, cnt)

    if max_unk_ratio <= 0.0:
        return to_tensors(sentences, vocab), []

    def criterion(ids, cnt):
        return (cnt / len(ids)) < max_unk_ratio

    results = [unkify_sent(sent) for sent in sentences]
    remove_idxs = [i for i, (ids, cnt) in enumerate(results) if not criterion(ids, cnt)]
    return [ids for ids, cnt in results if criterion(ids, cnt)], remove_idxs


def to_tensor_pairs(pairs, vocab):
    def to_id(sent):
        return torch.tensor([vocab.index_unked(w) for w in sent], dtype=torch.long)
    def conv_sent_pairs(sent_pairs):
        return [(to_id(s1), to_id(s2), cont_verb) for s1, s2, cont_verb in sent_pairs]
    return [conv_sent_pairs(sent_pairs) for sent_pairs in pairs]
    # def to_id(sent):
    #     return torch.tensor([vocab.index_unked(w) for w in sent], dtype=torch.long)
    # return [(to_id(s1), to_id(s2)) for s1, s2 in pairs]


def neg_examples_to_ids(negative_examples, vocab):
    def to_id(neg_token):
        return (neg_token[0], [vocab.index_unked(w) for w in neg_token[1]])
    def conv_neg_sent(neg_sent):
        #  [(0, ['is']), (3, ['himself', 'hersel'])]
        return [to_id(neg_token) for neg_token in neg_sent]
    return [conv_neg_sent(neg_sent) for neg_sent in negative_examples]


def to_tagged_tensors(sentences, vocab, tag_vocab):
    def to_id(sent):
        return torch.tensor([vocab.index_unked(w) for w in sent], dtype=torch.long)
    def to_tag_id(tags):
        return torch.tensor([tag_vocab.index_unked(w) for w in tags], dtype=torch.long)
    return [(to_id(s), to_tag_id(t)) for (s, t) in sentences]


def batches_in_buckets(sentences, batch_size):
    size = len(sentences) // 5
    for i in range(0, len(sentences), size):
        bucket = sentences[i:i+size]
        np.random.shuffle(bucket)

        for j in range(0, len(bucket), batch_size):
            yield bucket[j:j+batch_size]

