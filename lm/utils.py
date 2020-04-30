from collections import defaultdict
import inflect
import math
import re
import subprocess

import torch

import data
import unkifier
import word_processor

def mk_unkifier(unkify):
    if unkify == 'choe_charniak':
        return unkifier.ChoeCharniakUnkifier()
    elif unkifier == 'pos':
        return unkifier.POSUnkifier()
    else:
        return unkifier.ConstUnkifier(data.UNK)

def mk_preproc(preproc):
    def to_processor(v):
        if v == 'lower': return word_processor.Lower()
        elif v == 'delnum': return word_processor.DeleteNumber()
        else: raise ValueError("Unknown --preproc value: {}".format(v))

    items = [to_processor(v) for v in preproc]
    if len(items) == 1:
        return items[0]
    else:
        return word_processor.Combinator(items)


def repackage_hidden(h):
    """Wraps hidden states in new Tensors,
    to detach them from their history."""
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)


def batchify(data, bsz, batch_first=False):
    assert isinstance(data, list)
    data = torch.cat(data)
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    if batch_first:
        data = data.view(bsz, -1) # batch_first
    else:
        data = data.view(bsz, -1).t().contiguous()
    return data


def get_batch(source, i, bptt, seq_len=None, evaluation=False, batch_first=False):
    if batch_first:
        seq_len = min(seq_len if seq_len else bptt, source.size(1) - 1 - i)
        data = source[:, i:i+seq_len]
        target = source[:, i+1:i+1+seq_len].contiguous().view(-1)
        return data, target
    else:
        seq_len = min(seq_len if seq_len else bptt, len(source) - 1 - i)
        data = source[i:i+seq_len]
        target = source[i+1:i+1+seq_len].view(-1)
        return data, target

def get_sorted_sentences_batch(source, i, batch_size, padding_value, sorted=False):
    sentences = source[i:i+batch_size]

    # We later drop the last token <eos> from the input, so we subtract 1 here.
    lengths = torch.tensor([len(s)-1 for s in sentences])
    if not sorted:
        lengths, perm_idx = lengths.sort(0, descending=True)
        sentences = [sentences[i] for i in perm_idx]

    padded = torch.nn.utils.rnn.pad_sequence(
        sentences, batch_first=True, padding_value = padding_value)

    # targets can be obtained by ignoring all first tokens <eos> of the sentences.
    targets = padded[:, 1:].contiguous().view(-1)

    padded = padded[:, :-1]

    return padded, lengths, targets

def get_sorted_sentences_with_agreement_batch(
        source, i, batch_size, padding_value, sorted=False):
    sentences_with_ag = source[i:i+batch_size]

    lengths = torch.tensor([len(s)-1 for (s, a) in sentences_with_ag])
    if not sorted:
        lengths, perm_idx = lengths.sort(0, descending=True)
        sentences_with_ag = [sentences_with_ag[i] for i in perm_idx]
    sentences = [s for s, _ in sentences_with_ag]
    agreements = [a for _, a in sentences_with_ag]

    padded = torch.nn.utils.rnn.pad_sequence(
        sentences, batch_first=True, padding_value = padding_value)

    # targets can be obtained by ignoring all first tokens <eos> of the sentences.
    targets = padded[:, 1:].contiguous().view(-1)

    padded = padded[:, :-1]

    return padded, lengths, targets, agreements

def get_sorted_tagged_sentences_batch(
        tagged_source, i, batch_size, padding_value, sorted=False):
    sentences = tagged_source[i:i+batch_size]

    # Tag mode does not predict the tag of the last token <eos>.
    lengths = torch.tensor([len(s)-1 for s, tags in sentences])
    if not sorted:
        lengths, perm_idx = lengths.sort(0, descending=True)
        sentences = [sentences[i] for i in perm_idx]

    tokens, targets = zip(*sentences)
    tokens = torch.nn.utils.rnn.pad_sequence(
        tokens, batch_first=True, padding_value = padding_value)
    tokens = tokens[:, :-1]

    targets = torch.nn.utils.rnn.pad_sequence(
        targets, batch_first=True, padding_value = padding_value)
    targets = targets[:, :-1].contiguous().view(-1)

    return tokens, lengths, targets

def transform_agreement_pairs_to_final_binary_prediction(pairs, vocab):
    infl = inflect.engine()
    for pairs_for_sent in pairs:
        for gold, wrong, is_simple in pairs_for_sent:
            last = gold[-1]
            token = vocab.value(last)
            # last_token should not be unk; unk should be ignored beforehand.
            is_plural = token == 'themselves' or (infl.plural_verb(token) == token)
            gold[-1] = 1 if is_plural else 0
    return pairs

def find_idle_gpu():
    o = subprocess.check_output('nvidia-smi').decode('utf-8')
    o = o.split('\n')
    start = '|  GPU       PID   Type   Process name'

    avail_info = False
    gpus = defaultdict(int)
    for line in o:
        if not avail_info:
            if line.startswith(start):
                avail_info = True
                continue
            if re.match('^\|\s+\d', line):
                max_gpu = int([a for a in line[1:-1].split(' ') if a != ''][0])
        if avail_info:
            if not re.match('^\|\s+\d', line):
                continue
            vs = [a for a in line[1:-1].split(' ') if a != '']
            num = int(vs[0])
            mem = int(vs[4][:-3])
            gpus[num] += mem

    mems = [0 for _ in range(max_gpu + 1)]
    for gpu, mem in gpus.items():
        mems[gpu] += mem

    mems = [(m, i) for i, m in enumerate(mems)]
    mems = sorted(mems)
    return mems[0][1]

def log1mexp(x, expm1_guard = 1e-7):
    # (6) in https://cran.r-project.org/web/packages/Rmpfr/vignettes/log1mexp-note.pdf
    assert(all(x < .0))

    t = x < math.log(0.5)
    y = torch.zeros_like(x)
    y[t] = torch.log1p(-x[t].exp())

    # for x close to 0 we need expm1 for numerically stable computation
    # we furtmermore modify the backward pass to avoid instable gradients,
    # ie situations where the incoming output gradient is close to 0 and the gradient of expm1 is very large
    expxm1 = torch.expm1(x[~t])
    log1mexp_fw = (-expxm1).log()
    log1mexp_bw = (-expxm1+expm1_guard).log() # limits magnitude of gradient

    y[~t] = log1mexp_fw.detach() + (log1mexp_bw - log1mexp_bw.detach())
    return y
