
import argparse
import gzip
import logging
import torch

import batch_generator
import data
import evaluator
import train_lm
import utils

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s:%(name)s:%(levelname)s: %(message)s')

def run_test(args, device):

    batch_size = args.batch_size

    rnn, optimizer = train_lm.load_model(args.model, device)
    vocab = rnn.vocab
    pad_id = vocab.index(data.PAD)

    ev = evaluator.SentenceEvaluator(pad_id)

    sents = data.read_sentences(args.data, False, False, vocab.start_symbol, args.sentence_piece_model)
    if args.ignore_eos:
        assert sents[0][-2] == sents[0][-1] == data.EOS
        sents = [s[:-1] for s in sents] # remove dupicated EOS
    if args.capitalize:
        for sent in sents:
            sent[1] = sent[1].capitalize()

    tensors = data.to_tensors(sents, vocab)
    batch_gen = batch_generator.SentenceBatchGenerator(tensors, batch_size, pad_id)
    calc_entropy = not args.no_entropy
    if calc_entropy:
        logger.info('calc entropy')
    else:
        logger.info('not calc entropy')

    print('First sentences (after conversion):')
    for s in tensors[:3]:
        print(' '.join([vocab.value(t) for t in s]))

    def open_for_w(fn):
        if fn.endswith('.gz'):
            return gzip.open(fn, 'wt')
        else:
            return open(fn, 'w')

    def resolve_subwords(pieces, piece_surps, piece_ents):
        word_sent = []
        sent_surps = []
        sent_ents = []
        is_intermediate = False
        for i, piece in enumerate(pieces):
            if is_intermediate:
                if '▁' in piece:
                    word_sent[-1] += piece[:-1]
                    is_intermediate = False
                else:
                    word_sent[-1] += piece
                sent_surps[-1] += piece_surps[i]
            else:
                if '▁' in piece:
                    word_sent.append(piece[:-1])
                else:
                    word_sent.append(piece)
                    is_intermediate = True
                sent_surps.append(piece_surps[i])
                sent_ents.append(piece_ents[i])

        return word_sent, sent_surps, sent_ents

    with open_for_w(args.output) as o:
        if args.internal_token:
            o.write("word processed sentid sentpos wlen surp entropy\n")

            def report(word, sent_i, j, sent_surps, sent_ents):
                # transform-to-id-then-detransform results in the internal string rep.
                conved = vocab.value(vocab.index_unked(word))
                return '{} {} {} {} {} {} {}\n'.format(
                    word, conved, sent_i, j, len(word), sent_surps[j], sent_ents[j])
        else:
            o.write("word sentid sentpos wlen surp entropy\n")

            def report(word, sent_i, j, sent_surps, sent_ents):
                return '{} {} {} {} {} {}\n'.format(
                    word, sent_i, j, len(word), sent_surps[j], sent_ents[j])
        o.write("\n")

        sent_i = 0

        total_batchs = len(tensors) // batch_size
        for batch_i, i in enumerate(range(0, len(tensors), batch_size)):
            if batch_i > 0 and batch_i % 100 == 0:
                logger.info("{}/{} batches processed.".format(batch_i, total_batchs))
            sources = tensors[i:i+batch_size]
            with torch.no_grad():
                surps, entropys = ev.word_stats(rnn, sources, pad_id, calc_entropy=calc_entropy)

            for sent_surps, sent_ents in zip(surps, entropys):
                sent = sents[sent_i][1:] # remove begin of sentence
                if args.sentence_piece_model is not None:
                    sent, sent_surps, sent_ents = resolve_subwords(sent, sent_surps, sent_ents)
                assert len(sent) == len(sent_surps) == len(sent_ents)
                for j, word in enumerate(sent[:-1]): # ignore eos for evaluation
                    o.write(report(word, sent_i, j, sent_surps, sent_ents))
                sent_i += 1

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Calculate statistics of each word such as surprisals.')

    parser.add_argument('--ignore-eos', action='store_true',
                        help='Set True for text that already has <eos> for each end of sentence.')
    parser.add_argument('--internal-token', action='store_true',
                        help='If true, add column for preprocessed token (internally used) by a model-specific vocabulary.')
    parser.add_argument('--output', default='rnn.output')
    parser.add_argument('--data', default='data/penn/test.txt')
    parser.add_argument('--model', default='model.pt')
    parser.add_argument('--gpu', type=int, default=None)
    parser.add_argument('--batch-size', type=int, default=50)
    parser.add_argument('--no-entropy', action='store_true')
    parser.add_argument('--capitalize', action='store_true')
    parser.add_argument('--sentence_piece_model', default=None)

    args = parser.parse_args()

    if args.gpu is not None and args.gpu >= 0:
        device = torch.device("cuda:{}".format(args.gpu))
    else:
        device = torch.device("cuda:{}".format(utils.find_idle_gpu()) if torch.cuda.is_available() else "cpu")

    logger.info('device: {}'.format(device))

    run_test(args, device)
