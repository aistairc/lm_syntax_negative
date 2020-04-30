
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

    batch_size = 200

    rnn, optimizer = train_lm.load_model(args.model, device)
    vocab = rnn.vocab
    pad_id = vocab.index(data.PAD)

    ev = evaluator.SentenceEvaluator(pad_id)

    sents = data.read_sentences(args.data, False, False, vocab.start_symbol)
    if args.ignore_eos:
        assert sents[0][-2] == sents[0][-1] == data.EOS
        sents = [s[:-1] for s in sents] # remove dupicated EOS

    tensors = data.to_tensors(sents, vocab)
    batch_gen = batch_generator.SentenceBatchGenerator(tensors, batch_size, pad_id)

    print('First sentences (after conversion):')
    for s in tensors[:3]:
        print(' '.join([vocab.value(t) for t in s]))

    def open_for_w(fn):
        if fn.endswith('.gz'):
            return gzip.open(fn, 'wt')
        else:
            return open(fn, 'w')

    with open_for_w(args.output) as o:
        total_batchs = len(tensors) // batch_size
        for batch_i, i in enumerate(range(0, len(tensors), batch_size)):
            if batch_i > 0 and batch_i % 100 == 0:
                logger.info("{}/{} batches processed.".format(batch_i, total_batchs))
            sources = tensors[i:i+batch_size]
            surps, _ = ev.word_stats(rnn, sources, pad_id)

            for sent_surps, sent_ents in zip(surps, entropys):
                sent = sents[sent_i][1:] # remove begin of sentence
                assert len(sent) == len(sent_surps) == len(sent_ents)
                for j, word in enumerate(sent[:-1]): # ignore eos for evaluation
                    o.write(report(word, sent_i, j, sent_surps, sent_ents))
                sent_i += 1

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Calculate statistics of each word such as surprisals.')

    parser.add_argument('--ignore-eos', action='store_true',
                        help='Set True for text that already has <eos> for each end of sentence.')
    parser.add_argument('--output', default='rnn.output')
    parser.add_argument('--data', default='data/penn/test.txt')
    parser.add_argument('--model', default='model.pt')
    parser.add_argument('--gpu', type=int, default=None)

    args = parser.parse_args()

    if args.gpu and args.gpu >= 0:
        device = torch.device("cuda:{}".format(args.gpu))
    else:
        device = torch.device("cuda:{}".format(utils.find_idle_gpu()) if torch.cuda.is_available() else "cpu")

    logger.info('device: {}'.format(device))

    run_test(args, device)
