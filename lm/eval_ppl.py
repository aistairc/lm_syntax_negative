
import argparse
import gzip
import logging
import math
import torch

import batch_generator
import data
import evaluator
import train_lm
import utils

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s:%(name)s:%(levelname)s: %(message)s')

def eval_ppl(args, device):

    batch_size = 200

    rnn, optimizer = train_lm.load_model(args.model, device)
    vocab = rnn.vocab
    pad_id = vocab.index(data.PAD)

    exclude = []
    if args.exclude_begin:
        exclude.append(0)
    if args.exclude_eos:
        exclude.append(-1)
    if args.exclude_last:
        exclude.append(-2)

    ev = evaluator.SentenceEvaluator(pad_id, exclude=exclude)

    sents = data.read_sentences(args.data, False, False, vocab.start_symbol)

    tensors = data.to_tensors(sents, vocab)
    batch_gen = batch_generator.SentenceBatchGenerator(tensors, batch_size, pad_id)
    total_loss, avg_loss, _, _ = ev.evaluate(rnn, batch_gen)

    print('perplexity: {}'.format(math.exp(avg_loss)))
    print('loss: {}'.format(total_loss))

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Calculate perplexity on the test set.')

    parser.add_argument('--exclude-begin', action='store_true',
                        help='Exclude the first token of each sentence.')
    parser.add_argument('--exclude-last', action='store_true',
                        help='Exclude the last token before eos')
    parser.add_argument('--exclude-eos', action='store_true')

    parser.add_argument('--data', default='data/penn/test.txt')
    parser.add_argument('--model', default='model.pt')
    parser.add_argument('--gpu', type=int, default=None)

    args = parser.parse_args()

    if args.gpu and args.gpu >= 0:
        device = torch.device("cuda:{}".format(args.gpu))
    else:
        device = torch.device("cuda:{}".format(utils.find_idle_gpu()) if torch.cuda.is_available() else "cpu")

    eval_ppl(args, device)
