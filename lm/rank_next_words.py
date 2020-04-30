
import argparse
import gzip
import logging
import torch

import data
import train_lm
import utils

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s:%(name)s:%(levelname)s: %(message)s')

def run(args, device):

    model, optimizer = train_lm.load_model(args.model)
    vocab = model.vocab

    bos = vocab.start_symbol
    pad_id = vocab.index(data.PAD)

    model.eval()

    with torch.no_grad():
        while True:
            prefix = input('> ')
            # add a dummy final word to get prediction after the "given" final word
            prefix = [bos] + [w.strip() for w in prefix.split(' ')] + [bos]

            prefix_tensor = data.to_tensors([prefix], vocab)  # (1, len)

            sources, lengths, targets = utils.get_sorted_sentences_batch(prefix_tensor, 0, 1, pad_id)

            sources = sources.to(device)
            lengths = lengths.to(device)
            targets = targets.to(device)

            output, _ = model.rnn(sources, input_lengths = lengths)

            dist = model.log_dist(output[-1])
            probs, words = dist.topk(100)

            probs = probs.cpu().numpy()
            words = words.cpu().numpy()
            for p, w in zip(probs, words):
                print('{}\t{}'.format(vocab.value(w), p))

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Calculate statistics of each word such as surprisals.')

    parser.add_argument('--model', default='model.pt')

    args = parser.parse_args()

    device = torch.device("cuda:{}".format(utils.find_idle_gpu()) if torch.cuda.is_available() else "cpu")

    logger.info('device: {}'.format(device))

    run(args, device)
