
import argparse
import gzip
import inflect
import itertools

def open_f(fn, mode='rt'):
        if fn.endswith('.gz'):
            return gzip.open(fn, mode)
        else:
            return open(fn, mode)

class ReflexiveFinder(object):
    def __init__(self):
        # each position is a pair (idx, simple or not)
        # All reflexives are not simple. So the second is always False.
        self.single_positions = []
        self.plural_positions = []

        # self.vbz_set = set() # all vbz appearing in the corpus
        # self.vbp_set = set() # all vbp appearing in the corpus

        # self.infl = inflect.engine()

    def find_all_reflexives(self, fn):
        with open_f(args.source) as source:
            for line in source:
                sent = line[:-1].split(' ')
                self.record_positions(sent)

    def record_positions(self, sent):
        words = [t.lower() for t in sent]
        single_idx = [(i, False) for i, w in enumerate(words)
                      if w == 'himself' or w == 'herself']
        plural_idx = [(i, False) for i, w in enumerate(words)
                      if w == 'themselves']
        self.single_positions.append(single_idx)
        self.plural_positions.append(plural_idx)

        # for idx, simple in vbz_idx: self.vbz_set.add(sent.token[idx].word)
        # for idx, simple in vbp_idx: self.vbp_set.add(sent.token[idx].word)

def run(args):
    reflexive_finder = ReflexiveFinder()
    reflexive_finder.find_all_reflexives(args.source)

    single_positions = reflexive_finder.single_positions
    plural_positions = reflexive_finder.plural_positions

    def to_examples(sent, positions, is_single = True):
        def example_list(p):
            if is_single:
                # convert to themselves
                return [(p[0], 'themselves', p[1])]
            else:
                # both are incorrect
                return [(p[0], 'himself', p[1]), (p[0], 'herself', p[1])]
        items = [example_list(p) for p in positions]
        return list(itertools.chain(*items))  # flatten

    with open_f(args.source) as source, open_f(args.output, 'wt') as target:
        for i, line in enumerate(source):
            sent = line[:-1].split()

            single = to_examples(sent, single_positions[i], True)
            plural = to_examples(sent, plural_positions[i], False)

            examples = sorted(single + plural, key=lambda x: x[0])
            line = '\t'.join("{} {} {}".format(e[0], e[1], e[2]) for e in examples)
            target.write('{} {}'.format(i, line))
            target.write('\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Generate negative examples for LM agreement task.')

    parser.add_argument('--source', required=True, type=str)
    parser.add_argument('--output', default='reflexive_negative_examples.txt')

    args = parser.parse_args()
    run(args)
