
import argparse
import corenlp
import gzip
import inflect
from tqdm import tqdm

def open_f(fn, mode='rt'):
        if fn.endswith('.gz'):
            return gzip.open(fn, mode)
        else:
            return open(fn, mode)

class VerbFinder(object):
    def __init__(self):
        # each position is a pair (idx, simple or not)
        # simple means the last word is an (agreed) noun
        self.vbz_positions = [] # singular
        self.vbp_positions = [] # plural

        self.vbz_set = set() # all vbz appearing in the corpus
        self.vbp_set = set() # all vbp appearing in the corpus

        self.infl = inflect.engine()

    def get_converter(self):
        conv = {}
        for vbz in self.vbz_set:
            vbp = self.infl.plural_verb(vbz)
            if vbp in self.vbp_set:
                conv[vbz] = vbp
                conv[vbp] = vbz
        return conv

    def find_all_verbs(self, fn):
        props = {"tokenize.whitespace": "true",
                 "ssplit.eolonly": "true",
                 "tokenize.options": "\"normalizeParentheses=true,normalizeOtherBrackets=true\""}

        num_lines = sum(1 for line in open(fn, 'r'))
        with corenlp.CoreNLPClient(annotators="tokenize ssplit pos".split(),
                                   properties=props) as client, \
                                   open_f(args.source) as source:
            # To reduce network overhead we call corenlp on every chunk of 100 sentences.
            sents = []
            chunk_size = 100
            for line in tqdm(source, total=num_lines):
                if len(sents) >= chunk_size:
                    ann = client.annotate('\n'.join(sents))
                    assert(len(ann.sentence) == chunk_size)
                    self.record_positions(ann)
                    sents = []
                sents.append(line[:-1])
            if sents:
                ann = client.annotate('\n'.join(sents))
                self.record_positions(ann)

    def record_positions(self, annotation):
        sents = annotation.sentence
        for sent in sents:
            poses = [t.pos for t in sent.token]
            words = [t.word.lower() for t in sent.token]

            def is_singular_noun(p):
                return p == 'NN' or p == 'NNP'
            def is_plural_noun(p):
                return p == 'NNS' or p == 'NNPS'
            def is_third_pronoun(w):
                return w == 'he' or w == 'she' or w == 'it' or w == 'this'
            def is_nonthird_pronoun(w):
                return w == 'we' or w == 'they' or w == 'all' or w == 'i' or w == 'you'

            def simple_vbz(i):
                return i > 0 and (is_singular_noun(poses[i-1]) or is_third_pronoun(words[i-1]))
            def simple_vbp(i):
                return i > 0 and (is_plural_noun(poses[i-1]) or is_nonthird_pronoun(words[i-1]))

            vbz_idx = [(i, simple_vbz(i)) for i, p in enumerate(poses) if p == 'VBZ']
            vbp_idx = [(i, simple_vbp(i)) for i, p in enumerate(poses) if p == 'VBP']

            self.vbz_positions.append(vbz_idx)
            self.vbp_positions.append(vbp_idx)

            for idx, simple in vbz_idx: self.vbz_set.add(sent.token[idx].word)
            for idx, simple in vbp_idx: self.vbp_set.add(sent.token[idx].word)

def run(args):
    verb_finder = VerbFinder()
    verb_finder.find_all_verbs(args.source)

    conv = verb_finder.get_converter()
    vbz_positions = verb_finder.vbz_positions
    vbp_positions = verb_finder.vbp_positions

    def filter_cands(sent, positions):
        items = [(idx, conv.get(sent[idx]), simple) for (idx, simple) in positions]
        return [item for item in items if item[1] and item[1] != sent[item[0]]]

    with open_f(args.source) as source, open_f(args.output, 'wt') as target:
        for i, line in enumerate(source):
            sent = line[:-1].split()

            vbz = vbz_positions[i]
            vbp = vbp_positions[i]
            vbz = filter_cands(sent, vbz)
            vbp = filter_cands(sent, vbp)

            examples = sorted(vbz + vbp, key=lambda x: x[0])
            line = '\t'.join("{} {} {}".format(e[0], e[1], e[2]) for e in examples)
            target.write('{} {}'.format(i, line))
            target.write('\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Generate negative examples for LM agreement task.')

    parser.add_argument('--source', required=True, type=str)
    parser.add_argument('--output', default='verb_negative_examples.txt')

    args = parser.parse_args()
    run(args)
