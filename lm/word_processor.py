
class WordProcessor(object):
    pass

class Keep(WordProcessor):
    def __call__(self, word):
        return word

class Lower(WordProcessor):
    def __call__(self, word):
        return word.lower()

class DeleteNumber(WordProcessor):
    def __call__(self, word):
        if word[0].isdigit() and word[-1].isdigit():
            return 'NUM'
        else:
            return word

class Combinator(WordProcessor):
    def __init__(self, processors):
        assert all(isinstance(x, WordProcessor) for x in processors)
        self.processors = processors

    def __call__(self, word):
        for proc in self.processors:
            word = proc(word)
        return word

