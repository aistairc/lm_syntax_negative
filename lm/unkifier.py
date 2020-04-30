
class Unkifier(object):
    """Unkifier defines `unkify`, which receives a word that is not found on vocabulary,
    and returns the corresponding unked word.

    Some `unkify` may use POS tag, 
    """
    def unkify(word, pos = None):
        return word

class ConstUnkifier(Unkifier):
    def __init__(self, unk):
        self.unk = unk
    def unkify(self, word, pos = None):
        return self.unk

class POSUnkifier(Unkifier):
    def unkify(self, word, pos):
        return pos

class ChoeCharniakUnkifier(Unkifier):
    def unkify(self, ws, pos = None):
        uk = "unk"
        sz = len(ws) - 1
        if ws[0].isupper():
            uk = "c" + uk
        if ws[0].isdigit() and ws[sz].isdigit():
            uk = uk + "n"
        elif sz <= 2:
            pass
        elif ws[sz-2:sz+1] == "ing":
            uk = uk + "ing"
        elif ws[sz-1:sz+1] == "ed":
            uk = uk + "ed"
        elif ws[sz-1:sz+1] == "ly":
            uk = uk + "ly"
        elif ws[sz] == "s":
            uk = uk + "s"
        elif ws[sz-2:sz+1] == "est":
            uk = uk + "est"
        elif ws[sz-1:sz+1] == "er":
            uk = uk + 'ER'
        elif ws[sz-2:sz+1] == "ion":
            uk = uk + "ion"
        elif ws[sz-2:sz+1] == "ory":
            uk = uk + "ory"
        elif ws[0:2] == "un":
            uk = "un" + uk
        elif ws[sz-1:sz+1] == "al":
            uk = uk + "al"
        else:
            for i in range(sz):
                if ws[i] == '-':
                    uk = uk + "-"
                    break
                elif ws[i] == '.':
                    uk = uk + "."
                    break
        return "<" + uk + ">"

