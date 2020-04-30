import collections.abc
from lxml import etree

class TreebankNode(object):
    pass

class InternalTreebankNode(TreebankNode):
    def __init__(self, label, children):
        assert isinstance(label, (str, CCGCategory))
        self.label = label

        assert isinstance(children, collections.abc.Sequence)
        assert all(isinstance(child, TreebankNode) for child in children)
        assert children
        self.children = tuple(children)

    def linearize(self):
        return "({} {})".format(
            self.label, " ".join(child.linearize() for child in self.children))

    def leaves(self):
        for child in self.children:
            yield from child.leaves()

    def convert(self, index=0):
        tree = self
        sublabels = [self.label]

        while len(tree.children) == 1 and isinstance(
                tree.children[0], InternalTreebankNode):
            tree = tree.children[0]
            sublabels.append(tree.label)

        children = []
        for child in tree.children:
            children.append(child.convert(index=index))
            index = children[-1].right

        return InternalParseNode(tuple(sublabels), children)

class LeafTreebankNode(TreebankNode):
    def __init__(self, tag, word):
        assert isinstance(tag, (str, CCGCategory))
        self.tag = tag

        assert isinstance(word, str)
        self.word = word

    def linearize(self):
        return "({} {})".format(self.tag, self.word)

    def leaves(self):
        yield self

    def convert(self, index=0):
        return LeafParseNode(index, self.tag, self.word)

class ParseNode(object):
    pass

class InternalParseNode(ParseNode):
    def __init__(self, label, children):
        assert isinstance(label, tuple)
        assert all(isinstance(sublabel, (str, CCGCategory)) for sublabel in label)
        assert label
        self.label = label

        assert isinstance(children, collections.abc.Sequence)
        assert all(isinstance(child, ParseNode) for child in children)
        assert children
        assert len(children) > 1 or isinstance(children[0], LeafParseNode)
        assert all(
            left.right == right.left
            for left, right in zip(children, children[1:]))
        self.children = tuple(children)

        self.left = children[0].left
        self.right = children[-1].right

    def leaves(self):
        for child in self.children:
            yield from child.leaves()

    def convert(self):
        children = [child.convert() for child in self.children]
        tree = InternalTreebankNode(self.label[-1], children)
        for sublabel in reversed(self.label[:-1]):
            tree = InternalTreebankNode(sublabel, [tree])
        return tree

    def enclosing(self, left, right):
        assert self.left <= left < right <= self.right
        for child in self.children:
            if isinstance(child, LeafParseNode):
                continue
            if child.left <= left < right <= child.right:
                return child.enclosing(left, right)
        return self

    def oracle_label(self, left, right):
        enclosing = self.enclosing(left, right)
        if enclosing.left == left and enclosing.right == right:
            return enclosing.label
        return ()

    def oracle_splits(self, left, right):
        return [
            child.left
            for child in self.enclosing(left, right).children
            if left < child.left < right
        ]

    def find_spans(self, f):
        """`f` is a funciton ((Node, Node)=>Boolean) where first Node is parent and
        second is a child, and judge whether child matches the condition (of f).
        """
        return self.find_spans_recur(f, None, [])

    def find_spans_recur(self, f, parent, found):
        if f(parent, self):
            found.append(self)
        for child in self.children:
            found = child.find_spans_recur(f, self, found)
        return found

    def foreach(self, f):
        f(self)
        for c in self.children: c.foreach(f)

class LeafParseNode(ParseNode):
    def __init__(self, index, tag, word):
        assert isinstance(index, int)
        assert index >= 0
        self.left = index
        self.right = index + 1

        assert isinstance(tag, (str, CCGCategory))
        self.tag = tag

        assert isinstance(word, str)
        self.word = word

    def leaves(self):
        yield self

    def convert(self):
        return LeafTreebankNode(self.tag, self.word)

    def find_spans_recur(self, f, parent=None, found=[]):
        if f(parent, self):
            found.append(self)
        return found

    def foreach(self, f):
        f(self)

def load_trees(path, strip_top=True):
    with open(path) as infile:
        tokens = infile.read().replace("(", " ( ").replace(")", " ) ").split()

    def helper(index):
        trees = []

        while index < len(tokens) and tokens[index] == "(":
            paren_count = 0
            while tokens[index] == "(":
                index += 1
                paren_count += 1

            label = tokens[index]
            index += 1

            if tokens[index] == "(":
                children, index = helper(index)
                trees.append(InternalTreebankNode(label, children))
            else:
                word = tokens[index]
                index += 1
                trees.append(LeafTreebankNode(label, word))

            while paren_count > 0:
                assert tokens[index] == ")"
                index += 1
                paren_count -= 1

        return trees, index

    trees, index = helper(0)
    assert index == len(tokens)

    if strip_top:
        for i, tree in enumerate(trees):
            if tree.label == "TOP":
                assert len(tree.children) == 1
                trees[i] = tree.children[0]

    return trees

def load_ccgs(path, old=False):
    """Read CCG parses from the output of Jigg"""

    child_key = 'child' if old else 'children'
    symbol_key = 'category' if old else 'symbol'
    form_key = 'surf' if old else 'form'

    def read_ccg(sentence):
        assert sentence.tag == "sentence"
        ccg = sentence.xpath("./ccg")[0]
        root_id = ccg.attrib["root"]
        tokens = sentence.xpath("./tokens")[0]

        cat_reader = CategoryTreeReader()

        def find_token(token_id):
            return tokens.xpath("./token[@id='{}']".format(token_id))

        def find_span(span_id):
            return ccg.xpath("./span[@id='{}']".format(span_id))

        def mk_bottomup(cur_span):
            try:
                children = cur_span.attrib[child_key].split(" ")
            except:
                children = cur_span.attrib['terminal'].split(" ")
            label = cur_span.attrib[symbol_key]
            category = cat_reader.read_category(label)
            if len(children) == 1:
                # possibly leaf node
                child_span = find_span(children[0])
                if not child_span:
                    token = find_token(children[0])[0].attrib[form_key]
                    return LeafTreebankNode(category, token)

                else:
                    child_node = mk_bottomup(child_span[0])
                    return InternalTreebankNode(category, [child_node])
            else:
                child_spans = [find_span(child)[0] for child in children]
                assert len(child_spans) == len(children)
                child_nodes = [mk_bottomup(child_span) for child_span in child_spans]
                return InternalTreebankNode(category, child_nodes)

        return mk_bottomup(find_span(root_id)[0])

    root = etree.parse(path)
    sentences = root.xpath("//sentence")

    return [read_ccg(s) for s in sentences]


class CCGCategory(object):
    pass

class ComplexCategory(CCGCategory):
    def __init__(self, slash, left, right):
        self.slash = slash
        self.left = left
        self.right = right

    def __str__(self):
        def childstr(child):
            if isinstance(child, AtomicCategory):
                return str(child)
            else:
                return "(" + str(child) + ")"
        return childstr(self.left) + self.slash + childstr(self.right)

    def nofeat_str(self):
        def childstr(child):
            if isinstance(child, AtomicCategory):
                return child.nofeat_str()
            else:
                return "(" + child.nofeat_str() + ")"
        return childstr(self.left) + self.slash + childstr(self.right)

    def isadjunct(self, soft=False):
        """If soft=True, judge after removing features."""
        if soft:
            return self.left.nofeat_str() == self.right.nofeat_str()
        else:
            return str(self.left) == str(self.right)

    def ispunc(self):
        return False

class AtomicCategory(CCGCategory):
    def __init__(self, label):
        self.label = label

    def __str__(self):
        return self.label

    def nofeat_str(self):
        brackt = self.label.find("[")
        if brackt != -1:
            return self.label[:brackt]
        else:
            return self.label

    def isadjunct(self, soft=False):
        return False

    def ispunc(self):
        return self.label == '.' or self.label == ',' or self.label == ':' or self.label == ';'

class CategoryTreeReader(object):
    """This class is not thread safe."""

    def __init__(self):
        pass

    def read_category(self, symbol):
        # self.tokens = symbol.replace("(", " ( ").replace(")", " ) ").replace(" \\", " \\ ").replace(" /", " / ").split()
        self.tokens = symbol.replace("(", " ( ").replace(")", " ) ").replace("\\", " \\ ").replace("/", " / ").split()

        category, index = self.read_category_helper(0)
        assert index == len(self.tokens)

        return category

    def read_target(self, index):
        cur = self.tokens[index]
        if cur == "(":
            category, index = self.read_category_helper(index + 1)
            assert self.tokens[index] == ")"
            return category, index + 1
        else: # cur is an atomic category, e.g., NP
            return AtomicCategory(cur), index + 1

    def read_category_helper(self, index):
        # cur = self.tokens[index]
        target_tree, index = self.read_target(index)

        if index >= len(self.tokens) or not self.is_slash(index):
            return target_tree, index

        slash = self.tokens[index]
        assert(slash == "/" or slash == "\\")
        index += 1
        argument_tree, index = self.read_category_helper(index)

        return ComplexCategory(slash, target_tree, argument_tree), index

    def is_slash(self, index):
        return self.tokens[index] == "/" or self.tokens[index] == "\\"
