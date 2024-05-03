class Tree(object):
    def __init__(self):
        self._depth = None
        self._size = None
        self.parent = None
        self.num_children = 0
        self.children = list()
        self.op = 0  #
        self.value = ""  #
        self.opname = ""  #
        self.output = None
        self.idx = -1

    def add_child(self, child):
        child.parent = self
        self.num_children += 1
        self.children.append(child)

    def size(self):
        try:
            if getattr(self, '_size'):
                return self._size
        except AttributeError as e:
            count = 1
            for i in range(self.num_children):
                count += self.children[i].size()
            self._size = count
            return self._size

    def depth(self):
        if getattr(self, '_depth'):
            return self._depth
        count = 0
        if self.num_children > 0:
            for i in range(self.num_children):
                child_depth = self.children[i].depth()
                if child_depth > count:
                    count = child_depth
            count += 1
        self._depth = count
        return self._depth

    def get_children(self):
        return self.children

    def dump(self, indent=0):
        tab = '    ' * (indent - 1) + ' |- ' if indent > 0 else ''
        print('%s%s' % (tab, self.opname), '----', self.op)
        for i in range(len(self.children)):
            self.children[i].dump(indent + 1)

    def __str__(self):
        return self.opname