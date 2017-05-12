from OpenGL.GL import GLfloat
from os.path import join, splitext
from os import getcwd

class DictionaryOfLists(dict):
    def __getitem__(self, key):
        try:
            return super().__getitem__(key)
        except KeyError:
            super().__setitem__(key, [])
            return self[key]

    def __setitem__(self, key, value):
        raise NotImplementedError()

def mid(a, b, c):
    return sorted([a, b, c])[1]

def dict_get_value(dick, key, default):
    if key in dick:
        return dick[key]
    return default

# def get_data(list_):
#     array_type = GLfloat * len(list_)
#     return array_type(*list_)

def change_extension(path, ext):
    # ext should include '.'
    left, right = splitext(path)
    return left + ext

def read_file(filename):
    with open(join(getcwd(), filename), 'r') as f:
        return f.read()

def write_file(filename, data):
    with open(join(getcwd(), filename), 'w') as f:
        f.write(data)

def find_with_name(items, name):
    for i in items:
        if i.name == name:
            return i

class GenericDictionary:
    # this is pretty cool but not using
    def __init__(self):
        self.generic = {}  # indexed by cls

    def _values(self, cls):
        try:
            return self.generic[cls].values()
        except KeyError:
            return []

    def __getitem__(self, tup):
        try:
            cls, key = tup
            dic = self.generic[cls]
            return dic[key]
        except TypeError:
            return self._values(tup)  # tup is cls
            # could just return self.generic, would be more normal

    def __setitem__(self, tup, value):
        try:
            cls, key = tup
            if not cls in self.generic:
                self.generic[cls] = {}  # the normal dictionary
            dic = self.generic[cls]
            dic[key] = value
        except TypeError:
            self.generic[tup] = value  # tup is cls

    def __contains__(self, tup):
        cls, key = tup
        return cls in self.generic and key in self.generic[cls]
        # except TypeError:
        #     return tup in self.generic
