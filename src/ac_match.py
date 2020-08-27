# encoding=utf8
"""
ac自动机匹配
"""
import ahocorasick
from src.util import load_dir_util


class ACMatch:

    def __init__(self, path):
        self.A = ahocorasick.Automaton()
        for item in load_dir_util.load_data_from_dir(path):
            if len(item) == 3:
                self.A.add_word(item[0], "\t".join(item))
        self.A.make_automaton()

    def ac_match(self, q):
        l = []
        for item in self.A.iter(q):
            if len(l) > 0 and l[-1].split('\t')[0] in item[1].split('\t')[0]:
                l[-1] = item[1]
            else:
                l.append(item[1])
        d = {}
        for ll in l:
            tts = ll.split('\t')
            d[tts[0]] = tts + [1.0]
        return d


if __name__ == "__main__":
    s = ACMatch('../../data/entity')
    for e in s.ac_match('上海天气怎么样'):
        print(e)
