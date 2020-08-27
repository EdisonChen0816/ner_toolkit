# encoding=utf-8
import collections
import unicodedata
import tensorflow as tf


def convert_to_unicode(text):
    if isinstance(text, str):
        return text
    elif isinstance(text, bytes):
        return text.decode('utf-8', 'ignore')
    elif isinstance(text, list):
        return text
    else:
        raise ValueError('unsupported type')


def load_vocab(vocab_file):
    vocab = collections.OrderedDict()
    index = 0
    with tf.gfile.GFile(vocab_file, 'r') as reader:
        while True:
            token = convert_to_unicode(reader.readline())
            if not token:
                break
            vocab[token.strip()] = index
            index += 1
    return vocab


def _is_whitespace(char):
    return char == ' ' or char == '\t' or char == '\n' or char == '\r' or unicodedata.category(char) == 'Zs'


def _is_control(char):
    return unicodedata.category(char).startswith('C') and char != '\t' and char != '\n' and char != '\r'


def _is_punctuation(char):
    cp = ord(char)
    return (cp >= 33 and cp <= 47) or (cp >= 58 and cp <= 64) or \
           (cp >= 91 and cp <= 96) or (cp >= 123 and cp <= 126) or\
           unicodedata.category(char).startswith('P')


def whitespace_tokenize(text):
    text = text.strip()
    if not text:
        return []
    return text.split()


class BasicTokenizer(object):

    def __init__(self, do_lower_case=True):
        self.do_lower_case = do_lower_case

    def tokenize(self, text):
        text = convert_to_unicode(text)
        text = self._clean_text(text)
        text = self._tokenize_chinese_chars(text)
        orig_tokens = whitespace_tokenize(text)
        split_tokens = []
        for token in orig_tokens:
            if self.do_lower_case:
                token = token.lower()
                token = self._run_strip_accents(token)
            split_tokens.extend(self._run_split_on_punc(token))
        output_tokens = whitespace_tokenize(' '.join(split_tokens))
        return output_tokens

    def _clean_text(self, text):
        output = []
        for char in text:
            if ord(char) in (0, 0xfffd) or _is_control(char):
                continue
            if _is_whitespace(char):
                output.append(' ')
            else:
                output.append(char)
        return ''.join(output)

    def _tokenize_chinese_chars(self, text):
        output = []
        for char in text:
            if self._is_chinese_char(ord(char)):
                output.extend([' ',char,' '])
            else:
                output.append(char)
        return ''.join(output)

    def _is_chinese_char(self, cp):
        return (cp >= 0x20000 and cp <= 0x2A6DF) or\
               (cp >= 0x2A700 and cp <= 0x2B73F) or\
               (cp >= 0x2B740 and cp <= 0x2B81F) or\
               (cp >= 0x2B820 and cp <= 0x2CEAF) or\
               (cp >= 0x2F800 and cp <= 0x2FA1F) or\
               (cp >= 0x3400 and cp <= 0x4DBF) or\
               (cp >= 0x4E00 and cp <= 0x9FFF) or\
               (cp >= 0xF900 and cp <= 0xFAFF)

    def _run_strip_accents(self, text):
        text = unicodedata.normalize('NFD', text)
        output = []
        for char in text:
            cat = unicodedata.category(char)
            if cat == 'Mn':
                continue
            output.append(char)
        return ''.join(output)

    def _run_split_on_punc(self, text):
        chars = list(text)
        i = 0
        start = True
        output = []
        while i < len(chars):
            if _is_punctuation(chars[i]):
                output.append([chars[i]])
                start = True
            else:
                if start:
                    output.append([])
                output[-1].append(chars[i])
                start = False
            i += 1
        return [''.join(x) for x in output]


class WordpieceTokenizer(object):

    def __init__(self, vocab, unk_token='[UNK]', max_input_chars_per_word=200):
        self.vocab = vocab
        self.unk_token = unk_token
        self.max_input_chars_per_word = max_input_chars_per_word

    def tokenize(self, text):
        text = convert_to_unicode(text)
        output_tokens = []
        for token in whitespace_tokenize(text):
            chars = list(token)
            if len(chars) > self.max_input_chars_per_word:
                output_tokens.append(self.unk_token)
                continue
            is_bad = False
            start = 0
            sub_tokens = []
            while start < len(chars):
                end = len(chars)
                cur_substr = None
                while start < end:
                    substr = ''.join(chars[start:end])
                    if start > 0:
                        substr = '##'+substr
                    if substr in self.vocab:
                        cur_substr = substr
                        break
                    end -= 1
                if cur_substr is None:
                    is_bad = True
                    break
                sub_tokens.append(cur_substr)
                start = end
            output_tokens.extend([self.unk_token] if is_bad else sub_tokens)
        return output_tokens


class FullTokenizer(object):

    def __init__(self, vocab_file, do_lower_case=True):
        self.vocab = load_vocab(vocab_file)
        self.inv_vocab = {v: k for k, v in self.vocab.items()}
        self.basic_tokenizer = BasicTokenizer()
        self.wordpiece_tokenizer = WordpieceTokenizer(vocab=self.vocab)

    def tokenize(self, text):
        split_tokens = []
        for token in self.basic_tokenizer.tokenize(text):
            for sub_token in self.wordpiece_tokenizer.tokenize(token):
                split_tokens.append(sub_token)
        return split_tokens

    def convert_tokens_to_ids(self, tokens):
        return [self.vocab[item] for item in tokens]

    def convert_ids_to_tokens(self, ids):
        return [self.inv_vocab[item] for item in ids]