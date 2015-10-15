from __future__ import division
import getopt
import itertools
import nltk
import os.path
import pickle
import random
import sys
from collections import Counter, defaultdict
from curses.ascii import isdigit
from itertools import islice
from nltk.corpus import cmudict 
from nltk.tokenize import RegexpTokenizer


VOWELS = "AEIOU"
CONSONANTS = "BCDFGHJKLMNPQRSTVWXYZ"
phonedict = cmudict.dict()
cmuwords = cmudict.words()


class TextHandler:
    def __init__(self, order, files, rhyme_file, scary_file):
        self.order = order
        self.files = files
        self.scary_words = self.read_scary_words(scary_file)
        self.rhyme_dict = self.load_rhyme_dict(rhyme_file)

    def get_matrix(self):        
        content_text = self.merge_text(self.files)
        tokenizer = RegexpTokenizer(r'[\w\']+')
        corpus_words = tokenizer.tokenize(content_text)
        reverse_matrix = self.create_reverse_matrix(corpus_words)
        return reverse_matrix

    def merge_text(self, files):
        """Merge corpus texts"""
        text = ""
        for filename in files:
            with open(filename, "rb") as f:
                content = f.read()
                unicode_content = unicode(content, "utf-8")
                text = " ".join((text, unicode_content))
        return text

    def window_generator(self, seq, n=2):
        it = iter(seq)
        result = tuple(islice(it, n))
        if len(result) == n:
            yield result    
        for elem in it:
            result = result[1:] + (elem,)
            yield result

    def create_reverse_matrix(self, text):
        text = reversed(text)
        tokens = self.window_generator(text, self.order + 1)
        matrix = {}

        for sequence in tokens:
            current = sequence[:-1]
            succ = sequence[-1]

            if current not in matrix:
                matrix[current] = Counter()

            counter = matrix[current]
            counter[succ] += 1

        return matrix

    def read_scary_words(self, filename):
        with open(filename) as f:
            scary_words = f.read().split()
        return scary_words

    def load_rhyme_dict(self, filename):
        if os.path.isfile(filename):
            return pickle.load(open(filename, "rb"))
        
        # Create a new dict if file is not found
        # (may take a long time)
        #d = create_rhyme_dict(corpus_words + get_scary_words())
        #save_dict(d, filename) 
        #return d
        return

    def create_rhyme_dict(self, words):
        rhyme_dict = defaultdict(set)

        for word in words:
            if word.lower() not in cmuwords:
                continue    
            rhyme_dict[get_rhyme_phoneme(word)].add(word)

        return rhyme_dict

    def save_dict(self, dictionary, filename):
        pickle.dump(dictionary, open(filename, "wb"))


class Poet:
    def __init__(self, reverse_matrix, rhyme_dict, scary_words, min_lines, max_lines, min_syllables, max_syllables):
        self.reverse_matrix = reverse_matrix
        self.rhyme_dict = rhyme_dict
        self.scary_words = scary_words
        self.min_lines = min_lines
        self.max_lines = max_lines
        self.min_syllables = min_syllables
        self.max_syllables = max_syllables

    def matching_phonemes(self, words, mode):
        phonemes = []
        for word in words:
            word = word.lower()

            if word not in cmuwords:
                return False

            pronunciation = phonedict[word][0]
                
            if mode == "r":
                phoneme = next(p for p in reversed(pronunciation) if p[0] in VOWELS)
            elif mode == "a": 
                phoneme = pronunciation[0]
                if phoneme[0] not in CONSONANTS:
                    return False

            phonemes.append(phoneme)


        return all(p == phonemes[0] for p in phonemes)


    def rhyme(self, words):
        return matching_phonemes(words, "r")


    def alliterate(self, words):
        return matching_phonemes(words, "a")

    def generate_alliteration(self, keys):    
        item = random.choice(keys)
        prev = item
        for word in item:
            yield word

        for _ in xrange(7):
            words = list(self.reverse_matrix[item].elements())
            allit_words = filter(lambda x: alliterate(prev + (x, )), words)
            if not allit_words:
               return

            word = random.choice(allit_words)
            yield word
            item = item[1:] + (word,)
            prev = item


    def syllable_count(self, word):
        # Take the first pronunciation option
        syllable_count = [len(list(y for y in x if y[-1].isdigit())) for x in phonedict[word.lower()]][0]
        return syllable_count

    def get_rhyme_phoneme(self, word):
        pronunciation = phonedict[word.lower()][0]
        phonemes = []

        for p in reversed(pronunciation):
            phonemes.append(p)
            if p[0] in VOWELS:
                break  
        
        return "".join(phonemes)


    def get_first_phoneme(self,word):
        return phonedict[word.lower()][0][0]

    def get_rhyming_word(self, rhyme_dict, word):
        return random.choice(rhyme_dict[word])


    def create_rest_of_line(self, item):
        nsyl = 0
        line = []

        for word in item:
            if word.lower() not in cmuwords:
                return[]
            nsyl += self.syllable_count(word)
            line.append(word)

        while True: 
            if nsyl > self.max_syllables:
                return []
            if nsyl >= self.min_syllables:
                break
            words = list(self.reverse_matrix[item].elements())

            word = random.choice(words)
            if word.lower() not in cmuwords:
                return []

            line.append(word)
            nsyl += (self.syllable_count(word))
            item = item[1:] + (word,)

        return line

    def select_first_item(self, keys):
        best_options = []

        for _ in xrange(100):
            key = random.choice(keys)
            tagged_key = nltk.pos_tag(key)
            last_word, tag = tagged_key[0]
            if tag not in ("IN", "TO", "CC", "DT", "EX", "MD"):
                return key

        return random.choice(keys)

    def create_random_line(self):
        item = self.select_first_item(self.reverse_matrix.keys())
        line = self.create_rest_of_line(item)
        if not line:
            return

        yield " ".join(reversed(line))


    def create_rhyming_line(self, rhyme):
        rhyming_keys = [tup for tup in self.reverse_matrix.keys() if tup[0] in self.rhyme_dict[self.get_rhyme_phoneme(rhyme)]]
        if not rhyming_keys:
            return

        item = self.select_first_item(rhyming_keys)
        line = self.create_rest_of_line(item)
        if not line:
            return

        yield " ".join(reversed(line))


    def fix_capitalization(self, line):
        words = line.split()
        tagged_words = nltk.pos_tag(words)

        for i, (word, tag) in enumerate(tagged_words):
            if tag in ("NNP", "NNPS") or word.lower() == "i":
                words[i] = words[i].capitalize()
            else:
                words[i] = word.lower()

        new_line = " ".join(words)
        return new_line[0].upper() + new_line[1:]


    def create_poem(self):   
        line_count = random.randint(self.min_lines, self.max_lines)

        while True:   
            print "start"     
            first_line = "".join(self.create_random_line())
            if not first_line:
                continue

            first_line = self.make_scarier(first_line)
            rhyme = first_line.split()[-1]
            if len(self.rhyme_dict[self.get_rhyme_phoneme(rhyme)]) < line_count - 1:
                continue

            lines = [first_line]

            for _ in xrange(line_count - 1):
                line = "".join(self.create_rhyming_line(rhyme))
                lines.append(line)

            if not all(lines):
                continue

            return "\n".join(self.fix_capitalization(l) for l in lines) + "\n"


    def make_scarier(self, line):
        """Replace a random word with a scary word"""
        words = line.split()
        tagged_words = nltk.pos_tag(words)
        pos_to_change = random.choice(("NN", "NNS", "NNP", "JJ", ))
        new_words = [word for word, tag in nltk.pos_tag(self.scary_words) if tag == pos_to_change]
        indices = []
        for i, (_, tag) in enumerate(tagged_words):
            if tag == pos_to_change:
                indices.append(i)
        
        if not indices or not new_words:
            # use original line
            return line
        
        new_word = random.choice(new_words)
        index = random.choice(indices)
        words[index] = new_word
        return " ".join(words)



def main():
        order = int(sys.argv[1])
        files = sys.argv[2:]
        th = TextHandler(order, files, "rhymedict.p", "scary_words.txt")
        poet = Poet(th.get_matrix(), th.rhyme_dict, th.scary_words, 3, 5, 8, 16)
        print poet.create_poem()


if __name__ == "__main__":
    main()