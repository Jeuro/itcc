from __future__ import division
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
BORDER = "-----------------------------------"
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

    def add_ngrams_to_matrix(self, text, matrix):
        tokens = self.window_generator(text, self.order + 1)
        for sequence in tokens:
            current = sequence[:-1]
            succ = sequence[-1]

            if current not in matrix:
                matrix[current] = Counter()

            counter = matrix[current]
            counter[succ] += 1

    def create_reverse_matrix(self, text):
        text = reversed(text)
        matrix = {}
        tokens = self.window_generator(text, self.order + 1)
        for sequence in tokens:
            current = sequence[:-1]
            succ = sequence[-1]

            if current not in matrix:
                matrix[current] = Counter()

            counter = matrix[current]

            if succ in self.scary_words:
                counter[succ] += 2
            else:
                counter[succ] += 1
        return matrix

    def increase_matrix_probs(self, text, matrix):                
        """Teach a new text to given matrix."""
        lines = text.splitlines()
        for line in lines:
            line = reversed(line)
            tokens = self.window_generator(line, self.order + 1)
            for sequence in tokens:
                current = sequence[:-1]
                succ = sequence[-1]

                if current not in matrix:
                    matrix[current] = Counter()

                counter = matrix[current]
                counter[succ] += 1

    def decrease_matrix_probs(self, text, matrix):                
        """Decrease state transition probabilities found in a text in given matrix."""
        lines = text.splitlines()
        for line in lines:
            line = reversed(line)
            tokens = self.window_generator(line, self.order + 1)
            for sequence in tokens:
                current = sequence[:-1]
                succ = sequence[-1]

                if current not in matrix:
                    matrix[current] = Counter()

                counter = matrix[current]
                # do not entirely remove an existing transition 
                if counter[succ] > 1:
                    counter[succ] -= 1

    def update_matrix(self, good_texts, poor_texts, matrix):
        for gt in good_texts:
            self.increase_matrix_probs(gt, matrix)
        for pt in poor_texts:
            self.decrease_matrix_probs(pt, matrix)

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
    def __init__(self, th, line_lengths, min_syllables, max_syllables):
        self.th = th
        self.reverse_matrix = th.get_matrix()
        self.rhyme_dict = th.rhyme_dict
        self.scary_words = th.scary_words
        self.scary_pos_dict = self.create_pos_dict(self.scary_words)
        self.line_lengths = line_lengths
        self.min_syllables = min_syllables
        self.max_syllables = max_syllables
        self.switch_prob = 0.5

    def create_pos_dict(self, words):
        pos_dict = defaultdict(list)
        tagged_words = nltk.pos_tag(words)
        for (word, tag) in tagged_words:
            pos_dict[tag].append(word)
        return pos_dict

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
        rhyming_keys = [tup for tup in self.reverse_matrix.keys() if tup[0] != rhyme and tup[0] in self.rhyme_dict[self.get_rhyme_phoneme(rhyme)]]
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

    def fix_poem_format(self, lines):
        for i, line in enumerate(lines):
            line = self.fix_capitalization(line)
            if i % 2 != 0:
                line += "\n"
            lines[i] = line

    def make_scarier(self, lines):
        """Replace random words with scary words."""
        pos_to_change =("NN", "NNS", "NNP", "NNPS", "JJ", "VB", "VBG", "VBD")
        
        for i, line in enumerate(lines):            
            words = line.split()
            tagged_words = nltk.pos_tag(words)

            for j, (_, tag) in enumerate(tagged_words):
                if (tag in pos_to_change) and self.scary_pos_dict[tag] and (j < len(tagged_words) - 1) and (random.random() < self.switch_prob):                             
                    new_word = random.choice(self.scary_pos_dict[tag])
                    words[j] = new_word
            
            lines[i] = " ".join(words)

    def create_poem(self):   
        line_count = random.choice(self.line_lengths)

        while True: 
            lines = []
            for i in xrange(int(line_count/2)):    
                first_line = "".join(self.create_random_line())
                if not first_line:
                    break

                lines.append(first_line)
                rhyme = first_line.split()[-1]
                line = "".join(self.create_rhyming_line(rhyme))
                lines.append(line)
                
            if len(lines) < line_count or not all(lines):
                continue

            self.make_scarier(lines)
            self.fix_poem_format(lines)
            return "\n".join(lines)

    def update_matrix(self, good_poems, bad_poems):
        self.th.update_matrix(good_poems, bad_poems, self.reverse_matrix)


def main():
    order = 1
    files = sys.argv[1:]
    th = TextHandler(order, files, "rhymedict.p", "scary_words.txt")
    line_lengths = (4, )
    poet = Poet(th, line_lengths, 8, 16)
    round_counter = 1

    while True:
        print "Round", round_counter
        poem1 = poet.create_poem()
        poem2 = poet.create_poem()

        print BORDER + "\n"
        print poem1
        print BORDER + "\n"
        print poem2
        print BORDER

        selection = raw_input(
            "Which poem is a better Halloween poem?\n" + 
            "1 = first poem\n" +
            "2 = second poem\n" +
            "3 = both are good\n" +
            "4 = both are bad" +
            "\nEnter = skip evaluation:\n")

        if selection == 1:
            poet.update_matrix((poem1), (poem2))
        if selection == 2:
            poet.update_matrix((poem2), (poem1))
        if selection == 3:            
            poet.update_matrix((poem1, poem2), ())
        if selection == 4:
            poet.update_matrix((), (poem1, poem2))

        exit = raw_input("\nPress Enter to create new poems... (enter q to stop)\n")
        if exit == "q":
            return
        round_counter +=1


if __name__ == "__main__":
    main()
