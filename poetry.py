import sys
import itertools
import urllib2
import nltk
import random
import time
from nltk.corpus import cmudict 
from itertools import islice
from nltk.tokenize import RegexpTokenizer
from collections import Counter, defaultdict
from curses.ascii import isdigit


VOWELS = "AEIOU"
CONSONANTS = "BCDFGHJKLMNPQRSTVWXYZ"

order = int(sys.argv[1])
content_text = sys.stdin.read()
tokenizer = RegexpTokenizer('\w+')
phonedict = cmudict.dict()
wordsdict = cmudict.words()


def window_generator(seq, n=2):
    it = iter(seq)
    result = tuple(islice(it, n))
    if len(result) == n:
        yield result    
    for elem in it:
        result = result[1:] + (elem,)
        yield result


def matching_phonemes(words, mode):
    phonemes = []
    for word in words:
        word = word.lower()

        if word not in wordsdict:
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


def rhyme(words):
    return matching_phonemes(words, "r")


def alliterate(words):
    return matching_phonemes(words, "a")


def create_matrix():        
    text = tokenizer.tokenize(content_text)
    tokens = window_generator(text, order + 1)
    matrix = {}

    for sequence in tokens:
        current = sequence[:-1]
        succ = sequence[-1]

        if current not in matrix:
            matrix[current] = Counter()

        counter = matrix[current]
        counter[succ] += 1

    return matrix


def create_reverse_matrix(text):
    text = reversed(text)
    tokens = window_generator(text, order + 1)
    matrix = {}

    for sequence in tokens:
        current = sequence[:-1]
        succ = sequence[-1]

        if current not in matrix:
            matrix[current] = Counter()

        counter = matrix[current]
        counter[succ] += 1

    return matrix


def generate_alliteration(matrix, keys):    
    item = random.choice(keys)
    prev = item
    for word in item:
        yield word

    for _ in xrange(7):
        words = list(matrix[item].elements())
        allit_words = filter(lambda x: alliterate(prev + (x, )), words)
        if not allit_words:
           return

        word = random.choice(allit_words)
        yield word
        item = item[1:] + (word,)
        prev = item


def generate_rhyme(reverse_matrix, rhyme_dict, length):
    # first line
    first_line = []
    item = random.choice(reverse_matrix.keys())
    
    for i, word in enumerate(item):
        first_line.append(word)
        if i == 0:
            first_word = word
    
    for _ in xrange(length - 1):
        words = list(reverse_matrix[item].elements())
        word = random.choice(words)
        first_line.append(word)
        item = item[1:] + (word,)       
    
    yield " ".join(reversed(first_line))
 
    if first_word.lower() not in wordsdict:
        return    

    # second line
    second_line = []
    
    rhyming_keys = [tup for tup in reverse_matrix.keys() if tup[0] in rhyme_dict[get_rhyme_phoneme(first_word)]]
    if not rhyming_keys:
        return

    item = random.choice(rhyming_keys)
    for word in item:
        second_line.append(word)
    
    for _ in xrange(length - 1):
        words = list(reverse_matrix[item].elements())
        word = random.choice(words)
        second_line.append(word)
        item = item[1:] + (word,)

    yield " ".join(reversed(second_line))


def create_alliterations():
    matrix = create_matrix()
    allit_keys = [tup for tup in matrix.keys() if alliterate(tup)]

    while True:
        phrase = list(generate_alliteration(matrix, allit_keys))
        if len(phrase) > 2:
            print(" ".join(phrase))


def get_rhyme_phoneme(word):
    pronunciation = phonedict[word.lower()][0]
    phonemes = []

    for p in reversed(pronunciation):
        phonemes.append(p)
        if p[0] in VOWELS:
            break  
    
    return "".join(phonemes)


def create_rhyme_dict(words):
    rhyme_dict = defaultdict(set)

    for word in words:
        if word.lower() not in wordsdict:
            continue    
        rhyme_dict[get_rhyme_phoneme(word)].add(word)

    return rhyme_dict


def get_rhyming_word(rhyme_dict, word):
    return random.choice(rhyme_dict[word])


def create_rhymes():
    text = tokenizer.tokenize(content_text)
    rhyme_dict = create_rhyme_dict(text)
    reverse_matrix = create_reverse_matrix(text)
    
    length = random.randint(4, 7)

    while True:
       line = list(generate_rhyme(reverse_matrix, rhyme_dict, length))
       if len(line) > 0:
           print("\n".join(line) + "\n")
       time.sleep(5)


def syllable_count(word):
    # Take the first pronunciation option
    syllable_count = [len(list(y for y in x if y[-1].isdigit())) for x in phonedict[word.lower()]][0]
    return syllable_count


def create_rhyming_line(rhyme, nsyl):

    rhyming_keys = [tup for tup in reverse_matrix.keys() if tup[0] in rhyme_dict[get_rhyme_phoneme(first_word)]]
    if not rhyming_keys:
        return

    item = random.choice(rhyming_keys)
    for word in item:
        second_line.append(word)
    
    for _ in xrange(length - 1):
        words = list(reverse_matrix[item].elements())
        word = random.choice(words)
        second_line.append(word)
        item = item[1:] + (word,)

    yield " ".join(reversed(second_line))

def create_limerick():
    text = tokenizer.tokenize(content_text)
    rhyme_dict = create_rhyme_dict(text)
    reverse_matrix = create_reverse_matrix(text)
    # List of countries
    # List of people/professions/creatures
    person = None
    location = None
    first_line = "There was a {} from {}"

    



#create_rhymes()
print(syllable_count("anaconda"))
