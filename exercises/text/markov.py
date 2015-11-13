import sys
import itertools
import urllib2
import nltk
import random
import time
from itertools import islice
from nltk.tokenize import RegexpTokenizer
from collections import Counter


order = int(sys.argv[1])

def window_generator(seq, n=2):
    it = iter(seq)
    result = tuple(islice(it, n))
    if len(result) == n:
        yield result    
    for elem in it:
        result = result[1:] + (elem,)
        yield result


#text_source = "http://www.gutenberg.org/cache/epub/158/pg158.txt"
#data = urllib2.urlopen(text_source)
#content_text = data.read()

content_text = sys.stdin.read()
tokenizer = RegexpTokenizer('\[[^\]]*\( |[+/\-@&*]|\w+|\$[\d\.]+|\S+')
# For words
#text = tokenizer.tokenize(content_text)
# For characters
text = content_text

tokens = window_generator(text, order + 1)

matrix = {}

for sequence in tokens:
    current = sequence[:-1]
    succ = sequence[-1]

    if current not in matrix:
        matrix[current] = Counter()

    counter = matrix[current]
    counter[succ] += 1


def markov_generator(matrix, order):
    # Pick a random word to start
    item = random.choice(matrix.keys())
    for word in item:
        yield word

    while True:
        words = list(matrix[item].elements())
        word = random.choice(words)
        yield word
        item = item[1:] + (word,)


#for word in markov_generator(matrix, order):
#    time.sleep(0.5)
#    if word.startswith((".", ",", ";", "!", "?", ":")):
#        sys.stdout.write(word)
#    else:
#        sys.stdout.write(" " + word)
#    sys.stdout.flush()

#for char in markov_generator(matrix, order):
#    time.sleep(0.1)
#    sys.stdout.write(char)
#    sys.stdout.flush()



