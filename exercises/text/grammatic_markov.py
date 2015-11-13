from __future__ import division
import sys
import nltk
import random
import time
from itertools import islice
from nltk.tokenize import RegexpTokenizer
from collections import Counter


order = int(sys.argv[1])

content_text = sys.stdin.read()
tokenizer = RegexpTokenizer('\[[^\]]*\( |[+/\-@&*"]|\w+|\$[\d\.]+|\S+')
text = tokenizer.tokenize(content_text.decode('utf-8'))
tagged_tokens = nltk.pos_tag(text)
word_tag_dict = dict(tagged_tokens)

pos_dict = {} # {pos: {(word, count)}}

for word, pos in tagged_tokens:
    if pos not in pos_dict:
        pos_dict[pos] = Counter()

    # Convert word to lowercase if it's not a proper noun or "I"
    if pos not in ('NNP', 'NNPS') and word != "I":
        word = word.lower()

    counter = pos_dict[pos]
    counter[word] += 1


# Calculate the distribution of words in each pos
#distribution = {}
 
#for key, frequencies in pos_dict.iteritems():
#    total = sum(frequencies.values())
#    distribution[key] = Counter()

#    for word, count in frequencies.items():
#        distribution[key][word] = count / total


def window_generator(seq, word_tags, n=2):
    it = iter(seq)
    result = tuple((word_tags[x] for x in islice(it, n)))
    if len(result) == n:
        yield result    
    for elem in it:
        result = result[1:] + (word_tags[elem],)
        yield result

pos_tokens = window_generator(text, word_tag_dict, order + 1)

pos_matrix = {}

# Create the pos matrix
for pos_sequence in pos_tokens:
    current = pos_sequence[:-1]
    succ = pos_sequence[-1]

    if current not in pos_matrix:
        pos_matrix[current] = Counter()

    counter = pos_matrix[current]
    counter[succ] += 1


def sentence_generator(pos_matrix, tag_word_dict, order):
    # Pick a random pos to start
    item = random.choice(pos_matrix.keys())
    for pos in item:
        # Pick a random word of the pos
        word = random.choice(list(tag_word_dict[pos].elements()))
        yield word

    while True:
        pos_list = list(pos_matrix[item].elements())
        pos = random.choice(pos_list)
        word = random.choice(list(tag_word_dict[pos].elements()))
        yield word
        item = item[1:] + (pos,)

last_token = "."

for word in sentence_generator(pos_matrix, pos_dict, order):
    time.sleep(0.5)
    if word.startswith((".", ",", ";", "!", "?", ":", "'", "\"")):
        sys.stdout.write(word)
    elif last_token in ".!?":
        # Decide whether to stop writing
        if random.random() < 0.2:
            break
        sys.stdout.write(" " + word.capitalize())
    else:
        sys.stdout.write(" " + word)
    last_token = word
    sys.stdout.flush()

print("")
