import csv
import sys
import re
from collections import Counter

filename = sys.argv[1]
pattern = sys.argv[2]


def get_noun(filename, pattern):
    counter = Counter()

    with open(filename) as f:
        for row in csv.reader(f, delimiter="\t"):
            phrase = row[0]

            if phrase.startswith(pattern):
                last_word = phrase.split()[-1]
                counter[last_word] += int(row[2])

    for word, count in counter.most_common():
        print(word, count)


def get_adjective(filename, word):
    counter = Counter()
    regex = r"as [\S]* as [\S]* "
    pattern = re.compile("".join((regex, word, r"$")))

    with open(filename) as f:
        for row in csv.reader(f, delimiter="\t"):
            phrase = row[0]

            if pattern.match(phrase):    
                adjective = phrase.split()[1]                   
                if counter[adjective] == 0:
                    print(phrase)             
                counter[adjective] += int(row[2])

    for word, count in counter.most_common():
        print(word, count)


get_adjective(filename, pattern)
