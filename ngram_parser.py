import csv
import sys
import re
from collections import Counter


def get_word(filename, regex, index):
    counter = Counter()
    pattern = re.compile(regex)

    with open(filename) as f:
        for row in csv.reader(f, delimiter="\t"):
            phrase = row[0]

            if pattern.match(phrase):    
                adjective = phrase.split()[index]                   
                if counter[adjective] == 0:
                    print(phrase)             
                counter[adjective] += int(row[2])

    for word, count in counter.most_common():
        print(word, count)


filename = sys.argv[1]
pos = sys.argv[2]
word = sys.argv[3]

# query for adjectives
if pos == "a":
    regex = "".join((r"as [\S]* as [\S]* ", word, r"$"))
    get_word(filename, regex, 1)

# query for nouns
if pos == "n":
    regex = "".join((r"as ", word, r" as [\S]* [\S]*$"))
    get_word(filename, regex, -1)
