import csv
import sys
import re
import random
from collections import Counter


def get_word(filename, regex, index):
    counter = Counter()
    pattern = re.compile(regex)

    with open(filename) as f:
        for row in csv.reader(f, delimiter="\t"):
            phrase = row[0]

            if pattern.match(phrase):    
                expression = phrase.split()[index]                   
                if counter[expression] == 0:
                    print(phrase)             
                counter[expression] += int(row[2])

    #for word, count in counter.most_common():
    #    print(word, count)
    return counter


def generate_metaphor(filename, word, regex):
    noun = random.choice(list(get_word(filename, regex, -1)))
    article = random.choice(("the", "an" if noun[0] in "aeiou" else "a"))
    metaphor = " ".join(("She is", article, noun))
    print(metaphor)


filename = sys.argv[1]
word = sys.argv[2]
pos = sys.argv[3]

# query for adjectives
if pos == "a":
    regex = "".join((r"as [\S]* as [\S]* ", word, r"$"))
    get_word(filename, regex, 1)
# query for nouns
elif pos == "n":
    regex = "".join((r"as ", word, r" as [\S]* [\S]*$"))
    get_word(filename, regex, -1)
else:
    regex = "".join((r"as ", word, r" as [\S]* [\S]*$"))
    generate_metaphor(filename, word, regex)
