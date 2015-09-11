import csv
import sys
from collections import Counter

filename = sys.argv[1]
pattern = sys.argv[2]
counter = Counter()

with open(filename) as f:
    for row in csv.reader(f, delimiter="\t"):
        phrase = row[0]

        if phrase.startswith(pattern):
            last_word = phrase.split()[-1]
            counter[last_word] += int(row[2])


for word, count in counter.most_common():
    print(word, count)


