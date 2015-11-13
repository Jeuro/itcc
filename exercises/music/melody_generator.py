import sys
import re
import random
import fractions
from fractions import Fraction
from itertools import islice
from collections import Counter


def create_tokens(notes):
    tokens = [re.findall('\d*\D*', n) for n in notes.split()]
    pitches = [t[0] for t in tokens]

    durations = [t[1] for t in tokens]
    
    for i, d in enumerate(durations):
        if d == "16g'":
            d = '16'
            durations[i] = d
        if d != '':
            previous = d
        else:
            durations[i] = previous

    return pitches, durations

def create_file(notes, filename):
    score = "\score {\n <<\n { \key c \major\n %s\n }>> \n \midi { }\n \layout { }\n}" % notes
    with open(filename, "w") as outfile:
        outfile.write(score)


def window_generator(seq, n=2):
    it = iter(seq)
    result = tuple(islice(it, n))
    if len(result) == n:
        yield result    
    for elem in it:
        result = result[1:] + (elem,)
        yield result


def create_markov_chain(tokens):
    matrix = {}

    for sequence in tokens:
        current = sequence[:-1]
        succ = sequence[-1]

        if current not in matrix:
            matrix[current] = Counter()

        counter = matrix[current]
        counter[succ] += 1

    return matrix


def smooth(chain, p):
    for counter in chain.values():
        total = sum(counter.values())
        avg = total/len(counter)
        for k, v in counter.items():
            diff = avg - v
            new_diff = diff * p
            counter[k] += int(new_diff)


def chain_pitch(pitches, order, p=0):
    tokens = window_generator(pitches, order + 1)
    chain = create_markov_chain(tokens)
    smooth(chain, p)
    return chain


def chain_duration(durations, order, p=0):
    tokens = window_generator(durations, order + 1)
    chain = create_markov_chain(tokens)
    smooth(chain, p)
    return chain


def duration_to_fraction(duration):
    if "." in duration:
        d = duration[:-1]
        return Fraction(1, int(d)) + Fraction(1, 2 * int(d))

    return Fraction(1, int(duration))


def get_suitable_duration(current_duration):
    d = 1 - current_duration
    if d.numerator == 3:
        return "".join((str(d.denominator/2), ".")), d 

    return str(d.denominator), d


def create_melody(pitch_matrix, duration_matrix, nbar):
    # random first notes
    pitch_key = random.choice(list(pitch_matrix.keys()))
    dur_key = random.choice(list(duration_matrix.keys()))

    bar_duration = 0

    for d in dur_key:
        # assuming that the first bar is not filled by the first 
        bar_duration += duration_to_fraction(d)
    
    notes = ["".join(list(l)) for l in zip(pitch_key, dur_key)]

    # rest of the notes from Markov chains
    for _ in range(nbar):
        while True:
            if bar_duration == 1:
                break

            dnext = random.choice(list(duration_matrix[dur_key].elements()))             
            pnext = random.choice(list(pitch_matrix[pitch_key].elements()))
            dfrac = duration_to_fraction(dnext)

            if bar_duration + dfrac > 1:
                dnext, dfrac = get_suitable_duration(bar_duration)

            bar_duration += dfrac
            n = "".join((pnext, dnext))
            notes.append(n)
            pitch_key = pitch_key[1:] + (pnext,)
            dur_key = dur_key[1:] + (dnext,)

        bar_duration = 0
    
    return notes

order = int(sys.argv[1])
nbar = int(sys.argv[2])
outfile = sys.argv[3]
corpus = sys.stdin.read()

pitches, durations = create_tokens(corpus)
pitch_matrix = chain_pitch(pitches, order, 0.6)
print(pitch_matrix)
duration_matrix = chain_duration(durations, order, 0.6)
melody = " ".join(create_melody(pitch_matrix, duration_matrix, nbar))
create_file(melody, outfile)
