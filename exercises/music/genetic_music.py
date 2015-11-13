import random
import re
import sys
from itertools import islice
from fractions import Fraction
from collections import Counter
from deap import algorithms
from deap import base
from deap import creator
from deap import tools


pitches=["c", "d", "e", "f", "g", "a", "b", "cis", "dis", "eis", "fis", "gis", "ais", "bis", "ces", "des", "es", "fes", "ges", "aes", "bes"]
durations=["1", "2", "2.", "4", "4.", "8", "8.", "16", "16.", "32", "32.", "64", "64."]
octaves=["'", "''", ",", ""]


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


corpus = sys.stdin.read()

corpus_pitches, corpus_durs = create_tokens(corpus)
pitch_matrix = chain_pitch(corpus_pitches, 1)
duration_matrix = chain_duration(corpus_durs, 1)


def duration_to_fraction(duration):
    if "." in duration:
        d = duration[:-1]
        return Fraction(1, int(d)) + Fraction(1, 2 * int(d))

    return Fraction(1, int(duration))


def evaluation_nbar(individual):
    # the closer the notes are to full bars
    tokens = [re.findall('\d*\D*', n) for n in individual]
    durs = [t[1] for t in tokens]
    total = sum([duration_to_fraction(d) for d in durs])
    fit = abs(round(total) - total)

    return fit,

def get_octave(pitch):
    if "''" in pitch:
        return "''"
    if "'" in pitch:
        return "'"
    if "," in pitch:
        return ","
    return ""

def evaluation_octaves(individual):
    # the more notes are in the same octave
    tokens = [re.findall('\d*\D*', n) for n in individual]
    ps = [t[0] for t in tokens]
    counter = Counter(get_octave(p) for p in ps)    
    fit = counter.most_common(1)[0][1]

    return fit,

def evaluation_markov(individual):
    tokens = [re.findall('\d*\D*', n) for n in individual]
    ps = [t[0] for t in tokens]
    pfit = 0

    for cur, succ in window_generator(ps):
        key = (cur,)
        if key in pitch_matrix.keys() and succ in pitch_matrix[key]:
            pfit += 1

    ds = [t[1] for t in tokens]
    dfit = 0

    for cur, succ in window_generator(ds):
        key = (cur,)
        if key in duration_matrix.keys() and succ in duration_matrix[key]:
            dfit += 1

    return pfit + dfit,

def get_random_note():
    pitch = random.choice(pitches)
    octave = random.choice(octaves)
    duration = random.choice(durations)

    return "".join((pitch, octave, duration))

# We create a fitness for the individuals, because our eval-function gives us 
# "better" values the closer they are zero, we will give it weight -1.0.
# This creates a class creator.FitnessMin(), that is from now on callable in the
# code. (Think about Java's factories, etc.)
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))

# We create a class Individual, which has base type of list, it also uses our 
# just created creator.FitnessMin() class.
creator.create("Individual", list, fitness=creator.FitnessMin)

# We create DEAP's toolbox. Which will contain our mutation functions, etc.
toolbox = base.Toolbox()

# We create a function named 'random_digit', which calls random.randint 
# with fixed parameters, i.e. calling toolbox.random_digit() is the same as 
# calling random.randint(0, 9)
#toolbox.register('random_digit', random.randint, 0, 9)
toolbox.register('random_note', get_random_note)

# Now, we can make our individual (genotype) creation code. Here we make the function to create one instance of 
# creator.Individual (which has base type list), with tools.initRepeat function. tool.initRepeat 
# calls our just created toolbox.random_digit function n-times, where n is the 
# length of our target. This is about the same as: [random.randint(0,9) for i in xrange(len(target))].
# However, our created individual will also have fitness class attached to it (and 
# possibly other things not covered in this example.)
#toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.random_digit, n = len(target))
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.random_note, n=100)

# As we now have our individual creation code, we can create our population code
# by making a list of toolbox.individual (which we just created in last line).
# Here it is good to know, that n (population size), is not defined at this time 
# (but is needed by the initRepeat-function), and can be altered when calling the 
# toolbox.population. This can be achieved by something called partial functions, check 
# https://docs.python.org/2/library/functools.html#functools.partial if interested.
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# We register our evaluation function, which is now callable as toolbox.eval(individual).
toolbox.register("evaluate", evaluation_nbar)

# We use simple selection strategy where we select only the best individuals, 
# now callable in toolbox.select.
toolbox.register("select", tools.selBest)

def crossover(a, b):
    i = random.randint(0, len(a)-1)
    temp = a[:i+1]
    a[:i+1] = b[:i+1]
    b[:i+1] = temp

    return a, b


# We use one point crossover, now callable in toolbox.mate.
toolbox.register("mate", crossover)
#toolbox.register("mate", tools.cxOnePoint)

# Replace one index of an individual with a random note.
def mutate(individual):
    i = random.randint(0, len(individual)-1)
    individual[i] = toolbox.random_note()
    return individual,


def mutate2(individual):
    for i, val in enumerate(individual):
        if random.random() < 0.05:
            p = re.findall('\d*\D*', val)
            individual[i] = "".join((p[0], random.choice(durations)))
    
    # DEAP's mutation function has to return a tuple, thats why there is comma
    # after. 
    return individual,


# We register our own mutation function as toolbox.mutate
toolbox.register("mutate", mutate)

# Now we have defined basic functions with which the evolution algorithm (EA) can run.
# Next, we will define some parameters that can be changed between the EA runs.

# Maximum amount of generations for this run
generations = 10

# Create population of size 100 (Now we define n, which was missing when we 
# registered toolbox.population).
pop = toolbox.population(n=10)

# Create hall of fame which stores only the best individual
hof = tools.HallOfFame(1)

# Get some statistics of the evolution at run time. These will be printed to
# sys.stdout when the algorithm is running.
import numpy as np
stats = tools.Statistics(lambda ind: ind.fitness.values)
stats.register("avg", np.mean)
stats.register("std", np.std)
stats.register("min", np.min)
stats.register("max", np.max)

# Probability for crossover
crossover_prob = 0.7

# Probability for mutation
mutation_prob = 0.5
# Call our actual evolutionary algorithm that runs the evolution.
# eaSimple needs toolbox to have 'evaluate', 'select', 'mate' and 'mutate'
# functions defined. This is the most basic evolutionary algorithm. Here, we 
# have crossover probability of 0.7, and mutation probability 0.2.

algorithms.eaSimple(pop, toolbox, crossover_prob, mutation_prob, generations, stats, halloffame=hof)

# Print the best individual, and its fitness
print(hof[0], evaluation_nbar(hof[0]))

def create_file(notes, filename):
    score = "\score {\n <<\n { \key c \major\n %s\n }>> \n \midi { }\n \layout { }\n}" % notes
    with open(filename, "w") as outfile:
        outfile.write(score)


outfile = sys.argv[1]
create_file(" ".join(hof[0]), outfile)
#fit = []
#for _ in xrange(100):
#    algorithms.eaSimple(pop, toolbox, crossover_prob, mutation_prob, generations, stats, halloffame=hof)

    # Print the best individual, and its fitness
    #print hof[0], eval(hof[0])
#    fit.append(eval(hof[0])[0])

#print "Average fitness", sum(fit) / float(len(fit))

