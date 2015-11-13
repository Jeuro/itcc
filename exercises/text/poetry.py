import sys
import itertools
import urllib2
import nltk
import random
import time
import csv
import pickle
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


def syllable_count(word):
    # Take the first pronunciation option
    syllable_count = [len(list(y for y in x if y[-1].isdigit())) for x in phonedict[word.lower()]][0]
    return syllable_count


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


def get_first_phoneme(word):
    return phonedict[word.lower()][0][0]


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

def get_places():
    countries = []

    with open("countries.csv") as csvfile:
        reader = csv.reader(csvfile, delimiter=',')

        # skip headers
        next(reader, None)

        for row in reader:
            countries.append(row[1])

    return countries


def get_animals():
    animals = []

    with open("animals.csv") as csvfile:
        reader = csv.reader(csvfile, delimiter=',')

        # skip headers
        next(reader, None)

        for row in reader:
            a = row[1]
            if a:
                animals.append(a)

    return animals


def get_rhyming_places(rhyme_dict, places):
    rhyming_places = []

    for p in places:
        if p.lower() not in wordsdict:
            continue
        # test if anything rhymes with place
        if rhyme_dict[get_rhyme_phoneme(p)]:
            rhyming_places.append(p)

    return rhyming_places


def get_nsyl_animal(animals, nsyl):
    n_animals = []
    for a in animals:
        a = a.lower()
        if a not in wordsdict: 
            continue
        if syllable_count(a) == nsyl:
            n_animals.append(a)
    
    return n_animals


def create_rest_of_line(item, reverse_matrix, min_length, max_length):
    nsyl = 0
    line = []

    for word in item:
        if word.lower() not in wordsdict:
            return[]
        nsyl += syllable_count(word)
        line.append(word)

    while True: 
        if nsyl > max_length:
            return []
        if nsyl >= min_length:
            break
        words = list(reverse_matrix[item].elements())

        word = random.choice(words)
        if word.lower() not in wordsdict:
            return[]

        line.append(word)
        nsyl += (syllable_count(word))
        item = item[1:] + (word,)

    return line


def create_random_line(reverse_matrix, min_length, max_length):
    item = random.choice(reverse_matrix.keys())
    line = create_rest_of_line(item, reverse_matrix, min_length, max_length)
    if not line:
        return

    yield " ".join(reversed(line))


def create_rhyming_line(reverse_matrix, rhyme_dict, rhyme, min_length, max_length):
    rhyming_keys = [tup for tup in reverse_matrix.keys() if tup[0] in rhyme_dict[get_rhyme_phoneme(rhyme)]]
    if not rhyming_keys:
        return

    item = random.choice(rhyming_keys)
    line = create_rest_of_line(item, reverse_matrix, min_length, max_length)
    if not line:
        return

    yield " ".join(reversed(line))


def create_first_limerick_line(nsyl, places, rhyme_dict):
    # list only places that have rhyming words in the dict
    rhyming_places = get_rhyming_places(rhyme_dict, places)
    locations = [l for l in rhyming_places if syllable_count(l) < nsyl - 5]    
    location = random.choice(rhyming_places)

    nsyl_animal = nsyl - 4 - syllable_count(location)
    animals = get_nsyl_animal(get_animals(), nsyl - 4 - syllable_count(location))
    if not animals:
        return ""
    animal = random.choice(animals)
    article = "an" if get_first_phoneme(animal)[0][0] in VOWELS else "a"

    line = "There was {} {} from {}".format(article, animal, location)
    return line 


def create_limerick():
    text = tokenizer.tokenize(content_text)
    places = get_places()
    #rhyme_dict = create_rhyme_dict(text + places)
    #reverse_matrix = create_reverse_matrix(text)
    rhyme_dict = pickle.load(open("rhyme.p", "rb"))
    reverse_matrix = pickle.load(open("rmatrix.p", "rb"))
    print("Begin poetry generation\n")

    while True:
        print("New poem: \n")
        a_min = 7
        a_max = 9
        b_min = 5
        b_max = 6

        first = create_first_limerick_line(random.randint(a_min, a_max), places, rhyme_dict)
        if not first:
            continue

        location = first.split()[-1]

        second = "".join(create_rhyming_line(reverse_matrix, rhyme_dict, location, a_min, a_max))    
        fifth = "".join(create_rhyming_line(reverse_matrix, rhyme_dict, location, a_min, a_max))
        third = "".join(create_random_line(reverse_matrix, b_min, b_max))
        
        if not all((second, fifth, third)):
            continue

        rhyme = third.split()[-1]


        fourth = "".join(create_rhyming_line(reverse_matrix, rhyme_dict, rhyme, b_min, b_max))

        if not fourth:
            continue

        print("\n".join((first, second, third, fourth, fifth)) + "\n\n")
        time.sleep(8)


#create_rhymes()
create_limerick()
