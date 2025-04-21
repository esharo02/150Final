from music21 import *
import math
import random
import sys

POPULATION_SIZE = 1000
MUTATION_RATE = 0.1
CROSSOVER_RATE = 0.5
SOLOCHUNK_RATE = 0.2
GENERATIONS = 1000
GENES = [-1, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]


def getSolo(chords, melody, length, instrument):

    # define a number of sections to split the solo up into. do GA on each of 
    # these sections
    numSoloSplits = 4 # parameterize this?

    # general idea is that we want to be pretty close to the melody at the beginning of the solo,
    # then less close midway through the solo, then back to close again by the end
    # so define some numbers "what percentage of the fitness function should be defined by closeness"
    # for each split

    # closeness follows cosine, 0.6 at 0th split, 0.0 at middle of solo, 0.6 at last split
    closeness_percents = [0.3 * (math.cos((math.pi * 2 * i / numSoloSplits) + (math.pi / (2 * numSoloSplits))) + 1) for i in range(numSoloSplits)]
    correctness_percents = [1 - i for i in closeness_percents]

    offsets = []
    for i in range(numSoloSplits):
        beginoffset = i * length / numSoloSplits
        offsets.append(beginoffset)
        # endoffset = (i + 1) * length / numSoloSplits
    offsets.append(length) # ensure no weird cases
    melodySplits = [[] for _ in range(numSoloSplits)]
    chordsSplits = [[] for _ in range(numSoloSplits)] 

    print(melodySplits)
    def separateIntoSplits(s, target):
        currentSplit = 0
        for x in s:
            if x['offset'] >= offsets[currentSplit + 1]:
                currentSplit += 1
            target[currentSplit].append(x)
            
            
    
    separateIntoSplits(melody, melodySplits)
    separateIntoSplits(chords, chordsSplits)
    print([len(melodySplits[i]) for i in range(numSoloSplits)])
    print([len(chordsSplits[i]) for i in range(numSoloSplits)])
    
    # convert melody to chromosome for use in GA "chunk of melody" mutation and fitness
    print(melody)
    melodychromosome = [-1 for _ in range(int(length * 2))] # 2x length for 8th notes
    for n in melody:
        for o in range(int(n['offset'] * 2), int((n['offset'] + n['el'].quarterLength) * 2)):
            if isinstance(n['el'], note.Note):
                melodychromosome[o] = n['el'].pitch.pitchClass
            else:
                print(f"A non-note was found in the melody: {n['el']}", file=sys.stderr)
    
    # do the same for chords
    chordchromosome = []
    offset = 0
    extendedChords = chords + [{'el': -1, 'offset': length}]
    for i, n in enumerate(extendedChords): # add a dummy element to the end of the list
        if isinstance(n['el'], harmony.ChordSymbol):
            while offset < extendedChords[i+1]['offset']:
                chordchromosome.append([i.pitch.pitchClass for i in n['el'].notes])
                offset += 0.5
        elif n['el'] == -1: # dummy
            continue
        else:
            print(f"A non-chord was found in the chords: {n['el']}", file=sys.stderr)

    print(chordchromosome)
    print(melodychromosome)
    print(len(chordchromosome))
    print(len(melodychromosome))
    exit()


    def initializePopulation():
        """ Initialize the population of melodies. 
        The population is a list of lists of notes. Each note is represented by a 
        number from 0, where 0 is C, 1 is C#, 2 is D, etc. -1 is a rest."""
        population = []
        for i in range(POPULATION_SIZE):
            chromosome = []
            for j in range(int(length)):
                choice = random.choice(GENES) 
                if choice == -1:
                    # rest for more than an eighth note
                    for i in range(round(random.random() * 5) + 1): # maximum 3 beat rest
                        chromosome.append(-1)
                else:
                    chromosome.append(choice)

            population.append(chromosome)
        return population

    def nextGeneration(population, fitnessScores, whichsplit):
        """ Create the next generation of the population. 
        The next generation is created by selecting two parents from the population
        based on their fitness. The child is created by crossing over the parents at
        a random point, if crossover was selected. The child is then mutated at a 
        random point, if mutation was selected. Both of these can occur on the same
        child. The child is then added to the new population. """
        newPopulation = []
        for i in range(POPULATION_SIZE):
            parent1 = random.choices(population, weights=fitnessScores)[0]
            parent2 = None
            while parent2 == None or parent2 == parent1:
                parent2 = random.choices(population, weights=fitnessScores)[0]
            child = []
            if random.random() < CROSSOVER_RATE:
                crossoverPoint = random.randint(0, len(parent1))
                child = parent1[crossoverPoint:] + parent2[:crossoverPoint]
            else:
                child = random.choice([parent1, parent2])
            for i in range(len(child)):
                if random.random() < MUTATION_RATE:
                    child[i] = random.choice(GENES)
            if random.random() < SOLOCHUNK_RATE:
                # randomly replace a chunk of the child with the same chunk from the melody
                chunkStart = random.randint(0, len(child) - 1)
                chunkEnd = random.randint(chunkStart, len(child) - 1)
                chunk = melodychromosome[(length * whichsplit // 4 + chunkStart):(length * whichsplit // 4 + chunkEnd)]
                for i in range(chunkStart, chunkEnd):
                    child[i] = chunk[i]
            newPopulation.append(child)
        return newPopulation

    def getFitnessScores(population, chordsplit, melodysplit, closeness_percent, correctness_percent):
        """ Get the fitness scores for the population. 
        The fitness score is a number between 0 and 1. The higher the score, the 
        better the melody. The score is calculated by comparing the melody to the 
        chords and the melody. The closer the melody is to the chords and the 
        melody, the higher the score."""
        fitnessScores = [0 for _ in range(len(population))]
        for chromosome in population:
            fitnessScores += correctness(chromosome, chordsplit, melodysplit) * correctness_percent
            fitnessScores += closeness(chromosome, chordsplit, melodysplit) * closeness_percent
        return fitnessScores

    def correctness(chromosome, chordsplit, melodysplit):
        score = 0
        for i, (gene, chord) in enumerate(zip(chromosome, chordsplit)):
            pass



    def generateSplit(chordsplit, melodysplit, closeness_percent, correctness_percent, whichsplit):
        # global POPULATION_SIZE, GENERATIONS
        # if we want to modify ^

        pop = initializePopulation()
        for x in range(GENERATIONS):
            fitnessScores = getFitnessScores(pop, chordsplit, melodysplit, closeness_percent, correctness_percent)
            pop = nextGeneration(pop, fitnessScores, whichsplit)
        return sorted(zip(pop, fitnessScores), key=lambda pair: pair[1], reverse=True)[0][0]
        

    generateSplit(chordsSplits[0], melodySplits[0], closeness_percents[0], correctness_percents[0])


    print(length)
    print(closeness_percents)
    print(correctness_percents)
    



    # solo = []
    # for _ in range(int(length)):
    #     solo.append(note.Rest())
    # if length % 1 != 0:
    #     solo.append(note.Rest(quarterlength=length % 1))
    # return solo
    exit()

def transposeSolo(solo, instrument):
    return solo

from checkplayable import get_chords
if __name__ == "__main__":
    chords, melody, length, t = get_chords("..\leads\MrPC.musicxml")
    print(melody)
    getSolo(chords, melody, length, "horn")
    pass