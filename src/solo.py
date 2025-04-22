from music21 import *
import math
import random
import sys
from main import NOTES_FOR_CHORDS, SCALES_FOR_CHORDS

POPULATION_SIZE = 1000
MUTATION_RATE = 0.02
CROSSOVER_RATE = 0.7
SOLOCHUNK_RATE = 0.3
GENERATIONS = 100
GENES = [-1, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]


def getSolo(chords, melody, length, instrument):

    # define a number of sections to split the solo up into. do GA on each of these sections
    numSoloSplits = 4 # parameterize this?

    # general idea is that we want to be pretty close to the melody at the beginning of the solo,
    # then less close midway through the solo, then back to close again by the end
    # so define some numbers "what percentage of the fitness function should be defined by closeness"
    # for each split

    # closeness follows cosine, 0.75 at 0th split, 0.0 at middle of solo, 0.75 at last split (approx)
    # this only applies for saxophone, other instruments don't steal from melody as such
    closeness_percents = \
        [0.375 * (math.cos((math.pi * 2 * i / numSoloSplits) + 
                          (math.pi / (2 * numSoloSplits))) + 1) 
                for i in range(numSoloSplits)] \
            if instrument == "horn" else \
        [0 for _ in range(numSoloSplits)]
    correctness_percents = [1 - i for i in closeness_percents]

    # offsets = []
    # for i in range(numSoloSplits):
    #     beginoffset = i * length / numSoloSplits
    #     offsets.append(beginoffset)
    #     # endoffset = (i + 1) * length / numSoloSplits
    # offsets.append(length) # ensure no weird cases
    # melodySplits = [[] for _ in range(numSoloSplits)]
    # chordsSplits = [[] for _ in range(numSoloSplits)] 

    # print(melodySplits)
    # def separateIntoSplits(s, target):
    #     currentSplit = 0
    #     for x in s:
    #         if x['offset'] >= offsets[currentSplit + 1]:
    #             currentSplit += 1
    #         target[currentSplit].append(x)
            
            
    
    # separateIntoSplits(melody, melodySplits)
    # separateIntoSplits(chords, chordsSplits)
    # print([len(melodySplits[i]) for i in range(numSoloSplits)])
    # print([len(chordsSplits[i]) for i in range(numSoloSplits)])
    
    # convert melody to chromosome for use in GA "chunk of melody" mutation and fitness
    # print(melody)
    melodychromosome = [-1 for _ in range(int(length * 2))] # 2x length for 8th notes
    for n in melody:
        for o in range(int(n['offset'] * 2), int((n['offset'] + n['el'].quarterLength) * 2)):
            if isinstance(n['el'], note.Note):
                melodychromosome[o] = n['el'].pitch.pitchClass
            elif isinstance(n['el'], chord.Chord):
                melodychromosome[o] = n['el'].root().pitchClass
            else: # rest
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

    # print(chordchromosome)
    # print(melodychromosome)
    # print(len(chordchromosome))
    # print(len(melodychromosome))

    scalechromosome = []
    for ch in chordchromosome:
        temp = [(i - ch[0]) % 12 for i in ch] # normalize to C
        # python sucks
        tempscale = SCALES_FOR_CHORDS[list(NOTES_FOR_CHORDS.keys())[list(NOTES_FOR_CHORDS.values()).index(sorted(temp))]]
        tempscale = [(i + ch[0]) % 12 for i in tempscale]
        scalechromosome.append(tempscale)
    # print(scalechromosome)
    # print(len(scalechromosome))
    def chunk(li, n):
        """ Split a list into n chunks of approximately equal size. Stackoverflow helped here. """ 
        d, r = divmod(len(li), n)
        return [li[i * d + min(i, r):(i + 1) * d + min(i + 1, r)] for i in range(n)]
    chordsplits = chunk(chordchromosome, numSoloSplits)
    melodysplits = chunk(melodychromosome, numSoloSplits)
    scalesplits = chunk(scalechromosome, numSoloSplits)
    # this works
    
    



    def initializePopulation(splitlength, whichsplit):
        """ Initialize the population of melodies. 
        The population is a list of lists of notes. Each note is represented by a 
        number from 0, where 0 is C, 1 is C#, 2 is D, etc. -1 is a rest."""
        population = []
        for i in range(POPULATION_SIZE):
            chromosome = [-1 for _ in range(splitlength)]
            for j in range(len(chordsplits[whichsplit])):
                octaveUp = (scalesplits[whichsplit][j][k] + 12 for k in range(len(scalesplits[whichsplit][j])))
                choices = [-1, *scalesplits[whichsplit][j], *octaveUp]
                # print(choices)
                gene = random.choice(choices)
                # print(gene)
                chromosome[j] = gene
                if gene == -1:
                    rest_length = random.randint(2, 5)
                    for k in range(j, min(j + rest_length, len(chromosome))):
                        chromosome[k] = -1
                        j += rest_length - 1
            # chromosome = random.choices(GENES, k=splitlength) # random.choices is a list of length splitlength
            # for j in range(len(chromosome)):
            #     if chromosome[j] == -1:
            #         rest_length = random.randint(2, 5)
            #         for k in range(j, min(j + rest_length, len(chromosome))):
            #             chromosome[k] = -1
            #             j += rest_length - 1
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
            parent2 = random.choices(population, weights=fitnessScores)[0]
            child = []
            if random.random() < CROSSOVER_RATE:
                crossoverPoint = random.randint(0, len(parent1))
                child = parent1[crossoverPoint:] + parent2[:crossoverPoint]
            else:
                child = random.choice([parent1, parent2])
            # for i in range(len(child)):
            #     if random.random() < MUTATION_RATE:
            #         child[i] = random.choice(GENES)
            if random.random() < SOLOCHUNK_RATE:
                # randomly replace a chunk of the child with the same chunk from the melody
                chunkStart = random.randint(0, len(child) - 1)
                chunkEnd = random.randint(chunkStart, len(child) - 1)
                chunk = melodychromosome[int(length * whichsplit // 4 + chunkStart):int(length * whichsplit // 4 + chunkEnd)]
                for i in range(len(chunk)):
                    child[i] = chunk[i]
            newPopulation.append(child)
        return newPopulation

    def getFitnessScores(population, whichsplit):
        """ Get the fitness scores for the population. 
        The fitness score is a number between 0 and 1. The higher the score, the 
        better the melody. The score is calculated by comparing the melody to the 
        chords and the melody. The closer the melody is to the chords and the 
        melody, the higher the score."""
        fitnessScores = [0 for _ in range(len(population))]
        for i, chromosome in enumerate(population):
            corr = correctness(chromosome, whichsplit)
            fitnessScores[i] += corr * correctness_percents[whichsplit]
            clo = closeness(chromosome, whichsplit)
            fitnessScores[i] += clo * closeness_percents[whichsplit]
            # if fitnessScores[i] > 0.9:
                # print(f"Split {whichsplit}: {chromosome} {fitnessScores[i]} {corr} {clo}")
        return fitnessScores

    def correctness(chromosome, whichsplit):
        chordsplit = chordsplits[whichsplit]
        scalesplit = scalesplits[whichsplit]
        melodysplit = melodysplits[whichsplit]
        score = 0
        for i, ((gene, chord), (mNote, scale)) in enumerate(zip(zip(chromosome, chordsplit), zip(melodysplit, scalesplit))):
            if gene % 12 in chord or gene % 12 == mNote: # if the note is in the chord or the melody, yay
                score += .05
            elif (gene + 1) % 12 in chord or (gene - 1) % 12 in chord: # elif because we sometimes this the melody note is 1 away from a chord tone
                score -= .125
            if gene % 12 in scale: # play notes in the scale!
                score += .4
            else: # gene % 12 not in scale
                score -= .2 # don't think this can happen
            if i > 0 and gene % 12 == chromosome[i - 1] % 12:
                score += .2
            if i > 1 and gene % 12 == chromosome[i - 1] % 12 and i > 1 and gene % 12 == chromosome[i - 2] % 12:
                score += .2
            if i > 7 and all(gene % 12 == chromosome[i - j] % 12 for j in range(1, 8)):
                score -= 1  # Penalize heavily for the same note repeated 8 times in a row
            if i > 3 and all(gene % 12 == chromosome[i - j] % 12 for j in range(1, 4)):
                score -= 0.2 # not too long of the same note
            if i > 0 and abs((gene % 12) - (chromosome[i - 1] % 12)) > 4:
                score -= .6
            if i > 0 and 1 <= abs((gene % 12) - (chromosome[i - 1] % 12)) < 3:
                score += .2
            if i > 0 and 1 <= abs((gene % 12) - (chromosome[i - 1] % 12)) < 2:
                score += .6
                if instrument == "bass":
                    score += .6 # stepwise motion is good for bass
        percentRests = chromosome.count(-1) / float(len(chromosome))
        if percentRests > 0.45 or percentRests < 0.2:
            score -= .1

        # Count the number of attacks (notes that play after rests, or notes that change)
        attacks = sum(1 for i in range(1, len(chromosome)) if chromosome[i] != chromosome[i - 1] and chromosome[i] != -1)
        expectedattacks = 0
        if instrument == "bass":
            expectedattacks = 0.5 * len(chromosome) # one attack per beat
        elif instrument == "piano" or instrument == "horn":
            expectedattacks = 0.75 * len(chromosome) # one attack every 3/4 of a beat on average
        score -= abs(expectedattacks - attacks) * .25
        x = (math.tanh(score) + 1) / 2
        # return -4000/441 * x**3 + 4700/441 * x**2 - 100/63 * x
        return x
    

    def closeness(chromosome, whichsplit):
        count = 0
        melody = melodysplits[whichsplit]
        attacksmelody = sum(1 for i in range(1, len(melody)) if melody[i] != melody[i - 1] and melody[i] != -1)
        attackschromosome = sum(1 for i in range(1, len(chromosome)) if chromosome[i] != chromosome[i - 1] and chromosome[i] != -1)
        def attack_similarity(attacksmelody, attackschromosome):
            difference = abs(attacksmelody - attackschromosome)
            return max(0, 1 - (difference / max(attacksmelody, 1)))

        similarity_score = attack_similarity(attacksmelody, attackschromosome)
        melody_intervals = [abs((melody[i+1] % 12) - (melody[i] % 12)) for i in range(len(melody) - 1) if melody[i] != -1 and melody[i+1] != -1 and abs((melody[i+1] % 12) - (melody[i] % 12)) != 0]
        chromosome_intervals = [abs((chromosome[i+1] % 12) - (chromosome[i] % 12)) for i in range(len(chromosome) - 1) if chromosome[i] != -1 and chromosome[i+1] != -1 and abs((chromosome[i+1] % 12) - (chromosome[i] % 12)) != 0]
        intcount = 0 
        for interval in melody_intervals:
            if any(abs(interval - our_interval) <= 2 for our_interval in chromosome_intervals):
                intcount += 1
        interval_similarity = intcount / len(melody_intervals) if len(melody_intervals) > 0 else 1
        for i in range(len(chromosome)):
            if melody[i] == chromosome[i]:
                count += 1
        x = count / (len(chromosome))
        closeness_score = -4000/441 * x**3 + 4700/441 * x**2 - 100/63 * x
        # print(closeness_score)
        # print(closeness_score, similarity_score, interval_similarity, chromosome) if closeness_score * similarity_score * interval_similarity >= 0.90 else None
        return closeness_score * similarity_score * interval_similarity

    def generateSplit(whichsplit):
        # global POPULATION_SIZE, GENERATIONS
        # if we want to modify ^
        pop = initializePopulation(len(chordsplits[whichsplit]), whichsplit)
        best = 0
        tenthbest = 0
        x = 0
        okayWereDone = False
        while best < 0.99 and x < 100:
            
            fitnessScores = getFitnessScores(pop, whichsplit)
            correctnessScores = [correctness(chromosome, whichsplit) for chromosome in pop]
            closenessScores = [closeness(chromosome, whichsplit) for chromosome in pop]
            
            tenthbest = sorted(fitnessScores, reverse=True)[9]
            best = sorted(fitnessScores, reverse=True)[0]
            best_idx = fitnessScores.index(best)    
            tenthbestcorr = correctnessScores[best_idx]
            tenthbestclose = closenessScores[best_idx]
            # print(f"{x}: {fitnessScores}")
            x += 1
            # Remove the bottom half of the population based on fitness scores
            sorted_population = sorted(zip(pop, fitnessScores), key=lambda pair: pair[1], reverse=True)
            
            pop, fitnessScores = zip(*sorted_population[:POPULATION_SIZE // 2])
            pop, fitnessScores = list(pop), list(fitnessScores)
            
            pop = nextGeneration(pop, fitnessScores, whichsplit)
            if x % 10 == 0:
                if tenthbestclose == 0:
                    okayWereDone = True
                if okayWereDone:
                    break
                print(f"Generation {x} for split {whichsplit}, tenthbest = {best}, corr = {tenthbestcorr}, close = {tenthbestclose}", file=sys.stderr)
            
        # best = sorted(fitnessScores, reverse=True)[0]
        # best_idx = fitnessScores.index(best)    
        # print(f"Split {whichsplit}: corr={correctnessScores[best_idx]:.4f} at {correctness_percents[whichsplit]}, close={closenessScores[best_idx]:.4f} at {closeness_percents[whichsplit]}")
        return pop[best_idx]
        

    melodies = [generateSplit(i) for i in range(numSoloSplits)]
    # print(melodies)
    solo = []
    for melody in melodies:
        previous_note = None
        duration = 0.5
        for note_value in melody:
            if note_value == -1:  # Handle rests
                if isinstance(previous_note, note.Rest):
                    duration += 0.5
                else:
                    if previous_note:
                        previous_note.quarterLength = duration
                        solo.append(previous_note)
                    previous_note = note.Rest(quarterLength=0.5)
                    duration = 0.5
            else:  # Handle notes
                if isinstance(previous_note, note.Note) and previous_note.pitch.pitchClass == note_value:
                    duration += 0.5
                else:
                    if previous_note:
                        previous_note.quarterLength = duration
                        solo.append(previous_note)
                    previous_note = note.Note(note_value + 60, quarterLength=0.5)
                    duration = 0.5
        if previous_note:  # Add the last note or rest
            previous_note.quarterLength = duration
            solo.append(previous_note)
    return solo
    # print(length)
    # print(closeness_percents)
    # print(correctness_percents)
    # sc = stream.Score()
    # for chord_entry in chords:
    #     chord_symbol = chord_entry['el']
    #     offset = chord_entry['offset']
    #     if isinstance(chord_symbol, harmony.ChordSymbol):
    #         solo.insert(offset, chord_symbol)
    # sc.append(solo)
    # solo.show()
    # sc.makeMeasures()
    # sc.show('text')
    # sc.show()



    # solo = []
    # for _ in range(int(length)):
    #     solo.append(note.Rest())
    # if length % 1 != 0:
    #     solo.append(note.Rest(quarterlength=length % 1))
    # return solo
    exit()

def transposeSolo(solo, instrument):
    # Transpose the solo to the key of the instrument
    # transposition_interval = interval.Interval('P0')  # Default to no transposition
    if instrument.lower() in ["piano"]:
        for n in solo:
            if isinstance(n, note.Note):
                n.transpose("P8", inPlace=True)
            # n.transpose("P8", inPlace=True)
    elif instrument.lower() in ["bass"]:
        for n in solo:
            if isinstance(n, note.Note):
                n.transpose("-P15", inPlace=True)
            # n.transpose("-P15", inPlace=True)

    

    return solo

from checkplayable import get_chords
if __name__ == "__main__":
    chords, melody, length, t, i44 = get_chords("../leads/AllOfMe.musicxml")
    # print(melody)
    s = getSolo(chords, melody, length, "horn")
    st = stream.Stream()
    for n in s:
        st.append(n)
    sc = stream.Score()
    sc.append(st)
    for ch in chords:
        if isinstance(ch['el'], harmony.ChordSymbol):
            st.insert(ch['offset'], ch['el'])
    sc.makeMeasures()
    st.show()

    pass