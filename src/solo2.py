from music21 import *
import math
import random
import sys
from crap import NOTES_FOR_CHORDS, SCALES_FOR_CHORDS

POPULATION_SIZE = 1000
MUTATION_RATE = 0.02
CROSSOVER_RATE = 0.5
SOLOCHUNK_RATE = 0.4
GENERATIONS = 100
GENES = [-1, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]


def getSolo(chords, melody, length, instrument):
    numSoloSplits = 8

    # build weight curves
    closeness_percents = [
        0.38 * (math.cos((math.pi * 2 * i / numSoloSplits) + (math.pi / (2 * numSoloSplits))) + 1)
        for i in range(numSoloSplits)
    ]
    correctness_percents = [1 - c for c in closeness_percents]

    # convert melody & chords to "chromosomes"
    melodychromosome = [-1] * int(length * 2)
    for n in melody:
        for o in range(int(n['offset'] * 2), int((n['offset'] + n['el'].quarterLength) * 2)):
            if isinstance(n['el'], note.Note):
                melodychromosome[o] = n['el'].pitch.pitchClass
    
    chordchromosome = []
    extended = chords + [{'el': -1, 'offset': length}]
    off = 0
    for i, n in enumerate(extended[:-1]):
        if isinstance(n['el'], harmony.ChordSymbol):
            while off < extended[i+1]['offset']:
                chordchromosome.append([p.pitch.pitchClass for p in n['el'].notes])
                off += 0.5

    scalechromosome = []
    for ch in chordchromosome:
        base = ch[0]
        norm = [(i - base) % 12 for i in ch]
        sc = SCALES_FOR_CHORDS[list(NOTES_FOR_CHORDS.keys())[list(NOTES_FOR_CHORDS.values()).index(norm)]]
        scalechromosome.append([(i + base) % 12 for i in sc])

    def chunk(li, n):
        d, r = divmod(len(li), n)
        return [li[i * d + min(i, r):(i + 1) * d + min(i+1, r)] for i in range(n)]

    chordsplits = chunk(chordchromosome, numSoloSplits)
    melodysplits = chunk(melodychromosome, numSoloSplits)
    scalesplits = chunk(scalechromosome, numSoloSplits)

    def initializePopulation(splitlen):
        pop = []
        for _ in range(POPULATION_SIZE):
            chrom = random.choices(GENES, k=splitlen)
            j = 0
            while j < len(chrom):
                if chrom[j] == -1:
                    run = random.randint(2, 5)
                    for k in range(j, min(j+run, len(chrom))):
                        chrom[k] = -1
                    j += run
                else:
                    j += 1
            pop.append(chrom)
        return pop

    def correctness(chromosome, whichsplit):
        chordsplit = chordsplits[whichsplit]
        scalesplit = scalesplits[whichsplit]
        melodys = melodysplits[whichsplit]
        score = 0
        for i, gene in enumerate(chromosome):
            # winning conditions
            if gene in chordsplit[i] or gene == melodys[i]: score += .05
            elif ((gene + 1) % 12) in chordsplit[i] or ((gene - 1) % 12) in chordsplit[i]: score -= .125
            if gene in scalesplit[i]: score += 1.2
            else: score -= .2
            if i>0 and gene == chromosome[i-1]: score += .2
            if i>1 and gene==chromosome[i-1]==chromosome[i-2]: score += .2
            if i>3 and len({chromosome[i-k] for k in range(5)})==1: score -= 2
            if i>0:
                diff = abs((gene%12)-(chromosome[i-1]%12))
                score += (0.25 if diff<3 else -0.25)
        pR = chromosome.count(-1)/len(chromosome)
        if pR>0.25 or pR<0.1: score -= .25
        return (math.tanh(score)+1)/2

    def closeness(chromosome, whichsplit):
        mel = melodysplits[whichsplit]
        count = 0
        for i in range(len(chromosome)-1):
            if abs((chromosome[i+1]%12)-(chromosome[i]%12)) == abs((mel[i+1]%12)-(mel[i]%12)):
                count += 1
        for i in range(len(chromosome)):
            if chromosome[i] == mel[i]: count += 1
        return count/(2*len(chromosome))

    def nextGeneration(pop, fitnessScores, whichsplit):
        new = []
        for _ in range(POPULATION_SIZE):
            p1 = random.choices(pop, weights=fitnessScores)[0]
            p2 = p1
            while p2 == p1:
                p2 = random.choices(pop, weights=fitnessScores)[0]
            if random.random() < CROSSOVER_RATE:
                pt = random.randint(0, len(p1))
                child = p1[pt:]+p2[:pt]
            else:
                child = random.choice([p1,p2])
            if random.random()<SOLOCHUNK_RATE:
                a = random.randint(0,len(child)-1)
                b = random.randint(a,len(child)-1)
                chunk = melodychromosome[int(length*whichsplit//4 + a):int(length*whichsplit//4 + b)]
                for i,cg in enumerate(chunk): child[i]=cg
            new.append(child)
        return new

    def generateSplit(whichsplit):
        pop = initializePopulation(len(chordsplits[whichsplit]))
        tenthbest = 0
        x = 0
        while tenthbest<0.975 and x<400:
            # build lexicographic entries
            entries = []
            for chrom in pop:
                corr = correctness(chrom, whichsplit)
                close = closeness(chrom, whichsplit)
                entries.append((chrom,corr,close))
            # choose key
            if whichsplit in (0, numSoloSplits-1):
                keyfn = lambda e: (e[1], e[2])
            elif whichsplit==numSoloSplits//2:
                keyfn = lambda e: (e[2], e[1])
            else:
                keyfn = lambda e: (e[1], e[2])
            sorted_entries = sorted(entries, key=keyfn, reverse=True)
            top = sorted_entries[:POPULATION_SIZE//2]
            pop = [e[0] for e in top]
            fitnessScores = [e[1]*correctness_percents[whichsplit] + e[2]*closeness_percents[whichsplit] for e in top]
            pop = nextGeneration(pop, fitnessScores, whichsplit)
            tenthbest = sorted(fitnessScores, reverse=True)[4]
            x+=1

        # pick champion
        best_chrom, best_corr, best_close = sorted_entries[0]
        print(f"Split {whichsplit}: corr={best_corr:.4f}, close={best_close:.4f}")
        return best_chrom

    melodies = [generateSplit(i) for i in range(numSoloSplits)]
    solo = stream.Part()
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
    print(length)
    print(closeness_percents)
    print(correctness_percents)
    sc = stream.Score()
    for chord_entry in chords:
        chord_symbol = chord_entry['el']
        offset = chord_entry['offset']
        if isinstance(chord_symbol, harmony.ChordSymbol):
            solo.insert(offset, chord_symbol)
    sc.append(solo)
    solo.show()
    
    exit()

# driver
from checkplayable import get_chords
if __name__ == "__main__":
    chords, melody, length, inst = get_chords("../leads/AllOfMe.musicxml")
    getSolo(chords, melody, length, "horn")
