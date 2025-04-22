from music21 import *
import math
import random
import sys
from constants import *

def getSolo(chords, melody, length, instrument, trades=False):

    # define a number of sections to split the solo up into. do GA on each of these sections
    numSoloSplits = int(length // 16) # parameterize this?
    melodies = []
    # general idea is that we want to be pretty close to the melody at the beginning of the solo,
    # then less close midway through the solo, then back to close again by the end
    # so define some numbers "what percentage of the fitness function should be defined by closeness"
    # for each split

    if not trades:

        # closeness follows cosine, 0.75 at 0th split, 0.0 at middle of solo, 0.75 at last split (approx)
        closeness_percents = \
            [(math.cos((math.pi * 2 * i / numSoloSplits) + 
                            (math.pi / (2 * numSoloSplits))) + 3) / 4
                    for i in range(numSoloSplits)]
        if instrument == "horn":
            mod = 0.4
        elif instrument == "piano":
            mod = 0.4
        else:
            mod = 0.3

    else:
        closeness_percents = [0.2] + [0.6 for _ in range(numSoloSplits - 1)]
        mod = 1
    
    closeness_percents = [i * mod for i in closeness_percents]
    correctness_percents = [1 - i for i in closeness_percents]


    # convert melody to chromosome for use in GA "chunk of melody" mutation and fitness
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

    # and for scales
    scalechromosome = []
    for ch in chordchromosome:
        temp = list(set([(i - ch[0]) % 12 for i in ch])) # normalize to C
        # python sucks
        try:
            tempscale = SCALES_FOR_CHORDS[list(NOTES_FOR_CHORDS.keys())[list(NOTES_FOR_CHORDS.values()).index(sorted(temp))]]
        except ValueError:
            print(ch)
        tempscale = [(i + ch[0]) % 12 for i in tempscale]
        scalechromosome.append(tempscale)

    
    def chunk(li, n):
        """ Split a list into n chunks of approximately equal size. Stackoverflow helped here. """ 
        d, r = divmod(len(li), n)
        return [li[i * d + min(i, r):(i + 1) * d + min(i + 1, r)] for i in range(n)]
    
    # chunk chromosomes into splits
    chordsplits = chunk(chordchromosome, numSoloSplits)
    melodysplits = chunk(melodychromosome, numSoloSplits)
    scalesplits = chunk(scalechromosome, numSoloSplits)

    # define GA functions:
    def initializePopulation(splitlength, whichsplit):
        """ Initialize the population of melodies. 
        The population is a list of lists of notes. Each note is represented by a 
        number from 0, where 0 is C, 1 is C#, 2 is D, etc. -1 is a rest."""
        population = []
        for i in range(POPULATION_SIZE):
            chromosome = [-1 for _ in range(splitlength)]
            for j in range(len(chordsplits[whichsplit])):
                octaveUp = [scalesplits[whichsplit][j][k] + 12 for k in range(len(scalesplits[whichsplit][j]))]
                halfOctaveUp = [octaveUp[k] for k in range(len(octaveUp)) if octaveUp[k] <= 18]
                choices = [-1, *scalesplits[whichsplit][j], *halfOctaveUp]
                gene = random.choice(choices)
                chromosome[j] = gene
                if gene == -1: # if we get 1 rest, we should make the next few notes rests as well
                    rest_length = random.randint(2, 5)
                    for k in range(j, min(j + rest_length, len(chromosome))):
                        chromosome[k] = -1
                        j += rest_length - 1
            population.append(chromosome)
        return population

    def local_hill_climb(child, whichsplit):
        """
        Perform one-step hill climb on child:
        - pick a random index k
        - for each chord tone at that beat, try swapping in
        - keep the swap if it raises correctness()
        """
        # evaluate current
        current_score = correctness(child, whichsplit)
        best_child  = child
        best_score  = current_score

        # pick a random gene to tweak
        k = random.randrange(len(child))
        # try each valid chord-tone at that position
        for tone in chordsplits[whichsplit][k]:
            if tone == child[k]:
                continue
            candidate = child.copy()
            candidate[k] = tone
            cand_score = correctness(candidate, whichsplit)
            if cand_score > best_score:
                best_score  = cand_score
                best_child  = candidate

        return best_child

    def nextGeneration(population, fitnessScores, whichsplit):
        def tournament_selection(population, fitnessScores, k=3):
            candidates = random.sample(list(zip(population, fitnessScores)), k)
            return max(candidates, key=lambda x: x[1])[0]

        """ Create the next generation of the population. 
        The next generation is created by selecting two parents from the population
        based on their fitness. The child is created by crossing over the parents at
        a random point, if crossover was selected. The child is then mutated at a 
        random point, if mutation was selected. Both of these can occur on the same
        child. The child is then added to the new population. """
        newPopulation = []
        for _ in range(POPULATION_SIZE):
            # parent1 = random.choices(population, weights=fitnessScores)[0]
            # parent2 = random.choices(population, weights=fitnessScores)[0]
            parent1 = tournament_selection(population, fitnessScores)
            parent2 = tournament_selection(population, fitnessScores)
            child = []
            if random.random() < CROSSOVER_RATE:
                crossoverPoint = random.randint(0, len(parent1))
                child = parent1[:crossoverPoint] + parent2[crossoverPoint:]
            else:
                child = random.choice([parent1, parent2])
            for j in range(len(child)):
                if random.random() < MUTATION_RATE:
                    octaveUp = [scalesplits[whichsplit][j][k] + 12 for k in range(len(scalesplits[whichsplit][j]))]
                    if instrument == "piano":
                        halfOctaveUp = [octaveUp[k] for k in range(len(octaveUp)) if octaveUp[k] <= 18]
                        choices = [-1, *scalesplits[whichsplit][j], *halfOctaveUp]
                    else:
                        halfOctaveDown = [octaveUp[k] - 12 for k in range(len(octaveUp)) if octaveUp[k] - 12 >= 0]
                        choices = [-1, *halfOctaveDown, *scalesplits[whichsplit][j]]
                    child[j] = random.choice(choices)
            if random.random() < CHORD_TONE_MUTATION_RATE:
                for j, gene in enumerate(child):
                    if gene % 12 not in chordsplits[whichsplit][j]:
                        child[j] = random.choice(chordsplits[whichsplit][j])
            if random.random() < SOLOCHUNK_RATE:
                # randomly replace a chunk of the child with the same chunk from the melody
                chunkStart = random.randint(0, len(child) - 1)
                chunkEnd = random.randint(chunkStart, len(child) - 1)
                chunk = melodychromosome[int(length * whichsplit // 4 + chunkStart):int(length * whichsplit // 4 + chunkEnd)]
                for j in range(len(chunk)):
                    child[j] = chunk[j]
            if random.random() < QUOTESELF_RATE and whichsplit >= 1:
                # randomly replace a chunk of the child with a chunk from a previous split
                # and then make sure it fits in the new chords
                chunkStart = random.randint(0, len(child) - 1)
                chunkEnd = random.randint(chunkStart, len(child) - 1)
                # pick a random split that's already been completed
                samplesplit = random.randint(0, whichsplit - 1)
                # chunk = population[random.randint(0, POPULATION_SIZE // 2)][int(length * (whichsplit - 1) // 4 + chunkStart):int(length * (whichsplit - 1) // 4 + chunkEnd)]
                for j in range(chunkEnd - chunkStart):
                    # print(f"Chunk {chunkStart} to {chunkEnd} from {whichsplit} to {samplesplit}", file=sys.stderr)
                    # print(f"Length of child: {len(child)}", file=sys.stderr)
                    # print(f"Length of melodies: {len(melodies[samplesplit])}", file=sys.stderr)
                    child[j] = melodies[samplesplit][j + chunkStart]
                    # make sure the chunk fits in the new chords
                    if child[j] not in chordsplits[whichsplit][j]:
                        child[j] = random.choice(chordsplits[whichsplit][j])
            child = local_hill_climb(child, whichsplit)
            newPopulation.append(child)
        elite = [x for _, x in sorted(zip(fitnessScores, population), reverse=True)[:ELITE_COUNT]]
        newPopulation = elite + newPopulation[:-ELITE_COUNT]
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
            # print(f"Split {whichsplit}: {chromosome} {fitnessScores[i]} {corr} * {correctness_percents[whichsplit]}", file=sys.stderr)
            # exit()
            clo = closeness(chromosome, whichsplit)
            fitnessScores[i] += clo * closeness_percents[whichsplit]
            # if fitnessScores[i] > 0.9:
                # print(f"Split {whichsplit}: {chromosome} {fitnessScores[i]} {corr} {clo}")
        return fitnessScores

    def correctness(chromosome, whichsplit):
        chordsplit = chordsplits[whichsplit]
        scalesplit = scalesplits[whichsplit]
        melodysplit = melodysplits[whichsplit]
        longNoteCount = 0
        stepwiseCount = 2
        score = 0
        for i, ((gene, chord), (mNote, scale)) in enumerate(zip(zip(chromosome, chordsplit), zip(melodysplit, scalesplit))):
            if gene == -1: # rest
                score += 1
            else:
                if gene % 12 in chord or gene % 12 == mNote: # if the note is in the chord or the melody, yay
                    score += 1.5
                    if i == 0 or (chordsplit[i-1] != chordsplit[i]):
                        score += 2
                elif (gene + 1) % 12 in chord or (gene - 1) % 12 in chord: # elif because we sometimes this the melody note is 1 away from a chord tone
                    j = i
                    while j > 0 and gene % 12 == chromosome[j] % 12:
                        score -= 2
                        j -= 1
                # if gene % 12 not in chord: # play notes in the chord!
                #     score -= 0.5
                if i > 0 and gene % 12 == chromosome[i - 1] % 12:
                    score += .8
                if i > 1 and gene % 12 == chromosome[i - 1] % 12 and gene % 12 == chromosome[i - 2] % 12:
                    score += 1.4
                if i > 7 and all(gene == chromosome[i - j] for j in range(1, 8)):
                    score -= 6  # Penalize heavily for the same note repeated 8 times in a row
                if i > 3 and all(gene == chromosome[i - j] for j in range(1, 4)) and gene % 12 in chord:
                    score += (4 - (1 * min(longNoteCount, 2))) if gene in chord else -3
                    longNoteCount += 1 # not too long of the same note
                if i > 0 and abs((gene) - (chromosome[i - 1])) > 5:
                    if instrument != "piano":
                        score -= 3 
                    else:
                        score -= 1 # Less bad for piano
                if i > 0 and abs((gene) - (chromosome[i - 1])) > 7:
                    score -= 10
                    if instrument == "horn":
                        score -= 5 # this is even worse on sax
                # if i > 0 and 1 <= abs((gene) - (chromosome[i - 1])) < 3:
                #     score += 1
                if i > 0 and 1 <= abs((gene) - (chromosome[i - 1])) < 3:
                    score += 1.7 * stepwiseCount
                    stepwiseCount /= 2
                else: 
                    stepwiseCount = 2
        percentRests = chromosome.count(-1) / float(len(chromosome))
        if percentRests > 0.45 or percentRests < 0.2:
            score -= 10

        # Count the number of attacks (notes that play after rests, or notes that change)
        attacks = sum(1 for i in range(1, len(chromosome)) if chromosome[i] != chromosome[i - 1] and chromosome[i] != -1)
        expectedattacks = 0
        if instrument == "bass":
            expectedattacks = 0.5 * len(chromosome) # one attack per beat
        elif instrument == "piano" or instrument == "horn":
            expectedattacks = 0.75 * len(chromosome) # one attack every 3/4 of a beat on average
        score -= abs(expectedattacks - attacks) * 1.6
        # print(score)
        score /= 100
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
        while best < 0.95 and x < 400:
            
            fitnessScores = getFitnessScores(pop, whichsplit)
            correctnessScores = [correctness(chromosome, whichsplit) for chromosome in pop]
            closenessScores = [closeness(chromosome, whichsplit) for chromosome in pop]
            
            tenthbest = sorted(fitnessScores, reverse=True)[9]
            best = sorted(fitnessScores, reverse=True)[0]
            best_idx = fitnessScores.index(best)    
            tenthbestcorr = correctnessScores[best_idx]
            tenthbestclose = closenessScores[best_idx]
            average = sum(fitnessScores) / len(fitnessScores)
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
                print(f"Generation {x} for split {whichsplit}, average = {average}, tenthbest = {tenthbest}, corr = {tenthbestcorr}, close = {tenthbestclose}", file=sys.stderr)
            
        # best = sorted(fitnessScores, reverse=True)[0]
        # best_idx = fitnessScores.index(best)    
        # print(f"Split {whichsplit}: corr={correctnessScores[best_idx]:.4f} at {correctness_percents[whichsplit]}, close={closenessScores[best_idx]:.4f} at {closeness_percents[whichsplit]}")
        return pop[best_idx]
    
    def transposeSolo(solo):
        # Transpose the solo to the range of the instrument
        if trades:
            curNoteIdx = 0
            curNoteOffset = 0
            offset = 0
            for i, split in enumerate(melodysplits):
                offset += len(split) / 2
                while curNoteOffset < offset:
                    # Piano, Sax, Piano, Sax, etc.
                    if i % 2 == 0 and isinstance(solo[curNoteIdx], note.Note):
                        solo[curNoteIdx].transpose("P8", inPlace=True)
                    curNoteOffset += solo[curNoteIdx].quarterLength
                    curNoteIdx += 1
                    
        else:
            if instrument.lower() in ["piano"]:
                for n in solo:
                    if isinstance(n, note.Note):
                        n.transpose("P8", inPlace=True)
                    # n.transpose("P8", inPlace=True)
            elif instrument.lower() in ["bass"]:
                for n in solo:
                    if isinstance(n, note.Note):
                        n.transpose("-P15", inPlace=True)
                    n.transpose("-P15", inPlace=True)
        
    offset = 0
    for i in range(numSoloSplits):
        # print("Current melodychromosome:", melodychromosome)
        split = generateSplit(i)
        melodies.append(split)
        if trades and i < numSoloSplits - 1:
            if instrument == "horn":
                instrument = "piano:"
            else:
                instrument = "horn"
            melodychromosome = melodychromosome[:offset] + split + melodychromosome[offset + len(split):]
            # print("Keeping part of melodychromosome", melodychromosome[:offset])
            # print("New split:", split)
            # print("Length of melodychromosome:", len(melodychromosome))
            offset += len(split)



    # print(melodies)
    solo = []
    if trades:
        instrument = "piano"
    for melody in melodies:
        previous_note = None
        d = 0.5
        for note_value in melody:
            if note_value == -1:  # Handle rests
                if isinstance(previous_note, note.Rest):
                    d += 0.5
                else:
                    if previous_note:
                        previous_note.duration = duration.Duration(quarterLength=d)
                        solo.append(previous_note)
                    previous_note = note.Rest(quarterLength=0.5)
                    d = 0.5
            else:  # Handle notes
                if isinstance(previous_note, note.Note) and previous_note.pitch.pitchClass == note_value and (random.random() < 0.85 or instrument == "horn"):
                    d += 0.5
                else:
                    if previous_note:
                        previous_note.duration = duration.Duration(quarterLength=d)
                        solo.append(previous_note)
                    previous_note = note.Note(note_value + 60, quarterLength=0.5)
                    d = 0.5
        if previous_note:  # Add the last note or rest
            previous_note.duration = duration.Duration(quarterLength=d)
            solo.append(previous_note)
        if trades and instrument == "horn":
            instrument = "piano"
        elif trades and instrument == "piano":
            instrument = "horn"
    transposeSolo(solo)
    return solo, [len(split) / 2 for split in melodysplits]
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

    

    return solo

from ingestlead import get_chords
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