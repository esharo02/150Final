from music21 import *
import argparse
from ingestlead import get_chords
import copy
import re
import random
from fractions import Fraction
from solo import *
from constants import *

def main():

    argparser = argparse.ArgumentParser(description="Choose leadsheet file to run")
    argparser.add_argument("file", type=str, help="The file to be ran")
    argparser.add_argument("--swing", action='store_true', help="Whether to swing the file or not")
    argparser.add_argument('-m', dest='mode', action='store_const', const='-m', help='Midi output')
    argparser.add_argument('-s', dest='mode', action='store_const', const='-s', help='Sheet music output. Default.')

    args = argparser.parse_args()
    midi = args.mode == "-m"
    FILE = args.file

    chords, melody, length, t, isFourFour = get_chords(FILE)
    if len(chords) == 0:
        print("Error: No chords found in the file.")
        return

    # solo, lengths = getSolo(chords, melody, length, "piano", True)
    # print(lengths)
    # exit()

    theScore = buildScore(chords, melody, length, t.getQuarterBPM(), args.swing, midi, isFourFour)
    if midi:
        theScore.show('midi')
    else:
        theScore.show()
    
def buildScore(chords, melody, length, t, swing, midi, isFourFour):
    """
    Builds a jazz score with the given chords and melody. Head, Bass solo, Piano solo, Horn solo, and Head again.
    
    Reutrns the Score object.
    """

    theScore = stream.Score()
    t = tempo.MetronomeMark(number=t, referent="quarter") if t else None
    theScore.append(t)

    pianoPart = getPianoPart(chords, length, t, swing)
    pianoPart.makeMeasures(inPlace=True)

    pianoTrades, hornTrades = getTrades(chords, melody, length)

    pianoSoloPart = getPianoSoloPart(chords, melody, length, t, swing, pianoTrades)
    pianoSoloPart.makeMeasures(inPlace=True)
    
    hornPart = getHornPart(chords, melody, length, t, swing, midi, hornTrades)
    hornPart.makeMeasures(inPlace=True)

    bassPart = getBassPart(chords, melody, length, t, swing)
    bassPart.makeMeasures(inPlace=True)

    if isFourFour and swing:
        drumParts = get_drums(length, t, midi)
        for p in drumParts:
            p.makeMeasures(inPlace=True)
    else: drumParts = []

    longestLength = max(
        max(pianoPart.quarterLength, pianoSoloPart.quarterLength, hornPart.quarterLength, bassPart.quarterLength), 
        max([p.quarterLength for p in drumParts]) if drumParts else 0)
    
    fillUpToLength(hornPart, longestLength)
    hornPart.insert(0, dynamics.Dynamic('f'))
    hornPart.makeMeasures(inPlace=True)
    theScore.append(hornPart)

    fillUpToLength(pianoSoloPart, longestLength)
    pianoSoloPart.insert(0, dynamics.Dynamic('f'))
    pianoSoloPart.makeMeasures(inPlace=True)
    theScore.append(pianoSoloPart)

    fillUpToLength(pianoPart, longestLength)
    pianoPart.insert(0, dynamics.Dynamic('mf'))
    pianoPart.insert(length, dynamics.Dynamic('mp'))
    pianoPart.insert(length * 4, dynamics.Dynamic('mf'))
    pianoPart.makeMeasures(inPlace=True)
    theScore.append(pianoPart)

    fillUpToLength(bassPart, longestLength)
    bassPart.insert(0, dynamics.Dynamic('mf'))
    bassPart.makeMeasures(inPlace=True)
    theScore.append(bassPart)

    if drumParts:
        for p in drumParts:
            fillUpToLength(p, longestLength)
            p.insert(0, dynamics.Dynamic('mp'))
            p.makeMeasures(inPlace=True)
            theScore.append(p)

    return theScore

def getTrades(chords, melody, length):
    solo, lengths = getSolo(chords, melody, length, "piano", True)
    pianoTrades = []
    hornTrades = []
    curOffset = 0
    curLengthIdx = 0
    curLength = lengths[0]
    for n in solo:
        if curOffset >= curLength:
            curLengthIdx += 1
            curLength += lengths[curLengthIdx]
        if curLengthIdx % 2 == 0:
            pianoTrades.append(n)
            hornTrades.append(note.Rest(quarterLength=n.quarterLength))
        else: 
            hornTrades.append(n)
            pianoTrades.append(note.Rest(quarterLength=n.quarterLength))
        curOffset += n.quarterLength
    return pianoTrades, hornTrades

def fillUpToLength(part, length):
    while part.quarterLength < length:
        part.append(note.Rest())
    if part.quarterLength > length:
        part[-1].quarterLength -= part.quarterLength - length

def swingify(instream, inst):
    """
    WHY music21, WHY?!?!?!?!?!
    10+ hours of our lives spent on this function alone...
    """
    t = instream.getElementsByClass(tempo.MetronomeMark)
    instream.makeMeasures(inPlace=True)
    instream.makeRests(fillGaps=True, inPlace=True)
    newPart = stream.Part()
    newPart.append(inst)
    if t: newPart.append(t[0])
    s = instream.flatten()
    notesAndRests = list(s.notesAndRests)
    for i, elem in enumerate(notesAndRests):
        if elem.offset % 1 == 0.5 and elem.duration.quarterLength >= 0.5 and not isinstance(elem, note.Rest):
            offset = newPart.highestOffset
            prevElem = newPart.pop()
            prevElem.offset = offset
            prevElem.quarterLength += Fraction(1, 6)
            if prevElem.duration.quarterLength > Fraction(2, 3):
                if isinstance(prevElem, note.Rest):
                    newElem = note.Rest(quarterLength=prevElem.duration.quarterLength - Fraction(2, 3))
                elif isinstance(prevElem, note.Note):
                    newElem = note.Note(prevElem.pitch, quarterLength=prevElem.duration.quarterLength - Fraction(2, 3))
                elif isinstance(prevElem, chord.Chord):
                    newElem = chord.Chord(prevElem.pitches, quarterLength=prevElem.duration.quarterLength - Fraction(2, 3))
                elif isinstance(elem, note.Unpitched):
                    newElem = note.Unpitched(displayName=prevElem.displayName, quarterLength=prevElem.duration.quarterLength - Fraction(1, 6))
                    newElem.notehead = prevElem.notehead
                newElem.offset = prevElem.offset
                if not isinstance(prevElem, note.Rest):
                    newElem.tie = tie.Tie('start')
                newPart.insert(newElem.offset, newElem)
                if isinstance(prevElem, note.Rest):
                    prevElem = note.Rest(quarterLength=Fraction(2, 3))
                elif isinstance(prevElem, note.Note):
                    prevElem = note.Note(prevElem.pitch, quarterLength=Fraction(2, 3))
                elif isinstance(prevElem, chord.Chord):
                    prevElem = chord.Chord(prevElem.pitches, quarterLength=Fraction(2, 3))
                elif isinstance(elem, note.Unpitched):
                    temp = note.Unpitched(displayName=prevElem.displayName, quarterLength=prevElem.duration.quarterLength - Fraction(1, 6))
                    temp.notehead = prevElem.notehead
                    prevElem = temp
                prevElem.offset = elem.offset - 0.5
                if not isinstance(prevElem, note.Rest):
                    prevElem.tie = tie.Tie('stop')
                newPart.insert(prevElem.offset, prevElem)
            else:
                prevElem.duration.appendTuplet(duration.Tuplet(3, 2))
                prevElem.offset = elem.offset - 0.5
                newPart.insert(prevElem.offset, prevElem)
            if isinstance(elem, note.Rest):
                newElem = note.Rest(quarterLength=elem.duration.quarterLength - Fraction(1, 6))
            elif isinstance(elem, note.Note):
                newElem = note.Note(elem.pitch, quarterLength=elem.duration.quarterLength - Fraction(1, 6))
            elif isinstance(elem, chord.Chord):
                newElem = chord.Chord(elem.pitches, quarterLength=elem.duration.quarterLength - Fraction(1, 6))
            elif isinstance(elem, note.Unpitched):
                newElem = note.Unpitched(displayName=elem.displayName, quarterLength=elem.duration.quarterLength - Fraction(1, 6))
                newElem.notehead = elem.notehead
            if elem.tie:
                newElem.tie = elem.tie
            newElem.duration.appendTuplet(duration.Tuplet(3, 2))
            elem.offset = elem.offset + Fraction(1, 6)
            newPart.insert(elem.offset, newElem)
        else:
            newPart.insert(elem.offset, elem)
        prevElem = elem
    return newPart

def getHornPart(chords, melody, length, t, swung, midi, hornTrades):
    hornPart = stream.Part()
    hornPart.append(instrument.AltoSaxophone())
    hornPart.append(clef.TrebleClef())

    if t is not None:
        hornPart.append(t)

    #Head
    for n in melody:
        hornPart.insert(n["offset"], n["el"])

    #Piano solo
    fillUpToLength(hornPart, length * 2)

    #Horn solo
    solo, _ = getSolo(chords, melody, length, "horn")
    hornPart.append(solo)

    #Trading fours
    for n in hornTrades:
        hornPart.append(n)

    #Head
    for n in melody:
        hornPart.insert(n["offset"] + length * 4, copy.deepcopy(n["el"]))

    #Midi doesn't need transpose (at least in garageband, hope this is true for all DAWs)
    if not midi:
        hornPart.transpose("M6", inPlace=True) 

    hornPart.makeMeasures(inPlace=True)
    hornPart.makeTies(inPlace=True)
    newPart = hornPart if not swung else swingify(hornPart, instrument.AltoSaxophone())
    # newPart.show('text', addEndTimes=True, addStartTimes=True, addOffsets=True, addDurations=True, addClefs=True, addInstruments=True)
    # exit()
    return newPart

def getBassPart(chords, melody, length, t, swung):
    bassPart = stream.Part()
    bassPart.append(instrument.AcousticBass())
    bassPart.append(clef.BassClef())

    if t is not None:
        bassPart.append(t)

    for _ in range(5):
        for n in getBassPattern(chords, length):
            bassPart.append(n)

    bassPart.makeMeasures(inPlace=True)
    bassPart.makeTies(inPlace=True)
    newPart = bassPart if not swung else swingify(bassPart, instrument.AcousticBass())
    return newPart

def getBassPattern(chords, length):
    """
    Generates a bass pattern based on the given chords and length.
    The pattern is generated using a Markov chain based on the chord quality and notes in the chord.

    Returns a list of Note objects representing the bass pattern.
    """
    bassPattern = []
    chordIndex = -1
    curChain = {}
    chordNotes = []
    for offset in range(int(length)):
        #Looking at a new chord
        if chordIndex == -1 or (chordIndex != len(chords) - 1 and chords[chordIndex + 1]["offset"] <= offset):
            chordIndex += 1
            quality = chordSymbolToQuality(chords[chordIndex]["el"].figure)
            curRoot = chords[chordIndex]['el'].root().midi
            curChain = MARKOVS[quality]
            chordNotes = NOTES_FOR_CHORDS[quality]

            #First note for a new chord is the root, 3rd if present, or 5th if present
            choices = [i for i in list(set([0, 3, 4, 7]).intersection(chordNotes))]
            n = note.Note(random.choice(choices) + curRoot - 12)
        else:  
            #Use the Markov chain to generate the next note
            row = curChain[(bassPattern[-1].pitch.midi - curRoot) % 12]
            n = note.Note(random.choices(list(row.keys()), weights=list(row.values()), k=1)[0] + curRoot - 12)

        shuffledChordNotes = random.sample(chordNotes, len(chordNotes))
        resolveNote = None
        r = random.random()

        # If the note is not in the chord and spicy (one away from a chord tone), resolve it to a close chord tone
        if r < 0.4 and (n.pitch.midi - curRoot) % 12 not in chordNotes: 
            for i in shuffledChordNotes:
                if abs(((n.pitch.midi - curRoot) % 12) - i) == 1:
                    resolveNote = note.Note(i + curRoot - 12)
                    break
                    
            if resolveNote is None:
                for i in shuffledChordNotes:
                    if abs(((n.pitch.midi - curRoot) % 12) - i) == 2:
                        resolveNote = note.Note(i + curRoot - 12)
                        break

        if resolveNote is not None:
            n.duration.quarterLength = 0.5
            bassPattern[-1].quarterLength = 0.5
    
        bassPattern.append(n)

        if resolveNote is not None:
            bassPattern.append(resolveNote)

    return bassPattern

def makeMarkovChain(chord, scale):
    """
    Creates a Markov chain for the given chord and scale. 

    Returns a dictionary where the keys are previous notes and the values are dictionaries 
    of next notes with their probabilities.
    """
    mc = {}
    for prev in scale:
        row = {}
        buckets = [0] * len(scale) 
        for i, choice in enumerate(scale): 
            buckets[i] += 1 # each note has a chance of being selected
            if choice in chord and prev not in chord: # notes in the chord have a higher chance of being selected
                buckets[i] += 1
            if choice not in chord and prev in chord:
                buckets[i] += 1
            if choice == prev:
                if i != 0:
                    buckets[(i - 1)] += 1 # the note before it has additional chance of being selected
                buckets[i] == 0
                if i != len(scale) - 1:
                    buckets[(i + 1)] += 1 # the note after it has additional chance of being selected
        buckets[scale.index(prev)] == 0
        bucketrates = [0.95 * i / sum(buckets) for i in buckets]
        
        for i, choice in enumerate(scale):
            if prev == choice: 
                row[choice] = 0.05 # Low chance of repeated note
            else:
                row[choice] = bucketrates[i]
        mc[prev] = row
    return mc

MARKOVS = {k: makeMarkovChain(NOTES_FOR_CHORDS[k], SCALES_FOR_CHORDS[k]) for k in SCALES_FOR_CHORDS.keys()}

def chordSymbolToQuality(s):
    """
    Converts a chord symbol to its quality, removing its root and inversion.

    Returns the chord quality as a string.
    """
    match = re.search(r'[^A-Z]', s)
    if match:
        q = s[match.start():]
        if q[0] == '-' or q[0] == '#':
            q = q[1:]
        if '/' in q:
            q = q.split('/')[0]
        return q
    return ''

def getPianoPart(chords, length, t, swung):
    pianoPart = stream.Part()
    pianoPart.append(instrument.Piano())

    if t is not None:
        pianoPart.append(t)

    for _ in range(5):
        for n in getPianoPattern(chords, length):
            pianoPart.append(n)

    pianoPart.makeMeasures(inPlace=True)
    pianoPart.makeTies(inPlace=True)
    newPart = pianoPart if not swung else swingify(pianoPart, instrument.Piano())
    return newPart

def getPianoPattern(chords, length):
    """
    Generates a piano pattern based on the given chords and length.
    Rhythms are generated by a Monte Carlo method, and chord notes are chosen randomly 
    based on their interval from the root.

    Returns a list of Note objects representing the piano pattern.
    """
    chordIndex = -1
    pianoPattern = []
    totalmap = []
    for offset in range(int(length)):
        #Looking at a new chord
        if chordIndex == -1 or (chordIndex != len(chords) - 1 and chords[chordIndex + 1]["offset"] <= offset):
            chordIndex += 1
            quality = chordSymbolToQuality(chords[chordIndex]["el"].figure)
            choices = [i for i in NOTES_FOR_CHORDS[quality]]

            if chordIndex == len(chords) - 1:
                chord_length = length - chords[chordIndex]["offset"]

            elif chordIndex == 0:
                chord_length = chords[chordIndex + 1]["offset"]
                
            else:
                chord_length = chords[chordIndex + 1]["offset"] - chords[chordIndex]["offset"]

            rhythm = getRhythms(chord_length)
            mapping = mapRhythms(choices, chords[chordIndex]['el'].root().midi % 12, rhythm, chords[chordIndex]["offset"] % 4)
            totalmap.extend(mapping)
        #else:
            #Go until next chord starts to find length

    for el in totalmap:
        #Flatten notes
        if isinstance(el, list):
            for sub_el in el:
                pianoPattern.append(copy.deepcopy(sub_el))
        else:
            pianoPattern.append(copy.deepcopy(el))
        
    #Randomly choose octaves for notes in the pattern within reason
    for el in pianoPattern:
        if not isinstance(el, note.Rest):
            tryAgain = True
            while tryAgain:
                for n in el.notes:
                    n.octave = random.choice([3, 4])
                pitches = [n.midi for n in el.pitches]
                tryAgain = max(pitches) - min(pitches) > 15

    return pianoPattern

def getRhythms(length): 
    currentLength = 0
    rhythms = []
    # print(length)
    while currentLength != length:
        weights = SHORT_LENGTH_WEIGHTS if length < 6 else LONG_LENGTH_WEIGHTS
        curr = random.choices(LENGTH_OPTIONS, weights=weights, k=1)[0]
        # curr = random.choice(LENGTH_OPTIONS)
        if curr <= length - currentLength and \
            (len(rhythms) < 2 or rhythms[-1] != curr or rhythms[-1] != rhythms[-2] or currentLength == length - 0.5):
                rhythms.append(curr)
                currentLength += curr
    # print(length, sum(rhythms), rhythms)
    return rhythms

def mapRhythms(notes, root, rhythms, startingOffset): # notes will actually be note objects
    # comp = stream.Part()
    comp = []
    # comp.append(key.KeySignature(KEY_SIGNAURE))
    currentOffset = startingOffset
    addRest = False
    notesInChord = [note.Note(n + root) for n in notes]
    # print(notesInChord)
    currChordAmount = 0 # num chords played by accomp (not chords in the list)
    for i, rhythm in enumerate(rhythms):
        # print("Current offset:", currentOffset)
        # print("Current rhythm:", rhythm)
        # addRest = False ## no rests baby
       
        chordRate = 0.3 if len(rhythms) < 4 else 0.6
        rest_probability = MONTE_SHORT_LENGTH_REST_PROBABILITY if rhythm < 4 else MONTE_LONG_LENGTH_REST_PROBABILITY
        addRest = ((random.random() <= rest_probability and len(rhythms) > 1) or currChordAmount / rhythm > chordRate * len(rhythms)) and not (i == len(rhythms) - 1 and currChordAmount == 0)
         
        if not addRest:
            chordNotes = []
            while len(chordNotes) < 2:
                for n in notesInChord:
                    noteIntervalFromRoot = abs((n.pitch.midi - root) % 12)
                    if noteIntervalFromRoot == 0: # roots
                        prob = 0.15
                    elif noteIntervalFromRoot == 3 or noteIntervalFromRoot == 4:
                        prob = 0.9
                    elif noteIntervalFromRoot == 7:
                        prob = 0.25
                    elif noteIntervalFromRoot == 10 or noteIntervalFromRoot == 11:
                        prob = 0.9
                    else:
                        prob = 0.5
                    stat = random.random()
                    # print(stat, prob)
                    if stat < prob and n not in chordNotes:
                        chordNotes.append(n)
                        

                # print(chordNotes)

        if currentOffset + rhythm > 4:
            curr1 = copy.deepcopy(chord.Chord(chordNotes)) if not addRest else note.Rest()  
            curr2 = copy.deepcopy(chord.Chord(chordNotes)) if not addRest else note.Rest()
            curr1.quarterLength = 4 - currentOffset
            curr2.quarterLength = rhythm - 4 + currentOffset
            # print(curr1.quarterLength, curr2.quarterLength, rhythm, currentOffset)
            if not addRest:
                curr1.tie = tie.Tie("start")
                curr2.tie = tie.Tie("stop")
                currChordAmount += rhythm
            comp.extend([curr1, curr2])
        else:
            if addRest: n = note.Rest(duration=duration.Duration(quarterLength=rhythm))  
            else: 
                n = chord.Chord(chordNotes, duration=duration.Duration(quarterLength=rhythm))
                currChordAmount += rhythm
            comp.append(n)
        currentOffset += rhythm
        currentOffset %= 4
    return comp

def getPianoSoloPart(chords, melody, length, t, swung, pianoTrades):
    pianoPart = stream.Part()
    pianoPart.append(instrument.Piano())

    if t is not None:
        pianoPart.append(t)

    #Head
    fillUpToLength(pianoPart, length)

    #Piano solo
    solo, _ = getSolo(chords, melody, length, "piano")
    pianoPart.append(solo)

    #Horn solo
    fillUpToLength(pianoPart, length * 3)

    #Trading fours
    for n in pianoTrades:
        pianoPart.append(n)

    #Head
    fillUpToLength(pianoPart, length * 5)

    pianoPart.makeMeasures(inPlace=True)
    pianoPart.makeTies(inPlace=True)
    newPart = pianoPart if not swung else swingify(pianoPart, instrument.Piano())
    return newPart

def get_drums(length, t, midi):
    #Drums coded different in midi and sheet music
    if midi:
        rideNoteQuarter = note.Note('D#3')
        hihatNote = note.Note('G#2')
        bassNote = note.Note('C2')
        snareNote = note.Note('D2')
    else:
        rideNoteQuarter = note.Unpitched(displayName='F5')
        hihatNote = note.Unpitched(displayName='D4')
        bassNote = note.Unpitched(displayName='F4')
        snareNote = note.Unpitched(displayName='C5')

    rideNoteTrip1 = copy.deepcopy(rideNoteQuarter)
    rideNoteTrip1.duration = duration.Duration(0.5)
    rideNoteTrip2 = copy.deepcopy(rideNoteTrip1)
    rideNoteTrip2.duration = duration.Duration(0.5)
    quarterRest = note.Rest()
    rideNoteQuarter.notehead = 'x'
    rideNoteTrip1.notehead = 'x'
    rideNoteTrip2.notehead = 'x'
    hihatNote.notehead = 'x'
    inst = instrument.UnpitchedPercussion()

    ridePattern = [
        copy.deepcopy(rideNoteQuarter),
        copy.deepcopy(rideNoteTrip1),
        copy.deepcopy(rideNoteTrip2),
        copy.deepcopy(rideNoteQuarter),
        copy.deepcopy(rideNoteTrip1),
        copy.deepcopy(rideNoteTrip2)
    ]

    hihatPattern = [
        copy.deepcopy(quarterRest),
        copy.deepcopy(hihatNote),
        copy.deepcopy(quarterRest),
        copy.deepcopy(hihatNote)
    ]

    bassPattern = [
        copy.deepcopy(bassNote),
        copy.deepcopy(quarterRest),
        copy.deepcopy(bassNote),
        copy.deepcopy(quarterRest)
    ]

    snarePattern = [
        copy.deepcopy(quarterRest),
        copy.deepcopy(snareNote),
        copy.deepcopy(quarterRest),
        copy.deepcopy(snareNote)
    ]

    ridePart = stream.Part()
    ridePart.insert(0, inst)
    if t is not None:
        ridePart.append(t)
    
    hihatPart = stream.Part()
    hihatPart.insert(0, inst)
    if t is not None:
        hihatPart.append(t)

    bassPart = stream.Part()
    bassPart.insert(0, inst)
    if t is not None:
        bassPart.append(t)

    snarePart = stream.Part()
    snarePart.insert(0, inst)
    if t is not None:
        snarePart.append(t)
    
    for _ in range(int((length * 5) / 4)):
        for el in ridePattern:
            ridePart.append(copy.deepcopy(el))
        for el in hihatPattern:
            hihatPart.append(copy.deepcopy(el))
        for el in bassPattern:
            bassPart.append(copy.deepcopy(el))
        for el in snarePattern:
            snarePart.append(copy.deepcopy(el))
    
    return [swingify(ridePart, inst), swingify(hihatPart, inst), swingify(bassPart, inst), swingify(snarePart, inst)]

if __name__ == "__main__":
    main()
