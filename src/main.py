from music21 import *
import argparse
from checkplayable import get_chords
import copy
import re
import random
from fractions import Fraction
from solo import *

#https://igm.rit.edu/~jabics/BilesICMC94.pdf
NOTES_FOR_CHORDS = {
    "": [0, 4, 7], 
    "6": [0, 4, 7, 9],
    "6 add 9": [0, 2, 4, 7, 9],
    "sus": [0, 4, 5, 7],
    "sus add 7": [0, 4, 5, 7, 10],
    "sus add 7 add 9": [0, 2, 4, 5, 7, 10],
    "maj7": [0, 4, 7, 11], 
    "maj7 add 9": [0, 2, 4, 7, 11],
    "maj7 add #11": [0, 4, 6, 7, 11],
    "7": [0, 4, 7, 10],
    "9": [0, 2, 4, 7, 10],
    "11": [0, 2, 4, 5, 7, 10],
    "13": [0, 2, 4, 5, 7, 9, 10],
    "M13": [0, 2, 4, 6, 7, 9, 11],
    "m7": [0, 3, 7, 10],
    "m9": [0, 2, 3, 7, 10],
    "m11": [0, 2, 3, 5, 7, 10],
    "m13": [0, 2, 3, 5, 7, 9, 10],
    "7 alter b5": [0, 4, 6, 10],
    "7 alter #5": [0, 4, 8, 10],
    "7 add 9": [0, 2, 4, 7, 10], 
    "7 add #9": [0, 3, 4, 7, 10], 
    "7 add b13": [0, 4, 7, 8, 10], 
    "m7 add 9": [0, 2, 3, 7, 10],
    "m": [0, 3, 7], 
    "m6": [0, 3, 7, 9],
    "ø7": [0, 3, 6, 10],
    "o7": [0, 3, 6, 9],
    "m7 alter b5": [0, 3, 6, 10],
    "o": [0, 3, 6],
    "dim": [0, 3, 6],
    "+": [0, 4, 8],
    "7+": [0, 4, 8, 10],
    "7#9": [0, 3, 4, 7, 10],
    "7#11": [0, 4, 6, 7, 10],
    "7 add b9": [0, 1, 4, 7, 10],
    "m7 add b9": [0, 1, 3, 7, 10],
    "maj7#11": [0, 4, 6, 7, 11],
    "power": [0, 7],
    "mM7": [0, 3, 7, 11]
}
SCALES_FOR_CHORDS = {
    "": [0, 2, 4, 5, 7, 9, 11], # major, includes the iv
    "6": [0, 2, 4, 5, 7, 9, 10], # mixolydian
    "6 add 9": [0, 2, 4, 5, 7, 9, 10],
    "sus": [0, 2, 4, 5, 7, 9, 10],
    "sus add 7": [0, 2, 4, 5, 7, 9, 10],
    "sus add 7 add 9": [0, 2, 4, 5, 7, 9, 10],
    "maj7": [0, 2, 4, 7, 9, 11], 
    "maj7 add 9": [0, 2, 4, 7, 9, 11],
    "maj7 add #11": [0, 2, 4, 6, 7, 9, 11],
    "7": [0, 2, 4, 7, 9, 10],
    "9": [0, 2, 4, 7, 9, 10],
    "11": [0, 2, 4, 5, 7, 9, 10],
    "13": [0, 2, 4, 5, 7, 9, 10],
    "M13": [0, 2, 4, 6, 7, 9, 11],
    "m7": [0, 2, 3, 5, 7, 10],
    "m9": [0, 2, 3, 5, 7, 10],
    "m11": [0, 2, 3, 5, 7, 10],
    "m13": [0, 2, 3, 5, 7, 9, 10],
    "7 alter b5": [0, 2, 4, 5, 6, 9, 10],
    "7 alter #5": [0, 2, 4, 6, 8, 10],
    "7 add 9": [0, 2, 4, 7, 9, 10], 
    "7 add #9": [0, 1, 3, 4, 7, 9, 10], 
    "7 add b13": [0, 1, 3, 4, 6, 7, 8, 10], 
    "m7 add 9": [0, 2, 3, 5, 7, 10],
    "m": [0, 2, 3, 5, 7, 8, 10], 
    "m6": [0, 2, 3, 5, 7, 9, 11],
    "ø7": [0, 3, 5, 6, 8, 10],
    "o7": [0, 2, 3, 5, 6, 8, 9, 11],
    "m7 alter b5": [0, 3, 5, 6, 8, 10],
    "o": [0, 2, 3, 5, 6, 8, 9, 11],
    "dim": [0, 2, 3, 5, 6, 8, 9, 11],
    "+": [0, 2, 4, 6, 8, 9, 11],
    "7+": [0, 2, 4, 6, 8, 10],
    "7#9": [0, 1, 3, 4, 6, 7, 8, 10],
    "7#11": [0, 2, 4, 6, 7, 9, 10],
    "7 add b9": [0, 1, 3, 4, 6, 7, 9, 10],
    "m7 add b9": [0, 1, 3, 5, 7, 9, 10],
    "maj7#11": [0, 2, 4, 6, 7, 9, 11],
    "power": [0, 2, 4, 6, 7, 9, 10],
    "mM7": [0, 2, 3, 5, 7, 9, 11]
}



#SOME NOTES ARE BOTH IN CHORDS AND ONE AWAY FROM A CHORD TONE

def makeMarkovChain(chord, scale):
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
                row[choice] = 0.05 # 5% chance of picking itself
            else:
                row[choice] = bucketrates[i]
        mc[prev] = row
        
    # [sum(mc[i]) for i in range(len(row))]

    # # print(mc)
    # def prettyPrintMarkov(mc):
    #     print("Markov Chain Matrix:")
    #     print("   ", end="")
    #     for col in scale:
    #         print(f"{col:5}", end="")
    #     print()
    #     for prev, row in mc.items():
    #         print(f"{prev:3}", end="")
    #         for col in scale:
    #             print(f"{row[col]:5.2f}", end="")
    #         print()
    
    # # prettyPrintMarkov(mc)
    return mc

MARKOVS = {k: makeMarkovChain(NOTES_FOR_CHORDS[k], SCALES_FOR_CHORDS[k]) for k in SCALES_FOR_CHORDS.keys()}

def chordSymbolToQuality(s): # this removes the root and returns the quality of the chord
    match = re.search(r'[^A-Z]', s)
    if match:
        q = s[match.start():]
        if q[0] == '-' or q[0] == '#':
            q = q[1:]
        if '/' in q:
            q = q.split('/')[0]
        return q
    return ''

def main():

    # 2 options: import a file or check a file
    argparser = argparse.ArgumentParser(description="Choose leadsheet file to run")
    # argparse.add_argument("select", type=str, help="Select a file")
    # argparse.add_argument("input", type=str, help="Import a file")
    argparser.add_argument("file", type=str, help="The file to be ran")
    argparser.add_argument("--swing", action='store_true', help="Whether to swing the file or not")
    argparser.add_argument('-m', dest='mode', action='store_const', const='-m', help='Midi output')
    argparser.add_argument('-s', dest='mode', action='store_const', const='-s', help='Sheet music output. Default.')

    # parser.add_argument("--mode", help="run or load", default="load")


    args = argparser.parse_args()
    midi = args.mode == "-m"
    FILE = args.file

    # to access probabilities, markovs["chordquality"]["prev"]["next"]
    # for k in SCALES_FOR_CHORDS:
    #     makeMarkovChain(NOTES_FOR_CHORDS[k], SCALES_FOR_CHORDS[k])


    #directory_path = "../leads"
    #directory = [os.path.join(directory_path, file) for file in os.listdir(directory_path) if file.endswith((".mxl", ".musicxml"))]

    # dont include hero town
    # dont include OntheSunnySideOfTheStreet_I
    # dont inlcude beautifullove.mxl, it has multiple endings 
    # dont include OntheSunnySideOfTheStreet_II.musicxml, it has weird repeats
    # dont include StickyJuly.mxl, it has a repeat without a start repeat barline --- i thought we had a case for this...
    # dont include SummerSamba, it has multiple endings
    # dont inlcude theZone, there is a repeat without a start repeat barline
    # directory = [file for file in directory if "beautiful" not in file.lower() if "ii" not in file.lower() if "july" not in file.lower() if "samba" not in file.lower() if "zone" not in file.lower()]
    # directory = ["../leads/Caravan.musicxml"] # for testing
    # for f in directory:
    #     print("Song: ", f)
    chords, melody, length, t = get_chords(FILE)
    if len(chords) == 0:
        print("Error: No chords found in the file.")
        return
    #     chords, melody, length = get_chords(f)
    #     pieceLength = length * 5 # HEAD, BASS SOLO, PIANO SOLO, HORN SOLO, HEAD

    #     #TODO test with more leads to cover more chord types



    #     for c in chords:
    #         if chordSymbolToQuality(c["el"].figure) not in SCALES_FOR_CHORDS:
    #             print(chordSymbolToQuality(c["el"].figure))
            
    theScore = stream.Score()

    # build piano accompaniment
    # select rhythms with monte carlo, short notes are uncommon, aim for two attacks per chord
    # rests are somewhat common
    # then within each rhythm, select a voicing of that chord. 3rd and 7th are common, root and fifth are uncommon
    # if chord is 9th, 11th, etc, those are also common because they're cool
    # each chord tone has its own probability, and is either on or off for each voicing

    # walking bass
    # 1. the first beat in a chord is a chord tone (usually root or fifth)
    # 2. then we pick a note from either the set of non-chord tones (still in the chord scale) (more likely) or a chord tone (less likely)
    # 3. then we pick a chord tone (more likely) or a non-chord tone (less likely)
    # and continue to alternate between the two.
    # unless the chord is an odd # of beats long, in which case the last beat will still be more likely to be a non-chord tone
    # when a spicy note is selected, it resolves to a chord tone (maybe not? maybe we don't pick spicy notes ever in which case we just sometimes do the rhythm thing)

    #TODO Maybe add input parameter for swung or not
    
    ## main.py --mode select --file file.mxl
    ## main.py --mode input --file newfile.mxl

    # if MODE == "select" and FILE:
    #     print(f"Processing selected file: {FILE}")
        
    # elif MODE == "input" and FILE:
    #     print(f"Processing input file: {FILE}")
    
    # else:
    #     print('Error: Invalid mode. Mode must be "select" or "input".')

    #theScore.append(getChordPart(chords, length, args.swing))
    t = tempo.MetronomeMark(number=t.getQuarterBPM(), referent="quarter") if t else None
    hornPart = getHornPart(chords, melody, length, t, args.swing, midi)
    bassPart = getBassPart(chords, melody, length, t, args.swing)
    while hornPart.highestOffset < bassPart.highestOffset:
        hornPart.append(note.Rest())
    # hornPart.show('text', addEndTimes=True, addStartTimes=True, addOffsets=True, addDurations=True, addClefs=True, addInstruments=True)
    hornPart.makeMeasures(inPlace=True)
    theScore.append(hornPart)
    theScore.append(bassPart)
    #theScore.append(getPianoPart(chords, melody, length, t, args.swing))
    #theScore.makeMeasures(inPlace=True)
    # theScore.quantize(quarterLengthDivisors=(3,), processOffsets=True, processDurations=True, recurse=True, inPlace=True)
    if midi:
        theScore.show('midi')
    else:
        theScore.show()
    # theScore.show('text', addEndTimes=True, addStartTimes=True, addOffsets=True, addDurations=True, addClefs=True, addInstruments=True)
    # theScore.parts[0].measures(0, 5).show()


def swingify(instream, instrument):
    t = instream.getElementsByClass(tempo.MetronomeMark)
    instream.makeMeasures(inPlace=True)
    instream.makeRests(fillGaps=True, inPlace=True)
    # instream.show('text')
    newPart = stream.Part()
    newPart.append(instrument)
    if t:
        newPart.append(t[0])
    s = instream.flatten()
    notesAndRests = list(s.notesAndRests)
    for i, elem in enumerate(notesAndRests):
        if elem.offset % 1 == 0.5 and elem.duration.quarterLength >= 0.5 and not isinstance(elem, note.Rest):
            # print(elem, prevElem)
            offset = newPart.highestOffset
            prevElem = newPart.pop()
            prevElem.offset = offset
            # print(f"Popping element {prevElem}")
            # print("Current Build:")
            # newPart.show('text', addEndTimes=True, addStartTimes=True, addOffsets=True, addDurations=True, addClefs=True, addInstruments=True)
            prevElem.quarterLength += Fraction(1, 6)
            if prevElem.duration.quarterLength > Fraction(2, 3):
                if isinstance(prevElem, note.Rest):
                    newElem = note.Rest(quarterLength=prevElem.duration.quarterLength - Fraction(2, 3))
                elif isinstance(prevElem, note.Note):
                    newElem = note.Note(prevElem.pitch, quarterLength=prevElem.duration.quarterLength - Fraction(2, 3))
                elif isinstance(prevElem, chord.Chord):
                    newElem = chord.Chord(prevElem.pitches, quarterLength=prevElem.duration.quarterLength - Fraction(2, 3))
                newElem.offset = prevElem.offset
                if not isinstance(prevElem, note.Rest):
                    newElem.tie = tie.Tie('start')
                newPart.insert(newElem.offset, newElem)
                # print(f"Inserting element {newElem} at offset {newElem.offset}")
                # if(newElem.offset <= 0):
                #     print("1")
                # print("Current Build:")
                # newPart.show('text', addEndTimes=True, addStartTimes=True, addOffsets=True, addDurations=True, addClefs=True, addInstruments=True)
                if isinstance(prevElem, note.Rest):
                    prevElem = note.Rest(quarterLength=Fraction(2, 3))
                elif isinstance(prevElem, note.Note):
                    prevElem = note.Note(prevElem.pitch, quarterLength=Fraction(2, 3))
                elif isinstance(prevElem, chord.Chord):
                    prevElem = chord.Chord(prevElem.pitches, quarterLength=Fraction(2, 3))
                prevElem.offset = elem.offset - 0.5
                if not isinstance(prevElem, note.Rest):
                    prevElem.tie = tie.Tie('stop')
                newPart.insert(prevElem.offset, prevElem)
                # print(f"Inserting element {prevElem} at offset {prevElem.offset}")
                # if(prevElem.offset <= 0):
                #     print("2")
                # print("Current Build:")
                # newPart.show('text', addEndTimes=True, addStartTimes=True, addOffsets=True, addDurations=True, addClefs=True, addInstruments=True)
            else:
                prevElem.duration.appendTuplet(duration.Tuplet(3, 2))
                prevElem.offset = elem.offset - 0.5
                newPart.insert(prevElem.offset, prevElem)
                # print(f"Inserting element {prevElem} at offset {prevElem.offset}")
                # if(prevElem.offset <= 0):
                #     print("3")
                # print("Current Build:")
                # newPart.show('text', addEndTimes=True, addStartTimes=True, addOffsets=True, addDurations=True, addClefs=True, addInstruments=True)
            if isinstance(elem, note.Rest):
                newElem = note.Rest(quarterLength=elem.duration.quarterLength - Fraction(1, 6))
            elif isinstance(elem, note.Note):
                newElem = note.Note(elem.pitch, quarterLength=elem.duration.quarterLength - Fraction(1, 6))
            elif isinstance(elem, chord.Chord):
                newElem = chord.Chord(elem.pitches, quarterLength=elem.duration.quarterLength - Fraction(1, 6))
            if elem.tie:
                newElem.tie = elem.tie
            newElem.duration.appendTuplet(duration.Tuplet(3, 2))
            elem.offset = elem.offset + Fraction(1, 6)
            newPart.insert(elem.offset, newElem)
            # print(f"Inserting element {elem} at offset {elem.offset}")
            # if(elem.offset <= 0):
            #     print("4")
            # print("Current Build:")
            # newPart.show('text', addEndTimes=True, addStartTimes=True, addOffsets=True, addDurations=True, addClefs=True, addInstruments=True)
        else:
            newPart.insert(elem.offset, elem)
            # print(f"Inserting element {elem} at offset {elem.offset}")
            # if(elem.offset == 0):
            #     print("5")
            # print("Current Build:")
            # newPart.show('text', addEndTimes=True, addStartTimes=True, addOffsets=True, addDurations=True, addClefs=True, addInstruments=True)
        prevElem = elem

    # print("Current Build:")
    # newPart.show('text', addEndTimes=True, addStartTimes=True, addOffsets=True, addDurations=True, addClefs=True, addInstruments=True)
    return newPart

def getHornPart(chords, melody, length, t, swung, midi):
    hornPart = stream.Part()
    hornPart.append(instrument.AltoSaxophone())
    hornPart.append(clef.TrebleClef())

    if t is not None:
        hornPart.append(t)

    for n in melody:
        hornPart.insert(n["offset"], n["el"])

    #Two solos
    for _ in range(int(length * 2)):
        hornPart.append(note.Rest())

    if (length * 2) % 1 != 0:
        hornPart.append(note.Rest(quarterlength=((length * 2) % 1))) # add remainder 

    #Horn solo
    hornPart.append(transposeSolo(getSolo(chords, melody, length, "horn"), "horn"))

    for n in melody:
        hornPart.insert(n["offset"] + length * 4, copy.deepcopy(n["el"]))

    for n in hornPart.notesAndRests:
        if (n.offset % 4) + n.quarterLength > 4 and not isinstance(n, note.Rest):
            prevLength = n.quarterLength
            n.quarterLength = 4 - (n.offset % 4)
            n.tie = tie.Tie('start')
            if len(n.pitches) > 1:
                hornPart.insert(n.offset + n.quarterLength, chord.Chord(n.pitches, quarterLength=prevLength - n.quarterLength))
            else:
                hornPart.insert(n.offset + n.quarterLength, note.Note(n.pitch, quarterLength=prevLength - n.quarterLength))

    # hornPart.show('text')

    if not midi:
        hornPart.transpose("-m3", inPlace=True) 

    newPart = hornPart if not swung else swingify(hornPart, instrument.AltoSaxophone())
    # hornPart.show('text')
    # newPart.show('text', addEndTimes=True, addStartTimes=True, addOffsets=True, addDurations=True, addClefs=True, addInstruments=True)
    return newPart


def getBassPart(chords, melody, length, t, swung):
    bassPart = stream.Part()
    bassPart.append(instrument.AcousticBass())
    bassPart.append(clef.BassClef())

    if t is not None:
        bassPart.append(t)


    #TODO Need to change if we decide that bass pattern should be 
    # different for each time through the form
    bassPattern = getBassPattern(chords, length)

    # Head
    bassPart.append(bassPattern)

    # Bass solo
    bassPart.append(transposeSolo(getSolo(chords, melody, length, "bass"), "bass"))

    #TODO Need to change if we decide that bass pattern should be
    # different for each time through the form

    # Piano Solo, Horn solo, Head
    for _ in range(3):
        for n in bassPattern:
            bassPart.append(copy.deepcopy(n))

    while bassPart.highestOffset < length * 5 and bassPart.highestOffset % 4 != 0:
        bassPart.append(note.Rest())

    newPart = bassPart if not swung else swingify(bassPart, instrument.AcousticBass())
    newPart.makeMeasures(inPlace=True)

    return newPart


# weighted_choices = {
    
# }


def getBassPattern(chords, length):
    bassPattern = []
    chordIndex = -1
    curChain = {}
    chordNotes = []
    for offset in range(int(length)):
        if chordIndex == -1 or (chordIndex != len(chords) - 1 and chords[chordIndex + 1]["offset"] <= offset): # choose a new chord now
            chordIndex += 1
            quality = chordSymbolToQuality(chords[chordIndex]["el"].figure)
            curRoot = chords[chordIndex]['el'].root().midi
            curChain = MARKOVS[quality]
            chordNotes = NOTES_FOR_CHORDS[quality]
            # pick a note from root, 3rd if present, or 5th if present
            choices = [i for i in list(set([0, 3, 4, 7]).intersection(chordNotes))]
            n = note.Note(random.choice(choices) + curRoot - 12)
        else:  
            row = curChain[(bassPattern[-1].pitch.midi - curRoot) % 12]
            # print(row.keys())
            # print(type(row.keys()))
            n = note.Note(random.choices(list(row.keys()), weights=list(row.values()), k=1)[0] + curRoot - 12)
                

        shuffledChordNotes = random.sample(chordNotes, len(chordNotes))
        # print("Root:", note.Note(curRoot - 12).pitch)
        # print("Chord quality:", quality)
        # print("Chord notes:", [note.Note(c + curRoot - 12).pitch.name for c in shuffledChordNotes])
        # print("Chord notes:", [c for c in shuffledChordNotes])
        # print("Chosen note:", n.pitch)
        # print("Chosen note midi:", (n.pitch.midi - curRoot) % 12)
        
        resolveNote = None
        r = random.random()
        if r < 0.4 and (n.pitch.midi - curRoot) % 12 not in chordNotes: 
            # print("Spicy note found, resolving to a chord tone")
            for i in shuffledChordNotes:
                # print("Testing", i, "against", (n.pitch.midi - curRoot) % 12, ". Result: ", abs(((n.pitch.midi - curRoot) % 12) - i))
                if abs(((n.pitch.midi - curRoot) % 12) - i) == 1:
                    resolveNote = note.Note(i + curRoot - 12)
                    # print("Spicy note found one half step away, resolving to:", resolveNote.pitch)
                    break
                    
            if resolveNote is None:
                for i in shuffledChordNotes:
                    # print("Testing", i, "against", (n.pitch.midi - curRoot) % 12, ". Result: ", abs(((n.pitch.midi - curRoot) % 12) - i))
                    if abs(((n.pitch.midi - curRoot) % 12) - i) == 2:
                        resolveNote = note.Note(i + curRoot - 12)
                        # print("Spicy note found one whole step away, resolving to:", resolveNote.pitch)
                        break

        if resolveNote is not None:
            n.duration.quarterLength = 0.5
            bassPattern[-1].quarterLength = 0.5
                
        # print("Adding note:", n.pitch)
        bassPattern.append(n)

        if resolveNote is not None:
            # print("Adding note:", resolveNote.pitch)
            bassPattern.append(resolveNote)

    # for b in bassPattern:
    #     print(b, b.quarterLength)
        # random choice btwn each note
        

        # choices = the_weights[curr_chord][0]

        # choice_weights = the_weights[curr_chord][1]        
        # newNote = random.choices(notes, weights=choice_weights, k=1)
        # bassPattern.append(copy.deepcopy(newNote))

    return bassPattern

def getPianoPart(chords, melody, length, t, swung):
    pianoPart = stream.Part()
    pianoPart.append(instrument.Piano())

    pianoPattern = getPianoPattern(chords, length)

    # Head
    pianoPart.append(pianoPattern)

    # Bass solo
    for n in pianoPattern:
        pianoPart.append(copy.deepcopy(n))

    # Piano solo
    pianoPart.append(transposeSolo(getSolo(chords, melody, length, "piano"), "piano"))

    # Horn solo, Head
    for _ in range(2):
        for n in pianoPattern:
            pianoPart.append(copy.deepcopy(n))

    return pianoPart



## from comp2.py
def getRhythms(length): # 
    compositionQuarterLength = length
    currentLength = 0
    rhythms = []
    while currentLength != compositionQuarterLength:
        curr = random.choices(LENGTH_OPTIONS, weights=LENGTH_WEIGHTS, k=1)[0]
        # curr = random.choice(LENGTH_OPTIONS)
        if curr <= compositionQuarterLength - currentLength and \
            (len(rhythms) < 2 or rhythms[-1] != curr or rhythms[-1] != rhythms[-2] or currentLength == compositionQuarterLength - 0.5):
                rhythms.append(curr)
                currentLength += curr
    return rhythms

# LENGTH_OPTIONS = [0.5, 1, 1.5, 2, 2.5, 3, 4] # TODO add more options for length of notes
LENGTH_OPTIONS = [0.5, 1, 1.5, 2, 2.5] # TODO add more options for length of notes
LENGTH_WEIGHTS = [0.3, 0.25, 0.2, 0.15, 0.1] # TODO add more options for length of notes
MONTE_REST_PROBABILITY = 0.8 # probability of adding a rest, higher for accomp

## from comp2.py
def mapRhythms(notes, rhythms): # notes will actually be note objects
    comp = stream.Part()
    # comp.append(key.KeySignature(KEY_SIGNAURE))
    currentOffset = 0
    addRest = False
    for i, (n, rhythm) in enumerate(zip(notes, rhythms)):
        addRest = random.random() <= MONTE_REST_PROBABILITY and len(rhythms) > 1 and i < len(notes) - 1 and rhythm < 2 and not addRest
        if currentOffset + rhythm > 4:

            # curr1 = roman.RomanNumeral(n.figure, KEY) if not addRest else note.Rest()
            # curr2 = roman.RomanNumeral(n.figure, KEY) if not addRest else note.Rest()
            curr1 = copy.deepcopy(n) if not addRest else note.Rest()  
            curr2 = copy.deepcopy(n) if not addRest else note.Rest()
            curr1.duration.quarterLength = 4 - currentOffset
            curr2.duration.quarterLength = rhythm - 4 + currentOffset
            if not addRest:
                curr1.tie = tie.Tie("start")
                curr2.tie = tie.Tie("stop")
            comp.append([curr1, curr2])
        else:
            if addRest: n = note.Rest()  
            n.duration.quarterLength = rhythm
            comp.append(n)
        currentOffset += rhythm
        currentOffset %= 4
    return comp

def getPianoPattern(chords, length):
    
    chordIndex = -1
    pianoNotes = []
    pianoPattern = []
    # pianoPart = stream.Part()
    # pianoPart.append(clef.BassClef())
    # pianoPart.append(instrument.Piano())


    ## from basspattern. 
    # We need to change this to not choose individual notes, from a markov chain
    # Instead, we will generate some rhythms and choose them usijng monte carlo :)
    for offset in range(int(length)):
        if chordIndex == -1 or (chordIndex != len(chords) - 1 and chords[chordIndex + 1]["offset"] <= offset): # choose a new chord now
            
            # old chord
            # this should be defined from the prev iteration (else block)
            # row = curChain[(pianoNotes[-1].pitch.midi - curRoot) % 12]
            # chord_choices = [key + curRoot - 12 for key in row.keys()]
          
            chord_length = chords[chordIndex + 1 ["offset"]] - chords[chordIndex ["offset"]]
            print(f"getting rhythm for chord: {chords[chordIndex]}")
            print(f"measures {chords[chordIndex["offset"]]} to {chords[chordIndex + 1["offset"]]}")
            getRhythms(chord_length, chord_choices) # these will be choices generated from 
            

            # next chord
            chordIndex += 1
            quality = chordSymbolToQuality(chords[chordIndex]["el"].figure)
            curRoot = chords[chordIndex]['el'].root().midi
            curChain = MARKOVS[quality]
            # pick a note from root, 3rd if present, or 5th if present
            choices = [i for i in list(set([0, 3, 4, 7]).intersection(NOTES_FOR_CHORDS[quality]))]
            pianoNotes.append(note.Note(random.choice(choices) + curRoot - 12))
        else:  
            row = curChain[(pianoNotes[-1].pitch.midi - curRoot) % 12]
            chord_choices = [key + curRoot - 12 for key in row.keys()]
            pianoNotes.append(note.Note(random.choices(list(row.keys()), weights=list(row.values()), k=1)[0] + curRoot - 12))

            # pianoNotes.append(note.Note(random.choices(list(row.keys()), weights=list(row.values()), k=1)[0] + curRoot - 12))
    
    # print(pianoNotes)

    
    # rhythms = getRhythms(length)
    # mapping = mapRhythms(pianoNotes, rhythms)


    # for el in mapping:
    #     pianoPattern.append(copy.deepcopy(el))

    ##TODO ensure these notes actually match with the chords using this mapping strategy

    for el in pianoPattern:
        if isinstance(el, note.Note):
            el.octave += 3  # Move notes up two octaves for treble range
        elif isinstance(el, chord.Chord):
            el.transpose(+36, inPlace=True)  # Transpose chords up two octaves
    return pianoPattern


if __name__ == "__main__":
    main()

