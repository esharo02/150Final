from music21 import *
import argparse
from checkplayable import get_chords
import copy
import re

#https://igm.rit.edu/~jabics/BilesICMC94.pdf
SCALES_FOR_CHORDS = {
    "": [0, 2, 4, 5, 7, 9, 11], # major, includes the iv
    "6": [0, 2, 4, 5, 7, 9, 10], # mixolydian
    "sus": [0, 2, 4, 5, 7, 9, 10],
    "maj7": [0, 2, 4, 7, 9, 11], 
    "maj7 add 9": [0, 2, 4, 7, 9, 11],
    "7": [0, 2, 4, 7, 9, 10],
    "9": [0, 2, 4, 7, 9, 10],
    "11": [0, 2, 4, 7, 9, 10],
    "13": [0, 2, 4, 7, 9, 10],
    "m7": [0, 2, 3, 5, 7, 10],
    "7 add 9": [0, 2, 4, 7, 9, 10],
    "7 alter b5": [0, 2, 4, 5, 6, 9, 10], 
    "m7 add 9": [0, 2, 3, 5, 7, 10],
    "m": [0, 2, 3, 5, 7, 8, 10], 
    "m6": [0, 2, 3, 5, 7, 8, 10],
    "ø7": [0, 3, 5, 6, 8, 10],
    "o7": [0, 2, 3, 5, 6, 8, 9, 11],
    "o": [0, 2, 3, 5, 6, 8, 9, 11],
    "+": [0, 2, 4, 6, 8, 9, 11],
    "7+": [0, 2, 4, 6, 8, 10],
    "7#11": [0, 2, 4, 6, 7, 9, 10],
    "7#9": [0, 1, 3, 4, 6, 8, 10],
    "7 add b9": [0, 1, 3, 4, 6, 7, 9, 10],
    "m7 add b9": [0, 1, 3, 5, 7, 9, 10],
    "maj7#11": [0, 2, 4, 6, 7, 9, 11],
}

def chordSymbolToQuality(s):
    match = re.search(r'[^A-Z]', s)
    if match:
        str = s[match.start():]
        if str[0] == '-' or str[0] == '#':
            str = str[1:]
        if '/' in str:
            str = str.split('/')[0]
        return str
    return ''

def main():

    # 2 options: import a file or check a file
    argparser = argparse.ArgumentParser(description="Choose leadsheet file to run")
    # argparse.add_argument("select", type=str, help="Select a file")
    # argparse.add_argument("input", type=str, help="Import a file")
    argparser.add_argument("file", type=str, help="The file to be ran")
    #argparse.add_argument("mode", type=str, help="The file to be ran")
    

    args = argparser.parse_args()

    FILE = args.file

    chords, melody, length = get_chords(FILE)
    pieceLength = length * 5 # HEAD, BASS SOLO, PIANO SOLO, HORN SOLO, HEAD

    #TODO test with more leads to cover more chord types
    for c in chords:
        if chordSymbolToQuality(c["el"].figure) not in SCALES_FOR_CHORDS:
            print(c["el"].figure)
        
    exit()

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


    theScore.append(getHornPart(chords, melody, length))
    theScore.append(getBassPart(chords, melody, length))
    #theScore.append(getPianoPart(chords, melody, length))
    theScore.makeMeasures(inPlace=True)
    theScore.show()

def getHornPart(chords, melody, length):
    hornPart = stream.Part()
    hornPart.append(instrument.AltoSaxophone())
    hornPart.append(clef.TrebleClef())

    for n in melody:
        hornPart.append(n["el"])

    #Two solos
    for _ in range(int(length * 2)):
        hornPart.append(note.Rest())

    if (length * 2) % 1 != 0:
        hornPart.append(note.Rest(quarterlength=((length * 2) % 1))) # add remainder 

    #Horn solo
    hornPart.append(transposeSolo(getSolo(chords, length), "horn"))

    for n in melody:
        hornPart.append(copy.deepcopy(n["el"]))

    #Add chord symbols
    for i in range(5):
        for c in chords:
            hornPart.insert(c["offset"] + length * i, copy.deepcopy(c["el"]))

    return hornPart

def getBassPart(chords, melody, length):
    bassPart = stream.Part()
    bassPart.append(instrument.Bass())
    bassPart.append(clef.BassClef())


    #TODO Need to change if we decide that bass pattern should be 
    # different for each time through the form
    bassPattern = getBassPattern(chords, length)

    # Head
    bassPart.append(bassPattern)

    # Bass solo
    bassPart.append(transposeSolo(getSolo(chords, length), "bass"))

    #TODO Need to change if we decide that bass pattern should be
    # different for each time through the form

    # Piano Solo, Horn solo, Head
    for _ in range(3):
        for n in bassPattern:
            bassPart.append(copy.deepcopy(n))

    return bassPart

import random

# weighted_choices = {
    
# }


def getBassPattern(chords, length):
    bassPattern = []
    chordIndex = 0
    for offset in range(int(length)):
        if chordIndex != len(chords) - 1 and chords[chordIndex + 1]["offset"] <= offset:
            chordIndex += 1
        notes = chords[chordIndex]["el"].notes

        # random choice btwn each note
        

        # choices = the_weights[curr_chord][0]

        # choice_weights = the_weights[curr_chord][1]        
        # newNote = random.choices(notes, weights=choice_weights, k=1)
        # bassPattern.append(copy.deepcopy(newNote))

        bassPattern.append(copy.deepcopy(random.choice(notes)))

        

    return bassPattern

def getPianoPart(chords, melody, length):
    pianoPart = stream.Part()
    pianoPart.append(instrument.Piano())

    pianoPattern = getPianoPattern()

    # Head
    pianoPart.append(pianoPattern)

    # Bass solo
    for n in pianoPattern:
        pianoPart.append(copy.deepcopy(n))

    # Piano solo
    pianoPart.append(transposeSolo(getSolo(chords, length), "piano"))

    # Horn solo, Head
    for _ in range(2):
        for n in pianoPattern:
            pianoPart.append(copy.deepcopy(n))

    return pianoPart

def getPianoPattern():
    pass

def getSolo(chords, length):
    solo = []
    for _ in range(int(length)):
        solo.append(note.Rest())
    if length % 1 != 0:
        solo.append(note.Rest(quarterlength=length % 1))
    return solo

def transposeSolo(solo, instrument):
    return solo

if __name__ == "__main__":
    main()

