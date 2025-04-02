from music21 import *
import os

# get the chords, their offsets, and the entire length out of given mxl file

def get_chords(mxl_file):
    c = converter.parse(mxl_file)
    c = c.chordify() # this is unfair
    chords = []
    for element in c.flatten().notes:
        if isinstance(element, chord.Chord):
            chords.append((element, element.offset))
    chordsymbols = []
    for element in c.flatten().notes:
        if isinstance(element, harmony.ChordSymbol):
            chords.append((element, element.offset))
    return chords, chordsymbols

chords, chordsymbols = get_chords('All_Of_Me__Key_of_C.mxl')
print(chords)
print(chordsymbols)


# print version of music21
import music21
print(music21.__version__)
# 9.3.0