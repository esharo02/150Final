from music21 import *
import os

# get the chords, their offsets, and the entire length out of given mxl file

def get_chords(mxl_file):
    c = converter.parse(mxl_file)
    c = c.chordify() # this is unfair
    chords = []
    for element in c.flatten().notes:
        if isinstance(element, chord.Chord) and len(element.notes) > 1:
            chords.append((element, element.offset))

    
    return chords




chords = get_chords('../leads/All_Of_Me__Key_of_C.mxl')
print(chords)

chords = get_chords('../leads/There_Will_Never_Be_Another_You.mxl')
print(chords)

chords = get_chords('../leads/TRB_Autumn_Leaves.mxl')
print(chords)


