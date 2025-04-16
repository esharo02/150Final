from music21 import *
import argparse
from checkplayable import get_chords
import os


def main():

    # 2 options: import a file or check a file
    argparse = argparse.ArgumentParser(description="Choose leadsheet file to run")
    # argparse.add_argument("select", type=str, help="Select a file")
    # argparse.add_argument("input", type=str, help="Import a file")
    argparse.add_argument("file", type=str, help="The file to be ran")
    #argparse.add_argument("mode", type=str, help="The file to be ran")
    

    args = argparse.parse_args()

    FILE = args.file

    chords, melody, length = get_chords(FILE)
    pieceLength = length * 5 # HEAD, BASS SOLO, PIANO SOLO, HORN SOLO, HEAD

    ## assign lead to horn
    theScore = stream.Score()
    hornPart = stream.Part()
    hornPart.append(instrument.AltoSaxophone())

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

    
    ## main.py --mode select --file file.mxl
    ## main.py --mode input --file newfile.mxl

    # if MODE == "select" and FILE:
    #     print(f"Processing selected file: {FILE}")
        
    # elif MODE == "input" and FILE:
    #     print(f"Processing input file: {FILE}")
    
    # else:
    #     print('Error: Invalid mode. Mode must be "select" or "input".')



if __name__ == "__main__":
    main()

