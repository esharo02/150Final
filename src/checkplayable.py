from music21 import *
import os

# get the chords, their offsets, and the entire length out of given mxl file
#  {0.0} <music21.stream.Measure 0 offset=0.0>
#         {0.0} <music21.layout.SystemLayout>
#         {0.0} <music21.clef.TrebleClef>
#         {0.0} <music21.key.KeySignature of 3 flats>
#         {0.0} <music21.meter.TimeSignature 4/4>
#         {0.0} <music21.note.Note B->
#         {1.0} <music21.bar.Barline type=double>


def get_chords(mxl_file):
    c = converter.parse(mxl_file)
    # c.show('text')
     
    pickupLength = repeat.RepeatFinder(c).getQuarterLengthOfPickupMeasure()
    s = stream.Score()
    s.append(c)
    s.show('text')
    news = s.expandRepeats()
    news.show('text')
        # stre.append(part)

    # stre.show('text')
    
    chords, melody = [], []
    for element in c.flatten().notes:
        if isinstance(element, harmony.ChordSymbol):
            chords.append({"el": element, "offset": element.offset})
        if isinstance(element, note.Note):
            melody.append({"el": element, "offset": element.offset})
            
    #only take lead sheets, dw abt this       
    # if len(chords) == 0:
    #     print("empty chords:")
    #     c = c.chordify() # this is unfair
    #     for element in c.flatten().notes:
    #         if isinstance(element, chord.Chord) and len(element.notes) > 1:
    #             chords.append({"el": element, "offset": element.offset})
    #         elif isinstance(element, chord.Chord):
    #             melody.append({"el": element, "offset": element.offset})

    # print('chordified')
    # c.show('text')
    ## get chords and their offsets
    
    # totallength: len(chords) - puckuplen
    # print(chords[-1]["el"].quarterLength)
    totalLength = chords[-1]["offset"] + chords[-1]["el"].quarterLength - pickupLength
    return chords, melody, totalLength




# chords, melody, len = get_chords('../leads/All_Of_Me__Key_of_C.mxl')
# chords, melody, len = get_chords('../leads/autumn.mxl')
chords, melody, len = get_chords('../leads/Caravan.musicxml')
# print(chords)
# print(melody)
# print(len)

#chords = get_chords('../leads/There_Will_Never_Be_Another_You.mxl')
# print(chords)



