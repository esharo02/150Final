from music21 import *
import os
import copy
import sys
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
    #s.show('text')
        # stre.append(part)
    key = c.analyze('key')
    tempo = c.metronomeMarkBoundaries()[0][2] if c.metronomeMarkBoundaries() else None
    #print(f"Key: {key}, Tempo: {tempo}")
    # stre.show('text')
    
    repeatBars = []
    for element in s.flatten():
        if isinstance(element, bar.Repeat):
            try: 
                repeatBars.append({"el": element, "offset": s.flatten().getElementAfterElement(element).offset})
            except AttributeError:
                repeatBars.append({"el": element, "offset": -1}) 
            #print(f"offset after repeat:  {repeatBars[-1]['offset']}")
            #print(f"Repeat barline at measure: {element.measureNumber}, direction: {element.direction}")
    if len(repeatBars) > 0 and repeatBars[0]['el'].direction == "end": # handle the case where there's no begin repeat
        repeatBars.insert(0, {"el": bar.Repeat(direction="start", measureNumber=1), "offset": 0})

    if len(repeatBars) == 2 and repeatBars[0]['el'].measureNumber == 1 and repeatBars[1]['offset'] == -1:
        repeatBars = []

    chords = []
    melody = []
    for element in c.flatten():
        if isinstance(element, harmony.ChordSymbol):
            chords.append({"el": element, "offset": element.offset})
        elif isinstance(element, note.Note) or isinstance(element, note.Rest) or isinstance(element, chord.Chord):
            element.lyric = None
            melody.append({"el": element, "offset": element.offset})
    
    # print((repeatBars))
    
    totalLength = 0
    curStart = 0
    for i in range(0, len(repeatBars), 2):
        # print(f"In for loop {i} {repeatBars[i]['el'].measureNumber} {repeatBars[i + 1]['el'].measureNumber}")
        repeatChords, repeatMelody = [], []
        # print(c)
        repeatSection = copy.deepcopy(c.measures(repeatBars[i]['el'].measureNumber, repeatBars[i + 1]['el'].measureNumber))
        # repeatSection.show('text')
        curStart += repeatBars[i]['offset']
        # print(repeatBars[i]['offset'])
        # print(f"curStart: {curStart}")

        for element in repeatSection.flatten():
            #print(f"I am here and appending this {element} at {element.offset + curStart}")
            if isinstance(element, harmony.ChordSymbol):
                repeatChords.append({"el": element, "offset": element.offset + curStart})
            elif isinstance(element, note.Note) or isinstance(element, note.Rest) or isinstance(element, chord.Chord):
                repeatMelody.append({"el": element, "offset": element.offset + curStart})
        
        # A [repeat] B
        ## currStart = len(A), then len(B)
     
        for ch in chords:
            if ch["offset"] >= curStart:
                ch["offset"] += repeatSection.quarterLength
        for me in melody:
            if me["offset"] >= curStart:
                me["offset"] += repeatSection.quarterLength
        
        curStart += repeatSection.quarterLength
        # print(f"repeat length: {repeatSection.quarterLength}")
        chords = chords + repeatChords
        melody = melody + repeatMelody
        totalLength += repeatSection.quarterLength
    chords.sort(key=lambda x: x["offset"])
    melody.sort(key=lambda x: x["offset"])


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
    


    

    # print(repeatBars)

    # repeat beginning on 17 ending on 32
    # repeatBars.measurenum = [||: 17, 32 :||] 
    

    # Expand repeats into the stream

    
    
    
#   Key: f minor, Tempo: <music21.tempo.MetronomeMark Quarter=200 (playback only)>
#   Repeat barline at measure: 1, direction: start
#   Repeat barline at measure: 16, direction: end
#   [1, 16]

    totalLength += s.quarterLength - pickupLength
    return chords, melody, totalLength


# chords, melody, len = get_chords('../leads/All_Of_Me__Key_of_C.mxl')
# chords, melody, len = get_chords('../leads/autumn.mxl') # 
chords, melody, length = get_chords('../leads/Caravan.musicxml')
#print(chords)
#print(melody)
#for c in chords:
    #print(f"Element: {c['el']}, Offset: {c['offset']}")
#print(len)

# convert melody array back into music21 stream
melodyStream = stream.Stream()
for n in melody:
    melodyStream.append(n["el"])

#print(melody)
#print(chords)
    
if sys.argv[1] == "--fuckoff":
    melodyStream.show()


## what I have:
## part 1 
## part 2
## my function:
## part1 = part1+part2

#chords = get_chords('../leads/There_Will_Never_Be_Another_You.mxl')
# print(chords)



