from music21 import *
import copy

def get_chords(mxl_file):
    """
    Extracts chords and melody from a MusicXML file, handling repeats.
    """

    c = converter.parse(mxl_file)
    s = stream.Score()
    s.append(c)

    tempo = c.metronomeMarkBoundaries()[0][2] if c.metronomeMarkBoundaries() else None
    ts = s[meter.TimeSignature].first()
    isFourFour = ts.numerator == 4 and ts.denominator == 4

    #Repeat barlines
    repeatBars = []
    for element in s.flatten():
        if isinstance(element, bar.Repeat):
            repeatBars.append({"el": element, "offset": element.offset})

    #Repeat brackets (ones with 1st and 2nd endings)
    rbs = []
    for el in s.flatten():
        if isinstance(el, spanner.RepeatBracket):
            rbs.append(el)

    #Handle the case where there's no begin repeat
    if len(repeatBars) > 0 and repeatBars[0]['el'].direction == "end": 
        repeatBars.insert(0, {"el": bar.Repeat(direction="start"), "offset": 0})
    #Handle the case where there's no end repeat
    if len(repeatBars) == 2 and repeatBars[0]['el'].measureNumber == 1 and repeatBars[1]['offset'] == -1:
        repeatBars = []
    #Handle duplicate repeat bars (caused by multiple parts)
    unique = []
    seen = set()
    for repeat in repeatBars:
        obj = (repeat["el"].measureNumber, repeat["el"].direction)
        if obj not in seen:
            seen.add(obj)
            unique.append(repeat)
    repeatBars = unique

    #Group repeat brackets by their measure numbers
    rbsGroupedByRepeatMeasureNumber = {}
    for rb in rbs:
        for i in range(1, len(repeatBars), 2):
            measureNumber = repeatBars[i]['el'].measureNumber
            if rb.getSpannedElements()[-1].measureNumber >= measureNumber and rb.getSpannedElements()[0].measureNumber <= measureNumber + 1:
                break
        if measureNumber not in rbsGroupedByRepeatMeasureNumber:
            rbsGroupedByRepeatMeasureNumber[measureNumber] = []
        rbsGroupedByRepeatMeasureNumber[measureNumber].append(rb)

    chords = []
    melody = []
    for element in c.flatten():
        if isinstance(element, harmony.ChordSymbol):
            chords.append({"el": element, "offset": element.offset})
        elif isinstance(element, note.Note) or isinstance(element, note.Rest) or isinstance(element, chord.Chord):
            element.lyric = None
            melody.append({"el": element, "offset": element.offset})
    
    #Double repeated sections
    totalLength = 0
    curStart = 0
    prevWasEndingRepeat = False
    for i in range(0, len(repeatBars), 2):
        repeatChords, repeatMelody = [], []
        startFrom = repeatBars[i]['el'].measureNumber if repeatBars[i]['el'].measureNumber else 1
        if repeatBars[i + 1]['el'].measureNumber in rbsGroupedByRepeatMeasureNumber:
            repeatSection = copy.deepcopy(c.measures(startFrom, rbsGroupedByRepeatMeasureNumber[repeatBars[i + 1]['el'].measureNumber][0].getSpannedElements()[0].measureNumber - 1))
            if prevWasEndingRepeat:
                curStart += repeatBars[i + 1]['offset'] - repeatBars[i - 1]['offset']
            else:
                curStart += repeatBars[i + 1]['offset']
            prevWasEndingRepeat = True
        else:
            repeatSection = copy.deepcopy(c.measures(startFrom, repeatBars[i + 1]['el'].measureNumber))
            if prevWasEndingRepeat:
                curStart += repeatBars[i]['offset'] - repeatBars[i - 2]['offset']
            else:
                curStart += repeatBars[i]['offset']
            prevWasEndingRepeat = False

        for i, element in enumerate(repeatSection.flatten()):
            if isinstance(element, harmony.ChordSymbol):
                repeatChords.append({"el": element, "offset": element.offset + curStart})
            elif isinstance(element, note.Note) or isinstance(element, note.Rest) or isinstance(element, chord.Chord):
                repeatMelody.append({"el": element, "offset": element.offset + curStart})
                if i == len(repeatSection.flatten()) - 1 and element.tie and element.tie.type == "start":
                    element.addLyric("unfinished")
    
        for ch in chords:
            if ch["offset"] >= curStart:
                ch["offset"] += repeatSection.quarterLength
        for me in melody:
            if me["offset"] >= curStart:
                me["offset"] += repeatSection.quarterLength
    
        curStart += repeatSection.quarterLength
        chords = chords + repeatChords
        melody = melody + repeatMelody
        totalLength += repeatSection.quarterLength
    chords.sort(key=lambda x: x["offset"])
    melody.sort(key=lambda x: x["offset"])

    next = False
    for n in melody:
        if next:
            n['el'].tie = tie.Tie(type="stop")
            break
        if n['el'].lyric == "unfinished":
            next = True
            n['el'].lyric == None

    #Combine notes at the same offset
    temp = {}
    for m in melody:
        if m["offset"] not in temp:
            temp[m["offset"]] = []
        temp[m["offset"]].append(m["el"])
    newMelody = []
    for offset, notes in sorted(temp.items(), key=lambda x: x[0]):
        notes = [n for n in notes if not isinstance(n, note.Rest)]
        if len(notes) == 0:
            continue
        if len(notes) == 1:
            newMelody.append({"el": notes[0], "offset": offset})
        else:
            newMelody.append({"el": chord.Chord(notes), "offset": offset})
    melody = newMelody

    #Round offsets to the nearest whole number if they are at 0.25 or 0.75 (chords change too fast)
    for c in chords:
        if c["offset"] % 1 == 0.25 or c["offset"] % 1 == 0.75:
            c["offset"] = round(c["offset"])

    totalLength += s.quarterLength
    return chords, melody, totalLength, tempo, isFourFour
