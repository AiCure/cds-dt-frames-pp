import sys
sys.path.append('../dtpp')

from dtpp.derived.vocal_activity import PauseCharacteristics

def test_pause_characteristics():
    pc = PauseCharacteristics(0.5, './data/vad.csv', offset=1)
    assert pc.pause_count() == 3

    pc = PauseCharacteristics(0.5, './data/vad.csv', offset=0)
    assert pc.pause_count() == 4