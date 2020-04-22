import numpy as np
from tqdm import tqdm
import midi_manipulation
import os

def get_files(path="generated"):
    generated = []
    for a, b, c in os.walk(path):
        for f in c:
            generated.append(os.path.join(a, f))

    return generated


def menu(output="output"):
    files = get_files()
    songs = np.zeros((0,156))
    for f in tqdm(files):
        try:
            song = np.array(midi_manipulation.midiToNoteStateMatrix(f))
            songs = np.concatenate((songs,song))
        except Exception as e:
            raise e
    print("samples merging ...")
    print(np.shape(songs))
    if(not ".mid" in output):
        output+=".mid"

    midi_manipulation.noteStateMatrixToMidi(songs, output)
