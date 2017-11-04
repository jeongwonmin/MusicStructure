import os
import math

import librosa
import librosa.core
import numpy as np


class MusicReader(object):
    EXT = list(sorted(set([".mp3", ".wav"])))
    
    def __init__(self, music_folder):
        self.music_folder = music_folder
        self._music_list = self._make_music_list()
        
    def _make_music_list(self):
        musics = []
        for f in os.listdir(self.music_folder):
            rhythm_folder = os.path.join(self.music_folder, f)
            if not os.path.isdir(rhythm_folder):
                continue
            for musicfile in os.listdir(rhythm_folder):
                if not os.path.splitext(musicfile)[1] in self.EXT:
                    continue
                music_path = os.path.join(rhythm_folder, musicfile)
                musics.append(music_path)
        return musics
                
    # load original cqt matrix, mfcc matrix and k
    def _load_music(self, music_file):
        y, sr = librosa.load(music_file)
        C = librosa.hybrid_cqt(y, sr, fmin=librosa.note_to_hz('C2'), n_bins=72)
        CQT = librosa.amplitude_to_db(C, ref=np.max)
        mfcc = librosa.feature.mfcc(y=y, sr=sr)[:13,:]

        tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
        
        k = 1 + 2 * math.ceil(math.log(len(beats), 2))
        
        C = librosa.util.sync(C, beats) # mean aggregate
        mfcc = librosa.util.sync(mfcc, beats)
        C = librosa.feature.stack_memory(C)
        mfcc = librosa.feature.stack_memory(mfcc)
        C_t = C.transpose()
        mfcc_t = mfcc.transpose()
        return C_t, mfcc_t, k
     
    def __call__(self):
        for msc in self._music_list:
            filename = os.path.basename(msc)
            f_noext = os.path.splitext(filename)[0]
            yield f_noext, self._load_music(msc)
            yield f_noext, self._load_music(msc)