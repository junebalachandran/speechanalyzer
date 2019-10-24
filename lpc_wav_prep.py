import pandas as pd
import os, wave
import scipy.io.wavfile as swav
import librosa
import numpy as np

class LWavLoader:
    def __init__(self, wav_dir):
        self.wav_dir = wav_dir

    def _load(self, audio):
        wavefile = wave.open(audio, 'r')
        return wavefile


    def get_wav_df(self):
        wav_files = []
        for wav in os.listdir(self.wav_dir):
           if wav.endswith('.wav'):
               entry = dict()
               entry['Session'] = wav

               fs, signal = swav.read(self.wav_dir + '/' + wav)
               y, sr = librosa.load(self.wav_dir + '/' + wav)
               lpc = librosa.lpc(y,5)
               for no in range(0, len(lpc)):
                   entry['LIB_LPC{0}'.format(no)] = lpc[no]
               y, sr = librosa.load(self.wav_dir + '/' + wav)
               pitches, magnitudes = librosa.core.piptrack(y, sr)
               # Select out pitches with high energy
               pitches = pitches[magnitudes > np.median(magnitudes)]
               pit = librosa.pitch_tuning(pitches)

               entry['pitch'] = pit

               wav_files.append(entry)

        # wav_files = []
        # entry = dict()
        # iemocap_wav_list = self._load()
        # print(iemocap_wav_list.getframerate())
        # print(iemocap_wav_list)
        # entry['Session'] = glob.glob("*.wav", iemocap_wav_list)
        # if bool(entry):
        #     wav_files.append(entry)
        wav_df = pd.DataFrame(wav_files)
        return wav_df

