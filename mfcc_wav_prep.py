import pandas as pd
import os, wave
import librosa
import numpy as np


class MFCCWavLoader:
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
               SAMPLE_RATE = 44100

               b, _ = librosa.core.load(self.wav_dir + '/' + wav, sr=SAMPLE_RATE)
               y, sr = librosa.load(self.wav_dir + '/' + wav)
               entry['Mean_RMS'] = np.mean(librosa.feature.rms(y=y))
               entry['STD_RMS'] = np.std(librosa.feature.rms(y=y))
               assert _ == SAMPLE_RATE
               mfcc_feature = librosa.feature.mfcc(b, sr=SAMPLE_RATE, n_mfcc=20)
               delta_mfcc = librosa.feature.delta(mfcc_feature)
               d_delta_mfcc = librosa.feature.delta(mfcc_feature, order=2)
               mean_mfcc = np.mean(mfcc_feature, axis=1)
               std_mfcc = np.mean(mfcc_feature, axis=1)
               mean_ddmfcc = np.mean(d_delta_mfcc, axis=1)
               std_ddmfcc = np.std(d_delta_mfcc,axis=1)
               mean_dmfcc = np.mean(delta_mfcc, axis=1)
               std_dmfcc = np.std(delta_mfcc, axis=1)
               for no in range(0, len(np.mean(delta_mfcc, axis=1))):
                   entry['Mean_MFCC{0}'.format(no)] = mean_mfcc[no]
                   entry['STD_MFCC{0}'.format(no)] = std_mfcc[no]
                   entry['Mean_DDMFCC{0}'.format(no)] = mean_ddmfcc[no]
                   entry['STD_DDMFCC{0}'.format(no)] = std_ddmfcc[no]
                   entry['Mean_Delta_MFCC{0}'.format(no)] = mean_dmfcc[no]
                   entry['STD_Delta_MFCC{0}'.format(no)] = std_dmfcc[no]
               y, sr = librosa.load(self.wav_dir + '/' + wav)
               pitches, magnitudes = librosa.core.piptrack(y, sr)
               # Select out pitches with high energy
               pitches = pitches[magnitudes > np.median(magnitudes)]
               pit = librosa.pitch_tuning(pitches)
               entry['pitch'] = pit

               wav_files.append(entry)

        wav_df = pd.DataFrame(wav_files)
        return wav_df

