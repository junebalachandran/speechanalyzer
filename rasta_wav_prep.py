import pandas as pd
import os, glob, wave
from sidekit.frontend.features import plp
import librosa
import numpy as np
class RPLPWavLoader:
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


               spf = wave.open(self.wav_dir + '/' + wav, 'r')

               signal = spf.readframes(-1)
               input_sig = np.fromstring(signal, 'Int16')

               matrix = plp(input_sig, nwin=0.025, fs=sr, plp_order=13, shift=0.01, get_spec=False, get_mspec=False,
                            prefac=0.97, rasta=True)

               rasta_f_df = pd.DataFrame(matrix[0])
               mean_rastaplp = np.asarray((np.mean(rasta_f_df, axis=0)).tolist())
               std_rastaplp = np.asarray((np.std(rasta_f_df,axis=0)).tolist())
               delta_rastaplp = librosa.feature.delta(rasta_f_df)
               d_delta_rastaplp = librosa.feature.delta(rasta_f_df, order=2)


               mean_ddrastaplp = np.mean(d_delta_rastaplp, axis=0)
               std_ddrastaplp = np.std(d_delta_rastaplp, axis=0)
               mean_drastaplp = np.mean(delta_rastaplp, axis=0)
               std_drastaplp = np.std(delta_rastaplp, axis=0)



               for no in range(0, 13):
                   entry['Mean_RASTAPLP{0}'.format(no)] = mean_rastaplp[no]
                   entry['STD_RASTAPLP{0}'.format(no)] = std_rastaplp[no]
                   entry['Mean_DDRastaPLP{0}'.format(no)] = mean_ddrastaplp[no]
                   entry['STD_DDRastaPLP{0}'.format(no)] = std_ddrastaplp[no]
                   entry['Mean_Delta_RastaPLP{0}'.format(no)] = mean_drastaplp[no]
                   entry['STD_Delta_RastaPLP{0}'.format(no)] = std_drastaplp[no]
               y, sr = librosa.load(self.wav_dir + '/' + wav)
               pitches, magnitudes = librosa.core.piptrack(y, sr)
               # Select out pitches with high energy
               pitches = pitches[magnitudes > np.median(magnitudes)]
               pit = librosa.pitch_tuning(pitches)
               entry['pitch'] = pit

               wav_files.append(entry)
        wav_df = pd.DataFrame(wav_files)
        return wav_df

