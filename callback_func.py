# -*- coding: utf-8 -*-
import speech_recognition as sr
import pandas as pd
import wave
import librosa
import numpy as np
from sidekit.frontend.features import plp
from sklearn.externals import joblib
import plotly.graph_objects as go
import base64
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

vect = CountVectorizer()
tfidf_trans = TfidfTransformer()
def text_emo(text):

    text_df = pd.DataFrame([{'Transcript': text}])
    X_new_counts = vect.transform(text_df['Transcript'])
    X_new_tfidf = tfidf_trans.transform(X_new_counts)
    clf = joblib.load('pickles/text_model')
    result = str(clf.predict(X_new_tfidf)).strip("['']")
    bar = pd.DataFrame(clf.predict_proba(X_new_tfidf))
    bar.columns = clf.classes_
    bar_t = bar.T
    bar_t.columns = ['values']
    fig = go.Figure(data=[go.Pie(labels=clf.classes_, values=bar_t['values'],hole=.3),])
    return result, fig

def rasta_emotion_upload():
    wav_files = []
    entry = dict()
    SAMPLE_RATE = 44100
    b, _ = librosa.core.load('pickles/catalyst.wav', sr=SAMPLE_RATE)
    y, sr = librosa.load('pickles/catalyst.wav')
    entry['Mean_RMS'] = np.mean(librosa.feature.rms(y=y))
    entry['STD_RMS'] = np.std(librosa.feature.rms(y=y))
    assert _ == SAMPLE_RATE

    spf = wave.open('pickles/catalyst.wav')

    signal = spf.readframes(-1)
    input_sig = np.fromstring(signal, 'Int16')

    matrix = plp(input_sig, nwin=0.025, fs=sr, plp_order=13, shift=0.01, get_spec=False, get_mspec=False,
                 prefac=0.97, rasta=True)

    rasta_f_df = pd.DataFrame(matrix[0])
    mean_rastaplp = np.asarray((np.mean(rasta_f_df, axis=0)).tolist())
    std_rastaplp = np.asarray((np.std(rasta_f_df, axis=0)).tolist())
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
    y, sr = librosa.load('/pickles/catalyst.wav')
    pitches, magnitudes = librosa.core.piptrack(y, sr)
    # Select out pitches with high energy
    pitches = pitches[magnitudes > np.median(magnitudes)]
    pit = librosa.pitch_tuning(pitches)
    entry['pitch'] = pit

    wav_files.append(entry)
    wav_df = pd.DataFrame(wav_files)
    rasta_clf = joblib.load('pickles/rastaplp_model.sav')

    bar = pd.DataFrame(rasta_clf.predict_proba(wav_df))
    bar.columns = rasta_clf.classes_
    bar_t = bar.T
    bar_t.columns = ['values']
    print('HERE')

    fig = go.Figure(data=[go.Pie(labels=rasta_clf.classes_, values=bar_t['values'], hole=.3), ])
    return rasta_clf.predict(wav_df), fig
def lpc_emotion_upload():
    entry = dict()
    wav_files = []
    SAMPLE_RATE = 44100
    b, _ = librosa.core.load('pickles/catalyst.wav', sr=SAMPLE_RATE)
    y, sr = librosa.load('pickles/catalyst.wav')
    lpc = librosa.lpc(y, 5)
    for no in range(0, len(lpc)):
        entry['LIB_LPC{0}'.format(no)] = lpc[no]
    y, sr = librosa.load('pickles/catalyst.wav')
    pitches, magnitudes = librosa.core.piptrack(y, sr)
    # Select out pitches with high energy
    pitches = pitches[magnitudes > np.median(magnitudes)]
    pit = librosa.pitch_tuning(pitches)

    entry['pitch'] = pit

    wav_files.append(entry)
    wav_df = pd.DataFrame(wav_files)
    lpc_clf = joblib.load('pickles/lpc_model.sav')

    bar = pd.DataFrame(lpc_clf.predict_proba(wav_df))
    bar.columns = lpc_clf.classes_
    bar_t = bar.T
    bar_t.columns = ['values']
    print('HERE')

    fig = go.Figure(data=[go.Pie(labels=lpc_clf.classes_, values=bar_t['values'], hole=.3), ])
    return lpc_clf.predict(wav_df), fig

def decoder(value):
    org = value
    org = org.pop(0)
    orgi = org[22:]
    data = base64.b64decode(orgi)
    real_audio = open('pickles/catalyst.wav', 'wb')
    real_audio.write(data)
    boolean = 1
    return boolean

def recog(audio):
    r = sr.Recognizer()
    text = r.recognize_google(audio)
def figure_maker(input_value):

    text_df = pd.DataFrame([{'Transcript': input_value}])
    X_new_counts = vect.transform(text_df['Transcript'])
    X_new_tfidf = tfidf_trans.transform(X_new_counts)
    clf = joblib.load('pickles/text_model')
    result = str(clf.predict(X_new_tfidf)).strip("['']")
    bar = pd.DataFrame(clf.predict_proba(X_new_tfidf))
    bar.columns = clf.classes_
    bar_t = bar.T
    bar_t.columns = ['values']
    fig = go.Figure(data=[go.Pie(labels=clf.classes_, values=bar_t['values'],hole=.3),])
    return result, fig

def mfcc_emotion_upload():
    entry = dict()
    wav_files = []
    SAMPLE_RATE = 44100

    b, _ = librosa.core.load('pickles/catalyst.wav', sr=SAMPLE_RATE)
    y, sr = librosa.load('pickles/catalyst.wav')
    entry['Mean_RMS'] = np.mean(librosa.feature.rms(y=y))
    entry['STD_RMS'] = np.std(librosa.feature.rms(y=y))
    assert _ == SAMPLE_RATE
    mfcc_feature = librosa.feature.mfcc(b, sr=SAMPLE_RATE, n_mfcc=20)
    delta_mfcc = librosa.feature.delta(mfcc_feature)
    d_delta_mfcc = librosa.feature.delta(mfcc_feature, order=2)
    mean_mfcc = np.mean(mfcc_feature, axis=1)
    std_mfcc = np.mean(mfcc_feature, axis=1)
    mean_ddmfcc = np.mean(d_delta_mfcc, axis=1)
    std_ddmfcc = np.std(d_delta_mfcc, axis=1)
    mean_dmfcc = np.mean(delta_mfcc, axis=1)
    std_dmfcc = np.std(delta_mfcc, axis=1)
    for no in range(0, len(np.mean(delta_mfcc, axis=1))):
        entry['Mean_MFCC{0}'.format(no)] = mean_mfcc[no]
        entry['STD_MFCC{0}'.format(no)] = std_mfcc[no]
        entry['Mean_DDMFCC{0}'.format(no)] = mean_ddmfcc[no]
        entry['STD_DDMFCC{0}'.format(no)] = std_ddmfcc[no]
        entry['Mean_Delta_MFCC{0}'.format(no)] = mean_dmfcc[no]
        entry['STD_Delta_MFCC{0}'.format(no)] = std_dmfcc[no]
    pitches, magnitudes = librosa.core.piptrack(y, sr)
    # Select out pitches with high energy
    pitches = pitches[magnitudes > np.median(magnitudes)]
    pit = librosa.pitch_tuning(pitches)
    entry['pitch'] = pit

    wav_files.append(entry)
    wav_df = pd.DataFrame(wav_files)
    mfcc_clf = joblib.load('pickles/mfcc_model.sav')
    bar = pd.DataFrame(mfcc_clf.predict_proba(wav_df))
    bar.columns = mfcc_clf.classes_
    bar_t = bar.T
    bar_t.columns = ['values']
    fig = go.Figure(data=[go.Pie(labels=mfcc_clf.classes_, values=bar_t['values'], hole=.3), ])
    return mfcc_clf.predict(wav_df), fig
def emt(input_value):
    text_df = pd.DataFrame([{'Transcript': input_value}])
    X_new_counts = vect.transform(text_df['Transcript'])
    X_new_tfidf = tfidf_trans.transform(X_new_counts)
    clf = joblib.load('pickles/text_model')
    result = str(clf.predict(X_new_tfidf)).strip("['']")
    return 'Your emotion is "{}"'.format(result)
print('Callbacks done')