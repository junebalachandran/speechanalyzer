import os
import numpy as np
import pandas as pd
from mfcc_wav_prep import MFCCWavLoader
from transcript_prep import TranscriptLoader
from evaluation_prep import EvaluationLoader
import pickle
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

sessions = 5
# transcripts = "Data/Session{0}/transcriptions"
# evaluations = "Data/Session{0}/EmoEvaluation"
# wav_file_read = "Data/Session{0}/wav{0}"
transcript_list = []
evaluation_list = []
wav_list = []
if bool(os.path.exists('pickles/mfcc_features.pkl')):
    mfcc_wav_loader_df = pd.read_pickle('pickles/mfcc_features.pkl')
    mfcc_wav_loader_df['Session'] = mfcc_wav_loader_df['Session'].str.replace(r'.wav$', '')
    mfcc_wav_loader_df = mfcc_wav_loader_df.set_index('Session')
    transcript_loader_df = pd.read_pickle('pickles/alltrans.pkl')

    evaluation_loader_df = pd.read_pickle('pickles/alleval.pkl')

else:
    print('Error')
    # for session in range(1, sessions+1):
    #
    #     dir_transcript = transcripts.format(session)
    #     dir_evaluation = evaluations.format(session)
    #     dir_wav = wav_file_read.format(session)
    #
    #     #print("Loading Transcripts for session {0}:".format(session))
    #     os.chdir(dir_transcript)
    #     for transcript in os.listdir(dir_transcript):
    #         if transcript in os.listdir(dir_evaluation):
    #             #print(' Transcript: {0}'.format(transcript))
    #             transcript_loader = TranscriptLoader(transcript)
    #             transcript_loader_df = transcript_loader.get_transcript_df()
    #             transcript_list.append(transcript_loader_df)
    #
    #     #print("Loading Evaluations for session {0}:".format(session))
    #     os.chdir(dir_evaluation)
    #     for evaluation in os.listdir(dir_evaluation):
    #         if evaluation in os.listdir(dir_transcript):
    #             #print(' Evaluation: {0}'.format(evaluation))
    #             evaluation_loader = EvaluationLoader(evaluation)
    #             evaluation_loader_df = evaluation_loader.get_eval_df()
    #             evaluation_list.append(evaluation_loader_df)
    #     os.chdir(dir_wav)
    #     mfcc_wav_loader = MFCCWavLoader(dir_wav)
    #     mfcc_wav_loader_df = mfcc_wav_loader.get_wav_df()
    #     wav_list.append(mfcc_wav_loader_df)
    #
    #
    #
    #
    # transcript_loader_df = pd.concat(transcript_list)
    #
    # transcript_loader_df = transcript_loader_df.set_index('Session')
    # transcript_loader_df.to_pickle('/Users/junebalachandran/PycharmProjects/speechanalyzer/alltrans.pkl')
    # evaluation_loader_df = pd.concat(evaluation_list)
    # evaluation_loader_df = evaluation_loader_df.set_index('Session')
    # evaluation_loader_df.to_pickle('/Users/junebalachandran/PycharmProjects/speechanalyzer/alleval.pkl')
    #
    # mfcc_wav_loader_df = pd.concat(wav_list)
    # mfcc_wav_loader_df['Session'] = mfcc_wav_loader_df['Session'].str.replace(r'.wav$', '')
    # mfcc_wav_loader_df = mfcc_wav_loader_df.set_index('Session')
    # mfcc_wav_loader_df.to_pickle('/Users/junebalachandran/PycharmProjects/speechanalyzer/allwavnew.pkl')


merged_data = pd.merge(transcript_loader_df, evaluation_loader_df,  how='inner', on='Session')
merged_data = pd.merge(merged_data, mfcc_wav_loader_df, how='inner', on='Session')
merged_data.replace(to_replace='ang', value='Anger', inplace=True)
merged_data.replace(to_replace='fru', value='Frustration', inplace=True)
merged_data.replace(to_replace='sur', value='Surprise', inplace=True)
merged_data.replace(to_replace='sad', value='Sadness', inplace=True)
merged_data.replace(to_replace='hap', value='Happiness', inplace=True)
merged_data.replace(to_replace='neu', value='Neutral', inplace=True)
merged_data.replace(to_replace='exc', value='Excited', inplace=True)
merged_data.replace(to_replace='oth', value='Other', inplace=True)
merged_data.replace(to_replace='fea', value='Fear', inplace=True)
merged_data.replace(to_replace='dis', value='Disgust', inplace=True)
merged_data = merged_data[merged_data['G_Emotion'] != 'xxx']
merged_data['G_Valence'] = pd.to_numeric(merged_data['G_Valence'])
merged_data['G_Activation'] = pd.to_numeric(merged_data['G_Activation'])
merged_data['G_Dominance'] = pd.to_numeric(merged_data['G_Dominance'])
merged_data['Transcript'] = merged_data['Transcript'].str.replace(',', '')
merged_data['Transcript'] = merged_data['Transcript'].str.replace('.', '')
merged_data['Transcript'] = merged_data['Transcript'].str.replace('--', '')
X = merged_data[['Mean_DDMFCC0', 'Mean_DDMFCC1', 'Mean_DDMFCC10', 'Mean_DDMFCC11', 'Mean_DDMFCC12', 'Mean_DDMFCC13', 'Mean_DDMFCC14', 'Mean_DDMFCC15', 'Mean_DDMFCC16', 'Mean_DDMFCC17', 'Mean_DDMFCC18', 'Mean_DDMFCC19', 'Mean_DDMFCC2', 'Mean_DDMFCC3', 'Mean_DDMFCC4', 'Mean_DDMFCC5', 'Mean_DDMFCC6', 'Mean_DDMFCC7', 'Mean_DDMFCC8', 'Mean_DDMFCC9', 'Mean_Delta_MFCC0', 'Mean_Delta_MFCC1', 'Mean_Delta_MFCC10', 'Mean_Delta_MFCC11', 'Mean_Delta_MFCC12', 'Mean_Delta_MFCC13', 'Mean_Delta_MFCC14', 'Mean_Delta_MFCC15', 'Mean_Delta_MFCC16', 'Mean_Delta_MFCC17', 'Mean_Delta_MFCC18', 'Mean_Delta_MFCC19', 'Mean_Delta_MFCC2', 'Mean_Delta_MFCC3', 'Mean_Delta_MFCC4', 'Mean_Delta_MFCC5', 'Mean_Delta_MFCC6', 'Mean_Delta_MFCC7', 'Mean_Delta_MFCC8', 'Mean_Delta_MFCC9', 'Mean_MFCC0', 'Mean_MFCC1', 'Mean_MFCC10', 'Mean_MFCC11', 'Mean_MFCC12', 'Mean_MFCC13', 'Mean_MFCC14', 'Mean_MFCC15', 'Mean_MFCC16', 'Mean_MFCC17', 'Mean_MFCC18', 'Mean_MFCC19', 'Mean_MFCC2', 'Mean_MFCC3', 'Mean_MFCC4', 'Mean_MFCC5', 'Mean_MFCC6', 'Mean_MFCC7', 'Mean_MFCC8', 'Mean_MFCC9', 'Mean_RMS', 'STD_DDMFCC0', 'STD_DDMFCC1', 'STD_DDMFCC10', 'STD_DDMFCC11', 'STD_DDMFCC12', 'STD_DDMFCC13', 'STD_DDMFCC14', 'STD_DDMFCC15', 'STD_DDMFCC16', 'STD_DDMFCC17', 'STD_DDMFCC18', 'STD_DDMFCC19', 'STD_DDMFCC2', 'STD_DDMFCC3', 'STD_DDMFCC4', 'STD_DDMFCC5', 'STD_DDMFCC6', 'STD_DDMFCC7', 'STD_DDMFCC8', 'STD_DDMFCC9', 'STD_Delta_MFCC0', 'STD_Delta_MFCC1', 'STD_Delta_MFCC10', 'STD_Delta_MFCC11', 'STD_Delta_MFCC12', 'STD_Delta_MFCC13', 'STD_Delta_MFCC14', 'STD_Delta_MFCC15', 'STD_Delta_MFCC16', 'STD_Delta_MFCC17', 'STD_Delta_MFCC18', 'STD_Delta_MFCC19', 'STD_Delta_MFCC2', 'STD_Delta_MFCC3', 'STD_Delta_MFCC4', 'STD_Delta_MFCC5', 'STD_Delta_MFCC6', 'STD_Delta_MFCC7', 'STD_Delta_MFCC8', 'STD_Delta_MFCC9', 'STD_MFCC0', 'STD_MFCC1', 'STD_MFCC10', 'STD_MFCC11', 'STD_MFCC12', 'STD_MFCC13', 'STD_MFCC14', 'STD_MFCC15', 'STD_MFCC16', 'STD_MFCC17', 'STD_MFCC18', 'STD_MFCC19', 'STD_MFCC2', 'STD_MFCC3', 'STD_MFCC4', 'STD_MFCC5', 'STD_MFCC6', 'STD_MFCC7', 'STD_MFCC8', 'STD_MFCC9', 'STD_RMS', 'pitch']]
Y = merged_data[['G_Emotion']]


X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.4)
print(type(y_train))
print(y_train.shape)


clf = RandomForestClassifier(n_estimators=1000, max_features=0.2)
clf.fit(X_train, y_train)
importances = clf.feature_importances_
std = np.std([tree.feature_importances_ for tree in clf.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]

print("Feature ranking:")

for f in range(X.shape[1]):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))



filename = 'pickles/mfcc_model.sav'
pickle.dump(clf, open(filename, 'wb'))
y_pred = clf.predict(X_test)
y_test = y_test.reset_index(drop=True)
print(accuracy_score(y_test, y_pred))
y_total = pd.concat([pd.Series(y_pred.ravel()), y_test], ignore_index=True, axis=1)
y_total.columns = ['Predicted', 'GEmotion']
print(y_total)
print(len(y_total[y_total['Predicted'] == y_total['GEmotion']]))