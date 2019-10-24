import os
import numpy as np
import pandas as pd
from lpc_wav_prep import LWavLoader
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
if bool(os.path.exists('pickles/lpc_features.pkl')):
    lpc_wav_loader_df = pd.read_pickle('pickles/lpc_features.pkl')
    lpc_wav_loader_df['Session'] = lpc_wav_loader_df['Session'].str.replace(r'.wav$', '')
    lpc_wav_loader_df = lpc_wav_loader_df.set_index('Session')
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
    #     lpc_wav_loader = LWavLoader(dir_wav)
    #     lpc_wav_loader_df = lpc_wav_loader.get_wav_df()
    #     wav_list.append(lpc_wav_loader_df)
    #
    #
    #
    #
    # transcript_loader_df = pd.concat(transcript_list)
    #
    # transcript_loader_df = transcript_loader_df.set_index('Session')
    # transcript_loader_df.to_pickle('pickles/alltrans.pkl')
    # evaluation_loader_df = pd.concat(evaluation_list)
    # evaluation_loader_df = evaluation_loader_df.set_index('Session')
    # evaluation_loader_df.to_pickle('pickles/alleval.pkl')
    #
    # lpc_wav_loader_df = pd.concat(wav_list)
    # lpc_wav_loader_df['Session'] = lpc_wav_loader_df['Session'].str.replace(r'.wav$', '')
    # lpc_wav_loader_df = lpc_wav_loader_df.set_index('Session')
    # lpc_wav_loader_df.to_pickle('pickles/lpc_features.pkl')


merged_data = pd.merge(transcript_loader_df, evaluation_loader_df,  how='inner', on='Session')
merged_data = pd.merge(merged_data, lpc_wav_loader_df, how='inner', on='Session')
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
X = merged_data[['LIB_LPC0', 'LIB_LPC1', 'LIB_LPC2', 'LIB_LPC3', 'LIB_LPC4', 'LIB_LPC5', 'pitch']]
Y = merged_data[['G_Emotion']]


X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.4)
print(type(y_train))
print(y_train.shape)


clf = RandomForestClassifier(n_estimators=500, max_features=0.2)
clf.fit(X_train, y_train)
importances = clf.feature_importances_
std = np.std([tree.feature_importances_ for tree in clf.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]

print("Feature ranking:")

for f in range(X.shape[1]):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))



filename = 'pickles/lpc_model.sav'
pickle.dump(clf, open(filename, 'wb'))
y_pred = clf.predict(X_test)
y_test = y_test.reset_index(drop=True)
print(accuracy_score(y_test, y_pred))
y_total = pd.concat([pd.Series(y_pred.ravel()), y_test], ignore_index=True, axis=1)
y_total.columns = ['Predicted', 'GEmotion']
print(y_total)
print(len(y_total[y_total['Predicted'] == y_total['GEmotion']]))