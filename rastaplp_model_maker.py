import os
import numpy as np
import pandas as pd
from rasta_wav_prep import RPLPWavLoader
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
if bool(os.path.exists('pickles/rastaplp_features.pkl')):
    rasta_wav_loader_df = pd.read_pickle('pickles/rastaplp_features.pkl')
    transcript_loader_df = pd.read_pickle('pickles/alltrans.pkl')

    evaluation_loader_df = pd.read_pickle('pickles/alleval.pkl')

else:
    print('pickles were not used')
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
    #     rasta_wav_loader = RPLPWavLoader(dir_wav)
    #     rasta_wav_loader_df = rasta_wav_loader.get_wav_df()
    #     wav_list.append(rasta_wav_loader_df)
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
    # rasta_wav_loader_df = pd.concat(wav_list)
    # rasta_wav_loader_df['Session'] = rasta_wav_loader_df['Session'].str.replace(r'.wav$', '')
    # rasta_wav_loader_df = rasta_wav_loader_df.set_index('Session')
    # rasta_wav_loader_df.to_pickle('pickles/rastaplp_features.pkl')


merged_data = pd.merge(transcript_loader_df, evaluation_loader_df,  how='inner', on='Session')
merged_data = pd.merge(merged_data, rasta_wav_loader_df, how='inner', on='Session')
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
X = merged_data[['Mean_DDRastaPLP0', 'Mean_DDRastaPLP12', 'Mean_DDRastaPLP1', 'Mean_DDRastaPLP10', 'Mean_DDRastaPLP11', 'Mean_DDRastaPLP2', 'Mean_DDRastaPLP3', 'Mean_DDRastaPLP4', 'Mean_DDRastaPLP5', 'Mean_DDRastaPLP6', 'Mean_DDRastaPLP7', 'Mean_DDRastaPLP8', 'Mean_DDRastaPLP9', 'Mean_Delta_RastaPLP0', 'Mean_Delta_RastaPLP1', 'Mean_Delta_RastaPLP10', 'Mean_Delta_RastaPLP11', 'Mean_Delta_RastaPLP2', 'Mean_Delta_RastaPLP12', 'Mean_Delta_RastaPLP3', 'Mean_Delta_RastaPLP4', 'Mean_Delta_RastaPLP5', 'Mean_Delta_RastaPLP6', 'Mean_Delta_RastaPLP7', 'Mean_Delta_RastaPLP8', 'Mean_Delta_RastaPLP9', 'Mean_RASTAPLP0', 'Mean_RASTAPLP12' , 'Mean_RASTAPLP1', 'Mean_RASTAPLP10', 'Mean_RASTAPLP11', 'Mean_RASTAPLP2', 'Mean_RASTAPLP3', 'Mean_RASTAPLP4', 'Mean_RASTAPLP5', 'Mean_RASTAPLP6', 'Mean_RASTAPLP7', 'Mean_RASTAPLP8', 'Mean_RASTAPLP9', 'Mean_RMS', 'STD_DDRastaPLP0', 'STD_DDRastaPLP1', 'STD_DDRastaPLP10', 'STD_DDRastaPLP12','STD_DDRastaPLP11', 'STD_DDRastaPLP2', 'STD_DDRastaPLP3', 'STD_DDRastaPLP4', 'STD_DDRastaPLP5', 'STD_DDRastaPLP6', 'STD_DDRastaPLP7', 'STD_DDRastaPLP8', 'STD_DDRastaPLP9', 'STD_Delta_RastaPLP0', 'STD_Delta_RastaPLP1', 'STD_Delta_RastaPLP12','STD_Delta_RastaPLP10', 'STD_Delta_RastaPLP11', 'STD_Delta_RastaPLP2', 'STD_Delta_RastaPLP3', 'STD_Delta_RastaPLP4', 'STD_Delta_RastaPLP5', 'STD_Delta_RastaPLP6', 'STD_Delta_RastaPLP7', 'STD_Delta_RastaPLP8', 'STD_Delta_RastaPLP9', 'STD_RASTAPLP0', 'STD_RASTAPLP1', 'STD_RASTAPLP10', 'STD_RASTAPLP11','STD_RASTAPLP12','STD_RASTAPLP2', 'STD_RASTAPLP3', 'STD_RASTAPLP4', 'STD_RASTAPLP5', 'STD_RASTAPLP6', 'STD_RASTAPLP7', 'STD_RASTAPLP8', 'STD_RASTAPLP9', 'STD_RMS', 'pitch']]
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



filename = 'pickles/rastaplp_model.sav'
pickle.dump(clf, open(filename, 'wb'))
y_pred = clf.predict(X_test)
y_test = y_test.reset_index(drop=True)
print(accuracy_score(y_test, y_pred))
y_total = pd.concat([pd.Series(y_pred.ravel()), y_test], ignore_index=True, axis=1)
y_total.columns = ['Predicted', 'GEmotion']
print(y_total)
print(len(y_total[y_total['Predicted'] == y_total['GEmotion']]))