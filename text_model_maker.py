import os

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier

from sklearn.preprocessing import LabelEncoder
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split



sessions = 5
# transcripts = "Data/Session{0}/transcriptions"
# evaluations = "Data/Session{0}/EmoEvaluation"
# nrc_vad_word = "Data/nrc/NRC-VAD-Lexicon.txt"
# testfile = "Data/Session1/EmoEvaluation/Ses01M_script02_2.txt"
# testfile2 = "Data/Session1/transcriptions/Ses01M_script02_2.txt"
transcript_list = []
evaluation_list = []
nrc_list = []
vect = CountVectorizer()

tfidf_trans = TfidfTransformer()
le = LabelEncoder()
if bool(os.path.exists('pickles/alltrans.pkl')):
    transcript_loader_df = pd.read_pickle('pickles/alltrans.pkl')
    evaluation_loader_df = pd.read_pickle('pickles/alleval.pkl')

else:
    print('Error')
    # for session in range(1, sessions+1):
    #
    #     dir_transcript = transcripts.format(session)
    #     dir_evaluation = evaluations.format(session)
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
    #
    #
    #
    # transcript_loader_df = pd.concat(transcript_list)
    # transcript_loader_df = transcript_loader_df.set_index('Session')
    # evaluation_loader_df = pd.concat(evaluation_list)
    #


"""
testcase2 = EvaluationLoader(testfile)
testcase2_df = testcase2.get_eval_df()
print(len(testcase2_df))
testcase = EvaluationLoader(testfile)
testcase_df = testcase.get_eval_df()
print(len(testcase_df))
"""

merged_data = pd.merge(transcript_loader_df, evaluation_loader_df, how='inner', on='Session')
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


X_train, X_test, y_train, y_test = train_test_split(merged_data['Transcript'], merged_data['G_Emotion'], test_size=0.2)
X_train_ct = vect.fit_transform(X_train)
X_test_ct = vect.transform(X_test)
X_train_tfidf = tfidf_trans.fit_transform(X_train_ct)
X_test_tfidf = tfidf_trans.transform(X_test_ct)
clf = MultinomialNB().fit(X_train_tfidf, y_train)
joblib.dump(clf,'pickles/text_model')


    #print(clf.predict_proba(X_test_tfidf))

view_train = pd.DataFrame(clf.predict_proba(X_train_tfidf))
view = pd.DataFrame(clf.predict_proba(X_test_tfidf))

X_train = pd.concat([X_train, view_train], axis=1)
X_test = pd.concat([X_test, view], axis=1)
X_train = X_train.drop(columns=['Transcript'])
X_train.columns = clf.classes_
X_test = X_test.drop(columns=['Transcript'])
X_test.columns = clf.classes_

