# -*- coding: utf-8 -*-
import pandas as pd
import dash
import dash_html_components as html
import dash_core_components as dcc
from dash.dependencies import Input, Output, State
from callback_func import mfcc_emotion_upload,lpc_emotion_upload, decoder
import speech_recognition as sr
from flask import Flask
from sklearn.externals import joblib
from text_model_maker import vect, tfidf_trans
from dash.exceptions import PreventUpdate
import plotly.graph_objects as go
from sidekit.frontend.features import plp
import librosa
import numpy as np
import wave

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
    y, sr = librosa.load('pickles/catalyst.wav')
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
r = sr.Recognizer()
mic = sr.Microphone()

colors = {
    'background': '#111111',
    'text': '#7FDBFF'
}
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = Flask(__name__)
app = dash.Dash(__name__, external_stylesheets=external_stylesheets,meta_tags=[
        {
            "name": "viewport",
            "content": "width=device-width, initial-scale=1, maximum-scale=1.0, user-scalable=no",
        }
    ],)
app.config.suppress_callback_exceptions = True
app.layout = html.Div(style = {'border-radius': '45px',"box-shadow": "0 4px 8px 0 rgba(0, 0, 0, 0.2), 0 6px 20px 0 rgba(0, 0, 0, 0.19)",'background':'#d8dee8'},children =[
    html.Header(
        style={
            'background': '#3a769e',
            'height': '50px',
            'textAlign': 'left',
            'color': 'white',
            'padding':'4px',
            'font-size':'40px',
            'font-style': 'italic',
        }, children='Speech Analyzer'

    ),
    html.Br(),
    html.Br(),
    html.Div(id="tab-area", children=[
    dcc.Tabs(id="tabs",style = {"box-shadow": "0 4px 8px 0 rgba(0, 0, 0, 0.2), 0 6px 20px 0 rgba(0, 0, 0, 0.19)"}, children=[
        dcc.Tab(label='Description', children=[
            html.Br(),
            html.Br(),

            html.Div(children=[
                html.Div(style={'width': '5%', 'display': 'inline-block'}),
                html.Div(style={'border-radius': '15px','width': '90%',
                                "box-shadow": "0 4px 8px 0 rgba(0, 0, 0, 0.2), 0 6px 20px 0 rgba(0, 0, 0, 0.19)",
                                'display': 'inline-block','background-color':'white'}, children=[
                    html.H2(children=[html.P(style={'text-indent': '5%'},children ='Description'),html.Hr(style={'border-width': '3px'})]),
                    html.P(style={'padding':'10px','font-size':'20px'},children="Our goal is to design an application to recognize the details of the user’s speech and our second goal is to take those collected details then identify the emotion of that speech pattern and classify based on the level of percentage of these emotions i.e ’Anger’,’Happiness’. ‘Sadness’ ‘Neutral’, ‘Frustration’, ‘Excited' or ‘Others’")
                    ,
                    html.P(style={'padding':'10px','font-size':'20px'},children='The motivation behind this project is to help in the field of technology and psychology.'
                                    'We will be able to identify different emotional disorders or heavy moods.'
                                    'We can develop this application and embedd this technology in a user’s devices or any other electronic such as Alexa or Google Home'
                                    'It can help detecting suicidal thoughts or any kind of sadness.')]),

                html.Div(style={'width': '5%', 'display': 'inline-block'}),
            ], style={'border-radius': '15px','width': '100%', 'display': 'inline-block'})
        ]),






        dcc.Tab(label='Text-Based', children=[
            html.Br(),
            html.Br(),

            html.Div(children=[
                html.Div(style={'width':'10%','display':'inline-block','vertical-align':'top'}),
                html.Div(style={'width': '80%',
                                "box-shadow": "0 4px 8px 0 rgba(0, 0, 0, 0.2), 0 6px 20px 0 rgba(0, 0, 0, 0.19)",
                                'display': 'inline-block','vertical-align': 'top','border-radius': '5px 30px 30px 30px','background':'#87AFC7'}, children=[
                                    html.H3(style={'padding':'10px'},children='Text-Based Emotion Analysis'),
                                    html.Hr(style={'border': '1px solid black'}),
                                    html.Div(id='texttab',children=[dcc.Tabs(style = {"box-shadow": "0 4px 8px 0 rgba(0, 0, 0, 0.2), 0 6px 20px 0 rgba(0, 0, 0, 0.19)"}, children=[
                                        dcc.Tab(label='Write your statement', children=[
                                        html.Div(children=[
                                            html.Div(style={'width':'5%','display':'inline-block'}),
                                            html.Div(style={'width':'30%','display':'inline-block'},children=[html.Br(),html.H2(children='Write the statement of which you want to detect emotion'),dcc.Input(
                                                    id='text_input',
                                                    placeholder='Enter your statement',
                                                    style={'align': '', 'text-indent': '5%'}
                                                ),                                    html.Button('Submit', id='Tbutton'),html.Br(),html.H3(id='text_output',style={'color':'Green'}),html.Br(),html.H3(id='TEmotion',style={'color':'Red'})]),
                                            html.Div(style={'width': '10%', 'display': 'inline-block'}),
                                            html.Div(style={'width':'50%','display':'inline-block'},
                                             children=[html.Div(style={'padding':'10px'},children=[dcc.Graph(id = 'plot')])]),


                                            ],style={'width': '100%', 'display': 'inline-block'})
                                             ]),

                                        dcc.Tab(label='Upload your file', children=[html.Div(children=[
                                            html.Div(style={'width':'5%','display':'inline-block'}),
                                            html.Div(style={'width':'30%','display':'inline-block'},children=[html.Br(),html.H2(children='Choose the audio file of which you want to detect emotion'),dcc.Upload(
                                            id='upload-data',
                                            children=[html.Div([
                                                'Drag and Drop or ',
                                                html.A('Select Files')
                                            ]),html.H3(id='text_output2',style={'color':'Green'}),html.Br(),html.H3(id='TEmotion2',style={'color':'Red'})],
                                            style={
                                                'width': '90%',
                                                'height': '60px',
                                                'lineHeight': '60px',
                                                'borderWidth': '1px',
                                                'borderStyle': 'dashed',
                                                'borderRadius': '5px',
                                                'textAlign': 'center',
                                                'margin': '10px'
                                            },
                                            # Allow multiple files to be uploaded
                                            multiple=True
                                        ),]),
                                            html.Div(style={'width': '5%', 'display': 'inline-block'}),
                                            html.Div(style={'width':'50%','display':'inline-block'},
                                             children=[html.Div(style={'padding':'10px'}),html.Div(children=[dcc.Graph(id = 'plot2')])]),


                                            ],style={'width': '100%', 'display': 'inline-block'})]),
                                        dcc.Tab(label='Record your audio', children=[html.Div(children=[
                                            html.Div(style={'width': '5%', 'display': 'inline-block'}),
                                            html.Div(style={'width': '30%', 'display': 'inline-block'},
                                                     children=[html.Br(),html.H2(children='Press the record and record the statement(wait 2s)'),
                                                    html.Button('Record my audio', id='tb-button'),html.H3(id='text_output3',style={'color':'Green'}),html.Br(),html.H3(id='TEmotion3',style={'color':'Red'})
                                                     ]),
                                            html.Div(style={'width': '5%', 'display': 'inline-block'}),
                                            html.Div(style={'width': '50%', 'display': 'inline-block'},
                                                     children=[html.Div(children=[html.Div(style={'padding':'10px'}),
                                                                                  dcc.Graph(id='plot3')])]),

                                        ], style={'width': '100%', 'display': 'inline-block'})])])

                                    ]),


                                    html.Br()



                    ]),
                html.Div(style = {'width':'10%'})

            ], style={'width': '100%', 'display': 'inline-block'})
        ]),






        dcc.Tab(label='MFCC', children=[
            html.Br(),
            html.Br(),

            html.Div(children=[
                html.Div(style={'width':'10%','display':'inline-block','vertical-align':'top'}),
                html.Div(style={'width': '80%',
                                "box-shadow": "0 4px 8px 0 rgba(0, 0, 0, 0.2), 0 6px 20px 0 rgba(0, 0, 0, 0.19)",
                                'display': 'inline-block','vertical-align': 'top','border-radius': '5px 30px 30px 30px','background':'#87AFC7'}, children=[
                                    html.H3(style={'padding':'10px'},children='MFCC-Based Emotion Analysis'),
                                    html.Hr(style={'border': '1px solid black'}),
                                    html.Div(id='mfcctab',children=[dcc.Tabs(style = {"box-shadow": "0 4px 8px 0 rgba(0, 0, 0, 0.2), 0 6px 20px 0 rgba(0, 0, 0, 0.19)"}, children=[


                                        dcc.Tab(label='Upload your file', children=[html.Div(children=[
                                            html.Div(style={'width':'5%','display':'inline-block'}),
                                            html.Div(style={'width':'30%','display':'inline-block'},children=[html.H2(children='Choose the audio file of which you want to detect emotion'),dcc.Upload(
                                            id='uploadmfcc',
                                            children=html.Div([
                                                'Drag and Drop or ',
                                                html.A('Select Files')
                                            ],),
                                            style={
                                                'width': '90%',
                                                'height': '60px',
                                                'lineHeight': '60px',
                                                'borderWidth': '1px',
                                                'borderStyle': 'dashed',
                                                'borderRadius': '5px',
                                                'textAlign': 'center',
                                                'margin': '10px'
                                            },
                                            # Allow multiple files to be uploaded
                                            multiple=True
                                        ),html.H3(id='M_output',style={'color':'Green'}),html.Br(),html.H3(id='MEmotion',style={'color':'Red'})]),
                                            html.Div(style={'width': '10%', 'display': 'inline-block'}),
                                            html.Div(style={'width':'50%','display':'inline-block'},
                                             children=[html.Div(style={'padding':'10px'}),html.Div(children=[dcc.Graph(id = 'plotmfcc')])]),


                                            ],style={'width': '100%', 'display': 'inline-block'})]),
                                        dcc.Tab(label='Record your audio', children=[html.Div(children=[
                                            html.Div(style={'width': '5%', 'display': 'inline-block'}),
                                            html.Div(style={'width': '30%', 'display': 'inline-block'},
                                                     children=[html.Br(), html.H2(
                                                         children='Press the record and record the statement(wait 2s)'),
                                                               html.Button('Record my audio', id='mb-button'),
                                                               html.H3(id='M_output2', style={'color': 'Green'}),
                                                               html.Br(),
                                                               html.H3(id='MEmotion2', style={'color': 'Red'})
                                                               ]),
                                            html.Div(style={'width': '10%', 'display': 'inline-block'}),
                                            html.Div(style={'width': '50%', 'display': 'inline-block'},
                                                     children=[html.Div(children=[html.Div(style={'padding': '10px'}),
                                                                                  dcc.Graph(id='plotmfcc2')])]),

                                        ], style={'width': '100%', 'display': 'inline-block'})])])

                                    ]),


                                    html.Br()



                    ]),
                html.Div(style = {'width':'10%'})

            ], style={'width': '100%', 'display': 'inline-block'})
        ]),

        dcc.Tab(label='LPC', children=[
            html.Br(),
            html.Br(),

            html.Div(children=[
                html.Div(style={'width': '10%', 'display': 'inline-block', 'vertical-align': 'top'}),
                html.Div(style={'width': '80%',
                                "box-shadow": "0 4px 8px 0 rgba(0, 0, 0, 0.2), 0 6px 20px 0 rgba(0, 0, 0, 0.19)",
                                'display': 'inline-block', 'vertical-align': 'top',
                                'border-radius': '5px 30px 30px 30px', 'background': '#87AFC7'}, children=[
                    html.H3(style={'padding': '10px'}, children='LPC-Based Emotion Analysis'),
                    html.Hr(style={'border': '1px solid black'}),
                    html.Div(id='lpctab', children=[dcc.Tabs(
                        style={"box-shadow": "0 4px 8px 0 rgba(0, 0, 0, 0.2), 0 6px 20px 0 rgba(0, 0, 0, 0.19)"},
                        children=[

                            dcc.Tab(label='Upload your file', children=[html.Div(children=[
                                html.Div(style={'width': '5%', 'display': 'inline-block'}),
                                html.Div(style={'width': '30%', 'display': 'inline-block'}, children=[
                                    html.H2(children='Choose the audio file of which you want to detect emotion'),
                                    dcc.Upload(
                                        id='uploadlpc',
                                        children=html.Div([
                                            'Drag and Drop or ',
                                            html.A('Select Files')
                                        ], ),
                                        style={
                                            'width': '90%',
                                            'height': '60px',
                                            'lineHeight': '60px',
                                            'borderWidth': '1px',
                                            'borderStyle': 'dashed',
                                            'borderRadius': '5px',
                                            'textAlign': 'center',
                                            'margin': '10px'
                                        },
                                        # Allow multiple files to be uploaded
                                        multiple=True
                                    ), html.H3(id='L_output', style={'color': 'Green'}), html.Br(),
                                    html.H3(id='LEmotion', style={'color': 'Red'})]),
                                html.Div(style={'width': '10%', 'display': 'inline-block'}),
                                html.Div(style={'width': '50%', 'display': 'inline-block'},
                                         children=[html.Div(style={'padding': '10px'}),
                                                   html.Div(children=[dcc.Graph(id='plotlpc')])]),

                            ], style={'width': '100%', 'display': 'inline-block'})]),
                            dcc.Tab(label='Record your audio', children=[html.Div(children=[
                                html.Div(style={'width': '5%', 'display': 'inline-block'}),
                                html.Div(style={'width': '30%', 'display': 'inline-block'},
                                         children=[html.Br(), html.H2(
                                             children='Press the record and record the statement(wait 2s)'),
                                                   html.Button('Record my audio', id='lb-button'),
                                                   html.H3(id='L_output2', style={'color': 'Green'}),
                                                   html.Br(),
                                                   html.H3(id='LEmotion2', style={'color': 'Red'})
                                                   ]),
                                html.Div(style={'width': '10%', 'display': 'inline-block'}),
                                html.Div(style={'width': '50%', 'display': 'inline-block'},
                                         children=[html.Div(children=[html.Div(style={'padding': '10px'}),
                                                                      dcc.Graph(id='plotlpc2')])]),

                            ], style={'width': '100%', 'display': 'inline-block'})])])

                                                     ]),

                    html.Br()

                ]),
                html.Div(style={'width': '10%'})

            ], style={'width': '100%', 'display': 'inline-block'})
        ]),

        dcc.Tab(label='RASTA PLP', children=[
            html.Br(),
            html.Br(),

            html.Div(children=[
                html.Div(style={'width': '10%', 'display': 'inline-block', 'vertical-align': 'top'}),
                html.Div(style={'width': '80%',
                                "box-shadow": "0 4px 8px 0 rgba(0, 0, 0, 0.2), 0 6px 20px 0 rgba(0, 0, 0, 0.19)",
                                'display': 'inline-block', 'vertical-align': 'top',
                                'border-radius': '5px 30px 30px 30px', 'background': '#87AFC7'}, children=[
                    html.H3(style={'padding': '10px'}, children='RASTA PLP-Based Emotion Analysis'),
                    html.Hr(style={'border': '1px solid black'}),
                    html.Div(id='rastatab', children=[dcc.Tabs(
                        style={"box-shadow": "0 4px 8px 0 rgba(0, 0, 0, 0.2), 0 6px 20px 0 rgba(0, 0, 0, 0.19)"},
                        children=[

                            dcc.Tab(label='Upload your file', children=[html.Div(children=[
                                html.Div(style={'width': '5%', 'display': 'inline-block'}),
                                html.Div(style={'width': '30%', 'display': 'inline-block'}, children=[
                                    html.H2(children='Choose the audio file of which you want to detect emotion'),
                                    dcc.Upload(
                                        id='uploadrasta',
                                        children=html.Div([
                                            'Drag and Drop or ',
                                            html.A('Select Files')
                                        ], ),
                                        style={
                                            'width': '90%',
                                            'height': '60px',
                                            'lineHeight': '60px',
                                            'borderWidth': '1px',
                                            'borderStyle': 'dashed',
                                            'borderRadius': '5px',
                                            'textAlign': 'center',
                                            'margin': '10px'
                                        },
                                        # Allow multiple files to be uploaded
                                        multiple=True
                                    ), html.H3(id='R_output', style={'color': 'Green'}), html.Br(),
                                    html.H3(id='REmotion', style={'color': 'Red'})]),
                                html.Div(style={'width': '10%', 'display': 'inline-block'}),
                                html.Div(style={'width': '50%', 'display': 'inline-block'},
                                         children=[html.Div(style={'padding': '10px'}),
                                                   html.Div(children=[dcc.Graph(id='plotrasta')])]),

                            ], style={'width': '100%', 'display': 'inline-block'})]),
                            dcc.Tab(label='Record your audio', children=[html.Div(children=[
                                html.Div(style={'width': '5%', 'display': 'inline-block'}),
                                html.Div(style={'width': '30%', 'display': 'inline-block'},
                                         children=[html.Br(), html.H2(
                                             children='Press the record and record the statement(wait 2s)'),
                                                   html.Button('Record my audio', id='rb-button'),
                                                   html.H3(id='R_output2', style={'color': 'Green'}),
                                                   html.Br(),
                                                   html.H3(id='REmotion2', style={'color': 'Red'})
                                                   ]),
                                html.Div(style={'width': '10%', 'display': 'inline-block'}),
                                html.Div(style={'width': '50%', 'display': 'inline-block'},
                                         children=[html.Div(children=[html.Div(style={'padding': '10px'}),
                                                                      dcc.Graph(id='plotrasta2')])]),

                            ], style={'width': '100%', 'display': 'inline-block'})])])

                                                     ]),

                    html.Br()

                ]),
                html.Div(style={'width': '10%'})

            ], style={'width': '100%', 'display': 'inline-block'})
        ]),





    ]),
    ]),
    html.Br(),
    html.Br(),
    html.Br(),
    html.Br(),
    html.Br(),

    html.Footer(
        style={
            'background': '#3a769e',
            'height': '20%',
            'textAlign': 'left',
            'color': 'white'
        }, children=[html.H1(style={'textAlign': 'right'},children='JB')]

    )
])

#########################TEXT BASED ENTER STATEMENT#############################################
@app.callback(
    Output(component_id='text_output', component_property='children'),
    [dash.dependencies.Input('Tbutton', 'n_clicks')],
    [dash.dependencies.State('text_input', 'value')]
)
def update_output_div(n_clicks,input_value):
    return 'You\'ve entered "{}"'.format(input_value)
@app.callback(
    Output(component_id='TEmotion', component_property='children'),
    [dash.dependencies.Input('Tbutton', 'n_clicks')],
    [dash.dependencies.State('text_input', 'value')]
)
def update_EText(n_clicks,input_value):
    if n_clicks is None:
        return PreventUpdate
    else:
        emotion, fig = text_emo(input_value)
        return emotion

@app.callback(
    Output(component_id='plot', component_property='figure'),
    [dash.dependencies.Input('Tbutton', 'n_clicks')],
    [dash.dependencies.State('text_input', 'value')]
)
def update_EText(n_clicks,input_value):
    if n_clicks is None:
        return PreventUpdate
    else:
        emotion, fig = text_emo(input_value)

        return fig
###############################################################################################
#########################TEXT BASED Drag#############################################

@app.callback(
    [Output('text_output2', 'children'),
     Output('TEmotion2', 'children'),
     Output('plot2', 'figure')],
    [Input("upload-data", "filename"), Input("upload-data", "contents")])
def callback_a(uploaded_filenames, uploaded_file_contents):
    if uploaded_file_contents is None:
        return PreventUpdate
    else:

        boolean = decoder(uploaded_file_contents)
        if boolean == 1:
            testfile = sr.AudioFile('pickles/catalyst.wav')
            with testfile as source:
                audio = r.record(source)
            text = r.recognize_google(audio)
            emotion, fig = text_emo(text)
            return text, emotion, fig
        else:
            return PreventUpdate


###############################################################################################
#########################TEXT BASED REcord#############################################





@app.callback(
    [Output('text_output3', 'children'),
     Output('TEmotion3', 'children'),
     Output('plot3', 'figure')],
    [dash.dependencies.Input('tb-button', 'n_clicks')])
def callback_a(n_clicks):
        if n_clicks != None:
            print('Speak now text based')
            with mic as source:
                r.adjust_for_ambient_noise(source)
                audio = r.listen(source)
            text = r.recognize_google(audio)
            emotion, fig = text_emo(text)
            return text, emotion, fig
        else:
            return PreventUpdate


###############################################################################################
######################### mfcc BASED upload#############################################


@app.callback(
    [Output('M_output', 'children'),
     Output('MEmotion', 'children'),
     Output('plotmfcc', 'figure')],
    [Input("uploadmfcc", "filename"), Input("uploadmfcc", "contents")])
def callback_a(uploaded_filenames, uploaded_file_contents):
    if uploaded_file_contents is None:
        return PreventUpdate
    else:

        boolean = decoder(uploaded_file_contents)
        if boolean == 1:
            testfile = sr.AudioFile('pickles/catalyst.wav')
            with testfile as source:
                audio = r.record(source)
            text = r.recognize_google(audio)
            emotion, fig = mfcc_emotion_upload()
            return 'Detected Text: "{}"'.format(text), 'Detected Emotion"{}"'.format(emotion[0]), fig
        else:
            return PreventUpdate
###############################################################################################
######################### mfcc BASED record#############################################

@app.callback(
    [Output('M_output2', 'children'),
     Output('MEmotion2', 'children'),
     Output('plotmfcc2', 'figure')],
    [dash.dependencies.Input('mb-button', 'n_clicks')])
def callback_a(x):
    if x is None:
        return PreventUpdate
    else:
        with mic as source:
            r.adjust_for_ambient_noise(source)
            audio = r.listen(source)
        with open('pickles/catalyst.wav','wb') as f:
            f.write(audio.get_wav_data())
        text = r.recognize_google(audio)
        emotion, fig = mfcc_emotion_upload()
        return text, emotion, fig
###############################################################################################
######################### lpc BASED upload#############################################


@app.callback(
    [Output('L_output', 'children'),
     Output('LEmotion', 'children'),
     Output('plotlpc', 'figure')],
    [Input("uploadlpc", "filename"), Input("uploadlpc", "contents")])
def callback_a(uploaded_filenames, uploaded_file_contents):
    if uploaded_file_contents is None:
        return PreventUpdate
    else:

        boolean = decoder(uploaded_file_contents)
        if boolean == 1:
            testfile = sr.AudioFile('pickles/catalyst.wav')
            with testfile as source:
                audio = r.record(source)
            text = r.recognize_google(audio)
            emotion, fig = lpc_emotion_upload()
            return 'Detected Text: "{}"'.format(text), 'Detected Emotion"{}"'.format(emotion[0]), fig
        else:
            return PreventUpdate

###############################################################################################
######################### lpc BASED record#############################################

@app.callback(
    [Output('L_output2', 'children'),
     Output('LEmotion2', 'children'),
     Output('plotlpc2', 'figure')],
    [dash.dependencies.Input('lb-button', 'n_clicks')])
def callback_a(x):
    if x is None:
        return PreventUpdate
    else:
        with mic as source:
            r.adjust_for_ambient_noise(source)
            audio = r.listen(source)
        with open('pickles/catalyst.wav', 'wb') as f:
            f.write(audio.get_wav_data())
        text = r.recognize_google(audio)
        emotion, fig = lpc_emotion_upload()
        return text, emotion, fig

###############################################################################################
######################### rasta BASED upload#############################################


@app.callback(
    [Output('R_output', 'children'),
     Output('REmotion', 'children'),
     Output('plotrasta', 'figure')],
    [Input("uploadrasta", "filename"), Input("uploadrasta", "contents")])
def callback_a(uploaded_filenames, uploaded_file_contents):
    if uploaded_file_contents is None:
        return PreventUpdate
    else:

        boolean = decoder(uploaded_file_contents)
        if boolean == 1:
            testfile = sr.AudioFile('pickles/catalyst.wav')
            with testfile as source:
                audio = r.record(source)

            emotion, fig = rasta_emotion_upload()
            text = r.recognize_google(audio)
            return 'Detected Text: "{}"'.format(text), 'Detected Emotion"{}"'.format(emotion[0]), fig
        else:
            return PreventUpdate
###############################################################################################
######################### rasta BASED record#############################################

@app.callback(
    [Output('R_output2', 'children'),
     Output('REmotion2', 'children'),
     Output('plotrasta2', 'figure')],
    [dash.dependencies.Input('rb-button', 'n_clicks')])
def callback_a(x):
    if x is None:
        return PreventUpdate
    else:
        with mic as source:
            r.adjust_for_ambient_noise(source)
            audio = r.listen(source)
        with open('pickles/catalyst.wav', 'wb') as f:
            f.write(audio.get_wav_data())
        text = r.recognize_google(audio)
        emotion, fig = rasta_emotion_upload()
        return text, emotion, fig

if __name__ == '__main__':
    app.run_server(debug=False)