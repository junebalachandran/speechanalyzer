
Speech Analyzer
===
Speech analyzer is a web application which helps in detecting user's emotion via entering a text, uploading audio files or recording user audio.
We use four methods to detect a user's emotion and these four methods are Text-based and three ASR techniques like MFCC, LPC, RASTA.
Models created and used for this project is based on IEMOCAP dataset.
---
### Instructions

1. Install Python 3.7
2. Install PyCharm Community Edition from https://www.jetbrains.com/pycharm/download/download-thanks.html?platform=mac&code=PCC 
3. Open Pycharm and go to  File > Open and select the unzipped project folder speechanalyzer. 
The project should load under pycharm
4. Go to main menu of Pycharm>Preferences> Project:speechanalyzer> Project Interpreter, click on add Project Interpreter 
for speech analyzer
5. Create Virtualenv Environment> New Environment. Keep the location as the location of the project file
 speech analyzer.
6. Add Base Interpreter as the location of your python eg /usr/local/bin/python3
7. Once the Virtual environment is created, Go to main menu of Pycharm>Preferences> Project:speechanalyzer> Project Interpreter
 and click on the '+' in this window (after you load your interpreter) to add the following libraries.
 See link on how to add virtual environment for the project https://www.jetbrains.com/help/pycharm/creating-virtual-environment.html

Some of these will already be installed but this is the full list of packages you would need.

|  Package               |      Version  |
|:-----------------------|---------------|
| absl-py                |  0.7.1        | 
| antiorm                |  1.2.1        |
| appnope                |  0.1.0        |
| arrow                  |  0.15.2       |
| astor                  |  0.8.0        |
| attrs                  |  19.1.0       |
| aubio                  |  0.4.9        |
| audiolazy              |  0.6          |
| audioread              |  2.1.8        |
| backcall               |  0.1.0        |
| binaryornot            |  0.4.4        |
| biopython              |  1.74         |
| certifi                |  2019.6.16    |
| cffi                   |  1.12.3       |
| chardet                |  3.0.4        |
| chart-studio           |  1.0.0        |
| Click                  |  7.0          |
| colorama               |  0.4.1        |
| colour                 |  0.1.5        |
| cycler                 |  0.10.0       |
| dash                   |  1.2.0        |
| dash-bio               |  0.1.4        |
| dash-bio-utils         |  0.0.3        |
| dash-core-components   |  1.1.2        |
| dash-html-components   |  1.0.1        |
| dash-renderer          |  1.0.1        |
| dash-table             |  4.2.0        |
| db                     |  0.1.1        |
| decorator              |  4.4.0        |
| docutils               |  0.15.2       |
| Flask                  |  1.1.1        |
| Flask-Compress         |  1.4.0        |
| future                 |  0.17.1       |
| gast                   |  0.2.2        |
| GEOparse               |  1.2.0        |
| google-pasta           |  0.1.7        |
| grpcio                 |  1.22.0       |
| gunicorn               |  19.9.0       |
| h5py                   |  2.9.0        |
| idna                   |  2.8          |
| ipython                |  7.8.0        |
| ipython-genutils       |  0.2.0        |
| itsdangerous           |  1.1.0        |
| jedi                   |  0.15.1       |
| Jinja2                 |  2.10.1       |
| joblib                 |  0.13.2       |
| jsonschema             |  3.0.2        |
| kiwisolver             |  1.0.1        |
| librosa                |  0.7.0        |
| llvmlite               |  0.29.0       |
| load                   |  2019.4.1     |
| Markdown               |  3.1.1        |
| MarkupSafe             |  1.1.1        |
| matplotlib             |  3.0.3        |
| mock                   |  3.0.5        |
| nose                   |  1.3.7        |
| numba                  |  0.45.0       |
| numpy                  |  1.16.2       |
| optional-django        |  0.3.0        |
| pandas                 |  0.24.1       |
| ParmEd                 |  3.2.0        |
| parso                  |  0.5.1        |
| periodictable          |  1.5.1        |
| pexpect                |  4.7.0        |
| | pickleshare          |    0.7.5      |  
| Pillow                 |  6.1.0        |
| pip                    |  10.0.1       |
| pitch                  |  0.0.6        |
| plotly                 |  4.1.1        |
| poyo                   |  0.5.0        |
| prompt-toolkit         |  2.0.9        |
| protobuf               |  3.9.0        |
| ptyprocess             |  0.6.0        |
| public                 |  2019.3.22    |
| PyAudio                |  0.2.11       |
| pycosat                |  0.6.3        |
| pycparser              |  2.19         |
| Pygments               |  2.4.2        |
| pyparsing              |  2.4.0        |
| pyrsistent             |  0.15.4       |
| python-dateutil        |  2.8.0        |
| python-speech-features |  0.6          |
| pytz                   |  2018.9       |
| PyYAML                 |  5.1.1        |
| react                  |  4.3.0        |
| requests               |  2.22.0       |
| resampy                |  0.2.1        |
| retrying               |  1.3.3        |
| ruamel.yaml            |  0.16.5       |
| ruamel.yaml.clib       |  0.1.2        |
| scikit-learn           |  0.20.3       |
| scipy                  |  1.2.1        |
| scipyplot              |  0.0.6        |
| seaborn                |  0.9.0        |
| setuptools             |  39.1.0       |
| SIDEKIT                |  1.3.3        |
| singledispatch         |  3.4.0.3      |
| six                    |  1.12.0       |
| sklearn                |  0.0          |
| sounddevice            |  0.3.14       |
| SoundFile              |  0.10.2       |
| spectrum               |  0.7.5        |
| speechpy               |  2.4          |
| SpeechRecognition      |  3.8.1        |
| sqlparse               |  0.3.0        |
| standalone             |  1.0.1        |
| termcolor              |  1.1.0        |
| torch                  |  1.1.0.post2  |
| torchvision            |  0.3.0        |
| tqdm                   |  4.35.0       |
| traitlets              |  4.3.2        |
| umap-learn             |  0.3.10       |
| urllib3                |  1.25.3       |
| utils                  |  0.9.0        |
| wcwidth                |  0.1.7        |
| Werkzeug               |  0.15.5       |
| wheel                  |  0.33.4       |
3. Check all the pickle files and other required files exist (see list of files)
4. Launch the app:
Run newdash.py and go to link http://127.0.0.1:8050/. newdash.py uses backend functions from callback_func.py(no need to run) and text_model_maker.py(no need to run)

---
### List of files


1. pickles(Contains all the required files for the application)

    1. alleval.pkl -- Dataframe which contains ground emotions
    2. alltrans.pkl -- Dataframe which contains transciptions  
    3. catalyst.wave -- Audio data handler
    4. lpc_features -- Extracted LPC features
    5. lpc_model.sav -- LPC model
    6. mfcc_features.pkl -- Extracted LPC features
    7. mfcc_model.sav -- MFCC model
    8. rastaplp_features.pkl -- Extracted RASTA PLP features
    9. rastaplp_model.sav -- RASTA PLP model
    10.text_model -- Text-based model
    
2. callback_func.py -- contains all callback functions of newdash.py

3. evaluation_prep.py -- extracts evaluation results

4. lpc_model_maker.py -- builds model based on extracted LPC features

5. lpc_wav_prep.py - extracts LPC features

6. mfcc_model_maker.py -- builds model based on extracted MFCC features

7. mfcc_wav_prep.py - extracts MFCC features

8. newdash.py - Main web app

9. pickle_maker.py - makes direct model pickles for MFCC,LPC and RASTA-PLP

10. rasta_model_maker.py -- builds model based on extracted RASTA-PLP features

11. rastaplp_wav_prep.py - extracts RASTA-PLP features

12. text_model_maker.py - builds model based on text features

13. transcript_prep.py - extracts transcript results

14. Readme.md - description



    

