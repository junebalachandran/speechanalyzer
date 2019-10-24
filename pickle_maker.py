import os
import pandas as pd
from mfcc_wav_prep import MFCCWavLoader
from rasta_wav_prep import RPLPWavLoader
from lpc_wav_prep import LWavLoader

sessions = 5
# wav_file_read = "Data/Session{0}/wav{0}"
wav_list = []
allcols_mfcc = ['Mean_DDMFCC0', 'Mean_DDMFCC1', 'Mean_DDMFCC10', 'Mean_DDMFCC11', 'Mean_DDMFCC12', 'Mean_DDMFCC13', 'Mean_DDMFCC14', 'Mean_DDMFCC15', 'Mean_DDMFCC16', 'Mean_DDMFCC17', 'Mean_DDMFCC18', 'Mean_DDMFCC19', 'Mean_DDMFCC2', 'Mean_DDMFCC3', 'Mean_DDMFCC4', 'Mean_DDMFCC5', 'Mean_DDMFCC6', 'Mean_DDMFCC7', 'Mean_DDMFCC8', 'Mean_DDMFCC9', 'Mean_Delta_MFCC0', 'Mean_Delta_MFCC1', 'Mean_Delta_MFCC10', 'Mean_Delta_MFCC11', 'Mean_Delta_MFCC12', 'Mean_Delta_MFCC13', 'Mean_Delta_MFCC14', 'Mean_Delta_MFCC15', 'Mean_Delta_MFCC16', 'Mean_Delta_MFCC17', 'Mean_Delta_MFCC18', 'Mean_Delta_MFCC19', 'Mean_Delta_MFCC2', 'Mean_Delta_MFCC3', 'Mean_Delta_MFCC4', 'Mean_Delta_MFCC5', 'Mean_Delta_MFCC6', 'Mean_Delta_MFCC7', 'Mean_Delta_MFCC8', 'Mean_Delta_MFCC9', 'Mean_MFCC0', 'Mean_MFCC1', 'Mean_MFCC10', 'Mean_MFCC11', 'Mean_MFCC12', 'Mean_MFCC13', 'Mean_MFCC14', 'Mean_MFCC15', 'Mean_MFCC16', 'Mean_MFCC17', 'Mean_MFCC18', 'Mean_MFCC19', 'Mean_MFCC2', 'Mean_MFCC3', 'Mean_MFCC4', 'Mean_MFCC5', 'Mean_MFCC6', 'Mean_MFCC7', 'Mean_MFCC8', 'Mean_MFCC9', 'Mean_RMS', 'STD_DDMFCC0', 'STD_DDMFCC1', 'STD_DDMFCC10', 'STD_DDMFCC11', 'STD_DDMFCC12', 'STD_DDMFCC13', 'STD_DDMFCC14', 'STD_DDMFCC15', 'STD_DDMFCC16', 'STD_DDMFCC17', 'STD_DDMFCC18', 'STD_DDMFCC19', 'STD_DDMFCC2', 'STD_DDMFCC3', 'STD_DDMFCC4', 'STD_DDMFCC5', 'STD_DDMFCC6', 'STD_DDMFCC7', 'STD_DDMFCC8', 'STD_DDMFCC9', 'STD_Delta_MFCC0', 'STD_Delta_MFCC1', 'STD_Delta_MFCC10', 'STD_Delta_MFCC11', 'STD_Delta_MFCC12', 'STD_Delta_MFCC13', 'STD_Delta_MFCC14', 'STD_Delta_MFCC15', 'STD_Delta_MFCC16', 'STD_Delta_MFCC17', 'STD_Delta_MFCC18', 'STD_Delta_MFCC19', 'STD_Delta_MFCC2', 'STD_Delta_MFCC3', 'STD_Delta_MFCC4', 'STD_Delta_MFCC5', 'STD_Delta_MFCC6', 'STD_Delta_MFCC7', 'STD_Delta_MFCC8', 'STD_Delta_MFCC9', 'STD_MFCC0', 'STD_MFCC1', 'STD_MFCC10', 'STD_MFCC11', 'STD_MFCC12', 'STD_MFCC13', 'STD_MFCC14', 'STD_MFCC15', 'STD_MFCC16', 'STD_MFCC17', 'STD_MFCC18', 'STD_MFCC19', 'STD_MFCC2', 'STD_MFCC3', 'STD_MFCC4', 'STD_MFCC5', 'STD_MFCC6', 'STD_MFCC7', 'STD_MFCC8', 'STD_MFCC9', 'STD_RMS', 'Session', 'pitch']
allcols_rasta = ['Mean_DDRastaPLP0', 'Mean_DDRastaPLP12', 'Mean_DDRastaPLP1', 'Mean_DDRastaPLP10', 'Mean_DDRastaPLP11', 'Mean_DDRastaPLP2', 'Mean_DDRastaPLP3', 'Mean_DDRastaPLP4', 'Mean_DDRastaPLP5', 'Mean_DDRastaPLP6', 'Mean_DDRastaPLP7', 'Mean_DDRastaPLP8', 'Mean_DDRastaPLP9', 'Mean_Delta_RastaPLP0', 'Mean_Delta_RastaPLP1', 'Mean_Delta_RastaPLP10', 'Mean_Delta_RastaPLP11', 'Mean_Delta_RastaPLP2', 'Mean_Delta_RastaPLP12', 'Mean_Delta_RastaPLP3', 'Mean_Delta_RastaPLP4', 'Mean_Delta_RastaPLP5', 'Mean_Delta_RastaPLP6', 'Mean_Delta_RastaPLP7', 'Mean_Delta_RastaPLP8', 'Mean_Delta_RastaPLP9', 'Mean_RASTAPLP0', 'Mean_RASTAPLP12' , 'Mean_RASTAPLP1', 'Mean_RASTAPLP10', 'Mean_RASTAPLP11', 'Mean_RASTAPLP2', 'Mean_RASTAPLP3', 'Mean_RASTAPLP4', 'Mean_RASTAPLP5', 'Mean_RASTAPLP6', 'Mean_RASTAPLP7', 'Mean_RASTAPLP8', 'Mean_RASTAPLP9', 'Mean_RMS', 'STD_DDRastaPLP0', 'STD_DDRastaPLP1', 'STD_DDRastaPLP10', 'STD_DDRastaPLP12','STD_DDRastaPLP11', 'STD_DDRastaPLP2', 'STD_DDRastaPLP3', 'STD_DDRastaPLP4', 'STD_DDRastaPLP5', 'STD_DDRastaPLP6', 'STD_DDRastaPLP7', 'STD_DDRastaPLP8', 'STD_DDRastaPLP9', 'STD_Delta_RastaPLP0', 'STD_Delta_RastaPLP1', 'STD_Delta_RastaPLP12','STD_Delta_RastaPLP10', 'STD_Delta_RastaPLP11', 'STD_Delta_RastaPLP2', 'STD_Delta_RastaPLP3', 'STD_Delta_RastaPLP4', 'STD_Delta_RastaPLP5', 'STD_Delta_RastaPLP6', 'STD_Delta_RastaPLP7', 'STD_Delta_RastaPLP8', 'STD_Delta_RastaPLP9', 'STD_RASTAPLP0', 'STD_RASTAPLP1', 'STD_RASTAPLP10', 'STD_RASTAPLP11','STD_RASTAPLP12','STD_RASTAPLP2', 'STD_RASTAPLP3', 'STD_RASTAPLP4', 'STD_RASTAPLP5', 'STD_RASTAPLP6', 'STD_RASTAPLP7', 'STD_RASTAPLP8', 'STD_RASTAPLP9', 'STD_RMS', 'Session', 'pitch']
allcols_lpc = ['LIB_LPC0', 'LIB_LPC1', 'LIB_LPC2', 'LIB_LPC3', 'LIB_LPC4', 'LIB_LPC5', 'Session', 'pitch']

if bool(os.path.exists('pickles/mfcc_features.pkl')):
    mwav_loader_df = pd.read_pickle('pickles/mfcc_features.pkl')
else:
    print('Pickle was not used')

    # mfcc_bucket = pd.DataFrame(columns=allcols_mfcc)
    # for session in range(1, sessions+1):
    #     dir_wav = wav_file_read.format(session)
    #     os.chdir(dir_wav)
    #     mfcc_wav_loader = MFCCWavLoader(dir_wav)
    #     mwav_loader_df = mfcc_wav_loader.get_wav_df()
    #     print('Session{0} done for extracting MFCC Features'.format(session))
    #     mfcc_bucket = pd.concat([mfcc_bucket, mwav_loader_df])
    # mfcc_bucket.to_pickle('pickles/mfcc_features.pkl')



if bool(os.path.exists('pickles/rastaplp_features.pkl')):
    rwav_loader_df = pd.read_pickle('pickles/rastaplp_features.pkl')
else:
    print('Pickle was not used')

    # rasta_bucket = pd.DataFrame(columns=allcols_rasta)
    # for session in range(1, sessions+1):
    #     dir_wav = wav_file_read.format(session)
    #     os.chdir(dir_wav)
    #     rastaplp_wav_loader = RPLPWavLoader(dir_wav)
    #     rwav_loader_df = rastaplp_wav_loader.get_wav_df()
    #     print('Session{0} done for extracting RASTA_PLP features'.format(session))
    #     rasta_bucket = pd.concat([rasta_bucket, rwav_loader_df])
    # rasta_bucket.to_pickle('pickle/rastaplp_features.pkl')


if bool(os.path.exists('pickles/lpc_features.pkl')):
    lpc_wav_loader_df = pd.read_pickle('pickles/lpc_features.pkl')
else:
    print('Pickle was not used')

    # lpc_bucket = pd.DataFrame(columns=allcols_lpc)
    # for session in range(1, sessions+1):
    #     dir_wav = wav_file_read.format(session)
    #     os.chdir(dir_wav)
    #     lpc_wav_loader = LWavLoader(dir_wav)
    #     lpc_wav_loader_df = lpc_wav_loader.get_wav_df()
    #     print('Session{0} done for extracting LPC features'.format(session))
    #     lpc_bucket = pd.concat([lpc_bucket, lpc_wav_loader_df])
    # lpc_bucket.to_pickle('pickle/lpc_features.pkl')
    #
