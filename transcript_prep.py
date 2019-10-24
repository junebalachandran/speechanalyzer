import pandas as pd


class TranscriptLoader:
    def __init__(self, transcript):
        self.transcript = transcript

    def _load(self):
        with open(self.transcript, 'r') as fh:
            iemocap_tscript_list = fh.read().rstrip('\n').split('\n')
        return iemocap_tscript_list

    def get_transcript_df(self):
        iemocap_tscript_list = self._load()
        iemocap_tscript_list = [entry.split(' ', 2) for entry in iemocap_tscript_list if entry[0] == 'S']
        iemocap_tscript_df = pd.DataFrame(iemocap_tscript_list, columns=('Session', 'Time', 'Transcript'))
        iemocap_tscript_df['Time'] = iemocap_tscript_df["Time"].str.replace(':', ' ')
        return iemocap_tscript_df
