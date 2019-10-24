import pandas as pd
import re

class EvaluationLoader:
    def __init__(self, evaluation):
        self.evaluation = evaluation

    def _load(self):
        with open(self.evaluation, 'r') as fs:
            iemocap_eval_list = fs.read().rstrip('\n').split('\n')
        return iemocap_eval_list

    def get_eval_df(self):

        newlist = []
        entry = dict()
        iemocap_eval_list = self._load()

        for line in iemocap_eval_list:
            if line == '':
                if bool(entry):
                    newlist.append(entry)
                entry = dict()
            else:
                if line.startswith('['):
                    entry['Time'] = re.search('(\d*.\d* - \d*.\d*)', line).group(1)
                    entry['Session'] = re.search('(Ses.*_.\d*)', line).group(1)

                    entry['G_Emotion'] = re.search('\s([a-z].*)\s\[', line).group(1)
                    entry['G_Valence'] = re.search('\[(\d.\d*),', line).group(1)
                    entry['G_Activation'] = re.search(',\s(\d.\d*),', line).group(1)
                    entry['G_Dominance'] = re.search(',\s(\d.\d*)]', line).group(1)
                elif line.startswith('C-E1:'):
                    c_e1 = re.search(':\s([a-zA-Z].*);', line)
                    if not c_e1:
                        entry['C_E1'] = 'nan'
                    else:
                        entry['C_E1'] = re.search(':\s([a-zA-Z].*);', line).group(1)

                    entry['C_E1_note'] = re.search(';\s(\(.*\))', line).group(1)
                elif line.startswith('C-E2:'):
                    c_e2 = re.search(':\s([a-zA-Z].*);', line)
                    if not c_e2:
                        entry['C_E2'] = 'nan'
                    else:
                        entry['C_E2'] = re.search(':\s([a-zA-Z].*);', line).group(1)

                    entry['C_E2_note'] = re.search(';\s(\(.*\))', line).group(1)
                elif line.startswith('C-E3:'):
                    c_e3 = re.search(':\s([a-zA-Z].*);', line)
                    if not c_e3:
                        entry['C_E3'] = 'nan'
                    else:
                        entry['C_E3'] = re.search(':\s([a-zA-Z].*);', line).group(1)
                    entry['C_E3_note'] = re.search(';\s(\(.*\))', line).group(1)
                elif line.startswith('C-E4:'):
                    c_e4 = re.search(':\s([a-zA-Z].*);', line)
                    if not c_e4:
                        entry['C_E4'] = 'nan'
                    else:
                        entry['C_E4'] = re.search(':\s([a-zA-Z].*);', line).group(1)
                    entry['C_E4_note'] = re.search(';\s(\(.*\))', line).group(1)
                elif line.startswith('C-F1:') or line.startswith('C-M1:'):
                    c_mf1_reg = re.search(':\s([a-zA-Z].*);', line)
                    if not c_mf1_reg:
                        entry['C_MF1'] = 'nan'
                    else:
                        entry['C_MF1'] = re.search(':\s([a-zA-Z].*);', line).group(1)
                    entry['C_MF1_note'] = re.search(';\s(\(.*\))', line).group(1)
                elif line.startswith('A-E1:'):
                    a_e1_val = re.search('val\s(\d*.\d*);', line)
                    if not a_e1_val:
                        entry['A_E1_val'] = '0'
                    else:
                        entry['A_E1_val'] = re.search('val\s(\d*.\d*);', line).group(1)
                    a_e1_act = re.search('act\s(\d*.\d*);', line)
                    if not a_e1_act:
                        entry['A_E1_act'] = '0'
                    else:
                        entry['A_E1_act'] = re.search('act\s(\d*.\d*);', line).group(1)
                    a_e1_dom = re.search('dom\s(\d*.\d*);', line)
                    if not a_e1_dom:
                        entry['A_E1_dom'] = '0'
                    else:
                        entry['A_E1_dom'] = re.search('dom\s(\d*.\d*);', line).group(1)
                    entry['A_E1_note'] = re.search(';\s(\(.*\))', line).group(1)
                elif line.startswith('A-E2:'):
                    a_e2_val = re.search('val\s(\d*.\d*);', line)
                    if not a_e2_val:
                        entry['A_E2_val'] = '0'
                    else:
                        entry['A_E2_val'] = re.search('val\s(\d*.\d*);', line).group(1)
                    a_e2_act = re.search('act\s(\d*.\d*);', line)
                    if not a_e2_act:
                        entry['A_E2_act'] = '0'
                    else:
                        entry['A_E2_act'] = re.search('act\s(\d*.\d*);', line).group(1)
                    a_e2_dom = re.search('dom\s(\d*.\d*);', line)
                    if not a_e2_dom:
                        entry['A_E2_dom'] = '0'
                    else:
                        entry['A_E2_dom'] = re.search('dom\s(\d*.\d*);', line).group(1)
                    entry['A_E2_note'] = re.search(';\s(\(.*\))', line).group(1)
                elif line.startswith('A-E3:'):
                    a_e3_val = re.search('val\s(\d*.\d*);', line)
                    if not a_e3_val:
                        entry['A_E3_val'] = '0'
                    else:
                        entry['A_E3_val'] = re.search('val\s(\d*.\d*);', line).group(1)
                    a_e3_act = re.search('act\s(\d*.\d*);', line)
                    if not a_e3_act:
                        entry['A_E3_act'] = '0'
                    else:
                        entry['A_E3_act'] = re.search('act\s(\d*.\d*);', line).group(1)
                    a_e3_dom = re.search('dom\s(\d*.\d*);', line)
                    if not a_e3_dom:
                        entry['A_E3_dom'] = '0'
                    else:
                        entry['A_E3_dom'] = re.search('dom\s(\d*.\d*);', line).group(1)
                    entry['A_E3_note'] = re.search(';\s(\(.*\))', line).group(1)
                elif line.startswith('A-E4:'):
                    a_e4_val = re.search('val\s(\d*.\d*);', line)
                    if not a_e4_val:
                        entry['A_E4_val'] = '0'
                    else:
                        entry['A_E4_val'] = re.search('val\s(\d*.\d*);', line).group(1)
                    a_e4_act = re.search('act\s(\d*.\d*);', line)
                    if not a_e4_act:
                        entry['A_E4_act'] = '0'
                    else:
                        entry['A_E4_act'] = re.search('act\s(\d*.\d*);', line).group(1)
                    a_e4_dom = re.search('dom\s(\d*.\d*);', line)
                    if not a_e4_dom:
                        entry['A_E4_dom'] = '0'
                    else:
                        entry['A_E4_dom'] = re.search('dom\s(\d*.\d*);', line).group(1)
                    entry['A_E4_note'] = re.search(';\s(\(.*\))', line).group(1)
                elif line.startswith('A-F1:') or line.startswith('A-M1:'):
                    a_mf1_val = re.search('val\s(\d*.\d*);', line)
                    a_mf1_act = re.search('act\s(\d*.\d*);', line)
                    a_mf1_dom = re.search('dom\s(\d*.\d*);', line)

                    if not a_mf1_val:
                        entry['A_MF1_val'] = '0'
                    else:
                        entry['A_MF1_val'] = re.search('val\s(\d*.\d*);', line).group(1)
                    if not a_mf1_act:
                        entry['A_MF1_act'] = '0'
                    else:
                        entry['A_MF1_act'] = re.search('act\s(\d*.\d*);', line).group(1)
                    if not a_mf1_dom:
                        entry['A_MF1_dom'] = '0'
                    else:
                        entry['A_MF1_dom'] = re.search('dom\s(\d*.\d*);', line).group(1)
                    entry['A_MF1_note'] = re.search(';\s(\(.*\))', line).group(1)
        if bool(entry):
            newlist.append(entry)
        eval_df = pd.DataFrame(newlist)
        return eval_df
