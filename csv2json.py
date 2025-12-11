import json
import pandas as pd
import numpy as np

json_data = []
csv_paths = ['/projects/data/vision-team/shanmukha_sreevatsa/Patram/Imc_new_questions - Finance-2.csv','/projects/data/vision-team/shanmukha_sreevatsa/Patram/Imc_new_questions - Identity and Personal Records.csv', '/projects/data/vision-team/shanmukha_sreevatsa/Patram/Imc_new_questions - medical_questions_with_empty_answers.csv.csv', '/projects/data/vision-team/shanmukha_sreevatsa/Patram/Imc_new_questions - Misc.csv']
counter = 1
for csv_path in csv_paths:
    df = pd.read_csv(csv_path)
    for i, row in df.iterrows():
        # print(i)
        # print(row['question_easy_1'])
        dictt = {}
        dictt['doc_id'] = f'doc_{counter:05d}'
        dictt['img_path'] = row['IMAGE ID / Path']
        dictt['qa_pairs'] = []
        for question in ['question_easy_1','question_easy_2','question_medium_1','question_medium_2','question_hard_1','question_hard_2']:
            qa_dict = {}
            qa_dict['question'] = row[question]
            qa_dict['model1'] = ""
            qa_dict['model2'] = ""
            dictt['qa_pairs'].append(qa_dict)
        json_data.append(dictt)
        counter+=1  

    print(json_data)
with open('/projects/data/vision-team/shanmukha_sreevatsa/Patram/qa_pairs.json','w') as f:
    json.dump(json_data, f)
