import pandas as pd
import os 
# import question bank
QUESTIONS = pd.read_csv(os.environ["DATASET_PATH"])

def get_answer_text(row):
        return row[f"Option {row['Actual Answer']}"]

def get_answer_list(row):
        options = [x for x in list(QUESTIONS) if 'Option' in x]
        answers = ""
        for option in options:
               answers = answers + "{}. {} \n".format(option[-1], row[option])
        return answers


for index, row in QUESTIONS.iterrows():
    print(row['Question'])
    print(get_answer_list(row))

print(list(QUESTIONS))

