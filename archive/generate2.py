import os
import pickle
import numpy as np
import pandas as pd
from prompt_utils import get_openai_response

"""
The generate.py script will create a pkl of a list of generate dicts of type:
    {
        'id': String,
        'generated_answers': [Strings],
        'generated_logprobs': [[float, ...]...],
        'generated_perplexity': [float, ...],
        'true_answer': String,
        'question': String,
    }
""" 

# number of inference runs (answers) to generate
GENERATIONS = 10

# import question bank
QUESTIONS = pd.read_csv(os.environ["DATASET_PATH"])

# TODO implement multiple model logic here, for now just openai
MODEL = "openai"
API_RESPONSE = get_openai_response

print(list(QUESTIONS))

result = []

def get_answer_text(row):
        return row[f"Option {row['Actual Answer']}"]

def get_answer_list(row):
        options = [x for x in list(QUESTIONS) if 'Option' in x]
        answers = ""
        for option in options:
               answers = answers + "{}. {} \n".format(option[-1], row[option])
        return answers

for index, row in QUESTIONS.iterrows():
    question = row['Question']
    answers = get_answer_list(row)
    answer = row['Actual Answer']
    save = dict(
        id=index,
        question=question,
        true_answer=answer
    )
    generated_answers = []
    generated_logprobs = []
    generated_perplexity = []
    print("{}/{}: {}".format(index, QUESTIONS.shape[0], question))
    for i in range(GENERATIONS):
        res_text, res_logprobs = API_RESPONSE(
            question, answers=answers
        )
        print(index, res_text)
        generated_answers.append(res_text)
        generated_logprobs.append(res_logprobs)
        perplexity = np.exp(-np.mean(res_logprobs))
        generated_perplexity.append(perplexity)

    save["generated_answers"] = generated_answers
    save["generated_logprobs"] = generated_logprobs
    save["generated_perplexity"] = generated_perplexity

    result.append(save)

print(result)

with open(f'./data/{MODEL}_generations_mc.pkl', 'wb') as outfile:
    pickle.dump(result, outfile)
