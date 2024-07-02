import os
import pickle
import numpy as np
import pandas as pd
from prompt_utils import get_openai_response
import argparse
from typing import TypedDict

parser = argparse.ArgumentParser(
                    prog='generate',
                    description='Script 1: Generate M answers to questions from the MRCOG question bank',
                    epilog='')

parser.add_argument('M', default=10, type=int)     
parser.add_argument('model', default="openai", type=str,
                    choices=["openai"])     

args = parser.parse_args()

class Output(TypedDict):
    id: int
    generated_answers: list[str]
    generated_logprobs: list[list[float]]
    generated_perplexity: list[float]
    true_answer: str
    question: str

SaveData = list[Output]

# number of inference runs (answers) to generate
GENERATIONS = args.M

# import question bank
QUESTIONS = pd.read_csv(os.environ["DATASET_PATH"])

# TODO implement multiple model logic here, for now just openai
MODEL = args.model
API_RESPONSE = get_openai_response

print(list(QUESTIONS))

result: SaveData = []

def get_answer_text(row):
        return row[f"Option {row['Actual Answer']}"]

for index, row in QUESTIONS.iterrows():
    question = row['Question']
    answer = get_answer_text(row)
    id=index,
    question=question,
    true_answer=answer
    generated_answers = []
    generated_logprobs = []
    generated_perplexity = []
    print("{}/{}: {}".format(index, QUESTIONS.shape[0], question))
    for i in range(GENERATIONS):
        res_text, res_logprobs = API_RESPONSE(
            question
        )
        print(index, res_text)
        generated_answers.append(res_text)
        generated_logprobs.append(res_logprobs)
        perplexity = np.exp(-np.mean(res_logprobs))
        generated_perplexity.append(perplexity)

    output: Output = dict(
        id=id,
        question=question,
        true_answer=true_answer,
        generated_answers=generated_answers,
        generated_logprobs=generated_logprobs,
        generated_perplexity=generated_perplexity
    )
    result.append(output)

with open(f'./data/{MODEL}_generations.pkl', 'wb') as outfile:
    pickle.dump(result, outfile)
