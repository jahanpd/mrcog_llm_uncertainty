import json
import pickle
from pydantic import BaseModel
import random
from AesEverywhere import aes256
import os
import pandas as pd

class Item(BaseModel):
    id: int
    question: str
    generated_answers: list[str]
    clusters: list[int]
    perplexity: list[float]
    true_answer: str

path = "data/openai_temp=1.0_reasoning=False_generations.pkl"
path2 = "data/openai_gpt_reas=False_temp=1.0_semantic_similarity.pkl"
data = pickle.load(open(path, "rb"))
clusters = pickle.load(open(path2, "rb"))

questions = pd.read_csv("~/Jahan_Subset_v2.csv")

output: list[Item] = []
filter = {}
for i in range(1, 11):
    filter[i] = []

print(data[0])
print(clusters[data[0]["id"]])

for d in data:
    try:
        c = clusters[d["id"]]
        c = [c[key] for key in range(0, 10)]
        numc = len(set(c))
        item = Item(
            id = d["id"][0],
            question = d["question"][0],
            generated_answers=d["generated_answers"],
            clusters=c,
            perplexity=[perp if perp < 10000 else 10000 for perp in d["generated_perplexity"]],
            true_answer=d["true_answer"]
        )

        filter[numc].append(dict(item))
    except Exception as e:
        print("exception", e)

random.seed(42)

def get_subset(items):
    if len(items) > 20:
        subsample = random.sample(subset, 20)
        return subsample
    else:
        return items 

for i in range(1, 11):
    subset = filter[i]
    count_one = 0
    count_two = 0
    part1 = [item for item in subset if questions.loc[item["id"], :].Part == 'One' and questions.loc[item["id"], :].isnull().Table]
    part2 = [item for item in subset if questions.loc[item["id"], :].Part == 'Two' and questions.loc[item["id"], :].isnull().Table]

    if len(part1) > len(part2):
        part1 = random.sample(part1, len(part2))
    if len(part2) > len(part1):
        part2 = random.sample(part2, len(part1))
    subpart1 = get_subset(part1)
    subpart2 = get_subset(part2)
    output = output + subpart1 + subpart2
    print(i, len(filter[i]), len(subpart1), len(subpart2))

random.shuffle(output)

print(len(output))
jsonstring = json.dumps(output, indent=4)
print(jsonstring[:50])

encrypted = aes256.encrypt(jsonstring, os.environ["SECRET_KEY"])

print(encrypted[:50])

with open('data/encrypted.json', 'w') as f:
    f.write(encrypted.decode('utf-8'))