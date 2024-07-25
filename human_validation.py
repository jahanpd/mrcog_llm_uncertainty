import pickle

MODEL = 'openai'

with open(f'./data/{MODEL}_generations.pkl', 'rb') as infile:
    sequences = pickle.load(infile)

with open(f'./data/{MODEL}_generations.pkl', 'rb') as infile:
    sequences = pickle.load(infile)

with open(f'./data/{MODEL}_gpt_semantic_similarity.pkl', 'rb') as infile:
    semantic_set_ids = pickle.load(infile)

csv = open('check_answers.csv', 'w')
csv.write(f'id|question|answer_true|{"|".join([f"answer_{i}" for i in range(10)])}\n')
for s in sequences:
    print(semantic_set_ids[s['id']])
    idx = s['id'][0]
    csv.write(f"{idx}|{s['question']}|{s['true_answer']}|{'|'.join(s['generated_answers'])}\n")
    print([v for k, v in semantic_set_ids[s['id']].items() if k != 0])
    csv.write(f"{idx}||0|{'|'.join([str(v) for k, v in semantic_set_ids[s['id']].items() if k != 0])}\n")

csv.close()