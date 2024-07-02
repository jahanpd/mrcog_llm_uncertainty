import pickle
from entailment import get_set_dict, SemanticSet
import argparse

parser = argparse.ArgumentParser(
                    prog='Semantic Similarity',
                    description='Script 2: Measure semantic similarity and cluster into sets for generated and true answers',
                    epilog='')

parser.add_argument('model', default="openai", type=str,
                    choices=["openai"])     

args = parser.parse_args()

SetSemanticSets = dict[int, SemanticSet]

MODEL = args.model

with open(f'./data/{MODEL}_generations.pkl', 'rb') as infile:
    sequences = pickle.load(infile)

print(sequences[1])

semantic_sets: SetSemanticSets = {}
for s in sequences:
    generated_answers = []
    for g in s['generated_answers']:
        conditioned_response = '''Question: {question}
Answer: {answer}'''.format(
            question=s['question'],
            answer=g
        )
        generated_answers.append(conditioned_response)

    conditioned_truth = '''Question: {question}
Answer: {answer}'''.format(
        question=s['question'],
        answer=s['true_answer']
    )

    semantic_set_ids: SemanticSet = get_set_dict(
            conditioned_truth, generated_answers)
    
    print(semantic_set_ids)
    semantic_sets[s['id']] = semantic_set_ids

with open(f'./data/{MODEL}_semantic_similarity.pkl', 'wb') as outfile:
    pickle.dump(semantic_sets, outfile)
