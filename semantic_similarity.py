import pickle
from entailment import get_set_dict, SemanticSet
from prompt_utilss import get_gpt_entailment
import argparse

parser = argparse.ArgumentParser(
                    prog='Semantic Similarity',
                    description='Script 2: Measure semantic similarity and cluster into sets for generated and true answers',
                    epilog='')

parser.add_argument('model', default="openai", type=str,
                    choices=["openai"])     

parser.add_argument('entailment', default="gpt", type=str,
                    choices=["gpt", "deberta"])     

args = parser.parse_args()

# semantic set is dict[int, int]
SetSemanticSets = dict[int, SemanticSet]

MODEL = args.model
ENTAILMENT = args.entailment

with open(f'./data/{MODEL}_generations.pkl', 'rb') as infile:
    sequences = pickle.load(infile)

print(sequences[1])

semantic_sets: SetSemanticSets = {}

if args.entailment == "deberta":
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

if arg.entailment == "gpt":

    for s in sequences:
        question = s["question"]
        answers = s["generated_answers"]
        # add the true answer to index 0
        answers.insert(0, s["true_answer"])
        # base semantic set 
        semantic_set_ids: SemanticSet = {}
        for idx, answer in enumerate(s["generated_answers"]):
            semantic_set_ids[idx] = idx

        for i, answer1 in enumerate(answers):
            # this inner loop compared each gen ans with other answers
            for j in range(i + 1, len(answers)):
                entail_response = get_gpt_entailment(
                        question,
                        answer1,
                        answers[j]
                        )
                binary_response = entail_response.lower()[:30]
                if 'entailment' in binary_response:
                    semantic_set_ids[j] = semantic_set_ids[i]
        semantic_sets[s['id']] = semantic_set_ids
    

with open(f'./data/{MODEL}_{ENTAILMENT}_semantic_similarity.pkl', 'wb') as outfile:
    pickle.dump(semantic_sets, outfile)
