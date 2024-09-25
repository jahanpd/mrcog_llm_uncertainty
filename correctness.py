import pickle
import argparse
from entailment import get_gpt_entailment
import numpy as np
from joblib import Parallel, delayed

parser = argparse.ArgumentParser(
                    prog='Compute correctness',
                    description='Post clustering entailment to decide response correctness',
                    epilog='')

parser.add_argument('--model', default="openai", type=str,
                    choices=["openai"])     

parser.add_argument('--temp', default=1.0, type=float)     

parser.add_argument('--reasoning', action='store_true')     

parser.add_argument('--entailment', default="gpt", type=str,
                    choices=["gpt", "deberta"])     

args = parser.parse_args()

with open(f'./data/{args.model}_temp={args.temp}_reasoning={args.reasoning}_generations.pkl', 'rb') as infile:
    sequences = pickle.load(infile)

with open(f'./data/{args.model}_{args.entailment}_reas={args.reasoning}_temp={args.temp}_semantic_similarity.pkl', 'rb') as infile:
    semantic_set_ids = pickle.load(infile)

collected_correctness = []

def process_sequence(s):
    print(s["id"])
    idx = s["id"]
    question = s["question"]
    answers = s["generated_answers"]
    perplexity = s["generated_perplexity"]
    true_answer = s["true_answer"]
    semantic_set = semantic_set_ids[idx] # dict[id, id]

    # find the largest semantic set
    answer_labels = [semantic_set[i] for i in range(len(answers))]
    labels, label_counts = np.unique(answer_labels, return_counts=True)
    # sort from largest to smallest cluster
    labels = labels[np.argsort(-label_counts)]
    label_counts = label_counts[np.argsort(-label_counts)]

    cluster_correct_strict = False
    cluster_correct_relaxed = False

    if len(labels) == 1:
        # check true answer against all answers
        entailed = [get_gpt_entailment(question, answer, true_answer) for answer in answers]
        cluster_correct_strict = np.all(entailed)
        cluster_correct_relaxed = np.any(entailed)
    else:
        # check for a tie in largest label clusters
        if label_counts[0] == label_counts[1]:
            # if there's a tie then answer is wrong/uncertain
            pass
        else:
            answer_subset = [a for a, l in zip(answers, answer_labels) if l == labels[0]]
            entailed = [get_gpt_entailment(question, answer, true_answer) for answer in answer_subset]
            cluster_correct_strict = np.all(entailed)
            cluster_correct_relaxed = np.any(entailed)

    # check lowest perplexity answer entailment
    lowest_perp_answer = answers[np.argmin(perplexity)]
    perplexity_correct = get_gpt_entailment(question, lowest_perp_answer, true_answer)
    return dict(
        id=idx,
        cluster_correct_strict=cluster_correct_strict,
        cluster_correct_relaxed=cluster_correct_relaxed,
        perplexity_correct=perplexity_correct,
        question=question,
        answers=answers,
        true_answer=true_answer,
        answer_labels=answer_labels,
        perplexity=perplexity
    )

results = Parallel(n_jobs=10, prefer='threads')(delayed(process_sequence)(s) for s in sequences)
for r in results:
    collected_correctness.append(r)


with open(f'./data/{args.model}_{args.entailment}_temp={args.temp}_reas={args.reasoning}_correctness.pkl', 'wb') as outfile:
    pickle.dump(collected_correctness, outfile)

