import pickle
import torch
import numpy as np

"""
We use output from previous two scripts to compute confidence values.
"""

# TODO implement multiple model logic here, for now just openai
MODEL = "openai"

def compute_semantic_uncertainty(log_likelihoods, semantic_set_ids) -> tuple[float, bool, int]:
    avg_likelihoods = torch.tensor([
        torch.logsumexp(torch.tensor(ls), dim=0) - torch.log(torch.tensor(len(ls)))
        for ls in log_likelihoods
        ]) # tensor of shape (number of generations,)
    semantic_set = torch.tensor([value for key, value in semantic_set_ids.items() if key != 0])
    semantic_set_w = torch.tensor([value for key, value in semantic_set_ids.items()])
    aggregated_likelihoods = []
    C = torch.unique(semantic_set_w).shape[0]
    sets, counts = torch.unique(semantic_set_w, return_counts=True)
    correct_answer = sets[counts.argmax()] == 0
    for set_id in torch.unique(semantic_set):
        ag = torch.logsumexp(avg_likelihoods[semantic_set == set_id], dim=0)
        aggregated_likelihoods.append(
                ag
                )
    print(C)
    aggregated_likelihoods = torch.tensor(aggregated_likelihoods) - C
    entropy = -torch.sum(aggregated_likelihoods) / aggregated_likelihoods.shape[0]
    return (entropy, correct_answer, sets.shape[0])

def compute_perplexity(perplexity, semantic_set_ids) -> tuple[float, bool]:
    semantic_set = torch.tensor([value for key, value in semantic_set_ids.items() if key != 0])
    perplexity = torch.tensor(perplexity).clamp(0.0, 1000000.0)
    lowest_perp_set = semantic_set[perplexity.argmin()]
    correct_answer = lowest_perp_set == 0
    perplexity_norm = (perplexity - 1.0) / ((perplexity.max() - 1.0) + 1e-8)

    return perplexity.min(), correct_answer

with open(f'./data/{MODEL}_generations.pkl', 'rb') as infile:
    sequences = pickle.load(infile)

with open(f'./data/{MODEL}_semantic_similarity.pkl', 'rb') as infile:
    semantic_set_ids = pickle.load(infile)

final_results = dict(
        entropy = [],
        entropy_correct = [],
        sets = [],
        perplexity = [],
        perplexity_correct = []
        )
for sequence in sequences:
    log_likelihoods = sequence['generated_logprobs']
    semantic_set = semantic_set_ids[sequence['id']]
    # for each question, we calculate
    # 1. semantic uncertainty (entropy)
    # 2. if the highest count set contains the correct answer
    # 3. the number of semantic sets in the generated answers
    # 4. the lowest perplexity 
    # 5. if the lowest perplexity answer is in the same set as the correct answer - aka correct
    entropy, entropy_correct, sets = compute_semantic_uncertainty(log_likelihoods, semantic_set)
    perplexity, perplexity_var, perplexity_norm, perplexity_correct = compute_perplexity(sequence['generated_perplexity'], semantic_set) 
    final_results["entropy"].append(entropy)
    final_results["entropy_correct"].append(entropy_correct)
    final_results["sets"].append(sets)
    final_results["perplexity"].append(perplexity)
    final_results["perplexity_correct"].append(perplexity_correct)
    
with open(f'./data/{MODEL}_final_results.pkl', 'wb') as outfile:
    pickle.dump(final_results, outfile)
