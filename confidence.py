import pickle
import torch
import numpy as np

"""
We use output from previous two scripts to compute confidence values.
"""

# TODO implement multiple model logic here, for now just openai
MODEL = "openai"
ENTAILMENT = "gpt"

def logsumexp_by_id(
        semantic_ids: dict[int, int], 
        log_likelihoods: list[float], 
        agg='sum_normalized'
        ):
    """Sum probabilities with the same semantic id.

    Log-Sum-Exp because input and output probabilities in log space.
    """
    # https://github.com/jlko/semantic_uncertainty/blob/a8d9aa8cecd5f3bec09b19ae38ab13552e0846f4/semantic_uncertainty/uncertainty/uncertainty_measures/semantic_entropy.py#L208
    # note that semantic_set_id 0 is the true answer
    assert len(log_likelihoods) == (len(semantic_set_ids) - 1) # semantic sets include true answer
    log_likelihood_per_semantic_id = []
    # need to filter out the true answer as no logliks for it, has poisiton/id 0
    unique_ids = sorted(list(set([val for key, val in semantic_set_ids.items() if key != 0])))
    for uid in unique_ids:
        # Find positions in `semantic_ids` which belong to the active `uid`.
        # position 0 in dict is the true answer, and not in the log prop array
        id_indices = [pos for pos, x in semantic_ids.items() if x == uid]
        # Gather mean log likelihoods at these indices.
        id_log_likelihoods = [log_likelihoods[i] for i in id_indices] # list[float]
        if agg == 'sum_normalized':
            # log_lik_norm = id_log_likelihoods - np.prod(log_likelihoods)
            log_lik_norm = np.array(id_log_likelihoods) - np.log(np.sum(np.exp(log_likelihoods)))
            logsumexp_value = np.log(np.sum(np.exp(log_lik_norm)))
        else:
            raise ValueError
        log_likelihood_per_semantic_id.append(logsumexp_value)

    return log_likelihood_per_semantic_id, unique_ids


def categorical_empirical_loglik(
        semantic_ids: dict[int, int], 
        log_likelihoods: list[float], 
        ):
    """ Calculate counts for each set.

    Return logprob of each set"""
    unique, counts = np.unique(list(semantic_ids.values()), return_counts=True)
    logprobs = []
    return [np.log(c/np.sum(counts)) for c in counts], unique
    

def predictive_entropy_rao(log_probs):
    entropy = -np.sum(np.exp(log_probs) * log_probs)
    return entropy


def compute_semantic_uncertainty(
        log_likelihoods: list[list[float]], 
        semantic_set_ids: dict[int, int],
        agg='sum_normalized'
        ) -> tuple[float, bool, int]:

    # Length normalization of generation probabilities.
    log_liks_agg = [np.mean(log_lik) for log_lik in log_likelihoods]

    # returns list[float] of likelihoods where index corresponds
    # to unique semantic id, and list[int] of semantic ids
    # index 0 is the true answer semantic set if index 0 of unique_ids is 0
    log_likelihood_per_semantic_id, unique_ids_cont = logsumexp_by_id(
            semantic_set_ids, 
            log_likelihoods, 
            agg='sum_normalized')
    log_likelihood_per_semantic_id_discrete, unique_ids_disc = logsumexp_by_id(
            semantic_set_ids, 
            log_likelihoods)
    pe_continuous = predictive_entropy_rao(log_likelihood_per_semantic_id)
    pe_discrete = predictive_entropy_rao(log_likelihood_per_semantic_id_discrete)

    correct_continuous = 0 == unique_ids_cont[np.argmax(log_likelihood_per_semantic_id)]
    correct_discrete = 0 == unique_ids_disc[np.argmax(log_likelihood_per_semantic_id_discrete)]
    assert correct_continuous == correct_discrete

    return (pe_continuous, pe_discrete, correct_continuous, len(unique_ids_cont), len(unique_ids_disc))

def compute_perplexity(perplexity, semantic_set_ids) -> tuple[float, bool]:
    semantic_set = torch.tensor([value for key, value in semantic_set_ids.items() if key != 0])
    perplexity = torch.tensor(perplexity).clamp(0.0, 1000000.0)
    lowest_perp_set = semantic_set[perplexity.argmin()]
    correct_answer = lowest_perp_set == 0
    perplexity_norm = (perplexity - 1.0) / ((perplexity.max() - 1.0) + 1e-8)

    return perplexity.min(), correct_answer

with open(f'./data/{MODEL}_generations.pkl', 'rb') as infile:
    sequences = pickle.load(infile)

with open(f'./data/{MODEL}_{ENTAILMENT}_semantic_similarity.pkl', 'rb') as infile:
    semantic_set_ids = pickle.load(infile)

final_results = dict(
        entropy = [],
        entropy_discrete = [],
        entropy_correct = [],
        sets = [],
        sets_discrete = [],
        perplexity = [],
        perplexity_correct = []
        )
for sequence in sequences:
    log_likelihoods = sequence['generated_logprobs']
    semantic_set = semantic_set_ids[sequence['id']] # dict[id, id]
    # for each question, we calculate
    # 1. semantic uncertainty (entropy)
    # 2. if the highest count set contains the correct answer
    # 3. the number of semantic sets in the generated answers
    # 4. the lowest perplexity 
    # 5. if the lowest perplexity answer is in the same set as the correct answer - aka correct
    entropy, entropy_discrete, entropy_correct, sets, sets_disc = compute_semantic_uncertainty(
            log_likelihoods, 
            semantic_set)
    perplexity, perplexity_correct = compute_perplexity(sequence['generated_perplexity'], semantic_set) 
    final_results["entropy"].append(entropy)
    final_results["entropy_discrete"].append(entropy)
    final_results["entropy_correct"].append(entropy_correct)
    final_results["sets"].append(sets)
    final_results["sets_discrete"].append(sets)
    final_results["perplexity"].append(perplexity)
    final_results["perplexity_correct"].append(perplexity_correct)
    
with open(f'./data/{MODEL}_final_results.pkl', 'wb') as outfile:
    pickle.dump(final_results, outfile)
