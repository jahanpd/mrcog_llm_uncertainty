import pickle
import torch
import numpy as np
import argparse

"""
We use output from previous two scripts to compute confidence values.
"""

parser = argparse.ArgumentParser(
                    prog='Compute confidence',
                    description='Script 3: Measure semantic uncertainty (entropy), discrete SE, and perplexity',
                    epilog='')

parser.add_argument('--model', default="openai", type=str,
                    choices=["openai"])     

parser.add_argument('--temp', default=1.0, type=float)     

parser.add_argument('--reasoning', action='store_true')     

parser.add_argument('--entailment', default="gpt", type=str,
                    choices=["gpt", "deberta"])     

parser.add_argument('--agg', default="original", type=str,
                    choices=["sum_normalized", "original"])     

args = parser.parse_args()

MODEL = args.model
ENTAILMENT = args.entailment

# the original papers metric
# https://github.com/lorenzkuhn/semantic_uncertainty/blob/20e0ee1388e776e48c1ee285e00462aabc6cf35a/code/compute_confidence_measure.py#L123
def compute_semantic_uncertainty_original(
        log_likelihoods: list[list[float]], 
        semantic_set_ids: dict[int,int]
        ) -> tuple[float, bool, int]:
    
    # length normalisation
    # [logsumexp(loglik - log(length)) ...]
    avg_likelihoods = torch.tensor([
        torch.logsumexp(torch.tensor(ls), dim=0) - torch.log(torch.tensor(len(ls)))
        for ls in log_likelihoods
        ]) # tensor of shape (number of generations,)

    # exludes true value
    semantic_set = torch.tensor([value for key, value in semantic_set_ids.items() if key != 0])
    # with true value
    semantic_set_w = torch.tensor([value for key, value in semantic_set_ids.items()])
    C = torch.unique(semantic_set_w).shape[0] # number of sets
    sets, counts = torch.unique(semantic_set_w, return_counts=True) # counts for each set
    correct_answer = sets[counts.argmax()] == 0
    aggregated_likelihoods = []
    for set_id in torch.unique(semantic_set):
        # logsumexp for each unique set
        ag = torch.logsumexp(avg_likelihoods[semantic_set == set_id], dim=0)
        aggregated_likelihoods.append(
                ag
                )
    print(C)
    aggregated_likelihoods = torch.tensor(aggregated_likelihoods) - 5
    # assert C == aggregated_likelihoods.shape[0], f"{C} {aggregated_likelihoods.shape[0]}"
    entropy = -torch.sum(aggregated_likelihoods) / aggregated_likelihoods.shape[0]
    return (entropy, correct_answer, sets.shape[0])


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
    assert len(log_likelihoods) == (len(semantic_ids) - 1), f"{len(log_likelihoods)} vs {len(semantic_ids)}" # semantic sets include true answer
    log_likelihood_per_semantic_id = []
    # need to filter out the true answer as no logliks for it, has poisiton/id 0
    unique_ids = sorted(list(set([val for key, val in semantic_ids.items() if key != 0])))
    for uid in unique_ids:
        # Find positions in `semantic_ids` which belong to the active `uid`.
        # position 0 in dict is the true answer, and not in the log prop array
        id_indices = [pos - 1 for pos, x in semantic_ids.items() if x == uid]
        # Gather mean log likelihoods at these indices.
        id_log_likelihoods = [log_likelihoods[i] for i in id_indices if i > -1] # list[float]
        if agg == 'sum_normalized':
            # log_lik_norm = id_log_likelihoods - np.prod(log_likelihoods)
            log_lik_norm = np.array(id_log_likelihoods) - np.log(np.sum(np.exp(log_likelihoods)))
            logsumexp_value = np.log(np.sum(np.exp(log_lik_norm)))
        elif agg == 'original':
            logsumexp_value = np.log(np.sum(np.exp(id_log_likelihoods)))
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
    unique, counts = np.unique([val for key, val in semantic_ids.items() if key != 0], return_counts=True)
    return [np.log(c/np.sum(counts)) for c in counts], unique
    

def predictive_entropy_rao(log_probs):
    entropy = -np.sum(np.exp(log_probs) * log_probs)
    return entropy


def compute_semantic_uncertainty(
        log_likelihoods: list[list[float]], 
        semantic_ids: dict[int, int],
        agg='sum_normalized'
        ) -> tuple[float, bool, int]:

    # Length normalization of generation probabilities.
    # log_liks_agg = [np.mean(log_lik) for log_lik in log_likelihoods]
    log_liks_agg = [np.log(np.sum(np.exp(li - np.log(len(li))))) for li in log_likelihoods]

    # returns list[float] of likelihoods where index corresponds
    # to unique semantic id, and list[int] of semantic ids
    # index 0 is the true answer semantic set if index 0 of unique_ids is 0
    log_likelihood_per_semantic_id, unique_ids_cont = logsumexp_by_id(
            semantic_ids, 
            log_liks_agg, 
            agg=agg)
    log_likelihood_per_semantic_id_discrete, unique_ids_disc = categorical_empirical_loglik(
            semantic_ids, 
            log_likelihoods)
    pe_continuous = predictive_entropy_rao(log_likelihood_per_semantic_id)
    pe_discrete = predictive_entropy_rao(log_likelihood_per_semantic_id_discrete)

    correct_continuous = 0 == unique_ids_cont[np.argmax(log_likelihood_per_semantic_id)]
    # check for ties
    if correct_continuous:
        if np.sum([log_likelihood_per_semantic_id[0] == x for x in log_likelihood_per_semantic_id]) > 1:
            print("continuous tie detected")
            correct_continuous = False
    correct_discrete = 0 == unique_ids_disc[np.argmax(log_likelihood_per_semantic_id_discrete)]
    # check for ties
    if correct_discrete:
        if np.sum([log_likelihood_per_semantic_id_discrete[0] == x for x in log_likelihood_per_semantic_id_discrete]) > 1:
            print("discrete tie detected")
            correct_discrete = False

    # assert correct_continuous == correct_discrete, f"cont: {correct_continuous}, dis: {correct_discrete} {log_likelihood_per_semantic_id_discrete} {log_likelihood_per_semantic_id}"

    return (pe_continuous, pe_discrete, correct_continuous, correct_discrete, len(unique_ids_cont), len(unique_ids_disc))

def compute_perplexity(perplexity, semantic_ids) -> tuple[float, bool]:
    semantic_set = torch.tensor([value for key, value in semantic_ids.items() if key != 0])
    perplexity = torch.tensor(perplexity).clamp(0.0, 1000000.0)
    lowest_perp_set = semantic_set[perplexity.argmin()]
    correct_answer = lowest_perp_set == 0

    return perplexity.min(), correct_answer

with open(f'./data/{MODEL}_temp={args.temp}_reasoning={args.reasoning}_generations.pkl', 'rb') as infile:
    sequences = pickle.load(infile)

with open(f'./data/{MODEL}_{ENTAILMENT}_reas={args.reasoning}_temp={args.temp}_semantic_similarity.pkl', 'rb') as infile:
    semantic_set_ids = pickle.load(infile)

final_results = dict(
        ids = [],
        entropy = [],
        dentropy = [],
        og_entropy = [],
        entropy_correct = [],
        dentropy_correct = [],
        og_entropy_correct = [],
        sets = [],
        dsets = [],
        og_sets = [],
        perplexity = [],
        perplexity_correct = []
        )
for sequence in sequences:
    log_likelihoods = sequence['generated_logprobs']
    semantic_set = semantic_set_ids[sequence['id']] # dict[id, id]
    print(sequence['id'])
    # for each question, we calculate
    # 1. semantic uncertainty (entropy)
    # 2. if the highest count set contains the correct answer
    # 3. the number of semantic sets in the generated answers
    # 4. the lowest perplexity 
    # 5. if the lowest perplexity answer is in the same set as the correct answer - aka correct

    og_entropy, og_ent_correct, og_sets = compute_semantic_uncertainty_original(
        log_likelihoods,
        semantic_set,
    )
    entropy, entropy_discrete, entropy_correct, dentropy_correct, sets, sets_disc = compute_semantic_uncertainty(
            log_likelihoods, 
            semantic_set,
            agg=args.agg
            )
    perplexity, perplexity_correct = compute_perplexity(sequence['generated_perplexity'], semantic_set) 
    final_results["ids"].append(sequence['id'])
    final_results["entropy"].append(entropy)
    final_results["og_entropy"].append(og_entropy)
    final_results["dentropy"].append(entropy_discrete)
    final_results["entropy_correct"].append(entropy_correct)
    final_results["og_entropy_correct"].append(og_ent_correct)
    final_results["dentropy_correct"].append(dentropy_correct)
    final_results["sets"].append(sets)
    final_results["dsets"].append(sets)
    final_results["og_sets"].append(sets)
    final_results["perplexity"].append(perplexity)
    final_results["perplexity_correct"].append(perplexity_correct)
    print(f"e:{entropy:.2f} ed:{entropy_discrete:.2f} s:{sets} sd:{sets_disc}")
    
with open(f'./data/{MODEL}_{ENTAILMENT}_temp={args.temp}_reas={args.reasoning}_agg={args.agg}_final_results.pkl', 'wb') as outfile:
    pickle.dump(final_results, outfile)
