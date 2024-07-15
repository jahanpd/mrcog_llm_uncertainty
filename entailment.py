import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from typing import TypedDict

tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-large-mnli")
model = AutoModelForSequenceClassification.from_pretrained("microsoft/deberta-large-mnli").cuda()

"""
The semantic_similarity.py script will use the above output and create a pkl of a dict:
    {
        '{id}': semantic_set_ids
    }
"""

def check_bidirectional_entailment_deberta(phrase1, phrase2) -> bool:
    input = phrase1 + ' [SEP] ' + phrase2
    encoded_input = tokenizer.encode(input, padding=True)
    prediction = model(torch.tensor(torch.tensor([encoded_input]), device='cuda'))['logits']
    predicted_label = torch.argmax(prediction, dim=1)

    reversed_input = phrase2 + ' [SEP] ' + phrase1
    encoded_reversed_input = tokenizer.encode(reversed_input, padding=True)
    reverse_prediction = model(torch.tensor(torch.tensor([encoded_reversed_input]), device='cuda'))['logits']
    reverse_predicted_label = torch.argmax(reverse_prediction, dim=1)

    if 0 in predicted_label or 0 in reverse_predicted_label:
        return False
    else:
        return True

SemanticSet = dict[int, int]

def get_set_dict(truth, gen_list) -> SemanticSet:
    # for each generated answer compare it to each other answer
    # and also the ground truth answer
    gen_list.insert(0, truth)
    semantic_set_ids = {}
    for idx, answer in enumerate(gen_list):
        semantic_set_ids[idx] = idx

    for i, phrase1 in enumerate(gen_list):
        # this inner loop compared each gen ans with other answers
        for j in range(i + 1, len(gen_list)):
            gen_entailment = check_bidirectional_entailment(
                    phrase1,
                    gen_list[j])
            if gen_entailment:
                semantic_set_ids[j] = semantic_set_ids[i]

    return semantic_set_ids






test = check_bidirectional_entailment('capital of france is paris', 
                                      'the main city of governance for the french people is paris')
print(test)
