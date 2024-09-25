import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from openai import OpenAI
import os
import logging
from datetime import datetime

logger = logging.getLogger("")
logging.basicConfig(filename= f'./logs/entailment-{datetime.now().isoformat()}.log', encoding='utf-8', level=logging.INFO)


tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v2-xlarge-mnli")
model = AutoModelForSequenceClassification.from_pretrained("microsoft/deberta-v2-xlarge-mnli").cuda()

"""
The semantic_similarity.py script will use the above output and create a pkl of a dict:
    {
        '{id}': semantic_set_ids
    }
"""
def check_deberta_bidirectional(phrase1, phrase2) -> int:
    inputs = tokenizer(phrase1, phrase2, return_tensors="pt").to('cuda')
    # The model checks if text1 -> text2, i.e. if text2 follows from text1.
    # check_implication('The weather is good', 'The weather is good and I like you') --> 1
    # check_implication('The weather is good and I like you', 'The weather is good') --> 2
    outputs = model(**inputs)
    logits = outputs.logits
    # Deberta-mnli returns `neutral` and `entailment` classes at indices 1 and 2.
    largest_index = torch.argmax(F.softmax(logits, dim=1))  # pylint: disable=no-member
    prediction = largest_index.cpu().item()
    return prediction


def deberta_prompt(question, answer):
    return f'''Question: {question}
Answer: {answer}'''

def get_deberta_entailment(question, phrase1, phrase2, strict=False) -> bool:
    forward = check_deberta_bidirectional(
        deberta_prompt(question, phrase1), deberta_prompt(question, phrase2))
    reverse = check_deberta_bidirectional(
        deberta_prompt(question, phrase2), deberta_prompt(question, phrase1))

    if strict:
            semantically_equivalent = (forward == 2) and (reverse == 2)
    else:
        implications = [forward, reverse]
        # Check if none of the implications are 0 (contradiction) and not both of them are neutral.
        semantically_equivalent = (0 not in implications) and ([1, 1] != implications)

    return semantically_equivalent

SemanticSet = dict[int, int]

client = OpenAI(
    api_key=os.environ["OPENAI_KEY"],
)


def gpt_entailment_prompt(question, text1, text2):
    prompt = f"""We are evaluating answers to the question \"{question}\"\n"""
    prompt += "Here are two possible answers:\n"
    prompt += f"Possible Answer 1: {text1}\nPossible Answer 2: {text2}\n"
    prompt += "Does Possible Answer 1 semantically entail Possible Answer 2? Respond only with entailment, contradiction, or neutral.\n"""
    prompt += "Response:"""
    return prompt

def get_llm_entailement_response(prompt):
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        model="gpt-4o-mini",
        temperature=0.02,
        max_tokens=200,
    )

    binary_response = chat_completion.choices[0].message.content.lower()

    if 'entailment' in binary_response:
        return 2
    elif 'neutral' in binary_response:
        return 1
    elif 'contradiction' in binary_response:
        return 0
    else:
        logging.warning('MANUAL NEUTRAL!')
        logging.warning(prompt)
        logging.warning(binary_response)
        return 1

def get_gpt_entailment(question, text1, text2, strict=False) -> bool:
    forward = get_llm_entailement_response(gpt_entailment_prompt(question, text1, text2))
    reverse = get_llm_entailement_response(gpt_entailment_prompt(question, text2, text1))

    if strict:
            semantically_equivalent = (forward == 2) and (reverse == 2)
    else:
        implications = [forward, reverse]
        # Check if none of the implications are 0 (contradiction) and not both of them are neutral.
        semantically_equivalent = (0 not in implications) and ([1, 1] != implications)

    return semantically_equivalent




# test = check_bidirectional_entailment('capital of france is paris', 
                                    #   'the main city of governance for the french people is paris')
# print(test)
