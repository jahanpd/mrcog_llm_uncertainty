import os
from openai import OpenAI

BASE_PROMPT = open('./prompts/user/user_prompt','r').read()
BASE_PROMPT2 = open('./prompts/user/user_prompt2','r').read()
def create_prompt(question):
    return BASE_PROMPT.format(question=question)

def create_prompt_mc(question, answers):
    return BASE_PROMPT2.format(question=question, answers=answers)

SYSTEM_PROMPT = open('./prompts/system/system_prompt','r').read()

client = OpenAI(
    api_key=os.environ["OPENAI_KEY"],
)

# client = OpenAI(
#     api_key=os.environ["OPEN_AI_INT"],
# )

def get_openai_response(qn, answers=None):
    if answers:
        prompt = create_prompt_mc(qn, answers)
    else:
        prompt = create_prompt(qn)

    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content":SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": prompt,
            }
        ],
        model="gpt-4-turbo",
        temperature=1,
        max_tokens=40,
        logprobs=True
    )

    text = chat_completion.choices[0].message.content 
    logprobs = [token.logprob for token in chat_completion.choices[0].logprobs.content] 

    return text, logprobs

def get_gpt_entailment(question, text1, text2):
    prompt = f"""We are evaluating answers to the question \"{question}\"\n"""
    prompt += "Here are two possible answers:\n"
    prompt += f"Possible Answer 1: {text1}\nPossible Answer 2: {text2}\n"
    prompt += "Does Possible Answer 1 semantically entail Possible Answer 2? Respond only with entailment, contradiction, or neutral.\n"""
    prompt += "Response:"""

    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        model="gpt-4-turbo",
        temperature=0.02,
        max_tokens=200,
    )

    response = chat_completion.choices[0].message.content 

    return response

