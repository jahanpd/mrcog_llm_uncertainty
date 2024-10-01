import os
from pydantic import BaseModel
from openai import OpenAI
from enum import Enum


CAT_SYSTEM_PROMPT = open('./prompts/system/system_prompt_screen','r').read()
BASE_PROMPT = open('./prompts/user/user_prompt','r').read()
BASE_PROMPT_REASONING = open('./prompts/user/user_prompt_reasoning','r').read()
def create_prompt(question):
    return BASE_PROMPT.format(question=question)

def create_prompt_reasoning(question):
    return BASE_PROMPT_REASONING.format(question=question)

client = OpenAI(
    api_key=os.environ["OPENAI_KEY"],
)

class Question(str, Enum):
    knowledge = 'knowledge'
    reasoning = 'reasoning'

class QuestionType(BaseModel):
    category: Question

class AnswerReasoning(BaseModel):
    reasoning: str
    short_answer: str

class AnswerAlone(BaseModel):
    short_answer: str

def get_openai_response(qn, reasoning=False, temperature=1.0):
    # check whether question is knowledge or reasoning
    chat_completion = client.beta.chat.completions.parse(
        messages=[
            {
                "role": "system",
                "content":CAT_SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": qn,
            }
        ],
        model="gpt-4o-2024-08-06",
        temperature=0.02,
        max_tokens=400,
        response_format=QuestionType
    )

    category: QuestionType = chat_completion.choices[0].message.parsed 
    if category.category == "knowledge":
        # knowledge system prompt
        system = open('./prompts/system/system_prompt_reasoning','r').read()
        structure = AnswerAlone
    else: # reasoning
        # reasoning system prompt
        system = open('./prompts/system/system_prompt_reasoning','r').read()
        structure = AnswerReasoning


    chat_completion = client.beta.chat.completions.parse(
        messages=[
            {
                "role": "system",
                "content":system
            },
            {
                "role": "user",
                "content": qn,
            }
        ],
        model="gpt-4o-2024-08-06",
        temperature=temperature,
        max_tokens=400,
        logprobs=True,
        response_format=structure
    )

    text = chat_completion.choices[0].message.parsed 
    print(category.category, text)
    # only return the logprobs of the answer, not the reasoning
    logprobs_all = chat_completion.choices[0].logprobs.content
    idx = [idx for idx, el in enumerate(logprobs_all) if 'answer' in el.token][0]
    logprobs = [token.logprob for token in logprobs_all[idx+2:-1]]
    print()

    return text, logprobs, category.category