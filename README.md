# Understanding the codebase
A description of the key files and data types.

## generate.py
This file performs question inference.
The output of these inference runs are stored in a pickle with a data structure:
```
class Output(TypedDict):
    id: int
    generated_answers: list[str]
    generated_logprobs: list[list[float]]
    generated_perplexity: list[float]
    true_answer: str
    question: str

SaveData = list[Output]
```
### prompt_utils.py
Convenience functions for generating answers

## semantic_similarity.py
Compute the semantic sets among the generated and true answers.
The ouput of the script is stored in a data structure:
```
SemanticSet = dict[int, int]
SetSemanticSets = dict[int, SemanticSet] # where the key is the id
```

### entailment.py
Convenience functions for computing entailment.

## confidence.py
Script for computing confidence metrics based on the defined semantic sets.
Current metrics include:
- Semantic uncertainty
- Perplexity


