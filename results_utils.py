from pydantic import BaseModel
import os
from enum import Enum
import pickle
import pandas as pd
from typing import Optional
from roc import *
from sklearn.metrics import roc_auc_score
import itertools

class Entail(Enum):
    GPT = 0
    DEBERTA = 1

class Result(BaseModel):
    temp: float
    reasoning: bool
    entailment: Entail
    checker: Entail
    confidence: Optional[list[dict]] = None
    correctness: Optional[list[dict]] = None

class Results:
    def __init__(self, results: list[Result], dataset_path="./Jahan_Subset_v2.csv"):
        assert len(results) > 0

        questions = pd.read_csv(dataset_path)
        self.questions = questions
        self.parts = {
            "part1": self.check_part1,
            "part2" : self.check_part2,
            "full" : lambda x: True,
            }

        for r in results:
            path = f'./data/openai_{self.entail_str(r.entailment)}_temp={r.temp}_reas={r.reasoning}_agg=original_confidence.pkl'
            print(path)
            with open(path, 'rb') as infile:
                res = pickle.load(infile)
                r.confidence = [item for item in res if self.check_table(item["ids"])]

            path = f'./data/openai_{self.entail_str(r.entailment)}_temp={r.temp}_reas={r.reasoning}_checker={self.entail_str(r.checker)}_correctness.pkl'
            print(path)
            with open(path, 'rb') as infile:
                res = pickle.load(infile)
                r.correctness = [item for item in res if self.check_table(item["id"])]

        self.results: list[Result] = results

    def entail_str(self, entail: Entail):
        return "gpt" if entail == Entail.GPT else "deberta"
    
    def check_table(self, id):
        try:
            return bool(self.questions.loc[id[0], :].isnull().Table)
        except:
            return bool(self.questions.loc[id, :].isnull().Table)
    
    def check_part1(self, id):
        try:
            return self.questions.loc[id[0], :].Part == "One"
        except:
            return self.questions.loc[id, :].Part == "One"

    def check_part2(self, id):
        try:
            return self.questions.loc[id[0], :].Part == "Two"
        except:
            return self.questions.loc[id, :].Part == "Two"

    def filter_results(self, 
            temp=[0.2, 1.0, 1.1], 
            reasoning=[True, False],
            entailment=[Entail.GPT, Entail.DEBERTA]):
        filter = [r for r in self.results if r.temp in temp and r.reasoning in reasoning and r.entailment in entailment]
        names = [f'Temp={r.temp}|Reasoning={r.reasoning}|entailed with {self.entail_str(r.entailment)}' for r in filter]
        return filter, names

    def get_results_df(self):
        df = {
            "temp": [],
            "reasoning": [],
            "entailment": [],
            "checker": [],
            "metric": [],
            "correctness": [],
            "part": [],
            "acc": [],
            "auc": [],
        }
        metrics = ['entropy', 'dentropy', 'perplexity']
        part = ['full', 'part1', 'part2']
        correct_definition = ['cluster_correct_strict', 'cluster_correct_relaxed']

        combinations = list(itertools.product(metrics, part, correct_definition))
        for r in self.results:
            for mname, pname, cname in combinations:
                if mname == 'perplexity':
                    cname = 'perplexity_correct'
                p = ([item[mname] for item in r.confidence if self.parts[pname](item["ids"]) ],
                    [item[cname] for item in r.correctness if self.parts[pname](item["id"])])
                try:
                    acc = self.accuracy(p[1])
                    auc = self.auroc(p[0], p[1])
                    df["temp"].append(r.temp)
                    df["reasoning"].append(r.reasoning)
                    df["entailment"].append(r.entailment)
                    df["checker"].append(r.checker)
                    df["metric"].append(mname)
                    df["correctness"].append(cname)
                    df["part"].append(pname)
                    df["acc"].append(acc)
                    df["auc"].append(auc)
                except:
                    print(r)
        return pd.DataFrame(df).drop_duplicates()


    
    def accuracy(self, correct):
        arr = np.array(correct).astype(np.float32)
        return arr.sum() / arr.shape[0]

    def auroc(self, score, correct):
        auc = roc_auc_score(
            np.array(correct).astype(np.float32),
            -1*np.array(score).astype(np.float32)
        )
        return auc

    def plot_aurocs_sem_ent_full_gpt(self):
        """Plot AUROC curves for Semantic Uncertainty subset by LLM entailment"""
        res, names = self.filter_results(
            entailment=[Entail.GPT]
        )

        _, (ax1) = plt.subplots(1, 1, figsize=(8, 8))
        su_rocs_from_results(
            res, 
            ax1,
            names,
            "All SE Across Variables"
            )
            

    def plot_aurocs_metrics_standard(self, title="This is a title"):
        """Plot AUROC curves for all metrics in the base case of temp=1.0 and no reasoning"""
        res, names = self.filter_results(
            entailment=[Entail.GPT],
            temp=[1.0],
            reasoning=[False]
        )

        _, (ax1) = plt.subplots(1, 1, figsize=(8, 8))
        rocs_from_results(
            results_array=res,
            axes=[ax1 for _ in range(len(res))],
            titles=names
        )
        ax1.set_title(title)
