from pydantic import BaseModel
import os
from enum import Enum
import pickle
import pandas as pd
from typing import Optional
from roc import *
from sklearn.metrics import roc_auc_score

class Entail(Enum):
    GPT = 0
    DEBERTA = 1

class Result(BaseModel):
    temp: float
    reasoning: bool
    entailment: Entail
    data: Optional[dict] = None
    part1: Optional[dict] = None
    part2: Optional[dict] = None

class Results:
    def __init__(self, results: list[Result], dataset_path="./Jahan_Subset_v2.csv"):
        assert len(results) > 0

        for r in results:
            path = f'./data/openai_{self.entail_str(r.entailment)}_temp={r.temp}_reas={r.reasoning}_agg=original_final_results.pkl'
            print(path)
            with open(path, 'rb') as infile:
                res = pickle.load(infile)
                r.data = res
                self.get_parts(r, dataset_path=dataset_path)

        self.results: list[Result] = results


    def entail_str(self, entail: Entail):
        return "gpt" if entail == Entail.GPT else "deberta"
    
    def dlld(self, DL): 
        """ Convert a dict of lists to list of dicts """
        return [dict(zip(DL,t)) for t in zip(*DL.values())]

    def lddl(self, LD): 
        """ Convert a list of dicts to dict of lists """
        return {k: [dic[k] for dic in LD] for k in LD[0]}

    def filter_results(self, 
            temp=[0.2, 1.0, 1.1], 
            reasoning=[True, False],
            entailment=[Entail.GPT, Entail.DEBERTA]):
        filter = [r for r in self.results if r.temp in temp and r.reasoning in reasoning and r.entailment in entailment]
        names = [f'Temp={r.temp}|Reasoning={r.reasoning}|entailed with {self.entail_str(r.entailment)}' for r in filter]
        return filter, names

    def get_parts(self, result: Result, dataset_path):
        questions = pd.read_csv(dataset_path)
        part1 = ['part 1' in x.lower() for x in questions.Source]
        part2 = ['part 2' in x.lower() for x in questions.Source]

        result.part1 = self.lddl([res for inc, res in zip(part1, self.dlld(result.data)) if inc])
        result.part2 = self.lddl([res for inc, res in zip(part2, self.dlld(result.data)) if inc])


    def get_results_df(self):
        df = {
            "temp": [],
            "reasoning": [],
            "entailment": [],
            "metric": [],
            "part": [],
            "acc": [],
            "auc": [],
        }
        metrics = ['entropy', 'dentropy', 'perplexity']
        for r in self.results:
            part = [('full', r.data), ('part1', r.part1), ('part2', r.part2)]
            for pname, p in part:
                for m in metrics:
                    try:
                        acc = self.accuracy(p, m)
                        auc = self.auroc(p, m)
                        df["temp"].append(r.temp)
                        df["reasoning"].append(r.reasoning)
                        df["entailment"].append(r.entailment)
                        df["metric"].append(m)
                        df["part"].append(pname)
                        df["acc"].append(acc)
                        df["auc"].append(auc)
                    except:
                        print(r)
        return pd.DataFrame(df)


    
    def accuracy(self, data: dict, metric: str):
        arr = np.array(data[f'{metric}_correct']).astype(np.float32)
        return arr.sum() / arr.shape[0]

    def auroc(self, data: dict, metric: str):
        auc = roc_auc_score(
            np.array(data[f'{metric}_correct']).astype(np.float32),
            -1*np.array(data[f'{metric}']).astype(np.float32)
        )
        return auc


    def plot_aurocs_sem_ent_full_gpt(self):
        """Plot AUROC curves for Semantic Uncertainty subset by LLM entailment"""
        res, names = self.filter_results(
            entailment=[Entail.GPT]
        )

        _, (ax1) = plt.subplots(1, 1, figsize=(8, 8))
        su_rocs_from_results(
            [r.data for r in res], 
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
            results_array=[r.data for r in res],
            axes=[ax1 for _ in range(len(res))],
            titles=names
        )
        ax1.set_title(title)