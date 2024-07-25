import numpy as np
import matplotlib.pyplot as plt
from roc_utils import *
from tabulate import tabulate

def rocs_from_results(results_array, axes, titles):
    for results, ax, title in zip(results_array, axes, titles):
        negent = -1*np.array(results["entropy"])
        ent_correct = np.array(results["entropy_correct"]).astype(np.float32)

        negdent = -1*np.array(results["dentropy"])
        dent_correct = np.array(results["dentropy_correct"]).astype(np.float32)

        neg_og_ent = -1*np.array(results["og_entropy"])
        og_ent_correct = np.array(results["og_entropy_correct"]).astype(np.float32)

        negperp = -1*np.array(results["perplexity"])
        perp_correct = np.array(results["perplexity_correct"]).astype(np.float32)


        roc1 = compute_roc(X=negent, y=ent_correct, pos_label=1.0)
        roc2 = compute_roc(X=negperp, y=perp_correct, pos_label=1.0)
        roc3 = compute_roc(X=negdent, y=dent_correct, pos_label=1.0)
        roc4 = compute_roc(X=neg_og_ent, y=og_ent_correct, pos_label=1.0)

        plot_roc(roc1, label="Semantic Uncertainty", color="red", ax=ax)
        plot_roc(roc3, label="Discrete Semantic Uncertainty", color="orange", ax=ax)
        plot_roc(roc4, label="Original Semantic Uncertainty", color="blue", ax=ax)
        plot_roc(roc2, label="Perplexity", color="green", ax=ax)

        ax.set_title(title)


def table_from_results(results_array, headers):
    table = [
        ["SE"],
        ["SDE" ],
        ["OSE"],
        ["Perp"]
    ]
    for results in results_array:
        ent_correct = np.array(results["entropy_correct"]).astype(np.float32)
        table[0].append(ent_correct.sum() / ent_correct.shape[0])

        dent_correct = np.array(results["dentropy_correct"]).astype(np.float32)
        table[1].append(dent_correct.sum() / dent_correct.shape[0])

        og_ent_correct = np.array(results["og_entropy_correct"]).astype(np.float32)
        table[2].append(og_ent_correct.sum() / og_ent_correct.shape[0])

        perp_correct = np.array(results["perplexity_correct"]).astype(np.float32)
        table[3].append(perp_correct.sum() / perp_correct.shape[0])
 

    headers = ["Metric"] + headers
    print(tabulate(table, headers=headers))