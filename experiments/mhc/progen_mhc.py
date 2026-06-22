import torch
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
import esm
import pandas as pd
import numpy as np
from peptide_icl.progen import AutoRegressiveScorer
from tqdm import tqdm
from esm_mhc import prepare_dataset
from transformers import AutoModelForCausalLM
from tokenizers import Tokenizer

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device('cpu')
    # prompt_lengths = [2, 4, 8, 16, 32]
    prompt_lengths = [2, 4, 8, 16]
    batch_size = 8
   
    max_prompt_length = max(prompt_lengths)
    pos_train_seqs, neg_train_seqs, test_seqs, test_labels = prepare_dataset(
            max_prompt_length
        )
    
    # rand_choice = np.random.choice(len(test_seqs), size=10)
    # test_seqs = [test_seqs[i] for i in rand_choice]
    # test_labels = [test_labels[i] for i in rand_choice]


    progen_model = "hugohrban/progen2-medium"
    plm = AutoModelForCausalLM.from_pretrained(
        progen_model, trust_remote_code=True, torch_dtype="auto"
    )
    tokenizer = Tokenizer.from_pretrained(progen_model)
    

    roc_aucs = []

    for prompt_length in tqdm(prompt_lengths):
        positive_prompt_seqs = pos_train_seqs[:prompt_length]
        negative_prompt_seqs = neg_train_seqs[-prompt_length:]

        icl_scorer = AutoRegressiveScorer(
            positive_sequences=positive_prompt_seqs,
            negative_sequences=negative_prompt_seqs,
            progen_model=(plm, tokenizer),
            linker="GGGGG",
            # loop=False,
            loop=True,
        ).to(device)
        with torch.no_grad():
            scores = []
            for i, batch_idx in enumerate(tqdm(range(0, len(test_seqs), batch_size))):
                batch = test_seqs[batch_idx : (batch_idx + batch_size)] 
                score = icl_scorer(batch).detach().cpu()
                scores.append(score)
            predicted_scores = torch.cat(scores, dim=0).numpy()

        roc_auc = metrics.roc_auc_score(test_labels, predicted_scores)
        roc_aucs.append(roc_auc)

    plt.plot(prompt_lengths, roc_aucs, marker="o")
    plt.xlabel("Prompt Length")
    plt.ylabel("ROC AUC")
    plt.title("ICL Performance vs Prompt Length")
    plt.show()


if __name__ == "__main__":
    main()
