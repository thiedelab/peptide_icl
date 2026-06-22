from peptide_icl import ICLScorer
from tqdm import tqdm
import torch
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
import esm
import pandas as pd

def prepare_dataset(max_prompt_length):
    df = pd.read_csv('mhc_dataset.csv')
    sequences = list(df['sequence'])
    true_labels = torch.tensor(df['labels'])

    pos_train_seqs = sequences[:max_prompt_length]
    neg_train_seqs = sequences[-max_prompt_length:]
    test_seqs = sequences[max_prompt_length:-max_prompt_length]
    test_labels = [1] * (250 - max_prompt_length) + [0] * (250 - max_prompt_length)
    return pos_train_seqs, neg_train_seqs, test_seqs, test_labels

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    prompt_lengths = [2, 4, 8, 16, 32]
    max_prompt_length = max(prompt_lengths)
    esm_model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
    
    pos_train_seqs, neg_train_seqs, test_seqs, test_labels = prepare_dataset(
            max_prompt_length
        )

    roc_aucs = []

    for prompt_length in tqdm(prompt_lengths):
        positive_prompt_seqs = pos_train_seqs[:prompt_length]
        negative_prompt_seqs = neg_train_seqs[-prompt_length:]

        icl_scorer = ICLScorer(
            positive_sequences=positive_prompt_seqs,
            negative_sequences=negative_prompt_seqs,
            esm_model=(esm_model, alphabet),
            linker="GGGGG",
            query_len=15,
        ).to(device)
        with torch.no_grad():
            predicted_scores = icl_scorer(test_seqs).detach().cpu().numpy()
        roc_auc = metrics.roc_auc_score(test_labels, predicted_scores)
        roc_aucs.append(roc_auc)

    plt.plot(prompt_lengths, roc_aucs, marker="o")
    plt.xlabel("Prompt Length")
    plt.ylabel("ROC AUC")
    plt.title("ICL Performance vs Prompt Length")
    plt.show()


if __name__ == "__main__":
    main()
