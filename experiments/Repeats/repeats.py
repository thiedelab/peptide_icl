from peptide_icl import ICLScorer
from tqdm import tqdm
import torch
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
import esm

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    length = 10
    num_examples = 5
    one_pos = "W" * length
    positive_prompt_seqs = [one_pos] * num_examples
    
    linker = "Y" * length

    esm_model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()

    icl_scorer = ICLScorer(
        positive_sequences=positive_prompt_seqs,
        esm_model=(esm_model, alphabet),
        linker=linker,
        query_len=len(linker),
    ).to(device)
    
    print(icl_scorer.scorer.tok_to_idx)
    print(icl_scorer.log_prob_lookup_table.shape)
    print(torch.argmax(icl_scorer.log_prob_lookup_table, dim=-1))


if __name__ == "__main__":
    main()
