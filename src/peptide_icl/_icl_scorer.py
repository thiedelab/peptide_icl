import torch
import torch.nn as nn
import esm
from typing import List, Union, Tuple


def make_prompt_text(sequences, linker="GGGGG", query_len=15):
    mask = "<mask>" * query_len
    prompt = linker.join(sequences + [mask])
    return prompt


class ESMTokenizer(nn.Module):
    def __init__(self, alphabet):
        super().__init__()

        self.alphabet = alphabet
        self.batch_converter = self.alphabet.get_batch_converter()

    def tokenize_prompts(self, prompts: Union[str, List[str]]):
        if isinstance(prompts, str):
            prompts = [prompts]
        prompt_tuples = [(f"prompt_{i}", prompt) for i, prompt in enumerate(prompts)]
        _, _, toks = self.batch_converter(prompt_tuples)
        return toks

    def forward(self, prompts):
        toks = self.tokenize_prompts(prompts)
        return toks


class PromptProcessor(nn.Module):
    def __init__(self, esm_model, tokenizer):
        super().__init__()

        self.esm_model = esm_model
        self.tokenizer = tokenizer
        self.tok_to_idx = self.tokenizer.alphabet.tok_to_idx
        self.esm_model.eval()

    def get_mask_positions(self, toks):
        mask_token_idx = self.tok_to_idx["<mask>"]
        mask_positions = (toks[0] == mask_token_idx).nonzero(as_tuple=True)[0]
        return mask_positions

    def run_ESM(self, toks):
        try:
            device = next(self.esm_model.parameters()).device
        except StopIteration:
            device = torch.device("cpu")

        with torch.no_grad():
            results = self.esm_model(
                toks.to(device),
                repr_layers=[self.esm_model.num_layers],
                return_contacts=False,
            )
            representations = results["representations"][self.esm_model.num_layers]
        logits = self.esm_model.lm_head(representations)

        return logits

    def build_lookup_table(self, prompt):
        toks = self.tokenizer(prompt)
        logits = self.run_ESM(toks=toks)
        mask_positions = self.get_mask_positions(toks=toks)

        # TODO: Generalize this to work with batch of prompts
        log_probs = torch.log_softmax(logits[0], dim=-1)
        unmasked_tokens_log_prob_lookup_table = log_probs[mask_positions]

        return unmasked_tokens_log_prob_lookup_table

    def forward(self, positive_prompt, negative_prompt=None):
        if negative_prompt is None:
            return self.build_lookup_table(positive_prompt)
        else:
            positive_lookup_table = self.build_lookup_table(positive_prompt)
            negative_lookup_table = self.build_lookup_table(negative_prompt)
            return positive_lookup_table - negative_lookup_table


class QueryPeptideScorer(nn.Module):
    def __init__(self, log_prob_lookup_table, tokenizer):
        super().__init__()

        self.log_prob_lookup_table = log_prob_lookup_table
        self.tok_to_idx = tokenizer.alphabet.tok_to_idx
        self.tokenizer = tokenizer

    def score_peptides_torch(self, query_peptides):
        tokens = self.tokenizer(query_peptides)
        residue_tokens = tokens[:, 1:-1]  # remove BOS and EOS

        batch, seq_len = residue_tokens.shape
        positions = (
            torch.arange(seq_len, device=residue_tokens.device)
            .unsqueeze(0)
            .expand(batch, -1)
        )
        per_residue_scores = self.log_prob_lookup_table[
            positions, residue_tokens
        ]  # shape [batch, seq_len]
        scores = per_residue_scores.sum(dim=1)

        return scores

    def forward(self, query_peptides):
        return self.score_peptides_torch(query_peptides)


class ICLScorer(nn.Module):
    def __init__(
        self,
        positive_sequences,
        negative_sequences=None,
        esm_model: Union[
            str, Tuple[torch.nn.Module, torch.nn.Module]
        ] = "esm2_t6_8M_UR50D",
        linker="GGGGG",
        query_len=15,
    ):
        super().__init__()

        self.positive_prompt = make_prompt_text(
            positive_sequences, linker=linker, query_len=query_len
        )
        if negative_sequences is not None:
            self.negative_prompt = make_prompt_text(
                negative_sequences, linker=linker, query_len=query_len
            )
        else:
            self.negative_prompt = None

        if isinstance(esm_model, str):
            self.plm, self.alphabet = esm.pretrained.load_model_and_alphabet(esm_model)
        else:
            self.plm, self.alphabet = esm_model

        self.tokenizer = ESMTokenizer(self.alphabet)
        self.prompt_processor = PromptProcessor(self.plm, self.tokenizer)
        self.log_prob_lookup_table = self.prompt_processor(
            self.positive_prompt, self.negative_prompt
        )

        self.scorer = QueryPeptideScorer(
            log_prob_lookup_table=self.log_prob_lookup_table, tokenizer=self.tokenizer
        )

    def forward(self, query_peptides):
        scores = self.scorer(query_peptides)

        return scores
