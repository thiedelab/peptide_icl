from typing import Union, Tuple
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM
from tokenizers import Tokenizer as ProGenTokenizer


def make_prompt_text(sequences, linker="GGGGG"):
    """ """
    prompt = linker.join(sequences)
    prompt = "1" + prompt
    return prompt


class AutoRegressive_Scorer(nn.Module):
    def __init__(
        self,
        positive_sequences,
        negative_sequences=None,
        progen_model: Union[
            str, Tuple[torch.nn.Module, torch.nn.Module]
        ] = "hugohrban/projgen2-medium",
        linker: str = "GGGGG",
        loop: bool = True,
    ):
        super().__init__()

        self.positive_prompt = make_prompt_text(positive_sequences, linker=linker)
        if negative_sequences is not None:
            self.negative_prompt = make_prompt_text(negative_sequences, linker=linker)
        else:
            self.negative_prompt = None

        if isinstance(progen_model, str):
            self.plm = AutoModelForCausalLM.from_pretrained(
                progen_model, trust_remote_code=True, torch_dtype="auto"
            )
            self.tokenizer = ProGenTokenizer.from_pretrained(progen_model)
            self.tokenizer.no_padding()
        else:
            self.plm, self.tokenizer = progen_model
        self.loop = loop

    def _score_loop(self, query: str, prompt: str) -> str:
        score = 0
        query_ids = self.tokenizer.convert_tokens_to_ids(query)
        for k in range(len(query)):
            input_k = prompt + query[:k]
            logits = self.plm(input_k).logits[-1]
            score += logits[query_ids[k]]
        return score

    def _score_one_pass(self, query: str, prompt: str) -> str:
        query_ids = self.tokenizer.convert_tokens_to_ids(query)
        input_seq = prompt + query
        logits = self.plm(input_seq)
        score = logits[query_ids].sum(dim=-1)
        return score

    def forward(self, query_peptides):
        """ """
        if self.loop:
            score_fxn = self._score_loop
        else:
            score_fxn = self._score_one_pass

        scores = []
        for q in query_peptides:
            score = self._score_fxn(q, self.positive_prompt)
            if self.negative_prompt is not None:
                negative_score = score_fxn(q, self.negative_prompt)
                score = score - negative_score
            scores.append(score)
        return score
