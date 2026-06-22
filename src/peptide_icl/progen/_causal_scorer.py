from typing import Union, Tuple, List
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM
from tokenizers import Tokenizer as ProGenTokenizer


def make_prompt_text(sequences, linker="GGGGG"):
    """ """
    prompt = linker.join(sequences)
    prompt = "1" + prompt
    return prompt


class AutoRegressiveScorer(nn.Module):
    def __init__(
        self,
        positive_sequences,
        negative_sequences=None,
        progen_model: Union[
            str, Tuple[torch.nn.Module, torch.nn.Module]
        ] = "hugohrban/progen2-medium",
        linker: str = "GGGGG",
        loop: bool = True,
    ):
        super().__init__()

        self.positive_prompt = make_prompt_text(positive_sequences, linker=linker)
        if negative_sequences is not None:
            self.negative_prompt = make_prompt_text(negative_sequences, linker=linker)
        else:
            self.negative_prompt = None
        self.linker = linker

        if isinstance(progen_model, str):
            self.plm = AutoModelForCausalLM.from_pretrained(
                progen_model, trust_remote_code=True, torch_dtype="auto"
            )
            self.tokenizer = ProGenTokenizer.from_pretrained(progen_model)
            self.tokenizer.no_padding()
        else:
            self.plm, self.tokenizer = progen_model
        self.loop = loop

    def _prepare_inputs(self, query_batch: str, prompt: str):
        # Get IDs of amino acids in the query.
        query_encoded = self.tokenizer.encode_batch(query_batch)
        # Build the residues with the prompt.
        batch = [prompt + self.linker + q for q in query_batch]
        inputs_encoded = self.tokenizer.encode_batch(batch)
        input_ids = torch.tensor(
            [enc.ids for enc in inputs_encoded], device=self.plm.device
        )
        return input_ids, query_encoded

    def _score_loop(self, query_batch: List[str], prompt: str) -> str:
        score = torch.zeros(len(query_batch), device=self.plm.device)
        input_ids, query_encoded = self._prepare_inputs(query_batch, prompt)
        query_ids = torch.tensor([q.ids for q in query_encoded], device=self.plm.device)
        query_length = len(query_encoded[0])
        L = input_ids.shape[-1]
        for k in range(-query_length + 1, 1):
            input_ids_k = input_ids[:, : (L + k)]
            logits = self.plm(input_ids_k).logits[:, -1]
            query_ids_k = query_ids[:, k]

            score_k = logits[
                torch.arange(len(query_ids_k), device=query_ids_k.device), query_ids_k
            ]
            score += score_k
        return score

    def _score_one_pass(self, query_batch: List[str], prompt: str) -> str:
        K = len(query_batch[0])

        input_ids, query_encoded = self._prepare_inputs(query_batch, prompt)

        logits = self.plm(input_ids).logits[:, -K:]
        aa_scores = []
        for i, q_enc in enumerate(query_encoded):
            aa_logits = logits[i][torch.arange(K), q_enc.ids]
            aa_scores.append(aa_logits.sum())
        return torch.stack(aa_scores)

    def forward(self, batch):
        """ """
        if self.loop:
            scoring_fxn = self._score_loop
        else:
            scoring_fxn = self._score_one_pass

        score = scoring_fxn(batch, self.positive_prompt)
        if self.negative_prompt is not None:
            negative_score = scoring_fxn(batch, self.negative_prompt)
            score = score - negative_score
        return score
