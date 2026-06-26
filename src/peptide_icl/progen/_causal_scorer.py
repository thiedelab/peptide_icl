from typing import Union, Tuple, List
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM
from tokenizers import Tokenizer as ProGenTokenizer
from torch.nn.functional import softmax



class AutoRegressiveScorer(nn.Module):
    def __init__(
        self,
        positive_sequences,
        negative_sequences=None,
        progen_model: Union[
            str, Tuple[torch.nn.Module, torch.nn.Module]
        ] = "hugohrban/progen2-medium",
        linker: str = "GGGGG",
        use_end_token: bool=False,
        loop: bool = True,
    ):
        super().__init__()

        # General behavior flags
        self.loop = loop
        self.use_end_token = use_end_token

        # If using end token, read position must be shifted back by one
        self.read_shift = -1 * int(self.use_end_token)

        # Define prompts
        self.positive_prompt = linker.join(positive_sequences)
        if negative_sequences is not None:
            self.negative_prompt = linker.join(negative_sequences)
        else:
            self.negative_prompt = None
        self.linker = linker

        # Build the model
        if isinstance(progen_model, str):
            self.plm = AutoModelForCausalLM.from_pretrained(
                progen_model, trust_remote_code=True, torch_dtype="auto"
            )
            self.tokenizer = ProGenTokenizer.from_pretrained(progen_model)
            self.tokenizer.no_padding()
        else:
            self.plm, self.tokenizer = progen_model

    def _complete_prompt(self, query: str, base_prompt:str):
        prompt = self.linker.join([base_prompt, query])
        prompt = "1" + prompt
        if self.use_end_token:
            prompt = prompt + "2"
        return prompt

    def _prepare_inputs(self, query_batch: str, prompt: str):
        # Build the residues with the prompt.
        batch = [self._complete_prompt(q, prompt) for q in query_batch]

        inputs_encoded = self.tokenizer.encode_batch(batch)
        input_ids = torch.tensor(
            [enc.ids for enc in inputs_encoded], device=self.plm.device
        )
        return input_ids


    def _score_loop(self, query_batch: List[str], prompt: str) -> torch.tensor:
        score = torch.zeros(len(query_batch), device=self.plm.device)
        query_encoded = self.tokenizer.encode_batch(query_batch)
        query_ids = torch.tensor([q.ids for q in query_encoded], device=self.plm.device)
        batch_indices = torch.arange(len(query_batch), device=self.plm.device)
        length_of_one_query = len(query_batch[0])

        for k in range(1, length_of_one_query):
            query_batch_i = [q[:k] for q in query_batch]
            input_ids_k = self._prepare_inputs(query_batch_i, prompt)
            logits = self.plm(input_ids_k).logits[:, -1 + self.read_shift]
            log_probabilites = softmax(logits, dim=-1)
            score = score + log_probabilites[batch_indices, query_ids[:, k]]
        return score

    def _score_one_pass(self, query_batch: List[str], prompt: str) -> str:
        K = len(query_batch[0])

        input_ids = self._prepare_inputs(query_batch, prompt)
        query_encoded = self.tokenizer.encode_batch(query_batch)

        logits = self.plm(input_ids).logits
        logits = logits[:, logits.shape[1]-K+self.read_shift: logits.shape[1] + self.read_shift]
        log_probabilites = softmax(logits, dim=-1)
        aa_scores = []
        for i, q_enc in enumerate(query_encoded):
            aa_logits = log_probabilites[i][torch.arange(K), q_enc.ids]
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
