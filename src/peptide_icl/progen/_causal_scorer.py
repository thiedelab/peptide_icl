from typing import Union, Tuple, List
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM
from tokenizers import Tokenizer as ProGenTokenizer
from torch.nn.functional import log_softmax


class AutoRegressiveScorer(nn.Module):
    def __init__(
        self,
        positive_sequences, #can be '' or [] in the case n=0
        negative_sequences=None,
        progen_model: Union[
            str, Tuple[torch.nn.Module, torch.nn.Module] #either pass the name of model, or initialized (model, tokenizer)
        ] = "hugohrban/progen2-medium",
        linker: str = "GGGGG",
        use_end_token: bool=False,
        loop: bool = True, #NOTE: AA-by-AA (loop=True) or every AA at once (loop=False)
    ):
        super().__init__()

        # General behavior flags
        self.loop = loop
        self.use_end_token = use_end_token

        # If using end token, read position must be shifted back by one
        self.read_shift = -1 * int(self.use_end_token)

        # Define prompts
        self.positive_prompt = linker.join(positive_sequences)

        #NOTE: Cannot handle negative_sequences = torch tensor yet!
        if negative_sequences is None:
            self.negative_prompt = None
        else:
            self.negative_prompt = linker.join(negative_sequences)
        self.linker = linker

        # Build the model
        if isinstance(progen_model, str):
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.plm = AutoModelForCausalLM.from_pretrained(
                progen_model, trust_remote_code=True, torch_dtype="auto"
            )
            self.plm = self.plm.to(device)
            self.plm.eval() #make sure model in inference mode
            self.tokenizer = ProGenTokenizer.from_pretrained(progen_model)
            self.tokenizer.no_padding()
        else:
            self.plm, self.tokenizer = progen_model #NOTE: Assuming that model has been sent to device prior

    def _complete_prompt(self, query: str, base_prompt:str):
        # context + GGGGG + query (if context is not empty)
        prompt = self.linker.join([base_prompt, query]) if base_prompt else query 
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

        for k in range(0, length_of_one_query):
            query_batch_i = [q[:k] for q in query_batch]
            input_ids_k = self._prepare_inputs(query_batch_i, prompt)
            logits = self.plm(input_ids_k).logits[:, -1 + self.read_shift]
            log_probabilites = log_softmax(logits, dim=-1)
            score = score + log_probabilites[batch_indices, query_ids[:, k]]
        return score

    def _score_one_pass(self, query_batch: List[str], prompt: str) -> str:
        K = len(query_batch[0])

        input_ids = self._prepare_inputs(query_batch, prompt)
        query_encoded = self.tokenizer.encode_batch(query_batch)

        logits = self.plm(input_ids).logits
        logits = logits[:, logits.shape[1]-K+self.read_shift: logits.shape[1] + self.read_shift]
        log_probabilites = log_softmax(logits, dim=-1)
        aa_scores = []
        for i, q_enc in enumerate(query_encoded):
            aa_logits = log_probabilites[i][torch.arange(K), q_enc.ids]
            aa_scores.append(aa_logits.sum())
        return torch.stack(aa_scores)

    def _score(self, batch):
        scoring_fxn = self._score_loop if self.loop else self._score_one_pass

        score = scoring_fxn(batch, self.positive_prompt)

        if self.negative_prompt is not None:
            score -= scoring_fxn(batch, self.negative_prompt)

        return score

    def forward(self, batch, batch_size=50):
        outputs = []

        batch_start = 0
        while batch_start < len(batch):
            batch_end = min(batch_start + batch_size, len(batch))
            mini_batch = batch[batch_start:batch_end]

            try:
                with torch.no_grad():
                    outputs.append(self._score(mini_batch))

                batch_start = batch_end

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    torch.cuda.empty_cache()
                    batch_size = max(1, int(batch_size / 2.5))
                    print(f"[OOM] Reducing batch_size to {batch_size}, retrying from {batch_start}")
                else:
                    raise RuntimeError('[OOM] Batch can no longer be reduced. Consider shorter prompts.')

        return torch.cat(outputs)

    # def forward(self, batch):
    #     """ """
    #     if self.loop:
    #         scoring_fxn = self._score_loop
    #     else:
    #         scoring_fxn = self._score_one_pass

    #     score = scoring_fxn(batch, self.positive_prompt)
    #     if self.negative_prompt is not None:
    #         negative_score = scoring_fxn(batch, self.negative_prompt)
    #         score = score - negative_score
    #     return score
