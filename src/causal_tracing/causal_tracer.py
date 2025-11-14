from collections import defaultdict

import numpy as np
import torch
from tqdm import tqdm

from causal_tracing.layer_config import get_layer_config
from causal_tracing.noise_level import pick_noise_level, resize_list_by_samples, retrieved_embeddings
from utils.constants import DEFAULT_DEVICE, DEFAULT_PROMPT_FORMAT
from utils.layer_matching import collect_matching_layers, get_layer_name
from utils.logger import log_or_print
from utils.token_utils import compute_joint_probs, find_token_range, get_remaining_token_ranges, make_inputs, predict_from_input, predict_with_cf_from_input
from utils.trace import Trace
from utils.trace_dict import TraceDict


class HiddenFlow:
    def __init__(self, scores, low_score, high_score, input_ids, input_tokens, target_ranges, answer, kind, clean_run, corrupted_run, corrupted_with_restoration_run):
        self.scores = scores
        self.low_score = low_score
        self.high_score = high_score
        self.input_ids = input_ids
        self.input_tokens = input_tokens
        self.target_ranges = target_ranges
        self.answer = answer
        self.kind = kind
        self.clean_run = clean_run
        self.corrupted_run = corrupted_run
        self.corrupted_with_restoration_run = corrupted_with_restoration_run

class HiddenFlowQuery:
    def __init__(self, prompt, prompt_attrs=None, prompt_format=None, target=None, cf=None, noise=None, noise_position=None):
        self.prompt = prompt
        self.prompt_attrs = prompt_attrs if prompt_attrs else {}
        self.prompt_format = prompt_format if prompt_format else DEFAULT_PROMPT_FORMAT
        self.target = target
        self.cf = cf
        self.noise = noise
        self.noise_position = noise_position

    def offset(self, tokenizer):
        return len(tokenizer.tokenize(self.prompt_format.replace(" {prompt}", "").format(**self.prompt_attrs)))

    @property
    def text(self):
        return self.prompt_format.format(prompt=self.prompt, **self.prompt_attrs)

    @property
    def noisy_text(self):
        if self.noise is None:
            return self.prompt_format.format(prompt=self.prompt, **self.prompt_attrs)
        

        prompt = self.prompt.replace(self.target, self.noise) if self.noise_position == "prompt" else self.prompt
        prompt_attrs = {
            k: (v.replace(self.target, self.noise) if k == self.noise_position else v)
            for k, v in self.prompt_attrs.items()
        } if self.noise_position != "prompt" else self.prompt_attrs

        return self.prompt_format.format(prompt=prompt, **prompt_attrs)

class CausalTracer:
    def __init__(self, model, tokenizer, layer_config=None, noise=None, device=None):
        device = device if device else DEFAULT_DEVICE
        model.to(device).eval()
        
        model.eval()
        self.model = model
        self.tokenizer = tokenizer
        self.layer_config = layer_config or get_layer_config(model)
        self.noise = noise if noise is not None else pick_noise_level(model, self.layer_config.embedding_layer)
        self.device = device
    
    def count_layers(self):
        return len(collect_matching_layers(self.model, self.layer_config.hidden_layers_matcher))

    def get_layer_name(self, layer_num, layer_kind):
        if layer_kind == "hidden":
            return get_layer_name(self.model, self.layer_config.hidden_layers_matcher, layer_num)
        if layer_kind == "mlp":
            return get_layer_name(self.model, self.layer_config.mlp_layers_matcher, layer_num)
        if layer_kind == "attention":
            return get_layer_name(self.model, self.layer_config.attention_layers_matcher, layer_num)
        raise ValueError("Unknown layer kind")

    def get_layer_names_of_kind(self, layer_kind):
        return [self.get_layer_name(i, layer_kind) for i in range(self.count_layers())]

    def extract_patch(self, model, states_to_patch, input_dict):
        patch = {}
        for layer in states_to_patch:
            with Trace(model, layer[1], stop=True) as t:
                model(**input_dict)

            patch[layer[1]] = t.output[0] if isinstance(t.output, tuple) else t.output
        
        return patch

    @torch.no_grad()
    def trace_with_patch(
        self,
        model,  # The model
        input_dict,  # A set of inputs
        states_to_patch,  # A list of (token index, layername) triples to restore
        answer_token,  # Answer probabilities to collect
        target_ranges,  # Range of tokens to corrupt (begin, end)
        target_noise,
        trace_layers=None,  # List of traced outputs to return
        uniform_noise=False,
        replace=False, # True to replace with instead of add noise
        is_noise_generated=False,
        generated_noise=None,
        noisy_patch_spec=None,
        patch_seed=1,
    ):
        # For reproducibility, use pseudorandom noise
        rs = np.random.RandomState(patch_seed)  
        if uniform_noise:
            prng = lambda *shape: rs.uniform(-1, 1, shape)
        else:
            prng = lambda *shape: rs.randn(*shape)

        patch_spec = defaultdict(list)
        for t, l in states_to_patch:
            patch_spec[l].append(t)

        embed_layername = self.layer_config.embedding_layer

        def untuple(x):
            return x[0] if isinstance(x, tuple) else x

        # Define the model-patching rule.
        def patch_rep(x, layer):
            nonlocal is_noise_generated, generated_noise

            # print("HERE!")
            # print(f"is_noise_generated: {is_noise_generated}")
            # print(f"generated_noise: {generated_noise}")
            # print(f"input_dict: {input_dict}")
            # print(f"answer_token: {answer_token}")
            # print(f"target_range: {target_range}")
            # print(f"target_noise: {target_noise}")
            # raise

            if layer == embed_layername:
                # print("BEFORE 1", x[0, 0, :10])
                # print("BEFORE 2", x[1, 0, :10])
                # If requested, we corrupt a range of token embeddings on batch items x[1:]
                if target_ranges is not None:
                    for i, target_range in enumerate(target_ranges):
                        b, e = target_range

                        if not is_noise_generated[i]:
                            if not isinstance(target_noise, float):
                                noise = torch.from_numpy(np.array(target_noise)).to(x.device)
                            else:
                                noise = target_noise * torch.from_numpy(prng(x.shape[0] - 1, e - b, x.shape[2])).to(x.device)

                            generated_noise[i] = noise
                            is_noise_generated[i] = True


                        if replace:
                            x[1:, b:e] = generated_noise[i]
                        else:
                            x[1:, b:e] += generated_noise[i]

                # print("AFTER 1", x[0, 0, :10])
                # print("AFTER 2", x[1, 0, :10])
                return x

            if layer not in patch_spec:
                return x

            # If this layer is in the patch_spec, restore the uncorrupted hidden state
            # for selected tokens.
            h = untuple(x)
            for t in patch_spec[layer]:
                if noisy_patch_spec and layer in noisy_patch_spec:
                    h[1:, t] = noisy_patch_spec[layer][1:, t]
                else:
                    h[1:, t] = h[0, t]
            return x

        # With the patching rules defined, run the patched model in inference.
        additional_layers = [] if trace_layers is None else trace_layers
        with torch.no_grad(), TraceDict(
            model,
            [embed_layername] + list(patch_spec.keys()) + additional_layers,
            edit_output=patch_rep,
        ) as td:
            outputs_exp = model(**input_dict)
        
        logits = outputs_exp.logits
        score = torch.softmax(logits[1:, -1, :], dim=-1).mean(dim=0)[answer_token]
        # next_tokens = torch.argmax(torch.softmax(logits[:, -1, :], dim=-1), dim=-1).unsqueeze(-1)
        # probs = torch.softmax(logits[1:, -1, :], dim=-1).mean(dim=0)[answer_token]
        # scores = torch.softmax(logits[1:, -1, :], dim=-1).mean(dim=0)

        # If tracing all layers, collect all activations together to return.
        if trace_layers is not None:
            all_traced = torch.stack(
                [untuple(td[layer].output).detach().cpu() for layer in trace_layers], dim=2
            )
            return logits, score, is_noise_generated, generated_noise, all_traced 

        return logits, score, is_noise_generated, generated_noise

    @torch.no_grad()
    def trace_important_states(
        self,
        model,
        num_layers,
        input_dict,
        answer_tokens,
        target_ranges,
        target_noise,
        counterfactuals=None,
        kind="hidden",
        uniform_noise=False,
        replace=False,
        token_range=None,
        patch_seed=1,
        desc="trace_important_states"
    ):
        ntoks = input_dict["input_ids"].shape[1]
        table = []

        if token_range is None:
            token_range = range(ntoks)

        for tnum in tqdm(token_range, total=ntoks,  desc=desc):
            row = []
            for layer in range(num_layers):
                
                current_input_ids = input_dict["input_ids"].clone()
                current_attn_mask = input_dict["attention_mask"].clone()
                is_noise_generated, generated_noise = [False] * len(target_ranges), [None] * len(target_ranges)
                
                scores, ground_scores, cf_scores = [], [], []
                pred_tokens = []

                for i, t_id in enumerate(answer_tokens):
                    states_to_patch = [(tnum, self.get_layer_name(i, kind))]

                    noisy_patch_spec = None
                    if torch.is_tensor(input_dict["noisy_input_ids"]) and torch.is_tensor(input_dict["noisy_attention_mask"]):
                        noisy_patch_spec = self.extract_patch(self.model, states_to_patch, {"input_ids": input_dict["noisy_input_ids"], "attention_mask": input_dict["noisy_attention_mask"]})

                    _ground_scores, _cf_scores = [], []

                    logits, score, is_noise_generated, generated_noise = self.trace_with_patch(
                        model=self.model,
                        input_dict={"input_ids": current_input_ids, "attention_mask": current_attn_mask},
                        states_to_patch=states_to_patch, 
                        answer_token=t_id,
                        target_ranges=target_ranges,
                        target_noise=target_noise,
                        is_noise_generated=is_noise_generated,
                        generated_noise=generated_noise,
                        patch_seed=patch_seed,
                        replace=replace,
                        noisy_patch_spec=noisy_patch_spec
                    )

                    scores.append(score)
                    p = torch.softmax(logits[1:, -1, :], dim=1)

                    for j in range(current_input_ids.size(0) - 1):
                        _ground_scores.append(p[j, t_id])
                        if counterfactuals:
                            cf_token = self.tokenizer.encode(f" {counterfactuals[j]}", add_special_tokens=False)
                            _cf_scores.append(p[j, cf_token[i]])

                    ground_scores.append(_ground_scores)
                    cf_scores.append(_cf_scores)

                    next_tokens = torch.argmax(torch.softmax(logits[:, -1, :], dim=-1), dim=-1).unsqueeze(-1)
                    current_input_ids = torch.cat([current_input_ids, next_tokens], dim=-1)
                    current_attn_mask = torch.cat([current_attn_mask, torch.ones_like(next_tokens)], dim=-1)

                    pred_tokens.append(next_tokens)

                scores = compute_joint_probs(torch.tensor(scores).to(self.device))
                ground_scores = compute_joint_probs(torch.tensor(ground_scores).to(self.device).t())
                cf_scores = compute_joint_probs(torch.tensor(cf_scores).to(self.device).t())
                pred_tokens = torch.cat(pred_tokens, dim=-1)
                pred_tokens_str = self.tokenizer.batch_decode(pred_tokens)
                
                r = dict(
                    scores=scores,
                    ground_scores=ground_scores,
                    cf_scores=cf_scores,
                    pred_tokens=pred_tokens,
                    pred_tokens_str=pred_tokens_str,
                )

                row.append(r)
            table.append(row)
            torch.cuda.empty_cache()
        return table
    
    @torch.no_grad()
    def trace_important_window(
        self,
        model,
        num_layers,
        input_dict,
        answer_tokens,
        target_ranges,
        target_noise,
        kind,
        counterfactuals=None,
        window=3,
        uniform_noise=False,
        replace=False,
        token_range=None,
        patch_seed=1,
        desc="trace_important_window"
    ):
        ntoks = input_dict["input_ids"].shape[1]
        table = []

        if token_range is None:
            token_range = range(ntoks)

        for tnum in tqdm(token_range, total=ntoks,  desc=desc):
            row = []
            for layer in range(num_layers):
                
                current_input_ids = input_dict["input_ids"].clone()
                current_attn_mask = input_dict["attention_mask"].clone()
                is_noise_generated, generated_noise = [False] * len(target_ranges), [None] * len(target_ranges)
                
                scores, ground_scores, cf_scores = [], [], []
                pred_tokens = []

                for i, t_id in enumerate(answer_tokens):
                    states_to_patch = [(tnum, self.get_layer_name(i, kind)) for i in range(max(0, layer - window // 2), min(num_layers, layer + window // 2))]

                    noisy_patch_spec = None
                    if torch.is_tensor(input_dict["noisy_input_ids"]) and torch.is_tensor(input_dict["noisy_attention_mask"]):
                        noisy_patch_spec = self.extract_patch(self.model, states_to_patch, {"input_ids": input_dict["noisy_input_ids"], "attention_mask": input_dict["noisy_attention_mask"]})

                    _ground_scores, _cf_scores = [], []

                    logits, score, is_noise_generated, generated_noise = self.trace_with_patch(
                        model=self.model,
                        input_dict={"input_ids": current_input_ids, "attention_mask": current_attn_mask},
                        states_to_patch=states_to_patch, 
                        answer_token=t_id,
                        target_ranges=target_ranges,
                        target_noise=target_noise,
                        is_noise_generated=is_noise_generated,
                        generated_noise=generated_noise,
                        patch_seed=patch_seed,
                        replace=replace,
                        noisy_patch_spec=noisy_patch_spec
                    )

                    scores.append(score)
                    p = torch.softmax(logits[1:, -1, :], dim=1)

                    for j in range(current_input_ids.size(0) - 1):
                        _ground_scores.append(p[j, t_id])
                        if counterfactuals:
                            cf_token = self.tokenizer.encode(f" {counterfactuals[j]}", add_special_tokens=False)
                            _cf_scores.append(p[j, cf_token[i]])

                    ground_scores.append(_ground_scores)
                    cf_scores.append(_cf_scores)

                    next_tokens = torch.argmax(torch.softmax(logits[:, -1, :], dim=-1), dim=-1).unsqueeze(-1)
                    current_input_ids = torch.cat([current_input_ids, next_tokens], dim=-1)
                    current_attn_mask = torch.cat([current_attn_mask, torch.ones_like(next_tokens)], dim=-1)

                    pred_tokens.append(next_tokens)

                scores = compute_joint_probs(torch.tensor(scores).to(self.device))
                ground_scores = compute_joint_probs(torch.tensor(ground_scores).to(self.device).t())
                cf_scores = compute_joint_probs(torch.tensor(cf_scores).to(self.device).t())
                pred_tokens = torch.cat(pred_tokens, dim=-1)
                pred_tokens_str = self.tokenizer.batch_decode(pred_tokens)
                
                r = dict(
                    scores=scores,
                    ground_scores=ground_scores,
                    cf_scores=cf_scores,
                    pred_tokens=pred_tokens,
                    pred_tokens_str=pred_tokens_str,
                )

                row.append(r)
            table.append(row)
            torch.cuda.empty_cache()
        return table

    def calculate_hidden_flow(
        self, 
        prompt,
        target_str,
        answer_str,
        target_noise=None,
        prompt_attrs=None, 
        prompt_format=None, 
        counterfactuals=None,
        target_position=None,  # None, "prompt", "context"
        samples=3, 
        window=3, 
        kind="hidden", 
        patch_seed=1,
        replace=False, 
        reverse_patching=False,
        queue_id=0,
    ):
        answer_str = f" {answer_str}"
        prompt_attrs = prompt_attrs if prompt_attrs else {}
        prompt_format = prompt_format if prompt_format else DEFAULT_PROMPT_FORMAT
        noises = [None] + resize_list_by_samples(target_noise, samples) 
        counterfactuals = resize_list_by_samples(counterfactuals, samples)
        queries = [
            HiddenFlowQuery(prompt, prompt_attrs=prompt_attrs, prompt_format=prompt_format, target=target_str, cf=cf, noise=noises[i], noise_position=target_position)
            for i, cf in enumerate([None] + counterfactuals)
        ]


        noisy_input_dict = {}
        if reverse_patching and isinstance(target_noise, list):
            noisy_prompts = [query.noisy_text for query in queries]
            noisy_input_dict = make_inputs(self.tokenizer, noisy_prompts, self.device)


        prompts = [query.text for query in queries]
        input_dict = make_inputs(self.tokenizer, prompts, self.device)
        base_input_dict = make_inputs(self.tokenizer, prompts[:1], self.device)
        base_output = predict_with_cf_from_input(self.model, self.tokenizer, base_input_dict, answer_str, counterfactuals=list(map(lambda x: f" {x}", counterfactuals)) if counterfactuals else None)
        # print(f"base_output: {base_output}")

        clean_run = dict(
            scores=base_output["j_ans_probs"][0].detach().cpu(),
            ground_scores=base_output["j_ans_probs"].detach().cpu(),
            cf_scores=torch.tensor(base_output["j_cfs_probs"]).detach().cpu(),
            pred_tokens=base_output["pred_tokens"].detach().cpu(),
            pred_tokens_str=base_output["pred_tokens_str"],
        )
        log_or_print("**CLEAN RUN**", verbose=True)
        log_or_print(clean_run, verbose=True)

        # print(f"offset: {queries[0].offset(self.tokenizer)}")
        # raise
        # print(f"target_str: {target_str}")
        target_str = target_str.split("+") if "+" in target_str else target_str
        if isinstance(target_str, list):
            # filter_prompt_tokens(tokenizer, encoded_prompt, subject_range, object_range)
            # print(f"target_str: {target_str}")

            subj_obj_range = [find_token_range(self.tokenizer, input_dict["input_ids"][0], target) for target in target_str]
            target_ranges = get_remaining_token_ranges(input_dict["input_ids"][0], subj_obj_range[0], subj_obj_range[1], offset=queries[0].offset(self.tokenizer) + 1)

            # print(f"target_range: {target_range}")
            # for rng in target_range:
            #     print(input_dict["input_ids"][0][rng[0]:rng[1]], self.tokenizer.decode(input_dict["input_ids"][0][rng[0]:rng[1]]))
        else:
            target_ranges = [find_token_range(self.tokenizer, input_dict["input_ids"][0], target_str)]

        log_or_print("**TARGET RANGEs**", verbose=True)
        log_or_print(target_ranges, verbose=True)


        answer_token_ids = self.tokenizer.encode(answer_str, add_special_tokens=False)
        current_input_ids = input_dict["input_ids"].clone()
        current_attn_mask = input_dict["attention_mask"].clone()
        is_noise_generated, generated_noise = [False] * len(target_ranges), [None] * len(target_ranges)
        target_noise = self.noise if not isinstance(target_noise, list) else retrieved_embeddings(self.model, self.tokenizer, layer=self.layer_config.embedding_layer, prompts=list(map(lambda x: f" {x}", resize_list_by_samples(target_noise, samples))))

        scores, ground_scores, cf_scores = [], [], []
        pred_tokens = []

        for i, t_id in enumerate(answer_token_ids):    
            _ground_scores, _cf_scores = [], []

            logits, score, is_noise_generated, generated_noise = self.trace_with_patch(
                model=self.model,
                input_dict={"input_ids": current_input_ids, "attention_mask": current_attn_mask},
                states_to_patch=[], 
                answer_token=t_id,
                target_ranges=target_ranges,
                target_noise=target_noise,
                is_noise_generated=is_noise_generated,
                generated_noise=generated_noise,
                patch_seed=patch_seed,
                replace=replace
            )

            scores.append(score)
            p = torch.softmax(logits[1:, -1, :], dim=1)

            for j in range(current_input_ids.size(0) - 1):
                _ground_scores.append(p[j, t_id])
                if counterfactuals:
                    cf_token = self.tokenizer.encode(f" {counterfactuals[j]}", add_special_tokens=False)
                    _cf_scores.append(p[j, cf_token[i]])

            ground_scores.append(_ground_scores)
            cf_scores.append(_cf_scores)

            next_tokens = torch.argmax(torch.softmax(logits[:, -1, :], dim=-1), dim=-1).unsqueeze(-1)
            current_input_ids = torch.cat([current_input_ids, next_tokens], dim=-1)
            current_attn_mask = torch.cat([current_attn_mask, torch.ones_like(next_tokens)], dim=-1)
            pred_tokens.append(next_tokens)

        scores = compute_joint_probs(torch.tensor(scores).to(self.device))
        ground_scores = compute_joint_probs(torch.tensor(ground_scores).to(self.device).t())
        cf_scores = compute_joint_probs(torch.tensor(cf_scores).to(self.device).t())
        pred_tokens = torch.cat(pred_tokens, dim=-1)
        pred_tokens_str = self.tokenizer.batch_decode(pred_tokens)

        corrupted_run = dict(
            scores=scores.detach().cpu(),
            ground_scores=ground_scores.detach().cpu(),
            cf_scores=cf_scores.detach().cpu(),
            pred_tokens=pred_tokens.detach().cpu(),
            pred_tokens_str=pred_tokens_str,
        )
        log_or_print("**CORRUPTED RUN**", verbose=True)
        log_or_print(corrupted_run, verbose=True)

        if kind == "hidden":
            differences = self.trace_important_states(
                model=self.model,
                num_layers=self.count_layers(),
                input_dict={
                    "input_ids": input_dict["input_ids"],
                    "attention_mask": input_dict["attention_mask"],
                    "noisy_input_ids": noisy_input_dict.get("input_ids"),
                    "noisy_attention_mask": noisy_input_dict.get("attention_mask"),
                },
                answer_tokens=answer_token_ids,
                target_ranges=target_ranges,
                target_noise=target_noise,
                counterfactuals=counterfactuals,
                replace=replace,
                patch_seed=patch_seed,
                token_range=None,
                desc=f"trace_important_states {queue_id}"
            )

            corrupted_with_restoration_run = dict(
                scores=torch.stack([torch.stack([r["scores"].detach().cpu() for r in d]) for d in differences]),
                ground_scores=torch.stack([torch.stack([r["ground_scores"].detach().cpu() for r in d]) for d in differences]),
                cf_scores=torch.stack([torch.stack([r["cf_scores"].detach().cpu() for r in d]) for d in differences]),
                pred_tokens=[[r["pred_tokens"] for r in d] for d in differences],
                pred_tokens_str=[[r["pred_tokens_str"] for r in d] for d in differences],
            )

        else:
            differences = self.trace_important_window(
                model=self.model,
                num_layers=self.count_layers(),
                input_dict={
                    "input_ids": input_dict["input_ids"],
                    "attention_mask": input_dict["attention_mask"],
                    "noisy_input_ids": noisy_input_dict.get("input_ids"),
                    "noisy_attention_mask": noisy_input_dict.get("attention_mask"),
                },
                answer_tokens=answer_token_ids,
                target_ranges=target_ranges,
                target_noise=target_noise,
                counterfactuals=counterfactuals,
                window=window,
                kind=kind,
                replace=replace,
                patch_seed=patch_seed,
                token_range=None,
                desc=f"trace_important_window {queue_id}"
            )

            corrupted_with_restoration_run = dict(
                scores=torch.stack([torch.stack([r["scores"].detach().cpu() for r in d]) for d in differences]),
                ground_scores=torch.stack([torch.stack([r["ground_scores"].detach().cpu() for r in d]) for d in differences]),
                cf_scores=torch.stack([torch.stack([r["cf_scores"].detach().cpu() for r in d]) for d in differences]),
                pred_tokens=[[r["pred_tokens"].detach().cpu() for r in d] for d in differences],
                pred_tokens_str=[[r["pred_tokens_str"] for r in d] for d in differences],
            )
        

        # log_or_print("**CORRUPTED WITH RESTORATION RUN**", verbose=True)
        # log_or_print(corrupted_with_restoration_run, verbose=True)
        
        return HiddenFlow(
            scores=corrupted_with_restoration_run["scores"],
            low_score=corrupted_run["scores"], 
            high_score=clean_run["scores"], 
            input_ids=input_dict["input_ids"][0], 
            input_tokens=[self.tokenizer.decode(token_id) for token_id in input_dict["input_ids"][0]], 
            target_ranges=target_ranges, 
            answer=answer_str, 
            kind=kind,
            clean_run=clean_run,
            corrupted_run=corrupted_run,
            corrupted_with_restoration_run=corrupted_with_restoration_run
        )
