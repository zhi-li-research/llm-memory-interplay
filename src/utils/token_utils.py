import torch

from utils.constants import DEFAULT_DEVICE
from utils.utils import find_all_substring_indices


def make_inputs(tokenizer, prompts, device=None, add_pad_token=True, add_special_tokens=True):
    device = device if device else DEFAULT_DEVICE
    ensure_tokenizer_has_pad_token(tokenizer, add_pad_token=add_pad_token)
    return tokenizer(prompts, padding=True, return_tensors="pt", add_special_tokens=add_special_tokens).to(device)

def ensure_tokenizer_has_pad_token(tokenizer, add_pad_token=True):
    if not tokenizer.pad_token:
        if add_pad_token:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            raise ValueError("Tokenizer must have a pad token")    


def decode_tokens(tokenizer, token_array):
    return [tokenizer.decode([t]) for t in token_array]

    
def find_token_range(tokenizer, token_array, substring, find_last_match=False):
    substr_toks = decode_tokens(tokenizer, tokenizer(substring)["input_ids"])
    if tokenizer.bos_token and substr_toks[0] == tokenizer.bos_token:
        substr_toks = substr_toks[1:]
    recoded_substr = "".join(substr_toks)
    toks = decode_tokens(tokenizer, token_array)
    whole_string = "".join(toks)
    char_locs = find_all_substring_indices(whole_string, recoded_substr)
    if not char_locs:
        if substring[0] != " ":
            return find_token_range(tokenizer, token_array, " " + substring)
        raise ValueError(f"Could not find substring {recoded_substr} in {whole_string}")
    token_ranges = []
    for char_loc in char_locs:
        loc = 0
        tok_start, tok_end = None, None
        for i, t in enumerate(toks):
            loc += len(t)
            if tok_start is None and loc > char_loc:
                tok_start = i
            if tok_end is None and loc >= char_loc + len(recoded_substr):
                tok_end = i + 1
                break
        if tok_start is not None and tok_end is not None:
            token_ranges.append((tok_start, tok_end))
    if not token_ranges:
        raise ValueError(f"Could not find substring {recoded_substr} in {toks}")
    return token_ranges[-1] if find_last_match else token_ranges[0]

# def filter_prompt_tokens(tokenizer, encoded_prompt, subject_range, object_range):
#     """
#     Given a tokenizer, the encoded_prompt, and the token ranges for subject and object,
#     return the decoded text of all tokens except those belonging to the subject or object.
#     """
#     # 1. Create a set of all token indices
#     all_indices = set(range(len(encoded_prompt)))
    
#     # 2. Create sets of token indices for subject and object
#     subject_indices = set(range(subject_range[0], subject_range[1]))
#     object_indices = set(range(object_range[0], object_range[1]))
    
#     # 3. Subtract out subject and object from the total set
#     keep_indices = all_indices - subject_indices - object_indices
    
#     # 4. Build a new token array that excludes subject and object
#     filtered_tokens = [encoded_prompt[i] for i in sorted(keep_indices)]
    
#     # 5. Decode and return
#     filtered_text = tokenizer.decode(filtered_tokens)
#     return filtered_text

def get_remaining_token_ranges(encoded_prompt, subject_range, object_range, offset=0):
    """
    Return a list of contiguous (start, end) token-index ranges
    that remain after removing the subject_range and object_range.
    
    :param encoded_prompt: list of token IDs for the entire prompt
    :param subject_range: 2-tuple (s_start, s_end) for the subject
    :param object_range: 2-tuple (o_start, o_end) for the object
    :return: list of (segment_start, segment_end), covering all 
             remaining tokens in sorted order
    """
    n_tokens = len(encoded_prompt)
    n_tokens = offset if offset > 0 else n_tokens

    # 1. Create sets of indices to exclude:
    #    from s_start..s_end-1 and o_start..o_end-1
    subject_indices = set(range(subject_range[0], subject_range[1]))
    object_indices = set(range(object_range[0], object_range[1]))
    exclude_indices = subject_indices.union(object_indices)

    # 2. Build a sorted list of kept indices (i.e., not excluded)
    keep_indices = [i for i in range(1, n_tokens) if i not in exclude_indices]

    # 3. Group kept indices into contiguous ranges
    remaining_ranges = _contiguous_ranges(keep_indices)
    return remaining_ranges


def _contiguous_ranges(sorted_indices):
    """
    Given a sorted list of indices, chunk them into contiguous 
    segments and return them as [(start, end), ...].
    
    E.g. [0,1,2,5,6,9] -> [(0,3), (5,7), (9,10)]
    """
    if not sorted_indices:
        return []

    ranges = []
    start = sorted_indices[0]
    prev  = start

    for idx in sorted_indices[1:]:
        # If there's a gap, close off the previous segment
        if idx != prev + 1:
            ranges.append((start, prev + 1))  # (start inclusive, end exclusive)
            start = idx
        prev = idx
    
    # Add the last segment
    ranges.append((start, prev + 1))
    return ranges

def find_final_attention_positions(attention_mask):
    indices = torch.arange(attention_mask.size(1)).to(attention_mask.device)
    indices = indices[None, :].expand_as(attention_mask)
    indices = torch.where(attention_mask == 1, indices, -1)
    return indices.max(dim=1).values

def compute_joint_probs(probs, min_prob=1e-12, smoothing_const=False, thresholding=False):
    if smoothing_const:
        probs = probs + min_prob

    if thresholding:
        probs = torch.where(probs < min_prob,
                            torch.tensor(min_prob, device=probs.device, dtype=probs.dtype),
                            probs)

    if probs.numel() == 0:
        return torch.tensor(0, device=probs.device, dtype=probs.dtype)

    log_probs = torch.log(probs)
    log_joint = log_probs.sum(dim=-1)
    j_prob = torch.exp(log_joint)

    return j_prob

@torch.no_grad()
def predict_from_input(model, tokenizer, input_dict, answer_str):
    answer_token_ids = tokenizer.encode(answer_str, add_special_tokens=False)

    device = input_dict["input_ids"].device
    current_input_ids = input_dict["input_ids"].clone()
    current_attn_mask = input_dict["attention_mask"].clone()

    batch_size = current_input_ids.shape[0]
    ls_probs = []
    pred_tokens = []
    ls_pred_probs = []
    ls_ans_probs = []
    
    for t_id in answer_token_ids:
        # print(f"current_input_ids: {current_input_ids.shape}")

        # print(f"currect_input_ids: {tokenizer.batch_decode(current_input_ids)}")

        logits = model(input_ids=current_input_ids, attention_mask=current_attn_mask)["logits"]
        # print(f"logits: {logits.shape}")
        final_token_positions = find_final_attention_positions(current_attn_mask)
        batch_indices = torch.arange(logits.size(0))
        

        probs = torch.softmax(logits[batch_indices, final_token_positions], dim=-1)
        ls_probs.append(probs)
        # print(f"probs: {probs.shape}")

        p_probs, tokens = torch.max(probs, dim=1, keepdim=True)
        # print(f'predicted token: {pred}')
        # print(f'predicted token: {prob}')

        # print(f"probs[:, t_id]: {probs[:, t_id].shape}")
        pred_tokens.append(tokens)
        ls_pred_probs.append(p_probs)
        ls_ans_probs.append(probs[:, t_id])
        
        current_input_ids = torch.cat(
            [current_input_ids, torch.full((batch_size, 1), t_id, device=device, dtype=torch.long)],
            dim=1
        )
        current_attn_mask = torch.cat(
            [
                current_attn_mask, 
                torch.ones((batch_size, 1), device=device, dtype=current_attn_mask.dtype)
            ], 
            dim=1
        )
    
    pred_tokens = torch.stack(pred_tokens, dim=1).squeeze(dim=-1)
    ls_pred_probs = torch.stack(ls_pred_probs, dim=1).squeeze(dim=-1)
    ls_ans_probs = torch.stack(ls_ans_probs, dim=1)

    # print(f"pred_tokens: {pred_tokens}")
    # print(f"ls_pred_probs: {ls_pred_probs}")
    # print(f"ls_ans_probs: {ls_ans_probs}")
    # raise

    # print(f"pred_tokens: {tokenizer.batch_decode(pred_tokens)}")
    # print(f"j_pred_probs: {j_pred_probs.shape}")
    # print(f"j_pred_probs: {j_pred_probs} -- {compute_joint_probs(j_pred_probs)}")
    # print(f"j_probs: {j_probs} -- {compute_joint_probs(j_probs)}")

    return dict(
        pred_tokens=pred_tokens,
        pred_tokens_str=tokenizer.batch_decode(pred_tokens),
        pred_probs=ls_pred_probs,
        j_pred_probs=compute_joint_probs(ls_pred_probs),
        ans_probs=ls_ans_probs,
        j_ans_probs=compute_joint_probs(ls_ans_probs),
        probs=ls_probs,
    )

@torch.no_grad()
def predict_with_cf_from_input(model, tokenizer, inp, answer_str, counterfactuals=None):
    output = predict_from_input(model, tokenizer, inp, answer_str)

    if not counterfactuals:
        return output

    cfs_probs = []
    j_cfs_probs = []
    for cf in counterfactuals:
        cf_probs = []
        cf_token_ids = tokenizer.encode(cf, add_special_tokens=False)

        for i, t_id in enumerate(cf_token_ids):
            if i > len(output["probs"]):
                break

            cf_probs.append(output["probs"][i][:, t_id])
        
        cf_probs = torch.stack(cf_probs, dim=1)
        cfs_probs.append(cf_probs)
        j_cfs_probs.append(compute_joint_probs(cf_probs))
    
    return dict(
        pred_tokens=output["pred_tokens"],
        pred_tokens_str=output["pred_tokens_str"],
        pred_probs=output["pred_probs"],
        j_pred_probs=output["j_pred_probs"],
        ans_probs=output["ans_probs"],
        j_ans_probs=output["j_ans_probs"],
        cfs_probs=cfs_probs,
        j_cfs_probs=j_cfs_probs,
    )