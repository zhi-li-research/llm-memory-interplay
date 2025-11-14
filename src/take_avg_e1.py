import os
import json
import glob
import copy
import pprint
import re
import ast

import bisect

import itertools
from collections import Counter
from tqdm import tqdm

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

import seaborn as sns
from scipy.stats import gaussian_kde
from scipy.stats import ttest_ind

from IPython.display import display, Latex

import torch

# from utils import read_json_file


class Avg:
    def __init__(self, size=12, name=None):
        self.d = []
        self.size = size
        self.name = name

    def add(self, v):
        if v.size > 0:
            self.d.append(v[None])

    def add_all(self, vv):
        if vv.size > 0:
            self.d.append(vv)

    def avg(self):
        if len(self.d) > 0:
            non_empty_arrays = [arr for arr in self.d if arr.size > 0]
            
            if len(non_empty_arrays) > 0:
                return np.concatenate(non_empty_arrays).mean(axis=0)
            else:
                return np.zeros(self.size)

        return np.zeros(self.size)

    def std(self):
        if len(self.d) > 0:
            non_empty_arrays = [arr for arr in self.d if arr.size > 0]
            
            if len(non_empty_arrays) > 0:
                return np.concatenate(non_empty_arrays).std(axis=0)
            else:
                return np.zeros(self.size)

        return np.zeros(self.size)

    def size(self):
        return sum(datum.shape[0] for datum in self.d)

    def humanize(self):
        return self.name.replace("_", " ").capitalize()

    def __repr__(self):
        return f"<Avg name={self.name}>"


def plot_array(
    differences,
    ax,
    labels,
    kind=None,
    savepdf=None,
    title=None,
    low_score=None,
    high_score=None,
    archname="Atlas",
    show_y_labels=True
):
    if low_score is None:
        low_score = differences.min()
    if high_score is None:
        high_score = differences.max()
        
    answer = "AIE"
    labels = labels

    # fig, ax = plt.subplots(figsize=(3.5, 2), dpi=300)

    h = ax.pcolor(
        differences,
        cmap={None: "Purples", "mlp": "Greens", "attn": "Reds"}[kind],
        vmin=low_score,
        vmax=high_score,
    )
    
    if title:
        ax.set_title(title)
        
    ax.invert_yaxis()
    ax.set_yticks([0.5 + i for i in range(len(differences))])
    # ax.set_xticks([0.5 + i for i in range(0, differences.shape[1] - 6, 5)])
    # ax.set_xticklabels(list(range(0, differences.shape[1] - 6, 5)))
    ax.set_xticks([0.5 + i for i in range(0, differences.shape[1])])
    ax.set_xticklabels(list(range(0, differences.shape[1])))
    ax.set_yticklabels(labels)

    if show_y_labels:
        ax.set_yticklabels(labels)
    else:
        ax.set_yticklabels([])

    if kind is None:
        ax.set_xlabel(f"single patched layer")
        # ax.set_xlabel(f"single patched layer within {archname}")
    else:
        ax.set_xlabel(f"center of interval of 6 patched {kind} layers")
        # ax.set_xlabel(f"center of interval of 6 patched {kind} layers within {archname}")

    cb = plt.colorbar(h)
    # The following should be cb.ax.set_xlabel(answer), but this is broken in matplotlib 3.5.1.
    if answer:
        # cb.ax.set_title(str(answer).strip(), y=-0.16, fontsize=10)
        cb.ax.set_title(str(answer).strip(), y=-0.16)

    if savepdf:
        os.makedirs(os.path.dirname(savepdf), exist_ok=True)
        plt.savefig(savepdf, bbox_inches="tight")
    

def is_valid_attrs(attrs):
    # Check if attrs is a list with exactly two items
    if not isinstance(attrs, list) or len(attrs) != 2:
        return False
    
    # Check if each item in the list is a tuple with exactly two integers
    for item in attrs:
        if not isinstance(item, tuple) or len(item) != 2:
            return False
        if not all(isinstance(i, int) for i in item):
            return False

    return True
    
        
def healthy_r(r):
    logs = []

    if "scores" not in r:
        msg = f"no `scores`"
        logs.append(msg)
        return False, logs
    
    if "status" not in r or not r["status"]:
        msg = f"`status == False`"
        logs.append(msg)
        return False, logs
    
    attribute_loc = list(sorted(list(itertools.chain.from_iterable(r["attributes_loc"].values()))))
    if not attribute_loc:
        msg = f"no `attribute_loc`"
        logs.append(msg)
        return False, logs

    
    if np.isnan(r["te_log"]) or np.isinf(r["te_log"]):
        msg = f"no `te_log`"
        logs.append(msg)
        return False, logs

    
    if np.isnan(r["te"]) or np.isinf(r["te"]):
        msg = f"no `te`"
        logs.append(msg)
        return False, logs

    return True, logs


def find_insert_position(tuples_list, target_tuple):
    # Extract the end values of each tuple in the list
    ends = [t[1] for t in tuples_list]
    
    # Find the position using bisect_right
    position = bisect.bisect_right(ends, target_tuple[0])
    
    return position

def tokens_space_division(data, experiment_type, num_layers=12, do_log=True):
    avg_effects = [
        "question",

        # "answer",
        # "answer_token",
        
        "begining_of_context",

        "first_subject_token",
        "middle_subject_tokens",
        "last_subject_token",

        "context_in_between_tokens",

        "first_object_token",
        "middle_object_tokens",
        "last_object_token",

        "rest_of_context_tokens",
        "last_token"
    ]
    avg_scores = ["high_score", "low_score", "te", "fixed_score"]
    avg = {name: Avg(size=1, name=name) for name in avg_scores}
    avg.update({name: Avg(size=num_layers, name=name) for name in avg_effects})

    result = np.array([])
    result_std = np.array([])

    for r in tqdm(data, total=len(data)):
        attribute_locs = list(sorted(list(itertools.chain.from_iterable(r["attributes_loc"].values()))))

        attribute_loc = attribute_locs[-1]
        start_of_answer = r["input_tokens"].index("▁answer")
        start_of_context = r["input_tokens"].index("<extra_id_0>") + 1
        start_of_attr, end_of_attr = attribute_loc
        end_of_prompt = len(r["input_tokens"])

        input_segments = prompt_segmenter([r["input_tokens"]])

        object_ranges = find_token_ranges(model.reader_tokenizer, r["input_ids"], r["cf"][0], bounds=input_segments[0]["context"])
        object_ranges = object_ranges[0] if isinstance(object_ranges, list) and len(object_ranges) > 0 else []

        subject_ranges = find_token_ranges(model.reader_tokenizer, r["input_ids"], r["prompt"]["subj"], bounds=input_segments[0]["context"])
        subject_ranges = subject_ranges[0] if isinstance(subject_ranges, list) and len(subject_ranges) > 0 else []

        if do_log:
            te, ie = r["te_log"], r["ie_log"]
        else:
            te, ie = r["te"], r["ie"]


        subject_pos = object_pos = 0
        if experiment_type == "c":
            try:
                subject_pos = find_insert_position(list(sorted(list(itertools.chain.from_iterable(r["attributes_loc"].values())))), subject_ranges)
                object_pos = find_insert_position(list(sorted(list(itertools.chain.from_iterable(r["attributes_loc"].values())))), object_ranges)
            except:
                continue

        
        if experiment_type == "a":
            attrs = [subject_ranges, attribute_loc]
        elif experiment_type == "b":
            attrs = [attribute_loc, object_ranges]
        elif experiment_type == "c":
            attrs = [subject_ranges, object_ranges]

        
        if not is_valid_attrs(attrs):
            continue


        avg["high_score"].add(np.array(r["cr_ans_score"]))
        avg["low_score"].add(np.array(r["crr_score"]))
        avg["te"].add(np.array(te))
        avg["fixed_score"].add(ie.max())

        avg["question"].add_all(ie[0:start_of_answer])
        # avg["answer"].add_all(ie[start_of_answer:start_of_answer+3])
        # avg["answer_token"].add(ie[start_of_answer+3])
        avg["last_token"].add(ie[-1])

        if experiment_type == "a":
            attrs = [subject_ranges, attribute_loc]
            first_attr = attrs.index(min(attrs))
            second_attr = 0 if first_attr == 1 else 1

            avg["begining_of_context"].add_all(ie[start_of_context+4:attrs[first_attr][0]])
            avg["context_in_between_tokens"].add_all(ie[attrs[first_attr][1]:attrs[second_attr][0]])
            avg["rest_of_context_tokens"].add_all(ie[attrs[second_attr][1]:end_of_prompt-1])

            avg["first_subject_token"].add(ie[subject_ranges[0]])
            avg["middle_subject_tokens"].add_all(ie[subject_ranges[0]+1:subject_ranges[1]-1])
            avg["last_subject_token"].add(ie[subject_ranges[1]-1])

            avg["first_object_token"].add(ie[attribute_loc[0]])
            avg["middle_object_tokens"].add_all(ie[attribute_loc[0]+1:attribute_loc[1]-1])
            avg["last_object_token"].add(ie[attribute_loc[1]-1])
        elif experiment_type == "b":
            attrs = [attribute_loc, object_ranges]
            first_attr = attrs.index(min(attrs))
            second_attr = 0 if first_attr == 1 else 1

            avg["begining_of_context"].add_all(ie[start_of_context+4:attrs[first_attr][0]])
            avg["context_in_between_tokens"].add_all(ie[attrs[first_attr][1]:attrs[second_attr][0]])
            avg["rest_of_context_tokens"].add_all(ie[attrs[second_attr][1]:end_of_prompt-1])
            
            avg["first_subject_token"].add(ie[attribute_loc[0]])
            avg["middle_subject_tokens"].add_all(ie[attribute_loc[0]+1:attribute_loc[1]-1])
            avg["last_subject_token"].add(ie[attribute_loc[1]])

            avg["first_object_token"].add(ie[object_ranges[0]])
            avg["middle_object_tokens"].add_all(ie[object_ranges[0]+1:object_ranges[1]-1])
            avg["last_object_token"].add(ie[object_ranges[1]-1])
        elif experiment_type == "c":
            attrs = [subject_ranges, object_ranges]
            first_attr = attrs.index(min(attrs))
            second_attr = 0 if first_attr == 1 else 1
            
            avg["begining_of_context"].add_all(ie[start_of_context+4:attrs[first_attr][0]])
            avg["context_in_between_tokens"].add_all(ie[attrs[first_attr][1]:attrs[second_attr][0]])
            avg["rest_of_context_tokens"].add_all(ie[attrs[second_attr][1]:end_of_prompt-1])

            avg["first_subject_token"].add(ie[subject_ranges[0]])
            avg["middle_subject_tokens"].add_all(ie[subject_ranges[0]+1:subject_ranges[1]-1])
            avg["last_subject_token"].add(ie[subject_ranges[1]-1])

            avg["first_object_token"].add(ie[object_ranges[0]])
            avg["middle_object_tokens"].add_all(ie[object_ranges[0]+1:object_ranges[1]-1])
            avg["last_object_token"].add(ie[object_ranges[1]-1])

        
        result = [avg[name].avg() for name in avg_effects]
        result_std = [avg[name].std() for name in avg_effects]
    
    print_out = [
        {"METRIC": "Average Total Effect", "VALUE": avg["te"].avg()},
    ]

    return {
        "high_score": avg["high_score"].avg(),
        "low_score": avg["low_score"].avg(),
        "labels": [avg[name].humanize() for name in avg_effects],
        "result": result,
        "result_std": result_std,
        "size": num_layers,
        "print_out": print_out,
        "avg": avg,
    }


def safe_division(a, b, do_log=False, threshold=None):
    threshold = threshold if threshold else np.finfo(np.float32).max/10.0
    return torch.clamp(a/b, max=threshold) if not do_log else (torch.log(a) - torch.log(b))


def calculate_te_ie(r, samples, experiment_type="a", do_log=True, threshold=1e-40):
    cr_cf_score = torch.clamp(r["cr_cf_score"], min=threshold)
    cr_ans_score = torch.clamp(r["cr_ans_score"], min=threshold)

    crr_cf_score = torch.clamp(r["crr_cf_score"], min=threshold)
    crr_ans_score = torch.clamp(r["crr_ans_score"], min=threshold)

    crwrr_cf_score = torch.clamp(r["crwrr_cf_score"], min=threshold)
    crwrr_ans_score = torch.clamp(r["crwrr_ans_score"], min=threshold)
    

    te, ie = [], []

    for i in range(samples):
        if experiment_type == "a" or experiment_type == "aa":
            te_i = safe_division(crr_cf_score[i], crr_ans_score[i], do_log=do_log) - \
                   safe_division(cr_cf_score[i], cr_ans_score, do_log=do_log)
            ie_i = safe_division(crwrr_cf_score[:, :, i], crwrr_ans_score[:, :, i], do_log=do_log) - \
                   safe_division(cr_cf_score[i], cr_ans_score, do_log=do_log)
        else:
            te_i = safe_division(cr_cf_score[i], cr_ans_score, do_log=do_log) - \
                   safe_division(crr_cf_score[i], crr_ans_score[i], do_log=do_log)
            ie_i = safe_division(crwrr_cf_score[:, :, i], crwrr_ans_score[:, :, i], do_log=do_log) - \
                   safe_division(crr_cf_score[i], crr_ans_score[i], do_log=do_log)

                

        te.append(te_i)
        ie.append(ie_i.unsqueeze(-1))

    te = torch.stack(te).mean()
    ie = torch.cat(ie, axis=-1).mean(-1)

    return te, ie


def calculate_post_proc(r, samples, experiment_type="a"):
    r = {k: torch.from_numpy(v) if isinstance(v, np.ndarray) else v for k, v in r.items()}
    
    if r["status"]:
        te_log, ie_log = calculate_te_ie(r, samples=samples, experiment_type=experiment_type, do_log=True)
        r["te_log"] = te_log
        r["ie_log"] = ie_log

        te, ie = calculate_te_ie(r, samples=samples, experiment_type=experiment_type, do_log=False)
        r["te"] = te
        r["ie"] = ie
    else:
        r["te_log"] = None
        r["ie_log"] = None
        r["te"] = None
        r["ie"] = None

    r = {k: v.detach().cpu().numpy() if torch.is_tensor(v) else v for k, v in r.items()}

    return r


def read_data(rows, experiment_type="a", kind=None):
    data = {}

    for i, row in tqdm(enumerate(rows), total=len(rows)):
        if row["topic"] not in data:
            data[row["topic"]] = []

        for _id, path in zip(row[kind]["ids"], row[kind]["paths"]):
            if _id in row["unique_ids"]:
                r = torch.load(path)
                r = calculate_post_proc(r, samples=6, experiment_type=experiment_type)

                data[row["topic"]].append(r)

    return data

raw_a_data = {
    kind: retrieve_behaviors(read_data(adata, experiment_type="a", kind=kind), threshold=-1, pnp_threshold=6)
    for kind in [None, "mlp", "attn"]
}
print()
raw_b_data = {
    kind: retrieve_behaviors(read_data(bdata, experiment_type="b", kind=kind), threshold=-1, pnp_threshold=6)
    for kind in [None, "mlp", "attn"]
}
print()
raw_c_data = {
    kind: retrieve_behaviors(read_data(cdata, experiment_type="c", kind=kind), threshold=-1, pnp_threshold=6)
    for kind in [None, "mlp", "attn"]
}

def find_in_common(lists):
    if not lists:
        return []
    
    # Initialize the common set with the first list's elements
    common_set = set(lists[0])
    
    # Iterate through the remaining lists and perform intersection
    for lst in lists[1:]:
        common_set.intersection_update(lst)
        
    return list(common_set)
    
def extract_fname_from_path(path):
    fname = path.split("/")[-1].split(".")[0]
    fname = fname.replace("_mlp", "").replace("_attention", "")
    return fname

def experiment_selection(paths, experiment_type):
    known_ids = []
    known_paths = {}
    logs = []
    

    for i, path in enumerate(paths):
        known_id = extract_fname_from_path(path)

        try:
            r = torch.load(path, map_location='cpu')
            r = calculate_post_proc(r, samples=6, experiment_type=experiment_type)
            
            status, logs = healthy_r(r)
            if not status:
                logs.append({"path": path, "logs": log})
            else:
                known_ids.append(known_id)
                known_paths[known_id] = path

        except:
            logs.append({"path": path, "logs": ["problem with loading"]})
            continue

    return known_ids, known_paths, logs
    
def data_prep(cases_directory, experiment_type):
    candidates_ids = {}
    candidates_paths = {}
    candidates = []
    for directory in tqdm(cases_directory, desc="Data preparation ..."):
        if not os.path.exists(directory):
            continue
        
        if directory not in candidates_ids:
            candidates_ids[directory] = {}
        
        if directory not in candidates_paths:
            candidates_paths[directory] = {}

        do_loop = True
        for kind in [None, "mlp", "attention"]:
            if not do_loop:
                continue

            fpaths = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
            if kind is None:  
                fpaths = [f for f in fpaths if not any(k in f for k in ["_mlp", "_attention"])]
            else:
                fpaths = [f for f in fpaths if f"_{kind}" in f]

            fpaths = [os.path.join(directory, path) for path in sorted(fpaths)]

            known_ids, known_paths, logs = experiment_selection(fpaths, experiment_type=experiment_type)
            # print(f"logs: {logs}")

            if len(known_ids) == 0:
                do_loop = False
                continue

            candidates_ids[directory][kind] = known_ids
            candidates_paths[directory][kind] = known_paths


        unique_ids = sorted(find_in_common(list(candidates_ids[directory].values())), key=lambda x: int(x.split("_")[1]))

        for kind in [None, "mlp", "attention"]:
            candidates_ids[directory][kind] = list(filter(lambda x: x in unique_ids, candidates_ids[directory][kind]))
            candidates_paths[directory][kind] = [v for k, v  in candidates_paths[directory][kind].items() if k in unique_ids]
        
        candidates.append({
            None: {
                "ids": candidates_ids[directory][None],
                "paths": candidates_paths[directory][None],
            },
            "mlp": {
                "ids": candidates_ids[directory]["mlp"],
                "paths": candidates_paths[directory]["mlp"],
            },
            "attention": {
                "ids": candidates_ids[directory]["attention"],
                "paths": candidates_paths[directory]["attention"],
            },
            "topic": f'{directory.split("/")[2]}/{directory.split("/")[3]}/{directory.split("/")[6]}',
            "unique_ids": unique_ids,
            "unique_len": len(unique_ids),
        })


    min_cand, max_cand = min(list(map(lambda x: x['unique_len'], candidates))), max(list(map(lambda x: x['unique_len'], candidates)))
    topic_cand = [f"{cand['topic']} -- {cand['unique_len']}" for cand in candidates]
    display(topic_cand)
    display(f"  min_cand: {min_cand}")
    display(f"  max_cand: {max_cand}")

    
    return candidates

experiment_type = "e1"
cases_directory = [
    # f"../experiments/ct/llama/popqa/{experiment_type}/capital/cases",
    # f"../experiments/ct/llama/popqa/{experiment_type}/capital_of/cases",
    # f"../experiments/ct/llama/popqa/{experiment_type}/color/cases",
    # f"../experiments/ct/llama/popqa/{experiment_type}/composer/cases",
    # f"../experiments/ct/llama/popqa/{experiment_type}/country/cases",
    # f"../experiments/ct/llama/popqa/{experiment_type}/father/cases",
    # f"../experiments/ct/llama/popqa/{experiment_type}/genre/cases",
    # f"../experiments/ct/llama/popqa/{experiment_type}/occupation/cases",
    # f"../experiments/ct/llama/popqa/{experiment_type}/place_of_birth/cases",
    # f"../experiments/ct/llama/popqa/{experiment_type}/religion/cases",
    # f"../experiments/ct/llama/popqa/{experiment_type}/sport/cases",

    # f"../experiments/ct/llama/peq/{experiment_type}/P17/cases",
    # f"../experiments/ct/llama/peq/{experiment_type}/P19/cases",
    # f"../experiments/ct/llama/peq/{experiment_type}/P20/cases",
    # f"../experiments/ct/llama/peq/{experiment_type}/P36/cases",
    # f"../experiments/ct/llama/peq/{experiment_type}/P69/cases",
    # f"../experiments/ct/llama/peq/{experiment_type}/P106/cases",
    # f"../experiments/ct/llama/peq/{experiment_type}/P127/cases",
    # f"../experiments/ct/llama/peq/{experiment_type}/P131/cases",
    # f"../experiments/ct/llama/peq/{experiment_type}/P159/cases",
    f"../experiments/ct/llama/peq/{experiment_type}/P175/cases",
    f"../experiments/ct/llama/peq/{experiment_type}/P176/cases",
    # f"../experiments/ct/llama/peq/{experiment_type}/P276/cases",
    # f"../experiments/ct/llama/peq/{experiment_type}/P407/cases",
    # f"../experiments/ct/llama/peq/{experiment_type}/P413/cases",
    # f"../experiments/ct/llama/peq/{experiment_type}/P495/cases",
    # f"../experiments/ct/llama/peq/{experiment_type}/P740/cases",
]
adata = data_prep(cases_directory, experiment_type=experiment_type)
print()
