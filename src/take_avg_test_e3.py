import os
import torch
import numpy as np
from itertools import chain
from causal_tracing.plot_hidden_flow_heatmap import plot_hidden_flow_heatmap


def get_bin_files_by_keyword(directory, keyword):
    bin_files = [
        os.path.join(directory, f)
        for f in os.listdir(directory)
        if os.path.isfile(os.path.join(directory, f)) and f.endswith('.bin') and keyword in f
    ]
    return bin_files

def segment_tokens(flow):
    tokens = flow.input_tokens
    ie_dim = flow.ie.cpu().numpy() 

    target_ranges = flow.target_ranges
    # if isinstance(target_ranges, list) and len(target_ranges) == 1:
    #     target_ranges = target_ranges[0]
    # if not isinstance(target_ranges, (tuple, list)) or len(target_ranges) != 2:
    #     raise ValueError(f"Invalid target_ranges: {target_ranges}")
    
    total_indices = set(range(len(tokens)))

    for start, end in target_ranges:
        total_indices -= set(range(start, end))
  
    remaining_indices = sorted(total_indices)

    if len(remaining_indices) < 3:
        raise ValueError("Insufficient tokens for defining object and subject ranges.")

    segments = []
    current_segment = [remaining_indices[0]]

    for i in range(1, len(remaining_indices)):
        if remaining_indices[i] == remaining_indices[i - 1] + 1:
            current_segment.append(remaining_indices[i])
        else:
            segments.append(current_segment)
            current_segment = [remaining_indices[i]]
    segments.append(current_segment)

    if len(segments) < 2:
        raise ValueError("Could not find enough segments to define object and subject.")

    object_segment = segments[0]
    subject_segment = segments[1]

    obj_start = object_segment[0]
    obj_end = object_segment[-1] 
    subject_start = subject_segment[0]
    subject_end = subject_segment[-1]

    # print(f"Object range: {obj_start}-{obj_end}")
    # print(f"Subject range: {subject_start}-{subject_end}")

    # segments = {
    #     "beginning_of_context": ie_dim[:obj_start, :],
    #     "first_object_token": ie_dim[obj_start:obj_start + 1, :],
    #     "middle_object_tokens": ie_dim[obj_start + 1:obj_end - 1, :] if obj_end - obj_start > 1 else np.zeros((0, ie_dim.shape[1])),
    #     "last_object_token": ie_dim[obj_end - 1:obj_end, :] if obj_end - obj_start > 1 else np.zeros((0, ie_dim.shape[1])),
    #     "context_in_between_tokens": ie_dim[obj_end:obj_end + 4, :],
    #     "first_subject_token": ie_dim[obj_end + 4:obj_end + 5, :],
    #     "middle_subject_token": ie_dim[obj_end + 5:obj_end + 8, :],
    #     "last_subject_token": ie_dim[obj_end + 8:obj_end + 9, :],
    #     "rest_of_context_tokens": ie_dim[obj_end + 9:-2, :],
    #     "question": ie_dim[-2:-1, :],
    #     "last_token": ie_dim[-1:, :],
    # }

    segments = {
    "beginning_of_context": ie_dim[:obj_start, :],  # object 前的上下文
    "first_object_token": ie_dim[obj_start:obj_start + 1, :],  # object 的第一个 token
    "middle_object_tokens": ie_dim[obj_start + 1:obj_end - 1, :] if obj_end - obj_start > 1 else np.zeros((0, ie_dim.shape[1])),
    "last_object_token": ie_dim[obj_end - 1:obj_end, :] if obj_end - obj_start > 1 else np.zeros((0, ie_dim.shape[1])),
    "context_in_between_tokens": ie_dim[obj_end:subject_start, :],  # object 和 subject 之间的上下文
    "first_subject_token": ie_dim[subject_start:subject_start + 1, :],  # subject 的第一个 token
    "middle_subject_token": ie_dim[subject_start + 1:subject_end - 1, :] if subject_end - subject_start > 1 else np.zeros((0, ie_dim.shape[1])),
    "last_subject_token": ie_dim[subject_end - 1:subject_end, :] if subject_end - subject_start > 1 else np.zeros((0, ie_dim.shape[1])),
    "rest_of_context_tokens": ie_dim[subject_end:subject_end + 1, :],  # subject 结束后两个 token
    "question": ie_dim[subject_end + 1:-1, :],
    "last_token": ie_dim[-1:, :],  # 最后一个 token
}

    segment_avg = {
        key: np.mean(value, axis=0) if value.size > 0 else np.zeros(ie_dim.shape[1])
        for key, value in segments.items()
    }

    return segment_avg

def process_bin_files(bin_files, target_dim=(11, 32)):
    all_segmented_data = []  
    segment_names = None

    for bin_file in bin_files:
        flow = torch.load(bin_file)
        # print(flow.ie)

        # print(flow.__dict__)

        # print(dir(flow))
        # print(flow)

        segmented_data = segment_tokens(flow)

        if segment_names is None:
            segment_names = list(segmented_data.keys())

        all_segmented_data.append(np.array(list(segmented_data.values())))

    all_segmented_data = np.stack(all_segmented_data, axis=0)
    avg_ie = np.mean(all_segmented_data, axis=0) 

    return {"avg_ie": avg_ie, "segments": segment_names}


def generate_heatmap(avg_ie, segments, kind, savepdf):

    target_ranges = [(2, 4)] 
    _ = plot_hidden_flow_heatmap(
        torch.clamp(torch.tensor(avg_ie, dtype=torch.float32), min=0),  
        labels=segments, 
        target_ranges=target_ranges,
        kind=kind,
        savepdf=savepdf
    )
    print(f"Heatmap saved to: {savepdf}")


def collect_bin_files(cases_directories, keywords):
    keyword_files = {key: [] for key in keywords}

    for directory in cases_directories:
        for keyword in keywords:
            bin_files = get_bin_files_by_keyword(directory, keyword)
            keyword_files[keyword].extend(bin_files)

    return keyword_files

cases_directories = [
    # "../experiments/ct/llama/peq/e1/P17/cases",
    # "../experiments/ct/llama/peq/e1/P19/cases",
    # "../experiments/ct/llama/peq/e1/P20/cases",
    # "../experiments/ct/llama/peq/e1/P26/cases",
    # # "../experiments/ct/llama/peq/e1/P36/cases",
    # "../experiments/ct/llama/peq/e1/P40/cases",
    # "../experiments/ct/llama/peq/e1/P50/cases",
    # "../experiments/ct/llama/peq/e1/P69/cases",
    # "../experiments/ct/llama/peq/e1/P106/cases",
    # "../experiments/ct/llama/peq/e1/P112/cases",
    # "../experiments/ct/llama/peq/e1/P127/cases",
    # "../experiments/ct/llama/peq/e1/P131/cases",
    # "../experiments/ct/llama/peq/e1/P20/cases",
    # "../experiments/ct/llama/peq/e1/P159/cases",
    # "../experiments/ct/llama/peq/e1/P170/cases",
    # "../experiments/ct/llama/peq/e1/P175/cases",
    # "../experiments/ct/llama/peq/e1/P176/cases",
    # "../experiments/ct/llama/peq/e1/P264/cases",
    # "../experiments/ct/llama/peq/e1/P276/cases",
    # "../experiments/ct/llama/peq/e1/P407/cases",
    # "../experiments/ct/llama/peq/e1/P413/cases",
    # "../experiments/ct/llama/peq/e1/P495/cases",
    # "../experiments/ct/llama/peq/e1/P740/cases",
    # "../experiments/ct/llama/peq/e1/P800/cases",
    # "../experiments/ct/llama/popqa/e1/author/cases",
    # "../experiments/ct/llama/popqa/e1/capital_of/cases",
    # "../experiments/ct/llama/popqa/e1/capital/cases",
    # "../experiments/ct/llama/popqa/e1/color/cases",
    # "../experiments/ct/llama/popqa/e1/composer/cases",
    # "../experiments/ct/llama/popqa/e1/country/cases",
    # "../experiments/ct/llama/popqa/e1/director/cases",
    # "../experiments/ct/llama/popqa/e1/father/cases",
    # "../experiments/ct/llama/popqa/e1/genre/cases",
    # "../experiments/ct/llama/popqa/e1/mother/cases",
    # "../experiments/ct/llama/popqa/e1/occupation/cases",
    # "../experiments/ct/llama/popqa/e1/place_of_birth/cases",
    # "../experiments/ct/llama/popqa/e1/producer/cases",
    # "../experiments/ct/llama/popqa/e1/religion/cases",
    "../experiments/ct/llama/popqa/e22/screenwriter/cases",
    "../experiments/ct/llama/popqa/e22/sport/cases",
]
keywords = ["_hidden_", "_attention_", "_mlp_"]
output_filenames = {
    "_hidden_": "results/heatmaps/hidden_avg_heatmap_test_e3_1.pdf",
    "_attention_": "results/heatmaps/attention_avg_heatmap_test_e3_1.pdf",
    "_mlp_": "results/heatmaps/mlp_avg_heatmap_test_e3_1.pdf",
}
kind_mapping = {
    "_hidden_": "hidden",
    "_attention_": "attention",
    "_mlp_": "mlp"
}

keyword_files = collect_bin_files(cases_directories, keywords)

for kind in keywords:
    bin_files = keyword_files[kind]
    if bin_files:
        result = process_bin_files(bin_files)
        avg_ie = result["avg_ie"]
        segments = result["segments"]
        savepdf = output_filenames[kind]
        mapped_kind = kind_mapping[kind] 
        generate_heatmap(avg_ie, segments, mapped_kind, savepdf)
    else:
        print(f"Didn't find bin files included {kind}")

