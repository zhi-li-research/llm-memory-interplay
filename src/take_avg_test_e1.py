# E1
from itertools import chain
import os
import torch
import numpy as np
from itertools import chain
from causal_tracing.plot_hidden_flow_heatmap import plot_hidden_flow_heatmap


def get_bin_files_by_keyword(directory, keyword):
    """
    Return all .bin files in the given directory that contain the specified keyword.
    """
    bin_files = [
        os.path.join(directory, f)
        for f in os.listdir(directory)
        if os.path.isfile(os.path.join(directory, f)) and f.endswith('.bin') and keyword in f
    ]
    return bin_files
    

def segment_tokens(flow):
    """
    Segment tokens and IE data inside a flow object.

    Args:
        flow (object): The flow object loaded from .bin file.

    Returns:
        dict: A dictionary mapping segment names to the averaged IE vectors.
    """
    tokens = flow.input_tokens
    target_ranges = flow.target_ranges

    # If target_ranges is a list with one tuple, unpack it
    if isinstance(target_ranges, list) and len(target_ranges) == 1:
        target_ranges = target_ranges[0]

    # Validate target_ranges
    if not isinstance(target_ranges, (tuple, list)) or len(target_ranges) != 2:
        raise ValueError(f"Invalid target_ranges: {target_ranges}")

    obj_start, obj_end = target_ranges
    
    # Convert IE tensor to NumPy
    ie_dim = flow.ie.cpu().numpy()

    # Original segmentation logic
    segments = {
        "beginning_of_context": ie_dim[:obj_start, :],
        "first_object_token": ie_dim[obj_start:obj_start + 1, :],
        "middle_object_tokens": ie_dim[obj_start + 1:obj_end - 1, :] if obj_end - obj_start > 1 else np.zeros((0, ie_dim.shape[1])),
        "last_object_token": ie_dim[obj_end - 1:obj_end, :] if obj_end - obj_start > 1 else np.zeros((0, ie_dim.shape[1])),
        "context_in_between_tokens": ie_dim[obj_end:obj_end + 4, :],
        "first_subject_token": ie_dim[obj_end + 4:obj_end + 5, :],
        "middle_subject_token": ie_dim[obj_end + 5:obj_end + 8, :],
        "last_subject_token": ie_dim[obj_end + 8:obj_end + 9, :],
        # "rest_of_context_tokens": ie_dim[obj_end + 9:-2, :],
        # "question": ie_dim[-2:-1, :],
        "rest_of_context_tokens": ie_dim[obj_end + 9:obj_end + 9 + 1, :],  # Two tokens after the subject
        "question": ie_dim[obj_end + 9 + 1:-1, :],
        "last_token": ie_dim[-1:, :],
    }

    # Compute mean for each segment (ensure uniform dimensionality)
    segment_avg = {key: np.mean(value, axis=0) if value.size > 0 else np.zeros(ie_dim.shape[1]) for key, value in segments.items()}
    return segment_avg


def process_bin_files(bin_files, target_dim=(11, 32)):
    """
    Segment, reshape, and compute the average IE across all provided .bin files.

    Args:
        bin_files (list of str): Paths to .bin files.
        target_dim (tuple): Expected final shape of IE representation.

    Returns:
        dict: Contains averaged IE and segment names.
    """
    all_segmented_data = []
    segment_names = None

    for bin_file in bin_files:
        flow = torch.load(bin_file)
        segmented_data = segment_tokens(flow)

        # Record segment names during first iteration
        if segment_names is None:
            segment_names = list(segmented_data.keys())

        # Collect segmented vectors
        all_segmented_data.append(np.array(list(segmented_data.values())))

    # Stack over all files and take mean per segment
    all_segmented_data = np.stack(all_segmented_data, axis=0)
    avg_ie = np.mean(all_segmented_data, axis=0)

    return {"avg_ie": avg_ie, "segments": segment_names}


def collect_bin_files(cases_directories, keywords):
    """
    Collect .bin files across multiple directories, grouped by keyword.

    Args:
        cases_directories (list of str): List of directories to search.
        keywords (list of str): List of keywords.

    Returns:
        dict: Mapping from keyword to list of matched .bin file paths.
    """
    keyword_files = {key: [] for key in keywords}

    for directory in cases_directories:
        for keyword in keywords:
            bin_files = get_bin_files_by_keyword(directory, keyword)
            keyword_files[keyword].extend(bin_files)

    return keyword_files


import os

def generate_heatmap(avg_ie, segments, kind, savepdf):
    """
    Generate a heatmap from average IE data and save as a PDF.

    Args:
        avg_ie (np.ndarray): Averaged IE data (11 x 32).
        segments (list): Segment names (labels for Y-axis).
        kind (str): Type of plot ("hidden", "attention", "mlp").
        savepdf (str): Output PDF path.
    """
    # Create output directory if missing
    os.makedirs(os.path.dirname(savepdf), exist_ok=True)

    _ = plot_hidden_flow_heatmap(
        torch.clamp(torch.tensor(avg_ie, dtype=torch.float32), min=0),
        labels=segments,
        kind=kind,
        savepdf=savepdf
    )

    print(f"Heatmap saved to: {savepdf}")


cases_directories = [
    "../experiments/ct/llama/popqa/e1/screenwriter/cases",
    "../experiments/ct/llama/popqa/e1/sport/cases",
]

keywords = ["_hidden_", "_attention_", "_mlp_"]

output_filenames = {
    "_hidden_": "results/heatmaps/hidden_avg_heatmap_e1_test_1.pdf",
    "_attention_": "results/heatmaps/attention_avg_heatmap_e1_test_1.pdf",
    "_mlp_": "results/heatmaps/mlp_avg_heatmap_e1_test_1.pdf",
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
        print(f"Didn't find bin files including {kind}")
