import argparse
import copy
import itertools
import json
import os
import pprint
import random
import re
import sys

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from causal_tracing.causal_tracer import CausalTracer
from causal_tracing.metrics import calculate_te_ie
from causal_tracing.plot_hidden_flow_heatmap import plot_hidden_flow_heatmap
from utils.constants import DEFAULT_DEVICE
from utils.torch_utils import set_requires_grad
from utils.utils import read_json_file


def main():
    parser = argparse.ArgumentParser(description="Causal Tracer")
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--experiment_type", type=int, choices=[1, 2, 3], default=1, help="Experiment type")
    parser.add_argument(
        "--data_path",
        help="A space-separated path to jsonl-formatted evaluation sets",
    )
    parser.add_argument("--output_dir")
    parser.add_argument("--reverse_patching", type=int, choices=[0, 1], default=0, help="reverse patching to cover the restoration")
    parser.add_argument("--replace", type=int, choices=[0, 1], default=0, help="replace parameter as bool")
    parser.add_argument(
        "--prompt_format",
        type=str,
        default="question: {question} answer: <extra_id_0>",
        help="How to format question as input prompts when using --task qa",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=10,
        help="num counterfactual samples",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=10,
        help="num window for restoration section",
    )
    parser.add_argument(
        "--max_datapoints",
        type=int,
        default=0,
        help="maximum number of recoreds",
    )
    parser.add_argument(
        "--max_cf",
        type=int,
        default=3,
        help="maximum number of counterfactuals",
    )
    
    args = parser.parse_args()
    
    print("ARGS:", '---' * 10)
    pprint.pp(args.__dict__)
    print("/", '---' * 10, "\n")

    output_dir = args.output_dir
    result_dir = f"{output_dir}/cases"
    pdf_dir = f"{output_dir}/pdfs"
    os.makedirs(result_dir, exist_ok=True)
    os.makedirs(pdf_dir, exist_ok=True)


    tokenizer = AutoTokenizer.from_pretrained(args.model_name, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(args.model_name, low_cpu_mem_usage=True, local_files_only=True).to(DEFAULT_DEVICE)
    set_requires_grad(False, model)
    tracer = CausalTracer(model, tokenizer, device=DEFAULT_DEVICE)

    for i, row in enumerate(read_json_file(args.data_path, jsonl=True)):
        if args.max_datapoints > 0 and i > args.max_datapoints:
            print(f"More than specified `{args.max_datapoints}` datapoints!")
            break
            
        for kind in "hidden", "mlp", "attention":
            
            if args.experiment_type == 1:
                obj_safe = row["obj"].replace(" ", "-").replace("_", "-")
                filename = f"{result_dir}/{i}_{kind}_{obj_safe}.bin"
                
                if not os.path.isfile(filename):
                    flow = tracer.calculate_hidden_flow(
                        prompt=row["prompt_wo_ctx"], 
                        prompt_attrs={"context": row["passages"][0]["text"]}, 
                        prompt_format=args.prompt_format,
                        target_str=row["obj"],
                        target_noise=row["obj_cf"][:args.max_cf],
                        target_position="context",
                        counterfactuals=row["obj_cf"][:args.max_cf],
                        answer_str=row["answers"][0],
                        replace=bool(args.replace),
                        reverse_patching=bool(args.reverse_patching),
                        kind=kind, 
                        window=args.window, 
                        samples=args.samples,
                        queue_id=i
                    )
                    flow = calculate_te_ie(flow, samples=args.samples, reverse_patching=args.reverse_patching, do_log=True)
                    torch.save(flow, filename)
                else:
                    flow = torch.load(filename)
                

                if i > 100:
                    continue
                
                _ = plot_hidden_flow_heatmap(
                    torch.clamp(flow.ie, min=0), 
                    labels=flow.input_tokens, 
                    target_ranges=flow.target_ranges,
                    kind=kind,
                    savepdf=f'{pdf_dir}/{str(row["answers"][0]).strip()}_{i}_{kind}.pdf'
                )
            
            elif args.experiment_type == 2:
                for obj in row["obj_cf"][:args.max_cf]:
                    obj_safe = obj.replace(" ", "-").replace("_", "-")
                    filename = f"{result_dir}/{i}_{kind}_{obj_safe}.bin"

                    prompt_attrs = {"context": row["passages"][0]["text"]}
                    prompt_attrs["context"] = prompt_attrs["context"].replace(row["obj"], obj)

                    if not os.path.isfile(filename):
                        flow = tracer.calculate_hidden_flow(
                            prompt=row["prompt_wo_ctx"], 
                            prompt_attrs=prompt_attrs, 
                            prompt_format=args.prompt_format,
                            target_str=row["subj"],
                            target_noise=None,
                            target_position="context",
                            counterfactuals=[obj],
                            answer_str=row["answers"][0],
                            replace=bool(args.replace),
                            reverse_patching=bool(args.reverse_patching),
                            kind=kind, 
                            window=args.window, 
                            samples=args.samples,
                            queue_id=i
                        )
                        flow = calculate_te_ie(flow, samples=args.samples, reverse_patching=args.reverse_patching, do_log=True)
                        torch.save(flow, filename)
                    else:
                        flow = torch.load(filename)
                    

                    if i > 100:
                        continue
                    
                    _ = plot_hidden_flow_heatmap(
                        torch.clamp(flow.ie, min=0), 
                        labels=flow.input_tokens, 
                        target_ranges=flow.target_ranges,
                        kind=kind,
                        savepdf=f'{pdf_dir}/{str(row["answers"][0]).strip()}_{i}_{kind}_{obj_safe}.pdf'
                    )

            elif args.experiment_type == 3:
                for obj in row["obj_cf"][:args.max_cf]:
                    obj_safe = obj.replace(" ", "-").replace("_", "-")
                    filename = f"{result_dir}/{i}_{kind}_{obj_safe}.bin"

                    prompt_attrs = {"context": row["passages"][0]["text"]}
                    prompt_attrs["context"] = prompt_attrs["context"].replace(row["obj"], obj)

                    if not os.path.isfile(filename):
                        flow = tracer.calculate_hidden_flow(
                            prompt=row["prompt_wo_ctx"], 
                            prompt_attrs=prompt_attrs, 
                            prompt_format=args.prompt_format,
                            target_str=f'{row["subj"]}+{obj}',
                            target_noise=None,
                            target_position="context",
                            counterfactuals=[obj],
                            answer_str=row["answers"][0],
                            replace=bool(args.replace),
                            reverse_patching=bool(args.reverse_patching),
                            kind=kind, 
                            window=args.window, 
                            samples=args.samples,
                            queue_id=i
                        )
                        flow = calculate_te_ie(flow, samples=args.samples, reverse_patching=args.reverse_patching, do_log=True)
                        torch.save(flow, filename)
                    else:
                        flow = torch.load(filename)
                    
                    
                    if i > 100:
                        continue
                    
                    _ = plot_hidden_flow_heatmap(
                        torch.clamp(flow.ie, min=0), 
                        labels=flow.input_tokens, 
                        target_ranges=flow.target_ranges,
                        kind=kind,
                        savepdf=f'{pdf_dir}/{str(row["answers"][0]).strip()}_{i}_{kind}_{obj_safe}.pdf'
                    )
    

if __name__ == "__main__":
    main()