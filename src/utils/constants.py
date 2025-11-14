import torch

DEFAULT_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEFAULT_PROMPT_FORMAT = "{prompt}"
QA_T1_PROMPT_FORMAT = "C: {context} Q: {prompt} A:"
QA_T2_PROMPT_FORMAT = "Q: {prompt} C: {context} A:"