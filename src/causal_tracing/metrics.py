import math

import numpy as np
import torch


def safe_division(a, b, do_log=False, threshold=None):
    threshold = threshold if threshold else np.finfo(np.float32).max/10.0
    return torch.clamp(a/b, max=threshold) if not do_log else (torch.log(a) - torch.log(b))


def calculate_te_ie(flow, samples, reverse_patching=False, do_log=True, threshold=1e-40):
    cr_cf_score = torch.clamp(flow.clean_run["cf_scores"], min=threshold)
    cr_ans_score = torch.clamp(flow.clean_run["ground_scores"], min=threshold)

    crr_cf_score = torch.clamp(flow.corrupted_run["cf_scores"], min=threshold)
    crr_ans_score = torch.clamp(flow.corrupted_run["ground_scores"], min=threshold)

    crwrr_cf_score = torch.clamp(flow.corrupted_with_restoration_run["cf_scores"], min=threshold)
    crwrr_ans_score = torch.clamp(flow.corrupted_with_restoration_run["ground_scores"], min=threshold)

    te, ie = [], []

    for i in range(samples):
        if reverse_patching:
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


    flow.te = te
    flow.ie = ie

    return flow