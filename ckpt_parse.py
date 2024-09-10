import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
import torch
import numpy as np

import argparse
import json
import os

import torch

import models
from datasets.kg_dataset import KGDataset
from utils.train import avg_both, format_metrics

from matplotlib import pyplot as plt
import torch.nn.functional as F

def test(model_dir):
    # load config
    with open(os.path.join(model_dir, "config.json"), "r") as f:
        config = json.load(f)
    args = argparse.Namespace(**config)

    # create dataset
    dataset_path = os.path.join("/home/ruizhou/DVKGE/data", args.dataset)
    dataset = KGDataset(dataset_path, False)
    test_examples = dataset.get_examples("test")
    filters = dataset.get_filters()

    # load pretrained model weights
    model:models.DVKGE = getattr(models, args.model)(args)
    device = 'cuda'
    model.to(device)
    model.load_state_dict(torch.load(os.path.join(model_dir, 'model.pt')))
    model.set_eval_hyper_dist_n_space_idx(None)

    # eval
    test_metrics = avg_both(*model.compute_metrics(test_examples, filters))
    return test_metrics, model, test_examples, filters


model_dir = "/home/ruizhou/DVKGE/logs/03_10/WN18RR/DVKGE_13_41_18"
test_metrics, model, test_examples, filters = test(model_dir)
print(format_metrics(test_metrics, split='test'))

# for i in range(model.n_space):
#     model.set_eval_hyper_dist_n_space_idx(i)
#     test_metrics = avg_both(*model.compute_metrics(test_examples, filters))
#     print(f"space id: {i} | {format_metrics(test_metrics, split='test')}")

model.set_eval_hyper_dist_n_space_idx(None)
for rel_i in test_examples[:, 1].unique():
    rel_i_test_examples = test_examples[test_examples[:, 1] == rel_i]
    test_metrics = avg_both(*model.compute_metrics(rel_i_test_examples.cuda(), filters))
    print(f"rel id: {rel_i} | num: {rel_i_test_examples.shape[0]} | MRR: {test_metrics['MRR']}")
    
# mrr = []
# for i in range(model.n_space):
#     model.set_eval_hyper_dist_n_space_idx(i)
#     test_metrics = avg_both(*model.compute_metrics(test_examples[1:2, :].cuda(), filters))
#     print(f"space id: {i} | {format_metrics(test_metrics, split='test')}")
#     mrr.append(1.0 / test_metrics['MR'])
# print(f"Mean MRR: {np.array(mrr).mean()}")