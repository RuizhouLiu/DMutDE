import subprocess
import os
from copy import deepcopy
import time

def gpu_info():
    gpu_status = os.popen('nvidia-smi | grep %').read().split('|')

    gpus = []
    tmp_gpu_i = []
    for line_i in gpu_status[1:]:
        if line_i != '\n':
            tmp_gpu_i.append(line_i)
        else:
            gpu_memory = int(tmp_gpu_i[1].split('/')[0].split('M')[0].strip())
            gpu_power = int(tmp_gpu_i[0].split('   ')[-1].split('/')[0].split('W')[0].strip())
            tmp_gpu_i = []
            gpus.append((gpu_memory, gpu_power))
    
    return gpus

def find_avail_cuda():
    cudas = gpu_info()
    cuda_stats = {}
    for i, (cuda_mem, cuda_power) in enumerate(cudas):
        cuda_stats[i] = (cuda_mem < 600 and cuda_power < 60)
    for k, v in cuda_stats.items():
        if v: return k
    return None

def reform(cmd):
    out = []
    for cmd_i in cmd:
        out.extend(cmd_i.split())
    
    return out

def gen_one_cmd(cfg):
    exec = "python"
    file = "run.py"
    args = [exec, file]
    for k, v in cfg.items():
        if isinstance(v, bool):
            if v:
                args.append(f"--{k}")
        else:
            args.append(f"--{k} {v}")
    
    return reform(args)


def gen_cmds():
    out = []
    dataset = ["FB237"]
    local_ranks = [32]
    for local_rank_i in local_ranks:
        for datasset_i in dataset:
            for global_model, local_model in pairs_stu_tea:
                for i in range(1, 6):
                    model_config_cp = deepcopy(eval(f"{global_model}_{local_model}_{datasset_i}_{i}"))
                    out.append(model_config_cp)
    return [gen_one_cmd(out_i) for out_i in out]


# student_auxiliary_dataset forms
# TransE_TransE_FB237 = {
#     "dataset": "FB237", "model": "EnsembleModel", "regularizer": "N3", "reg": "0.0", "optimizer": "AdamW", "max_epochs":"500", "patience": "100",
#     "valid": "5", "batch_size": "5000", "neg_sample_size": "100", "init_size": "0.001", "learning_rate": "1e-3", "gamma": "0.0", "bias": "learn",
#     "dtype": "double", "multi_c": False, "double_neg": False, "KD_iter": "40", "global_model": "TransE",  "local_model": "TransE", "global_model_rank": "512", 
#     "local_model_rank": "32",  "global_kd_weight": "0.01", "local_kd_weight": "0.1", "pretained": True
# }
# TransE_RotatE_FB237 = {
#     "dataset": "FB237", "model": "EnsembleModel", "regularizer": "N3", "reg": "0.0", "optimizer": "AdamW", "max_epochs":"500", "patience": "100",
#     "valid": "5", "batch_size": "5000", "neg_sample_size": "100", "init_size": "0.001", "learning_rate": "1e-3", "gamma": "0.0", "bias": "learn",
#     "dtype": "double", "multi_c": False, "double_neg": False, "KD_iter": "40", "global_model": "TransE",  "local_model": "RotatE", "global_model_rank": "512", 
#     "local_model_rank": "32",  "global_kd_weight": "0.01", "local_kd_weight": "0.1", "pretained": True
# }
# RotatE_RotatE_FB237 = {
#     "dataset": "FB237", "model": "EnsembleModel", "regularizer": "N3", "reg": "0.0", "optimizer": "AdamW", "max_epochs":"500", "patience": "100",
#     "valid": "5", "batch_size": "5000", "neg_sample_size": "100", "init_size": "0.001", "learning_rate": "1e-3", "gamma": "0.0", "bias": "learn",
#      "dtype": "double", "multi_c": False, "double_neg": False, "KD_iter": "40", "global_model": "RotatE",  "local_model": "RotatE", "global_model_rank": "512", 
#     "local_model_rank": "32",  "global_kd_weight": "0.01", "local_kd_weight": "0.1", "pretained": True
# }
# RotatE_TransE_FB237 = {
#     "dataset": "FB237", "model": "EnsembleModel", "regularizer": "N3", "reg": "0.0", "optimizer": "AdamW", "max_epochs":"500", "patience": "100",
#     "valid": "5", "batch_size": "5000", "neg_sample_size": "100", "init_size": "0.001", "learning_rate": "1e-3", "gamma": "0.0", "bias": "learn",
#      "dtype": "double", "multi_c": False, "double_neg": False, "KD_iter": "40", "global_model": "RotatE",  "local_model": "TransE", "global_model_rank": "512", 
#     "local_model_rank": "32",  "global_kd_weight": "0.01", "local_kd_weight": "0.1", "pretained": True
# }
# AttH_AttH_FB237 = {
#     "dataset": "FB237", "model": "EnsembleModel", "regularizer": "N3", "reg": "0.0", "optimizer": "Adagrad", "max_epochs":"500", "patience": "100",
#     "valid": "5", "batch_size": "4000", "neg_sample_size": "100", "init_size": "0.001", "learning_rate": "0.1", "gamma": "0.0", "bias": "learn",
#     "dtype": "double", "multi_c": True, "double_neg": False, "KD_iter": "40", "global_model": "AttH", "local_model": "AttH", "global_model_rank": "512", 
#     "local_model_rank": "32", "global_kd_weight": "0.01", "local_kd_weight": "0.1", "pretained": True
# }
# AttH_RotH_FB237 = {
#     "dataset": "FB237", "model": "EnsembleModel", "regularizer": "N3", "reg": "0.0", "optimizer": "Adagrad", "max_epochs":"500", "patience": "100",
#     "valid": "5", "batch_size": "4000", "neg_sample_size": "100", "init_size": "0.001", "learning_rate": "0.1", "gamma": "0.0", "bias": "learn",
#     "dtype": "double", "multi_c": True, "double_neg": False, "KD_iter": "40", "global_model": "AttH", "local_model": "RotH", "global_model_rank": "512", 
#     "local_model_rank": "32", "global_kd_weight": "0.01", "local_kd_weight": "0.1", "pretained": True
# }
# AttH_RefH_FB237 = {
#     "dataset": "FB237", "model": "EnsembleModel", "regularizer": "N3", "reg": "0.0", "optimizer": "Adagrad", "max_epochs":"500", "patience": "100",
#     "valid": "5", "batch_size": "4000", "neg_sample_size": "100", "init_size": "0.001", "learning_rate": "0.1", "gamma": "0.0", "bias": "learn",
#     "dtype": "double", "multi_c": True, "double_neg": False, "KD_iter": "40", "global_model": "AttH", "local_model": "RefH", "global_model_rank": "512", 
#     "local_model_rank": "32", "global_kd_weight": "0.01", "local_kd_weight": "0.1", "pretained": True
# }
# RotH_RotH_FB237 = {
#     "dataset": "FB237", "model": "EnsembleModel", "regularizer": "N3", "reg": "0.0", "optimizer": "Adagrad", "max_epochs":"500", "patience": "100",
#     "valid": "5", "batch_size": "4000", "neg_sample_size": "50", "init_size": "0.001", "learning_rate": "0.1", "gamma": "0.0", "bias": "learn",
#     "dtype": "double", "multi_c": True, "double_neg": False, "KD_iter": "40", "global_model": "RotH", "local_model": "RotH", "global_model_rank": "512", 
#     "local_model_rank": "32", "global_kd_weight": "0.01", "local_kd_weight": "0.1", "pretained": True
# }
# RotH_AttH_FB237 = {
#     "dataset": "FB237", "model": "EnsembleModel", "regularizer": "N3", "reg": "0.0", "optimizer": "Adagrad", "max_epochs":"500", "patience": "100",
#     "valid": "5", "batch_size": "4000", "neg_sample_size": "50", "init_size": "0.001", "learning_rate": "0.1", "gamma": "0.0", "bias": "learn",
#     "dtype": "double", "multi_c": True, "double_neg": False, "KD_iter": "40", "global_model": "RotH", "local_model": "AttH", "global_model_rank": "512", 
#     "local_model_rank": "32", "global_kd_weight": "0.01", "local_kd_weight": "0.1", "pretained": True
# }
# RotH_RefH_FB237 = {
#     "dataset": "FB237", "model": "EnsembleModel", "regularizer": "N3", "reg": "0.0", "optimizer": "Adagrad", "max_epochs":"500", "patience": "100",
#     "valid": "5", "batch_size": "2000", "neg_sample_size": "50", "init_size": "0.001", "learning_rate": "0.1", "gamma": "0.0", "bias": "learn",
#     "dtype": "double", "multi_c": True, "double_neg": False, "KD_iter": "40", "global_model": "RotH", "local_model": "RefH", "global_model_rank": "512", 
#     "local_model_rank": "32", "global_kd_weight": "0.01", "local_kd_weight": "0.1", "pretained": True
# }
# RefH_RefH_FB237 = {
#     "dataset": "FB237", "model": "EnsembleModel", "regularizer": "N3", "reg": "0.0", "optimizer": "Adagrad", "max_epochs":"500", "patience": "100",
#     "valid": "5", "batch_size": "4000", "neg_sample_size": "100", "init_size": "0.001", "learning_rate": "0.1", "gamma": "0.0", "bias": "learn",
#     "dtype": "double", "multi_c": True, "double_neg": False, "KD_iter": "40", "global_model": "RefH", "local_model": "RefH", "global_model_rank": "512", 
#     "local_model_rank": "32", "global_kd_weight": "0.01", "local_kd_weight": "0.1", "pretained": True
# }
# RefH_AttH_FB237 = {
#     "dataset": "FB237", "model": "EnsembleModel", "regularizer": "N3", "reg": "0.0", "optimizer": "Adagrad", "max_epochs":"500", "patience": "100",
#     "valid": "5", "batch_size": "4000", "neg_sample_size": "100", "init_size": "0.001", "learning_rate": "0.1", "gamma": "0.0", "bias": "learn",
#     "dtype": "double", "multi_c": True, "double_neg": False, "KD_iter": "40", "global_model": "RefH", "local_model": "AttH", "global_model_rank": "512", 
#     "local_model_rank": "32", "global_kd_weight": "0.01", "local_kd_weight": "0.1", "pretained": True
# }
# RefH_RotH_FB237 = {
#     "dataset": "FB237", "model": "EnsembleModel", "regularizer": "N3", "reg": "0.0", "optimizer": "Adagrad", "max_epochs":"500", "patience": "100",
#     "valid": "5", "batch_size": "4000", "neg_sample_size": "100", "init_size": "0.001", "learning_rate": "0.1", "gamma": "0.0", "bias": "learn",
#     "dtype": "double", "multi_c": True, "double_neg": False, "KD_iter": "40", "global_model": "RefH", "local_model": "RotH", "global_model_rank": "512",
#     "local_model_rank": "32", "global_kd_weight": "0.01", "local_kd_weight": "0.1", "pretained": True
# }
# AttH_LocAttH_FB237 = {
#     "dataset": "FB237", "model": "EnsembleModel", "regularizer": "N3", "reg": "0.0", "optimizer": "AdamW", "max_epochs":"500", "patience": "100",
#     "valid": "5", "batch_size": "3000", "neg_sample_size": "100", "init_size": "0.001", "learning_rate": "1e-3", "gamma": "0.0", "bias": "learn",
#     "dtype": "double", "multi_c": True, "double_neg": False, "KD_iter": "40", "global_model": "AttH", "local_model": "LocAttH", "global_model_rank": "32", 
#     "local_model_rank": "32", "global_kd_weight": "0.1", "local_kd_weight": "0.1"
# }
# RotH_LocRotH_FB237 = {
#     "dataset": "FB237", "model": "EnsembleModel", "regularizer": "N3", "reg": "0.0", "optimizer": "AdamW", "max_epochs":"500", "patience": "100",
#     "valid": "5", "batch_size": "3000", "neg_sample_size": "50", "init_size": "0.001", "learning_rate": "1e-3", "gamma": "0.0", "bias": "learn",
#     "dtype": "double", "multi_c": True, "double_neg": False, "KD_iter": "40", "global_model": "RotH", "local_model": "LocRotH", "global_model_rank": "32", 
#     "local_model_rank": "32", "global_kd_weight": "0.1", "local_kd_weight": "0.1"
# }
# RefH_LocRefH_FB237 = {
#     "dataset": "FB237", "model": "EnsembleModel", "regularizer": "N3", "reg": "0.0", "optimizer": "AdamW", "max_epochs":"500", "patience": "100",
#     "valid": "5", "batch_size": "3000", "neg_sample_size": "50", "init_size": "0.001", "learning_rate": "1e-3", "gamma": "0.0", "bias": "learn",
#     "dtype": "double", "multi_c": True, "double_neg": False, "KD_iter": "40", "global_model": "RefH", "local_model": "LocRefH", "global_model_rank": "32", 
#     "local_model_rank": "32", "global_kd_weight": "0.1", "local_kd_weight": "0.1"
# }

# LocAttH_AttH_FB237 = {
#     "dataset": "FB237", "model": "EnsembleModel", "regularizer": "N3", "reg": "0.0", "optimizer": "AdamW", "max_epochs":"500", "patience": "100",
#     "valid": "5", "batch_size": "3000", "neg_sample_size": "100", "init_size": "0.001", "learning_rate": "1e-3", "gamma": "0.0", "bias": "learn",
#     "dtype": "double", "multi_c": True, "double_neg": False, "KD_iter": "40", "global_model": "AttH", "local_model": "LocAttH", "global_model_rank": "32", 
#     "local_model_rank": "32", "global_kd_weight": "0.1", "local_kd_weight": "0.1"
# }
# LocRotH_RotH_FB237 = {
#     "dataset": "FB237", "model": "EnsembleModel", "regularizer": "N3", "reg": "0.0", "optimizer": "AdamW", "max_epochs":"500", "patience": "100",
#     "valid": "5", "batch_size": "3000", "neg_sample_size": "50", "init_size": "0.001", "learning_rate": "1e-3", "gamma": "0.0", "bias": "learn",
#     "dtype": "double", "multi_c": True, "double_neg": False, "KD_iter": "40", "global_model": "RotH", "local_model": "LocRotH", "global_model_rank": "32", 
#     "local_model_rank": "32", "global_kd_weight": "0.1", "local_kd_weight": "0.1"
# }
# LocRefH_RefH_FB237 = {
#     "dataset": "FB237", "model": "EnsembleModel", "regularizer": "N3", "reg": "0.0", "optimizer": "AdamW", "max_epochs":"500", "patience": "100",
#     "valid": "5", "batch_size": "3000", "neg_sample_size": "50", "init_size": "0.001", "learning_rate": "1e-3", "gamma": "0.0", "bias": "learn",
#     "dtype": "double", "multi_c": True, "double_neg": False, "KD_iter": "40", "global_model": "RefH", "local_model": "LocRefH", "global_model_rank": "32", 
#     "local_model_rank": "32", "global_kd_weight": "0.1", "local_kd_weight": "0.1"
# }
# AttH_LocAttH_FB237 = {
#     "dataset": "FB237", "model": "EnsembleModel", "regularizer": "N3", "reg": "0.0", "optimizer": "AdamW", "max_epochs":"500", "patience": "100",
#     "valid": "5", "batch_size": "3000", "neg_sample_size": "100", "init_size": "0.001", "learning_rate": "1e-3", "gamma": "0.0", "bias": "learn",
#     "dtype": "double", "multi_c": True, "double_neg": False, "KD_iter": "40", "global_model": "LocAttH", "local_model": "AttH", "global_model_rank": "32", 
#     "local_model_rank": "32", "global_kd_weight": "0.01", "local_kd_weight": "0.1"
# }
# RotH_LocRotH_FB237 = {
#     "dataset": "FB237", "model": "EnsembleModel", "regularizer": "N3", "reg": "0.0", "optimizer": "AdamW", "max_epochs":"500", "patience": "100",
#     "valid": "5", "batch_size": "3000", "neg_sample_size": "50", "init_size": "0.001", "learning_rate": "1e-3", "gamma": "0.0", "bias": "learn",
#     "dtype": "double", "multi_c": True, "double_neg": False, "KD_iter": "40", "global_model": "LocRotH", "local_model": "RotH", "global_model_rank": "32", 
#     "local_model_rank": "32", "global_kd_weight": "0.01", "local_kd_weight": "0.1"
# }
# RefH_LocRefH_FB237 = {
#     "dataset": "FB237", "model": "EnsembleModel", "regularizer": "N3", "reg": "0.0", "optimizer": "AdamW", "max_epochs":"500", "patience": "100",
#     "valid": "5", "batch_size": "3000", "neg_sample_size": "50", "init_size": "0.001", "learning_rate": "1e-3", "gamma": "0.0", "bias": "learn",
#     "dtype": "double", "multi_c": True, "double_neg": False, "KD_iter": "40", "global_model": "LocRefH", "local_model": "RefH", "global_model_rank": "32", 
#     "local_model_rank": "32", "global_kd_weight": "0.01", "local_kd_weight": "0.1"
# }

# TransE_AttH_FB237 = {
#     "dataset": "FB237", "model": "EnsembleModel", "regularizer": "N3", "reg": "0.0", "optimizer": "AdamW", "max_epochs":"500", "patience": "100",
#     "valid": "5", "batch_size": "3000", "neg_sample_size": "100", "init_size": "0.001", "learning_rate": "1e-3", "gamma": "0.0", "bias": "learn",
#     "dtype": "double", "multi_c": True, "double_neg": False, "KD_iter": "40", "global_model": "TransE",  "local_model": "AttH", "global_model_rank": "32", 
#     "local_model_rank": "32",  "global_kd_weight": "0.1", "local_kd_weight": "0.1"
# }
# TransE_RotH_FB237 = {
#     "dataset": "FB237", "model": "EnsembleModel", "regularizer": "N3", "reg": "0.0", "optimizer": "AdamW", "max_epochs":"500", "patience": "100",
#     "valid": "5", "batch_size": "3000", "neg_sample_size": "100", "init_size": "0.001", "learning_rate": "1e-3", "gamma": "0.0", "bias": "learn",
#     "dtype": "double", "multi_c": True, "double_neg": False, "KD_iter": "40", "global_model": "TransE",  "local_model": "RotH", "global_model_rank": "32", 
#     "local_model_rank": "32",  "global_kd_weight": "0.1", "local_kd_weight": "0.1"
# }
# RotatE_AttH_FB237 = {
#     "dataset": "FB237", "model": "EnsembleModel", "regularizer": "N3", "reg": "0.0", "optimizer": "AdamW", "max_epochs":"500", "patience": "100",
#     "valid": "5", "batch_size": "3000", "neg_sample_size": "100", "init_size": "0.001", "learning_rate": "1e-3", "gamma": "0.0", "bias": "learn",
#     "dtype": "double", "multi_c": True, "double_neg": False, "KD_iter": "40", "global_model": "RotatE",  "local_model": "AttH", "global_model_rank": "32", 
#     "local_model_rank": "32",  "global_kd_weight": "0.1", "local_kd_weight": "0.1"
# }
# RotatE_RotH_FB237 = {
#     "dataset": "FB237", "model": "EnsembleModel", "regularizer": "N3", "reg": "0.0", "optimizer": "AdamW", "max_epochs":"500", "patience": "100",
#     "valid": "5", "batch_size": "3000", "neg_sample_size": "100", "init_size": "0.001", "learning_rate": "1e-3", "gamma": "0.0", "bias": "learn",
#     "dtype": "double", "multi_c": True, "double_neg": False, "KD_iter": "40", "global_model": "RotatE",  "local_model": "RotH", "global_model_rank": "32", 
#     "local_model_rank": "32",  "global_kd_weight": "0.1", "local_kd_weight": "0.1"
# }
# AttH_TransE_FB237 = {
#     "dataset": "FB237", "model": "EnsembleModel", "regularizer": "N3", "reg": "0.0", "optimizer": "AdamW", "max_epochs":"500", "patience": "100",
#     "valid": "5", "batch_size": "3000", "neg_sample_size": "100", "init_size": "0.001", "learning_rate": "1e-3", "gamma": "0.0", "bias": "learn",
#     "dtype": "double", "multi_c": True, "double_neg": False, "KD_iter": "40", "global_model": "AttH",  "local_model": "TransE", "global_model_rank": "32", 
#     "local_model_rank": "32",  "global_kd_weight": "0.1", "local_kd_weight": "0.1"
# }
# AttH_RotatE_FB237 = {
#     "dataset": "FB237", "model": "EnsembleModel", "regularizer": "N3", "reg": "0.0", "optimizer": "AdamW", "max_epochs":"500", "patience": "100",
#     "valid": "5", "batch_size": "3000", "neg_sample_size": "100", "init_size": "0.001", "learning_rate": "1e-3", "gamma": "0.0", "bias": "learn",
#     "dtype": "double", "multi_c": True, "double_neg": False, "KD_iter": "40", "global_model": "AttH",  "local_model": "RotatE", "global_model_rank": "32", 
#     "local_model_rank": "32",  "global_kd_weight": "0.1", "local_kd_weight": "0.1"
# }
# RotH_TransE_FB237 = {
#     "dataset": "FB237", "model": "EnsembleModel", "regularizer": "N3", "reg": "0.0", "optimizer": "AdamW", "max_epochs":"500", "patience": "100",
#     "valid": "5", "batch_size": "3000", "neg_sample_size": "100", "init_size": "0.001", "learning_rate": "1e-3", "gamma": "0.0", "bias": "learn",
#     "dtype": "double", "multi_c": True, "double_neg": False, "KD_iter": "40", "global_model": "RotH",  "local_model": "TransE", "global_model_rank": "32", 
#     "local_model_rank": "32",  "global_kd_weight": "0.1", "local_kd_weight": "0.1"
# }
# RotH_RotatE_FB237 = {
#     "dataset": "FB237", "model": "EnsembleModel", "regularizer": "N3", "reg": "0.0", "optimizer": "AdamW", "max_epochs":"500", "patience": "100",
#     "valid": "5", "batch_size": "3000", "neg_sample_size": "100", "init_size": "0.001", "learning_rate": "1e-3", "gamma": "0.0", "bias": "learn",
#     "dtype": "double", "multi_c": True, "double_neg": False, "KD_iter": "40", "global_model": "RotH",  "local_model": "RotatE", "global_model_rank": "32", 
#     "local_model_rank": "32",  "global_kd_weight": "0.1", "local_kd_weight": "0.1"
# }




# TransE_TransE_WN18RR = {
#     "dataset": "WN18RR", "model": "EnsembleModel", "regularizer": "N3", "reg": "0.0", "optimizer": "AdamW", "max_epochs":"500", "patience": "100",
#     "valid": "5", "batch_size": "3000", "neg_sample_size": "100", "init_size": "0.001", "learning_rate": "1e-3", "gamma": "0.0", "bias": "learn",
#     "dtype": "double", "multi_c": False, "double_neg": True, "KD_iter": "40", "global_model": "TransE",  "local_model": "TransE", "global_model_rank": "512", 
#     "local_model_rank": "32",  "global_kd_weight": "0.01", "local_kd_weight": "0.1", "pretained": True
# }

# TransE_RotatE_WN18RR = {
#     "dataset": "WN18RR", "model": "EnsembleModel", "regularizer": "N3", "reg": "0.0", "optimizer": "AdamW", "max_epochs":"500", "patience": "100",
#     "valid": "5", "batch_size": "3000", "neg_sample_size": "100", "init_size": "0.001", "learning_rate": "1e-3", "gamma": "0.0", "bias": "learn",
#     "dtype": "double", "multi_c": False, "double_neg": True, "KD_iter": "40", "global_model": "TransE",  "local_model": "RotatE", "global_model_rank": "512", 
#     "local_model_rank": "32",  "global_kd_weight": "0.01", "local_kd_weight": "0.1", "pretained": True
# }
# RotatE_RotatE_WN18RR = {
#     "dataset": "WN18RR", "model": "EnsembleModel", "regularizer": "N3", "reg": "0.0", "optimizer": "AdamW", "max_epochs":"500", "patience": "100",
#     "valid": "5", "batch_size": "3000", "neg_sample_size": "100", "init_size": "0.001", "learning_rate": "1e-3", "gamma": "0.0", "bias": "learn",
#      "dtype": "double", "multi_c": False, "double_neg": True, "KD_iter": "40", "global_model": "RotatE",  "local_model": "RotatE", "global_model_rank": "512", 
#     "local_model_rank": "32",  "global_kd_weight": "0.01", "local_kd_weight": "0.1", "pretained": True
# }
# RotatE_TransE_WN18RR = {
#     "dataset": "WN18RR", "model": "EnsembleModel", "regularizer": "N3", "reg": "0.0", "optimizer": "AdamW", "max_epochs":"500", "patience": "100",
#     "valid": "5", "batch_size": "3000", "neg_sample_size": "100", "init_size": "0.001", "learning_rate": "1e-3", "gamma": "0.0", "bias": "learn",
#      "dtype": "double", "multi_c": False, "double_neg": True, "KD_iter": "40", "global_model": "RotatE",  "local_model": "TransE", "global_model_rank": "512", 
#     "local_model_rank": "32",  "global_kd_weight": "0.01", "local_kd_weight": "0.1", "pretained": True
# }
# AttH_AttH_WN18RR = {
#     "dataset": "WN18RR", "model": "EnsembleModel", "regularizer": "N3", "reg": "0.0", "optimizer": "AdamW", "max_epochs":"400", "patience": "100",
#     "valid": "5", "batch_size": "500", "neg_sample_size": "50", "init_size": "0.001", "learning_rate": "1e-3", "gamma": "0.0", "bias": "learn",
#     "dtype": "double", "multi_c": True, "double_neg": True, "KD_iter": "40", "global_model": "AttH", "local_model": "AttH", "global_model_rank": "512", 
#     "local_model_rank": "32", "global_kd_weight": "0.01", "local_kd_weight": "0.02", "pretained": True
# }
# AttH_RotH_WN18RR = {
#     "dataset": "WN18RR", "model": "EnsembleModel", "regularizer": "N3", "reg": "0.0", "optimizer": "AdamW", "max_epochs":"400", "patience": "100",
#     "valid": "5", "batch_size": "500", "neg_sample_size": "50", "init_size": "0.001", "learning_rate": "1e-3", "gamma": "0.0", "bias": "learn",
#     "dtype": "double", "multi_c": True, "double_neg": True, "KD_iter": "40", "global_model": "AttH", "local_model": "RotH", "global_model_rank": "512", 
#     "local_model_rank": "32", "global_kd_weight": "0.01", "local_kd_weight": "0.02", "pretained": True
# }
# AttH_RefH_WN18RR = {
#     "dataset": "WN18RR", "model": "EnsembleModel", "regularizer": "N3", "reg": "0.0", "optimizer": "AdamW", "max_epochs":"400", "patience": "100",
#     "valid": "5", "batch_size": "500", "neg_sample_size": "50", "init_size": "0.001", "learning_rate": "1e-3", "gamma": "0.0", "bias": "learn",
#     "dtype": "double", "multi_c": True, "double_neg": True, "KD_iter": "40", "global_model": "AttH", "local_model": "RefH", "global_model_rank": "512", 
#     "local_model_rank": "32", "global_kd_weight": "0.01", "local_kd_weight": "0.1", "pretained": True
# }
# RotH_RotH_WN18RR = {
#     "dataset": "WN18RR", "model": "EnsembleModel", "regularizer": "N3", "reg": "0.0", "optimizer": "AdamW", "max_epochs":"400", "patience": "100",
#     "valid": "5", "batch_size": "500", "neg_sample_size": "50", "init_size": "0.001", "learning_rate": "1e-3", "gamma": "0.0", "bias": "learn",
#     "dtype": "double", "multi_c": True, "double_neg": True, "KD_iter": "40", "global_model": "RotH", "local_model": "RotH", "global_model_rank": "512", 
#     "local_model_rank": "32", "global_kd_weight": "0.01", "local_kd_weight": "0.1", "pretained": True
# }
# RotH_AttH_WN18RR = {
#     "dataset": "WN18RR", "model": "EnsembleModel", "regularizer": "N3", "reg": "0.0", "optimizer": "AdamW", "max_epochs":"400", "patience": "100",
#     "valid": "5", "batch_size": "500", "neg_sample_size": "50", "init_size": "0.001", "learning_rate": "1e-3", "gamma": "0.0", "bias": "learn",
#     "dtype": "double", "multi_c": True, "double_neg": True, "KD_iter": "40", "global_model": "RotH", "local_model": "AttH", "global_model_rank": "512", 
#     "local_model_rank": "32", "global_kd_weight": "0.01", "local_kd_weight": "0.1", "pretained": True
# }
# RotH_RefH_WN18RR = {
#     "dataset": "WN18RR", "model": "EnsembleModel", "regularizer": "N3", "reg": "0.0", "optimizer": "AdamW", "max_epochs":"400", "patience": "100",
#     "valid": "5", "batch_size": "500", "neg_sample_size": "50", "init_size": "0.001", "learning_rate": "1e-3", "gamma": "0.0", "bias": "learn",
#     "dtype": "double", "multi_c": True, "double_neg": True, "KD_iter": "40", "global_model": "RotH", "local_model": "RefH", "global_model_rank": "512", 
#     "local_model_rank": "32", "global_kd_weight": "0.01", "local_kd_weight": "0.1", "pretained": True
# }

# RefH_RefH_WN18RR = {
#     "dataset": "WN18RR", "model": "EnsembleModel", "regularizer": "N3", "reg": "0.0", "optimizer": "AdamW", "max_epochs":"400", "patience": "100",
#     "valid": "5", "batch_size": "500", "neg_sample_size": "100", "init_size": "0.001", "learning_rate": "1e-3", "gamma": "0.0", "bias": "learn",
#     "dtype": "double", "multi_c": True, "double_neg": True, "KD_iter": "40", "global_model": "RefH", "local_model": "RefH", "global_model_rank": "512", 
#     "local_model_rank": "32", "global_kd_weight": "0.01", "local_kd_weight": "0.1", "pretained": True
# }
# RefH_AttH_WN18RR = {
#     "dataset": "WN18RR", "model": "EnsembleModel", "regularizer": "N3", "reg": "0.0", "optimizer": "AdamW", "max_epochs":"400", "patience": "100",
#     "valid": "5", "batch_size": "500", "neg_sample_size": "100", "init_size": "0.001", "learning_rate": "1e-3", "gamma": "0.0", "bias": "learn",
#     "dtype": "double", "multi_c": True, "double_neg": True, "KD_iter": "40", "global_model": "RefH", "local_model": "AttH", "global_model_rank": "512", 
#     "local_model_rank": "32", "global_kd_weight": "0.01", "local_kd_weight": "0.1", "pretained": True
# }
# RefH_RotH_WN18RR = {
#     "dataset": "WN18RR", "model": "EnsembleModel", "regularizer": "N3", "reg": "0.0", "optimizer": "AdamW", "max_epochs":"400", "patience": "100",
#     "valid": "5", "batch_size": "500", "neg_sample_size": "100", "init_size": "0.001", "learning_rate": "1e-3", "gamma": "0.0", "bias": "learn",
#     "dtype": "double", "multi_c": True, "double_neg": True, "KD_iter": "40", "global_model": "RefH", "local_model": "RotH", "global_model_rank": "512", 
#     "local_model_rank": "32", "global_kd_weight": "0.01", "local_kd_weight": "0.1", "pretained": True
# }
# AttH_LocAttH_WN18RR = {
#     "dataset": "WN18RR", "model": "EnsembleModel", "regularizer": "N3", "reg": "0.0", "optimizer": "AdamW", "max_epochs":"400", "patience": "100",
#     "valid": "5", "batch_size": "500", "neg_sample_size": "100", "init_size": "0.001", "learning_rate": "1e-3", "gamma": "0.0", "bias": "learn",
#     "dtype": "double", "multi_c": True, "double_neg": True, "KD_iter": "40", "global_model": "AttH", "local_model": "LocAttH", "global_model_rank": "32", 
#     "local_model_rank": "32", "global_kd_weight": "0.1", "local_kd_weight": "0.1"
# }
# RotH_LocRotH_WN18RR = {
#     "dataset": "WN18RR", "model": "EnsembleModel", "regularizer": "N3", "reg": "0.0", "optimizer": "AdamW", "max_epochs":"400", "patience": "100",
#     "valid": "5", "batch_size": "500", "neg_sample_size": "100", "init_size": "0.001", "learning_rate": "1e-3", "gamma": "0.0", "bias": "learn",
#     "dtype": "double", "multi_c": True, "double_neg": True, "KD_iter": "40", "global_model": "RotH", "local_model": "LocRotH", "global_model_rank": "32", 
#     "local_model_rank": "32", "global_kd_weight": "0.1", "local_kd_weight": "0.1"
# }
# RefH_LocRefH_WN18RR = {
#     "dataset": "WN18RR", "model": "EnsembleModel", "regularizer": "N3", "reg": "0.0", "optimizer": "AdamW", "max_epochs":"400", "patience": "100",
#     "valid": "5", "batch_size": "500", "neg_sample_size": "100", "init_size": "0.001", "learning_rate": "1e-3", "gamma": "0.0", "bias": "learn",
#     "dtype": "double", "multi_c": True, "double_neg": True, "KD_iter": "40", "global_model": "RefH", "local_model": "LocRefH", "global_model_rank": "32", 
#     "local_model_rank": "32", "global_kd_weight": "0.1", "local_kd_weight": "0.1"
# }
# LocAttH_AttH_WN18RR = {
#     "dataset": "WN18RR", "model": "EnsembleModel", "regularizer": "N3", "reg": "0.0", "optimizer": "AdamW", "max_epochs":"400", "patience": "100",
#     "valid": "5", "batch_size": "500", "neg_sample_size": "100", "init_size": "0.001", "learning_rate": "1e-3", "gamma": "0.0", "bias": "learn",
#     "dtype": "double", "multi_c": True, "double_neg": True, "KD_iter": "40", "global_model": "LocAttH", "local_model": "AttH", "global_model_rank": "32", 
#     "local_model_rank": "32", "global_kd_weight": "0.01", "local_kd_weight": "0.1"
# }
# LocRotH_RotH_WN18RR = {
#     "dataset": "WN18RR", "model": "EnsembleModel", "regularizer": "N3", "reg": "0.0", "optimizer": "AdamW", "max_epochs":"400", "patience": "100",
#     "valid": "5", "batch_size": "500", "neg_sample_size": "100", "init_size": "0.001", "learning_rate": "1e-3", "gamma": "0.0", "bias": "learn",
#     "dtype": "double", "multi_c": True, "double_neg": True, "KD_iter": "40", "global_model": "LocRotH", "local_model": "RotH", "global_model_rank": "32", 
#     "local_model_rank": "32", "global_kd_weight": "0.01", "local_kd_weight": "0.1"
# }
# LocRefH_RefH_WN18RR = {
#     "dataset": "WN18RR", "model": "EnsembleModel", "regularizer": "N3", "reg": "0.0", "optimizer": "AdamW", "max_epochs":"400", "patience": "100",
#     "valid": "5", "batch_size": "500", "neg_sample_size": "100", "init_size": "0.001", "learning_rate": "1e-3", "gamma": "0.0", "bias": "learn",
#     "dtype": "double", "multi_c": True, "double_neg": True, "KD_iter": "40", "global_model": "LocRefH", "local_model": "RefH", "global_model_rank": "32", 
#     "local_model_rank": "32", "global_kd_weight": "0.01", "local_kd_weight": "0.1"
# }

# TransE_AttH_WN18RR = {
#     "dataset": "WN18RR", "model": "EnsembleModel", "regularizer": "N3", "reg": "0.0", "optimizer": "AdamW", "max_epochs":"500", "patience": "100",
#     "valid": "5", "batch_size": "3000", "neg_sample_size": "100", "init_size": "0.001", "learning_rate": "1e-3", "gamma": "0.0", "bias": "learn",
#     "dtype": "double", "multi_c": True, "double_neg": True, "KD_iter": "40", "global_model": "TransE",  "local_model": "AttH", "global_model_rank": "32", 
#     "local_model_rank": "32",  "global_kd_weight": "0.1", "local_kd_weight": "0.1"
# }
# TransE_RotH_WN18RR = {
#     "dataset": "WN18RR", "model": "EnsembleModel", "regularizer": "N3", "reg": "0.0", "optimizer": "AdamW", "max_epochs":"500", "patience": "100",
#     "valid": "5", "batch_size": "3000", "neg_sample_size": "100", "init_size": "0.001", "learning_rate": "1e-3", "gamma": "0.0", "bias": "learn",
#     "dtype": "double", "multi_c": True, "double_neg": True, "KD_iter": "40", "global_model": "TransE",  "local_model": "RotH", "global_model_rank": "32", 
#     "local_model_rank": "32",  "global_kd_weight": "0.1", "local_kd_weight": "0.1"
# }
# RotatE_AttH_WN18RR = {
#     "dataset": "WN18RR", "model": "EnsembleModel", "regularizer": "N3", "reg": "0.0", "optimizer": "AdamW", "max_epochs":"500", "patience": "100",
#     "valid": "5", "batch_size": "3000", "neg_sample_size": "100", "init_size": "0.001", "learning_rate": "1e-3", "gamma": "0.0", "bias": "learn",
#     "dtype": "double", "multi_c": True, "double_neg": True, "KD_iter": "40", "global_model": "RotatE",  "local_model": "AttH", "global_model_rank": "32", 
#     "local_model_rank": "32",  "global_kd_weight": "0.1", "local_kd_weight": "0.1"
# }
# RotatE_RotH_WN18RR = {
#     "dataset": "WN18RR", "model": "EnsembleModel", "regularizer": "N3", "reg": "0.0", "optimizer": "AdamW", "max_epochs":"500", "patience": "100",
#     "valid": "5", "batch_size": "3000", "neg_sample_size": "100", "init_size": "0.001", "learning_rate": "1e-3", "gamma": "0.0", "bias": "learn",
#     "dtype": "double", "multi_c": True, "double_neg": True, "KD_iter": "40", "global_model": "RotatE",  "local_model": "RotH", "global_model_rank": "32", 
#     "local_model_rank": "32",  "global_kd_weight": "0.1", "local_kd_weight": "0.1"
# }
# AttH_TransE_WN18RR = {
#     "dataset": "WN18RR", "model": "EnsembleModel", "regularizer": "N3", "reg": "0.0", "optimizer": "AdamW", "max_epochs":"500", "patience": "100",
#     "valid": "5", "batch_size": "3000", "neg_sample_size": "100", "init_size": "0.001", "learning_rate": "1e-3", "gamma": "0.0", "bias": "learn",
#     "dtype": "double", "multi_c": True, "double_neg": True, "KD_iter": "40", "global_model": "AttH",  "local_model": "TransE", "global_model_rank": "32", 
#     "local_model_rank": "32",  "global_kd_weight": "0.1", "local_kd_weight": "0.1"
# }
# AttH_RotatE_WN18RR = {
#     "dataset": "WN18RR", "model": "EnsembleModel", "regularizer": "N3", "reg": "0.0", "optimizer": "AdamW", "max_epochs":"500", "patience": "100",
#     "valid": "5", "batch_size": "3000", "neg_sample_size": "100", "init_size": "0.001", "learning_rate": "1e-3", "gamma": "0.0", "bias": "learn",
#     "dtype": "double", "multi_c": True, "double_neg": True, "KD_iter": "40", "global_model": "AttH",  "local_model": "RotatE", "global_model_rank": "32", 
#     "local_model_rank": "32",  "global_kd_weight": "0.1", "local_kd_weight": "0.1"
# }
# RotH_TransE_WN18RR = {
#     "dataset": "WN18RR", "model": "EnsembleModel", "regularizer": "N3", "reg": "0.0", "optimizer": "AdamW", "max_epochs":"500", "patience": "100",
#     "valid": "5", "batch_size": "3000", "neg_sample_size": "100", "init_size": "0.001", "learning_rate": "1e-3", "gamma": "0.0", "bias": "learn",
#     "dtype": "double", "multi_c": True, "double_neg": True, "KD_iter": "40", "global_model": "RotH",  "local_model": "TransE", "global_model_rank": "32", 
#     "local_model_rank": "32",  "global_kd_weight": "0.1", "local_kd_weight": "0.1"
# }
# RotH_RotatE_WN18RR = {
#     "dataset": "WN18RR", "model": "EnsembleModel", "regularizer": "N3", "reg": "0.0", "optimizer": "AdamW", "max_epochs":"500", "patience": "100",
#     "valid": "5", "batch_size": "3000", "neg_sample_size": "100", "init_size": "0.001", "learning_rate": "1e-3", "gamma": "0.0", "bias": "learn",
#     "dtype": "double", "multi_c": True, "double_neg": True, "KD_iter": "40", "global_model": "RotH",  "local_model": "RotatE", "global_model_rank": "32", 
#     "local_model_rank": "32",  "global_kd_weight": "0.1", "local_kd_weight": "0.1"
# }


AttH_RotH_FB237_1 = {
    "dataset": "FB237", "model": "EnsembleModel", "regularizer": "N3", "reg": "0.0", "optimizer": "AdamW", "max_epochs":"500", "patience": "100",
    "valid": "5", "batch_size": "8000", "neg_sample_size": "100", "init_size": "0.001", "learning_rate": "1e-3", "gamma": "0.0", "bias": "learn",
    "dtype": "double", "multi_c": True, "double_neg": False, "KD_iter": "40", "global_model": "AttH", "local_model": "RotH", "global_model_rank": "32", 
    "local_model_rank": "32", "global_kd_weight": "0.00", "local_kd_weight": "0.00", "pretained": False, 'feat_kd_weight': '0.0'
}
AttH_RotH_FB237_2 = {
    "dataset": "FB237", "model": "EnsembleModel", "regularizer": "N3", "reg": "0.0", "optimizer": "AdamW", "max_epochs":"500", "patience": "100",
    "valid": "5", "batch_size": "8000", "neg_sample_size": "100", "init_size": "0.001", "learning_rate": "1e-3", "gamma": "0.0", "bias": "learn",
    "dtype": "double", "multi_c": True, "double_neg": False, "KD_iter": "40", "global_model": "AttH", "local_model": "RotH", "global_model_rank": "32", 
    "local_model_rank": "32", "global_kd_weight": "0.02", "local_kd_weight": "0.02", "pretained": False, 'feat_kd_weight': '0.02'
}
AttH_RotH_FB237_3 = {
    "dataset": "FB237", "model": "EnsembleModel", "regularizer": "N3", "reg": "0.0", "optimizer": "AdamW", "max_epochs":"500", "patience": "100",
    "valid": "5", "batch_size": "8000", "neg_sample_size": "100", "init_size": "0.001", "learning_rate": "1e-3", "gamma": "0.0", "bias": "learn",
    "dtype": "double", "multi_c": True, "double_neg": False, "KD_iter": "40", "global_model": "AttH", "local_model": "RotH", "global_model_rank": "32", 
    "local_model_rank": "32", "global_kd_weight": "0.04", "local_kd_weight": "0.04", "pretained": False, 'feat_kd_weight': '0.04'
}
AttH_RotH_FB237_4 = {
    "dataset": "FB237", "model": "EnsembleModel", "regularizer": "N3", "reg": "0.0", "optimizer": "AdamW", "max_epochs":"500", "patience": "100",
    "valid": "5", "batch_size": "8000", "neg_sample_size": "100", "init_size": "0.001", "learning_rate": "1e-3", "gamma": "0.0", "bias": "learn",
    "dtype": "double", "multi_c": True, "double_neg": False, "KD_iter": "40", "global_model": "AttH", "local_model": "RotH", "global_model_rank": "32", 
    "local_model_rank": "32", "global_kd_weight": "0.06", "local_kd_weight": "0.06", "pretained": False, 'feat_kd_weight': '0.06'
}
AttH_RotH_FB237_6 = {
    "dataset": "FB237", "model": "EnsembleModel", "regularizer": "N3", "reg": "0.0", "optimizer": "AdamW", "max_epochs":"500", "patience": "100",
    "valid": "5", "batch_size": "8000", "neg_sample_size": "100", "init_size": "0.001", "learning_rate": "1e-3", "gamma": "0.0", "bias": "learn",
    "dtype": "double", "multi_c": True, "double_neg": False, "KD_iter": "40", "global_model": "AttH", "local_model": "RotH", "global_model_rank": "32", 
    "local_model_rank": "32", "global_kd_weight": "0.08", "local_kd_weight": "0.08", "pretained": False, 'feat_kd_weight': '0.1'
}
AttH_RotH_FB237_7 = {
    "dataset": "FB237", "model": "EnsembleModel", "regularizer": "N3", "reg": "0.0", "optimizer": "AdamW", "max_epochs":"500", "patience": "100",
    "valid": "5", "batch_size": "8000", "neg_sample_size": "100", "init_size": "0.001", "learning_rate": "1e-3", "gamma": "0.0", "bias": "learn",
    "dtype": "double", "multi_c": True, "double_neg": False, "KD_iter": "40", "global_model": "AttH", "local_model": "RotH", "global_model_rank": "32", 
    "local_model_rank": "32", "global_kd_weight": "0.08", "local_kd_weight": "0.08", "pretained": False, 'feat_kd_weight': '0.2'
}
AttH_RotH_FB237_5 = {
    "dataset": "FB237", "model": "EnsembleModel", "regularizer": "N3", "reg": "0.0", "optimizer": "AdamW", "max_epochs":"500", "patience": "100",
    "valid": "5", "batch_size": "8000", "neg_sample_size": "100", "init_size": "0.001", "learning_rate": "1e-3", "gamma": "0.0", "bias": "learn",
    "dtype": "double", "multi_c": True, "double_neg": False, "KD_iter": "40", "global_model": "AttH", "local_model": "RotH", "global_model_rank": "32", 
    "local_model_rank": "32", "global_kd_weight": "0.08", "local_kd_weight": "0.08", "pretained": False, 'feat_kd_weight': '0.3'
}
AttH_RotH_FB237_8 = {
    "dataset": "FB237", "model": "EnsembleModel", "regularizer": "N3", "reg": "0.0", "optimizer": "AdamW", "max_epochs":"500", "patience": "100",
    "valid": "5", "batch_size": "8000", "neg_sample_size": "100", "init_size": "0.001", "learning_rate": "1e-3", "gamma": "0.0", "bias": "learn",
    "dtype": "double", "multi_c": True, "double_neg": False, "KD_iter": "40", "global_model": "AttH", "local_model": "RotH", "global_model_rank": "32", 
    "local_model_rank": "32", "global_kd_weight": "0.08", "local_kd_weight": "0.08", "pretained": False, 'feat_kd_weight': '0.4'
}
AttH_RotH_FB237_5 = {
    "dataset": "FB237", "model": "EnsembleModel", "regularizer": "N3", "reg": "0.0", "optimizer": "AdamW", "max_epochs":"500", "patience": "100",
    "valid": "5", "batch_size": "8000", "neg_sample_size": "100", "init_size": "0.001", "learning_rate": "1e-3", "gamma": "0.0", "bias": "learn",
    "dtype": "double", "multi_c": True, "double_neg": False, "KD_iter": "40", "global_model": "AttH", "local_model": "RotH", "global_model_rank": "32", 
    "local_model_rank": "32", "global_kd_weight": "0.08", "local_kd_weight": "0.08", "pretained": False, 'feat_kd_weight': '0.5'
}
AttH_RotH_FB237_5 = {
    "dataset": "FB237", "model": "EnsembleModel", "regularizer": "N3", "reg": "0.0", "optimizer": "AdamW", "max_epochs":"500", "patience": "100",
    "valid": "5", "batch_size": "8000", "neg_sample_size": "100", "init_size": "0.001", "learning_rate": "1e-3", "gamma": "0.0", "bias": "learn",
    "dtype": "double", "multi_c": True, "double_neg": False, "KD_iter": "40", "global_model": "AttH", "local_model": "RotH", "global_model_rank": "32", 
    "local_model_rank": "32", "global_kd_weight": "0.08", "local_kd_weight": "0.08", "pretained": False, 'feat_kd_weight': '0.6'
}
AttH_RotH_FB237_5 = {
    "dataset": "FB237", "model": "EnsembleModel", "regularizer": "N3", "reg": "0.0", "optimizer": "AdamW", "max_epochs":"500", "patience": "100",
    "valid": "5", "batch_size": "8000", "neg_sample_size": "100", "init_size": "0.001", "learning_rate": "1e-3", "gamma": "0.0", "bias": "learn",
    "dtype": "double", "multi_c": True, "double_neg": False, "KD_iter": "40", "global_model": "AttH", "local_model": "RotH", "global_model_rank": "32", 
    "local_model_rank": "32", "global_kd_weight": "0.08", "local_kd_weight": "0.08", "pretained": False, 'feat_kd_weight': '0.7'
}
AttH_RotH_FB237_5 = {
    "dataset": "FB237", "model": "EnsembleModel", "regularizer": "N3", "reg": "0.0", "optimizer": "AdamW", "max_epochs":"500", "patience": "100",
    "valid": "5", "batch_size": "8000", "neg_sample_size": "100", "init_size": "0.001", "learning_rate": "1e-3", "gamma": "0.0", "bias": "learn",
    "dtype": "double", "multi_c": True, "double_neg": False, "KD_iter": "40", "global_model": "AttH", "local_model": "RotH", "global_model_rank": "32", 
    "local_model_rank": "32", "global_kd_weight": "0.08", "local_kd_weight": "0.08", "pretained": False, 'feat_kd_weight': '0.8'
}
AttH_RotH_FB237_5 = {
    "dataset": "FB237", "model": "EnsembleModel", "regularizer": "N3", "reg": "0.0", "optimizer": "AdamW", "max_epochs":"500", "patience": "100",
    "valid": "5", "batch_size": "8000", "neg_sample_size": "100", "init_size": "0.001", "learning_rate": "1e-3", "gamma": "0.0", "bias": "learn",
    "dtype": "double", "multi_c": True, "double_neg": False, "KD_iter": "40", "global_model": "AttH", "local_model": "RotH", "global_model_rank": "32", 
    "local_model_rank": "32", "global_kd_weight": "0.08", "local_kd_weight": "0.08", "pretained": False, 'feat_kd_weight': '0.9'
}
AttH_RotH_FB237_5 = {
    "dataset": "FB237", "model": "EnsembleModel", "regularizer": "N3", "reg": "0.0", "optimizer": "AdamW", "max_epochs":"500", "patience": "100",
    "valid": "5", "batch_size": "8000", "neg_sample_size": "100", "init_size": "0.001", "learning_rate": "1e-3", "gamma": "0.0", "bias": "learn",
    "dtype": "double", "multi_c": True, "double_neg": False, "KD_iter": "40", "global_model": "AttH", "local_model": "RotH", "global_model_rank": "32", 
    "local_model_rank": "32", "global_kd_weight": "0.08", "local_kd_weight": "0.08", "pretained": False, 'feat_kd_weight': '1.0'
}
# AttH_RotH_FB237_6 = {
#     "dataset": "FB237", "model": "EnsembleModel", "regularizer": "N3", "reg": "0.0", "optimizer": "AdamW", "max_epochs":"500", "patience": "100",
#     "valid": "5", "batch_size": "8000", "neg_sample_size": "100", "init_size": "0.001", "learning_rate": "1e-3", "gamma": "0.0", "bias": "learn",
#     "dtype": "double", "multi_c": True, "double_neg": False, "KD_iter": "40", "global_model": "AttH", "local_model": "RotH", "global_model_rank": "32", 
#     "local_model_rank": "32", "global_kd_weight": "0.5", "local_kd_weight": "0.5", "pretained": False
# }
# AttH_RotH_FB237_7 = {
#     "dataset": "FB237", "model": "EnsembleModel", "regularizer": "N3", "reg": "0.0", "optimizer": "AdamW", "max_epochs":"500", "patience": "100",
#     "valid": "5", "batch_size": "8000", "neg_sample_size": "100", "init_size": "0.001", "learning_rate": "1e-3", "gamma": "0.0", "bias": "learn",
#     "dtype": "double", "multi_c": True, "double_neg": False, "KD_iter": "40", "global_model": "AttH", "local_model": "RotH", "global_model_rank": "32", 
#     "local_model_rank": "32", "global_kd_weight": "0.7", "local_kd_weight": "0.7", "pretained": False
# }
# AttH_RotH_FB237_8 = {
#     "dataset": "FB237", "model": "EnsembleModel", "regularizer": "N3", "reg": "0.0", "optimizer": "AdamW", "max_epochs":"500", "patience": "100",
#     "valid": "5", "batch_size": "8000", "neg_sample_size": "100", "init_size": "0.001", "learning_rate": "1e-3", "gamma": "0.0", "bias": "learn",
#     "dtype": "double", "multi_c": True, "double_neg": False, "KD_iter": "40", "global_model": "AttH", "local_model": "RotH", "global_model_rank": "32", 
#     "local_model_rank": "32", "global_kd_weight": "0.8", "local_kd_weight": "0.8", "pretained": False
# }
# AttH_RotH_FB237_9 = {
#     "dataset": "FB237", "model": "EnsembleModel", "regularizer": "N3", "reg": "0.0", "optimizer": "AdamW", "max_epochs":"500", "patience": "100",
#     "valid": "5", "batch_size": "8000", "neg_sample_size": "100", "init_size": "0.001", "learning_rate": "1e-3", "gamma": "0.0", "bias": "learn",
#     "dtype": "double", "multi_c": True, "double_neg": False, "KD_iter": "40", "global_model": "AttH", "local_model": "RotH", "global_model_rank": "32", 
#     "local_model_rank": "32", "global_kd_weight": "0.9", "local_kd_weight": "0.9", "pretained": False
# }


# pairs_stu_tea = [("TransE", "TransE"), ("TransE", "RotatE"), ("RotatE", "TransE"), ("RotatE", "RotatE"), ("AttH", "AttH"), ("AttH", "RotH"), ("AttH", "RefH"), ("RotH", "RotH"), ("RotH", "RefH"), ("RotH", "AttH"), ("RefH", "AttH"), ("RefH", "RotH"), ("RefH", "RefH")]
# pairs_stu_tea = [("TransE", "TransE"), ("TransE", "RotatE"), ("RotatE", "TransE"), ("RotatE", "RotatE")]
# pairs_stu_tea = [("LocAttH", "AttH"), ("LocRotH", "RotH"), ("LocRefH", "RefH"), ("AttH", "LocAttH"), ("RotH", "LocRotH"), ("RefH", "LocRefH")]
# pairs_stu_tea = [("TransE", "AttH"), ("TransE", "RotH"), ("RotatE", "AttH"), ("RotatE", "RotH"), ("AttH", "TransE"), ("AttH", "RotatE"), ("RotH", "TransE"), ("RotH", "RotatE")]
pairs_stu_tea = [("AttH", "RotH")]


if __name__ == "__main__":
    trials = []
    cmds = gen_cmds()
    for i, cmd in enumerate(cmds):
        avai_cuda = find_avail_cuda()
        while avai_cuda == None: 
            avai_cuda = find_avail_cuda()
            pass
        cmd.append(f"--cuda_id {avai_cuda}")
        cmd_cp = reform(cmd)
        print(" ".join(cmd_cp))
        print()
        trials.append(subprocess.Popen(args=cmd_cp))
        time.sleep(15)

    
    try: 
        while True: pass
    except KeyboardInterrupt:
        for t in trials:
            t.terminate()
        print("over")
    
        