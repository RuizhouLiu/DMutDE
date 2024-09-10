import subprocess
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2,3,4,5,6,7'
from copy import deepcopy
import time
import ipdb

def get_cuda_scop():
    try:
        cuda_ids = os.environ['CUDA_VISIBLE_DEVICES'].split(',')
        cuda_ids = [int(cuda_i) for cuda_i in cuda_ids]
    except:
        cuda_ids = list(range(8))
    
    return cuda_ids

def gpu_info():
    cuda_scop = get_cuda_scop()
    gpu_status = os.popen('nvidia-smi | grep %').read().split('|')
    # ipdb.set_trace()
    gpus = []
    tmp_gpu_i = []
    gpu_counter = 0
    for line_i in gpu_status[1:]:
        if line_i != '\n':
            tmp_gpu_i.append(line_i)
        else:
            if gpu_counter in cuda_scop:
                gpu_memory = int(tmp_gpu_i[1].split('/')[0].split('M')[0].strip())
                gpu_power = int(tmp_gpu_i[0].split('   ')[-1].split('/')[0].split('W')[0].strip())
                tmp_gpu_i = []
                gpus.append((gpu_memory, gpu_power))
            else:
                tmp_gpu_i.pop(0)
                tmp_gpu_i.pop(0)
                tmp_gpu_i.pop(0)
            gpu_counter += 1
            
    
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
                for feat_kd_weight_i in feat_kd_weights:
                    model_config_cp = deepcopy(eval(f"{global_model}_{local_model}_{datasset_i}"))
                    model_config_cp['feat_kd_weight'] = feat_kd_weight_i
                    out.append(model_config_cp)
    return [gen_one_cmd(out_i) for out_i in out]




AttH_RotH_FB237 = {
    "dataset": "FB237", "model": "EnsembleModel", "regularizer": "N3", "reg": "0.0", "optimizer": "AdamW", "max_epochs":"500", "patience": "100",
    "valid": "5", "batch_size": "8000", "neg_sample_size": "100", "init_size": "0.001", "learning_rate": "1e-3", "gamma": "0.0", "bias": "learn",
    "dtype": "double", "multi_c": True, "double_neg": False, "KD_iter": "40", "global_model": "AttH", "local_model": "RotH", "global_model_rank": "32", 
    "local_model_rank": "32", "global_kd_weight": "0.1", "local_kd_weight": "0.1", "pretained": False, 'feat_kd_weight': '0.0'
}

feat_kd_weights = ['0.00', '0.02', '0.04', '0.06', '0.08', '0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9', '1.0']
pairs_stu_tea = [("AttH", "RotH")]


if __name__ == "__main__":
    trials = []
    cmds = gen_cmds()
    for i, cmd in enumerate(cmds):
        avai_cuda = find_avail_cuda()
        # ipdb.set_trace()
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
    
        