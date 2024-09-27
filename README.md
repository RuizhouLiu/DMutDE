## DMutDE

Use `pip install -r requirements.txt` to install required packages. (Please install PyTorch 2.x version  [https://pytorch.org/get-started/locally/]) 

1. `cd` to top-level folder
2. Download dataset by `sh datasets/download.sh`.
3. `python datasets/process.py`

4. `sh run_FB15K-237.sh` to reproduce the experimental results of `RotH (AttH)` on FB15K-237 dataset.
5. `sh run_WN18RR.sh` to reproduce the experimental results of `RotH (RefH)` on WN18RR dataset.

6. you can replace `--global_model` and `--local_model` with other KGE model, like `AttH`, `RotH`, `RefH`, `LocAttH`, `LocRotH` and `LocRefH` etc.

## Citation

If you use the codes, please cite the following paper [1]:

## References

[1] Chami, Ines, et al. "Low-Dimensional Hyperbolic Knowledge Graph Embeddings."
Annual Meeting of the Association for Computational Linguistics. 2020.

Some of the code was forked from the original ComplEx-N3 implementation which can be found at: [https://github.com/facebookresearch/kbc](https://github.com/facebookresearch/kbc)

