# FieldE

This is the code for implementation of [Knowledge Graph Representation Learning usingOrdinary Differential Equations](https://aclanthology.org/2021.emnlp-main.750.pdf) (EMNLP 2021).

## Library Overview

This implementation includes the following models:

#### Complex embeddings:

*   Complex [1]
*   Complex-N3 [2]
*   RotatE (without self-adversarial sampling) [3]

#### Euclidean embeddings:

*   CTDecomp [2]
*   TransE [4]
*   MurE [5]
*   RotE [6]
*   RefE [6]
*   AttE [6]
*   FieldE [7]

#### Hyperbolic embeddings:

*   RotH [6]
*   RefH [6]
*   AttH [6]
*   FieldP [7]
*   FieldH [7]

## Installation

The starting point is to install KGEmb framework. To this end, first, create a python 3.7 environment and install dependencies:

```bash
virtualenv -p python3.7 hyp_kg_env
source hyp_kg_env/bin/activate
pip install -r requirements.txt
```

Then, set environment variables and activate your environment:

```bash
source set_env.sh
```

## Datasets

Download and pre-process the datasets:

```bash
source datasets/download.sh
python datasets/process.py
```

## Usage

To train and evaluate a KG embedding model for the link prediction task, use the run.py script:

```bash
usage: run.py [-h] [--dataset {FB15K,WN,WN18RR,FB237,YAGO3-10}]
              [--model {TransE,CP,MurE,RotE,RefE,AttE,RotH,RefH,AttH,ComplEx,RotatE, FieldE, FieldP, FieldH}]
              [--regularizer {N3,N2}] [--reg REG]
              [--optimizer {Adagrad,Adam,SGD,SparseAdam,RSGD,RAdam}]
              [--max_epochs MAX_EPOCHS] [--patience PATIENCE] [--valid VALID]
              [--rank RANK] [--batch_size BATCH_SIZE]
              [--neg_sample_size NEG_SAMPLE_SIZE] [--dropout DROPOUT]
              [--init_size INIT_SIZE] [--learning_rate LEARNING_RATE]
              [--gamma GAMMA] [--bias {constant,learn,none}]
              [--dtype {single,double}] [--double_neg] [--debug] [--multi_c]

Knowledge Graph Embedding

optional arguments:
  -h, --help            show this help message and exit
  --dataset {FB15K,WN,WN18RR,FB237,YAGO3-10}
                        Knowledge Graph dataset
  --model {TransE,CP,MurE,RotE,RefE,AttE,RotH,RefH,AttH,ComplEx,RotatE, FieldE, FieldP, FieldH}
                        Knowledge Graph embedding model
  --regularizer {N3,N2}
                        Regularizer
  --reg REG             Regularization weight
  --optimizer {Adagrad,Adam,SparseAdam}
                        Optimizer
  --max_epochs MAX_EPOCHS
                        Maximum number of epochs to train for
  --patience PATIENCE   Number of epochs before early stopping
  --valid VALID         Number of epochs before validation
  --rank RANK           Embedding dimension
  --batch_size BATCH_SIZE
                        Batch size
  --neg_sample_size NEG_SAMPLE_SIZE
                        Negative sample size, -1 to not use negative sampling
  --dropout DROPOUT     Dropout rate
  --init_size INIT_SIZE
                        Initial embeddings' scale
  --learning_rate LEARNING_RATE
                        Learning rate
  --gamma GAMMA         Margin for distance-based losses
  --bias {constant,learn,none}
                        Bias type (none for no bias)
  --dtype {single,double}
                        Machine precision
  --double_neg          Whether to negative sample both head and tail entities
  --debug               Only use 1000 examples for debugging
  --multi_c             Multiple curvatures per relation
```

## Citation

If you use the codes, please cite the following paper [6][7]:

## References

[1] Trouillon, Théo, et al. "Complex embeddings for simple link prediction."
International Conference on Machine Learning. 2016.

[2] Lacroix, Timothee, et al. "Canonical Tensor Decomposition for Knowledge Base
Completion." International Conference on Machine Learning. 2018.

[3] Sun, Zhiqing, et al. "Rotate: Knowledge graph embedding by relational
rotation in complex space." International Conference on Learning
Representations. 2019.

[4] Bordes, Antoine, et al. "Translating embeddings for modeling
multi-relational data." Advances in neural information processing systems. 2013.

[5] Balažević, Ivana, et al. "Multi-relational Poincaré Graph Embeddings."
Advances in neural information processing systems. 2019.

[6] Chami, Ines, et al. "Low-Dimensional Hyperbolic Knowledge Graph Embeddings."
Annual Meeting of the Association for Computational Linguistics. 2020.

[7] Nayyeri, Mojtaba, et al. "Knowledge Graph Representation Learning usingOrdinary Differential Equations."
Conference on Empirical Methods in Natural Language Processing. 2021.

Some of the code was forked from the original ComplEx-N3 implementation which can be found at: [https://github.com/facebookresearch/kbc](https://github.com/facebookresearch/kbc)

