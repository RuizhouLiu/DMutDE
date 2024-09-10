"""Train Knowledge Graph embeddings for link prediction."""

import argparse
import json
import logging
import os

import shutil
import torch
import torch.optim

import models
import optimizers.regularizers as regularizers
from datasets.kg_dataset import KGDataset
from models import all_models
from optimizers.kg_optimizer import KGOptimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau
# from optimizers.my_kgOptimizer import KGOptimizer
from utils.train import get_savedir, avg_both, format_metrics, count_params, cal_alpha

parser = argparse.ArgumentParser(
    description="Knowledge Graph Embedding"
)
parser.add_argument(
    "--dataset", default="WN18RR", choices=["FB15K", "WN", "WN18RR", "FB237", "YAGO3-10"],
    help="Knowledge Graph dataset"
)
parser.add_argument(
    "--model", default="RotE", choices=all_models, help="Knowledge Graph embedding model"
)
parser.add_argument(
    "--regularizer", choices=["N3", "F2"], default="N3", help="Regularizer"
)
parser.add_argument(
    "--reg", default=0, type=float, help="Regularization weight"
)
parser.add_argument(
    "--optimizer", choices=["Adagrad", "Adam", "SparseAdam", "AdamW"], default="Adagrad",
    help="Optimizer"
)
parser.add_argument(
    "--max_epochs", default=50, type=int, help="Maximum number of epochs to train for"
)
parser.add_argument(
    "--patience", default=10, type=int, help="Number of epochs before early stopping"
)
parser.add_argument(
    "--valid", default=3, type=float, help="Number of epochs before validation"
)
parser.add_argument(
    "--batch_size", default=256, type=int, help="Batch size"
)
parser.add_argument(
    "--neg_sample_size", default=50, type=int, help="Negative sample size, -1 to not use negative sampling"
)
parser.add_argument(
    "--dropout", default=0, type=float, help="Dropout rate"
)
parser.add_argument(
    "--init_size", default=1e-3, type=float, help="Initial embeddings' scale"
)
parser.add_argument(
    "--learning_rate", default=1e-1, type=float, help="Learning rate"
)
parser.add_argument(
    "--gamma", default=0, type=float, help="Margin for distance-based losses"
)
parser.add_argument(
    "--bias", default="constant", type=str, choices=["constant", "learn", "none"], help="Bias type (none for no bias)"
)
parser.add_argument(
    "--dtype", default="double", type=str, choices=["single", "double"], help="Machine precision"
)
parser.add_argument(
    "--double_neg", action="store_true",
    help="Whether to negative sample both head and tail entities"
)
parser.add_argument(
    "--debug", action="store_true", 
    help="Only use 1000 examples for debugging"
)
parser.add_argument(
    "--multi_c", action="store_true", default=True, help="Multiple curvatures per relation"
)

parser.add_argument(
    "--valid_batchsize", default=500, type=int, help="validation batchsize"
)

parser.add_argument(
    "--KD_iter", default=40, type=int, help="After 'KD iteration', start knowledge distillation."
)

parser.add_argument(
    "--pretained", default=False, action="store_true"
)

parser.add_argument(
    "--cuda_id", default=0, type=int
)

parser.add_argument(
    "--global_model", required=True, type=str, choices=all_models
)

parser.add_argument(
    "--global_model_rank", required=True, type=int
)

parser.add_argument(
    "--local_model_rank", required=True, type=int
)

parser.add_argument(
    "--local_model", required=True, type=str, choices=all_models
)
parser.add_argument(
    "--KD_SLF_weight", default=1.0, type=float, help="Specifying the weight of KGE loss."
)
parser.add_argument(
    "--KD_EED_weight", default=1.0, type=float, help="Specifying the weight of KGE loss."
)
parser.add_argument(
    "--KD_omega1", default=0.5, type=float, help="Specifying the weight of KGE loss."
)
parser.add_argument(
    "--KD_omega2", default=0.5, type=float, help="Specifying the weight of KGE loss."
)
parser.add_argument(
    "--feat_kd_weight", default=0.0, type=float
)




def train(args):
    save_dir = get_savedir(args.model, args.dataset)

    # file logger
    logging.basicConfig(
        format="%(asctime)s %(levelname)-8s %(message)s",
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
        filename=os.path.join(save_dir, "train.log")
    )

    # stdout logger
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(levelname)-8s %(message)s")
    console.setFormatter(formatter)
    logging.getLogger("").addHandler(console)
    logging.info("Saving logs in: {}".format(save_dir))

    # create dataset
    dataset_path = os.path.join("./KGE_datasets/data", args.dataset)
    dataset = KGDataset(dataset_path, args.debug)
    args.sizes = dataset.get_shape()

    # load data
    logging.info("\t " + str(dataset.get_shape()))
    train_examples = dataset.get_examples("train")
    valid_examples = dataset.get_examples("valid")
    test_examples = dataset.get_examples("test")
    filters = dataset.get_filters()

    # save config
    with open(os.path.join(save_dir, "config.json"), "w") as fjson:
        json.dump(vars(args), fjson, indent=4)
    
    # copy model code file to log folder
    folder_path = os.path.split(os.path.abspath(__file__))[0]
    shutil.copytree(os.path.join(folder_path, "models"), os.path.join(save_dir, "code", "models"))
    shutil.copytree(os.path.join(folder_path, "utils"), os.path.join(save_dir, "code", "utils"))
    shutil.copytree(os.path.join(folder_path, "optimizers"), os.path.join(save_dir, "code", "optimizers"))


    # set deivce
    device = torch.device(f"cuda:{args.cuda_id}" if torch.cuda.is_available() else 'cpu')
    args.device = device

    # create model
    model = getattr(models, args.model)(args)
    total = count_params(model)
    logging.info("Total number of parameters {}".format(total))
    model.to(device)

    # get optimizer
    regularizer = getattr(regularizers, args.regularizer)(args.reg)
    optim_method = getattr(torch.optim, args.optimizer)(model.parameters(), lr=args.learning_rate)
    optimizer = KGOptimizer(model, regularizer, optim_method, bool(args.double_neg), args)
    scheduler = ReduceLROnPlateau(optim_method, 'min', factor=0.5, verbose=True, patience=10, threshold=1e-3)
    counter = 0
    if args.pretained == True:
        noml = False
    else:
        noml = True
    best_mrr = None
    best_epoch = None
    logging.info("\t Start training")
    for step in range(args.max_epochs):

        # Train step
        model.train()
        train_loss = optimizer.epoch(train_examples, noml)
        logging.info("\t Epoch {} | average train loss: {:.4f}".format(step, train_loss))

        # Valid step
        model.eval()
        valid_loss = optimizer.calculate_valid_loss(valid_examples, noml)
        logging.info("\t Epoch {} | average valid loss: {:.4f}".format(step, valid_loss))

        if (step + 1) % args.valid == 0:
            valid_metrics_both, valid_metrics_global, valid_metrics_local = model.compute_metrics(valid_examples, filters, batch_size=args.valid_batchsize)
            test_metrics_both, test_metrics_global, test_metrics_local = model.compute_metrics(test_examples, filters)

            valid_metrics_global = avg_both(*valid_metrics_global)
            valid_metrics_local = avg_both(*valid_metrics_local)
            valid_metrics_both = avg_both(*valid_metrics_both)
            test_metrics_global = avg_both(*test_metrics_global)
            test_metrics_local = avg_both(*test_metrics_local)
            test_metrics_both = avg_both(*test_metrics_both)

            logging.info(f'global({args.global_model}):' + format_metrics(valid_metrics_global, split="valid"))
            logging.info(f'local ({args.local_model}): ' + format_metrics(valid_metrics_local, split="valid"))
            logging.info('both:  ' + format_metrics(valid_metrics_both, split="valid"))
            
            logging.info(f'global({args.global_model}):' + format_metrics(test_metrics_global, split="test"))
            logging.info(f'local ({args.local_model}): ' + format_metrics(test_metrics_local, split="test"))
            logging.info('both:  ' + format_metrics(test_metrics_both, split="test"))

            # alpha = cal_alpha(valid_metrics_global["MRR"], valid_metrics_local["MRR"], valid_metrics_both["MRR"])
            # optimizer.alpha = alpha

            # logging.info(f'alpha: {alpha:.4f}')


            valid_mrr = valid_metrics_both["MRR"]
            if not best_mrr or valid_mrr > best_mrr:
                best_mrr = valid_mrr
                counter = 0
                best_epoch = step
                logging.info("\t Saving model at epoch {} in {}".format(step, save_dir))
                torch.save(model.cpu().state_dict(), os.path.join(save_dir, "model.pt"))
                model.to(device)
            else:
                counter += 1
                if counter == args.patience:
                    logging.info("\t Early stopping")
                    break
                elif counter == args.patience // 2:
                    pass
                    # logging.info("\t Reducing learning rate")
                    # optimizer.reduce_lr()
                
                scheduler.step(valid_metrics_both['MRR'])

            if (valid_mrr - best_mrr <= 1e-3) and (step + 1) > args.KD_iter:
                noml = False
            
    logging.info("\t Optimization finished")
    if not best_mrr:
        torch.save(model.cpu().state_dict(), os.path.join(save_dir, "model.pt"))
    else:
        logging.info("\t Loading best model saved at epoch {}".format(best_epoch))
        model.load_state_dict(torch.load(os.path.join(save_dir, "model.pt")))
    model.to(device)
    model.eval()

    # Validation metrics
    valid_metrics = avg_both(*model.compute_metrics(valid_examples, filters, batch_size=args.valid_batchsize))
    logging.info(format_metrics(valid_metrics, split="valid"))

    # Test metrics
    test_metrics = avg_both(*model.compute_metrics(test_examples, filters, batch_size=args.valid_batchsize))
    logging.info(format_metrics(test_metrics, split="test"))


if __name__ == "__main__":
    train(parser.parse_args())
