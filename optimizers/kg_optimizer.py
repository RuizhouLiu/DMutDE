"""Knowledge Graph embedding model optimizer."""
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
from torch import nn
from models import *


class KGOptimizer(object):
    """Knowledge Graph embedding model optimizer.

    KGOptimizers performs loss computations with negative sampling and gradient descent steps.

    Attributes:
        model: models.base.KGModel
        regularizer: regularizers.Regularizer
        optimizer: torch.optim.Optimizer
        batch_size: An integer for the training batch size
        neg_sample_size: An integer for the number of negative samples
        double_neg: A boolean (True to sample both head and tail entities)
    """

    def __init__(
            self, model: EnsembleModel, regularizer, optimizer, double_neg, args, verbose=True):
        """Inits KGOptimizer."""
        self.model = model
        self.regularizer = regularizer
        self.optimizer = optimizer
        self.batch_size = args.batch_size
        self.verbose = verbose
        self.double_neg = double_neg
        self.loss_fn = nn.CrossEntropyLoss(reduction='mean')
        self.neg_sample_size = args.neg_sample_size
        self.n_entities = model.sizes[0]
        self.device = self.model.device
        self.KD_SLF_weight = args.KD_SLF_weight
        self.KD_EED_weight = args.KD_EED_weight
        self.local_kd_weight = 0.06
        self.global_kd_weight = 0.06
        self.neg_kd_weight = 0.1
        self.args = args

    def reduce_lr(self, factor=0.8):
        """Reduce learning rate.

        Args:
            factor: float for the learning rate decay
        """
        for param_group in self.optimizer.param_groups:
            param_group['lr'] *= factor

    def get_neg_samples(self, input_batch):
        """Sample negative examples.

        Args:
            input_batch: torch.LongTensor of shape (batch_size x 3) with ground truth training triples

        Returns:
            negative_batch: torch.Tensor of shape (neg_sample_size x 3) with negative examples
        """
        negative_batch = input_batch.repeat(self.neg_sample_size, 1)
        batch_size = input_batch.shape[0]
        negsamples = torch.Tensor(np.random.randint(
            self.n_entities,
            size=batch_size * self.neg_sample_size)
        ).to(input_batch.dtype)
        negative_batch[:, 2] = negsamples
        if self.double_neg:
            negsamples = torch.Tensor(np.random.randint(
                self.n_entities,
                size=batch_size * self.neg_sample_size)
            ).to(input_batch.dtype)
            negative_batch[:, 0] = negsamples
        
        negative_batch = negative_batch.view(self.neg_sample_size, -1, 3).permute((1,0,2)).reshape(-1, 3)
        return negative_batch

    def neg_sampling_loss(self, input_batch):
        """Compute KG embedding loss with negative sampling.

        Args:
            input_batch: torch.LongTensor of shape (batch_size x 3) with ground truth training triples.

        Returns:
            loss: torch.Tensor with negative sampling embedding loss
            factors: torch.Tensor with embeddings weights to regularize
        """

        # local model be teacher, global model be student.
        # positive samples
        pos_score_local, pos_score_global, factors_local, factors_global = self.model(input_batch)
        neg_score_local, neg_score_global, _, _ = self.model(neg_samples)

        kge_loss_global = - torch.cat([pos_pred_global, neg_pred_global], dim=0).mean()






        pos_pred_local = F.logsigmoid(pos_score_local)
        pos_pred_global = F.logsigmoid(pos_score_global)


        # negative samples 
        neg_samples = self.get_neg_samples(input_batch)
        


        neg_pred_local = F.logsigmoid(-neg_score_local)
        neg_pred_global = F.logsigmoid(-neg_score_global)





        loss = - (torch.cat([pos_pred_local, neg_pred_local], dim=0).mean() + 
                  torch.cat([pos_pred_global, neg_pred_global], dim=0).mean())
        return loss, factors_local, factors_global

    def no_neg_sampling_loss(self, input_batch):
        """Compute KG embedding loss without negative sampling.

        Args:
            input_batch: torch.LongTensor of shape (batch_size x 3) with ground truth training triples

        Returns:
            loss: torch.Tensor with embedding loss
            factors: torch.Tensor with embeddings weights to regularize
        """
        predictions, factors = self.model(input_batch, eval_mode=True)
        truth = input_batch[:, 2]
        log_prob = F.logsigmoid(-predictions)
        idx = torch.arange(0, truth.shape[0], dtype=truth.dtype)
        pos_scores = F.logsigmoid(predictions[idx, truth]) - F.logsigmoid(-predictions[idx, truth])
        log_prob[idx, truth] += pos_scores
        loss = - log_prob.mean()
        loss += self.regularizer.forward(factors)
        return loss, factors

    def calculate_loss(self, input_batch):
        """Compute KG embedding loss and regularization loss.

        Args:
            input_batch: torch.LongTensor of shape (batch_size x 3) with ground truth training triples

        Returns:
            loss: torch.Tensor with embedding loss and regularization loss
        """
        if self.neg_sample_size > 0:
            loss, factors_local, factors_global = self.neg_sampling_loss(input_batch)
        else:
            predictions, factors = self.model(input_batch, eval_mode=True)
            truth = input_batch[:, 2]
            loss = self.loss_fn(predictions, truth)
            # loss, factors = self.no_neg_sampling_loss(input_batch)

        # regularization loss
        loss += self.regularizer.forward(factors_local)
        loss += self.regularizer.forward(factors_global)
        return loss

    def calculate_valid_loss(self, examples, noml=True):
        """Compute KG embedding loss over validation examples.

        Args:
            examples: torch.LongTensor of shape (N_valid x 3) with validation triples

        Returns:
            loss: torch.Tensor with loss averaged over all validation examples
        """
        b_begin = 0
        loss = 0.0
        counter = 0
        with torch.no_grad():
            while b_begin < examples.shape[0]:
                input_batch = examples[
                              b_begin:b_begin + self.batch_size
                              ].to(self.device)
                b_begin += self.batch_size
                ### gradient step
                neg_samples = self.get_neg_samples(input_batch)

                if not noml:
                    ## global be a teacher, local be a student
                    pos_score_local, factors_local = self.model.local_model(input_batch)
                    neg_score_local, _ = self.model.local_model(neg_samples)
                    neg_score_local = neg_score_local.view(-1, self.neg_sample_size)

                    pos_score_global, factors_global = self.model.global_model(input_batch)
                    neg_score_global, _ = self.model.global_model(neg_samples)
                    neg_score_global = neg_score_global.view(-1, self.neg_sample_size)

                    # cal kge loss for local model
                    pos_pred_local =  F.logsigmoid(pos_score_local)
                    neg_pred_local = F.logsigmoid(-neg_score_local)
                    kge_loss_local = - torch.cat([pos_pred_local, neg_pred_local], dim=-1).mean()
                    kd_loss_local = self.model.mutual_learning(score_t=(pos_score_global, neg_score_global), score_s=(pos_score_local, neg_score_local))
                    reg_loss_local = self.regularizer.forward(factors_local)

                    loss_local = kge_loss_local + kd_loss_local + reg_loss_local

                    ## local be a teacher, global be a student
                    pos_score_local, _ = self.model.local_model(input_batch)
                    neg_score_local, _ = self.model.local_model(neg_samples)
                    neg_score_local = neg_score_local.view(-1, self.neg_sample_size)

                    # cal kge loss for global model
                    pos_pred_global = F.logsigmoid(pos_score_global)
                    neg_pred_global = F.logsigmoid(-neg_score_global)
                    kge_loss_global = - torch.cat([pos_pred_global, neg_pred_global], dim=-1).mean()
                    reg_loss_global = self.regularizer.forward(factors_global)
                    kd_loss_global = self.model.mutual_learning(score_t=(pos_score_local, neg_score_local), score_s=(pos_score_global, neg_score_global))

                    loss_global = kge_loss_global + kd_loss_global + reg_loss_global

                    loss = loss_global + loss_local
                else:
                    pos_score_local, pos_score_global, factors_local, factors_global = self.model(input_batch)
                    neg_score_local, neg_score_global, _, _ = self.model(neg_samples)

                    pos_pred_local = F.logsigmoid(pos_score_local)
                    pos_pred_global = F.logsigmoid(pos_score_global)

                    neg_pred_local = F.logsigmoid(-neg_score_local)
                    neg_pred_global = F.logsigmoid(-neg_score_global)

                    loss = - (torch.cat([pos_pred_local, neg_pred_local], dim=0).mean() + 
                              torch.cat([pos_pred_global, neg_pred_global], dim=0).mean())
                    
                    loss += self.regularizer.forward(factors_local)
                    loss += self.regularizer.forward(factors_global)

                counter += 1
        loss /= counter
        return loss

    def epoch(self, examples, noml=True):
        """Runs one epoch of training KG embedding model.

        Args:
            examples: torch.LongTensor of shape (N_train x 3) with training triples

        Returns:
            loss: torch.Tensor with loss averaged over all training examples
        """
        actual_examples = examples[torch.randperm(examples.shape[0]), :]
        with tqdm.tqdm(total=examples.shape[0], unit='ex', disable=not self.verbose) as bar:
            bar.set_description(f'train loss')
            b_begin = 0
            total_loss = 0.0
            counter = 0
            while b_begin < examples.shape[0]:
                input_batch = actual_examples[
                              b_begin:b_begin + self.batch_size
                              ].to(self.device)

                ### gradient step
                neg_samples = self.get_neg_samples(input_batch)

                if not noml:
                    ## global be a teacher, local be a student
                    pos_score_local, factors_local = self.model.local_model(input_batch)
                    neg_score_local, _ = self.model.local_model(neg_samples)
                    neg_score_local = neg_score_local.view(-1, self.neg_sample_size)

                    pos_score_global, factors_global = self.model.global_model(input_batch)
                    neg_score_global, _ = self.model.global_model(neg_samples)
                    neg_score_global = neg_score_global.view(-1, self.neg_sample_size)

                    # cal kge loss for local model
                    pos_pred_local =  F.logsigmoid(pos_score_local)
                    neg_pred_local = F.logsigmoid(-neg_score_local)
                    kge_loss_local = - torch.cat([pos_pred_local, neg_pred_local], dim=-1).mean()
                    # kd_loss_local1 = self.model.mutual_learning_adv_v2(score_t=(pos_score_global, neg_score_global), score_s=(pos_score_local, neg_score_local))
                    # kd_loss_local1 = torch.tensor(0.0)
                    # mix_kd_loss_local = self.model.mutual_learning_adv(score_s=(pos_score_local, neg_score_local), score_t=(mix_pos_score, mix_neg_score))
                    # kd_loss_local1 = self.model.mutual_learning_v2(pos_samples=input_batch, neg_samples=neg_samples.view(-1, self.neg_sample_size, 3),
                    #                                                model_t=self.model.global_model, model_s=self.model.local_model)
                    kd_loss_local1 = self.model.feature_learning(pos_samples=input_batch, neg_samples=neg_samples.view(-1, self.neg_sample_size, 3),
                                                                   model_t=self.model.global_model, model_s=self.model.local_model)
                    kd_loss_local2 = self.model.mutual_learning(score_t=(pos_score_global, neg_score_global), score_s=(pos_score_local, neg_score_local))
                    reg_loss_local = self.regularizer.forward(factors_local)
                    
                    loss_local = (1. - self.local_kd_weight) * (kge_loss_local + reg_loss_local) + self.local_kd_weight * (self.KD_EED_weight * kd_loss_local1  + self.KD_SLF_weight * kd_loss_local2)

                    self.optimizer.zero_grad()
                    loss_local.backward()
                    self.optimizer.step()

                    ## local be a teacher, global be a student
                    pos_score_local, _ = self.model.local_model(input_batch)
                    neg_score_local, _ = self.model.local_model(neg_samples)
                    neg_score_local = neg_score_local.view(-1, self.neg_sample_size)

                    # cal kge loss for global model
                    pos_pred_global = F.logsigmoid(pos_score_global)
                    neg_pred_global = F.logsigmoid(-neg_score_global)
                    kge_loss_global = - torch.cat([pos_pred_global, neg_pred_global], dim=-1).mean()
                    # kd_loss_global1 = self.model.mutual_learning_adv_v2(score_t=(pos_score_local, neg_score_local), score_s=(pos_score_global, neg_score_global))
                    # kd_loss_global1 = torch.tensor(0.0)
                    # mix_kd_loss_global = self.model.mutual_learning_adv(score_s=(pos_score_global, neg_score_global), score_t=(mix_pos_score, mix_neg_score))
                    # kd_loss_global1 = self.model.mutual_learning_v2(pos_samples=input_batch, neg_samples=neg_samples.view(-1, self.neg_sample_size, 3),
                    #                                                 model_t=self.model.local_model, model_s=self.model.global_model)
                    kd_loss_global1 = self.model.feature_learning(pos_samples=input_batch, neg_samples=neg_samples.view(-1, self.neg_sample_size, 3),
                                                                    model_t=self.model.local_model, model_s=self.model.global_model)
                    kd_loss_global2 = self.model.mutual_learning(score_t=(pos_score_local, neg_score_local), score_s=(pos_score_global, neg_score_global))
                    reg_loss_global = self.regularizer.forward(factors_global)

                    loss_global = (1. - self.global_kd_weight) * (kge_loss_global + reg_loss_global) + self.global_kd_weight * (self.KD_EED_weight * kd_loss_global1 + self.KD_SLF_weight * kd_loss_global2)

                    self.optimizer.zero_grad()
                    loss_global.backward()
                    self.optimizer.step()

                    l = loss_global + loss_local

                    kge_l = kge_loss_global + kge_loss_local
                    kd_l = kd_loss_global1 + kd_loss_global2 + kd_loss_local1 + kd_loss_local2

                else:
                    pos_score_local, pos_score_global, factors_local, factors_global = self.model(input_batch)
                    neg_score_local, neg_score_global, _, _ = self.model(neg_samples)

                    pos_pred_local = F.logsigmoid(pos_score_local)
                    pos_pred_global = F.logsigmoid(pos_score_global)

                    neg_pred_local = F.logsigmoid(-neg_score_local)
                    neg_pred_global = F.logsigmoid(-neg_score_global)

                    l = - (torch.cat([pos_pred_local, neg_pred_local], dim=0).mean() + 
                              torch.cat([pos_pred_global, neg_pred_global], dim=0).mean())
                    
                    l += self.regularizer.forward(factors_local)
                    l += self.regularizer.forward(factors_global)

                    kge_l = l.clone()

                    # kd_loss_local1 = self.model.mutual_learning_adv_v2(score_t=(pos_score_global, neg_score_global.view(-1, self.neg_sample_size)), score_s=(pos_score_local, neg_score_local.view(-1, self.neg_sample_size)))
                    # kd_loss_global1 = self.model.mutual_learning_adv_v2(score_t=(pos_score_local, neg_score_local.view(-1, self.neg_sample_size)), score_s=(pos_score_global, neg_score_global.view(-1, self.neg_sample_size)))
                    kd_loss_local1 = torch.tensor(0.0)
                    kd_loss_global1 = torch.tensor(0.0)

                    kd_l =  kd_loss_local1 + kd_loss_global1

                    l += kd_loss_local1
                    l += kd_loss_global1

                    self.optimizer.zero_grad()
                    l.backward()
                    self.optimizer.step()

                # dist = self.model.dist()
                dist = torch.tensor(0.0) 
                b_begin += self.batch_size
                total_loss += l
                counter += 1
                bar.update(input_batch.shape[0])
                bar.set_postfix(
                    kge_l=f'{kge_l.item():.4f}', 
                    kd_l=f'{kd_l.item():.4f}', 
                    dist=f'{dist.item():.4f}'
                )
        total_loss /= counter
        return total_loss
