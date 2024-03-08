"""Hyperbolic Knowledge Graph embedding models where all parameters are defined in tangent spaces."""
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from typing import Dict, Tuple, List
import tqdm
from copy import deepcopy

import models
import os
from models.base import KGModel
from utils.euclidean import givens_rotations, givens_reflection
from utils.hyperbolic import mobius_add, expmap0, project, hyp_distance_multi_c, expmapP, expmapH, projectH

HYP_MODELS = ["RotH", "RefH", "AttH", "FieldP", "FieldH", "EnsembleModel", "EnsembleModel_v2", "LocRotH", "LocAttH", "LocRefH"]


class BaseH(KGModel):
    """Trainable curvature for each relationship."""

    def __init__(self, args):
        super(BaseH, self).__init__(args.sizes, args.rank, args.dropout, args.gamma, args.dtype, args.bias,
                                    args.init_size)
        self.entity.weight.data = self.init_size * torch.randn((self.sizes[0], self.rank), dtype=self.data_type)
        self.rel.weight.data = self.init_size * torch.randn((self.sizes[1], 2 * self.rank), dtype=self.data_type)
        self.rel_diag = nn.Embedding(self.sizes[1], self.rank)
        self.rel_diag.weight.data = 2 * torch.rand((self.sizes[1], self.rank), dtype=self.data_type) - 1.0
        self.multi_c = args.multi_c
        if self.multi_c:
            c_init = torch.ones((self.sizes[1], 1), dtype=self.data_type)
        else:
            c_init = torch.ones((1, 1), dtype=self.data_type)
        self.c = nn.Parameter(c_init, requires_grad=True)

    def get_rhs(self, queries, eval_mode):
        """Get embeddings and biases of target entities."""
        if eval_mode:
            return self.entity.weight, self.bt.weight
        else:
            return self.entity(queries[:, 2]), self.bt(queries[:, 2])

    def similarity_score(self, lhs_e, rhs_e, eval_mode):
        """Compute similarity scores or queries against targets in embedding space."""
        lhs_e, c = lhs_e
        return - hyp_distance_multi_c(lhs_e, rhs_e, c, eval_mode) ** 2


class RotH(BaseH):
    """Hyperbolic 2x2 Givens rotations"""

    def get_queries(self, queries):
        """Compute embedding and biases of queries."""
        c = F.softplus(self.c[queries[:, 1]])
        head = expmap0(self.entity(queries[:, 0]), c)
        rel1, rel2 = torch.chunk(self.rel(queries[:, 1]), 2, dim=1)
        rel1 = expmap0(rel1, c)
        rel2 = expmap0(rel2, c)
        lhs = project(mobius_add(head, rel1, c), c)
        res1 = givens_rotations(self.rel_diag(queries[:, 1]), lhs)
        res2 = mobius_add(res1, rel2, c)
        return (res2, c), self.bh(queries[:, 0])


class LocRotH(BaseH):
    def __init__(self, args):
        super().__init__(args)

        self.trans1 = nn.Embedding(self.sizes[1], 2 * self.rank)
        self.trans2 = nn.Embedding(self.sizes[1], 2 * self.rank)
        self.trans3 = nn.Embedding(self.sizes[1], 2 * self.rank)
        self.trans = nn.Embedding(self.sizes[1], self.rank)

        self.rot1 = nn.Embedding(self.sizes[1], self.rank)
        self.rot2 = nn.Embedding(self.sizes[1], self.rank)
        self.rot3 = nn.Embedding(self.sizes[1], self.rank)

        self.c1 = nn.Parameter(0.001 * torch.randn((self.sizes[1], 1), dtype=self.data_type) + 1., requires_grad=True)
        self.c2 = nn.Parameter(0.001 * torch.randn((self.sizes[1], 1), dtype=self.data_type) + 1., requires_grad=True)
        self.c3 = nn.Parameter(0.001 * torch.randn((self.sizes[1], 1), dtype=self.data_type) + 1., requires_grad=True)

        self.context_vec = nn.Embedding(self.sizes[1], self.rank)
        self.act = nn.Softmax(dim=1)

        if args.dtype == "double":
            self.scale = torch.Tensor([1. / np.sqrt(self.rank)]).double().to(args.device)
        else:
            self.scale = torch.Tensor([1. / np.sqrt(self.rank)]).to(args.device)

        self.param_init()
    
    def param_init(self):
        nn.init.normal_(self.trans1.weight, std=self.init_size)
        nn.init.normal_(self.trans2.weight, std=self.init_size)
        nn.init.normal_(self.trans3.weight, std=self.init_size)
        nn.init.normal_(self.trans.weight, std=self.init_size)

        nn.init.normal_(self.rot1.weight, mean=-1)
        nn.init.normal_(self.rot2.weight, mean=-1)
        nn.init.normal_(self.rot3.weight, mean=-1)
        self.context_vec.weight.data = self.init_size * torch.randn((self.sizes[1], self.rank), dtype=self.data_type)
    
    def get_queries(self, queries):
        ########### prepare basic elements ##########
        c1 = F.softplus(self.c1[queries[:, 1]])
        c2 = F.softplus(self.c2[queries[:, 1]])
        c3 = F.softplus(self.c3[queries[:, 1]])

        lhs = self.entity(queries[:, 0])
        lhs_c1 = expmap0(lhs, c1)
        lhs_c2 = expmap0(lhs, c2)
        lhs_c3 = expmap0(lhs, c3)

        trans1_c1, trans2_c1 = torch.chunk(self.trans1(queries[:, 1]), 2, dim=1)
        trans1_c2, trans2_c2 = torch.chunk(self.trans2(queries[:, 1]), 2, dim=1)
        trans1_c3, trans2_c3 = torch.chunk(self.trans3(queries[:, 1]), 2, dim=1)
        trans1_c1, trans2_c1 = expmap0(trans1_c1, c1), expmap0(trans2_c1, c1)
        trans1_c2, trans2_c2 = expmap0(trans1_c2, c2), expmap0(trans2_c2, c2)
        trans1_c3, trans2_c3 = expmap0(trans1_c3, c3), expmap0(trans2_c3, c3)

        rot_c1 = self.rot1(queries[:, 1])
        rot_c2 = self.rot2(queries[:, 1])
        rot_c3 = self.rot3(queries[:, 1])

        context_vec = self.context_vec(queries[:, 1]).view((-1, 1, self.rank))

        ####### transformation ########

        lhs_c1 = mobius_add(lhs_c1, trans1_c1, c1)
        res1 = givens_rotations(rot_c1, lhs_c1)
        res1 = mobius_add(res1, trans2_c1, c1)

        lhs_c2 = mobius_add(lhs_c2, trans1_c2, c2)
        res2 = givens_rotations(rot_c2, lhs_c2)
        res2 = mobius_add(res2, trans2_c2, c2)

        lhs_c3 = mobius_add(lhs_c3, trans1_c3, c3)
        res3 = givens_rotations(rot_c3, lhs_c3)
        res3 = mobius_add(res3, trans2_c3, c3)

        res1 = res1.unsqueeze(1)
        res2 = res2.unsqueeze(1)
        res3 = res3.unsqueeze(1)

        cands = torch.cat([res1, res2, res3], dim=1)
        cands_c = torch.cat([c1, c2, c3], dim=-1).unsqueeze(-1)
        att_weights = torch.sum(context_vec * cands * self.scale, dim=-1, keepdim=True)
        att_weights = self.act(att_weights)
        att_q = torch.sum(att_weights * cands, dim=1)
        c = torch.sum(att_weights * cands_c, dim=1)

        trans_c = expmap0(self.trans(queries[:, 1]), c)
        att_q = mobius_add(att_q, trans_c, c)

        return (att_q, c), self.bh(queries[:, 0])


class LocRotH(BaseH):
    def __init__(self, args):
        super().__init__(args)

        self.trans1 = nn.Embedding(self.sizes[1], 2 * self.rank)
        self.trans2 = nn.Embedding(self.sizes[1], 2 * self.rank)
        self.trans3 = nn.Embedding(self.sizes[1], 2 * self.rank)
        self.trans = nn.Embedding(self.sizes[1], self.rank)

        self.rot1 = nn.Embedding(self.sizes[1], self.rank)
        self.rot2 = nn.Embedding(self.sizes[1], self.rank)
        self.rot3 = nn.Embedding(self.sizes[1], self.rank)

        self.c1 = nn.Parameter(0.001 * torch.randn((self.sizes[1], 1), dtype=self.data_type) + 1., requires_grad=True)
        self.c2 = nn.Parameter(0.001 * torch.randn((self.sizes[1], 1), dtype=self.data_type) + 1., requires_grad=True)
        self.c3 = nn.Parameter(0.001 * torch.randn((self.sizes[1], 1), dtype=self.data_type) + 1., requires_grad=True)

        self.context_vec = nn.Embedding(self.sizes[1], self.rank)
        self.act = nn.Softmax(dim=1)

        if args.dtype == "double":
            self.scale = torch.Tensor([1. / np.sqrt(self.rank)]).double().to(args.device)
        else:
            self.scale = torch.Tensor([1. / np.sqrt(self.rank)]).to(args.device)

        self.param_init()
    
    def param_init(self):
        nn.init.normal_(self.trans1.weight, std=self.init_size)
        nn.init.normal_(self.trans2.weight, std=self.init_size)
        nn.init.normal_(self.trans3.weight, std=self.init_size)
        nn.init.normal_(self.trans.weight, std=self.init_size)

        nn.init.normal_(self.rot1.weight, mean=-1)
        nn.init.normal_(self.rot2.weight, mean=-1)
        nn.init.normal_(self.rot3.weight, mean=-1)
        self.context_vec.weight.data = self.init_size * torch.randn((self.sizes[1], self.rank), dtype=self.data_type)
    
    def get_queries(self, queries):
        ########### prepare basic elements ##########
        c1 = F.softplus(self.c1[queries[:, 1]])
        c2 = F.softplus(self.c2[queries[:, 1]])
        c3 = F.softplus(self.c3[queries[:, 1]])

        lhs = self.entity(queries[:, 0])
        lhs_c1 = expmap0(lhs, c1)
        lhs_c2 = expmap0(lhs, c2)
        lhs_c3 = expmap0(lhs, c3)

        trans1_c1, trans2_c1 = torch.chunk(self.trans1(queries[:, 1]), 2, dim=1)
        trans1_c2, trans2_c2 = torch.chunk(self.trans2(queries[:, 1]), 2, dim=1)
        trans1_c3, trans2_c3 = torch.chunk(self.trans3(queries[:, 1]), 2, dim=1)
        trans1_c1, trans2_c1 = expmap0(trans1_c1, c1), expmap0(trans2_c1, c1)
        trans1_c2, trans2_c2 = expmap0(trans1_c2, c2), expmap0(trans2_c2, c2)
        trans1_c3, trans2_c3 = expmap0(trans1_c3, c3), expmap0(trans2_c3, c3)

        rot_c1 = self.rot1(queries[:, 1])
        rot_c2 = self.rot2(queries[:, 1])
        rot_c3 = self.rot3(queries[:, 1])

        context_vec = self.context_vec(queries[:, 1]).view((-1, 1, self.rank))

        ####### transformation ########

        lhs_c1 = mobius_add(lhs_c1, trans1_c1, c1)
        res1 = givens_rotations(rot_c1, lhs_c1)
        res1 = mobius_add(res1, trans2_c1, c1)

        lhs_c2 = mobius_add(lhs_c2, trans1_c2, c2)
        res2 = givens_rotations(rot_c2, lhs_c2)
        res2 = mobius_add(res2, trans2_c2, c2)

        lhs_c3 = mobius_add(lhs_c3, trans1_c3, c3)
        res3 = givens_rotations(rot_c3, lhs_c3)
        res3 = mobius_add(res3, trans2_c3, c3)

        res1 = res1.unsqueeze(1)
        res2 = res2.unsqueeze(1)
        res3 = res3.unsqueeze(1)

        cands = torch.cat([res1, res2, res3], dim=1)
        cands_c = torch.cat([c1, c2, c3], dim=-1).unsqueeze(-1)
        att_weights = torch.sum(context_vec * cands * self.scale, dim=-1, keepdim=True)
        att_weights = self.act(att_weights)
        att_q = torch.sum(att_weights * cands, dim=1)
        c = torch.sum(att_weights * cands_c, dim=1)

        trans_c = expmap0(self.trans(queries[:, 1]), c)
        att_q = mobius_add(att_q, trans_c, c)

        return (att_q, c), self.bh(queries[:, 0])


class LocAttH(BaseH):
    def __init__(self, args):
        super().__init__(args)

        self.rot1 = nn.Embedding(self.sizes[1], self.rank)
        self.rot2 = nn.Embedding(self.sizes[1], self.rank)
        self.rot3 = nn.Embedding(self.sizes[1], self.rank)

        self.ref1 = nn.Embedding(self.sizes[1], self.rank)
        self.ref2 = nn.Embedding(self.sizes[1], self.rank)
        self.ref3 = nn.Embedding(self.sizes[1], self.rank)

        self.trans = nn.Embedding(self.sizes[1], self.rank)
        self.trans1 = nn.Embedding(self.sizes[1], self.rank)
        self.trans2 = nn.Embedding(self.sizes[1], self.rank)
        self.trans3 = nn.Embedding(self.sizes[1], self.rank)

        self.c1 = nn.Parameter(0.001 * torch.randn((self.sizes[1], 1), dtype=self.data_type) + 1., requires_grad=True)
        self.c2 = nn.Parameter(0.001 * torch.randn((self.sizes[1], 1), dtype=self.data_type) + 1., requires_grad=True)
        self.c3 = nn.Parameter(0.001 * torch.randn((self.sizes[1], 1), dtype=self.data_type) + 1., requires_grad=True)

        self.context_vec1 = nn.Embedding(self.sizes[1], self.rank)
        self.context_vec2 = nn.Embedding(self.sizes[1], self.rank)
        self.context_vec3 = nn.Embedding(self.sizes[1], self.rank)

        self.gie_vec = nn.Embedding(self.sizes[1], self.rank)
        self.act = nn.Softmax(dim=1)

        if args.dtype == "double":
            self.context_scale1 = torch.Tensor([1. / np.sqrt(self.rank)]).double().to(args.device)
            self.context_scale2 = torch.Tensor([1. / np.sqrt(self.rank)]).double().to(args.device)
            self.context_scale3 = torch.Tensor([1. / np.sqrt(self.rank)]).double().to(args.device)
            self.gie_scale = torch.Tensor([1. / np.sqrt(self.rank)]).double().to(args.device)
        else:
            self.context_scale1 = torch.Tensor([1. / np.sqrt(self.rank)]).to(args.device)
            self.context_scale2 = torch.Tensor([1. / np.sqrt(self.rank)]).to(args.device)
            self.context_scale3 = torch.Tensor([1. / np.sqrt(self.rank)]).to(args.device)
            self.gie_scale = torch.Tensor([1. / np.sqrt(self.rank)]).to(args.device)
        
        self.param_init()
        
    def param_init(self):
        nn.init.normal_(self.rot1.weight, mean=-1)
        nn.init.normal_(self.rot2.weight, mean=-1)
        nn.init.normal_(self.rot3.weight, mean=-1)

        nn.init.normal_(self.ref1.weight, mean=-1)
        nn.init.normal_(self.ref2.weight, mean=-1)
        nn.init.normal_(self.ref3.weight, mean=-1)

        nn.init.normal_(self.trans.weight, std=self.init_size)
        nn.init.normal_(self.trans1.weight, std=self.init_size)
        nn.init.normal_(self.trans2.weight, std=self.init_size)
        nn.init.normal_(self.trans3.weight, std=self.init_size)

        nn.init.normal_(self.context_vec1.weight, std=self.init_size)
        nn.init.normal_(self.context_vec2.weight, std=self.init_size)
        nn.init.normal_(self.context_vec3.weight, std=self.init_size)

        nn.init.normal_(self.gie_vec.weight, std=self.init_size)
    
    def get_queries(self, queries):
        ########### prepare basic elements ##########
        c1 = F.softplus(self.c1[queries[:, 1]])
        c2 = F.softplus(self.c2[queries[:, 1]])
        c3 = F.softplus(self.c3[queries[:, 1]])

        lhs = self.entity(queries[:, 0])

        rot1 = self.rot1(queries[:, 1])
        rot2 = self.rot2(queries[:, 1])
        rot3 = self.rot3(queries[:, 1])

        ref1 = self.ref1(queries[:, 1])
        ref2 = self.ref2(queries[:, 1])
        ref3 = self.ref3(queries[:, 1])

        trans1 = expmap0(self.trans1(queries[:, 1]), c1)
        trans2 = expmap0(self.trans2(queries[:, 1]), c2)
        trans3 = expmap0(self.trans3(queries[:, 1]), c3)

        context_vec1 = self.context_vec1(queries[:, 1]).view((-1, 1, self.rank))
        context_vec2 = self.context_vec2(queries[:, 1]).view((-1, 1, self.rank))
        context_vec3 = self.context_vec3(queries[:, 1]).view((-1, 1, self.rank))
        gie_vec = self.gie_vec(queries[:, 1]).view((-1, 1, self.rank))

        ####### transformation ########

        lhs_ref1 = givens_reflection(ref1, lhs).view((-1, 1, self.rank))
        lhs_rot1 = givens_rotations(rot1, lhs).view((-1, 1, self.rank))
        lhs_cands1 = torch.cat([lhs_ref1, lhs_rot1], dim=1)
        att_weights1 = torch.sum(context_vec1 * lhs_cands1 * self.context_scale1, dim=-1, keepdim=True)
        att_weights1 = self.act(att_weights1)
        att_lhs1 = torch.sum(att_weights1 * lhs_cands1, dim=1)
        att_lhs1 = expmap0(att_lhs1, c1)
        att_lhs1 = mobius_add(att_lhs1, trans1, c1).view((-1, 1, self.rank))

        lhs_ref2 = givens_reflection(ref2, lhs).view((-1, 1, self.rank))
        lhs_rot2 = givens_rotations(rot2, lhs).view((-1, 1, self.rank))
        lhs_cands2 = torch.cat([lhs_ref2, lhs_rot2], dim=1)
        att_weights2 = torch.sum(context_vec2 * lhs_cands2 * self.context_scale2, dim=-1, keepdim=True)
        att_weights2 = self.act(att_weights2)
        att_lhs2 = torch.sum(att_weights2 * lhs_cands2, dim=1)
        att_lhs2 = expmap0(att_lhs2, c2)
        att_lhs2 = mobius_add(att_lhs2, trans2, c2).view((-1, 1, self.rank))

        lhs_ref3 = givens_reflection(ref3, lhs).view((-1, 1, self.rank))
        lhs_rot3 = givens_rotations(rot3, lhs).view((-1, 1, self.rank))
        lhs_cands3 = torch.cat([lhs_ref3, lhs_rot3], dim=1)
        att_weights3 = torch.sum(context_vec3 * lhs_cands3 * self.context_scale3, dim=-1, keepdim=True)
        att_weights3 = self.act(att_weights3)
        att_lhs3 = torch.sum(att_weights3 * lhs_cands3, dim=1)
        att_lhs3 = expmap0(att_lhs3, c3)
        att_lhs3 = mobius_add(att_lhs3, trans3, c3).view((-1, 1, self.rank))

        cands_lhs = torch.cat([att_lhs1, att_lhs2, att_lhs3], dim=1)
        cands_c = torch.cat([c1, c2, c3], dim=-1).unsqueeze(-1)
        gie_weights = torch.sum(gie_vec * cands_lhs * self.gie_scale, dim=-1, keepdim=True)
        gie_weights = self.act(gie_weights)

        gie_q = torch.sum(gie_weights * cands_lhs, dim=1)
        c = torch.sum(gie_weights * cands_c, dim=1)

        trans_c = expmap0(self.trans(queries[:, 1]), c)
        gie_q = mobius_add(gie_q, trans_c, c)

        return (gie_q, c), self.bh(queries[:, 0])
    

class LocRefH(BaseH):
    def __init__(self, args):
        super().__init__(args)

        self.trans = nn.Embedding(self.sizes[1], self.rank)
        self.trans1 = nn.Embedding(self.sizes[1], self.rank)
        self.trans2 = nn.Embedding(self.sizes[1], self.rank)
        self.trans3 = nn.Embedding(self.sizes[1], self.rank)

        self.ref1 = nn.Embedding(self.sizes[1], self.rank)
        self.ref2 = nn.Embedding(self.sizes[1], self.rank)
        self.ref3 = nn.Embedding(self.sizes[1], self.rank)

        self.c1 = nn.Parameter(0.001 * torch.randn((self.sizes[1], 1), dtype=self.data_type) + 1., requires_grad=True)
        self.c2 = nn.Parameter(0.001 * torch.randn((self.sizes[1], 1), dtype=self.data_type) + 1., requires_grad=True)
        self.c3 = nn.Parameter(0.001 * torch.randn((self.sizes[1], 1), dtype=self.data_type) + 1., requires_grad=True)

        self.gie_vec = nn.Embedding(self.sizes[1], self.rank)
        self.act = nn.Softmax(dim=1)

        if args.dtype == "double":
            self.gie_scale = torch.Tensor([1. / np.sqrt(self.rank)]).double().to(args.device)
        else:
            self.gie_scale = torch.Tensor([1. / np.sqrt(self.rank)]).to(args.device)

        self.param_init()
    
    def param_init(self):
        nn.init.normal_(self.trans.weight, std=self.init_size)
        nn.init.normal_(self.trans1.weight, std=self.init_size)
        nn.init.normal_(self.trans2.weight, std=self.init_size)
        nn.init.normal_(self.trans3.weight, std=self.init_size)

        nn.init.normal_(self.ref1.weight, mean=-1)
        nn.init.normal_(self.ref2.weight, mean=-1)
        nn.init.normal_(self.ref3.weight, mean=-1)

        nn.init.normal_(self.gie_vec.weight, std=self.init_size)
    
    def get_queries(self, queries):
        ########### prepare basic elements ##########
        c1 = F.softplus(self.c1[queries[:, 1]])
        c2 = F.softplus(self.c2[queries[:, 1]])
        c3 = F.softplus(self.c3[queries[:, 1]])

        lhs = self.entity(queries[:, 0])

        ref1 = self.ref1(queries[:, 1])
        ref2 = self.ref2(queries[:, 1])
        ref3 = self.ref3(queries[:, 1])

        trans1 = expmap0(self.trans1(queries[:, 1]), c1)
        trans2 = expmap0(self.trans2(queries[:, 1]), c2)
        trans3 = expmap0(self.trans3(queries[:, 1]), c3)

        gie_vec = self.gie_vec(queries[:, 1]).view((-1, 1, self.rank))

        ####### transformation ########

        lhs1 = givens_reflection(ref1, lhs)
        lhs1 = expmap0(lhs1, c1)
        lhs1 = mobius_add(lhs1, trans1, c1)

        lhs2 = givens_reflection(ref2, lhs)
        lhs2 = expmap0(lhs2, c2)
        lhs2 = mobius_add(lhs2, trans2, c2)

        lhs3 = givens_reflection(ref3, lhs)
        lhs3 = expmap0(lhs3, c3)
        lhs3 = mobius_add(lhs3, trans3, c3)

        lhs1 = lhs1.unsqueeze(1)
        lhs2 = lhs2.unsqueeze(1)
        lhs3 = lhs3.unsqueeze(1)

        cands = torch.cat([lhs1, lhs2, lhs3], dim=1)
        cands_c = torch.cat([c1, c2, c3], dim=-1).unsqueeze(-1)
        att_weights = torch.sum(gie_vec * cands * self.gie_scale, dim=-1, keepdim=True)
        att_weights = self.act(att_weights)

        att_q = torch.sum(att_weights * cands, dim=1)
        c = torch.sum(att_weights * cands_c, dim=1)

        trans_c = expmap0(self.trans(queries[:, 1]), c)
        att_q = mobius_add(att_q, trans_c, c)

        return (att_q, c), self.bh(queries[:, 0])
    

class RefH(BaseH):
    """Hyperbolic 2x2 Givens reflections"""

    def get_queries(self, queries):
        """Compute embedding and biases of queries."""
        c = F.softplus(self.c[queries[:, 1]])
        rel, _ = torch.chunk(self.rel(queries[:, 1]), 2, dim=1)
        rel = expmap0(rel, c)
        lhs = givens_reflection(self.rel_diag(queries[:, 1]), self.entity(queries[:, 0]))
        lhs = expmap0(lhs, c)
        res = project(mobius_add(lhs, rel, c), c)
        return (res, c), self.bh(queries[:, 0])

class AttH(BaseH):
    """Hyperbolic attention model combining translations, reflections and rotations"""

    def __init__(self, args):
        super(AttH, self).__init__(args)
        self.rel_diag = nn.Embedding(self.sizes[1], 2 * self.rank)
        self.rel_diag.weight.data = 2 * torch.rand((self.sizes[1], 2 * self.rank), dtype=self.data_type) - 1.0
        self.context_vec = nn.Embedding(self.sizes[1], self.rank)
        self.context_vec.weight.data = self.init_size * torch.randn((self.sizes[1], self.rank), dtype=self.data_type)
        self.act = nn.Softmax(dim=1)
        if args.dtype == "double":
            self.scale = torch.Tensor([1. / np.sqrt(self.rank)]).double().to(args.device)
        else:
            self.scale = torch.Tensor([1. / np.sqrt(self.rank)]).to(args.device)

    def get_queries(self, queries):
        """Compute embedding and biases of queries."""
        c = F.softplus(self.c[queries[:, 1]])
        head = self.entity(queries[:, 0])
        rot_mat, ref_mat = torch.chunk(self.rel_diag(queries[:, 1]), 2, dim=1)
        rot_q = givens_rotations(rot_mat, head).view((-1, 1, self.rank))
        ref_q = givens_reflection(ref_mat, head).view((-1, 1, self.rank))
        cands = torch.cat([ref_q, rot_q], dim=1)
        context_vec = self.context_vec(queries[:, 1]).view((-1, 1, self.rank))
        att_weights = torch.sum(context_vec * cands * self.scale, dim=-1, keepdim=True)
        att_weights = self.act(att_weights)
        att_q = torch.sum(att_weights * cands, dim=1)
        lhs = expmap0(att_q, c)
        rel, _ = torch.chunk(self.rel(queries[:, 1]), 2, dim=1)
        rel = expmap0(rel, c)
        res = project(mobius_add(lhs, rel, c), c)
        return (res, c), self.bh(queries[:, 0])
    
class EnsembleModel(nn.Module):
    def __init__(self, args) -> None:
        super().__init__()
        
        self.sizes = args.sizes
        self.dropout = args.dropout
        self.bias = args.bias
        self.init_size = args.init_size
        self.device = args.device
        self.scaling = nn.Parameter(torch.ones(1))

        self.global_model = self.get_global_model(args)
        self.local_model = self.get_local_model(args)
        # self.local_model = RotH(args=args)
        # self.global_model = AttH(args=args)

        self.local_model.kd_mapping = nn.Sequential(*[
            nn.Linear(self.local_model.rank, self.local_model.rank // 3, dtype=self.local_model.data_type),
            nn.LeakyReLU(inplace=True),
            nn.Linear(self.local_model.rank // 3, self.local_model.rank // 3, dtype=self.local_model.data_type)
        ])

        self.global_model.kd_mapping = nn.Sequential(*[
            nn.Linear(self.global_model.rank, self.global_model.rank // 3, dtype=self.global_model.data_type),
            nn.LeakyReLU(inplace=True),
            nn.Linear(self.global_model.rank // 3, self.global_model.rank // 3, dtype=self.global_model.data_type)
        ])

        if args.pretained == True:
            self.global_model.load_state_dict(torch.load(os.path.join("path_to_pretrain", args.dataset, f"{args.global_model}_rank{args.global_model_rank}", "model.pt")), strict=False)

        self.kl_loss = nn.KLDivLoss(reduction="batchmean")
        self.huber_loss = nn.HuberLoss()

        self.param_init()
    
    def get_local_model(self, args):
        args_cp = deepcopy(args)
        args_cp.rank = args_cp.local_model_rank
        return getattr(models, args.local_model)(args=args_cp)


    def get_global_model(self, args):
        args_cp = deepcopy(args)
        args_cp.rank = args_cp.global_model_rank
        return getattr(models, args.global_model)(args=args_cp)

    
    def param_init(self):
        for name, param in self.local_model.kd_mapping.named_parameters():
            if str.endswith(name, 'weight'):
                nn.init.kaiming_normal_(param)
            elif str.endswith(name, 'bias'):
                nn.init.zeros_(param)
        
        for name, param in self.global_model.kd_mapping.named_parameters():
            if str.endswith(name, 'weight'):
                nn.init.kaiming_normal_(param)
            elif str.endswith(name, 'bias'):
                nn.init.zeros_(param)


    def forward(self, queries, eval_mode=False):
        predictions_local, factors_local = self.local_model(queries, eval_mode)
        predictions_global, factors_global = self.global_model(queries, eval_mode)

        return predictions_local, predictions_global, factors_local, factors_global
    
    def mutual_learning(self, score_t, score_s):
        """
        score_t: (pos_score_t, neg_score_t)
        score_s: (pos_score_s, neg_score_s)
        """
        def minmax_norm(scores):
           scores_norm = torch.cat(scores, dim=-1)
           scores_max, _ = scores_norm.max(-1, keepdim=True)
           scores_min, _ = scores_norm.min(-1, keepdim=True)

           scores_norm = (scores_norm - scores_min) / (scores_max - scores_min)

           return scores_norm[:, 0:1], scores_norm[:, 1:]
        
        def zscore_norm(scores, eps=1e-6):
            scores_norm = torch.cat(scores, dim=-1)
            scores_mean = scores_norm.mean(-1, keepdim=True)
            scores_sqrtvar = torch.sqrt(scores_norm.var(-1, keepdim=True) + eps)

            scores_norm = (scores_norm - scores_mean) / scores_sqrtvar
            return scores_norm[:, 0:1], scores_norm[:, 1:]


        pos_score_t, neg_score_t = score_t
        pos_score_s, neg_score_s = score_s

        pos_score_t_norm, neg_score_t_norm = minmax_norm(score_t)
        pos_score_s_norm, neg_score_s_norm = minmax_norm(score_s)

        pos_indicats = (pos_score_t_norm > pos_score_s_norm).float().detach()
        neg_indicats = (neg_score_t_norm < neg_score_s_norm).float().detach()

        pos_mixsocre_t = pos_indicats * pos_score_t + (1. - pos_indicats) * pos_score_s
        neg_mixscore_t = neg_indicats * neg_score_t + (1. - neg_indicats) * neg_score_s

        
        student_prob = torch.cat([neg_score_s, pos_score_s.clone()], dim=1)
        teacher_prob = torch.cat([neg_mixscore_t, pos_mixsocre_t], dim=1)


        student_prob = F.log_softmax(student_prob, dim=1)
        teacher_prob = F.softmax(teacher_prob, dim=1)
        distill_loss = self.kl_loss(student_prob, teacher_prob.detach())

        neg_dist_loss = self.neg_dist_learning(neg_score_s=neg_score_s, neg_score_t=neg_score_t)

        return 0.5 * distill_loss + 0.5 * neg_dist_loss
    
    def neg_dist_learning(self, neg_score_s, neg_score_t):
        neg_score_s_ = F.softmax(self.scaling * neg_score_s, dim=1).clone().detach() * neg_score_s
        neg_score_t_ = F.softmax(self.scaling * neg_score_t, dim=1).clone().detach() * neg_score_t
        return self.kl_loss(F.log_softmax(neg_score_s_, dim=1), F.softmax(neg_score_t_.detach(), dim=1))
    
    def neg_dist_learning_v2(self, neg_score_s, neg_score_t):
        neg_indicats = (neg_score_t > neg_score_s).float().detach()
        neg_mixscore_t = neg_indicats * neg_score_t + (1. - neg_indicats) * neg_score_s

        neg_score_s_ = F.log_softmax(neg_score_s, dim=1)
        neg_mixscore_t_ = F.softmax(neg_mixscore_t, dim=1)
        distill_loss = self.kl_loss(neg_score_s_, neg_mixscore_t_.detach())

        return distill_loss


    def mutual_learning_adv(self, score_t, score_s):
        """
        score_t: (pos_score_t, neg_score_t)
        score_s: (pos_score_s, neg_score_s)
        """
        pos_score_t, neg_score_t = score_t
        pos_score_s, neg_score_s = score_s

        student_prob = torch.cat((neg_score_s, pos_score_s.clone()), dim=1)
        teacher_prob = torch.cat([neg_score_t, pos_score_t], dim=1)

        student_prob = F.softmax(-student_prob, dim=-1).detach() * student_prob
        student_prob = F.log_softmax(student_prob, dim=1)

        teacher_prob = F.softmax(-teacher_prob, dim=-1).detach() * teacher_prob
        teacher_prob = F.softmax(teacher_prob, dim=1)

        distill_loss = self.kl_loss(student_prob, teacher_prob.detach())

        return distill_loss
    
    def mutual_learning_adv_v2(self, score_t, score_s):
        """
        score_t: (pos_score_t, neg_score_t)
        score_s: (pos_score_s, neg_score_s)
        """
        pos_score_t, neg_score_t = score_t
        pos_score_s, neg_score_s = score_s

        student_prob = torch.cat((neg_score_s, pos_score_s.clone()), dim=1)
        teacher_prob = torch.cat([neg_score_t, pos_score_t], dim=1)

        student_prob = F.softmax(-teacher_prob, dim=-1).detach() * student_prob
        student_prob = F.log_softmax(student_prob, dim=1)

        teacher_prob = F.softmax(-student_prob, dim=-1).detach() * teacher_prob
        teacher_prob = F.softmax(teacher_prob, dim=1)

        distill_loss = self.kl_loss(student_prob, teacher_prob.detach())

        return distill_loss
    
    def mutual_learning_v2(self, pos_samples, neg_samples, model_t: BaseH, model_s: BaseH):

        pos_h, pos_t = pos_samples[:, 0], pos_samples[:, -1]
        neg_h, neg_t = neg_samples[:, :, 0], neg_samples[:, :, -1]

        pos_h_embed_teacher, pos_t_embed_teacher = model_t.entity(pos_h), model_t.entity(pos_t)
        pos_h_embed_student, pos_t_embed_student = model_s.entity(pos_h), model_s.entity(pos_t)
        neg_h_embed_teacher, neg_t_embed_teacher = model_t.entity(neg_h), model_t.entity(neg_t)
        neg_h_embed_student, neg_t_embed_student = model_s.entity(neg_h), model_s.entity(neg_t)

        pos_teacher_cos = F.cosine_similarity(pos_h_embed_teacher, pos_t_embed_teacher, dim=-1).unsqueeze(-1)
        pos_student_cos = F.cosine_similarity(pos_h_embed_student, pos_t_embed_student, dim=-1).unsqueeze(-1)
        neg_teacher_cos = F.cosine_similarity(neg_h_embed_teacher, neg_t_embed_teacher, dim=-1)
        neg_student_cos = F.cosine_similarity(neg_h_embed_student, neg_t_embed_student, dim=-1)

        pos_d_angle = self.huber_loss(pos_student_cos, pos_teacher_cos.detach())
        neg_d_angle = self.huber_loss(neg_student_cos, neg_teacher_cos.detach())

        pos_teacher_lr = pos_h_embed_teacher.norm(dim=-1, keepdim=True) / pos_t_embed_teacher.norm(dim=-1, keepdim=True)
        pos_student_lr = pos_h_embed_student.norm(dim=-1, keepdim=True) / pos_t_embed_student.norm(dim=-1, keepdim=True)
        neg_teacher_lr = neg_h_embed_teacher.norm(dim=-1) / neg_t_embed_teacher.norm(dim=-1)
        neg_student_lr = neg_h_embed_student.norm(dim=-1) / neg_t_embed_student.norm(dim=-1)

        pos_d_lr = self.huber_loss(pos_student_lr, pos_teacher_lr.detach())
        neg_d_lr = self.huber_loss(neg_student_lr, neg_teacher_lr.detach())

        pos_d_stru = pos_d_angle + pos_d_lr
        neg_d_stru = neg_d_angle + neg_d_lr


        return pos_d_stru + neg_d_stru
    
    def feature_learning(self, pos_samples, neg_samples, model_t: BaseH, model_s: BaseH):
        pos_h, pos_t = pos_samples[:, 0], pos_samples[:, -1]
        neg_h, neg_t = neg_samples[:, :, 0], neg_samples[:, :, -1]

        pos_h_embed_teacher, pos_t_embed_teacher = model_t.entity(pos_h), model_t.entity(pos_t)
        pos_h_embed_student, pos_t_embed_student = model_s.entity(pos_h), model_s.entity(pos_t)
        neg_h_embed_teacher, neg_t_embed_teacher = model_t.entity(neg_h), model_t.entity(neg_t)
        neg_h_embed_student, neg_t_embed_student = model_s.entity(neg_h), model_s.entity(neg_t)

        pos_h_embed_teacher = model_t.kd_mapping(pos_h_embed_teacher)
        pos_t_embed_teacher = model_t.kd_mapping(pos_t_embed_teacher)
        neg_h_embed_teacher = model_t.kd_mapping(neg_h_embed_teacher)
        neg_t_embed_teacher = model_t.kd_mapping(neg_t_embed_teacher)

        pos_h_embed_student = model_s.kd_mapping(pos_h_embed_student)
        pos_t_embed_student = model_s.kd_mapping(pos_t_embed_student)
        neg_h_embed_student = model_s.kd_mapping(neg_h_embed_student)
        neg_t_embed_student = model_s.kd_mapping(neg_t_embed_student)

        pos_teacher_cos = F.cosine_similarity(pos_h_embed_teacher, pos_t_embed_teacher, dim=-1).unsqueeze(-1)
        pos_student_cos = F.cosine_similarity(pos_h_embed_student, pos_t_embed_student, dim=-1).unsqueeze(-1)
        neg_teacher_cos = F.cosine_similarity(neg_h_embed_teacher, neg_t_embed_teacher, dim=-1)
        neg_student_cos = F.cosine_similarity(neg_h_embed_student, neg_t_embed_student, dim=-1)

        pos_d_angle = self.huber_loss(pos_student_cos, pos_teacher_cos.detach())
        neg_d_angle = self.huber_loss(neg_student_cos, neg_teacher_cos.detach())

        pos_teacher_lr = pos_h_embed_teacher.norm(dim=-1, keepdim=True) / pos_t_embed_teacher.norm(dim=-1, keepdim=True)
        pos_student_lr = pos_h_embed_student.norm(dim=-1, keepdim=True) / pos_t_embed_student.norm(dim=-1, keepdim=True)
        neg_teacher_lr = neg_h_embed_teacher.norm(dim=-1) / neg_t_embed_teacher.norm(dim=-1)
        neg_student_lr = neg_h_embed_student.norm(dim=-1) / neg_t_embed_student.norm(dim=-1)

        pos_d_lr = self.huber_loss(pos_student_lr, pos_teacher_lr.detach())
        neg_d_lr = self.huber_loss(neg_student_lr, neg_teacher_lr.detach())

        pos_d_stru = pos_d_angle + pos_d_lr
        neg_d_stru = neg_d_angle + neg_d_lr


        return pos_d_stru + neg_d_stru
    
    @torch.no_grad()
    def dist(self):
        X = self.local_model.entity.weight
        Y = self.global_model.entity.weight

        M = torch.linalg.inv(Y.T @ Y).T @ (Y.T @ X)
        dist = (1.0 / self.sizes[0]) * torch.trace((X - Y @ M).T @ (X - Y @ M))
        return dist
    
    def get_ranking(self, queries, filters, batch_size = 1000):

        ranks_both = torch.ones(len(queries))
        ranks_global = torch.ones(len(queries))
        ranks_local = torch.ones(len(queries))

        with torch.no_grad():
            b_begin = 0
            local_cands = self.local_model.get_rhs(queries, eval_mode=True)
            global_cands = self.global_model.get_rhs(queries, eval_mode=True)

            with tqdm.tqdm(total=queries.shape[0], unit='ex') as bar:
                bar.set_description(f'evaluation')
                while b_begin < len(queries):
                    these_queries = queries[b_begin:b_begin + batch_size].to(self.device)
                    target_idxs = these_queries[:, 2:3]

                    local_q = self.local_model.get_queries(these_queries)
                    scores_local = self.local_model.score(local_q, local_cands, eval_mode=True)

                    global_q = self.global_model.get_queries(these_queries)
                    scores_global = self.global_model.score(global_q, global_cands, eval_mode=True)

                    scores_both = scores_local + scores_global

                    targets_both = torch.gather(scores_both, index=target_idxs, dim=-1)
                    targets_global = torch.gather(scores_global, index=target_idxs, dim=-1)
                    targets_local = torch.gather(scores_local, index=target_idxs, dim=-1)

                    for i, query in enumerate(these_queries):
                        filter_out = filters[(query[0].item(), query[1].item())]
                        filter_out += [queries[b_begin + i, 2].item()]  
                        # set training sample scores with low value 
                        scores_both[i, torch.LongTensor(filter_out)] = -1e6
                        scores_local[i, torch.LongTensor(filter_out)] = -1e6
                        scores_global[i, torch.LongTensor(filter_out)] = -1e6

                    ranks_both[b_begin:b_begin + batch_size] += torch.sum(
                        (scores_both >= targets_both).float(), dim=1
                    ).cpu()
                    ranks_global[b_begin:b_begin + batch_size] += torch.sum(
                        (scores_global >= targets_global).float(), dim=1
                    ).cpu()
                    ranks_local[b_begin:b_begin + batch_size] += torch.sum(
                        (scores_local >= targets_local).float(), dim=1
                    ).cpu()
                    b_begin += batch_size

                    bar.update(these_queries.shape[0])

        return ranks_both, ranks_global, ranks_local
    
    def compute_metrics(self, examples, filters, batch_size=500):
        """Compute ranking-based evaluation metrics.
    
        Args:
            examples: torch.LongTensor of size n_examples x 3 containing triples' indices
            filters: Dict with entities to skip per query for evaluation in the filtered setting
            batch_size: integer for batch size to use to compute scores

        Returns:
            Evaluation metrics (mean rank, mean reciprocical rank and hits)
        """
        mean_rank_both, mean_rank_global, mean_rank_local = {}, {}, {}
        mean_reciprocal_rank_both, mean_reciprocal_rank_global, mean_reciprocal_rank_local = {}, {}, {}
        hits_at_both, hits_at_global, hits_at_local = {}, {}, {}

        for m in ["rhs", "lhs"]:
            q = examples.clone()
            if m == "lhs":
                tmp = torch.clone(q[:, 0])
                q[:, 0] = q[:, 2]
                q[:, 2] = tmp
                q[:, 1] += self.sizes[1] // 2
            
            ranks_both, ranks_global, ranks_local = self.get_ranking(q, filters[m], batch_size=batch_size)

            # parsing ranks results
            # both
            mean_rank_both[m] = torch.mean(ranks_both).item()
            mean_reciprocal_rank_both[m] = torch.mean(1. / ranks_both).item()
            hits_at_both[m] = torch.FloatTensor((list(map(
                lambda x: torch.mean((ranks_both <= x).float()).item(),
                (1, 3, 10)
            ))))

            # global
            mean_rank_global[m] = torch.mean(ranks_global).item()
            mean_reciprocal_rank_global[m] = torch.mean(1. / ranks_global).item()
            hits_at_global[m] = torch.FloatTensor((list(map(
                lambda x: torch.mean((ranks_global <= x).float()).item(),
                (1, 3, 10)
            ))))

            # local
            mean_rank_local[m] = torch.mean(ranks_local).item()
            mean_reciprocal_rank_local[m] = torch.mean(1. / ranks_local).item()
            hits_at_local[m] = torch.FloatTensor((list(map(
                lambda x: torch.mean((ranks_local <= x).float()).item(),
                (1, 3, 10)
            ))))

        return (
            (mean_rank_both, mean_reciprocal_rank_both, hits_at_both),
            (mean_rank_global, mean_reciprocal_rank_global, hits_at_global),
            (mean_rank_local, mean_reciprocal_rank_local, hits_at_local)
        )
