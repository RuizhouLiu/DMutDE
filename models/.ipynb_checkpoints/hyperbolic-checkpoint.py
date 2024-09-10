"""Hyperbolic Knowledge Graph embedding models where all parameters are defined in tangent spaces."""
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from models.base import KGModel
from utils.euclidean import givens_rotations, givens_reflection
from utils.hyperbolic import mobius_add, expmap0, project, hyp_distance_multi_c, expmapP, expmapH, projectH
from utils import pmath


HYP_MODELS = ['DVKGE']


class BaseH(KGModel):
    """Trainable curvature for each relationship."""

    def __init__(self, args):
        super(BaseH, self).__init__(args.sizes, args.rank, args.dropout, args.gamma, args.dtype, args.bias,
                                    args.init_size)
        

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


class DVKGE(BaseH):

    def __init__(self, args):
        super(DVKGE, self).__init__(args)

        self.n_space = args.n_space
        self.entity = nn.Embedding(self.sizes[0], self.rank * self.n_space, dtype=self.data_type)
        self.rel_trans_2g = nn.Embedding(self.sizes[1], 2 * self.rank * self.n_space, dtype=self.data_type)
        self.rel_rot = nn.Embedding(self.sizes[1], self.rank * self.n_space, dtype=self.data_type)
        self.rel_trans = nn.Embedding(self.sizes[1], self.rank * self.n_space, dtype=self.data_type)
        self.rel_scale = nn.Embedding(self.sizes[1], (self.rank // 4) * self.n_space, dtype=self.data_type)

        if args.dtype == "double":
            self.scale = torch.Tensor([1. / np.sqrt(self.rank * self.n_space)]).double().cuda()
        else:
            self.scale = torch.Tensor([1. / np.sqrt(self.rank * self.n_space)]).cuda()

        # set curvatures params
        self.c_local = nn.Embedding(self.sizes[1], self.n_space, dtype=self.data_type)
        self.c_global = nn.Embedding(self.sizes[1], 1, dtype=self.data_type)

        # bias
        self.bh = nn.Embedding(self.sizes[0], 1, dtype=self.data_type)
        self.bt = nn.Embedding(self.sizes[0], 1, dtype=self.data_type)

        self.eval_hyper_dist_n_space_idx = None

        self.params_init()
    
    def params_init(self):
        nn.init.xavier_normal_(self.entity.weight.data)
        nn.init.xavier_normal_(self.rel_trans_2g.weight.data)
        nn.init.xavier_normal_(self.rel_rot.weight.data)
        nn.init.xavier_normal_(self.rel_trans.weight.data)
        nn.init.xavier_normal_(self.rel_scale.weight.data)
        nn.init.ones_(self.c_global.weight.data)
        nn.init.zeros_(self.bh.weight.data)
        nn.init.zeros_(self.bt.weight.data)
    
    def set_eval_hyper_dist_n_space_idx(self, n_space_idx):
        if not hasattr(self, 'eval_hyper_dist_n_space_idx'):
            self.eval_hyper_dist_n_space_idx = None
        
        print(f"eval_hyper_dist_n_space_idx is set from {self.eval_hyper_dist_n_space_idx} to {n_space_idx}")

        self.eval_hyper_dist_n_space_idx = n_space_idx
        
    
    def get_queries(self, queries):
        c_global = F.softplus(self.c_global(queries[:, 1])).view(-1, 1, 1).expand((-1, self.n_space, -1))
        c_local = F.softplus(self.c_local(queries[:, 1])).view(-1, self.n_space, 1)
        head = self.entity(queries[:, 0]).view(-1, self.n_space, self.rank)
        rel_trans_g_1, rel_trans_g_2 = torch.chunk(self.rel_trans_2g(queries[:, 1]).view(-1, self.n_space, 2 * self.rank), 2, dim=-1)
        rel_rot = self.rel_rot(queries[:, 1]).view(-1, self.n_space, self.rank)
        rel_scale = self.rel_scale(queries[:, 1]).view(-1, self.n_space, self.rank // 4)

        # global
        head = expmap0(head, c_global)
        rel_trans_g_1 = expmap0(rel_trans_g_1, c_global)
        head = project(mobius_add(head, rel_trans_g_1, c_global), c_global)

        # local
        head = expmap0(head, c_local)
        rel_trans_g_2 = expmap0(rel_trans_g_2, c_local)
        head = self.quaternions(rel_rot, head, scale=rel_scale, c=c_local)
        
        head = project(mobius_add(head, rel_trans_g_2, c_local), c_local)

        return (head, c_local), self.bh(queries[:, 0])

    def _hyp_givens_rotations(self, r, x, c):
        givens = r.view((r.shape[0], self.n_space, -1, 2))
        givens = givens / torch.norm(givens, p=2, dim=-1, keepdim=True).clamp_min(1e-15)
        x = x.view((r.shape[0], self.n_space, -1, 2))
        c = c.unsqueeze(-1)

        x_real_part = pmath.mobius_pointwise_mul(givens[:, :, :, 0:1], x, c=c, dim=-2)
        x_imag_part = pmath.mobius_pointwise_mul(givens[:, :, :, 1:], torch.cat((-x[:, :, :, 1:], x[:, :, :, 0:1]), dim=-1), c=c, dim=-2)

        return (x_real_part + x_imag_part).view((-1, self.n_space, self.rank))
        # x_rot = givens[:, :, 0:1] * x + givens[:, :, 1:] * torch.cat((-x[:, :, 1:], x[:, :, 0:1]), dim=-1)
        # return x_rot.view((r.shape[0], -1))
    
    def _hyp_givens_reflections(self, r, x, c):
        givens = r.view((r.shape[0], self.n_space, -1, 2))
        givens = givens / torch.norm(givens, p=2, dim=-1, keepdim=True).clamp_min(1e-15)
        x = x.view((r.shape[0], self.n_space, -1, 2))
        c = c.unsqueeze(-1)

        real_part = pmath.mobius_pointwise_mul(givens[:, :, :, 0:1], torch.cat((x[:, :, :, 0:1], -x[:, :, :, 1:]), dim=-1), c=c, dim=-2)
        imag_part = pmath.mobius_pointwise_mul(givens[:, :, :, 1:], torch.cat((x[:, :, :, 1:], x[:, :, :, 0:1]), dim=-1), c=c, dim=-2)

        return (real_part + imag_part).view((-1, self.n_space, self.rank))
    
    def quaternions(self, r, h, *, scale=None, c=None):
        s_r, x_r, y_r, z_r = torch.chunk(r, 4, dim=-1)
        s_h, x_h, y_h, z_h = torch.chunk(h, 4, dim=-1)

        denominator_r = torch.sqrt(s_r ** 2 + x_r ** 2 + y_r ** 2 + z_r ** 2)
        s_r = s_r / denominator_r
        x_r = x_r / denominator_r
        y_r = y_r / denominator_r
        z_r = z_r / denominator_r


        A = s_h * s_r - x_h * x_r - y_h * y_r - z_h * z_r
        B = s_h * x_r + s_r * x_h + y_h * z_r - y_r * z_h
        C = s_h * y_r + s_r * y_h + z_h * x_r - z_r * x_h
        D = s_h * z_r + s_r * z_h + x_h * y_r - x_r * y_h

        if c != None:
            ABCD = torch.stack([A, B, C, D], dim=-1)
            ABCD = pmath.mobius_scalar_mul(scale.unsqueeze(-1), ABCD, c=c.unsqueeze(-1))
            A, B, C, D = ABCD[..., 0], ABCD[..., 1], ABCD[..., 2], ABCD[..., 3]

        return torch.cat([A, B, C, D], dim=-1)
    

    def givens_rotations(self, r, x):
        givens = r.view((r.shape[0], self.n_space, -1, 2))
        givens = givens / torch.norm(givens, p=2, dim=-1, keepdim=True).clamp_min(1e-15)
        x = x.view((r.shape[0], self.n_space, -1, 2))

        x_real_part = givens[:, :, :, 0:1] * x
        x_imag_part = givens[:, :, :, 1:] * torch.cat((-x[:, :, :, 1:], x[:, :, :, 0:1]), dim=-1)

        return (x_real_part + x_imag_part).view((-1, self.n_space, self.rank))
    

    def givens_reflections(self, r, x):
        givens = r.view((r.shape[0], self.n_space, -1, 2))
        givens = givens / torch.norm(givens, p=2, dim=-1, keepdim=True).clamp_min(1e-15)
        x = x.view((r.shape[0], self.n_space, -1, 2))

        real_part = givens[:, :, :, 0:1] * torch.cat((x[:, :, :, 0:1], -x[:, :, :, 1:]), dim=-1)
        imag_part = givens[:, :, :, 1:] * torch.cat((x[:, :, :, 1:], x[:, :, :, 0:1]), dim=-1)

        return (real_part + imag_part).view((-1, self.n_space, self.rank))



    def get_rhs(self, queries, eval_mode):
        """Get embeddings and biases of target entities."""
        if eval_mode:
            return self.entity.weight.view(-1, self.n_space, self.rank), self.bt.weight
        else:
            return self.entity(queries[:, 2]).view(-1, self.n_space, self.rank), self.bt(queries[:, 2])

    def similarity_score(self, lhs_e, rhs_e, eval_mode):
        """Compute similarity scores or queries against targets in embedding space."""
        lhs_e, c = lhs_e
        dist = hyp_distance_multi_c(lhs_e, rhs_e, c, eval_mode, n_space_idx=self.eval_hyper_dist_n_space_idx)
        if eval_mode:
            return -(dist ** 2).mean(dim=0)
        else:
            return -(dist ** 2).mean(dim=1)

    def get_factors(self, queries):
        head_e = self.entity(queries[:, 0])
        rel_e = torch.cat([
            self.rel_trans_2g(queries[:, 1]), self.rel_rot(queries[:, 1]), self.rel_trans(queries[:, 1]), self.rel_scale(queries[:, 1])
            ], dim=-1)
        rhs_e = self.entity(queries[:, 2])
        return head_e, rel_e, rhs_e


