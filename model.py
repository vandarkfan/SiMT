import torch
import torch.nn as nn
import torch.nn.functional as F
from PLA import SiAttention

from einops import rearrange, einsum

class SiTM(nn.Module):
    def __init__(self, num_ent, num_rel, ent_vis, rel_vis, dim_vis, ent_txt, rel_txt, dim_txt, ent_vis_mask, rel_vis_mask, \
                 dim_str, num_head, dim_hid, num_layer_enc_ent, num_layer_enc_rel, num_layer_dec, dropout = 0.1, \
                 emb_dropout = 0.6, vis_dropout = 0.1, txt_dropout = 0.1):
        super(SiTM, self).__init__()
        self.dim_str = dim_str
        self.num_head = num_head
        self.dim_hid = dim_hid
        self.num_ent = num_ent
        self.num_rel = num_rel

        self.ent_vis = ent_vis
        self.rel_vis = rel_vis
        self.ent_txt = ent_txt.unsqueeze(dim = 1)
        self.rel_txt = rel_txt.unsqueeze(dim = 1)

        self.ent_token = nn.Parameter(torch.Tensor(1, 1, dim_str))
        self.rel_token = nn.Parameter(torch.Tensor(1, 1, dim_str))
        self.ent_embeddings = nn.Parameter(torch.Tensor(num_ent, 1, dim_str))
        self.rel_embeddings = nn.Parameter(torch.Tensor(num_rel, 1 ,dim_str))
        self.lp_token = nn.Parameter(torch.Tensor(1,dim_str))

        self.str_ent_ln = nn.LayerNorm(dim_str)
        self.str_rel_ln = nn.LayerNorm(dim_str)
        self.vis_ln = nn.LayerNorm(dim_str)
        self.txt_ln = nn.LayerNorm(dim_str)

        self.embdr = nn.Dropout(p = emb_dropout)
        self.visdr = nn.Dropout(p = vis_dropout)
        self.txtdr = nn.Dropout(p = txt_dropout)


        self.pos_str_ent = nn.Parameter(torch.Tensor(1,1,dim_str))
        self.pos_vis_ent = nn.Parameter(torch.Tensor(1,1,dim_str))
        self.pos_txt_ent = nn.Parameter(torch.Tensor(1,1,dim_str))

        self.pos_str_rel = nn.Parameter(torch.Tensor(1,1,dim_str))
        self.pos_vis_rel = nn.Parameter(torch.Tensor(1,1,dim_str))
        self.pos_txt_rel = nn.Parameter(torch.Tensor(1,1,dim_str))

        self.pos_head = nn.Parameter(torch.Tensor(1,1,dim_str))
        self.pos_rel = nn.Parameter(torch.Tensor(1,1,dim_str))
        self.pos_tail = nn.Parameter(torch.Tensor(1,1,dim_str))
        
        self.proj_ent_vis = nn.Linear(dim_vis, dim_str)
        self.proj_txt = nn.Linear(dim_txt, dim_str)

        self.proj_rel_vis = nn.Linear(dim_vis * 3, dim_str)

        self.ent_encoder = nn.Sequential(
            SiAttentionSimOnly(dim=dim_str, num_patches=3, num_heads=num_head, qkv_bias=False, qk_scale=None, attn_drop=0.,
                                proj_drop=0., sr_ratio=1,
                                kernel_size=5, alpha=4),
            SiAttentionSimOnly(dim=dim_str, num_patches=3, num_heads=num_head, qkv_bias=False, qk_scale=None, attn_drop=0.,
                                proj_drop=0., sr_ratio=1,
                                kernel_size=5, alpha=4),)
        self.rel_encoder = SiAttentionSimOnly(dim=dim_str, num_patches=3, num_heads=num_head, qkv_bias=False, qk_scale=None, attn_drop=0.,
                                proj_drop=0., sr_ratio=1,
                                kernel_size=5, alpha=4)

        self.init_weights()
    def init_weights(self):
        nn.init.xavier_uniform_(self.ent_embeddings)
        nn.init.xavier_uniform_(self.rel_embeddings)
        nn.init.xavier_uniform_(self.proj_ent_vis.weight)
        nn.init.xavier_uniform_(self.proj_rel_vis.weight)
        nn.init.xavier_uniform_(self.proj_txt.weight)
        

        nn.init.xavier_uniform_(self.ent_token)
        nn.init.xavier_uniform_(self.rel_token)
        nn.init.xavier_uniform_(self.lp_token)
        nn.init.xavier_uniform_(self.pos_str_ent)
        nn.init.xavier_uniform_(self.pos_vis_ent)
        nn.init.xavier_uniform_(self.pos_txt_ent)
        nn.init.xavier_uniform_(self.pos_str_rel)
        nn.init.xavier_uniform_(self.pos_vis_rel)
        nn.init.xavier_uniform_(self.pos_txt_rel)
        nn.init.xavier_uniform_(self.pos_head)
        nn.init.xavier_uniform_(self.pos_rel)
        nn.init.xavier_uniform_(self.pos_tail)

        self.proj_ent_vis.bias.data.zero_()
        self.proj_rel_vis.bias.data.zero_()
        self.proj_txt.bias.data.zero_()

    def forward(self):
        rep_ent_str = self.embdr(self.str_ent_ln(self.ent_embeddings)) + self.pos_str_ent
        rep_ent_vis = self.visdr(self.vis_ln(self.proj_ent_vis(self.ent_vis))) + self.pos_vis_ent
        rep_ent_txt = self.txtdr(self.txt_ln(self.proj_txt(self.ent_txt))) + self.pos_txt_ent
        modalities = torch.cat([rep_ent_str, rep_ent_vis, rep_ent_txt], dim = 1)  # [batchsize, 4, 64]
        ent_embs = self.ent_encoder(modalities)[:,0,:]

        rep_rel_str = self.embdr(self.str_rel_ln(self.rel_embeddings)) + self.pos_str_rel
        rep_rel_vis = self.visdr(self.vis_ln(self.proj_rel_vis(self.rel_vis))) + self.pos_vis_rel
        rep_rel_txt = self.txtdr(self.txt_ln(self.proj_txt(self.rel_txt))) + self.pos_txt_rel
        modalities = torch.cat([rep_rel_str, rep_rel_vis, rep_rel_txt], dim = 1)  # [batchsize, 4, 64]
        rel_embs = self.rel_encoder(modalities)[:,0,:]
        return ent_embs, rel_embs
    def complex(self, lhs, rel, rhs, ent_embs, rel_embs):
        rank = self.dim_str // 2
        lhs = lhs[:, :rank], lhs[:, rank:]
        rel = rel[:, :rank], rel[:, rank:]
        rhs = rhs[:, :rank], rhs[:, rank:]
        output_dec_rhs = torch.cat([
            lhs[0] * rel[0] - lhs[1] * rel[1],
            lhs[0] * rel[1] + lhs[1] * rel[0]
        ], 1)
        # [1000,2000]
        output_dec_lhs = torch.cat([
            rhs[0] * rel[0] - rhs[1] * rel[1],
            rhs[0] * rel[1] + rhs[1] * rel[0]
        ], 1)

        rhs_scores = torch.inner(output_dec_rhs, ent_embs)
        lhs_scores = torch.inner(output_dec_lhs, ent_embs)
        return rhs_scores, lhs_scores
    def score(self, emb_ent, emb_rel, triplets):
        h_seq = emb_ent[triplets[:,0]]
        r_seq = emb_rel[triplets[:,1]]
        t_seq = emb_ent[triplets[:,2]]
        rhs_scores, lhs_scores = self.complex(h_seq, r_seq, t_seq, emb_ent, emb_rel)
        return rhs_scores, lhs_scores