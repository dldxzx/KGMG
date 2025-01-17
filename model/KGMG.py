import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_sum, scatter_mean
from tqdm.auto import tqdm

from model.common import compose_context, ShiftedSoftplus
from model.egnn import EGNN
from model.moe import MoE
from model.uni_transformer import UniTransformerO2TwoUpdateGeneral
from model.function import *
from model.attention import BasicTransformerBlock

# Time embedding
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = np.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

# 自适应动态融合
class DynamicFusionModule(nn.Module):
    def __init__(self, h_dim, pos_dim):
        super(DynamicFusionModule, self).__init__()
        # 用于融合坐标的全连接层
        self.pos_mlp = nn.Linear(pos_dim, pos_dim)
        # 用于融合特征的全连接层
        self.h_mlp = nn.Linear(h_dim, h_dim)

        # 可学习权重
        self.w_gloab = nn.Parameter(torch.Tensor(1))  # 全局坐标权重
        self.w_h_gloab = nn.Parameter(torch.Tensor(1))  # 全局特征权重
        
        # 初始化权重
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.ones_(self.w_gloab)  # 初始化为1
        nn.init.ones_(self.w_h_gloab)  # 初始化为1

    def forward(self, final_pos_gloab, final_pos_local, final_h_gloab, final_h_local):
        # 动态计算权重
        w_gloab = torch.sigmoid(self.w_gloab)  # 全局坐标权重
        w_local = 1 - w_gloab  # 局部坐标权重
        
        # 坐标融合
        final_pos = w_gloab * final_pos_gloab + w_local * final_pos_local
        
        # 动态计算特征权重
        w_h_gloab = torch.sigmoid(self.w_h_gloab)  # 全局特征权重
        w_h_local = 1 - w_h_gloab  # 局部特征权重
        # 特征融合
        final_h = w_h_gloab * final_h_gloab + w_h_local * final_h_local

        # 通过MLP进一步处理融合结果
        final_pos = self.pos_mlp(final_pos)
        final_h = self.h_mlp(final_h)
        return final_pos, final_h
# 门控融合
class AdvancedFusionModule(nn.Module):
    def __init__(self, h_dim, pos_dim):
        super(AdvancedFusionModule, self).__init__()
        # 用于融合坐标和特征的门控网络
        self.pos_gate = nn.Sequential(
            nn.Linear(pos_dim * 2, pos_dim),
            nn.ReLU(),
            nn.Linear(pos_dim, 1),
            nn.Sigmoid()
        )
        self.h_gate = nn.Sequential(
            nn.Linear(h_dim * 2, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, 1),
            nn.Sigmoid()
        )

        # 用于进一步处理融合后的坐标和特征的全连接层
        self.pos_mlp = nn.Linear(pos_dim, pos_dim)
        self.h_mlp = nn.Linear(h_dim, h_dim)

    def forward(self, final_pos_gloab, final_pos_local, final_h_gloab, final_h_local):
        # 计算坐标的融合权重
        pos_concat = torch.cat([final_pos_gloab, final_pos_local], dim=-1)
        w_gloab_pos = self.pos_gate(pos_concat)  # 全局坐标的门控权重
        w_local_pos = 1 - w_gloab_pos            # 局部坐标的门控权重
        
        # 坐标融合
        final_pos = w_gloab_pos * final_pos_gloab + w_local_pos * final_pos_local

        # 计算特征的融合权重
        h_concat = torch.cat([final_h_gloab, final_h_local], dim=-1)
        w_gloab_h = self.h_gate(h_concat)        # 全局特征的门控权重
        w_local_h = 1 - w_gloab_h                # 局部特征的门控权重
        
        # 特征融合
        final_h = w_gloab_h * final_h_gloab + w_local_h * final_h_local

        # 通过 MLP 进一步处理融合结果
        final_pos = self.pos_mlp(final_pos)
        final_h = self.h_mlp(final_h)
        
        return final_pos, final_h

class AttentionFusionModule(nn.Module):
    def __init__(self, h_dim, pos_dim, num_heads=4):
        super(AttentionFusionModule, self).__init__()

        self.new_pos_dim = (pos_dim // 2) * 16
        self.new_h_dim = (h_dim // 2) * 4

        self.pos_proj = nn.Linear(pos_dim, self.new_pos_dim)
        self.h_proj = nn.Linear(h_dim, self.new_h_dim)

        self.pos_attention = nn.MultiheadAttention(embed_dim=self.new_pos_dim, num_heads=num_heads, batch_first=True)
        self.h_attention = nn.MultiheadAttention(embed_dim=self.new_h_dim, num_heads=num_heads, batch_first=True)

        self.pos_mlp = nn.Linear(self.new_pos_dim, pos_dim)
        self.h_mlp = nn.Linear(self.new_h_dim, h_dim)

    def forward(self, final_pos_gloab, final_pos_local, final_h_gloab, final_h_local):
        # 将全局和局部坐标、特征分别通过投影层
        final_pos_gloab = self.pos_proj(final_pos_gloab)
        final_pos_local = self.pos_proj(final_pos_local)

        final_h_gloab = self.h_proj(final_h_gloab)
        final_h_local = self.h_proj(final_h_local)

        # 将投影后的全局和局部坐标拼接在一起
        pos_concat = torch.stack([final_pos_gloab, final_pos_local], dim=1)  # (batch_size, 2, new_pos_dim)
        h_concat = torch.stack([final_h_gloab, final_h_local], dim=1)        # (batch_size, 2, new_h_dim)
       
        # 使用多头注意力机制进行坐标融合
        attn_output_pos, _ = self.pos_attention(pos_concat, pos_concat, pos_concat)
        final_pos = attn_output_pos.mean(dim=1)  # 将多头的输出平均作为最终融合的坐标
        # 使用多头注意力机制进行特征融合
        attn_output_h, _ = self.h_attention(h_concat, h_concat, h_concat)
        final_h = attn_output_h.mean(dim=1)      # 将多头的输出平均作为最终融合的特征

        # 通过 MLP 将结果投影回原始维度
        final_pos = self.pos_mlp(final_pos)
        final_h = self.h_mlp(final_h)
        return final_pos, final_h




class SurfDM(nn.Module):
    def __init__(self, config, protein_atom_feature_dim, ligand_atom_feature_dim):
        super().__init__()
        self.config = config

        self.model_mean_type = config.model_mean_type  # ['noise', 'C0']
        self.loss_v_weight = config.loss_v_weight
        self.sample_time_method = config.sample_time_method  # ['importance', 'symmetric']

        if config.beta_schedule == 'cosine':
            alphas = cosine_beta_schedule(config.num_diffusion_timesteps, config.pos_beta_s) ** 2
            # print('cosine pos alpha schedule applied!')
            betas = 1. - alphas
        else:
            betas = get_beta_schedule(
                beta_schedule=config.beta_schedule,
                beta_start=config.beta_start,
                beta_end=config.beta_end,
                num_diffusion_timesteps=config.num_diffusion_timesteps,
            )
            alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])

        self.betas = to_torch_const(betas)
        self.num_timesteps = self.betas.size(0)
        self.alphas_cumprod = to_torch_const(alphas_cumprod)
        self.alphas_cumprod_prev = to_torch_const(alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = to_torch_const(np.sqrt(alphas_cumprod))
        self.sqrt_one_minus_alphas_cumprod = to_torch_const(np.sqrt(1. - alphas_cumprod))
        self.sqrt_recip_alphas_cumprod = to_torch_const(np.sqrt(1. / alphas_cumprod))
        self.sqrt_recipm1_alphas_cumprod = to_torch_const(np.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        self.posterior_mean_c0_coef = to_torch_const(betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        self.posterior_mean_ct_coef = to_torch_const(
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod))
        # log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.posterior_var = to_torch_const(posterior_variance)
        self.posterior_logvar = to_torch_const(np.log(np.append(self.posterior_var[1], self.posterior_var[1:])))

        # atom type diffusion schedule in log space
        if config.v_beta_schedule == 'cosine':
            alphas_v = cosine_beta_schedule(self.num_timesteps, config.v_beta_s)
            # print('cosine v alpha schedule applied!')
        else:
            raise NotImplementedError
        log_alphas_v = np.log(alphas_v)
        log_alphas_cumprod_v = np.cumsum(log_alphas_v)
        self.log_alphas_v = to_torch_const(log_alphas_v)
        self.log_one_minus_alphas_v = to_torch_const(log_1_min_a(log_alphas_v))
        self.log_alphas_cumprod_v = to_torch_const(log_alphas_cumprod_v)
        self.log_one_minus_alphas_cumprod_v = to_torch_const(log_1_min_a(log_alphas_cumprod_v))

        self.register_buffer('Lt_history', torch.zeros(self.num_timesteps))
        self.register_buffer('Lt_count', torch.zeros(self.num_timesteps))

        # model definition
        self.hidden_dim = config.hidden_dim
        self.num_classes = ligand_atom_feature_dim
        if self.config.node_indicator:
            emb_dim = self.hidden_dim - 1
        else:
            emb_dim = self.hidden_dim

        # atom embedding
        self.protein_atom_emb = nn.Linear(protein_atom_feature_dim, emb_dim)
        # center pos
        self.center_pos_mode = config.center_pos_mode  # ['none', 'protein']

        # time embedding
        self.time_emb_dim = config.time_emb_dim
        self.time_emb_mode = config.time_emb_mode  # ['simple', 'sin']
        if self.time_emb_dim > 0:
            if self.time_emb_mode == 'simple':
                self.ligand_atom_emb = nn.Linear(ligand_atom_feature_dim + 1, emb_dim)
            elif self.time_emb_mode == 'sin':
                self.time_emb = nn.Sequential(
                    SinusoidalPosEmb(self.time_emb_dim),
                    nn.Linear(self.time_emb_dim, self.time_emb_dim * 4),
                    nn.GELU(),
                    nn.Linear(self.time_emb_dim * 4, self.time_emb_dim)
                )
                self.ligand_atom_emb = nn.Linear(ligand_atom_feature_dim + self.time_emb_dim, emb_dim)
            else:
                raise NotImplementedError
        else:
            self.ligand_atom_emb = nn.Linear(ligand_atom_feature_dim, emb_dim)

        self.refine_net_type = config.model_type
        # self.atten_layer = BasicTransformerBlock(self.hidden_dim-1, 4, self.hidden_dim-1 // 4, 0.1, self.hidden_dim-1)
        self.encoder_local = EGNN(
            num_layers=config.num_layers,
            hidden_dim=config.hidden_dim,
            edge_feat_dim=config.edge_feat_dim,
            num_r_gaussian=1,
            cutoff=config.l_cutoff,
            k=config.l_knn,
            cutoff_mode=config.cutoff_mode
        )
        # self.encoder_gloab = EGNN(
        #     num_layers=config.num_layers,
        #     hidden_dim=config.hidden_dim,
        #     edge_feat_dim=config.edge_feat_dim,
        #     num_r_gaussian=1,
        #     cutoff=config.g_cutoff,
        #     k=config.g_knn,
        #     cutoff_mode=config.cutoff_mode
        # )
        # self.encoder_local = UniTransformerO2TwoUpdateGeneral(
        #     num_blocks=config.num_blocks,
        #     num_layers=config.num_layers,
        #     hidden_dim=config.hidden_dim,
        #     n_heads=config.n_heads,
        #     cutoff=config.l_cutoff,
        #     k=config.l_knn,
        #     edge_feat_dim=config.edge_feat_dim,
        #     num_r_gaussian=config.num_r_gaussian,
        #     num_node_types=config.num_node_types,
        #     act_fn=config.act_fn,
        #     norm=config.norm,
        #     cutoff_mode=config.cutoff_mode,
        #     ew_net_type=config.ew_net_type,
        #     num_x2h=config.num_x2h,
        #     num_h2x=config.num_h2x,
        #     r_max=config.r_max,
        #     x2h_out_fc=config.x2h_out_fc,
        #     sync_twoup=config.sync_twoup
        # )
        self.encoder_gloab = UniTransformerO2TwoUpdateGeneral(
            num_blocks=config.num_blocks,
            num_layers=config.num_layers,
            hidden_dim=config.hidden_dim,
            n_heads=config.n_heads,
            cutoff=config.g_cutoff,
            k=config.g_knn,
            edge_feat_dim=config.edge_feat_dim,
            num_r_gaussian=config.num_r_gaussian,
            num_node_types=config.num_node_types,
            act_fn=config.act_fn,
            norm=config.norm,
            cutoff_mode=config.cutoff_mode,
            ew_net_type=config.ew_net_type,
            num_x2h=config.num_x2h,
            num_h2x=config.num_h2x,
            r_max=config.r_max,
            x2h_out_fc=config.x2h_out_fc,
            sync_twoup=config.sync_twoup
        )
        self.pos_gloab_expert = MoE(3, 3, 5, 3, noisy_gating=True, k=2)
        # self.pos_local_expert = MoE(3, 3, 5, 3, noisy_gating=True, k=2)
        # self.atom_gloab_expert = MoE(128, 128, 5, 128, noisy_gating=True, k=2)
        # self.atom_local_expert = MoE(128, 128, 5, 128, noisy_gating=True, k=2)
        # self.fusion = DynamicFusionModule(h_dim=128,pos_dim=3)
        # self.advanced_fusion = AdvancedFusionModule(h_dim=128,pos_dim=3)
        self.attention_fusion = AttentionFusionModule(h_dim=128,pos_dim=3)
        # self.attention_fusion = FeatureFusionNetwork(h_input_dim=128,pos_input_dim=3)

        self.v_inference = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            ShiftedSoftplus(),
            nn.Linear(self.hidden_dim, ligand_atom_feature_dim),
        )
    def q_v_pred(self, log_v0, t, batch):
        # compute q(vt | v0)
        log_cumprod_alpha_t = extract(self.log_alphas_cumprod_v, t, batch)
        log_1_min_cumprod_alpha = extract(self.log_one_minus_alphas_cumprod_v, t, batch)

        log_probs = log_add_exp(
            log_v0 + log_cumprod_alpha_t,
            log_1_min_cumprod_alpha - np.log(self.num_classes)
        )
        return log_probs
    def q_v_sample(self, log_v0, t, batch):
        log_qvt_v0 = self.q_v_pred(log_v0, t, batch)
        sample_index = log_sample_categorical(log_qvt_v0)
        log_sample = index_to_log_onehot(sample_index, self.num_classes)
        return sample_index, log_sample
    def _predict_x0_from_eps(self, xt, eps, t, batch):
        pos0_from_e = extract(self.sqrt_recip_alphas_cumprod, t, batch) * xt - \
                      extract(self.sqrt_recipm1_alphas_cumprod, t, batch) * eps
        return pos0_from_e
    def q_pos_posterior(self, x0, xt, t, batch):
        # Compute the mean and variance of the diffusion posterior q(x_{t-1} | x_t, x_0)
        pos_model_mean = extract(self.posterior_mean_c0_coef, t, batch) * x0 + \
                         extract(self.posterior_mean_ct_coef, t, batch) * xt
        return pos_model_mean
    # atom type generative process
    def q_v_posterior(self, log_v0, log_vt, t, batch):
        # q(vt-1 | vt, v0) = q(vt | vt-1, x0) * q(vt-1 | x0) / q(vt | x0)
        t_minus_1 = t - 1
        # Remove negative values, will not be used anyway for final decoder
        t_minus_1 = torch.where(t_minus_1 < 0, torch.zeros_like(t_minus_1), t_minus_1)
        log_qvt1_v0 = self.q_v_pred(log_v0, t_minus_1, batch)
        unnormed_logprobs = log_qvt1_v0 + self.q_v_pred_one_timestep(log_vt, t, batch)
        log_vt1_given_vt_v0 = unnormed_logprobs - torch.logsumexp(unnormed_logprobs, dim=-1, keepdim=True)
        return log_vt1_given_vt_v0
    # atom type diffusion process
    def q_v_pred_one_timestep(self, log_vt_1, t, batch):
        # q(vt | vt-1)
        log_alpha_t = extract(self.log_alphas_v, t, batch)
        log_1_min_alpha_t = extract(self.log_one_minus_alphas_v, t, batch)

        # alpha_t * vt + (1 - alpha_t) 1 / K
        log_probs = log_add_exp(
            log_vt_1 + log_alpha_t,
            log_1_min_alpha_t - np.log(self.num_classes)
        )
        return log_probs
    # def categorical_kl(log_prob1, log_prob2):
    #     kl = (log_prob1.exp() * (log_prob1 - log_prob2)).sum(dim=1)
    #     return kl
    # def log_categorical(log_x_start, log_prob):
    #     return (log_x_start.exp() * log_prob).sum(dim=1)
    def compute_v_Lt(self, log_v_model_prob, log_v0, log_v_true_prob, t, batch):
        kl_v = categorical_kl(log_v_true_prob, log_v_model_prob)  # [num_atoms, ]
        decoder_nll_v = -log_categorical(log_v0, log_v_model_prob)  # L0
        assert kl_v.shape == decoder_nll_v.shape
        mask = (t == 0).float()[batch]
        loss_v = scatter_mean(mask * decoder_nll_v + (1. - mask) * kl_v, batch, dim=0)
        return loss_v
    def sample_time(self, num_graphs, device, method):
        if method == 'importance':
            if not (self.Lt_count > 10).all():
                return self.sample_time(num_graphs, device, method='symmetric')

            Lt_sqrt = torch.sqrt(self.Lt_history + 1e-10) + 0.0001
            Lt_sqrt[0] = Lt_sqrt[1]  # Overwrite decoder term with L1.
            pt_all = Lt_sqrt / Lt_sqrt.sum()

            time_step = torch.multinomial(pt_all, num_samples=num_graphs, replacement=True)
            pt = pt_all.gather(dim=0, index=time_step)
            return time_step, pt

        elif method == 'symmetric':
            time_step = torch.randint(
                0, self.num_timesteps, size=(num_graphs // 2 + 1,), device=device)
            time_step = torch.cat(
                [time_step, self.num_timesteps - time_step - 1], dim=0)[:num_graphs]
            pt = torch.ones_like(time_step).float() / self.num_timesteps
            return time_step, pt

        else:
            raise ValueError

    def forward(self, protein_pos, protein_v, batch_protein, ligand_pos, ligand_v, batch_ligand, time_step=None,
                return_all=False, fix_x=False):
        num_graphs = batch_protein.max().item() + 1
        protein_pos, ligand_pos, _ = center_pos(
            protein_pos, ligand_pos, batch_protein, batch_ligand, mode=self.center_pos_mode)

        # 1. sample noise levels
        if time_step is None:
            time_step, pt = self.sample_time(num_graphs, protein_pos.device, self.sample_time_method)
        else:
            pt = torch.ones_like(time_step).float() / self.num_timesteps
        a = self.alphas_cumprod.index_select(0, time_step)  # (num_graphs, )

        # 2. perturb pos and v
        a_pos = a[batch_ligand].unsqueeze(-1)  # (num_ligand_atoms, 1)
        pos_noise = torch.zeros_like(ligand_pos)
        pos_noise.normal_()
        # Xt = a.sqrt() * X0 + (1-a).sqrt() * eps
        ligand_pos_perturbed = a_pos.sqrt() * ligand_pos + (1.0 - a_pos).sqrt() * pos_noise  # pos_noise * std
        # Vt = a * V0 + (1-a) / K
        log_ligand_v0 = index_to_log_onehot(ligand_v, self.num_classes)
        ligand_v_perturbed, log_ligand_vt = self.q_v_sample(log_ligand_v0, time_step, batch_ligand)

        batch_size = batch_protein.max().item() + 1
        init_ligand_v = F.one_hot(ligand_v_perturbed, self.num_classes).float()
        # time embedding
        if self.time_emb_dim > 0:
            if self.time_emb_mode == 'simple':
                input_ligand_feat = torch.cat([
                    init_ligand_v,
                    (time_step / self.num_timesteps)[batch_ligand].unsqueeze(-1)
                ], -1)
            elif self.time_emb_mode == 'sin':
                time_feat = self.time_emb(time_step)
                input_ligand_feat = torch.cat([init_ligand_v, time_feat], -1)
            else:
                raise NotImplementedError
        else:
            input_ligand_feat = init_ligand_v
        h_protein = self.protein_atom_emb(protein_v)
        init_ligand_h = self.ligand_atom_emb(input_ligand_feat)
        # init_ligand_h, h_protein = self.atten_layer(init_ligand_h, h_protein)
        if self.config.node_indicator:
            h_protein = torch.cat([h_protein, torch.zeros(len(h_protein), 1).to(h_protein)], -1)
            init_ligand_h = torch.cat([init_ligand_h, torch.ones(len(init_ligand_h), 1).to(h_protein)], -1)
        # print(f'batch_porein:{batch_protein},batch_ligand:{batch_ligand}')
        h_all, pos_all, batch_all, mask_ligand = compose_context(
            h_protein=h_protein,
            h_ligand=init_ligand_h,
            pos_protein=protein_pos,
            pos_ligand=ligand_pos_perturbed,
            batch_protein=batch_protein,
            batch_ligand=batch_ligand,
        )
        # print(f'batch_all:{batch_all}')
        # POS MOE
        # pos_all, pos_moe_gloab_loss = self.pos_gloab_expert(pos_all)
        outputs_gloab = self.encoder_gloab(h_all, pos_all, mask_ligand, batch_all, return_all=return_all,fix_x=fix_x)
        outputs_local = self.encoder_local(h_all, pos_all, mask_ligand, batch_all, return_all=return_all)
        final_pos_gloab, final_h_gloab = outputs_gloab['x'], outputs_gloab['h']
        final_pos_local, final_h_local = outputs_local['x'], outputs_local['h']

        # final_pos, final_h = self.fusion(final_pos_gloab,final_pos_local,final_h_gloab,final_h_local)
        # final_pos, final_h = self.advanced_fusion(final_pos_gloab,final_pos_local,final_h_gloab,final_h_local)
        final_pos, final_h = self.attention_fusion(final_pos_gloab,final_pos_local,final_h_gloab,final_h_local)
        # print(final_pos.shape,final_h.shape)
        pred_ligand_pos, final_ligand_h = final_pos[mask_ligand], final_h[mask_ligand]

        pred_ligand_v = self.v_inference(final_ligand_h) 

        pred_pos_noise = pred_ligand_pos - ligand_pos_perturbed
        # atom position
        if self.model_mean_type == 'noise':
            pos0_from_e = self._predict_x0_from_eps(
                xt=ligand_pos_perturbed, eps=pred_pos_noise, t=time_step, batch=batch_ligand)
            pos_model_mean = self.q_pos_posterior(
                x0=pos0_from_e, xt=ligand_pos_perturbed, t=time_step, batch=batch_ligand)
        elif self.model_mean_type == 'C0':
            pos_model_mean = self.q_pos_posterior(
                x0=pred_ligand_pos, xt=ligand_pos_perturbed, t=time_step, batch=batch_ligand)
        else:
            raise ValueError

        # atom pos loss
        if self.model_mean_type == 'C0':
            target, pred = ligand_pos, pred_ligand_pos
        elif self.model_mean_type == 'noise':
            target, pred = pos_noise, pred_pos_noise
        else:
            raise ValueError
        loss_pos = scatter_mean(((pred - target) ** 2).sum(-1), batch_ligand, dim=0)
        loss_pos = torch.mean(loss_pos)

        # atom type loss
        log_ligand_v_recon = F.log_softmax(pred_ligand_v, dim=-1)
        log_v_model_prob = self.q_v_posterior(log_ligand_v_recon, log_ligand_vt, time_step, batch_ligand)
        log_v_true_prob = self.q_v_posterior(log_ligand_v0, log_ligand_vt, time_step, batch_ligand)
        kl_v = self.compute_v_Lt(log_v_model_prob=log_v_model_prob, log_v0=log_ligand_v0,
                                 log_v_true_prob=log_v_true_prob, t=time_step, batch=batch_ligand)
        loss_v = torch.mean(kl_v)
        loss = loss_pos + loss_v * self.loss_v_weight
        # loss = loss_pos + loss_v * self.loss_v_weight
        return {
            'loss_pos': loss_pos,
            'loss_v': loss_v,
            'loss': loss,
            'x0': ligand_pos,
            'pred_ligand_pos': pred_ligand_pos,
            'pred_ligand_v': pred_ligand_v,
            'pred_pos_noise': pred_pos_noise,
            'ligand_v_recon': F.softmax(pred_ligand_v, dim=-1)
        }
    @torch.no_grad()
    def sample_diffusion(self, protein_pos, protein_v, batch_protein,
                         init_ligand_pos, init_ligand_v, batch_ligand,
                         num_steps=None, center_pos_mode=None, pos_only=False):

        if num_steps is None:
            num_steps = self.num_timesteps
        num_graphs = batch_protein.max().item() + 1

        protein_pos, init_ligand_pos, offset = center_pos(
            protein_pos, init_ligand_pos, batch_protein, batch_ligand, mode=center_pos_mode)

        pos_traj, v_traj = [], []
        v0_pred_traj, vt_pred_traj = [], []
        ligand_pos, ligand_v = init_ligand_pos, init_ligand_v
        # time sequence
        time_seq = list(reversed(range(self.num_timesteps - num_steps, self.num_timesteps)))
        for i in tqdm(time_seq, desc='sampling', total=len(time_seq)):
            t = torch.full(size=(num_graphs,), fill_value=i, dtype=torch.long, device=protein_pos.device)
            init_ligand_v = F.one_hot(ligand_v, self.num_classes).float()
            # time embedding
            if self.time_emb_dim > 0:
                if self.time_emb_mode == 'simple':
                    input_ligand_feat = torch.cat([
                        init_ligand_v,
                        (t / self.num_timesteps)[batch_ligand].unsqueeze(-1)
                    ], -1)
                elif self.time_emb_mode == 'sin':
                    time_feat = self.time_emb(t)
                    input_ligand_feat = torch.cat([init_ligand_v, time_feat], -1)
                else:
                    raise NotImplementedError
            else:
                input_ligand_feat = init_ligand_v

            h_protein = self.protein_atom_emb(protein_v)
            init_ligand_h = self.ligand_atom_emb(input_ligand_feat)
            # init_ligand_h, h_protein = self.atten_layer(init_ligand_h, h_protein)
            if self.config.node_indicator:
                h_protein = torch.cat([h_protein, torch.zeros(len(h_protein), 1).to(h_protein)], -1)
                init_ligand_h = torch.cat([init_ligand_h, torch.ones(len(init_ligand_h), 1).to(h_protein)], -1)
            
            h_all, pos_all, batch_all, mask_ligand = compose_context(
                h_protein=h_protein,
                h_ligand=init_ligand_h,
                pos_protein=protein_pos,
                pos_ligand=ligand_pos,
                batch_protein=batch_protein,
                batch_ligand=batch_ligand,
            )
            outputs_gloab = self.encoder_gloab(h_all, pos_all, mask_ligand, batch_all, return_all=False, fix_x=False)
            outputs_local = self.encoder_local(h_all, pos_all, mask_ligand, batch_all, return_all=False)
            final_pos_gloab, final_h_gloab = outputs_gloab['x'], outputs_gloab['h']
            final_pos_local, final_h_local = outputs_local['x'], outputs_local['h']
            # exlplanier_pos_gloab, explanier_h_gloab = final_pos_gloab[mask_ligand], final_h_gloab[mask_ligand]
            # exlplanier_pos_local, explanier_h_local = final_pos_local[mask_ligand], final_h_local[mask_ligand]
            # final_pos, final_h = self.fusion(final_pos_gloab,final_pos_local,final_h_gloab,final_h_local)
            final_pos, final_h = self.attention_fusion(final_pos_gloab,final_pos_local,final_h_gloab,final_h_local)
            pred_ligand_pos, final_ligand_h = final_pos[mask_ligand], final_h[mask_ligand]
            pred_ligand_v = self.v_inference(final_ligand_h)
            # explainer_h_final = pred_ligand_v
            # Compute posterior mean and variance
            if self.model_mean_type == 'noise':
                pred_pos_noise = pred_ligand_pos - ligand_pos
                pos0_from_e = self._predict_x0_from_eps(xt=ligand_pos, eps=pred_pos_noise, t=t, batch=batch_ligand)
                v0_from_e = pred_ligand_v
            elif self.model_mean_type == 'C0':
                pos0_from_e = pred_ligand_pos
                v0_from_e = pred_ligand_v
            else:
                raise ValueError

            pos_model_mean = self.q_pos_posterior(x0=pos0_from_e, xt=ligand_pos, t=t, batch=batch_ligand)
            pos_log_variance = extract(self.posterior_logvar, t, batch_ligand)
            # no noise when t == 0
            nonzero_mask = (1 - (t == 0).float())[batch_ligand].unsqueeze(-1)
            ligand_pos_next = pos_model_mean + nonzero_mask * (0.5 * pos_log_variance).exp() * torch.randn_like(
                ligand_pos)
            ligand_pos = ligand_pos_next

            if not pos_only:
                log_ligand_v_recon = F.log_softmax(v0_from_e, dim=-1)
                log_ligand_v = index_to_log_onehot(ligand_v, self.num_classes)
                log_model_prob = self.q_v_posterior(log_ligand_v_recon, log_ligand_v, t, batch_ligand)
                ligand_v_next = log_sample_categorical(log_model_prob)

                v0_pred_traj.append(log_ligand_v_recon.clone().cpu())
                vt_pred_traj.append(log_model_prob.clone().cpu())
                ligand_v = ligand_v_next

            ori_ligand_pos = ligand_pos + offset[batch_ligand]
            pos_traj.append(ori_ligand_pos.clone().cpu())
            v_traj.append(ligand_v.clone().cpu())
            # print(i)
            # if i == 800:
            #     break
            
        # torch.save(explanier_h_local.to('cpu'),'explainer_h_local.pt')
        # torch.save(explanier_h_gloab.to('cpu'),'explainer_h_gloab.pt')
        # torch.save(explainer_h_final.to('cpu'),'explainer_h.pt')
        ligand_pos = ligand_pos + offset[batch_ligand]
        return {
            'pos': ligand_pos,
            'v': ligand_v,
            'pos_traj': pos_traj,
            'v_traj': v_traj,
            'v0_traj': v0_pred_traj,
            'vt_traj': vt_pred_traj,
        }
