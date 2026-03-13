from __future__ import annotations

import math

import torch
import torch.nn as nn


class FeedForward(nn.Module):
    def __init__(self, hidden_dim: int, mlp_ratio: float = 4.0, dropout: float = 0.0) -> None:
        super().__init__()
        inner_dim = int(hidden_dim * mlp_ratio)
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, inner_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(inner_dim, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SelfAttentionBlock(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, mlp_ratio: float, dropout: float) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ffn = FeedForward(hidden_dim, mlp_ratio=mlp_ratio, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x), need_weights=False)
        x = x + attn_out
        x = x + self.ffn(self.norm2(x))
        return x


class CrossAttentionBlock(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, mlp_ratio: float, dropout: float) -> None:
        super().__init__()
        self.q_norm = nn.LayerNorm(hidden_dim)
        self.kv_norm = nn.LayerNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.out_norm = nn.LayerNorm(hidden_dim)
        self.ffn = FeedForward(hidden_dim, mlp_ratio=mlp_ratio, dropout=dropout)

    def forward(self, query: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        q = self.q_norm(query)
        kv = self.kv_norm(memory)
        cross_out, _ = self.attn(q, kv, kv, need_weights=False)
        query = query + cross_out
        query = query + self.ffn(self.out_norm(query))
        return query


class LatentMixer(nn.Module):
    """
    Linear-complexity token mixer:
    1) latents attend to tokens (cross attention)
    2) latents run self-attention
    3) tokens attend back to latents (cross attention)
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        num_latents: int,
        mlp_ratio: float,
        dropout: float,
    ) -> None:
        super().__init__()
        self.latent_seed = nn.Parameter(torch.randn(num_latents, hidden_dim) * 0.02)
        self.latent_from_tokens = CrossAttentionBlock(hidden_dim, num_heads, mlp_ratio, dropout)
        self.latent_self = SelfAttentionBlock(hidden_dim, num_heads, mlp_ratio, dropout)
        self.tokens_from_latent = CrossAttentionBlock(hidden_dim, num_heads, mlp_ratio, dropout)

    def forward(self, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = int(tokens.shape[0])
        latents = self.latent_seed.unsqueeze(0).expand(batch_size, -1, -1)
        latents = self.latent_from_tokens(latents, tokens)
        latents = self.latent_self(latents)
        tokens = self.tokens_from_latent(tokens, latents)
        return tokens, latents


class EntityDecoderLayer(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        num_latents: int,
        mlp_ratio: float,
        dropout: float,
    ) -> None:
        super().__init__()
        self.from_modal_memory = CrossAttentionBlock(hidden_dim, num_heads, mlp_ratio, dropout)
        self.latent_mixer = LatentMixer(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_latents=num_latents,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
        )

    def forward(self, tokens: torch.Tensor, memory_tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        tokens = self.from_modal_memory(tokens, memory_tokens)
        tokens, latents = self.latent_mixer(tokens)
        return tokens, latents


class EntityDecoder(nn.Module):
    def __init__(
        self,
        *,
        max_tokens: int,
        hidden_dim: int,
        num_layers: int,
        num_heads: int,
        num_latents: int,
        mlp_ratio: float,
        dropout: float,
    ) -> None:
        super().__init__()
        self.query_embed = nn.Parameter(torch.randn(max_tokens, hidden_dim) * 0.02)
        self.context_proj = nn.Linear(hidden_dim, hidden_dim)
        self.layers = nn.ModuleList(
            [
                EntityDecoderLayer(
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    num_latents=num_latents,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, *, context_token: torch.Tensor, memory_tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = int(context_token.shape[0])
        context_bias = self.context_proj(context_token).unsqueeze(1)
        tokens = self.query_embed.unsqueeze(0).expand(batch_size, -1, -1) + context_bias

        latents = tokens.new_zeros((batch_size, 0, tokens.shape[-1]))
        for layer in self.layers:
            tokens, latents = layer(tokens, memory_tokens)
        return tokens, latents


class BRepHead(nn.Module):
    """
    B-rep decoder head with mixed attention:
    - cross-attention from modality memory tokens (pcd/voxel/fused)
    - latent self-attention mixer for each entity type
    - cross-type latent exchanges (V->E->F->V)
    """

    def __init__(
        self,
        *,
        pcd_feature_dim: int,
        voxel_feature_dim: int,
        fused_feature_dim: int,
        num_curve_types: int = 9,
        num_surface_types: int = 11,
        max_vertices: int = 4000,
        max_edges: int = 2000,
        max_faces: int = 1000,
        v_feat_dim: int = 3,
        e_feat_dim: int = 73,
        f_feat_dim: int = 174,
        hidden_dim: int = 256,
        num_heads: int = 8,
        num_decoder_layers: int = 2,
        num_latents: int = 64,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.max_vertices = max_vertices
        self.max_edges = max_edges
        self.max_faces = max_faces
        self.num_curve_types = num_curve_types
        self.num_surface_types = num_surface_types

        self.pcd_proj = nn.Linear(pcd_feature_dim, hidden_dim)
        self.voxel_proj = nn.Linear(voxel_feature_dim, hidden_dim)
        self.fused_proj = nn.Linear(fused_feature_dim, hidden_dim)
        self.memory_pos = nn.Parameter(torch.randn(3, hidden_dim) * 0.02)
        self.global_context_proj = nn.Linear(fused_feature_dim, hidden_dim)

        self.vertex_decoder = EntityDecoder(
            max_tokens=max_vertices,
            hidden_dim=hidden_dim,
            num_layers=num_decoder_layers,
            num_heads=num_heads,
            num_latents=num_latents,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
        )
        self.edge_decoder = EntityDecoder(
            max_tokens=max_edges,
            hidden_dim=hidden_dim,
            num_layers=num_decoder_layers,
            num_heads=num_heads,
            num_latents=num_latents,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
        )
        self.face_decoder = EntityDecoder(
            max_tokens=max_faces,
            hidden_dim=hidden_dim,
            num_layers=num_decoder_layers,
            num_heads=num_heads,
            num_latents=num_latents,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
        )

        self.edge_latent_from_vertex = CrossAttentionBlock(hidden_dim, num_heads, mlp_ratio, dropout)
        self.face_latent_from_edge = CrossAttentionBlock(hidden_dim, num_heads, mlp_ratio, dropout)
        self.vertex_latent_from_face = CrossAttentionBlock(hidden_dim, num_heads, mlp_ratio, dropout)

        self.edge_tokens_from_latent = CrossAttentionBlock(hidden_dim, num_heads, mlp_ratio, dropout)
        self.face_tokens_from_latent = CrossAttentionBlock(hidden_dim, num_heads, mlp_ratio, dropout)
        self.vertex_tokens_from_latent = CrossAttentionBlock(hidden_dim, num_heads, mlp_ratio, dropout)

        self.vertex_feat_head = nn.Linear(hidden_dim, v_feat_dim)
        self.edge_feat_head = nn.Linear(hidden_dim, e_feat_dim)
        self.face_feat_head = nn.Linear(hidden_dim, f_feat_dim)
        self.edge_type_head = nn.Linear(hidden_dim, num_curve_types)
        self.face_type_head = nn.Linear(hidden_dim, num_surface_types)
        self.edge_ori_head = nn.Linear(hidden_dim, 1)
        self.face_ori_head = nn.Linear(hidden_dim, 1)

        self.vertex_topology_proj = nn.Linear(hidden_dim, hidden_dim)
        self.edge_topology_proj = nn.Linear(hidden_dim, hidden_dim)
        self.face_topology_proj = nn.Linear(hidden_dim, hidden_dim)
        self.topology_scale = 1.0 / math.sqrt(float(hidden_dim))

        self.count_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 3),
        )
        self.register_buffer(
            "max_counts",
            torch.tensor([float(max_vertices), float(max_edges), float(max_faces)]),
            persistent=False,
        )

    def _build_memory_tokens(
        self,
        *,
        pcd_feature: torch.Tensor,
        voxel_feature: torch.Tensor,
        fused_feature: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        pcd_token = self.pcd_proj(pcd_feature)
        voxel_token = self.voxel_proj(voxel_feature)
        fused_token = self.fused_proj(fused_feature)
        memory = torch.stack([pcd_token, voxel_token, fused_token], dim=1)
        memory = memory + self.memory_pos.unsqueeze(0)
        context = self.global_context_proj(fused_feature)
        return memory, context

    def forward(
        self,
        *,
        pcd_feature: torch.Tensor,
        voxel_feature: torch.Tensor,
        fused_feature: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        memory_tokens, context_token = self._build_memory_tokens(
            pcd_feature=pcd_feature,
            voxel_feature=voxel_feature,
            fused_feature=fused_feature,
        )

        v_tokens, v_latent = self.vertex_decoder(context_token=context_token, memory_tokens=memory_tokens)
        e_tokens, e_latent = self.edge_decoder(context_token=context_token, memory_tokens=memory_tokens)
        f_tokens, f_latent = self.face_decoder(context_token=context_token, memory_tokens=memory_tokens)

        if v_latent.shape[1] > 0 and e_latent.shape[1] > 0:
            e_latent = self.edge_latent_from_vertex(e_latent, v_latent)
            e_tokens = self.edge_tokens_from_latent(e_tokens, e_latent)
        if e_latent.shape[1] > 0 and f_latent.shape[1] > 0:
            f_latent = self.face_latent_from_edge(f_latent, e_latent)
            f_tokens = self.face_tokens_from_latent(f_tokens, f_latent)
        if f_latent.shape[1] > 0 and v_latent.shape[1] > 0:
            v_latent = self.vertex_latent_from_face(v_latent, f_latent)
            v_tokens = self.vertex_tokens_from_latent(v_tokens, v_latent)

        pred_v_feat = self.vertex_feat_head(v_tokens)
        pred_e_feat = self.edge_feat_head(e_tokens)
        pred_f_feat = self.face_feat_head(f_tokens)
        pred_e_type_logits = self.edge_type_head(e_tokens)
        pred_f_type_logits = self.face_type_head(f_tokens)
        pred_e_ori_logits = self.edge_ori_head(e_tokens).squeeze(-1)
        pred_f_ori_logits = self.face_ori_head(f_tokens).squeeze(-1)

        v_topo = self.vertex_topology_proj(v_tokens)
        e_topo = self.edge_topology_proj(e_tokens)
        f_topo = self.face_topology_proj(f_tokens)

        pred_adj_ev = torch.tanh(torch.einsum("beh,bvh->bev", e_topo, v_topo) * self.topology_scale)
        pred_adj_fe = torch.tanh(torch.einsum("bfh,beh->bfe", f_topo, e_topo) * self.topology_scale)

        raw_counts = self.count_head(context_token)
        pred_counts = torch.sigmoid(raw_counts) * self.max_counts.unsqueeze(0)

        return {
            "v_feat": pred_v_feat,
            "e_feat": pred_e_feat,
            "f_feat": pred_f_feat,
            "e_type_logits": pred_e_type_logits,
            "f_type_logits": pred_f_type_logits,
            "e_ori_logits": pred_e_ori_logits,
            "f_ori_logits": pred_f_ori_logits,
            "adj_ev": pred_adj_ev,
            "adj_fe": pred_adj_fe,
            "counts": pred_counts,
        }
