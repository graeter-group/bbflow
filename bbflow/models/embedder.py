# Copyright (c) 2025 Max Planck Institute for Polymer Research
# Copyright (c) 2024 HITS gGmbH
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import functools as fn
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

import gafl.models.utils as mu
from gafl.data import utils as du

from gafl.models.ipa_pytorch import Linear
from gafl.models.utils import get_index_embedding, get_time_embedding, get_length_embedding

class MLP(nn.Module):
    def __init__(self, c1, c2, num_layers=2, use_layer_norm=True, dropout=0) -> None:
        super(MLP, self).__init__()

        self.layers = []
        for _ in range(num_layers-1):
            self.layers.append(Linear(c1, c1, init="relu"))
            self.layers.append(nn.ReLU())
            self.layers.append(nn.Dropout(dropout))
        self.layers.append(Linear(c1, c1, init="final"))
        self.layers = nn.Sequential(*self.layers)

        if c1 != c2:
            self.final_layer = nn.Linear(c1, c2)
        else:
            self.final_layer = nn.Identity()
        
        self.use_layer_norm = use_layer_norm
        if use_layer_norm:
            self.layer_norm = nn.LayerNorm(c2)

    def forward(self, x):
        x = self.layers(x) + x
        x = self.final_layer(x)
        if self.use_layer_norm:
            x = self.layer_norm(x)
        return x


def embed_direction(trans, rots, dist_cutoff=np.inf):
    """
    Compute the position of the other residues from the
    perspective of each residue.
    
    Args:
        trans: [B, N, 3] translation vectors
        rots: [B, N, 3, 3] rotation matrices
    Returns:
        [B, N, N, 3] relative positions from each residue to each other residue
    """

    B, N, _ = trans.shape
    device = trans.device

    frames = du.create_rigid(rots, trans)
    rel_pos = frames[:,:,None].invert_apply(trans[:,None,:])
    # rel_pos.shape = [B, N, N, 3]
    distances = torch.linalg.norm(rel_pos, dim=-1)
    rel_pos = rel_pos / (distances[...,None] + 1e-6)
    indices = torch.arange(N, device=device)
    rel_pos[:, indices, indices, :] = torch.zeros(3, device=device)
    if dist_cutoff < np.inf:
        # Set all direction vectors to zero if the distance is greater than dist_cutoff
        mask = distances > dist_cutoff
        rel_pos[mask,:] = 0

    return rel_pos




class Embedder(nn.Module):
    def __init__(self, model_conf):
        super(Embedder, self).__init__()
        self.model_conf = model_conf
        self.embed_aatype = model_conf.embed.embed_aatype

        if "time_embed_dim" in model_conf.embed:
            self.time_embed_dim = model_conf.embed.time_embed_dim
        else:
            self.time_embed_dim = 0

        if "embed_distogram" in model_conf.embed:
            self.embed_distogram = model_conf.embed.embed_distogram
        else:
            self.embed_distogram = True

        if "embed_self_conditioning" in model_conf.embed:
            self.embed_self_conditioning = model_conf.embed.embed_self_conditioning
        else:
            self.embed_self_conditioning = True


        length_embed_dim = model_conf.embed.length_embed_dim

        node_dim_in = self.time_embed_dim + length_embed_dim

        edge_dim_in = 2 * self.time_embed_dim
        edge_dim_in += 2 * model_conf.node_embed_size
        edge_dim_in += self.model_conf.embed.num_bins_distance
        if self.embed_distogram:
            edge_dim_in += self.model_conf.embed.num_bins_distance
        if self.embed_self_conditioning:
            edge_dim_in += self.model_conf.embed.num_bins_distance 

        self.embed_direction = model_conf.embed.embed_direction
        if self.embed_direction:
            if hasattr(model_conf.embed, "embed_direction_cutoff") and model_conf.embed.embed_direction_cutoff is not None:
                self.embed_direction_cutoff = model_conf.embed.embed_direction_cutoff
            else:
                self.embed_direction_cutoff = np.inf
            edge_dim_in += 3

        # positional encoding
        # if model_conf.embed.index_embed_dim > 0:
        if hasattr(model_conf.embed, "index_embed_type"):
            self.index_embed_type = model_conf.embed.index_embed_type
            self.index_embed_dim = model_conf.embed.index_embed_dim
        else:
            self.index_embed_dim = model_conf.embed.index_embed_dim
            self.index_embed_type = None if self.index_embed_dim == 0 else "full"

        if self.index_embed_type is None or self.index_embed_type == "none":
            self.index_embed_type = None
        elif self.index_embed_type == "full":
            assert self.index_embed_dim > 0, "index_embed_dim must be > 0 if index_embed_type is 'full'."
            node_dim_in += self.index_embed_dim
            edge_dim_in += self.index_embed_dim
            self.linear_relpos = nn.Linear(
                self.index_embed_dim, self.index_embed_dim
            )
        elif self.index_embed_type == "neighbor":
            edge_dim_in += 1
        else:
            raise ValueError("index_embed_type must be None/\"none\", 'full' or 'neighbor'.")


        if self.embed_aatype:
            self.aatype_embedding = nn.Linear(21, model_conf.embed.aatype_embed_dim)
            node_dim_in += model_conf.embed.aatype_embed_dim


        self.linear_s_p = nn.Linear(model_conf.node_embed_size, model_conf.node_embed_size)

        self.node_embedding = MLP(node_dim_in, model_conf.node_embed_size, num_layers=2)
        self.edge_embedding = MLP(edge_dim_in, model_conf.edge_embed_size, num_layers=2)


        self.timestep_embedder = fn.partial(
            get_time_embedding,
            embedding_dim=self.time_embed_dim,
        )
        self.index_embedder = fn.partial(
            get_index_embedding,
            embed_size=self.model_conf.embed.index_embed_dim,
        )
        self.length_embedder = fn.partial(
            get_length_embedding,
            embed_size=self.model_conf.embed.length_embed_dim,
            max_len=2056
        )

    def forward(self, input_features):
        """
        Args:
            t:
                [*, ] time step sampled from [0, 1]
            fixed_mask:
                [*, N_res] mask of fixed (motif) residues
            self_conditioning_ca:
                [*, N_res, 3] Ca positions of self-conditioning input
            rigids:
                [*, N_res] Frame object
        Returns:
            [*, N_res, node_dim] initial node features
            [*, N_res, N_res, edge_dim] initial edge features
        """
        B, N = input_features["B,N"]
        device = input_features["device"]
        
        h_0 = []
        z_0 = []

        if self.time_embed_dim > 0:
            t_embed = self.timestep_embedder(input_features["time"][:,0])[:, None, :].repeat(1, N, 1)
            h_0.append(t_embed)
            z_0.append(self._cross_concat(t_embed, B, N))
        
        if self.embed_aatype:
            h_0.append(self.aatype_embedding(input_features["seq"]))

        for ca_pos_name, ca_pos in input_features["CA_pos"].items():
            if not self.embed_distogram and ca_pos_name == "trans_equilibrium":
                continue
            ca_dgram = mu.calc_distogram(
                ca_pos,
                self.model_conf.embed.min_bin_distance,
                self.model_conf.embed.max_bin_distance,
                self.model_conf.embed.num_bins_distance,
            )

            z_0.append(ca_dgram.reshape([B, N**2, -1]))

        if self.embed_direction:
            direction_embedding = embed_direction(
                input_features["CA_pos"]["trans_equilibrium"],
                input_features["rotmats_equilibrium"],
                dist_cutoff=self.embed_direction_cutoff
            )
            z_0.append(direction_embedding.reshape([B, N**2, -1]))


        # TODO: get_length_embedding expects a tensor of shape [...,N]. 
        # It ignores elements and just uses .shape[-1] and device.
        # Don't want to change it, so we just pass in an empty tensor.
        length_embedding = self.length_embedder(
            torch.empty(1, N).to(device=device)
        )
        length_embedding = length_embedding.repeat([B, 1, 1])
        h_0.append(length_embedding)

        if self.index_embed_type is None:
            pass
        elif self.index_embed_type == "full":
            seq_idx = torch.arange(N).repeat(B, 1).to(device=device)
            h_0.append(self.index_embedder(seq_idx))
            rel_seq_offset = seq_idx[:, :, None] - seq_idx[:, None, :]
            rel_seq_offset = rel_seq_offset.reshape([B, N**2])
            z_0.append(self.linear_relpos(self.index_embedder(rel_seq_offset)))
        elif self.index_embed_type == "neighbor":
            neighbors = torch.diag_embed(torch.ones(N-1, device=device), offset=1) + \
                torch.diag_embed(torch.ones(N-1, device=device), offset=-1)
            neighbors = neighbors[None].repeat(B, 1, 1).reshape([B, N**2])[...,None]
            z_0.append(neighbors)
        else:
            raise ValueError("index_embed_type must be None/'none', 'full' or 'neighbor'.")

        h_0 = torch.cat(h_0, dim=-1)
        h_0 = self.node_embedding(h_0)

        z_0.append(self._cross_concat(self.linear_s_p(h_0), B, N))

        z_0 = torch.cat(z_0, dim=-1)
        z_0 = self.edge_embedding(z_0)
        z_0 = z_0.reshape([B, N, N, -1])

        return h_0, z_0

    def _cross_concat(self, feats_1d, B, N):
        return torch.cat([
            torch.tile(feats_1d[:, :, None, :], (1, 1, N, 1)),
            torch.tile(feats_1d[:, None, :, :], (1, N, 1, 1)),
        ], dim=-1).float().reshape([B, N**2, -1])
    
