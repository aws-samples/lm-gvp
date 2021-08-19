# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""
Utils for preparing self-attention maps and contact maps.
"""
import numpy as np
import torch
from transformers import BertModel
from sklearn.metrics import pairwise_distances


def calc_contact_map(protein: dict, dist_thres=10, k=6):
    """
    Calculate a protein's contact map
    Args:
        - protein: a dict object from LM-GVP formatted data (json record).
        - dist_thres: threshold for C-alpha distance to create an edge
        - k: number of amino acids away to filter out local contacts
    """
    coords = np.asarray(protein["coords"])
    # coordinates of C-alpha atoms
    X_ca = coords[:, 1]
    mask = np.isfinite(X_ca.sum(axis=1))
    # remove residues with NaNs in C-alpha coords
    X_ca = X_ca[mask]

    seqlen = X_ca.shape[0]
    # take the upper triangle of the symetric matrix
    idx = np.triu_indices(seqlen, k)
    dist_mat = pairwise_distances(X_ca, metric="euclidean")[idx]

    contact_map = (dist_mat < dist_thres).astype(int)

    protein["contact_map"] = contact_map
    protein["idx"] = idx
    return


def calc_self_attn(
    bert_model: BertModel, protein: dict, device="cuda:0", **kwargs
):
    """Calculate self-attention matrices given Bert model for one protein.
    Args:
        - bert_model: a BertModel instance
        - protein: a dict object from LM-GVP formatted data (json record).
        - device: device to do the computation
    Returns:
        - a torch.tensor of shape: [n_maps, seqlen, seqlen]
    """
    bert_model = bert_model.to(device)
    bert_model.eval()
    with torch.no_grad():
        self_attn_mats = bert_model(
            protein["input_ids"].unsqueeze(0).to(device),
            attention_mask=protein["attention_mask"].unsqueeze(0).to(device),
            output_attentions=True,
        ).attentions

    # gather self-attention map from all layers together
    n_layers = len(self_attn_mats)
    batch_size, n_heads, seqlen, _ = self_attn_mats[0].size()
    self_attn_mats = torch.stack(self_attn_mats, dim=1).view(
        batch_size, n_layers * n_heads, seqlen, seqlen
    )
    # remove [CLS] and [SEP]
    self_attn_mats = self_attn_mats[..., 1:-1, 1:-1]

    if self_attn_mats.size()[0] == 1:
        self_attn_mats = self_attn_mats.squeeze(0)

    self_attn_mats = self_attn_mats.detach().cpu()

    return self_attn_mats


def symmetrize(x):
    """Make layer symmetric in the final two
    dimensions, used for contact prediction."""
    return x + x.transpose(-1, -2)


def apc(x):
    """Perform average product correct, used for contact prediction."""
    a1 = x.sum(-1, keepdims=True)
    a2 = x.sum(-2, keepdims=True)
    a12 = x.sum((-1, -2), keepdims=True)

    avg = a1 * a2
    avg.div_(a12)  # in-place to reduce memory
    normalized = x - avg
    return normalized


def extract_process_attns(bert_model: BertModel, protein: dict, **kwargs):
    """Calculate self-attention matrix, process with apc, then take the upper
    triangle.
    Args:
        - bert_model: a BertModel instance
        - protein: a dict object from LM-GVP formatted data (json record).
    """
    attns = calc_self_attn(bert_model, protein, **kwargs)
    attns = apc(symmetrize(attns)).numpy()
    idx = protein["idx"]
    attns = np.asarray([mat[idx] for mat in attns])
    attns = attns.T
    return attns


def calc_dataset_attn_and_contact(
    dataset: list, bert_model: BertModel, **kwargs
):
    """Calculate attention matrices and contacts for a list of protein
    records.
    Args:
        - dataset: a list of protein records
        - bert_model: a BertModel instance
    """
    self_attn_mats = []  # collect tensors of processed attn maps
    contact_maps = []  # collect tensors of contact map

    for protein in dataset:
        self_attn_mat = extract_process_attns(bert_model, protein, **kwargs)
        self_attn_mats.append(self_attn_mat)
        contact_maps.append(protein["contact_map"])

    self_attn_mats = np.vstack(self_attn_mats)
    contact_maps = np.concatenate(contact_maps)

    return self_attn_mats, contact_maps
