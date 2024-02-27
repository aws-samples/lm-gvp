# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""
Pytorch map style dataset classes for proteins (seq, struct, seq+struct).
Modified from https://github.com/drorlab/gvp-pytorch/blob/main/gvp/data.py
"""
import math
import multiprocessing
import os
import tqdm
import numpy as np
import torch
import torch.utils.data as data
import torch.nn.functional as F
import torch_geometric
import torch_cluster




def _normalize(tensor, dim=-1):
    """
    Normalizes a `torch.Tensor` along dimension `dim` without `nan`s.

    Args:
        tensor: Torch tensor to be normalized.
        dim: Integer. Dimension to normalize across.

    Returns:
        Normalized tensor with zeros instead of nan's or infinity values.

    """
    return torch.nan_to_num(
        torch.div(tensor, torch.norm(tensor, dim=dim, keepdim=True))
    )


def _rbf(D, D_min=0.0, D_max=20.0, D_count=16, device="cpu"):
    """
    From https://github.com/jingraham/neurips19-graph-protein-design

    Returns an RBF embedding of `torch.Tensor` `D` along a new axis=-1.
    That is, if `D` has shape [...dims], then the returned tensor will have
    shape [...dims, D_count].


    Args:
        D: generic torch tensor
        D_min: Float. Minimum of the sequence of numbers created.
        D_max: Float. Max of the sequence of numbers created.
        D_count: Positive integer. Count of the numbers in the sequence. It is also lenght of the new dimension (-1) created in D.
        device: Device where D is stored.

    Return:
        Input `D` matrix with an RBF embedding along axis -1.
    """
    D_mu = torch.linspace(D_min, D_max, D_count, device=device)
    D_mu = D_mu.view([1, -1])
    D_sigma = (D_max - D_min) / D_count
    D_expand = torch.unsqueeze(D, -1)

    RBF = torch.exp(-(((D_expand - D_mu) / D_sigma) ** 2))
    return RBF



class BaseProteinGraphDataset(data.Dataset):
    """Dataset for the Base Protein Graph."""

    def __init__(
        self,
        data_list,
        num_positional_embeddings=16,
        top_k=10,
        num_rbf=16,
        device="cpu",
        preprocess=True,
    ):
        """
        Initializes the dataset

        Args:
            data_list: List containint the initial dataset
            num_positional_embeddings: Integer specifying the number of positional embeddings.
            top_k: Integer k to use in kNN when constructing the graph
            num_rbf: Integer specifying number of radial basis functions
            device: Device to allocate the tensors.
            preprocess: Whether to preprocess the data_list.

        Returns:
            None
        """

        super(BaseProteinGraphDataset, self).__init__()

        self.data_list = data_list
        self.top_k = top_k
        self.queue = []
        self.num_rbf = num_rbf
        self.num_positional_embeddings = num_positional_embeddings
        self.device = device
        if preprocess:
            print("Preprocessing data...")
            self._preprocess()

    def _preprocess(self):

        """Preprocess all the records in `data_list` with
        `_featurize_as_graph`"""
        for i in tqdm.tqdm(range(len(self.data_list))):
            self.data_list[i] = self._featurize_as_graph(self.data_list[i])

    def __getitem__(self, i):
        if isinstance(self.data_list[i], dict):
            # if not processed, process this instance and update
            self.data_list[i] = self._featurize_as_graph(self.data_list[i])
        return self.data_list[i]

    def __len__(self):
        return len(self.data_list)

    def _featurize_as_graph(self, protein):
        """Placeholder for the _featurize_as_graph method implemented in child classes.

        Args:
            protein: a dict representing a data object

        Returns:
            None
        """
        raise NotImplementedError

    def _dihedrals(self, X, eps=1e-7):
        """Compute sines and cosines dihedral angles (phi, psi, and omega)

        Args:
            X: torch.Tensor specifying coordinates of key atoms (N, CA, C, O) in 3D space with shape [seq_len, 4, 3]
            eps: Float defining the epsilon using to clamp the angle between normals: min= -1*eps, max=1-eps

        Returns:
            Sines and cosines dihedral angles as a torch.Tensor of shape [seq_len, 6]
        """
        # From https://github.com/jingraham/neurips19-graph-protein-design

        X = torch.reshape(X[:, :3], [3 * X.shape[0], 3])
        dX = X[1:] - X[:-1]
        U = _normalize(dX, dim=-1)
        u_2 = U[:-2]
        u_1 = U[1:-1]
        u_0 = U[2:]

        # Backbone normals
        n_2 = _normalize(torch.cross(u_2, u_1), dim=-1)
        n_1 = _normalize(torch.cross(u_1, u_0), dim=-1)

        # Angle between normals
        cosD = torch.sum(n_2 * n_1, -1)
        cosD = torch.clamp(cosD, -1 + eps, 1 - eps)
        D = torch.sign(torch.sum(u_2 * n_1, -1)) * torch.acos(cosD)

        # This scheme will remove phi[0], psi[-1], omega[-1]
        D = F.pad(D, [1, 2])
        D = torch.reshape(D, [-1, 3])
        # Lift angle representations to the circle
        D_features = torch.cat([torch.cos(D), torch.sin(D)], 1)
        return D_features

    def _positional_embeddings(self, edge_index, num_embeddings=None):
        """Creates and returns the positional embeddings.

        Args:
            edge_index: torch.Tensor representing edges in COO format with shape [2, num_edges].
            num_embeddings: Integer representing the number of embeddings.

        Returns:
            Positional embeddings as a torch tensor
        """
        # From https://github.com/jingraham/neurips19-graph-protein-design
        num_embeddings = num_embeddings or self.num_positional_embeddings
        d = edge_index[0] - edge_index[1]

        frequency = torch.exp(
            torch.arange(
                0, num_embeddings, 2, dtype=torch.float32, device=self.device
            )
            * -(np.log(10000.0) / num_embeddings)
        )
        angles = d.unsqueeze(-1) * frequency
        E = torch.cat((torch.cos(angles), torch.sin(angles)), -1)
        return E

    def _orientations(self, X):
        """Compute orientations between pairs of atoms from neighboring residues.

        Args:
            X: torch.Tensor representing atom coordinates with shape [n_atoms, 3]

        Returns:
            torch.Tensor atom pair orientations
        """
        forward = _normalize(X[1:] - X[:-1])
        backward = _normalize(X[:-1] - X[1:])
        forward = F.pad(forward, [0, 0, 0, 1])
        backward = F.pad(backward, [0, 0, 1, 0])
        return torch.cat([forward.unsqueeze(-2), backward.unsqueeze(-2)], -2)

    def _sidechains(self, X):
        """Compute the unit vector representing the imputed side chain directions (C_beta - C_alpha).

        Args:
            X: torch.Tensor specifying coordinates of key atoms (N, CA, C, O) in 3D space with shape [seq_len, 4, 3]

        Returns:
            Torch tensor representing side chain directions with shape [seq_len, 3]
        """
        n, origin, c = X[:, 0], X[:, 1], X[:, 2]
        c, n = _normalize(c - origin), _normalize(n - origin)
        bisector = _normalize(c + n)
        perp = _normalize(torch.cross(c, n))
        vec = -bisector * math.sqrt(1 / 3) - perp * math.sqrt(2 / 3)
        return vec


class ProteinGraphDataset(BaseProteinGraphDataset):
    """
    A map-syle `torch.utils.data.Dataset` which transforms JSON/dictionary
    -style protein structures into featurized protein graphs as described in
    the GVP manuscript (https://arxiv.org/abs/2009.01411). Modified for
    ProtBERT feature extractor.

    Returned graphs are of type `torch_geometric.data.Data` with attributes
    -x          alpha carbon coordinates, shape [n_nodes, 3]
    -name       name of the protein structure, string
    -node_s     node scalar features, shape [n_nodes, 6]
    -node_v     node vector features, shape [n_nodes, 3, 3]
    -edge_s     edge scalar features, shape [n_edges, 32]
    -edge_v     edge scalar features, shape [n_edges, 1, 3]
    -edge_index edge indices, shape [2, n_edges]
    -mask       node mask, `False` for nodes with missing data that are
        excluded from message passing
    -input_ids  amino acid id encoding from ProtBERT tokenizer
    -attention_mask attention_mask from ProtBERT tokenizer

    Modified from https://github.com/drorlab/gvp-pytorch/blob/main/gvp/data.py.
    """

    def __init__(self, data_list, **kwargs):
        """
        Initializes the dataset

        Args:
            data_list: List containint the initial data

        Returns:
            None
        """
        super(ProteinGraphDataset, self).__init__(data_list, **kwargs)

    def _featurize_as_graph(self, protein, index=None, q=None):
        """Featurizes the protein information as a graph for the GNN

        Args:
            protein: Dictionary with the protein seq, coord and name.

        Returns:
            Torch geometric data instance representing with the protein information
        """
        name = protein["name"]
        input_ids = protein["input_ids"]
        attention_mask = protein["attention_mask"]
        #adding sequence context
       # input_ids = torch.cat((protein["related_sequences"]["input_ids"].flatten(),  input_ids))
       # attention_mask = torch.cat((protein["related_sequences"]["attention_mask"].flatten(),  attention_mask))
        #for protein_sequence in protein["related_sequences"]:


        with torch.no_grad():
            coords = torch.as_tensor(
                protein["coords"], device=self.device, dtype=torch.float32
            )

            mask = torch.isfinite(coords.sum(dim=(1, 2)))
            coords[~mask] = np.inf

            X_ca = coords[:, 1]
       # edge_index = torch_cluster.knn_graph(X_ca, k=self.top_k)
      #  xs = []
      #  p_dist = torch.nn.PairwiseDistance(p=2)
           # index1=-1
            dist = torch.cdist(X_ca, X_ca, p=2)
            edge_index = (dist < 10).nonzero().t()
      #  [[xs.append(torch.as_tensor([index1, index2], device=self.device)) for index2, x2 in enumerate(X_ca) if index1!=index2 and p_dist(x1,x2) < 10]  for index1, x1 in enumerate(X_ca)]
         #   for x1 in X_ca:
         #       index1 += 1
         #       index2 = -1
         #       for x2 in X_ca:
         #           index2 += 1
         #           if index1 == index2:
         #               continue
         #           elif torch.sqrt(torch.sum((x1 - x2) * 2)) < 10:
         #               xs.append(torch.as_tensor([index1, index2], device=self.device))
       # if len(xs) == 0:
       #     raise RuntimeError("no edges")
       # edge_index = torch.stack(xs,dim=0).t()
            pos_embeddings = self._positional_embeddings(edge_index)
            E_vectors = X_ca[edge_index[0]] - X_ca[edge_index[1]]
            rbf = _rbf(
                E_vectors.norm(dim=-1),
                D_count=self.num_rbf,
                device=self.device,
            )

            dihedrals = self._dihedrals(coords)
            orientations = self._orientations(X_ca)
            sidechains = self._sidechains(coords)

            node_s = dihedrals
            node_v = torch.cat(
                [orientations, sidechains.unsqueeze(-2)], dim=-2
            )
            edge_s = torch.cat([rbf, pos_embeddings], dim=-1)
            edge_v = _normalize(E_vectors).unsqueeze(-2)

            node_s, node_v, edge_s, edge_v = map(
                torch.nan_to_num, (node_s, node_v, edge_s, edge_v)
            )
 #           try:
 #               protein["go_labels"]
 #           except:
 #               print(name)
        data = torch_geometric.data.Data(
            x=X_ca,
            input_ids=input_ids,
            attention_mask=attention_mask,
            name=name,
            node_s=node_s,
            node_v=node_v,
            edge_s=edge_s,
            edge_v=edge_v,
            edge_index=edge_index,
            mask=mask,
#            go_terms=torch.as_tensor(protein["go_labels"], dtype=torch.float32, device=self.device)
        )
        return data


class BertProteinGraphDatasetWithTarget(ProteinGraphDataset):
    """Thin wrapper for ProteinGraphDataset to include targets.
    Intended for all BERT+GNN models"""

    def __init__(self, data_list, **kwargs):
        """Initializes the dataset

        Args:
            data_list: a list of data records (dicts with `input_ids`, `attention_mask`, `target`)

        Return:
            None
        """
        super(BertProteinGraphDatasetWithTarget, self).__init__(
            data_list, **kwargs
        )

    def __getitem__(self, idx):
        """Retrieve protein information by index.

        Args:
            idx: Integer representing the position of the protein.

        Return:
            None
        """
        if not isinstance(self.data_list[idx], tuple):
            #manager = multiprocessing.Manager()
           # q = manager.Queue()
           # thread = torch.multiprocessing.Process(target=self._featurize_as_graph, args=[self.data_list[idx], q])
           # thread.start()
           # thread.join()
            self.data_list[idx] = (
                self._featurize_as_graph(self.data_list[idx]),
           #     q.get(),
                self.data_list[idx]["target"]
            )
        return self.data_list[idx]

    def _preprocess(self):
        """Preprocess all the records in `data_list` with
        `_featurize_as_graph`. Directly modifies self.data_list

        Args:

        Returns:
            None
        """
        #for i in tqdm.tqdm(range(len(self.data_list))):
        manager = multiprocessing.Manager()
        q = manager.Queue()
        #    self.data_list[i] = (
       # my_pool= multiprocessing.pool.ThreadPool(30)
        #torch.multiprocessing.Man
        threads = [torch.multiprocessing.Process(target= self._featurize_as_graph, args=[self.data_list[i], i]) for i in tqdm.tqdm(range(len(self.data_list)))]
       # started = [ thread.start() for thread in tqdm.tqdm(threads) if count<=30:]
       # data_graphs =my_pool.map_async(self._featurize_as_graph, self.data_list)
       # for index, graph in  tqdm.tqdm(enumerate(data_graphs.get())):
       #     self.data_list[index] = (graph,
       #     self.data_list[index]["target"])
     #   for i in tqdm.tqdm(range(len(self.data_list))):
     #       self.data_list[i] = (
     #           data_graphs[i] ,
     #           self.data_list[i]["target"]
     #       )
        started_threads = []
        last_index = 0
        for index, thread in tqdm.tqdm(enumerate(threads)):
            if len(started_threads )<=os.cpu_count() :
                thread.start()
                started_threads.append(thread)
            else:
                [threads.join() for threads in started_threads]
                for i in range(last_index, index):
                    self.data_list[i] = (
                        torch.load(str(i) + ".tmp"),
                        self.data_list[i]["target"]
                    )
                    os.system("rm  " + str(i)+".tmp")
                last_index = index
                started_threads = []
      #  for i in tqdm.tqdm(range(len(self.data_list))):
      #      self.data_list[i] = (
      #                 torch.load(str(i)+".tmp"),
      #                 self.data_list[i]["target"]
      #             )
      #  os.system("rm -r *.tmp")
      #  [thread.join() for thread in tqdm.tqdm(threads) ]
        #             self._featurize_as_graph(self.data_list[i]),
         #       self.data_list[i]["target"],
         #   )
