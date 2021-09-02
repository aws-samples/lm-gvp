# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""Modified from https://github.com/flatironinstitute/DeepFRI/blob/fa9409cca7dc7b475f71ab4bab0aa7b6b1091448/deepfrier/utils.py"""
import csv
import numpy as np

from sklearn import metrics
from joblib import Parallel, delayed

from Bio import SeqIO
from Bio.PDB.PDBParser import PDBParser


def load_FASTA(filename):
    """
    Loads a FASTA file and returns the protein ids and their sequences.

    Args:
        filename: String representing the path to the FASTA file.
    Returns
        Tuple where the first elemnent is a list of protein ids and the second element is a list of protein sequences.
    """
    # Loads fasta file and returns a list of the Bio SeqIO records
    infile = open(filename, "rU")
    entries = []
    proteins = []
    for entry in SeqIO.parse(infile, "fasta"):
        entries.append(str(entry.seq))
        proteins.append(str(entry.id))
    return proteins, entries


def load_GO_annot(filename):
    """
    Loads the GO annotations.

    Args:
        filename: String representing the path to the GO annotations file.
    Returns
        Quatruple where elements are
            1/ a dict of dict with protein annotations: {protein: {'cc': np.array([...])}}
            2/ a dict with metadata of GO terms: {'cc': [goterm1, ...]}
            3/ a dict with metadata of GO names: {'cc': [goname1, ...]}
            4/ a dict with protein counts of GO terms: {'cc': np.array(...)}
    """
    # Load GO annotations
    onts = ["mf", "bp", "cc"]
    prot2annot = {}
    goterms = {ont: [] for ont in onts}
    gonames = {ont: [] for ont in onts}
    with open(filename, mode="r") as tsvfile:
        reader = csv.reader(tsvfile, delimiter="\t")

        # molecular function
        next(reader, None)  # skip the headers
        goterms[onts[0]] = next(reader)
        next(reader, None)  # skip the headers
        gonames[onts[0]] = next(reader)

        # biological process
        next(reader, None)  # skip the headers
        goterms[onts[1]] = next(reader)
        next(reader, None)  # skip the headers
        gonames[onts[1]] = next(reader)

        # cellular component
        next(reader, None)  # skip the headers
        goterms[onts[2]] = next(reader)
        next(reader, None)  # skip the headers
        gonames[onts[2]] = next(reader)

        next(reader, None)  # skip the headers
        counts = {
            ont: np.zeros(len(goterms[ont]), dtype=float) for ont in onts
        }
        for row in reader:
            prot, prot_goterms = row[0], row[1:]
            prot2annot[prot] = {ont: [] for ont in onts}
            for i in range(3):
                goterm_indices = [
                    goterms[onts[i]].index(goterm)
                    for goterm in prot_goterms[i].split(",")
                    if goterm != ""
                ]
                prot2annot[prot][onts[i]] = np.zeros(len(goterms[onts[i]]))
                prot2annot[prot][onts[i]][goterm_indices] = 1.0
                counts[onts[i]][goterm_indices] += 1.0
    return prot2annot, goterms, gonames, counts


def norm_adj(A, symm=True):
    """
    Normalize adj matrix

    Args:
        A: numpy array representing the adjacency matrix to be normalized.
        symm: Boolean representing if the adjacency matrix is symmetric (i.e. undirected graph)
    Returns
        Numpy array representing the normalized adjacency matrix.
    """
    A += np.eye(A.shape[1])
    if symm:
        d = 1.0 / np.sqrt(A.sum(axis=1))
        D = np.diag(d)
        A = D.dot(A.dot(D))
    else:
        A /= A.sum(axis=1)[:, np.newaxis]
    return A


def _micro_aupr(y_true, y_test):
    """
    Computes the micro AUPR

    Args:
        y_true: array with the GT observations.
        y_test: array with the predictions.
    Returns
        float representing the micro aupr score
    """
    return metrics.average_precision_score(y_true, y_test, average="micro")


def compute_f1_score_at_threshold(
    y_true: np.ndarray, y_pred: np.ndarray, t: float
):
    """Calculate protein-centric F1 score based on DeepFRI's description.
    ref: https://www.nature.com/articles/nmeth.2340
    Online method -> Evaluation metrics

    Args:
        y_true: [n_proteins, n_functions], binary matrix of ground truth labels
        y_pred: [n_proteins, n_functions], probabilities from model predictions after sigmoid.
        t: Float representing the threshold to use to compute the f1 score.

    Returns:
        float representing the f1 score
    """
    n_proteins = y_true.shape[0]
    y_pred_bin = y_pred >= t  # binarize predictions
    pr = []
    rc = []
    for i in range(n_proteins):
        if y_pred_bin[i].sum() > 0:
            pr_i = metrics.precision_score(y_true[i], y_pred_bin[i])
            pr.append(pr_i)

        rc_i = metrics.recall_score(y_true[i], y_pred_bin[i])
        rc.append(rc_i)

    pr = np.mean(pr)
    rc = np.mean(rc)
    return 2 * pr * rc / (pr + rc)


def evaluate_multilabel(
    y_true: np.ndarray, y_pred: np.ndarray, n_thresholds=100
):
    """Calculate protein-centric F_max and function-centric AUPR
    based on DeepFRI's description.
    ref: https://www.nature.com/articles/nmeth.2340
    Online method -> Evaluation metrics
    Args:
        y_true: [n_proteins, n_functions], binary matrix of ground truth labels
        y_pred: [n_proteins, n_functions], logits from model predictions
        n_thresholds (int): number of thresholds to estimate F_max

    Returns:
        Tuple where the first element is the F1 score and the second element is the micro AUPR
    """
    # function-centric AUPR
    micro_aupr = _micro_aupr(y_true, y_pred)

    # apply sigmoid to logits
    y_pred = 1 / (1 + np.exp(-y_pred))

    thresholds = np.linspace(0.0, 1.0, n_thresholds, endpoint=False)
    f_scores = Parallel(n_jobs=-1, verbose=10)(
        delayed(compute_f1_score_at_threshold)(y_true, y_pred, thresholds[i])
        for i in range(n_thresholds)
    )

    return np.nanmax(f_scores), micro_aupr
