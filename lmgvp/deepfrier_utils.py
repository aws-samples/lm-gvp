# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""Modified from https://github.com/flatironinstitute/DeepFRI/blob/fa9409cca7dc7b475f71ab4bab0aa7b6b1091448/deepfrier/utils.py"""
import csv
import numpy as np

from sklearn import metrics
from joblib import Parallel, delayed

from Bio import SeqIO
from Bio.PDB.PDBParser import PDBParser


def load_predicted_PDB(pdbfile):
    # Generate (diagonalized) C_alpha distance matrix from a pdbfile
    parser = PDBParser()
    structure = parser.get_structure(
        pdbfile.split("/")[-1].split(".")[0], pdbfile
    )
    residues = [r for r in structure.get_residues()]

    # sequence from atom lines
    records = SeqIO.parse(pdbfile, "pdb-atom")
    seqs = [str(r.seq) for r in records]

    distances = np.empty((len(residues), len(residues)))
    for x in range(len(residues)):
        for y in range(len(residues)):
            one = residues[x]["CA"].get_coord()
            two = residues[y]["CA"].get_coord()
            distances[x, y] = np.linalg.norm(one - two)

    return distances, seqs[0]


def load_FASTA(filename):
    # Loads fasta file and returns a list of the Bio SeqIO records
    infile = open(filename, "rU")
    entries = []
    proteins = []
    for entry in SeqIO.parse(infile, "fasta"):
        entries.append(str(entry.seq))
        proteins.append(str(entry.id))
    return proteins, entries


def load_GO_annot(filename):
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


def load_EC_annot(filename):
    # Load EC annotations """
    prot2annot = {}
    with open(filename, mode="r") as tsvfile:
        reader = csv.reader(tsvfile, delimiter="\t")

        # molecular function
        next(reader, None)  # skip the headers
        ec_numbers = {"ec": next(reader)}
        next(reader, None)  # skip the headers
        counts = {"ec": np.zeros(len(ec_numbers["ec"]), dtype=float)}
        for row in reader:
            prot, prot_ec_numbers = row[0], row[1]
            ec_indices = [
                ec_numbers["ec"].index(ec_num)
                for ec_num in prot_ec_numbers.split(",")
            ]
            prot2annot[prot] = {
                "ec": np.zeros(len(ec_numbers["ec"]), dtype=np.int64)
            }
            prot2annot[prot]["ec"][ec_indices] = 1.0
            counts["ec"][ec_indices] += 1
    return prot2annot, ec_numbers, ec_numbers, counts


def norm_adj(A, symm=True):
    #  Normalize adj matrix
    A += np.eye(A.shape[1])
    if symm:
        d = 1.0 / np.sqrt(A.sum(axis=1))
        D = np.diag(d)
        A = D.dot(A.dot(D))
    else:
        A /= A.sum(axis=1)[:, np.newaxis]
    return A


def _micro_aupr(y_true, y_test):
    return metrics.average_precision_score(y_true, y_test, average="micro")


def compute_f1_score_at_threshold(
    y_true: np.ndarray, y_pred: np.ndarray, t: float
):
    """Calculate protein-centric F1 score based on DeepFRI's description.
    ref: https://www.nature.com/articles/nmeth.2340
    Online method -> Evaluation metrics
    Args:
        - y_true: [n_proteins, n_functions], binary matrix of ground truth
            labels
        - y_pred: [n_proteins, n_functions], probabilities from model
            predictions after sigmoid.
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
        - y_true: [n_proteins, n_functions], binary matrix of ground truth
            labels
        - y_pred: [n_proteins, n_functions], logits from model predictions
        - n_thresholds (int): number of thresholds to estimate F_max
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
