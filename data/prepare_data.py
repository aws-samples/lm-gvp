# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""
Parse PDB files to extract the coordinates of the 4 key atoms from AAs to
generate json records compatible to the LM-GVP model.

This script is intended for Fluorescence and Protease datasets from TAPE.
"""

import json
import os
import argparse
from collections import defaultdict
import pandas as pd
import numpy as np

from tqdm import tqdm
from joblib import Parallel, delayed
from Bio.PDB import PDBParser
from Bio.PDB.Polypeptide import three_to_one

from .xpdb import SloppyStructureBuilder
from .contact_map_utils import parse_pdb_structure


def parse_args():
    """Prepare argument parser.

    Args:

    Return:

    """
    parser = argparse.ArgumentParser(
        description="Generate GVP ProteinGraph datasets representing protein"
        + "structures."
    )
    parser.add_argument(
        "--data-file",
        help="Path to protein data frame, including sequences, paths to"
        + " structure files and labels",
        required=True,
    )
    parser.add_argument(
        "-t",
        "--target-variable",
        help="target variable in the protein data frame",
        required=True,
    )
    parser.add_argument("-o", "--output", help="output dir for graphs")

    args = parser.parse_args()
    return args


def get_atom_coords(residue, target_atoms=["N", "CA", "C", "O"]):
    """Extract the coordinates of the target_atoms from an AA residue.

    Args:
        residue: a Bio.PDB.Residue object representing the residue.
        target_atoms: Target atoms which residues will be returned.

    Retruns:
        Array of residue's target atoms (in the same order as target atoms).
    """
    return np.asarray([residue[atom].coord for atom in target_atoms])


def structure_to_coords(struct, target_atoms=["N", "CA", "C", "O"], name=""):
    """Convert a PDB structure in to coordinates of target atoms from all AAs

    Args:
        struct: a Bio.PDB.Structure object representing the protein structure
        target_atoms: Target atoms which residues will be returned.
        name: String. Name of the structure

    Return:
        Dictionary with the pdb sequence, atom 3D coordinates and name.
    """
    output = {}
    # get AA sequence in the pdb structure
    pdb_seq = "".join(
        [three_to_one(res.get_resname()) for res in struct.get_residues()]
    )
    output["seq"] = pdb_seq
    # get the atom coords
    coords = np.asarray(
        [
            get_atom_coords(res, target_atoms=target_atoms)
            for res in struct.get_residues()
        ]
    )
    output["coords"] = coords.tolist()
    output["name"] = name
    return output


def parse_pdb_gz_to_json_record(parser, sequence, pdb_file_path, name=""):
    """
    Reads and reformats a pdb strcuture into a dictionary.

    Args:
        parser: a Bio.PDB.PDBParser or Bio.PDB.MMCIFParser instance.
        sequence: String. Sequence of the structure.
        pdb_file_path: String. Path to the pdb file.
        name: String. Name of the protein.

    Return:
        Dictionary with the pdb sequence, atom 3D coordinates and name.
    """
    struct = parse_pdb_structure(parser, sequence, pdb_file_path)
    record = structure_to_coords(struct, name=name)
    return record


def main():
    """
    Data preparation main script: Load data, parses PDB, processes structures, segregate records and write to disk. Configuration via commandline arguments.

    Args:

    Return:

    """
    args = parse_args()
    # 1. Load data
    df = pd.read_csv(args.data_file)

    # PDB parser
    sloppyparser = PDBParser(
        QUIET=True,
        PERMISSIVE=True,
        structure_builder=SloppyStructureBuilder(),
    )

    # 2. Parallel parsing structures and converting to protein records
    records = Parallel(n_jobs=-1)(
        delayed(parse_pdb_gz_to_json_record)(
            sloppyparser,
            df.iloc[i]["primary"],
            df.iloc[i]["structure_path"],
            df.iloc[i]["structure_path"].split("/")[-1],
        )
        for i in tqdm(range(df.shape[0]))
    )

    # 3. Segregate records by splits
    splitted_records = defaultdict(list)
    for i, rec in enumerate(records):
        row = df.iloc[i]
        target = row[args.target_variable]
        split = row["split"]
        rec["target"] = target
        splitted_records[split].append(rec)

    # 4. write to disk
    for split, records in splitted_records.items():
        print(split, "number of proteins:", len(records))
        outfile = os.path.join(args.output, f"proteins_{split}.json")
        json.dump(records, open(outfile, "w"))

    return None


if __name__ == "__main__":
    main()
