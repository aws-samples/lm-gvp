# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""
Prase structure files downloaded from RCSB for the GO dataset from the
DeepFRI paper.
GO datasets was downloaded from:
https://github.com/flatironinstitute/DeepFRI/tree/master/preprocessing/data
"""
import threading

import json
import os
from Bio import SeqIO
from functools import reduce
import pandas as pd
import numpy as np

from tqdm import tqdm
from joblib import Parallel, delayed
from Bio.PDB import PDBParser, MMCIFParser
from Bio.PDB.Polypeptide import is_aa
import dask.dataframe as df

from data.contact_map_utils import (
    parse_structure,
    three_to_one_standard,
)
import xpdb
from lib.const import ALPHA_FOLD_STRUCTURE_EXT, ALPHA_FOLD_PAE_EXT
from query import AlphaFoldQuery

DATA_DIR = "/home/ec2-user/SageMaker/efs/paper_data/DeepFRI_GO_PDB"
OUTPUT_DIR = "/home/ec2-user/SageMaker/efs/gvp-datasets/DeepFRI_GO"


def get_structure_alignments(sequence_list: str, working_directory):
    complete_dataframe: df.DataFrame = df.read_csv("~/.config/structcompare/data/accession_ids.csv", header=None)
    # uniprot_id_strs = [uniprot.id for uniprot in self._uniprot_id_query_list]
    queried_dataframe: df.DataFrame = complete_dataframe[complete_dataframe[0].apply(lambda x: x in sequence_list)]
    queried_dataframe = queried_dataframe.compute()
    threads = [(threading.Thread(target=AlphaFoldQuery(working_directory.joinpath(accession)).query,
                                 args=[alpha_fold_model_name + ALPHA_FOLD_STRUCTURE_EXT % version + ".pdb"])
                , threading.Thread(target=AlphaFoldQuery(working_directory.joinpath(accession)).query,
                                   args=[alpha_fold_model_name + ALPHA_FOLD_PAE_EXT % version]))
               for accession, alpha_fold_model_name, version in
               zip(queried_dataframe[0], queried_dataframe[3], queried_dataframe[4])]
    [(thread[0].start(), thread[1].start()) for thread in threads]
    [(thread[0].join(), thread[1].join()) for thread in threads]
    sequences = []
    for accession in sequence_list:
        pdb_parser = SeqIO.parse(working_directory.joinpath(accession), "pdb-atom")
        sequence = ""
        for record in pdb_parser:
            sequence += record.seq
        sequences.append(sequence)
    return sequences

    #SeqIO.(working_directory.joinpath(query_structure_entry),"pdb")

def get_atom_coords(residue, target_atoms=["N", "CA", "C", "O"]):
    """Extract the coordinates of the target_atoms from an AA residue.
    Handles exception where residue doesn't contain certain atoms
    by setting coordinates to np.nan

    Args:
        residue: a Bio.PDB.Residue object.
        target_atoms: Target atoms which residues will be resturned.

    Returns:
        np arrays with target atoms 3D coordinates in the order of target atoms.
    """
    atom_coords = []
    for atom in target_atoms:
        try:
            coord = residue[atom].coord
        except KeyError:
            coord = [np.nan] * 3
        atom_coords.append(coord)
    return np.asarray(atom_coords)


def chain_to_coords(chain, target_atoms=["N", "CA", "C", "O"], name=""):
    """Convert a PDB chain in to coordinates of target atoms from all
    AAs

    Args:
        chain: a Bio.PDB.Chain object
        target_atoms: Target atoms which residues will be resturned.
        name: String. Name of the protein.
    Returns:
        Dictonary containing protein sequence `seq`, 3D coordinates `coord` and name `name`.

    """
    output = {}
    # get AA sequence in the pdb structure
    pdb_seq = "".join(
        [
            three_to_one_standard(res.get_resname())
            for res in chain.get_residues()
            if is_aa(res)
        ]
    )
    if len(pdb_seq) <= 1:
        # has no or only 1 AA in the chain
        return None
    output["seq"] = pdb_seq
    # get the atom coords
    coords = np.asarray(
        [
            get_atom_coords(res, target_atoms=target_atoms)
            for res in chain.get_residues()
            if is_aa(res)
        ]
    )
    output["coords"] = coords.tolist()
    output["name"] = "{}-{}".format(name, chain.id)
    return output


def parse_structure_file_to_json_record(
    pdb_parser, cif_parser, sequence, pdb_file_path, name=""
):
    """Parse a protein structure file (.pdb or .cif) to extract all the chains
    to json records for LM-GVP model.

    Args:
        pdb_parser: a Bio.PDB.PDBParser instance to parse the PDB files.
        cif_parser: a Bio.PDB.MMCIFParser instance to parse the CIF files.
        sequence: String representing the protein sequence
        pdb_file_path: String. Path to the PDB file.
        name: String. Name of the protein.

    Returns:
        List of parsed protein chain records (Dictonary containing protein sequence `seq`, 3D coordinates `coord` and name `name`.)
    """

    try:
        struct = parse_structure(
            pdb_parser, cif_parser, sequence, pdb_file_path
        )
    except Exception as e:
        print(pdb_file_path, "raised an error:")
        print(e)
        return []
    else:
        records = []
        chain_ids = set()
        for chain in struct.get_chains():
            if chain.id in chain_ids:  # skip duplicated chains
                continue
            chain_ids.add(chain.id)
            record = chain_to_coords(chain, name=name)
            if record is not None:
                records.append(record)
        return records


if __name__ == "__main__":

    # 0. Prepare structure parser
    # PDB parser
    pdb_parser = PDBParser(
        QUIET=True,
        PERMISSIVE=True,
        structure_builder=xpdb.SloppyStructureBuilder(),
    )

    # CIF parser
    cif_parser = MMCIFParser(
        QUIET=True,
        structure_builder=xpdb.SloppyStructureBuilder(),
    )

    # 1. Load metadata
    df = []
    for split in ["train", "valid", "test"]:
        split_df = pd.read_csv(
            os.path.join(DATA_DIR, "data", f"nrPDB-GO_2019.06.18_{split}.txt"),
            sep="\t",
            header=None,
        )
        split_df["split"] = split
        print(split, split_df.shape)
        df.append(split_df)

    df = pd.concat(df)
    df = df.set_index(df.columns[0], verify_integrity=True)
    print(df.shape)

    # 2. Parse the structure files and save to json files
    for split in ["train", "valid", "test"]:
        structure_file_dir = os.path.join(
            DATA_DIR, f"cif-nrPDB-GO_2019.06.18_{split}"
        )
        files = os.listdir(structure_file_dir)

        records = Parallel(n_jobs=-1)(
            delayed(parse_structure_file_to_json_record)(
                pdb_parser,
                cif_parser,
                files[i].split(".")[0],
                os.path.join(structure_file_dir, files[i]),
                files[i].split(".")[0],
            )
            for i in tqdm(range(len(files)))
        )

        # concat inner lists
        records = reduce(lambda x, y: x + y, records)
        print(split, len(records))

        # keep chains in df only
        chains_for_split = set(df.loc[df["split"] == split].index)
        records = [rec for rec in records if rec["name"] in chains_for_split]

        # check if there is any chains missing
        missed_chains = chains_for_split - set(
            [rec["name"] for rec in records]
        )
        if len(missed_chains) > 0:
            print("Missing chains:", len(missed_chains))

        # write to json file
        json.dump(
            records,
            open(os.path.join(OUTPUT_DIR, f"proteins_{split}.json"), "w"),
        )
