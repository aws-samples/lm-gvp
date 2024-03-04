# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""
Parse PDB files to extract the coordinates of the 4 key atoms from AAs to
generate json records compatible to the LM-GVP model.

This script is intended for Fluorescence and Protease datasets from TAPE.
"""

import threading

import json
import os
import argparse
from collections import defaultdict

from Bio import SeqIO
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed
from Bio.PDB import PDBParser
from Bio.PDB.Polypeptide import three_to_one
import xpdb
from data.contact_map_utils import parse_pdb_structure
import warnings

warnings.filterwarnings('ignore')
from lib.const import ALPHA_FOLD_STRUCTURE_EXT, ALPHA_FOLD_PAE_EXT
from query import AlphaFoldQuery


# pyr.init()
# scorefxn = pyr.get_fa_scorefxn()

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


def get_structure_alignments(sequence_list: str, working_directory, dataframe):
    # uniprot_id_strs = [uniprot.id for uniprot in self._uniprot_id_query_list]
    queried_dataframe: df.DataFrame = dataframe[dataframe[0].apply(lambda x: x in sequence_list)]
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
    try:
        pdb_seq = "".join(
            [three_to_one(res.get_resname()) for res in struct.get_residues()]
        )
    except:
        raise RuntimeError(f"could grab pdb for {struct}")
    output["seq"] = pdb_seq
    # print(pdb_seq)
    if len(pdb_seq) > 1500:
        return
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


def parse_pdb_gz_to_json_record(parser, sequence, pdb_file_path, name="", working_dir=""):
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
    #  print(sequence)
    #  print(Path(pdb_file_path).name)
    # if Path(pdb_file_path).name == "6RWY.pdb":
    #    print("im right here BIIIITIHCHSDHFIUH")
    #    print(sequence)
    # if len(sequence) >=700:
    #    print("returning nothing")
    #    return
    struct = parse_pdb_structure(parser, sequence, pdb_file_path)
    try:
        record = structure_to_coords(struct, name=pdb_file_path)
        # while(dataframe[index] )
    #  vectorized_structurally_similar_sequences = dataframe[dataframe[0] == str(Path(pdb_file_path).name)].head()[[1,2,6]].to_numpy()
    #    print(vectorized_structurally_similar_sequences)
    #   if len(vectorized_structurally_similar_sequences) > 0:
    #       record["related_structures"] = vectorized_structurally_similar_sequences

    # get_structure_alignments(pdb_file_path.name, working_dir, dataframe)
    except KeyError as ex:
        raise KeyError(f"Need to fix {name} as {ex}")
    except TypeError as ex:
        print(f"Returned nontype from {ex}")
    except RuntimeError as ex:
        print(f"Need to fix {pdb_file_path} as {ex}")
        return
    # raise RuntimeError(f"Need to fix {pdb_file_path} as {ex}")
    #  pyr.init()
    #  scorefxn = pyr.get_fa_scorefxn()
    #  pose = pyr.pose_from_pdb(pdb_file_path)
    #  scorefxn(pose)
    #  res_ene = pose.energies().residue_total_energies_array()
    #  print(res_ene.shape)
    #  try:
    #      record["energies"] = res_ene
    #  except TypeError:
    #      print(f"Fucked up {pdb_file_path}")
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
        structure_builder=xpdb.SloppyStructureBuilder(),
    )
    complete_dataframe = pd.read_csv("~/Research/results.m8", sep="\t", header=None)
    #complete_dataframe = complete_dataframe.to_numpy()
    # complete_dataframe: df.DataFrame = df.read_csv("~/.config/structcompare/data/accession_ids.csv", header=None)
    # 2. Parallel parsing structures and converting to protein records
    records = Parallel(n_jobs=-1)(
        delayed(parse_pdb_gz_to_json_record)(
            sloppyparser,
            df.iloc[i]["primary"],
            df.iloc[i]["structure_path"],
            df.iloc[i]["structure_path"].split("/")[-1],
            "/media/felix/Research/",
        )
        for i in tqdm(range(df.shape[0]))
    )

    # 3. Segregate records by splits
    splitted_records = defaultdict(list)
    current_pdb = ""
    count = 0
    records_temp = []
    checked = False
   # for rec in records:
   #     if rec is None:
   #         continue
   #     items = complete_dataframe[complete_dataframe[0] == Path(rec["name"]).name].head()[[1,2,6]]
   #     rec["related_sequences"] = items.to_numpy()

   # for item in complete_dataframe:
   #     if item[0] != current_pdb:
   #         current_pdb = item[0]
   #         records_temp = [item[[1, 2, 6]]]
   #         count = 1
   #         checked=False
   #     elif count < 5:
   #         records_temp.append(item[[1, 2, 6]])
   #         count += 1
   #     if count == 5 and not checked:
   #         found = False
   #         for rec in records:
   #             if rec is None:
   #                 continue
   #             if Path(rec["name"]).name.split(".pdb")[0] == current_pdb.split(".pdb")[0]:
   #                 rec["related_seqs"] = records_temp
   #                 found = True
   #                 break
   #         if not found:
   #              logging.warning(f"couldnt find my boi {current_pdb}")
   #         checked = True

  #  for rec in records:
      #  df[]

    for i, rec in enumerate(records):
        if rec is None:
            continue

        row = df.iloc[i]
        target = row[args.target_variable]
        split = row["split"]
        rec["target"] = target
        splitted_records[split].append(rec)

        items = complete_dataframe[complete_dataframe[0] == Path(rec["name"]).name].head()[[1,2,9]]
        rec["related_sequences"] = items.to_numpy().tolist()
    # 4. write to disk
    for split, records in splitted_records.items():
        print(split, "number of proteins:", len(records))
        outfile = os.path.join(args.output, f"proteins_{split}.json")
        json.dump(records, open(outfile, "w"))

    return None


if __name__ == "__main__":
    main()
