# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""
Util functions for loading datasets.
"""
import numpy
import os
import json
import numpy as np
import torch
from transformers import BertTokenizer, BertModel

from lmgvp.utils import prep_seq
from lmgvp.datasets import (
#    SequenceDatasetWithTarget,
#    ProteinGraphDatasetWithTarget,
    BertProteinGraphDatasetWithTarget,
)
from lmgvp.deepfrier_utils import load_GO_annot

DATA_ROOT_DIR = "/home/felix/"


def load_gvp_data(
    gvp_data_dir="{}/gvp-datasets".format(DATA_ROOT_DIR),
    split="train",
    seq_only=False,
):
    """For GVP models only.
    These prepared graph data files are generated by `generate_gvp_dataset.py`

    Args:
        task: choose from ['protease/with_tags', 'Fluorescence', 'DeepFRI_GO']
        seq_only: retain only the sequences in the returned list of objects
        split: String. Split of the dataset to be loaded. One of ['train', 'valid', 'test'].
        seq_only: Bool. Wheather or not to return only sequences without coordinates.

    Retrun:
        Dictionary containing the GVP dataset of proteins.
    """
    filename = os.path.join(gvp_data_dir, f"proteins_{split}.json")
    dataset = json.load(open(filename, "rb"))
    if seq_only:
        # delete the "coords" in data objects
        for obj in dataset:
            obj.pop("coords", None)
    return dataset


def preprocess_seqs(tokenizer, dataset):
    """Preprocess seq in dataset and bind the input_ids, attention_mask.

    Args:
        tokenizer: hugging face artifact. Tokenization to be used in the sequence.
        dataset: Dictionary containing the GVP dataset of proteins.

    Return:
        Input dataset with `input_ids` and `attention_mask`
    """
    seqs = [prep_seq(rec["seq"]) for rec in dataset]
    encodings = tokenizer(seqs, return_tensors="pt", padding=True, truncation=True, max_length=1600)
  #  bert = BertModel.from_pretrained("yarongef/DistilProtBert", torch_dtype="auto", ).to("cuda")

  #  return node_embeddings    # add input_ids, attention_mask to the json records
    for i, rec in enumerate(dataset):
        rec["input_ids"] = encodings["input_ids"][i]
        rec["attention_mask"] = encodings["attention_mask"][i]
     #   node_embeddings = bert(
     #       encodings["input_ids"][i].unsqueeze(-2), attention_mask=encodings["attention_mask"][i].unsqueeze(-2)
     #   ).last_hidden_state[:, 1:-1, :]
     #   attention_masks_1d = encodings["attention_mask"][i][:, 2:].reshape(-1)
        # remove embeddings from padding nodes
     #   node_embeddings = node_embeddings.reshape(-1, 1024)[
     #       attention_masks_1d == 1
     #       ]
     #   rec["node_embedding"] = node_embeddings
    return dataset

my_dict = {
    "antibiotic_resistance_repaired":            np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
    "avirulence_plant_repaired":                 np.array([0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
    "bacterial_counter_signaling_repaired":      np.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
    "counter_immunoglobin_repaired":             np.array([0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
    "cytotoxicity_repaired":                     np.array([0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
    "degrade_ecm_repaired":                      np.array([0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
    "development_in_host_repaired":              np.array([0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
    "disable_organ_repaired":                    np.array([0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
    "host_cell_cycle_repaired":                  np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
    "host_cell_death_repaired":                  np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
    "host_cytoskeleton_repaired":                np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
    "host_GTPase_repaired":                      np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
    "host_trancription_repaired":                np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
    "host_translation_repaired":                 np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
    "host_ubiquitin_repaired":                   np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
    "host_xenophagy_repaired":                   np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
    "induce_inflammation":                       np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
    "invasion_hostcell_Viral_repaired":          np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
    "nonviral_invasion_repaired":                np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
    "plant_RNA_silencing_viral_repaired":        np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
    "resist_host_complement_repaired":           np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
    "resist_oxidative_repaired":                 np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
    "secreted_effector_repaired":                np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]),
    "secretion_repaired":                        np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]),
    "suppress_detection_repaired":               np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]),
    "toxin_synthase_repaired":                   np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]),
    "viral_adhesion":                            np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]),
    "viral_counter_signaling_repaird":           np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]),
    "viral_movement_repaired":                   np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]),
    "virulence_activity_repaired":               np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]),
    "virulence_regulator_repaired":              np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]),

}
def load_FunSoc_labels( split, gvp_data_dir="{}/gvp-datasets".format(DATA_ROOT_DIR)):
    filename = os.path.join(gvp_data_dir, f"proteins_{split}.json")
    json_of_data = json.load(open(filename))
    prot2funsoc = {}
    funsocs_amount = len(my_dict.keys())
    classes = open("/media/felix/Research/Data/class_sizes.json","r")
    output = classes.readlines()[0]
    class_weight = numpy.zeros(funsocs_amount).astype("float64")
    classes = json.loads(output)
    for funsoc in classes:
         class_weight += my_dict[funsoc]*(classes[funsoc]/sum(classes.values()))
    sample_weights = np.zeros(len(json_of_data))
    for index, rec in enumerate(json_of_data):
        targets=numpy.zeros(funsocs_amount)
        count= 0
        for target in rec["target"].replace('[','').replace(']','').split(", "):
            count+=1
            if count > 2:
                print(rec["target"])
            targets =  targets + my_dict.get(target.replace("\'",''))
            sample_weights[index] = sample_weights[index] + np.sum(my_dict.get(target.replace("\'",''))*(1/classes[target.replace("\'",'')]))
        if 2 in targets:
            print(targets)
            raise RuntimeError("somehow a 2 got in my labels")
        prot2funsoc[rec["name"]] =targets
    print(sample_weights)
    return prot2funsoc, funsocs_amount, torch.from_numpy(class_weight.astype(np.float32)), torch.from_numpy(sample_weights.astype(np.float32))

def add_labels(dataset, prot2soc):
    """
    Add GO labels to a dataset

    Args:
        dataset: list of dict (output from `load_gvp_data`)
        prot2annot: output from `load_GO_labels`
        go_ont: String. GO ontology/task to be used. One of: 'cc', 'bp', 'mf'

    Return:
        Dataset formatted as a list. Where, for each element (dictionary), a `target` field has been added.

    """
    for rec in dataset:
        rec["target"] = torch.from_numpy(
            prot2soc[rec["name"]].astype(np.float32)
        )
    return dataset
def load_GO_labels(task="cc"):
    """Load the labels in the GO dataset

    Args:
        task: String. GO task. One of: 'cc', 'bp', 'mf'

    Return:
        Tuple where the first element is a dictionary mapping proteins to their target, second element is an integer with the number of outputs of the task and the third element is a matrix with the weight of each target.
    """
    prot2annot, goterms, gonames, counts = load_GO_annot(
        os.path.join(
            DATA_ROOT_DIR,
            "DeepFRI_GO_PDB/data/nrPDB-GO_2019.06.18_annot.tsv",
        )
    )
    goterms = goterms[task]
    gonames = gonames[task]
    num_outputs = len(goterms)
    # task =cc
    # computing weights for imbalanced go classes
    class_sizes = counts[task]
    mean_class_size = np.mean(class_sizes)
    pos_weights = mean_class_size / class_sizes
    pos_weights = np.maximum(1.0, np.minimum(10.0, pos_weights))
    # to tensor
    pos_weights = torch.from_numpy(pos_weights.astype(np.float16))
    return prot2annot, num_outputs, pos_weights


def add_GO_labels(dataset, prot2annot, go_ont="cc"):
    """
    Add GO labels to a dataset

    Args:
        dataset: list of dict (output from `load_gvp_data`)
        prot2annot: output from `load_GO_labels`
        go_ont: String. GO ontology/task to be used. One of: 'cc', 'bp', 'mf'

    Return:
        Dataset formatted as a list. Where, for each element (dictionary), a `target` field has been added.

    """
    for rec in dataset:
        rec["target"] = torch.from_numpy(
            prot2annot[rec["name"]][go_ont].astype(np.float16)
        )
    return dataset


def get_dataset(split="train"):
    """Load data from files, then transform into appropriate
    Dataset objects.
    Args:
        model_type: one of ['seq', 'struct', 'seq_struct']
        split: one of ['train', 'valid', 'test']

    Return:
        Torch dataset.
    """
#    seq_only = True if model_type == "seq" else False

    tokenizer = None
    #if model_type != "struct":
        # need to add BERT
    print("Loading BertTokenizer...")
    tokenizer = BertTokenizer.from_pretrained(
        "Rostlab/prot_bert", do_lower_case=False
    )

    # Load data from files
   # if task in ("cc", "bp", "mf"):  # GO dataset
        # load labels
    prot2funsoc, funsocs_amount,pos_weights, sample_weights  = load_FunSoc_labels(split)
    #prot2annot, num_outputs, pos_weights = load_GO_labels(task)
        # load features
    dataset = load_gvp_data(split=split, seq_only=False)
    dataset= add_labels(dataset, prot2funsoc)
    #add_GO_labels(dataset, prot2annot, go_ont=task)
    #else:
    #    data_dir = {"protease": "protease/with_tags", "flu": "Fluorescence"}
    #    dataset = load_gvp_data(
    #        task=data_dir[task], split=split, seq_only=seq_only
    #    )
    #    num_outputs = 1
    #    pos_weights = None

    # Convert data into Dataset objects
   # if model_type == "seq":
        #if num_outputs == 1:
        #    targets = torch.tensor(
        #        [obj["target"] for obj in dataset], dtype=torch.float32
        #    ).unsqueeze(-1)
        #else:
   #     targets = [obj["target"] for obj in dataset]
   #     dataset = SequenceDatasetWithTarget(
   #         [obj["seq"] for obj in dataset],
   #         targets,
   #         tokenizer=tokenizer,
   #         preprocess=True,
   #     )
   # else:
       # if num_outputs == 1:
            # convert target to f32 [1] tensor
       #     for obj in dataset:
       #         obj["target"] = torch.tensor(
       #             obj["target"], dtype=torch.float32
       #         ).unsqueeze(-1)
    #if model_type == "struct":
    #        dataset = ProteinGraphDatasetWithTarget(dataset, preprocess=False)
    #elif model_type == "seq_struct":
    dataset = preprocess_seqs(tokenizer, dataset)
    dataset = BertProteinGraphDatasetWithTarget(dataset, preprocess=False)

    dataset.num_outputs = funsocs_amount
    dataset.pos_weights = pos_weights
    dataset.sample_weights = sample_weights
    return dataset
