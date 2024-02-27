# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""
Train seq-only, struct-only or seq+struct model on Fluores, protease or GO
datasets using Pytorch-lightning.
"""
import os
from pprint import pprint
from torch.utils.data import WeightedRandomSampler
import argparse

from collections.abc import Sequence
import numpy as np
from pytorch_lightning.strategies import DDPStrategy
from sklearn import metrics
from scipy import stats
from finetuning_scheduler import FinetuningScheduler
import lightning.pytorch as L
import torch
import torch_geometric
import lightning.pytorch as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint

from lmgvp.modules import (
    BertGATModel,
    BertMQAModel,
    BertFinetuneModel
)
from lmgvp import deepfrier_utils, data_loaders
from lmgvp.transfer import load_state_dict_to_model

torch.backends.cudnn.allow_t32 = True

# to determine model type based on model name
MODEL_TYPES = {
    "bert_gat": "seq_struct",
}

# mapping model names to constructors
MODEL_CONSTRUCTORS = {
    "bert_gat": BertGATModel,
}


def init_model(
        datum=None,
        model_name="bert",
        num_outputs=32,
        classify=True,
        weights=None,
        **kwargs
):
    """Initialize a model.

    Args:
        datum: a Data object to determine input shapes for GVP-based models.
        model_name: choose from ['bert', 'gvp', 'bert_gvp', 'gat', 'bert_gat']
        num_outputs: number of output units
        weights: label weights for multi-output models

    Returns:
        model object (One of: bert, gat, bert_gat, gvp or bert_gvp)

    """
    print("Init {} model with args:".format("BertGAT"))
    pprint(kwargs)
    if model_name in "bert":
   #     model = BertGATModel(
   #         num_outputs=32,
   #         weights=weights,
   #         classify=classify,
   #         **kwargs
   #     )
    #elif model_name in "bert":

        model = BertFinetuneModel(
            num_outputs=num_outputs,
            weights=weights,
            classify=classify,
            **kwargs
        )
    elif model_name in ("gvp", "bert_gvp"):
        node_in_dim = (datum.node_s.shape[1], datum.node_v.shape[1])
        node_h_dim = (kwargs["node_h_dim_s"], kwargs["node_h_dim_v"])
        edge_in_dim = (datum.edge_s.shape[1], datum.edge_v.shape[1])
        edge_h_dim = (kwargs["edge_h_dim_s"], kwargs["edge_h_dim_v"])
        print("node_h_dim:", node_h_dim)
        print("edge_h_dim:", edge_h_dim)
        model = BertMQAModel(
            node_in_dim=node_in_dim,
            node_h_dim=node_h_dim,
            edge_in_dim=edge_in_dim,
            edge_h_dim=edge_h_dim,
            num_layers=3,
            drop_rate=0.1,
            weights=weights,
            num_outputs=num_outputs,
            classify=classify,
            **kwargs
        )
    return model


def main(args):
    """
    Load data, train and evaluate model and save scores. Configuration in the args object.

    Args:
        args: Parsed command line arguments. Must include: pytorchlighting pre-defined args, task, node_h_dim_s, node_h_dim_v, edge_h_dim_s, edge_h_dim_v, pretrained_weights, ls, bs, early_stopping_patience, num_workers.

    Returns:
        None
    """
    pl.seed_everything(42, workers=True)
    # 1. Load data
    train_dataset = data_loaders.get_dataset(split="train"
                                             )
    valid_dataset = data_loaders.get_dataset(split="valid")
    print("Data loaded:", len(train_dataset), len(valid_dataset))
    # 2. Prepare data loaders
    DataLoader = torch_geometric.loader.DataLoader
    sampler1 = WeightedRandomSampler(train_dataset.sample_weights, len(train_dataset), replacement=True,)
    # sampler2 = WeightedRandomSampler(valid_dataset.sample_weights, len(valid_dataset))

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.bs,
        sampler=sampler1,
        #  shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=args.bs,
        #    sampler=sampler2,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    # 3. Prepare model
    # datum = None
    # getting the dims from dataset
    datum = train_dataset[0][0]
    dict_args = vars(args)
    model = init_model(
        datum=datum,
        num_outputs=train_dataset.num_outputs,
        weights=train_dataset.pos_weights,
        classify=True,
        **dict_args
    )
    if args.pretrained_weights:
        # load pretrained weights
        checkpoint = torch.load(
            args.pretrained_weights, map_location=torch.device("cpu")
        )
        load_state_dict_to_model(model, checkpoint["state_dict"])
    # 4. Training
    # callbacks
    early_stop_callback = EarlyStopping(
        monitor="val_loss", patience=args.early_stopping_patience
    )
    # Init ModelCheckpoint callback, monitoring 'val_loss'
    checkpoint_callback = ModelCheckpoint(monitor="val_loss")
    fine_tune =  FinetuningScheduler()
    # init pl.Trainer
    #   pl.Trainer()
    #  pl.Trainer(benchmark=True)
    trainer = L.Trainer(
        # accelerator='dp',
        # plugins='ddp_sharded',
        #    args,
        # deterministic=True,

        #  accelerator='gpu',
        #  strategy="auto",
        devices=1,
    #    accumulate_grad_batches=8,
     #   enable_checkpointing=True,
        strategy='ddp_find_unused_parameters_true',
        callbacks= [fine_tune] #[early_stop_callback,checkpoint_callback]
     #   profiler="simple",
     #   benchmark=True,
        # precision=""

    )
    # train
   # model = torch.compile(model, mode="reduce-overhead")
    # model(torch.randn(32,3,32,32))
    # model.model = torch.compile(model.model)
    trainer.fit(model, train_loader, valid_loader)
    print("Training finished")
    print(
        "checkpoint_callback.best_model_path:",
        checkpoint_callback.best_model_path,
    )
    # 5. Evaluation
    # load the best model
    model = model.load_from_checkpoint(
        checkpoint_path=checkpoint_callback.best_model_path,
        weights=train_dataset.pos_weights,
    )
    print("Testing performance on test set")
    # load test data

    test_dataset = data_loaders.get_dataset(split="test")
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.bs,
        shuffle=False,
        num_workers=args.num_workers,
    )
    trainer.test(model, test_loader)
    # scores = evaluate(model, test_loader)
    # save scores to file
    # json.dump(
    #     scores,
    #     open(os.path.join(trainer.log_dir, "scores.json"), "w"),
    # )
    return None


if __name__ == "__main__":
    # os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:4096,garbage_collection_threshold:0.6"
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    # torch.multiprocessing.set_start_method('spawn', force=True)  # good solution !!!!
    parser = argparse.ArgumentParser()
    # add all the available trainer options to argparse
    #   parser = pl.Trainer.add_argparse_args(parser)
    # figure out which model to use
    parser.add_argument(
        "--model_name",
        type=str,
        default='bert',
        help="Choose from %s" % ", ".join(list(MODEL_TYPES.keys())),
    )

    # THIS LINE IS KEY TO PULL THE MODEL NAME
    #  temp_args, _ = parser.parse_known_args()
    # add model specific args
    #   model_name = temp_args.model_name
  #  parser = BertMQAModel.add_model_specific_args(parser)

    # Additional params
    # dataset params
    # parser.add_argument(
    #    "--task",
    #    help="Task to perform: ['flu', 'protease', 'cc', 'bp', 'mf']",
    #    type=str,
    #    required=True,
    # )
    # model hparams
    parser.add_argument(
        "--node_h_dim_s", type=int, default=100, help="node_h_dim[0] in GVP"
    )
    parser.add_argument(
        "--node_h_dim_v", type=int, default=16, help="node_h_dim[1] in GVP"
    )
    parser.add_argument(
        "--edge_h_dim_s", type=int, default=4, help="edge_h_dim[0] in GVP"
    )
    parser.add_argument(
        "--edge_h_dim_v", type=int, default=1, help="edge_h_dim[1] in GVP"
    )
    parser.add_argument(
        "--pretrained_weights",
        type=str,
        default=None,
        help="path to pretrained weights (such as GAE) to initialize model",
    )
    # training hparams
    parser.add_argument("--lr", type=float, default=1e-2, help="learning rate")
    parser.add_argument("--bs", type=int, default=1, help="batch size")
    parser.add_argument("--early_stopping_patience", type=int, default=3)
    parser.add_argument(
        "--num_workers",
        type=int,
        default=30,
        help="num_workers used in DataLoader",
    )

    args = parser.parse_args()

    print("args:", args)
    # train
    main(args)
