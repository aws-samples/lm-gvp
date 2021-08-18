"""
Train seq-only, struct-only or seq+struct model on Fluores, protease or GO
datasets using Pytorch-lightning.
"""

import os
import json
from pprint import pprint
import argparse
from collections.abc import Sequence

import numpy as np
from sklearn import metrics
from scipy import stats

import torch
import torch_geometric

import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint

from lmgvp.modules import (
    BertFinetuneModel,
    MQAModel,
    BertMQAModel,
    GATModel,
    BertGATModel,
)
from lmgvp import deepfrier_utils, data_loaders
from lmgvp.transfer import load_state_dict_to_model

# to determine model type based on model name
MODEL_TYPES = {
    "gvp": "struct",
    "bert": "seq",
    "bert_gvp": "seq_struct",
    "gat": "struct",
    "bert_gat": "seq_struct",
}

# mapping model names to constructors
MODEL_CONSTRUCTORS = {
    "gvp": MQAModel,
    "bert": BertFinetuneModel,
    "bert_gvp": BertMQAModel,
    "gat": GATModel,
    "bert_gat": BertGATModel,
}

# to determine problem type based on task
IS_CLASSIFY = {
    "flu": False,
    "protease": False,
    "cc": True,
    "mf": True,
    "bp": True,
}


def init_model(
    datum=None,
    model_name="gvp",
    num_outputs=1,
    classify=False,
    weights=None,
    **kwargs
):
    """Initialize a model.
    Args:
        - datum: a Data object to determine input shapes for GVP-based models.
        - model_name: choose from ['bert', 'gvp', 'bert_gvp', 'gat',
            'bert_gat']
        - num_outputs: number of output units
        - weights: label weights for multi-output models
    """
    print("Init {} model with args:".format(model_name))
    pprint(kwargs)
    if model_name in ("bert", "gat", "bert_gat"):
        model = MODEL_CONSTRUCTORS[model_name](
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
        model = MODEL_CONSTRUCTORS[model_name](
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


def evaluate(model, data_loader, task):
    """Evaluate model on dataset and return metrics."""
    # make predictions on test set
    device = torch.device("cuda:0")
    model = model.to(device)
    model.eval()
    y_preds = []
    y_true = []
    with torch.no_grad():
        for batch in data_loader:
            if isinstance(batch, Sequence):
                y_true.append(batch[-1])
                batch = [b.to(device) for b in batch]
            else:
                y_true.append(batch["labels"])
                batch = {key: val.to(device) for key, val in batch.items()}
            y_pred = model(batch)
            if y_pred.ndim == 1:
                y_pred = y_pred.unsqueeze(1)
            y_preds.append(y_pred.cpu())
    y_preds = torch.vstack(y_preds).numpy()
    y_true = torch.vstack(y_true).numpy()
    print(y_preds.shape, y_true.shape)
    if task in ("cc", "bp", "mf"):
        # multi-label classification
        f_max, micro_aupr = deepfrier_utils.evaluate_multilabel(
            y_true.numpy(), y_preds.numpy()
        )
        scores = {"f_max": f_max, "aupr": micro_aupr}
        print("F_max = {:1.3f}".format(scores["f_max"]))
        print("AUPR = {:1.3f}".format(scores["aupr"]))
    else:
        # single task regression
        mse = metrics.mean_squared_error(y_true, y_preds)
        rmse = np.sqrt(mse)
        r2 = metrics.r2_score(y_true, y_preds)
        rho, _ = stats.spearmanr(y_true, y_preds)
        scores = {"mse": float(mse), "rmse": float(rmse), "r2": r2, "rho": rho}
        for key, score in scores.items():
            print("{} = {:1.3f}".format(key, score))
    return scores


def main(args):
    pl.seed_everything(42, workers=True)
    # 1. Load data
    train_dataset = data_loaders.get_dataset(
        args.task, MODEL_TYPES[args.model_name], split="train"
    )
    valid_dataset = data_loaders.get_dataset(
        args.task, MODEL_TYPES[args.model_name], split="valid"
    )
    print("Data loaded:", len(train_dataset), len(valid_dataset))
    # 2. Prepare data loaders
    if MODEL_TYPES[args.model_name] == "seq":
        DataLoader = torch.utils.data.DataLoader
    else:
        DataLoader = torch_geometric.data.DataLoader

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.bs,
        shuffle=True,
        num_workers=args.num_workers,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=args.bs,
        shuffle=False,
        num_workers=args.num_workers,
    )
    # 3. Prepare model
    datum = None
    if MODEL_TYPES[args.model_name] != "seq":
        # getting the dims from dataset
        datum = train_dataset[0][0]
    dict_args = vars(args)
    model = init_model(
        datum=datum,
        num_outputs=train_dataset.num_outputs,
        weights=train_dataset.pos_weights,
        classify=IS_CLASSIFY[args.task],
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
    # init pl.Trainer
    trainer = pl.Trainer.from_argparse_args(
        args,
        deterministic=True,
        callbacks=[early_stop_callback, checkpoint_callback],
    )
    # train
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
    test_dataset = data_loaders.get_dataset(
        args.task, MODEL_TYPES[args.model_name], split="test"
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.bs,
        shuffle=False,
        num_workers=args.num_workers,
    )
    scores = evaluate(model, test_loader, args.task)
    # save scores to file
    json.dump(
        scores,
        open(os.path.join(trainer.log_dir, "scores.json"), "w"),
    )
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # add all the available trainer options to argparse
    parser = pl.Trainer.add_argparse_args(parser)
    # figure out which model to use
    parser.add_argument(
        "--model_name",
        type=str,
        default="gvp",
        help="Choose from %s" % ", ".join(list(MODEL_TYPES.keys())),
    )

    # THIS LINE IS KEY TO PULL THE MODEL NAME
    temp_args, _ = parser.parse_known_args()
    # add model specific args
    model_name = temp_args.model_name
    parser = MODEL_CONSTRUCTORS[model_name].add_model_specific_args(parser)

    # Additional params
    # dataset params
    parser.add_argument(
        "--task",
        help="Task to perform: ['flu', 'protease', 'cc', 'bp', 'mf']",
        type=str,
        required=True,
    )
    # model hparams
    parser.add_argument(
        "--node_h_dim_s", type=int, default=100, help="node_h_dim[0] in GVP"
    )
    parser.add_argument(
        "--node_h_dim_v", type=int, default=16, help="node_h_dim[1] in GVP"
    )
    parser.add_argument(
        "--edge_h_dim_s", type=int, default=32, help="edge_h_dim[0] in GVP"
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
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
    parser.add_argument("--bs", type=int, default=32, help="batch size")
    parser.add_argument("--early_stopping_patience", type=int, default=5)
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="num_workers used in DataLoader",
    )

    args = parser.parse_args()

    print("args:", args)
    # train
    main(args)
