# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""
pl.LightningModules for LM-GVP and other baseline models, modified to support model interpretation
"""
import torchmetrics
from typing import Tuple
import torch
from gvp.models import GVP, GVPConvLayer, LayerNorm
from torch_scatter import scatter_mean
from transformers import BertModel, EsmModel
from torch_geometric.nn import GATv2Conv
import lightning.pytorch as pl
#torch.set_float32_matmul_precision('medium')
torch.backends.cuda.matmul.allow_tf32=True
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

def one_hot(index, classes):
    size = index.size() + (classes,)
    view = index.size() + (1,)

    mask = torch.Tensor(*size).fill_(0)
    mask = mask.to("cuda:0")
    index = index.view(*view)
    ones = 1.

    if isinstance(index, Variable):
        x = torch.Tensor(index.size()).fill_(1)
        ones = Variable(x.to("cuda:0"))
        mask = Variable(mask, volatile=index.volatile)
    index = index.long()
    return mask.scatter_(1, index, ones)


class FocalLoss(nn.modules.loss._WeightedLoss):
    def __init__(self, weight=None, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__(weight,reduction=reduction)
        self.gamma = gamma
        self.weight = weight #weight parameter will act as the alpha parameter to balance class weights

    def forward(self, input, target):

        ce_loss = F.cross_entropy(input, target,reduction=self.reduction,weight=self.weight)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma * ce_loss).mean()
        return focal_loss

#@torch.compile()
def _bert_forward(
    bert_model: BertModel,
    embeding_dim: int,
    input_ids=None,
    attention_masks=None,
):
    """Forward pass throught BERT to get token-level embeddings

    Args:
        bert_model: HuggingFace bert model
        embeding_dim: Dimension of the embeddings
        input_ids: indexes of the input sequence
        attention_masks: Mask to be used for Bert attention

    Returns:
        Output token-level embeddings
    """
    # skip [CLS] and [SEP]
    node_embeddings = bert_model(
            input_ids, attention_mask=attention_masks
        ).last_hidden_state[:, 1:-1, :]
    attention_masks_1d = attention_masks[:, 2:].reshape(-1)
    # remove embeddings from padding nodes
    node_embeddings = node_embeddings.reshape(-1, embeding_dim)[
        attention_masks_1d == 1
    ]
    return node_embeddings


def _freeze_bert(
    bert_model: BertModel, freeze_bert=False, freeze_layer_count=-1
):
    """Freeze parameters in BertModel (in place)

    Args:
        bert_model: HuggingFace bert model
        freeze_bert: Bool whether or not to freeze the bert model
        freeze_layer_count: If freeze_bert, up to what layer to freeze.

    Returns:
        bert_model
    """
    if freeze_bert:
        # freeze the entire bert model
        for param in bert_model.parameters():
            param.requires_grad = False
    else:
        # freeze the embeddings
        for param in bert_model.embeddings.parameters():
            param.requires_grad = False
        if freeze_layer_count != -1:
            # freeze layers in bert_model.encoder
            for layer in bert_model.encoder.layer[:freeze_layer_count]:
                for param in layer.parameters():
                    param.requires_grad = False
    return None


class BaseModule(pl.LightningModule):
    """A generic pl.LightningModule with the following functionalities:
    - save hyperparams
    - compute loss based on problem types from flags `classify`, `multiclass`

    """

    def __init__(
        self,
        num_outputs=32,
        classify=True,
        multiclass=False,
        weights=None,
        **kwargs,
    ):
        """

        Args:
            num_outputs: number of output units to enable multi-task problems
            classify: if True: classification; else: regression problem
            multiclass: if True, multiclass; else: multi-label
            weights: a tensor of class weights

        Returns:
            None

        """
        super(BaseModule, self).__init__()
        self.save_hyperparameters(
            "num_outputs", "lr", "classify", "multiclass"
        )
        self.classify = classify
        self.multiclass = multiclass
        self.register_buffer("weights", weights)
       # self.accuracy = torchmetrics.Accuracy(task="multilabel", num_labels=num_outputs, average=None )
        self.average_precision = torchmetrics.Precision(task="multilabel", num_labels=num_outputs, average=None)
        self.recall = torchmetrics.Recall(task="multilabel", num_labels=num_outputs, average=None)

    @staticmethod
    def add_model_specific_args(parent_parser):
        """Adds model specific args to the base/parent parser.

        Args:
            parent_parser: Base/parent parser

        Returns:
            parent parser with additional model-specific args

        """
        return parent_parser

    def configure_optimizers(self):
        """Initialize optimizer

        Args:
            None

        Returns:
            None

        """
        optimizer = torch.optim.Adam (self.parameters(), lr=self.hparams.lr)
        return optimizer

    def _compute_loss(self, logits, targets):
        """Compute x-ent or MSE loss based on `classify` and `multiclass`
        flags.

        Args:
            logits: Predicted logits
            targets: Ground truth

        Returns:
            loss
        """
        if self.classify:
            if self.multiclass:
                # single-label multiclass
                loss = F.cross_entropy(logits, targets, weight=self.weights)
            else:
                # multi-label classification

               # loss = F.cross_entropy (logits, targets, reduction="mean",) #self.weights)
                #F_loss = (1 - pt) ** 5 * BCE_loss
                #loss= F_loss.sum()
                #loss_func = FocalLoss()
                #loss =  loss_func.forward(logits, targets)
                loss = F.binary_cross_entropy_with_logits(
                    logits, targets, reduction="mean",# weight=self.weights
                )
               # loss = (loss * self.weights).sum()
              #  self.accuracy(logits, targets)
                self.average_precision(logits, targets.int())
                self.recall(logits, targets.int())
                # self.log('acc', self.accuracy, on_step=False, on_epoch=True)
             #   myavg = self.accuracy.compute()
                myprec = self.average_precision.compute()
                myrecall = self.recall.compute()
              #  count = 0
              #  for i in myavg:
              #      count += 1
              #      self.log("class_average{}".format(count), i, batch_size=len(logits[0]), sync_dist=True)
                count = 0
                for i in myprec:
                    count += 1
                    self.log("class_precision{}".format(count), i, batch_size=len(logits[0]), sync_dist=True)
                count = 0
                for i in myrecall:
                    count += 1
                    self.log("class_recall{}".format(count), i, batch_size=len(logits[0]), sync_dist=True)

               # self.confusion_matrix(logits, targets.int())

                #print(self.accuracy)
                #print(self.average_precision)
                #print(self.confusion_matrix)
                #self.log('acc', self.accuracy, on_step=False, on_epoch=True)
                #self.log('prec', self.average_precision, on_step=False, on_epoch=True)
              #  self.log('confusion', self.confusion_matrix, on_step=False, on_epoch=True)
              #  self.logger.log_metrics()
        else:  # regression
            loss = F.mse_loss(logits, targets)
        return loss

    def _step(self, batch, batch_idx, prefix="train"):
        """Will be used in train/validation loop, independent of `forward`.
        To be implemented by child classes

        Args:
            batch: (torch_geometric.data.Data, targets)
            batch_idx: index of current batch
            prefix: Prefix for the loss: XXX_loss (train, validation, test)

        Returns:
            Loss

        """
        raise NotImplementedError

    def training_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, prefix="train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, prefix="val")

    def test_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, prefix="test")
class BertFinetuneModel(BaseModule):
    """Sequence-only baseline: Bert + linear layer on pooler_output"""

    def __init__(self, **kwargs):
        """Initializes the module

        Args:
            None

        Returns:
            None
        """
        super(BertFinetuneModel, self).__init__(**kwargs)
      #  self.bert_model = BertModel.from_pretrained("yarongef/DistilProtBert",torch_dtype="auto" )
        self.bert_model = EsmModel.from_pretrained("facebook/esm2_t33_650M_UR50D",torch_dtype="auto")
        # freeze the embeddings
        _freeze_bert(self.bert_model, freeze_bert=False, freeze_layer_count=-2 )
        self.dense = nn.Sequential(
           # nn.Linear(4800, 2048), # changed to 4800 for 2752 go terms
      #      nn.Linear(4800,n_hidden), #added
            nn.Linear(self.bert_model.pooler.dense.out_features, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512 ,self.hparams.num_outputs)

       #     nn.Linear(n_hidden, self.hparams.num_outputs),
        )
    #    self.output = nn.Linear(
    #        self.bert_model.pooler.dense.out_features+2752, self.hparams.num_outputs
    #    )

    def _forward(self, input_ids, attention_mask):#,go):
        """Helper function to perform the forward pass.

        Args:
            batch: torch_geometric.data.Data
            input_ids: IDs of the embeddings to be used in the model.
            attention_mask: Masking to use durinig BERT's self-attention.

        Returns:
            logits
        """
        x = self.bert_model(
            input_ids, attention_mask=attention_mask
        ).pooler_output
       # x = torch.cat((x, go), dim=-1)
        #outputs = self.output(x)
        outputs=self.dense(x)
        return outputs

    def forward(self, batch):
        """Performs the forward pass.

        Args:
            batch: (torch_geometric.data.Data, targets)

        Returns:
            logits
        """
        batch_size = batch.num_graphs
       # go = batch["go_terms"].reshape(batch_size, -1)
        input_ids = batch.input_ids.reshape(batch_size, -1)
        attention_mask = batch.attention_mask.reshape(batch_size, -1)
        outputs = self._forward(input_ids, attention_mask)#, go)
        return outputs

    def _step(self, batch, batch_idx, prefix="train"):
        """Used in train/validation loop, independent of `forward`

        Args:
            batch: (torch_geometric.data.Data, targets)
            batch_idx: index of current batch
            prefix: Prefix for the loss: XXX_loss (train, validation, test)

        Returns:
            Loss
        """
        x, targets = batch
        logits = self.forward(x)
        loss = self._compute_loss(logits, targets)
        self.log("{}_loss".format(prefix), loss, batch_size=len(x), sync_dist=True)
        return loss
       # batch_size = batch.num_graphs
       # go = batch["go_terms"].reshape(batch_size, -1)
       # logits = self._forward(batch["input_ids"], batch["attention_mask"], go)
       # loss = self._compute_loss(logits, batch["labels"])
       # self.log("{}_loss".format(prefix), loss)
       # return loss


class BertGATModel(BaseModule):
    """Bert + GAT head."""

    def __init__(
        self,
        n_hidden=512,
        drop_rate=0.2,
        freeze_bert=True,
        freeze_layer_count=-1,
        **kwargs,
    ):
        """Initializes the module

        Args:
            n_hidden: number of hidden units in classification head.
            drop_rate: rate to use in the dropout layer.
            freeze_bert: Whether or not to freeze bert layers for training.
            freeze_layer_count: If freeze_bert is true, up to what layer to freeze. -1 freeze all of them.

        Returns:
            None
        """

        super(BertGATModel, self).__init__(**kwargs)
        self.save_hyperparameters(
            "n_hidden",
            "drop_rate",
            "freeze_bert",
            "freeze_layer_count",
        )
        self.bert_model = BertModel.from_pretrained("yarongef/DistilProtBert",torch_dtype="auto" , )
        self.embeding_dim = self.bert_model.pooler.dense.out_features
        self.conv1 = GATv2Conv(self.embeding_dim, 128, 4)
        self.conv2 = GATv2Conv(512, 128, 4)
        self.conv3 = GATv2Conv(512, 256, 4)

        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=drop_rate)

        self.dense = nn.Sequential(
           # nn.Linear(4800, 2048), # changed to 4800 for 2752 go terms
            nn.Linear(2048,n_hidden), #added
            nn.ReLU(inplace=True),
            nn.Linear(n_hidden, self.hparams.num_outputs),
        )

        _freeze_bert(
            self.bert_model,
            freeze_bert=freeze_bert,
            freeze_layer_count=freeze_layer_count,
        )

    @staticmethod
    def add_model_specific_args(parent_parser):
        """Adds model specific args to the base/parent parser.

        Args:
            parent_parser: Base/parent parser

        Returns:
            parent parser with additional model-specific args

        """
        parser = parent_parser.add_argument_group("BertGATModel")
        parser.add_argument("--freeze_bert", type=bool, default=True)
        parser.add_argument(
            "--freeze_layer_count",
            type=int,
            default=-1,
            help="Number of bert.encoding layers to freeze, "
            + "if -1, unfreeze all",
        )
        return parent_parser

    def _step(self, batch, batch_idx, prefix="train"):
        """Forward pass and computation of the loss.

        Args:
            batch: (torch_geometric.data.Data, targets)
            batch_idx: index of current batch
            prefix: Prefix for the loss: XXX_loss (train, validation, test)

        Returns:
            Loss
        """
        x, targets = batch
        logits = self._forward(x)
        loss = self._compute_loss(logits, targets)
        self.log("{}_loss".format(prefix), loss, batch_size=len(x),  sync_dist=True)

      #  self.log("class2", myprec[1])
      #  self.log("class3", mypraec[3])
     #   self.log('prec', self.average_precision, on_epoch=True, on_step=False)

        return loss

    def forward(self, batch):
        """Does the forward pass through the model for batch[0]

        Args:
            batch: (torch_geometric.data.Data, targets)

        Returns:
            Inferenced logits
        """
        x, targets = batch
        logits = self._forward(x)
        return logits

    def _forward(self, batch):
        """Does the forward pass through the model for batch

        Args:
            batch: torch_geometric.data.Data

        Returns:
            Inferenced logits

        """
        batch_size = batch.num_graphs
        input_ids = batch.input_ids.reshape(batch_size, -1)
        attention_mask = batch.attention_mask.reshape(batch_size, -1)
      #  go = batch["go_terms"].reshape(batch_size, -1)
        node_embeddings = _bert_forward(
            self.bert_model, self.embeding_dim, input_ids, attention_mask
        )
        # GAT forward
      #  conv1_out = self.conv1(
            #_bert_forward(self.bert_model, self.embeding_dim, batch.input_ids.reshape(batch.num_graphs, -1),
                      #    batch.attention_mask.reshape(batch.num_graphs, -1)),
       #                            batch.node_embedding ,batch.edge_index)
        conv1_out = self.conv1(node_embeddings, batch.edge_index)
        conv2_out = self.conv2(conv1_out, batch.edge_index)
        conv3_out = self.conv3(conv2_out, batch.edge_index)
        # residual concat
        out = torch.cat((conv1_out, conv2_out, conv3_out), dim=-1)
        out = self.dropout(self.relu(out))  # [n_nodes, 2048]

        # aggregate node vectors to graph
     #   out = out.scatter(out, batch.batch, dim=0)
        out = scatter_mean(out, batch.batch, dim=0)  # [bs, 2048]
       # out = torch.cat((out, go), dim=-1)
        return self.dense(out).squeeze(-1)# + 0.5  # [bs]


class BertMQAModel(BaseModule):
    """Bert + GVP-GNN head (LM-GVP).
    Takes in protein structure graphs of type `torch_geometric.data.Data`
    or `torch_geometric.data.Batch` and returns a scalar score for
    each graph in the batch in a `torch.Tensor` of shape [n_nodes]
    Should be used with `data.ProteinGraphDataset`, or with generators
    of `torch_geometric.data.Batch` objects with the same attributes.
    """

    def __init__(
        self,
        node_in_dim: Tuple[int, int],
        node_h_dim: Tuple[int, int],
        edge_in_dim: Tuple[int, int],
        edge_h_dim: Tuple[int, int],
        num_layers=3,
        drop_rate=0.1,
        residual=True,
        freeze_bert=True,
        freeze_layer_count=-1,
        **kwargs,
    ):
        """
        Args:
            node_in_dim: node dimensions (s, V) in input graph
            node_h_dim: node dimensions to use in GVP-GNN layers
            edge_in_dim: edge dimensions (s, V) in input graph
            edge_h_dim: edge dimensions to embed to before use in GVP-GNN layers
            weights: a tensor of class weights
            num_layers: number of GVP-GNN layers
            drop_rate: rate to use in all dropout layers
            residual: whether to have residual connections among GNN layers
            freeze_bert: wheter to freeze the entire bert model
            freeze_layer_count: number of bert.embedding layers to freeze.
        Returns:
            None
        """

        super(BertMQAModel, self).__init__(**kwargs)
        self.save_hyperparameters(
            "node_in_dim",
            "node_h_dim",
            "edge_in_dim",
            "edge_h_dim",
            "num_layers",
            "drop_rate",
            "freeze_bert",
            "freeze_layer_count",
            "residual",
        )
        self.identity = nn.Identity()
        self.bert_model = BertModel.from_pretrained("yarongef/DistilProtBert",torch_dtype="auto" )
        self.embeding_dim = self.bert_model.pooler.dense.out_features
        self.residual = residual
        _freeze_bert(
            self.bert_model,
            freeze_bert=freeze_bert,
            freeze_layer_count=freeze_layer_count,
        )

        node_in_dim = (node_in_dim[0] + 1024, node_in_dim[1])

        self.W_v = nn.Sequential(
            LayerNorm(node_in_dim),
            GVP(node_in_dim, node_h_dim, activations=(None, None)),
        )
        self.W_e = nn.Sequential(
            LayerNorm(edge_in_dim),
            GVP(edge_in_dim, edge_h_dim, activations=(None, None)),
        )

        self.layers = nn.ModuleList(
            GVPConvLayer(node_h_dim, edge_h_dim, drop_rate=drop_rate)
            for _ in range(num_layers)
        )

        if self.residual:
            # concat outputs from GVPConvLayer(s)
            node_h_dim = (
                node_h_dim[0] * num_layers,
                node_h_dim[1] * num_layers,
            )
        ns, _ = node_h_dim
        self.W_out = nn.Sequential(
            LayerNorm(node_h_dim), GVP(node_h_dim, (ns, 0))
        )
        self.dense = nn.Sequential(
            nn.Linear(ns, 2 * ns),
            nn.ReLU(inplace=True),
            nn.Dropout(p=drop_rate),
            nn.Linear(2 * ns, self.hparams.num_outputs),
        )

    @staticmethod
    def add_model_specific_args(parent_parser):
        """Adds model specific args to the base/parent parser.
        Args:
            parent_parser: Base/parent parser
        Returns:
            parent parser with additional model-specific args
        """
        parser = parent_parser.add_argument_group("BertGVPModel")
        parser.add_argument("--freeze_bert", type=bool, default=True)
        parser.add_argument(
            "--freeze_layer_count",
            type=int,
            default=-1,
            help="how many bert.encoding layers to freeze. -1 -> unfreeze all",
        )
        parser.add_argument("--residual", type=bool, default=True)
        return parent_parser

    def _step(self, batch, batch_idx, prefix="train"):
        """Forward pass and computation of the loss.
        Args:
            batch: (torch_geometric.data.Data, targets)
            batch_idx: index of current batch
            prefix: Prefix for the loss: XXX_loss (train, validation, test)
        Returns:
            Loss
        """
        x, targets = batch
        logits = self._forward(x)
        loss = self._compute_loss(logits, targets)
        self.log("{}_loss".format(prefix), loss,batch_size=len(x), sync_dist=True)
        return loss

    def forward(self, batch, input_ids=None):
        """Perform the forward pass.
        Args:
            batch: (torch_geometric.data.Data, targets)
            input_ids: IDs of the embeddings to be used in the model.
        Returns:
            logits
        """
        x, targets = batch
        logits = self._forward(x, input_ids=input_ids)
        return logits
    def _forward(self, batch, input_ids=None):
        """
        Helper function to perform the forward pass.
        Args:
            batch: torch_geometric.data.Data
            input_ids: IDs of the embeddings to be used in the model.
        Returns:
            logits
        """
        h_V = (batch.node_s, batch.node_v)
        h_E = (batch.edge_s, batch.edge_v)
        edge_index = batch.edge_index
        batch_size = batch.num_graphs
    #    go = batch["go_terms"].reshape(batch_size, -1)

        batch_size = batch.num_graphs

        if input_ids is None:
            input_ids = batch.input_ids.reshape(batch_size, -1)
        attention_mask = batch.attention_mask.reshape(batch_size, -1)

        node_embeddings = _bert_forward(
            self.bert_model, self.embeding_dim, input_ids, attention_mask
        )
        node_embeddings = self.identity(node_embeddings)

        h_V = (torch.cat([h_V[0], node_embeddings], dim=-1), h_V[1])

        h_V = self.W_v(h_V)
        h_E = self.W_e(h_E)
        if not self.residual:
            for layer in self.layers:
                h_V = layer(h_V, edge_index, h_E)
            out = self.W_out(h_V)
        else:
            h_V_out = []  # collect outputs from GVPConvLayers
            h_V_in = h_V
            for layer in self.layers:
                h_V_out.append(layer(h_V_in, edge_index, h_E))
                h_V_in = h_V_out[-1]
            # concat outputs from GVPConvLayers (separatedly for s and V)
            h_V_out = (
                torch.cat([h_V[0] for h_V in h_V_out], dim=-1),
                torch.cat([h_V[1] for h_V in h_V_out], dim=-2),
            )
            out = self.W_out(h_V_out)

        out = scatter_mean(out, batch.batch, dim=0)
     #   out = torch.cat((out, go), dim=-1)
        return self.dense(out).squeeze(-1) + 0.5