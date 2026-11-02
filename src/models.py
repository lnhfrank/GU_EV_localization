"""GNN models for graph-level multi-label classification.

All models share:
  - 3 conv layers, BatchNorm, Dropout
  - Readout, switch `readout`:
      'dual_pool': mean + max pooling -> 2-layer MLP head, the submitted model
      'station':   per-EVCS readout, each station scored from its own bus
  - Output: raw logits (no sigmoid)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, GINConv, BatchNorm
from torch_geometric.nn import global_mean_pool, global_max_pool


READOUTS = ('dual_pool', 'station')


def _pool(x, batch):
    """Mean and max pooled node embeddings, [B, 2*hid_dim]."""
    return torch.cat([global_mean_pool(x, batch), global_max_pool(x, batch)], dim=-1)


class _Readout(nn.Module):
    """Readout shared by the three backbones.

    'dual_pool' reproduces the submitted model: the pooled graph embedding goes
    through a 2-layer MLP with one output per EVCS. 'station' scores EVCS k from
    the embedding of its own bus next to the pooled graph embedding, with an MLP
    shared by the stations and one bias per station, so a label is no longer
    read from an average over all buses (R1-3(iii)). Subclasses define
    node_embed(data) and call _init_readout after building their layers, so the
    parameters of 'dual_pool' are created in the submitted order.
    """

    def _init_readout(self, hid_dim, out_dim, readout, station_nodes):
        if readout not in READOUTS:
            raise ValueError(f"readout must be one of {list(READOUTS)}, got {readout!r}")
        self.readout = readout
        self.graph_dim = hid_dim * 2
        if readout == 'dual_pool':
            self.fc1 = nn.Linear(hid_dim * 2, hid_dim)
            self.fc2 = nn.Linear(hid_dim, out_dim)
            return
        if station_nodes is None or len(station_nodes) != out_dim:
            raise ValueError("readout='station' needs one bus index per EVCS output")
        self.fc1 = nn.Linear(hid_dim * 3, hid_dim)
        self.fc2 = nn.Linear(hid_dim, 1)
        self.station_bias = nn.Parameter(torch.zeros(out_dim))
        self.register_buffer('station_idx', torch.as_tensor(list(station_nodes), dtype=torch.long),
                             persistent=False)

    def encode(self, data):
        """Return pooled graph embedding [B, 2*hid_dim] (pre-head)."""
        return _pool(self.node_embed(data), data.batch)

    def head(self, x_nodes, g, data):
        """Localization logits [B, K] from the node embeddings and the pooled embedding g."""
        if self.readout == 'dual_pool':
            x = F.relu(self.fc1(g))
            x = F.dropout(x, p=self.dropout, training=self.training)
            return self.fc2(x)
        ptr = getattr(data, 'ptr', None)
        if ptr is None:
            counts = torch.bincount(data.batch, minlength=g.shape[0])
            ptr = torch.cat([counts.new_zeros(1), counts.cumsum(0)])
        idx = ptr[:-1].unsqueeze(1) + self.station_idx.unsqueeze(0)          # [B, K]
        z = torch.cat([x_nodes[idx], g.unsqueeze(1).expand(-1, idx.shape[1], -1)], dim=-1)
        x = F.relu(self.fc1(z))
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.fc2(x).squeeze(-1) + self.station_bias

    def forward(self, data):
        x_nodes = self.node_embed(data)
        return self.head(x_nodes, _pool(x_nodes, data.batch), data)


class GCN_Graph(_Readout):
    """GCN with dual pooling or the per-EVCS readout."""

    def __init__(self, in_dim, hid_dim=128, out_dim=3, n_layers=3, dropout=0.3,
                 readout='dual_pool', station_nodes=None):
        super().__init__()
        self.hid_dim = hid_dim
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.convs.append(GCNConv(in_dim, hid_dim))
        self.bns.append(BatchNorm(hid_dim))
        for _ in range(n_layers - 1):
            self.convs.append(GCNConv(hid_dim, hid_dim))
            self.bns.append(BatchNorm(hid_dim))
        self._init_readout(hid_dim, out_dim, readout, station_nodes)
        self.dropout = dropout

    def node_embed(self, data):
        """Node embeddings [N, hid_dim] after the last conv layer."""
        x, ei = data.x, data.edge_index
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, ei)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return x


class GAT_Graph(_Readout):
    """GAT with multi-head attention and dual pooling or the per-EVCS readout."""

    def __init__(self, in_dim, hid_dim=128, out_dim=3, n_layers=3, dropout=0.3, heads=4,
                 readout='dual_pool', station_nodes=None):
        super().__init__()
        self.hid_dim = hid_dim
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        # Layer 0: in_dim → hid_dim*heads
        self.convs.append(GATConv(in_dim, hid_dim, heads=heads, concat=True, dropout=dropout))
        self.bns.append(BatchNorm(hid_dim * heads))
        # Middle layers: hid_dim*heads → hid_dim*heads
        for _ in range(n_layers - 2):
            self.convs.append(GATConv(hid_dim * heads, hid_dim, heads=heads, concat=True, dropout=dropout))
            self.bns.append(BatchNorm(hid_dim * heads))
        # Last layer: hid_dim*heads → hid_dim (heads=1)
        self.convs.append(GATConv(hid_dim * heads, hid_dim, heads=1, concat=False, dropout=dropout))
        self.bns.append(BatchNorm(hid_dim))
        self._init_readout(hid_dim, out_dim, readout, station_nodes)
        self.dropout = dropout

    def node_embed(self, data):
        """Node embeddings [N, hid_dim] after the last conv layer."""
        x, ei = data.x, data.edge_index
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, ei)
            x = bn(x)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return x


class GIN_Graph(_Readout):
    """GIN with dual pooling or the per-EVCS readout."""

    def __init__(self, in_dim, hid_dim=128, out_dim=3, n_layers=3, dropout=0.3,
                 readout='dual_pool', station_nodes=None):
        super().__init__()
        self.hid_dim = hid_dim
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        mlp0 = nn.Sequential(nn.Linear(in_dim, hid_dim), nn.ReLU(), nn.Linear(hid_dim, hid_dim))
        self.convs.append(GINConv(mlp0))
        self.bns.append(BatchNorm(hid_dim))
        for _ in range(n_layers - 1):
            mlp = nn.Sequential(nn.Linear(hid_dim, hid_dim), nn.ReLU(), nn.Linear(hid_dim, hid_dim))
            self.convs.append(GINConv(mlp))
            self.bns.append(BatchNorm(hid_dim))
        self._init_readout(hid_dim, out_dim, readout, station_nodes)
        self.dropout = dropout

    def node_embed(self, data):
        """Node embeddings [N, hid_dim] after the last conv layer."""
        x, ei = data.x, data.edge_index
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, ei)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return x


MODEL_CLASSES = {
    'GCN': GCN_Graph,
    'GAT': GAT_Graph,
    'GIN': GIN_Graph,
}


# ======================================================================
#  AuxWrapper: wraps a backbone with an attack-type auxiliary head
# ======================================================================

class AuxWrapper(nn.Module):
    """Wraps any single-head GNN with an auxiliary attack-type classification head.

    forward()      -> loc_logits only (compatible with existing unlearning code)
    forward_both() -> (loc_logits, type_logits) for joint training
    forward_aux()  -> type_logits for evaluation

    The auxiliary head reads the pooled graph embedding under either readout.
    """

    def __init__(self, backbone, n_types=5, aux_hidden=64):
        super().__init__()
        self.backbone = backbone
        D = backbone.graph_dim  # equals fc1.in_features under the submitted dual-pool readout
        self.aux_head = nn.Sequential(
            nn.Linear(D, aux_hidden), nn.ReLU(), nn.Linear(aux_hidden, n_types),
        )

    @property
    def convs(self):
        return self.backbone.convs

    @property
    def bns(self):
        return self.backbone.bns

    @property
    def hid_dim(self):
        return self.backbone.hid_dim

    def encode(self, data):
        return self.backbone.encode(data)

    def forward(self, data):
        return self.backbone(data)

    def forward_both(self, data):
        x_nodes = self.backbone.node_embed(data)
        emb = _pool(x_nodes, data.batch)
        loc = self.backbone.head(x_nodes, emb, data)
        typ = self.aux_head(emb)
        return loc, typ

    def forward_aux(self, data):
        return self.aux_head(self.backbone.encode(data))

    def freeze_aux(self):
        for p in self.aux_head.parameters():
            p.requires_grad = False

    def unfreeze_aux(self):
        for p in self.aux_head.parameters():
            p.requires_grad = True
