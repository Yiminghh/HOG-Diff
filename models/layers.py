
from torch.nn.modules.dropout import Dropout
from torch.nn import Parameter
from torch.nn.modules.normalization import LayerNorm
from models.blocks import *


class ATTN(nn.Module):
    def __init__(self, node_dim, edge_dim, dx, n_head, dropout=0.1):
        """
        Args:
            node_dim: node feature dim.
            edge_dim: edge feature dim.
            dx: projected node feature dim; must equal n_head * df.
            n_head: number of attention heads.
        """
        super(ATTN, self).__init__()

        self.dx = dx
        self.n_head = n_head
        self.df = dx // n_head  # per-head feature dim

        # Node feature projections.
        self.q = nn.Linear(node_dim, dx)
        self.k = nn.Linear(node_dim, dx)
        self.v = nn.Linear(node_dim, dx)

        # Edge feature projections.
        self.phi_0 = nn.Linear(edge_dim, self.df)
        self.phi_1 = nn.Linear(edge_dim, self.df)

        # out
        self.dropX = Dropout(dropout)
        self.norm1 = LayerNorm(dx)

        self.outX = nn.Sequential(nn.Linear(dx, dx), nn.ReLU(), Dropout(dropout))
        self.norm2 = LayerNorm(dx)

        self.se = SELayer(dx)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.q.weight)
        nn.init.xavier_uniform_(self.k.weight)
        nn.init.xavier_uniform_(self.v.weight)
        nn.init.xavier_uniform_(self.phi_0.weight)
        nn.init.xavier_uniform_(self.phi_1.weight)

    def forward(self, X, E, masker):
        """
        :param X: bs, n, d        node features
        :param E: bs, n, n, de     edge features
        :param y: bs, dz           global features
        :param node_mask: bs, n
        :return: newX, newE, new_y with the same shape.
        """
        bs, n, d = X.shape
        x_mask = masker.get_atom_mask().unsqueeze(-1)               # bs, n, 1
        e_mask = masker.get_bond_mask(with_loop=True).unsqueeze(-1) # Edge mask (bs, n, n, 1)

        # 1. Map X to keys and queries
        Q = self.q(X) * x_mask           # (bs, n, dx)
        K = self.k(X) * x_mask           # (bs, n, dx)
        V = self.v(X) * x_mask
        # Edge feature mappings.
        edge_phi_0 = torch.tanh(self.phi_0(E)) * e_mask  # (bs, n, n, df)
        edge_phi_1 = torch.tanh(self.phi_1(E)) * e_mask  # (bs, n, n, df)

        # 2. Reshape to (bs, n, n_head, df) with dx = n_head * df
        Q = Q.reshape((bs, n, self.n_head, self.df))
        K = K.reshape((bs, n, self.n_head, self.df))
        V = V.reshape((bs, n, self.n_head, self.df))

        Q = Q.unsqueeze(2) * edge_phi_0.unsqueeze(-2)   # (bs, n, 1, n_head, df)
        K = K.unsqueeze(1)                              # (bs, 1, n, n head, df)
        V = edge_phi_1.unsqueeze(-2) * V.unsqueeze(1)   # (bs, 1, n, n head, df)

        # Compute unnormalized attentions
        attn_scores = (Q * K).sum(dim=-1) / math.sqrt(self.df)  # (bs, n, n, n_head)

        # Compute attentions
        attn_weights = F.softmax(attn_scores, dim=2)  # Normalize (bs, n, n, n_head)

        # Compute weighted values
        weighted_V = torch.einsum('bnnk,bnnkd->bnkd', attn_weights, V)  # (bs, n, n_head, df)

        # Reshape back to original dimensions
        out = weighted_V.flatten(start_dim=2)  # (bs, n, dx)
        out = self.norm1(self.dropX(out) + X)

        out = self.outX(out)
        out = self.norm2(out+X)

        out = self.se(out)

        return out





class DenseGCNConv(nn.Module):
    r"""See :class:`torch_geometric.nn.conv.GCNConv`.
    """
    def __init__(self, in_channels, out_channels, dropout=0.1, improved=False, bias=True):
        super(DenseGCNConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.improved = improved

        self.weight = Parameter(torch.Tensor(self.in_channels, out_channels))

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        # we modify
        self.out_layer = nn.Sequential(nn.ReLU(), Dropout(dropout))
        self.out_norm = LayerNorm(out_channels)
        self.se = SELayer(out_channels)

        self.reset_parameters()

    def zeros(self, tensor):
        if tensor is not None:
            tensor.data.fill_(0)

    def glorot(self, tensor):
        if tensor is not None:
            stdv = math.sqrt(6.0 / (tensor.size(-2) + tensor.size(-1)))
            tensor.data.uniform_(-stdv, stdv)

    def reset_parameters(self):
        self.glorot(self.weight)
        self.zeros(self.bias)


    def forward(self, x, adj, mask=None, add_loop=True):
        r"""
        Args:
            x (Tensor): Node feature tensor :math:`\mathbf{X} \in \mathbb{R}^{B
                \times N \times F}`, with batch-size :math:`B`, (maximum)
                number of nodes :math:`N` for each graph, and feature
                dimension :math:`F`.
            adj (Tensor): Adjacency tensor :math:`\mathbf{A} \in \mathbb{R}^{B
                \times N \times N}`. The adjacency tensor is broadcastable in
                the batch dimension, resulting in a shared adjacency matrix for
                the complete batch.
            mask (BoolTensor, optional): Mask matrix
                :math:`\mathbf{M} \in {\{ 0, 1 \}}^{B \times N}` indicating
                the valid nodes for each graph. (default: :obj:`None`)
            add_loop (bool, optional): If set to :obj:`False`, the layer will
                not automatically add self-loops to the adjacency matrices.
                (default: :obj:`True`)
        """
        x = x.unsqueeze(0) if x.dim() == 2 else x
        adj = adj.unsqueeze(0) if adj.dim() == 2 else adj
        B, N, _ = adj.size()

        if add_loop:
            adj = adj.clone()
            idx = torch.arange(N, dtype=torch.long, device=adj.device)
            adj[:, idx, idx] = 1 if not self.improved else 2

        out = torch.matmul(x, self.weight)
        deg_inv_sqrt = adj.sum(dim=-1).clamp(min=1).pow(-0.5)

        adj = deg_inv_sqrt.unsqueeze(-1) * adj * deg_inv_sqrt.unsqueeze(-2)
        out = torch.matmul(adj, out)

        if self.bias is not None:
            out = out + self.bias

        if mask is not None:
            out = out * mask.view(B, N, 1).to(x.dtype)

        out = self.out_layer(out)
        out = self.out_norm(out + x)
        out = self.se(out)

        return out


    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)

