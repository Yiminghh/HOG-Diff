from models import utils
from models.layers import *





@utils.register_model(name='ConScoreNet')
class ConScoreNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.data_name = config.data.name
        self.n_layers = config.model.num_gnn_layers
        self.edge_th = config.model.edge_th  # edge threshold
        self.rw_depth = rw_depth = config.model.rw_depth  # random-walk depth
        nf = config.model.nf
        act = get_act(config)
        max_node_num = config.data.max_node
        dropout = config.model.dropout
        n_head = config.model.heads
        atom_ch = config.data.atom_channels
        bond_ch = 1  # config.data.bond_channels

        # input layers
        self.time_modules = nn.Sequential(SinusoidalPosEmb(nf), nn.Linear(nf, nf * 2),
                                          nn.GELU(), nn.Linear(nf * 2, nf))

        self.pre_atom = nn.Sequential(nn.Linear((bond_ch + atom_ch + rw_depth) * 2, nf * 2), nn.LayerNorm(nf * 2), act,
                                      nn.Linear(nf * 2, nf), act)
        self.pre_bond1 = nn.Sequential(conv1x1((1 + 1 + rw_depth + 1) * 2, nf), nn.GroupNorm(1, nf), act)
        self.pre_bond2 = nn.Linear(nf, nf)

        self.glayers1, self.glayers2, self.attn = torch.nn.ModuleList(), torch.nn.ModuleList(), torch.nn.ModuleList()
        for _ in range(self.n_layers):
            self.glayers1.append(DenseGCNConv(nf, nf, dropout))
            self.glayers2.append(DenseGCNConv(nf, nf, dropout))
            self.attn.append(ATTN(node_dim=nf, edge_dim=nf, dx=nf, n_head=n_head, dropout=dropout))
        self.FilM, self.FilM1, self.FilM2 = FilM(nf, dropout), FilM(nf, dropout), FilM(nf, dropout)

        # process y
        self.y_y = nn.Linear(nf, nf)
        self.yPAN, self.yPAN1, self.yPAN2 = PNA(nf, nf), PNA(nf, nf), PNA(nf, nf)
        self.post_y = nn.Sequential(nn.Linear(nf, nf), nn.ReLU(), Dropout(dropout), nn.Linear(nf, nf))
        self.norm_y = LayerNorm(nf)

        # out
        fdim = 2 * atom_ch + nf + 3 * self.n_layers * nf
        self.post_h = nn.Sequential(nn.Linear(fdim, 2 * nf), act, Dropout(dropout), LayerNorm(2 * nf),
                                    nn.Linear(2 * nf, nf), act, Dropout(dropout), LayerNorm(nf))

        self.out_x = nn.Linear(nf, atom_ch)
        self.final_x = nn.Sequential(nn.Linear(2 * atom_ch, atom_ch), act,
                                     nn.Linear(atom_ch, atom_ch))

        self.out_la = nn.Sequential(nn.Linear(nf, 2 * max_node_num), act,
                                    nn.Linear(2 * max_node_num, max_node_num // 2), act,
                                    nn.Linear(max_node_num // 2, 1))
        self.final_la = nn.Sequential(nn.Linear(3 * max_node_num, 2 * max_node_num), act,
                                      nn.Linear(2 * max_node_num, max_node_num))


    def set_mu(self, mu):
        self.X1, self.la1, self.adj1 = mu
        self.atom_degree1 = torch.sum(self.adj1, dim=-1).unsqueeze(-1)
        with torch.no_grad():
            self.adj1_b = (self.adj1 > self.edge_th).int()
            self.rw_landing1, self.spd_onehot1 = utils.get_rw_feat(self.rw_depth, self.adj1_b)

        # if self.data_name in ['qm9', 'zinc250k']:
        #     #print(torch.unique(self.X1))
        #     assert torch.equal(torch.unique(self.X1), torch.tensor([-0.5, 0, 0.5], device=self.X1.device)), f"Not scale properly, torch,unique={torch.unique(self.X1)}"

    def forward(self, graph, timestep, masker):
        X0, la0, adj0 = graph  # X:(bs,N,ch), la: (bs,N), E:(bs,N,N)
        bond_mask = masker.get_bond_mask().unsqueeze(-1)  # [bs,N,N,1]

        #x_to_out = torch.cat([X0, self.X1], dim=-1)  # (128,20,10)->(128,20,20)
        la_to_out = torch.cat([la0, self.la1], dim=-1)  # (128,20)->(128,40)

        with torch.no_grad():
            adj0_b = (adj0 > self.edge_th).int()
            rw_landing0, spd_onehot0 = utils.get_rw_feat(self.rw_depth, adj0_b)  # [B, N, rw_depth], [B,rw_depth+1,N,N]


        atom_degree = torch.sum(adj0, dim=-1).unsqueeze(-1)
        hX = self.pre_atom(torch.cat([X0, atom_degree, rw_landing0,
                                      self.X1, self.atom_degree1, self.rw_landing1], dim=-1))

        hE = self.pre_bond1(torch.cat([adj0.unsqueeze(1), adj0_b.unsqueeze(1), spd_onehot0,
                                      self.adj1.unsqueeze(1), self.adj1_b.unsqueeze(1), self.spd_onehot1], dim=1)).permute(0, 2, 3, 1)  # [bs,E_ch,N,N]->[bs,N,N,E_ch]
        hE = self.pre_bond2(hE)
        hE = (hE + hE.transpose(1, 2)) / 2 * bond_mask

        hT = self.time_modules(timestep)  # (bs,nf)

        h_list = [X0, self.X1, hX]  ##(bs,N,4)+(bs,N,nf)
        hX1, hX2, = hX.clone(), hX.clone()
        for _ in range(self.n_layers):
            newT = self.y_y(hT) + self.yPAN(hX) + self.yPAN1(hX1) + self.yPAN2(hX2)
            hX = self.attn[_](hX, hE, masker)
            hX1 = self.glayers1[_](hX1, adj0)
            hX2 = self.glayers2[_](hX2, self.adj1)
            h_list.extend([self.FilM(hX, hT), self.FilM1(hX1, hT), self.FilM2(hX2, hT)])
            hT = self.norm_y(self.post_y(newT) + hT)

        hs = self.post_h(torch.cat(h_list, dim=-1))

        x = self.out_x(hs)
        x = self.final_x(torch.cat([x, X0], dim=-1))

        hla = self.out_la(hs).squeeze(-1)
        hla = self.final_la(torch.cat([hla, la_to_out], dim=-1))

        return masker.mask_atom(x), masker.mask_la(hla)


@utils.register_model(name='ScoreNet')
class ScoreNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.data_name = config.data.name
        self.n_layers = config.model.num_gnn_layers
        self.edge_th = config.model.edge_th  # edge threshold
        self.rw_depth = rw_depth = config.model.rw_depth  # random-walk depth
        nf = config.model.nf
        act = get_act(config)
        max_node_num = config.data.max_node
        dropout = config.model.dropout
        n_head = config.model.heads
        atom_ch = config.data.atom_channels
        bond_ch = 1  # config.data.bond_channels

        # input layers
        self.time_modules = nn.Sequential(SinusoidalPosEmb(nf), nn.Linear(nf, nf * 2),
                                          nn.GELU(), nn.Linear(nf * 2, nf))

        self.pre_atom = nn.Sequential(nn.Linear(bond_ch + atom_ch + rw_depth, nf), nn.LayerNorm(nf), act,
                                      nn.Linear(nf, nf), act)
        self.pre_bond1 = nn.Sequential(conv1x1(1 + 1 + rw_depth + 1, nf), nn.GroupNorm(1, nf), act)
        self.pre_bond2 = nn.Linear(nf, nf)

        self.glayers1, self.attn = torch.nn.ModuleList(), torch.nn.ModuleList()
        for _ in range(self.n_layers):
            self.glayers1.append(DenseGCNConv(nf, nf, dropout))
            self.attn.append(ATTN(node_dim=nf, edge_dim=nf, dx=nf, n_head=n_head, dropout=dropout))
        self.FilM, self.FilM1 = FilM(nf, dropout), FilM(nf, dropout)

        # process y
        self.y_y = nn.Linear(nf, nf)
        self.yPAN, self.yPAN1 = PNA(nf, nf), PNA(nf, nf)
        self.post_y = nn.Sequential(nn.Linear(nf, nf), nn.ReLU(), Dropout(dropout), nn.Linear(nf, nf))
        self.norm_y = LayerNorm(nf)

        # out
        fdim = atom_ch + nf + 2 * self.n_layers * nf
        self.post_h = nn.Sequential(nn.Linear(fdim, 2 * nf), act, Dropout(dropout), LayerNorm(2 * nf),
                                    nn.Linear(2 * nf, nf), act, Dropout(dropout), LayerNorm(nf))

        self.out_x = nn.Linear(nf, atom_ch)
        self.final_x = nn.Sequential(nn.Linear(2 * atom_ch, atom_ch), act,
                                     nn.Linear(atom_ch, atom_ch))

        self.out_la = nn.Sequential(nn.Linear(nf, 2 * max_node_num), act,
                                    nn.Linear(2 * max_node_num, max_node_num // 2), act,
                                    nn.Linear(max_node_num // 2, 1))
        self.final_la = nn.Sequential(nn.Linear(2 * max_node_num, max_node_num), act,
                                      nn.Linear(max_node_num, max_node_num))



    def forward(self, graph, timestep, masker):
        X0, la0, adj0 = graph  # X:(bs,N,ch), la: (bs,N), E:(bs,N,N)
        bond_mask = masker.get_bond_mask().unsqueeze(-1)  # [bs,N,N,1]

        with torch.no_grad():
            adj0_b = (adj0 > self.edge_th).int()
            rw_landing0, spd_onehot0 = utils.get_rw_feat(self.rw_depth, adj0_b)  # [B, N, rw_depth], [B,rw_depth+1,N,N]

        atom_degree = torch.sum(adj0, dim=-1).unsqueeze(-1)
        hX = self.pre_atom(torch.cat([X0, atom_degree, rw_landing0], dim=-1))

        hE = self.pre_bond1(torch.cat([adj0.unsqueeze(1), adj0_b.unsqueeze(1), spd_onehot0], dim=1)).permute(0, 2, 3, 1)  # [bs,E_ch,N,N]->[bs,N,N,E_ch]
        hE = self.pre_bond2(hE)
        hE = (hE + hE.transpose(1, 2)) / 2 * bond_mask

        hT = self.time_modules(timestep)  # (bs,nf)

        h_list = [X0, hX]  ##(bs,N,4)+(bs,N,nf)
        hX1 = hX.clone()
        for _ in range(self.n_layers):
            newT = self.y_y(hT) + self.yPAN(hX) + self.yPAN1(hX1)
            hX = self.attn[_](hX, hE, masker)
            hX1 = self.glayers1[_](hX1, adj0)
            h_list.extend([self.FilM(hX, hT), self.FilM1(hX1, hT)])
            hT = self.norm_y(self.post_y(newT) + hT)

        hs = self.post_h(torch.cat(h_list, dim=-1))

        x = self.out_x(hs)
        x = self.final_x(torch.cat([x, X0], dim=-1))

        hla = self.out_la(hs).squeeze(-1)
        hla = self.final_la(torch.cat([hla, la0], dim=-1))

        return masker.mask_atom(x), masker.mask_la(hla)





