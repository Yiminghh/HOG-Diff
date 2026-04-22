
import torch
class data_masker(object):
    def __init__(self, graph, config, device=None, initial=False, eps=1e-5):
        device = graph.device if device is None else device
        self.bs, self.N, tmp_N = graph.shape
        assert self.N == tmp_N, "Data_masker input is not a square graph"
        self.latent_space = config.data.latent_space

        self.atom_mask = torch.abs(graph).sum(-1).gt(eps).to(dtype=torch.float32, device=device)
        if initial:
            self._init_bond_mask()
            self._init_la_mask()
        else:
            self.bond_mask = None
            self.bond_mask_with_loop = None
            self.la_mask = None

    def _init_bond_mask(self):
        bond_mask = self.atom_mask[:, None, :] * self.atom_mask[:, :, None]
        self.bond_mask_with_loop = bond_mask
        bond_mask = torch.tril(bond_mask, -1)
        self.bond_mask = bond_mask + bond_mask.transpose(-1, -2)


    def _init_la_mask(self):
        if self.latent_space == 'Laplacian':
            self.la_mask = torch.sort(self.atom_mask, dim=-1, descending=True)[0]
        elif self.latent_space == 'adj':
            self.la_mask = torch.zeros((self.bs, self.N)).to(self.atom_mask.device)
            for i in range(self.bs):
                count = torch.count_nonzero(self.atom_mask[i])
                self.la_mask[i, :count // 2] = 1
                self.la_mask[i, -count // 2:] = 1


    def mask_atom(self, x):
        # x: [bs, N, channels]
        return x * self.atom_mask.unsqueeze(-1)

    def mask_bond(self, x):
        # x: [bs, N, N]
        if self.bond_mask is None:
            self._init_bond_mask()
        return x * self.bond_mask

    def mask_la(self, x):
        # x: [bs,N]
        if self.la_mask is None:
            self._init_la_mask()
        return x * self.la_mask

    def get_atom_mask(self):
        return self.atom_mask

    def get_bond_mask(self, with_loop=False):
        if self.bond_mask is None:
            self._init_bond_mask()
        if with_loop:
            # Diagonal entries are 1.
            return self.bond_mask_with_loop
        else:
            return self.bond_mask



    def get_la_mask(self):
        if self.la_mask is None:
            self._init_la_mask()
        return self.la_mask

    def to(self, device):
        attribute_list = ['atom_mask', 'bond_mask', 'la_mask', 'bond_mask_with_loop']
        for attr_name in attribute_list:
            attr = getattr(self, attr_name)
            if attr is not None:
                setattr(self, attr_name, attr.to(device))
        return self


def get_data_scaler(config):
    """Data normalizer. Assume data are always in [0, 1]."""

    centered = config.data.centered
    atom_norm, bond_norm = config.data.norm
    if 'la_norm' in config.data:
        la_norm = config.data.la_norm

    def scale_fn(x, type):

        if type == 'atom':
            # {0, 1} -> {-0.5, 0.5}
            if centered:
                x = x * 2. - 1.
            x = x / atom_norm
        elif type == 'bond':
            # Bond scaling is handled in the dataloader.
            pass
        elif type == 'eig':
            pass
        return x

    return scale_fn



def get_data_inverse_scaler(config):
    """Inverse data normalizer."""

    centered = config.data.centered
    atom_norm, bond_norm = config.data.norm

    def inverse_scale_fn(x, type):
        if type == 'atom':

            x = x * atom_norm
            if centered:
                x = (x + 1.) / 2.
        elif type == 'bond':
            pass
        elif type == 'eig':
            pass
        return x

    return inverse_scale_fn

