import logging

import torch
from tqdm import tqdm

from utils.losses import get_score_fn
from utils import sde, data_utils
from utils.sde import _SDES

_PREDICTORS = {}
_CORRECTORS = {}

def register_predictor(cls=None, *, name=None):
    """A decorator for registering predictor classes."""

    def _register(cls):
        local_name = cls.__name__ if name is None else name
        assert local_name not in _PREDICTORS, ValueError(f'Already registered predictor with name: {local_name}')
        _PREDICTORS[local_name] = cls
        return cls

    if cls is None:
        return _register
    else:
        return _register(cls)

def register_corrector(cls=None, *, name=None):
    """A decorator for registering corrector classes."""

    def _register(cls):
        local_name = cls.__name__ if name is None else name
        assert local_name not in _CORRECTORS, ValueError(f'Already registered corrector with name: {local_name}')
        _CORRECTORS[local_name] = cls
        return cls

    if cls is None:
        return _register
    else:
        return _register(cls)

@register_predictor(name='Euler')
class EulerMaruyamaPredictor():
  def __init__(self, sde_x, sde_adj, score_fn,  probability_flow=False):
    # Compute the reverse SDE/ODE
    self.dt = - torch.tensor(sde_x.dt)
    self.rsde_x  = sde_x.reverse(probability_flow)
    self.rsde_adj = sde_adj.reverse(probability_flow)
    self.score_fn = score_fn

  def update_fn(self, graph_tuple, t):
    x, la, adj = graph_tuple
    score_x, score_la = self.score_fn(graph_tuple=graph_tuple, t=t)

    # atom update
    drift_x, diffusion_x = self.rsde_x.sde(x, score_x, t)
    z_x = torch.randn_like(x, device='cpu').to(x.device)
    x_mean = x + drift_x * self.dt
    x = x_mean + diffusion_x[:, None, None] * torch.sqrt(-self.dt) * z_x

    # la update
    drift_la, diffusion_la = self.rsde_adj.sde(la, score_la, t)
    z_la = torch.randn_like(la, device='cpu').to(la.device)
    la_mean = la + drift_la * self.dt
    la = la_mean + diffusion_la[:, None] * torch.sqrt(-self.dt) * z_la


    return (x, la), (x_mean, la_mean)

@register_corrector(name='None')
class NoneCorrector():
    def __init__(self, sde_x, sde_adj, score_fn, snr, n_steps, scale_eps):
        pass

    def update_fn(self, graph_tuple, t):
        x, la, u, adj, masker = graph_tuple
        return (x, la), (x, la)


@register_corrector(name='Langevin')
class LangevinCorrector():
  def __init__(self, sde_x, sde_adj, score_fn, snr, n_steps, scale_eps):

    self.sde_x = sde_x
    self.sde_adj = sde_adj
    self.score_fn = score_fn
    self.snr = snr
    self.scale_eps = scale_eps
    self.n_steps = n_steps

  def update_fn(self, graph_tuple, t):


    x, la, u, adj, masker = graph_tuple
    u_T = torch.transpose(u, -1, -2)
    atom_snr, bond_snr = self.snr

    if isinstance(self.sde_x, (sde.VPSDE)):
      timestep = (t * (self.sde_x.N - 1) / self.sde_x.T).long()
      alpha_atom = self.sde_x.alphas.to(t.device)[timestep]
      alpha_bond = self.sde_adj.alphas.to(t.device)[timestep]
    else:
      alpha_atom = torch.ones_like(t)
      alpha_bond = torch.ones_like(t)


    for _ in range(self.n_steps):
      score_x, score_la = self.score_fn(graph_tuple=(x, la, adj), t=t)

      noise = torch.randn_like(x, device='cpu').to(x.device)
      grad_norm = torch.norm(score_x.reshape(score_x.shape[0], -1), dim=-1).mean()
      # Guard against division by zero when the score vanishes.
      grad_norm[grad_norm == 0] = 1.0
      noise_norm = torch.norm(noise.reshape(noise.shape[0], -1), dim=-1).mean()
      step_size = (atom_snr * noise_norm / grad_norm) ** 2 * 2 * alpha_atom
      x_mean = x + step_size[:, None, None] * score_x
      x = x_mean + torch.sqrt(step_size * 2)[:, None, None] * noise * self.scale_eps

      noise = torch.randn_like(la, device='cpu').to(la.device)
      grad_norm = torch.norm(score_la.reshape(score_la.shape[0], -1), dim=-1).mean()
      grad_norm[grad_norm == 0] = 1.0
      noise_norm = torch.norm(noise.reshape(noise.shape[0], -1), dim=-1).mean()
      step_size = (bond_snr * noise_norm / grad_norm) ** 2 * 2 * alpha_bond
      la_mean = la + step_size[:, None] * score_la
      la = la_mean + torch.sqrt(step_size * 2)[:, None] * noise * self.scale_eps


      x, la = masker.mask_atom(x), masker.mask_la(la)
      adj = cal_adj_from_eig(la, u, u_T, masker)


    return (x, la, adj), (x_mean, la_mean, cal_adj_from_eig(la_mean, u, u_T, masker))


def cal_adj_from_eig(la, u, u_T, masker, latent_space='Laplacian'):
    """Update adjacency matrix based on latent space and masking rules."""
    la_diag = torch.diag_embed(masker.mask_la(la))
    if latent_space == 'Laplacian':
        # For Laplacian latents: zero the diagonal via masking and negate to recover the adjacency.
        L = torch.bmm(torch.bmm(u, la_diag), u_T)
        adj = masker.mask_bond(-L)
    elif latent_space == 'adj':
        adj = masker.mask_bond(torch.bmm(torch.bmm(u, la_diag), u_T))
    return adj

def load_pc_sampler(config, device='cuda', eps=1e-3, disable_tqdm=False):
    """ Create a Predictor-Corrector (PC) sampler

    Args:
        config:
        eps: A `float` number. The reverse-time SDE and ODE are integrated to `epsilon` to avoid numerical issues.
        device:

    Returns:

    """
    continuous = config.training.continuous

    snr = (config.sampling.atom_snr, config.sampling.bond_snr)
    denoise = config.sampling.noise_removal
    n_steps = config.sampling.n_steps
    scale_eps = config.sampling.scale_eps
    probability_flow = config.sampling.probability_flow
    bs = config.eval.batch_size
    max_node = config.data.max_node
    latent_space = config.data.latent_space

    diff_steps = config.sde.atom.num_scales
    assert diff_steps == config.sde.bond.num_scales, 'sdes have different number of scales'

    atom_sde = _SDES[config.sde.type](config.sde.atom)
    bond_sde = _SDES[config.sde.type](config.sde.bond)

    if config.sde.type in ['VPSDE']:
        timesteps = torch.linspace(atom_sde.T, eps, diff_steps, device=device)
    elif config.sde.type in ['OUSDE', 'OUBridge']:
        timesteps = torch.linspace(diff_steps, 1, diff_steps, device=device).long()
    else:
        raise ValueError(f'Unknown sde type: {config.sde.type}')

    x_shape = (bs, max_node, config.data.atom_channels)
    la_shape = (bs, max_node)


    predictor = _PREDICTORS[config.sampling.predictor]
    corrector = _CORRECTORS[config.sampling.corrector]

    inverse_scaler = data_utils.get_data_inverse_scaler(config)



    def pc_sampler(model, mu, save_trajectory=None):
        u, masker = mu['u0'], mu['masker']
        u_T = torch.transpose(u, -1, -2)
        mode = 'OU' if 'ho_x' in mu else 'higher-order'
        if mode == 'OU':
            mu_x, mu_la, mu_adj = mu['ho_x'], mu['ho_la'], mu['ho_adj']
            atom_sde.set_mu(mu_x)
            bond_sde.set_mu(mu_la)
            model.set_mu((mu_x, mu_la, mu_adj))


        score_fn = get_score_fn(atom_sde, bond_sde, model, masker, train=False, continuous=continuous)
        predictor_obj = predictor(atom_sde, bond_sde, score_fn, probability_flow)
        corrector_obj = corrector(atom_sde, bond_sde, score_fn, snr, n_steps, scale_eps)

        with (torch.no_grad()):
            # Initial sample
            x = mu_x if mode=='OU' else masker.mask_atom(atom_sde.prior_sampling(x_shape).to(device))
            la = mu_la if mode=='OU' else masker.mask_la(bond_sde.prior_sampling(la_shape).to(device))
            adj = cal_adj_from_eig(la, u, u_T, masker, latent_space=latent_space)

            # -------- Reverse diffusion process --------
            for i in tqdm(range(0, diff_steps), desc='[PC Sampling]', disable=disable_tqdm):
                vec_t = timesteps[i].expand(bs)

                if config.sampling.corrector != 'None':
                    (x, la, adj), (_, _, _) = corrector_obj.update_fn((x, la, u, adj, masker), vec_t)

                (x, la), (x_mean, la_mean) = predictor_obj.update_fn((x, la, adj), vec_t)
                x, la = masker.mask_atom(x), masker.mask_la(la)
                adj = cal_adj_from_eig(la, u, u_T, masker, latent_space=latent_space)

                if torch.isnan(x_mean).any() and torch.isnan(la_mean).any():
                    logging.info(f'NaNs in predictor output: {x_mean}, {la_mean}')
                    raise ValueError(f'NaNs in predictor output: {x_mean}, {la_mean}')
                if save_trajectory is not None:
                    save_trajectory.append([
                        masker.mask_atom(inverse_scaler(x_mean if denoise else x, type='atom')),
                        cal_adj_from_eig(la_mean if denoise else la, u, u_T, masker, latent_space=latent_space),
                        masker.get_atom_mask().sum(-1).to(torch.int)
                        ])

            return masker.mask_atom(inverse_scaler(x_mean if denoise else x, type='atom')), \
                   masker.mask_la(inverse_scaler(la_mean if denoise else la, type='eig')), \
                   cal_adj_from_eig(la_mean if denoise else la, u, u_T, masker, latent_space=latent_space)

    return pc_sampler
