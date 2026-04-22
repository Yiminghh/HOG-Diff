
import torch
import utils.loader as loader
from utils import sde
from utils.sde import _SDES, VPSDE, OUSDE, OUBridge
from utils.data_utils import get_data_scaler, get_data_inverse_scaler

def get_score_fn(atom_sde, bond_sde, model, masker, train=True, continuous=True):
    """Wraps `score_fn` so that the model output corresponds to a real time-dependent score function.

        Args:
            atom_sde: An `sde_lib.SDE` object that represents the forward SDE.
            bond_sde: An `sde_lib.SDE` object that represents the forward SDE.
            model: A score model.
            train: `True` for training and `False` for evaluation.
            continuous: If `True`, the score-based model is expected to directly take continuous time steps.

        Returns:
            A score function.
        """

    supported_sdes = (VPSDE, OUSDE, OUBridge)
    assert isinstance(atom_sde, supported_sdes), f"SDE class {atom_sde.__class__.__name__} not supported."
    assert isinstance(bond_sde, supported_sdes), f"SDE class {bond_sde.__class__.__name__} not supported."
    assert continuous, "Discrete score function is not supported currently!"

    if train:
        model.train()
    else:
        model.eval()


    def normalized_score(pred, sde, t):
        """Scale neural network output by standard deviation and flip sign"""
        std = sde.marginal_prob_std(t)
        if isinstance(sde, OUBridge):
            # For OUBridge, std = 0 at t = 0 and at t = T.
            std[std == 0] = float('inf')

        if pred.ndim == 2:
            score = - pred / std[:, None]
        elif pred.ndim == 3:
            score = - pred / std[:, None, None]
        elif pred.ndim == 4:
            score = - pred / std[:, None, None, None]
        return score

    def score_fn(graph_tuple, t):
        # t:(bs,)
        if isinstance(atom_sde, sde.VPSDE):
            scaled_t = t
        elif isinstance(atom_sde, (sde.OUSDE, sde.OUBridge)):
            scaled_t = t / atom_sde.N

        pred_atom, pred_bond = model(graph_tuple, scaled_t, masker)

        atom_score = normalized_score(pred_atom, atom_sde, t)
        bond_score = normalized_score(pred_bond, bond_sde, t)
        return atom_score, bond_score

    return score_fn


def DenoisingScoreMatching(config, train, eps=1e-5):
    """ Create a loss function for training with arbitrary node SDE and edge SDE.

        Args:
            atom_sde, bond_sde: An `sde_lib.SDE` object that represents the forward SDE.
            train: `True` for training loss and `False` for evaluation loss.
            reduce_mean: If `True`, average the loss across data dimensions. Otherwise, sum the loss across data dimensions.
            continuous: `True` indicates that the model is defined to take continuous time steps.
                        Otherwise, it requires ad-hoc interpolation to take continuous time steps.
            eps: A `float` number. The smallest time step to sample from.

        Returns:
            A loss function.
        """

    continuous = config.training.continuous
    reduce_mean = config.training.reduce_mean
    likelihood_weighting = config.training.likelihood_weighting
    latent_space = config.data.latent_space
    data_scaler = get_data_scaler(config)
    inverse_scaler = get_data_inverse_scaler(config)  # The inverse data normalizer.
    weight_atom = config.training.weight_atom if 'weight_atom' in config.training else 1.0
    weight_bond = config.training.weight_bond if 'weight_bond' in config.training else 1.0

    atom_sde = _SDES[config.sde.type](config.sde.atom)
    bond_sde = _SDES[config.sde.type](config.sde.bond)

    def loss_fn(model, batch):
        """Compute the loss function.
            model: A score model.
            batch: A mini-batch of training data, including node_features, adjacency matrices, node mask and adj mask.
        """
        device = batch[0].device
        bs = batch[0].shape[0]
        # batches are already scaled and masked by load_batch.
        if len(batch) == 5:
            x, la, u, adj, masker = batch
            t = (torch.rand(bs) * (atom_sde.T - eps) + eps).to(device)
        elif len(batch) == 9:
            x, la, u, adj, ho_x, ho_la, ho_u, ho_adj, masker = batch
            atom_sde.set_mu(ho_x)
            bond_sde.set_mu(ho_la)
            model.set_mu((ho_x, ho_la, ho_adj))
            t = torch.randint(1, atom_sde.N, (bs,)).to(device)


        # perturbing atom
        z_x = torch.randn_like(x, device='cpu').to(device)
        mean_x, std_x = atom_sde.marginal_prob(x, t)
        perturbed_x = masker.mask_atom(mean_x + std_x[:, None, None] * z_x)

        # perturbing bond
        z_la = torch.randn_like(la, device='cpu').to(device)
        mean_la, std_la = bond_sde.marginal_prob(la, t)
        perturbed_la = masker.mask_la(mean_la + std_la[:, None] * z_la)

        u_T = torch.transpose(u, -1, -2)
        perturbed_la_diag = torch.diag_embed(masker.mask_la(inverse_scaler(perturbed_la, type='eig')))
        perturbed_adj = torch.bmm(torch.bmm(u, perturbed_la_diag), u_T)
        if latent_space == 'Laplacian':
            # Mask zeros out the diagonal, so negation recovers the adjacency matrix.
            perturbed_adj = masker.mask_bond(data_scaler(-perturbed_adj, type='bond'))
        elif latent_space == 'adj':
            perturbed_adj = masker.mask_bond(data_scaler(perturbed_adj, type='bond'))

        score_fn = get_score_fn(atom_sde, bond_sde, model, masker, train=train, continuous=continuous)
        score_x, score_la = score_fn((perturbed_x, perturbed_la, perturbed_adj), t)

        # atom loss
        atom_mask = masker.get_atom_mask().unsqueeze(-1).repeat(1, 1, x.shape[-1])
        atom_mask = atom_mask.reshape(bs, -1)
        losses_atom = torch.square(score_x * std_x[:, None, None] + z_x)
        losses_atom = losses_atom.reshape(losses_atom.shape[0], -1)
        if reduce_mean:
            losses_atom = torch.sum(losses_atom * atom_mask, dim=-1) / torch.sum(atom_mask, dim=-1)
        else:
            losses_atom = 0.5 * torch.sum(losses_atom * atom_mask, dim=-1)
        loss_atom = losses_atom.mean()

        # la loss
        la_mask = masker.get_la_mask()  # (bs,N)
        losses_bond = torch.square(score_la * std_la[:, None] + z_la)
        if reduce_mean:
            losses_bond = torch.sum(losses_bond * la_mask, dim=-1) / (torch.sum(la_mask, dim=-1) + 1e-8)
        else:
            losses_bond = 0.5 * torch.sum(losses_bond * la_mask, dim=-1)
        loss_bond = losses_bond.mean()

        return weight_atom*loss_atom + weight_bond*loss_bond, loss_atom, loss_bond

    return loss_fn


def MaxLikelihood(config, train, eps=1e-5):
    continuous = config.training.continuous
    reduce_mean = config.training.reduce_mean
    likelihood_weighting = config.training.likelihood_weighting
    latent_space = config.data.latent_space
    data_scaler = get_data_scaler(config)
    inverse_scaler = get_data_inverse_scaler(config)  # The inverse data normalizer.

    atom_sde = _SDES[config.sde.type](config.sde.atom)
    bond_sde = _SDES[config.sde.type](config.sde.bond)

    def loss_fn(model, batch):
        """Compute the loss function.
            model: A score model.
            batch: A mini-batch of training data, including node_features, adjacency matrices, node mask and adj mask.
        """
        device = batch[0].device
        bs = batch[0].shape[0]
        if len(batch) == 3:
            atom_feat, bond_feat, masker = batch
            t = torch.rand(bs, device=device) * (atom_sde.T - eps) + eps
        elif len(batch) == 4:
            atom_feat, bond_feat, ho_bond, masker = batch
            atom_sde.set_mu(atom_feat)
            bond_sde.set_mu(ho_bond)
            if is_parallel:
                model.module.set_mu((atom_feat, ho_bond))
            else:
                model.set_mu((atom_feat, ho_bond))
            t = torch.randint(1, atom_sde.N, (bs,), device=device)


        # perturbing atom
        z_atom = torch.randn_like(atom_feat, device=device)
        mean_atom, std_atom = atom_sde.marginal_prob(atom_feat, t)
        perturbed_atom = masker.mask_atom(mean_atom + std_atom[:, None, None] * z_atom)

        # perturbing bond
        z_bond = torch.randn_like(bond_feat, device=device)
        z_bond = torch.tril(z_bond, -1)
        z_bond = z_bond + z_bond.transpose(-1, -2)
        mean_bond, std_bond = bond_sde.marginal_prob(bond_feat, t)
        perturbed_bond = masker.mask_bond(mean_bond + std_bond[:, None, None] * z_bond)

        score_fn = get_score_fn(atom_sde, bond_sde, model, masker, train=train, continuous=continuous)
        atom_score, bond_score = score_fn((perturbed_atom, perturbed_bond), t)

        # atom loss
        atom_exp = perturbed_atom - atom_sde.reverse().sde(perturbed_atom, atom_score, t)[0] * atom_sde.dt
        atom_opt = atom_sde.reverse_optimum_step(perturbed_atom, atom_feat, t)

        bond_exp = perturbed_bond - bond_sde.reverse().sde(perturbed_bond, bond_score, t)[0] * bond_sde.dt
        bond_opt = bond_sde.reverse_optimum_step(bond_feat, bond_feat, t)

        # bond loss
        import torch.nn.functional as F
        loss_atom = F.l1_loss(atom_exp, atom_opt)
        loss_bond = F.l1_loss(bond_exp, bond_opt)

        return loss_atom + loss_bond

    return loss_fn



def get_step_fn(config, train):
    """Create a one-step training/evaluation function.

    Args:
        sde: An `sde_lib.SDE` object that represents the forward SDE.
             Tuple (`sde_lib.SDE`, `sde_lib.SDE`) that represents the forward node SDE and edge SDE.
        optimize_fn: An optimization function.
        reduce_mean: If `True`, average the loss across data dimensions.
            Otherwise, sum the loss across data dimensions.
        continuous: `True` indicates that the model is defined to take continuous time steps.
        likelihood_weighting: If `True`, weight the mixture of score matching losses according to
            https://arxiv.org/abs/2101.09258; otherwise, use the weighting recommended by score-sde.

    Returns:
        A one-step function for training or evaluation.
    """

    optimize_fn = loader.optimization_manager(config)

    if config.training.training_strategy == 'MaxLikelihood':
        loss_fn = MaxLikelihood(config, train)
    else:
        loss_fn = DenoisingScoreMatching(config, train)

    def step_fn(state, batch):
        """Running one step of training or evaluation.

        For jax version: This function will undergo `jax.lax.scan` so that multiple steps can be pmapped and
            jit-compiled together for faster execution.

        Args:
            state: A dictionary of training information, containing the score model, optimizer,
                EMA status, and number of optimization steps.
            batch: A mini-batch of training/evaluation data, including min-batch adjacency matrices and mask.

        Returns:
            loss: The average loss value of this state.
        """
        model = state['model']
        if train:
            optimizer = state['optimizer']
            optimizer.zero_grad()
            loss, loss_atom, loss_bond = loss_fn(model, batch)
            loss.backward()
            optimize_fn(optimizer, model.parameters(), step=state['step'])
            state['step'] += 1
            state['ema'].update(model.parameters())
        else:
            with torch.no_grad():
                ema = state['ema']
                ema.store(model.parameters())
                ema.copy_to(model.parameters())
                loss, loss_atom, loss_bond = loss_fn(model, batch)
                ema.restore(model.parameters())

        return loss, loss_atom, loss_bond

    return step_fn
