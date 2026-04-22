import abc
import torch
import numpy as np

_SDES={}

def register_sde(cls=None, *, name=None):
  """A decorator for registering model classes."""

  def _register(cls):
    local_name = cls.__name__ if name is None else name
    assert local_name not in _SDES, ValueError(f'Already registered sde with name: {local_name}')
    _SDES[local_name] = cls
    return cls

  if cls is None:
    return _register
  else:
    return _register(cls)




class SDE(abc.ABC):
  """SDE abstract class. Functions are designed for a mini-batch of inputs."""

  def __init__(self):
    """Construct an SDE.
    Args:
      N: number of discretization time steps.
    """
    super().__init__()

  @property
  @abc.abstractmethod
  def T(self):
    """End time of the SDE."""
    pass

  @abc.abstractmethod
  def sde(self, x, t):
    pass

  @abc.abstractmethod
  def marginal_prob(self, x, t):
    """Parameters to determine the marginal distribution of the SDE, $p_t(x)$."""
    pass


  def prior_sampling(self, shape):
    return torch.randn(*shape, device='cpu')

  def prior_sampling_sym(self, shape):
    x = torch.randn(*shape, device='cpu').tril(-1)
    return x + x.transpose(-1,-2)


  def discretize(self, x, t):
    """Discretize the SDE in the form: x_{i+1} = x_i + f_i(x_i) + G_i z_i.
    Useful for reverse diffusion sampling and probabiliy flow sampling.
    Defaults to Euler-Maruyama discretization.
    Args:
      x: a torch tensor
      t: a torch float representing the time step (from 0 to `self.T`)
    Returns:
      f, G
    """
    drift, diffusion = self.sde(x, t)
    f = drift * self.dt
    G = diffusion * torch.sqrt(torch.tensor(self.dt, device=t.device))
    return f, G

  def reverse(self, probability_flow=False):
    """Create the reverse-time SDE/ODE.
    Args:
      score_fn: A time-dependent score-based model that takes x and t and returns the score.
      probability_flow: If `True`, create the reverse-time ODE used for probability flow sampling.
    """
    N = self.N
    T = self.T
    sde_fn = self.sde
    discretize_fn = self.discretize

    # -------- Build the class for reverse-time SDE --------
    class RSDE(self.__class__):
      def __init__(self):
        self.N = N
        self.probability_flow = probability_flow

      @property
      def T(self):
        return T

      def sde(self, x, score, t):
        """Create the drift and diffusion functions for the reverse SDE/ODE."""
        drift, diffusion = sde_fn(x, t)

        if score.ndim == 2:
          drift = drift - diffusion[:, None] ** 2 * score * (0.5 if self.probability_flow else 1.)
        elif score.ndim == 3:
          drift = drift - diffusion[:, None, None] ** 2 * score * (0.5 if self.probability_flow else 1.)
        elif score.ndim == 4:
          drift = drift - diffusion[:, None, None, None] ** 2 * score * (0.5 if self.probability_flow else 1.)


          # -------- Set the diffusion function to zero for ODEs. --------
        diffusion = torch.tensor([0.], device=x.device) if self.probability_flow else diffusion
        return drift, diffusion


      def discretize(self, x, score, t):
        """Create discretized iteration rules for the reverse diffusion sampler."""
        f, G = discretize_fn(x, t)

        if score.ndim == 2:
          rev_f = f - G[:, None] ** 2 * score * (0.5 if self.probability_flow else 1.)
        elif score.ndim == 3:
          rev_f = f - G[:, None, None] ** 2 * score * (0.5 if self.probability_flow else 1.)

        rev_G = torch.zeros_like(G) if self.probability_flow else G
        return rev_f, rev_G

    return RSDE()

@register_sde(name='VPSDE')
class VPSDE(SDE):
  def __init__(self, sde_config):
    """Construct a Variance Preserving SDE.
    Args:
      beta_min: value of beta(0)
      beta_max: value of beta(1)
      N: number of discretization steps
    """
    super().__init__()
    self.beta_0 = sde_config.beta_min
    self.beta_1 = sde_config.beta_max
    self.N = sde_config.num_scales
    self.dt = 1. / self.N
    self.schedule = sde_config.schedule



    if self.schedule=="linear":
      self.discrete_betas = torch.linspace(self.beta_0 / self.N, self.beta_1 / self.N, self.N)
    elif self.schedule=="exp":
      t = torch.linspace(self.beta_0 / self.N, self.beta_1 / self.N, self.N)
      self.discrete_betas = torch.exp(t*torch.log(torch.tensor(self.beta_1 - self.beta_0 +1))) -1 + self.beta_0
    elif self.schedule=="cosine":
      t = torch.linspace(self.beta_0 / self.N, self.beta_1 / self.N, self.N)
      self.discrete_betas = torch.cos(torch.tensor(3.14 + t/(3.14/2)))* (self.beta_1 - self.beta_0) + self.beta_0 + 1

    self.alphas = 1. - self.discrete_betas
    self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
    self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
    self.sqrt_1m_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)

  @property
  def T(self):
    return 1

  def sde(self, x, t):

    beta_t = self.beta_0 + t * (self.beta_1 - self.beta_0)
    if x.ndim == 2:
      drift = -0.5 * beta_t[:, None] * x
    elif x.ndim == 3:
      drift = -0.5 * beta_t[:, None, None] * x
    else:
      raise ValueError("Unsupported input shape")
    diffusion = torch.sqrt(beta_t)
    return drift, diffusion


  def marginal_prob(self, x, t):
    """Parameters to determine the marginal distribution of the SDE, $p_t(x)$.
    return mean, std of the perturbation kernel """
    if self.schedule == "linear":
      log_mean_coeff = -0.25 * t ** 2 * (self.beta_1 - self.beta_0) - 0.5 * t * self.beta_0

    elif self.schedule == "exp":
      temp = torch.tensor(self.beta_1 - self.beta_0 + 1).float()
      log_part = torch.log(temp)
      log_mean_coeff = -0.5 * (1 / log_part) * torch.exp(t * log_part) - 0.5 * t * (self.beta_0 - 1)

    elif self.schedule == "cosine":
      log_mean_coeff = -0.5 * (3.14 / 2 * (self.beta_1 - self.beta_0) * (
        torch.sin(t / (3.14 / 2) + 3.14)) + t * (1 + self.beta_0))

    std = torch.sqrt(1. - torch.exp(2. * log_mean_coeff))

    if x.ndim == 2:
      mean = torch.exp(log_mean_coeff[:, None]) * x
    elif x.ndim == 3:
      mean = torch.exp(log_mean_coeff[:, None, None]) * x
    else:
      raise ValueError("Unsupported input shape")

    return mean, std

  def marginal_prob_std(self, t):
    " equal to marginal_prob(x,t)[1]"
    if self.schedule == "linear":
      log_mean_coeff = -0.25 * t ** 2 * (self.beta_1 - self.beta_0) - 0.5 * t * self.beta_0

    elif self.schedule == "exp":
      temp = torch.tensor(self.beta_1 - self.beta_0 + 1).float()
      log_part = torch.log(temp)
      log_mean_coeff = -0.5 * (1 / log_part) * torch.exp(t * log_part) - 0.5 * t * (self.beta_0 - 1)

    elif self.schedule == "cosine":
      log_mean_coeff = -0.5 * (3.14 / 2 * (self.beta_1 - self.beta_0) * (
        torch.sin(t / (3.14 / 2) + 3.14)) + t * (1 + self.beta_0))

    std = torch.sqrt(1. - torch.exp(2. * log_mean_coeff))
    return std




  def prior_logp(self, z):
    shape = z.shape
    N = np.prod(shape[1:])
    logps = -N / 2. * np.log(2 * np.pi) - torch.sum(z ** 2, dim=(1, 2)) / 2.
    return logps

  def discretize(self, x, t):
    """DDPM discretization."""
    timestep = (t * (self.N - 1) / self.T).long()
    beta = self.discrete_betas.to(x.device)[timestep]
    alpha = self.alphas.to(x.device)[timestep]
    sqrt_beta = torch.sqrt(beta)
    if len(x.shape) == 2:
      f = torch.sqrt(alpha)[:, None] * x - x
    elif len(x.shape) == 3:
      f = torch.sqrt(alpha)[:, None, None] * x - x
    else:
      raise ValueError(f"Unsupported shape for discretization: {x.shape}")
    G = sqrt_beta
    return f, G




class VESDE(SDE):
  def __init__(self, sigma_min=0.01, sigma_max=50, N=1000):
    """Construct a Variance Exploding SDE.
    Args:
      sigma_min: smallest sigma.
      sigma_max: largest sigma.
      N: number of discretization steps
    """
    super().__init__(N)
    self.sigma_min = sigma_min
    self.sigma_max = sigma_max
    self.discrete_sigmas = torch.exp(torch.linspace(np.log(self.sigma_min), np.log(self.sigma_max), N))
    self.N = N
    self.dt = 1. / N

  @property
  def T(self):
    return 1

  def sde(self, x, t):
    sigma = self.sigma_min * (self.sigma_max / self.sigma_min) ** t
    drift = torch.zeros_like(x)
    diffusion = sigma * torch.sqrt(torch.tensor(2 * (np.log(self.sigma_max) - np.log(self.sigma_min)),
                                                device=t.device))
    return drift, diffusion

  def marginal_prob(self, x, t):
    std = self.sigma_min * (self.sigma_max / self.sigma_min) ** t
    mean = x
    return mean, std

  def prior_logp(self, z):
    shape = z.shape
    N = np.prod(shape[1:])
    return -N / 2. * np.log(2 * np.pi * self.sigma_max ** 2) - torch.sum(z ** 2, dim=(1, 2, 3)) / (2 * self.sigma_max ** 2)

  def discretize(self, x, t):
    """SMLD(NCSN) discretization."""
    timestep = (t * (self.N - 1) / self.T).long()
    sigma = self.discrete_sigmas.to(t.device)[timestep]
    adjacent_sigma = torch.where(timestep == 0, torch.zeros_like(t),
                                 self.discrete_sigmas[timestep - 1].to(t.device))
    f = torch.zeros_like(x)
    G = torch.sqrt(sigma ** 2 - adjacent_sigma ** 2)
    return f, G


class subVPSDE(SDE):
  def __init__(self, beta_min=0.1, beta_max=20, N=1000):
    """Construct the sub-VP SDE that excels at likelihoods.
    Args:
      beta_min: value of beta(0)
      beta_max: value of beta(1)
      N: number of discretization steps
    """
    super().__init__(N)
    self.beta_0 = beta_min
    self.beta_1 = beta_max
    self.N = N
    self.discrete_betas = torch.linspace(beta_min / N, beta_max / N, N)
    self.alphas = 1. - self.discrete_betas
    self.dt = 1./N

  @property
  def T(self):
    return 1

  def sde(self, x, t):
    beta_t = self.beta_0 + t * (self.beta_1 - self.beta_0)
    drift = -0.5 * beta_t[:, None, None] * x
    discount = 1. - torch.exp(-2 * self.beta_0 * t - (self.beta_1 - self.beta_0) * t ** 2)
    diffusion = torch.sqrt(beta_t * discount)
    return drift, diffusion

  def marginal_prob(self, x, t):
    log_mean_coeff = -0.25 * t ** 2 * (self.beta_1 - self.beta_0) - 0.5 * t * self.beta_0
    mean = torch.exp(log_mean_coeff)[:, None, None] * x
    std = 1 - torch.exp(2. * log_mean_coeff)
    return mean, std

  def marginal_prob_std(self, t):
    " equal to marginal_prob(x,t)[1]"
    log_mean_coeff = -0.25 * t ** 2 * (self.beta_1 - self.beta_0) - 0.5 * t * self.beta_0
    std = 1 - torch.exp(2. * log_mean_coeff)
    return std


  def prior_logp(self, z):
    shape = z.shape
    N = np.prod(shape[1:])
    return -N / 2. * np.log(2 * np.pi) - torch.sum(z ** 2, dim=(1, 2, 3)) / 2.

@register_sde(name='OUSDE')
class OUSDE(SDE):
  """IR-SDE reference implementation."""
  def __init__(self, sde_config, eps=0.005):
    super().__init__()
    self.max_sigma = sde_config.max_sigma
    self.N = sde_config.num_scales
    schedule = sde_config.schedule


    def get_thetas_from_t(timesteps, schedule):

      if schedule == 'constant':
        timesteps = timesteps + 1  # T from 1 to 100
        return torch.ones(timesteps, dtype=torch.float32)
      elif schedule == 'linear':
        timesteps = timesteps + 1  # T from 1 to 100
        scale = 1000 / timesteps
        theta_start = scale * 0.0001
        theta_end = scale * 0.02
        return torch.linspace(theta_start, theta_end, timesteps, dtype=torch.float32)
      elif schedule == 'cosine':
        s = 0.008
        timesteps = timesteps + 2  # for truncating from 1 to -1
        x = torch.linspace(0, timesteps, timesteps + 1, dtype=torch.float32)
        f_t = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2 #alphas_cumprod
        alphas_cumprod = f_t / f_t[0]
        thetas = 1 - alphas_cumprod[1:-1]
        return thetas

    self.thetas = get_thetas_from_t(self.N, schedule)
    self.sigmas = torch.sqrt(2 * self.max_sigma**2 * self.thetas)
    self.thetas_cumsum = torch.cumsum(self.thetas, dim=0) - self.thetas[0]
    # Following the trick in IR-SDE Appendix D.
    self.dt = - (np.log(eps) / self.thetas_cumsum[-1]).cpu().numpy()
    # v_t in Eq. (6).
    self.sigma_bars = torch.sqrt(self.max_sigma**2 * (1 - torch.exp(-2 * self.thetas_cumsum * self.dt)))


  @property
  def T(self):
    return 1

  def set_mu(self, mu):
    self.mu = mu
    self.thetas = self.thetas.to(mu.device)
    self.sigmas = self.sigmas.to(mu.device)
    self.thetas_cumsum = self.thetas_cumsum.to(mu.device)
    self.sigma_bars = self.sigma_bars.to(mu.device)

  def sde(self, x, t):
    """Forward SDE: dx = theta_t (mu - x) dt + sigma_t dw."""
    if x.ndim == 2:
      drift = self.thetas[t, None] * (self.mu - x)
    elif x.ndim == 3:
      drift = self.thetas[t, None, None] * (self.mu - x)
    else:
      raise ValueError('x should be 2 or 3 dimensional')

    diffusion = self.sigmas[t]
    return drift, diffusion

  def marginal_prob(self, x,  t):
    """p_{0t} or p_t; see Eq. (6) of the IR-SDE paper."""
    theta_t_bar = self.thetas_cumsum[t] * self.dt
    if x.ndim == 2:
      mean = self.mu + (x - self.mu) * torch.exp(-theta_t_bar[:,None])
    elif x.ndim == 3:
      mean = self.mu + (x - self.mu) * torch.exp(-theta_t_bar[:, None, None])
    else:
      raise ValueError('x should be 2 or 3 dimensional')

    std = self.sigma_bars[t]
    return mean, std

  def marginal_prob_std(self, t):
    " equal to marginal_prob(x,t)[1]"
    return self.sigma_bars[t]

  def reverse_optium_step(self, xt, x0, t):
    """Optimum x_{t-1}; corresponds to Eq. (14) of the IR-SDE paper."""
    if xt.ndim == 2:
      t_ = t[:, None]
    elif xt.ndim == 3:
      t_ = t[:, None, None]
    else:
      raise ValueError('xt should be 2 or 3 dimensional')
    A = torch.exp(-self.thetas[t_] * self.dt)
    B = torch.exp(-self.thetas_cumsum[t_] * self.dt)
    C = torch.exp(-self.thetas_cumsum[t_ - 1] * self.dt)

    term1 = A * (1 - C ** 2) / (1 - B ** 2)
    term2 = C * (1 - A ** 2) / (1 - B ** 2)

    return term1 * (xt - self.mu) + term2 * (x0 - self.mu) + self.mu

  def discretize(self, x, t):
    """Not implemented for OUSDE; the SDE parent class provides a default discretize."""
    raise NotImplementedError(f"discretize has not be instantiated for OUSDE!")


  def reverse_optimum_step(self, xt, x0, t):
    """Optimum x_{t-1}; corresponds to Eq. (14) of the IR-SDE paper."""
    if xt.ndim == 2:
      t_ = t[:,None]
    elif xt.ndim == 3:
      t_ = t[:,None,None]
    else:
      raise ValueError('xt should be 2 or 3 dimensional')
    A = torch.exp(-self.thetas[t_] * self.dt)
    B = torch.exp(-self.thetas_cumsum[t_] * self.dt)
    C = torch.exp(-self.thetas_cumsum[t_ - 1] * self.dt)

    term1 = A * (1 - C ** 2) / (1 - B ** 2)
    term2 = C * (1 - A ** 2) / (1 - B ** 2)

    return term1 * (xt - self.mu) + term2 * (x0 - self.mu) + self.mu

@register_sde(name="OUBridge")
class OUBridge(SDE):
  """GOUB bridge (IR-SDE family)."""
  def __init__(self, sde_config, eps=0.005):
    super().__init__()
    self.max_sigma = sde_config.max_sigma
    self.N = sde_config.num_scales
    self.dt = 1.0 / self.N
    schedule = sde_config.schedule

    def get_thetas_from_t(timesteps, schedule):

      if schedule == 'constant':
        timesteps = timesteps + 1  # T from 1 to 100
        return torch.ones(timesteps, dtype=torch.float32)
      elif schedule == 'linear':
        timesteps = timesteps + 1  # T from 1 to 100
        scale = 1000 / timesteps
        theta_start = scale * 0.0001
        theta_end = scale * 0.02
        return torch.linspace(theta_start, theta_end, timesteps, dtype=torch.float32)
      elif schedule == 'cosine':
        s = 0.008
        timesteps = timesteps + 2  # for truncating from 1 to -1
        x = torch.linspace(0, timesteps, timesteps + 1, dtype=torch.float32)
        f_t = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2 #alphas_cumprod
        alphas_cumprod = f_t / f_t[0]
        thetas = 1 - alphas_cumprod[1:-1]
        return thetas

    self.thetas = get_thetas_from_t(self.N, schedule)
    self.g = torch.sqrt(2 * self.max_sigma**2 * self.thetas)
    self.thetas_cumsum = torch.cumsum(self.thetas, dim=0) - self.thetas[0]
    # Following the trick in IR-SDE Appendix D.
    self.dt = - (np.log(eps) / self.thetas_cumsum[-1]).cpu().numpy()
    # v_t in Eq. (6).
    self.sigma_bars = torch.sqrt(self.max_sigma**2 * (1 - torch.exp(-2 * self.thetas_cumsum * self.dt)))
    self.sigma_bars_t_T = torch.sqrt(self.max_sigma**2 * (1 - torch.exp(-2 * (self.thetas_cumsum[-1] - self.thetas_cumsum) * self.dt)))

    self.sigma_prime = self.sigma_bars * self.sigma_bars_t_T / self.sigma_bars[-1]

  @property
  def T(self):
    return 1

  def set_mu(self, mu):
    self.mu = mu
    self.thetas = self.thetas.to(mu.device)
    self.g = self.g.to(mu.device)
    self.thetas_cumsum = self.thetas_cumsum.to(mu.device)
    self.sigma_bars = self.sigma_bars.to(mu.device)
    self.sigma_bars_t_T = self.sigma_bars_t_T.to(mu.device)
    self.sigma_prime = self.sigma_prime.to(mu.device)

  def sde(self, x, t):
    """Forward SDE for the OU bridge (GOUB drift + dispersion)."""

    thetas_t_T = torch.exp(2 * (self.thetas_cumsum[t] - self.thetas_cumsum[-1]) * self.dt)
    drift_h =  self.g[t] ** 2 * thetas_t_T / (self.sigma_bars_t_T[t] ** 2).to(self.mu.device)

    drift_h[torch.where(t==self.N)] = 0.0

    if x.ndim == 2:
      drift = (self.thetas[t, None] + drift_h[:, None]) * (self.mu - x)
    elif x.ndim == 3:
      drift = (self.thetas[t, None, None] + drift_h[:, None, None]) * (self.mu - x)
    else:
      raise ValueError('x should be 2 or 3 dimensional')

    diffusion = self.g[t]
    return drift, diffusion


  def marginal_prob(self, x0,  t):
    """p_{0t} or p_t; see Eq. (8) of the GOUB paper."""

    theta_bar = self.thetas_cumsum[t] * self.dt

    # coefficient of x0 in the marginal forward process
    m = torch.exp(-theta_bar) * self.sigma_bars_t_T[t]**2 / (self.sigma_bars[-1]**2).to(self.mu.device)
    # coefficient of xT (mu) in the marginal forward process
    n = (1-torch.exp(-theta_bar)) * self.sigma_bars_t_T[t]**2 / (self.sigma_bars[-1]**2) + \
         torch.exp(-2 * (self.thetas_cumsum[-1] - self.thetas_cumsum[t]) * self.dt) * self.sigma_bars[t]**2 / (self.sigma_bars[-1]**2).to(self.mu.device)

    if x0.ndim == 2:
      mean = m[:, None] * x0 + n[:, None] * self.mu
    elif x0.ndim == 3:
      mean = m[:, None, None] * x0 + n[:, None, None] * self.mu
    else:
      raise ValueError('x0 should be 2 or 3 dimensional')

    std = self.sigma_prime[t]
    return mean, std


  def m(self, t):
    # coefficient of x0 in the marginal forward process
    m = torch.exp(-self.thetas_cumsum[t] * self.dt) * self.sigma_bars_t_T[t] ** 2 / (self.sigma_bars[-1] ** 2).to(self.mu.device)
    return m

  def n(self, t):
    # coefficient of xT (mu) in the marginal forward process
    n = ((1 - torch.exp(-self.thetas_cumsum[t] * self.dt)) * self.sigma_bars_t_T[t] ** 2 + \
         torch.exp(-2 * (self.thetas_cumsum[-1] - self.thetas_cumsum[t]) * self.dt) * self.sigma_bars[t] ** 2) / (
                   self.sigma_bars[-1] ** 2).to(self.mu.device)
    return n


  def marginal_prob_std(self, t):
    " equal to marginal_prob(x,t)[1]"
    return self.sigma_prime[t]





  def reverse_optimum_step(self, xt, x0, t):
    """Optimum x_{t-1}; corresponds to Eq. (13) of the GOUB paper."""
    if xt.ndim == 2:
      t = t[:,None]
    elif xt.ndim == 3:
      t = t[:,None,None]
    else:
      raise ValueError('xt should be 2 or 3 dimensional')

    a = self.m(t) / self.m(t - 1)
    b = self.n(t) - self.n(t - 1) * a
    m_prime_t_1 = self.m(t-1) * x0 + self.n(t-1) * self.mu

    x_t_1 = (self.sigma_prime[t - 1]**2 * a * (xt - b * self.mu) +
             (self.sigma_prime[t]**2 - self.sigma_prime[t-1]**2 * a ** 2) * m_prime_t_1   ) / (self.sigma_prime[t] ** 2)
    return x_t_1




