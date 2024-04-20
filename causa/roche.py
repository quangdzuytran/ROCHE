import numpy as np
import torch

from torch import nn
from torch.distributions import StudentT
from torch.nn.utils import parameters_to_vector
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm.autonotebook import trange

from causa.hsic_torch import HSIC
from causa.utils import TensorDataLoader


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class HetSpindlyHead(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lin1 = nn.Linear(1, 1, bias=False)
        self.lin2 = nn.Linear(1, 1, bias=False)
        self.lin2.weight.data.fill_(0.0)
        self.lin3 = nn.Linear(1, 1, bias=False)
        self.lin3.weight.data.fill_(0.0)

    def forward(self, input):
        out1 = self.lin1(input[:, 0].unsqueeze(-1))
        out2 = torch.exp(self.lin2(input[:, 1].unsqueeze(-1))) + 1
        out3 = torch.exp(self.lin3(input[:, 2].unsqueeze(-1)))
        return torch.cat([out1, out2, out3], 1)


def build_het_network(in_dim=1):
    return nn.Sequential(
        nn.Linear(in_dim, 100),
        nn.Tanh(),
        nn.Linear(100, 3),
        HetSpindlyHead()
    )


def loss_func(y, f):
    assert f.shape[0] == y.shape[0]
    assert y.ndim == 1
    assert f.shape[1] == 3
    assert all(f[:, 1] >= 1)
    assert all(f[:, 2] >= 0)

    df = f[:, 1] * 2
    scale_sq = f[:, 2] / f[:, 1]
    scale = torch.sqrt(scale_sq)
    loc = f[:, 0]  

    py_x = StudentT(df=df, loc=loc, scale=scale)
    loss = - torch.sum(py_x.log_prob(y))
    constraint = torch.mean(torch.square(df / (df - 2)))

    return loss, constraint


def map_optimization(model,
                     train_loader,
                    #  valid_loader,
                     n_epochs=500,
                     lr=1e-3,
                     lr_min=None,
                     verbose=False):
    if lr_min is None:  # don't decay lr
        lr_min = lr
    device = parameters_to_vector(model.parameters()).device
    N = len(train_loader.dataset)
    
    n_steps = n_epochs * len(train_loader)
    optimizer = Adam(model.parameters(), lr=lr)
    scheduler = CosineAnnealingLR(optimizer, n_steps, eta_min=lr_min)

    losses = list()
    if verbose:
        with trange(1, n_epochs + 1, desc="Training") as pbar:
            for epoch in pbar:
                epoch_loss = 0
                epoch_constraint = 0
                for X, y in train_loader:
                    X, y = X.to(device), y.to(device)
                    optimizer.zero_grad()
                    f = model(X)
                    loss, constraint = loss_func(y, f)
                    loss = loss / N
                    constraint = constraint / N
                    (loss + constraint).backward()
                    optimizer.step()
                    scheduler.step()
                    epoch_loss += loss.cpu().item() / len(train_loader)
                    epoch_constraint += constraint.cpu().item() / len(train_loader)
                losses.append(epoch_loss * N)
                pbar.set_postfix(loss=epoch_loss, constraint=epoch_constraint)
    else:
        for epoch in range(1, n_epochs + 1):
            epoch_loss = 0
            for X, y in train_loader:
                X, y = X.to(device), y.to(device)
                optimizer.zero_grad()
                f = model(X)
                loss, constraint = loss_func(y, f)
                loss = loss / N
                constraint =  constraint / N
                (loss + constraint).backward()
                optimizer.step()
                scheduler.step()
                epoch_loss += loss.cpu().item() / len(train_loader)
            losses.append(epoch_loss * N)
    
    return model, losses


def het_fit_nn(x, y, n_steps=None, seed=711, device='cpu', verbose=False):
    """Fit heteroscedastic noise model with convex estimator using neural network.
    More precisely we fit y = f(x) + g(x) N with N Gaussian noise and return a joint
    function for f and g.

    Returns
    -------
    log_lik : float
        log likelihood of the fit
    f : method
        method that takes vector of x values and returns mean and standard deviation. 
    """
    n_steps = 5000 if n_steps is None else n_steps
    x, y = torch.from_numpy(x).double(), torch.from_numpy(y).double()
    map_kwargs = dict(
        lr=1e-2,
        lr_min=1e-6,
        n_epochs=n_steps,
    )
    loader = TensorDataLoader(
        x.reshape(-1, 1).to(device), y.flatten().to(device), 
        batch_size=len(x),
    )
    set_seed(seed)
    model, losses = map_optimization(
        build_het_network().to(device).double(),
        loader,
        verbose=verbose,
        **map_kwargs
    )

    @torch.no_grad()
    def f(x_):
        x_ = torch.from_numpy(x_[:, np.newaxis]).double().to(device)
        f = model(x_)
        df = f[:, 1] * 2
        scale_sq = f[:, 2] / f[:, 1]
        scale = torch.sqrt(scale_sq * df / (df - 2))
        loc = f[:, 0]
        return loc.squeeze(), scale.squeeze()
    
    log_lik = - np.nanmin(losses) / len(x)
    return log_lik, f


def roche(x, y, independence_test=False, return_function=False, n_steps=None, device='cpu', verbose=False):
    """Robust estimation of causal heteroscedastic noise models (ROCHE) for bivariate pairs. 
    By default, the method returns a score for the x -> y causal direction where above 0
    indicates evidence for it and negative values indicate y -> x.

    Note: data x, y should be standardized or preprocessed in some way.
    
    Parameters
    ----------
    x : np.ndarray
        cause/effect vector 1-dimensional
    y : np.ndarray
        cause/effect vector 1-dimensional
    independence_test : bool, optional
        whether to run subsequent independence test of residuals, by default True
    neural_network : bool, optional
        whether to use neural network heteroscedastic estimator, by default True
    return_function : bool, optional
        whether to return functions to predict mean/std in both directions, by default False
    n_steps : int, optional
        number of epochs to train neural network or steps to optimize convex model
    """
    assert x.ndim == y.ndim == 1, 'x and y have to be 1-dimensional arrays'

    log_lik_forward, f_forward = het_fit_nn(x, y, n_steps, device=device, verbose=verbose)
    log_lik_reverse, f_reverse = het_fit_nn(y, x, n_steps, device=device, verbose=verbose)

    if independence_test:
        x_ = torch.from_numpy(x).double().to(device)
        y_ = torch.from_numpy(y).double().to(device)
        my, sy = f_forward(x)
        indep_forward = HSIC(x_, (y_ - my) / sy)
        mx, sx = f_reverse(y)
        indep_reverse = HSIC(y_, (x_ - mx) / sx)
        score = indep_reverse - indep_forward
    else:
        score = log_lik_forward - log_lik_reverse

    if return_function:
        return score, f_forward, f_reverse
    return score