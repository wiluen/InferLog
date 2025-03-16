import numpy as np
import sklearn.gaussian_process as gp
from scipy.stats import norm
from scipy.optimize import minimize
from env import run,read_performance,get_initial_point
from attmamldrop_bo import MetaLearner
import torch
import random

def expected_improvement(x, model, evaluated_loss,proxy, fast_weights=None,greater_is_better=False, n_params=1):
    """ expected_improvement

    Expected improvement acquisition function.

    Arguments:
    ----------
        x: array-like, shape = [n_samples, n_hyperparams]
            The point for which the expected improvement needs to be computed.
        gaussian_process: GaussianProcessRegressor object.
            Gaussian process trained on previously evaluated hyperparameters.
        evaluated_loss: Numpy array.
            Numpy array that contains the values off the loss function for the previously
            evaluated hyperparameters.
        greater_is_better: Boolean.
            Boolean flag that indicates whether the loss function is to be maximised or minimised.
        n_params: int.
            Dimension of the hyperparameter space.

    """

    x_to_predict = x.reshape(-1, n_params) 
   

    if proxy=='gp':
        mu, sigma = model.predict(x_to_predict, return_std=True) 
    elif proxy=='meta':
        x_to_predict=torch.from_numpy(x_to_predict).float()
        mu, sigma = model.predict(x_to_predict,fast_weights)

    if greater_is_better:
        loss_optimum = np.max(evaluated_loss)
    else:
        loss_optimum = np.min(evaluated_loss)

    scaling_factor = (-1) ** (not greater_is_better)

    with np.errstate(divide='ignore'):
        Z = scaling_factor * (mu - loss_optimum) / sigma
        expected_improvement = scaling_factor * (mu - loss_optimum) * norm.cdf(Z) + sigma * norm.pdf(Z)
        expected_improvement[sigma == 0.0] == 0.0

    return -1 * expected_improvement 


def sample_next_hyperparameter(acquisition_func, gaussian_process, evaluated_loss, greater_is_better=False,
                               bounds=(0, 10), n_restarts=25):
    """ sample_next_hyperparameter

    Proposes the next hyperparameter to sample the loss function for.

    Arguments:
    ----------
        acquisition_func: function.
            Acquisition function to optimise.
        gaussian_process: GaussianProcessRegressor object.
            Gaussian process trained on previously evaluated hyperparameters.
        evaluated_loss: array-like, shape = [n_obs,]
            Numpy array that contains the values off the loss function for the previously
            evaluated hyperparameters.
        greater_is_better: Boolean.
            Boolean flag that indicates whether the loss function is to be maximised or minimised.
        bounds: Tuple.
            Bounds for the L-BFGS optimiser.
        n_restarts: integer.
            Number of times to run the minimiser with different starting points.

    """
    best_x = None
    best_acquisition_value = 1
    n_params = bounds.shape[0]

    for starting_point in np.random.uniform(bounds[:, 0], bounds[:, 1], size=(n_restarts, n_params)):

        res = minimize(fun=acquisition_func,
                       x0=starting_point.reshape(-1),
                       bounds=bounds,
                       method='L-BFGS-B',
                       args=(gaussian_process, evaluated_loss, greater_is_better, n_params))

        if res.fun < best_acquisition_value:
            best_acquisition_value = res.fun
            best_x = res.x

    return best_x


def bayesian_optimisation(n_iters, sample_loss, bounds, task,x0=None, n_pre_samples=5,
                          gp_params=None, random_search=False, alpha=1e-5, epsilon=1e-7,proxy='gp',meta_learner=None):
    """ bayesian_optimisation

    Uses Gaussian Processes to optimise the loss function `sample_loss`.

    Arguments:
    ----------
        n_iters: integer.
            Number of iterations to run the search algorithm.
        sample_loss: function.
            Function to be optimised.
        bounds: array-like, shape = [n_params, 2].
            Lower and upper bounds on the parameters of the function `sample_loss`.
        x0: array-like, shape = [n_pre_samples, n_params].
            Array of initial points to sample the loss function for. If None, randomly
            samples from the loss function.
        n_pre_samples: integer.
            If x0 is None, samples `n_pre_samples` initial points from the loss function.
        gp_params: dictionary.
            Dictionary of parameters to pass on to the underlying Gaussian Process.
        random_search: integer.
            Flag that indicates whether to perform random search or L-BFGS-B optimisation
            over the acquisition function.
        alpha: double.
            Variance of the error term of the GP.
        epsilon: double.
            Precision tolerance for floats.
    """

    x_list = []
    y_list = []

    n_params = bounds.shape[0]

    if x0 is None:
        for params in np.random.uniform(bounds[:, 0], bounds[:, 1], (n_pre_samples, bounds.shape[0])):
            x_list.append(params)
            y_list.append(sample_loss(params,task))
    else:
        for params in x0:
            x_list.append(params)
            y_list.append(sample_loss(params,task))

    xp = np.array(x_list)
    yp = np.array(y_list)

    # Create the GP
    if proxy=='gp':
        fast_weights=None
        if gp_params is not None:
            model = gp.GaussianProcessRegressor(**gp_params)
        else:
            kernel = gp.kernels.Matern()
            model = gp.GaussianProcessRegressor(kernel=kernel,
                                                alpha=alpha,
                                                n_restarts_optimizer=10,
                                                normalize_y=True)
    elif proxy=='meta':
        model=meta_learner
        state_dict = torch.load(f'model/attmaml_{task}.pth')
        model.net.load_state_dict(state_dict)
        fast_weights = list(state_dict.values())[:6]

    for iter in range(n_iters):
        if proxy=='gp':
            model.fit(xp, yp)
        elif proxy=='meta':
             #MAML predict
            xp_tensor=torch.tensor(xp).float()
            yp_tensor=torch.tensor(yp).float()
           
            if iter==0:
                for param in fast_weights:
                    param.requires_grad = True
            else:
                fast_weights = [param.clone().requires_grad_() for param in fast_weights]
            fast_weights=model.finetune(xp_tensor,yp_tensor,fast_weights)

        if random_search:
            x_random = np.random.uniform(bounds[:, 0], bounds[:, 1], size=(random_search, n_params))
            ei = -1 * expected_improvement(x_random, model, yp, proxy,fast_weights,greater_is_better=False, n_params=n_params)
            next_sample = x_random[np.argmax(ei), :]
        else:
            next_sample = sample_next_hyperparameter(expected_improvement, model, yp, greater_is_better=True, bounds=bounds, n_restarts=100)
        if np.any(np.abs(next_sample - xp) <= epsilon):
            next_sample = np.random.uniform(bounds[:, 0], bounds[:, 1], bounds.shape[0])

        cv_score = sample_loss(next_sample,task)

        x_list.append(next_sample)
        y_list.append(cv_score)

        xp = np.array(x_list)
        yp = np.array(y_list)
        print(f'run {iter},conf {next_sample}, result is {cv_score}')

    return xp, yp

def target_function(params,dataset):
    """target object"""
    conf1=int(params[0])
    conf2=int(params[1])
    conf3=int(params[2])
    run(conf1,conf2,conf3,dataset)
    target=read_performance(dataset)
    return float(target)

def main():
    n_iters=15
    task='HPC'
    method='meta' #or gp
    bounds_dict ={'n_token_bounds': (0, 15), 'n_seq_bounds': (0, 7),'n_delay_bounds': (0, 5)}
    pbounds=np.array(list(bounds_dict.values()))
    init_points=get_initial_point(task)
    meta=MetaLearner()
    if method=='meta':
        list_x,list_y=bayesian_optimisation(n_iters, target_function, bounds=pbounds, task=task,x0=init_points, n_pre_samples=5,
                        gp_params=None, random_search=100, alpha=1e-5, epsilon=1e-7,proxy='meta',meta_learner=meta)
    else:
        list_x,list_y=bayesian_optimisation(n_iters, target_function, bounds=pbounds, task=task,x0=None, n_pre_samples=1,
                        gp_params=None, random_search=100, alpha=1e-5, epsilon=1e-7,proxy='gp',meta_learner=meta)
    for (x,y) in zip(list_x,list_y):
        print(x,y)

if __name__=='__main__':
    main()