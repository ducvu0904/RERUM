"""
Optuna Hyperparameter Optimization for Tarnet Model
Optimizes: uplift_lambda and response_lambda
"""

import optuna
from optuna.logging import get_logger
import torch
import numpy as np
import random
import os
import logging
from tarnet import Tarnet

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def seed_everything(seed):
    """
    Set random seed for reproducibility
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_and_evaluate_single_seed(
    seed,
    train_loader,
    val_loader,
    input_dim,
    uplift_lambda,
    response_lambda,
    lr,
    ranking_start_epoch,
    verbose=True
):
    """
    Train model with a single seed and return best validation loss.
    
    Parameters
    ----------
    seed : int
        Random seed for reproducibility
    train_loader : DataLoader
        Training data loader
    val_loader : DataLoader
        Validation data loader
    input_dim : int
        Input dimension for the model
    uplift_lambda : float
        Lambda for uplift ranking loss
    response_lambda : float
        Lambda for response ranking loss
    lr : float
        Learning rate
    ranking_start_epoch : int
        Epoch to start ranking loss
    verbose : bool
        Whether to print progress
        
    Returns
    -------
    float
        Best validation loss
    """
    # Fixed hyperparameters
    epochs = 150
    alpha = 0
    beta = 0
    wd = 1e-4
    early_stop_metric = "loss"
    ema = True
    ema_alpha = 0.05
    patience = 30
    shared_dropout = 0
    outcome_dropout = 0
    shared_hidden = 200
    outcome_hidden = 100
    early_stop_start = 50
    
    # Set seed
    seed_everything(seed)
    
    # Create model
    model = Tarnet(
        input_dim=input_dim,
        epochs=epochs,
        alpha=alpha,
        beta=beta,
        learning_rate=lr,
        weight_decay=wd,
        use_ema=ema,
        ema_alpha=ema_alpha,
        patience=patience,
        shared_hidden=shared_hidden,
        outcome_hidden=outcome_hidden,
        outcome_droupout=outcome_dropout,
        shared_dropout=shared_dropout,
        early_stop_metric=early_stop_metric,
        uplift_ranking=uplift_lambda,
        response_ranking=response_lambda,
        early_stop_start_epoch=early_stop_start,
        ranking_start_epoch=ranking_start_epoch
    )
    
    # Train model (suppress output during optuna search)
    if not verbose:
        import sys
        from io import StringIO
        old_stdout = sys.stdout
        sys.stdout = StringIO()
    
    try:
        model.fit(train_loader, val_loader)
    finally:
        if not verbose:
            sys.stdout = old_stdout
    
    # Return best validation loss from training
    return model.best_loss


def objective(
    trial,
    train_loader,
    val_loader,
    input_dim,
    seeds=[412312, 42, 1874, 902745, 1],
    verbose=True
):
    """
    Optuna objective function.
    
    Parameters
    ----------
    trial : optuna.Trial
        Optuna trial object
    train_loader : DataLoader
        Training data loader
    val_loader : DataLoader
        Validation data loader
    input_dim : int
        Input dimension for the model
    seeds : list
        List of seeds to run for each trial
    verbose : bool
        Whether to print detailed progress
        
    Returns
    -------
    float
        Mean validation loss across all seeds
    """
    # Sample hyperparameters (log scale for wide ranges)
    uplift_lambda = trial.suggest_float("uplift_lambda", 0.1, 100.0, log=True)
    response_lambda = trial.suggest_float("response_lambda", 1e-4, 10.0, log=True)
    lr = trial.suggest_float("lr", 5e-5, 5e-3, log=True)
    ranking_start_epoch = trial.suggest_int("ranking_start_epoch", 0, 30)
    
    if verbose:
        logger.info(f"\n{'='*60}")
        logger.info(f"Trial {trial.number + 1}")
        logger.info(f"  uplift_lambda: {uplift_lambda:.6f}")
        logger.info(f"  response_lambda: {response_lambda:.6f}")
        logger.info(f"  lr: {lr:.6f}")
        logger.info(f"  ranking_start_epoch: {ranking_start_epoch}")
        logger.info(f"{'='*60}")
    
    loss_scores = []
    
    for i, seed in enumerate(seeds):
        if verbose:
            logger.info(f"  Running seed {i+1}/{len(seeds)}: {seed}")
        
        loss_score = train_and_evaluate_single_seed(
            seed=seed,
            train_loader=train_loader,
            val_loader=val_loader,
            input_dim=input_dim,
            uplift_lambda=uplift_lambda,
            response_lambda=response_lambda,
            lr=lr,
            ranking_start_epoch=ranking_start_epoch,
            verbose=False  # Suppress model training output
        )
        
        loss_scores.append(loss_score)
        
        if verbose:
            logger.info(f"    Seed {seed} Loss: {loss_score:.4f}")
    
    mean_loss = np.mean(loss_scores)
    std_loss = np.std(loss_scores)
    
    if verbose:
        logger.info(f"\n  📊 Trial {trial.number + 1} Results:")
        logger.info(f"     Mean Loss: {mean_loss:.4f} ± {std_loss:.4f}")
        logger.info(f"     Individual scores: {[f'{s:.4f}' for s in loss_scores]}")
    
    return mean_loss


def optimize_tarnet(
    train_loader,
    val_loader,
    input_dim,
    n_trials=30,
    seeds=[412312, 42, 1874, 902745, 1],
    verbose=True,
    study_name="tarnet_optimization"
):
    """
    Run Optuna optimization for Tarnet model.
    
    NOTE: Optimization is done on VALIDATION set to avoid data leakage.
    After finding best parameters, evaluate on TEST set separately.
    
    Parameters
    ----------
    train_loader : DataLoader
        Training data loader
    val_loader : DataLoader
        Validation data loader
    input_dim : int
        Input dimension for the model
    n_trials : int
        Number of optimization trials (default: 30)
    seeds : list
        List of seeds to run for each trial (default: [412312, 42, 1874, 902745, 1])
    verbose : bool
        Whether to print detailed progress (default: True)
    study_name : str
        Name of the Optuna study (default: "tarnet_optimization")
        
    Returns
    -------
    optuna.Study
        The completed Optuna study object
        
    Example
    -------
    >>> # Optimize on validation set
    >>> study = optimize_tarnet(
    ...     train_loader=train_loader,
    ...     val_loader=val_loader,
    ...     input_dim=x_men_train_t.shape[1],
    ...     n_trials=30,
    ...     verbose=True
    ... )
    >>> print(f"Best parameters: {study.best_params}")
    >>> print(f"Best Val Loss: {study.best_value}")
    >>> # Then evaluate best params on TEST set separately
    """
    
    print("=" * 70)
    print("🚀 OPTUNA HYPERPARAMETER OPTIMIZATION FOR TARNET")
    print("=" * 70)
    print(f"📋 Configuration:")
    print(f"   - Number of trials: {n_trials}")
    print(f"   - Seeds per trial: {len(seeds)} {seeds}")
    print(f"   - Total model trainings: {n_trials * len(seeds)}")
    print(f"   - Metric to minimize: Loss (on VALIDATION set)")
    print(f"   ⚠️  Using validation set for optimization (no data leakage)")
    print(f"\n📋 Search Space:")
    print(f"   - uplift_lambda: [0.1, 100.0] (log scale)")
    print(f"   - response_lambda: [1e-4, 10.0] (log scale)")
    print(f"   - lr: [5e-5, 5e-3] (log scale)")
    print(f"   - ranking_start_epoch: [0, 30] (integer)")
    print(f"\n📋 Fixed Parameters:")
    print(f"   - epochs: 150")
    print(f"   - wd: 1e-4")
    print(f"   - early_stop_metric: loss")
    print(f"   - ema: True, ema_alpha: 0.05")
    print(f"   - patience: 30")
    print(f"   - shared_hidden: 200, outcome_hidden: 100")
    print(f"   - early_stop_start: 50")
    print("=" * 70)
    
    # Create Optuna study (minimize loss)
    study = optuna.create_study(
        study_name=study_name,
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=42)  # For reproducibility
    )
    
    # Define objective with fixed parameters
    def objective_wrapper(trial):
        return objective(
            trial=trial,
            train_loader=train_loader,
            val_loader=val_loader,
            input_dim=input_dim,
            seeds=seeds,
            verbose=verbose
        )
    
    # Run optimization
    if verbose:
        optuna.logging.set_verbosity(optuna.logging.INFO)
    else:
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    
    study.optimize(
        objective_wrapper,
        n_trials=n_trials,
        show_progress_bar=True,
        gc_after_trial=True  # Garbage collection after each trial
    )
    
    # Print final results
    print("\n" + "=" * 70)
    print("🏆 OPTIMIZATION COMPLETE!")
    print("=" * 70)
    print(f"\n📊 Best Trial: #{study.best_trial.number + 1}")
    print(f"   Best Mean Loss (Validation): {study.best_value:.4f}")
    print(f"   ⚠️  Remember to evaluate on TEST set with best params!")
    print(f"\n🎯 Best Parameters:")
    print(f"   uplift_lambda: {study.best_params['uplift_lambda']:.6f}")
    print(f"   response_lambda: {study.best_params['response_lambda']:.6f}")
    print(f"   lr: {study.best_params['lr']:.6f}")
    print(f"   ranking_start_epoch: {study.best_params['ranking_start_epoch']}")
    
    # Print top 5 trials
    print(f"\n📈 Top 5 Trials:")
    trials_df = study.trials_dataframe()
    trials_df = trials_df.sort_values("value", ascending=True).head(5)
    for idx, row in trials_df.iterrows():
        print(f"   Trial {int(row['number'])+1}: Loss={row['value']:.4f} | "
              f"uplift_λ={row['params_uplift_lambda']:.4f} | "
              f"response_λ={row['params_response_lambda']:.6f} | "
              f"lr={row['params_lr']:.6f} | "
              f"rank_start={int(row['params_ranking_start_epoch'])}")
    
    print("=" * 70)
    
    return study


if __name__ == "__main__":
    print("This module provides Optuna optimization for Tarnet model.")
    print("Import and use optimize_tarnet() function from a notebook.")
    print("\nExample usage:")
    print("  from searching import optimize_tarnet")
    print("  # Optimize on VALIDATION set (no data leakage)")
    print("  study = optimize_tarnet(train_loader, val_loader, input_dim)")
    print("  # Then evaluate best params on TEST set separately")
