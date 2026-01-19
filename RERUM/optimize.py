"""
Optuna-based Hyperparameter Optimization for Dragonnet models.
This module provides a class to optimize hyperparameters using Optuna,
testing across multiple seeds for robust results.
"""

import optuna
from optuna.trial import Trial
import numpy as np
import torch
import random
import os
from dragonnet import Dragonnet
from metrics import auqc


def seed_everything(seed: int):
    """Set seed for reproducibility across all random number generators."""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class DragonnetOptimizer:
    """
    Optuna-based hyperparameter optimizer for Dragonnet models.
    
    This class optimizes the following parameters:
    - alpha: Weight for propensity loss (0.001 to 0.5)
    - learning_rate: Learning rate for optimizer (1e-5 to 5e-4)
    - weight_decay: L2 regularization strength (1e-6 to 5e-4)
    
    Fixed parameters:
    - shared_hidden: 200
    - outcome_hidden: 100
    
    Each configuration is tested across multiple seeds for robustness.
    
    Parameters
    ----------
    input_dim : int
        Input dimension for covariates
    train_t_loader : DataLoader
        DataLoader for treatment group training data
    train_c_loader : DataLoader
        DataLoader for control group training data
    val_loader : DataLoader
        DataLoader for validation data
    test_loader : DataLoader
        DataLoader for test data (used for final evaluation)
    seeds : list, optional
        List of random seeds to test each configuration (default: [42, 123, 456, 789, 1024])
    n_trials : int, optional
        Number of Optuna trials to run (default: 50)
    epochs : int, optional
        Number of training epochs per trial (default: 100)
    beta : float, optional
        Fixed weight for targeted regularization (default: 0.0)
    response_lambda : float, optional
        Weight for response ranking loss (default: 0.0)
    uplift_lambda : float, optional
        Weight for uplift ranking loss (default: 0.0)
    max_samples : int, optional
        Max samples for ranking loss computation (default: 200)
    early_stop_metric : str, optional
        Metric for early stopping: 'loss' or 'qini' (default: 'qini')
    direction : str, optional
        Optimization direction: 'maximize' for qini, 'minimize' for loss (default: 'maximize')
    verbose : bool, optional
        Whether to print progress during optimization (default: True)
    """
    
    def __init__(
        self,
        input_dim: int,
        train_t_loader,
        train_c_loader,
        val_loader,
        test_loader,
        seeds: list = None,
        n_trials: int = 50,
        epochs: int = 100,
        beta: float = 0.0,  # Fixed parameter (not optimized)
        response_lambda: float = 0.0,
        uplift_lambda: float = 0.0,
        max_samples: int = 200,
        early_stop_metric: str = 'loss',
        direction: str = 'minimize',
        verbose: bool = True
    ):
        self.input_dim = input_dim
        self.train_t_loader = train_t_loader
        self.train_c_loader = train_c_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.seeds = seeds if seeds is not None else [42, 10, 456, 789, 1110]
        self.n_trials = n_trials
        self.epochs = epochs
        self.beta = beta
        self.response_lambda = response_lambda
        self.uplift_lambda = uplift_lambda
        self.max_samples = max_samples
        self.early_stop_metric = early_stop_metric
        self.direction = direction
        self.verbose = verbose        
        # Fixed architecture parameters
        self.shared_hidden = 200
        self.outcome_hidden = 100        
        # Storage for results
        self.study = None
        self.best_params = None
        self.best_score = None
        self.trial_results = []
        
    def _train_and_evaluate(
        self, 
        alpha: float, 
        learning_rate: float, 
        weight_decay: float,
        seed: int
    ) -> tuple:
        """
        Train a Dragonnet model with given hyperparameters and return both loss and qini.
        
        Returns
        -------
        tuple
            (val_loss, val_qini) - Both metrics for selection strategy
        """
        # Set seed for reproducibility
        seed_everything(seed)
        
        # Create model with current hyperparameters (fixed architecture)
        model = Dragonnet(
            input_dim=self.input_dim,
            shared_hidden=self.shared_hidden,  # Fixed: 200
            outcome_hidden=self.outcome_hidden,  # Fixed: 100
            alpha=alpha,
            beta=self.beta,  # Fixed parameter
            epochs=self.epochs,
            learning_rate=learning_rate,  # Optimized
            weight_decay=weight_decay,
            response_lambda=self.response_lambda,
            uplift_lambda=self.uplift_lambda,
            max_samples=self.max_samples,
            early_stop_metric=self.early_stop_metric
        )
        
        # Train the model (suppress output during optimization)
        import sys
        from io import StringIO
        
        # Capture stdout to suppress training output
        old_stdout = sys.stdout
        sys.stdout = StringIO()
        
        try:
            model.fit(self.train_t_loader, self.train_c_loader, self.val_loader)
        finally:
            sys.stdout = old_stdout
        
        # Evaluate on validation set - get BOTH loss and qini
        val_loss = model.validate(self.val_loader, epoch=self.epochs)
        val_qini = model.validate_qini(self.val_loader)
        
        # Clean up GPU memory
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return val_loss, val_qini
    
    def _objective(self, trial: Trial) -> float:
        """
        Optuna objective function.
        
        Suggests hyperparameters, trains models across multiple seeds,
        and returns the mean validation metric.
        """
        # Suggest hyperparameters (only alpha, lr, weight_decay)
        alpha = trial.suggest_float('alpha', 0.001, 0.5, log=True)  # Changed low from 0.0 to 0.001
        learning_rate = trial.suggest_float('learning_rate', 1e-5, 5e-4, log=True)
        weight_decay = trial.suggest_float('weight_decay', 1e-6, 5e-4, log=True)
        
        if self.verbose:
            print(f"\n🔍 Trial {trial.number + 1}/{self.n_trials}")
            print(f"   Params: alpha={alpha:.4f}, lr={learning_rate:.6f}, weight_decay={weight_decay:.6f}")
            print(f"   Fixed: shared_hidden={self.shared_hidden}, outcome_hidden={self.outcome_hidden}")
            print(f"   Testing across {len(self.seeds)} seeds...")
        
        # Train and evaluate across all seeds - track both loss and qini
        seed_losses = []
        seed_qinis = []
        for i, seed in enumerate(self.seeds):
            if self.verbose:
                print(f"   Seed {i+1}/{len(self.seeds)} ({seed})...", end=" ")
            
            val_loss, val_qini = self._train_and_evaluate(
                alpha=alpha,
                learning_rate=learning_rate,
                weight_decay=weight_decay,
                seed=seed
            )
            seed_losses.append(val_loss)
            seed_qinis.append(val_qini)
            
            if self.verbose:
                print(f"Loss: {val_loss:.4f}, Qini: {val_qini:.4f}")
            
            # Report intermediate value for pruning (use loss for optimization)
            trial.report(np.mean(seed_losses), i)
            
            # Handle pruning if enabled
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
        
        # Calculate statistics for both metrics
        mean_loss = np.mean(seed_losses)
        std_loss = np.std(seed_losses)
        mean_qini = np.mean(seed_qinis)
        std_qini = np.std(seed_qinis)
        
        if self.verbose:
            print(f"   📊 Mean Loss: {mean_loss:.4f} ± {std_loss:.4f}")
            print(f"   📊 Mean Qini: {mean_qini:.4f} ± {std_qini:.4f}")
        
        # Store detailed results with both metrics
        self.trial_results.append({
            'trial': trial.number,
            'alpha': alpha,
            'learning_rate': learning_rate,
            'weight_decay': weight_decay,
            'seed_losses': seed_losses,
            'seed_qinis': seed_qinis,
            'mean_loss': mean_loss,
            'std_loss': std_loss,
            'mean_qini': mean_qini,
            'std_qini': std_qini
        })
        
        # Return LOSS for Optuna optimization (minimize)
        return mean_loss
    
    def optimize(self, study_name: str = "dragonnet_optimization", pruner=None) -> dict:
        """
        Run the hyperparameter optimization.
        
        Parameters
        ----------
        study_name : str, optional
            Name for the Optuna study (default: "dragonnet_optimization")
        pruner : optuna.pruners.BasePruner, optional
            Optuna pruner for early trial termination (default: None)
        
        Returns
        -------
        dict
            Dictionary containing:
            - 'best_params': Best hyperparameters found
            - 'best_score': Best validation metric achieved
            - 'study': The Optuna study object
            - 'all_results': List of all trial results
        """
        print("=" * 60)
        print("🚀 Starting Dragonnet Hyperparameter Optimization")
        print("=" * 60)
        print(f"📋 Configuration:")
        print(f"   - Number of trials: {self.n_trials}")
        print(f"   - Seeds per trial: {len(self.seeds)} {self.seeds}")
        print(f"   - Epochs per training: {self.epochs}")
        print(f"   - Optimization direction: {self.direction} (minimize loss)")
        print(f"   - Selection criterion: highest Qini")
        print(f"   - Early stop metric: {self.early_stop_metric}")
        print("=" * 60)
        
        # Create Optuna study - always minimize loss
        self.study = optuna.create_study(
            study_name=study_name,
            direction='minimize',  # Always minimize loss
            pruner=pruner
        )
        
        # Run optimization
        self.study.optimize(
            self._objective, 
            n_trials=self.n_trials,
            show_progress_bar=not self.verbose  # Show progress bar if not verbose
        )
        
        # Get best results BY QINI (not by loss)
        # Find trial with highest mean qini
        best_by_qini = max(self.trial_results, key=lambda x: x['mean_qini'])
        
        self.best_params = {
            'alpha': best_by_qini['alpha'],
            'learning_rate': best_by_qini['learning_rate'],
            'beta': self.beta,
            'weight_decay': best_by_qini['weight_decay'],
            'shared_hidden': self.shared_hidden,  # Fixed: 200
            'outcome_hidden': self.outcome_hidden  # Fixed: 100
        }
        self.best_qini = best_by_qini['mean_qini']
        self.best_loss = best_by_qini['mean_loss']
        
        # Also store Optuna's best (by loss) for comparison
        self.optuna_best_params = self.study.best_params
        self.optuna_best_loss = self.study.best_value
        
        print("\n" + "=" * 60)
        print("🏆 Optimization Complete!")
        print("=" * 60)
        print(f"\n📉 Best by Loss (Optuna):")
        print(f"   Loss: {self.optuna_best_loss:.4f}")
        
        print(f"\n📈 Best by Qini (Selected):")
        print(f"   Qini: {self.best_qini:.4f}")
        print(f"   Loss: {self.best_loss:.4f}")
        print(f"   Trial: {best_by_qini['trial']}")
        print(f"\nBest Parameters (by Qini):")
        print(f"   - alpha: {self.best_params['alpha']:.4f}")
        print(f"   - beta: {self.best_params['beta']:.4f}")
        print(f"   - weight_decay: {self.best_params['weight_decay']:.6f}")
        print(f"   - shared_hidden: {self.best_params['shared_hidden']}")
        print(f"   - outcome_hidden: {self.best_params['outcome_hidden']}")
        print("=" * 60)
        
        return {
            'best_params': self.best_params,
            'best_qini': self.best_qini,
            'best_loss': self.best_loss,
            'optuna_best_params': self.optuna_best_params,
            'optuna_best_loss': self.optuna_best_loss,
            'study': self.study,
            'all_results': self.trial_results
        }
    
    def get_best_model(self, seed: int = 42) -> Dragonnet:
        """
        Create and train a Dragonnet model with the best found hyperparameters.
        
        Parameters
        ----------
        seed : int, optional
            Random seed for training (default: 42)
        
        Returns
        -------
        Dragonnet
            Trained Dragonnet model with best hyperparameters
        """
        if self.best_params is None:
            raise ValueError("No optimization has been run yet. Call optimize() first.")
        
        print(f"🔧 Training best model with seed={seed}...")
        seed_everything(seed)
        
        model = Dragonnet(
            input_dim=self.input_dim,
            shared_hidden=self.best_params['shared_hidden'],
            outcome_hidden=self.best_params['outcome_hidden'],
            alpha=self.best_params['alpha'],
            beta=self.best_params['beta'],
            epochs=self.epochs,
            learning_rate=self.best_params['learning_rate'],
            weight_decay=self.best_params['weight_decay'],
            response_lambda=self.response_lambda,
            uplift_lambda=self.uplift_lambda,
            max_samples=self.max_samples,
            early_stop_metric=self.early_stop_metric
        )
        
        model.fit(self.train_t_loader, self.train_c_loader, self.val_loader)
        
        return model
    
    def evaluate_best_on_test(self, model: Dragonnet = None, seed: int = 42) -> dict:
        """
        Evaluate the best model on the test set.
        
        Parameters
        ----------
        model : Dragonnet, optional
            Pre-trained model to evaluate. If None, trains a new one.
        seed : int, optional
            Random seed if training a new model (default: 42)
        
        Returns
        -------
        dict
            Dictionary with test metrics
        """
        if model is None:
            model = self.get_best_model(seed=seed)
        
        # Evaluate on test set
        from ziln import zero_inflated_lognormal_pred
        
        model.model.eval()
        y_true_list = []
        t_true_list = []
        uplift_list = []
        
        device = model.device
        
        with torch.no_grad():
            for x, t, y in self.test_loader:
                x = x.to(device)
                y0_pred, y1_pred, t_pred, eps = model.model(x)
                
                # Convert ZILN predictions
                y0_pred = zero_inflated_lognormal_pred(y0_pred)
                y1_pred = zero_inflated_lognormal_pred(y1_pred)
                
                # Calculate uplift
                uplift = (y1_pred - y0_pred).cpu().numpy()
                
                y_true_list.extend(y.cpu().numpy())
                t_true_list.extend(t.cpu().numpy())
                uplift_list.extend(uplift)
        
        # Calculate Qini score
        test_qini = auqc(
            y_true=np.array(y_true_list),
            t_true=np.array(t_true_list),
            uplift_pred=np.array(uplift_list),
            bins=50,
            plot=False
        )
        
        print(f"\n📊 Test Set Evaluation:")
        print(f"   Test Qini Score: {test_qini:.4f}")
        
        return {
            'test_qini': test_qini,
            'model': model
        }
    
    def plot_optimization_history(self):
        """Plot the optimization history using Optuna's visualization."""
        if self.study is None:
            raise ValueError("No optimization has been run yet. Call optimize() first.")
        
        try:
            import matplotlib.pyplot as plt
            
            # Plot optimization history
            fig1 = optuna.visualization.matplotlib.plot_optimization_history(self.study)
            plt.title("Optimization History")
            plt.tight_layout()
            plt.show()
            
            # Plot parameter importances
            fig2 = optuna.visualization.matplotlib.plot_param_importances(self.study)
            plt.title("Hyperparameter Importances")
            plt.tight_layout()
            plt.show()
            
            # Plot parallel coordinate
            fig3 = optuna.visualization.matplotlib.plot_parallel_coordinate(self.study)
            plt.title("Parallel Coordinate Plot")
            plt.tight_layout()
            plt.show()
            
        except ImportError:
            print("⚠️ Matplotlib visualization requires: pip install optuna[visualization]")
            print("Falling back to text summary...")
            self.print_summary()
    
    def print_summary(self, sort_by='qini'):
        """
        Print a summary of all trials.
        
        Parameters
        ----------
        sort_by : str
            Sort by 'qini' (descending) or 'loss' (ascending)
        """
        if not self.trial_results:
            print("No trials have been run yet.")
            return
        
        print("\n" + "=" * 90)
        print("📋 Optimization Summary")
        print("=" * 90)
        print(f"Fixed Architecture: shared_hidden={self.shared_hidden}, outcome_hidden={self.outcome_hidden}")
        print("=" * 90)
        
        # Sort by specified metric
        if sort_by == 'qini':
            sorted_results = sorted(self.trial_results, key=lambda x: x['mean_qini'], reverse=True)
            print("📈 Sorted by Qini (highest first)")
        else:
            sorted_results = sorted(self.trial_results, key=lambda x: x['mean_loss'])
            print("📉 Sorted by Loss (lowest first)")
        
        print(f"\n{'Trial':<7} {'Alpha':<9} {'LR':<11} {'WD':<11} {'Loss':<10} {'Qini':<10}")
        print("-" * 90)
        
        for r in sorted_results[:10]:  # Show top 10
            print(f"{r['trial']:<7} {r['alpha']:<9.5f} {r['learning_rate']:<11.6f} {r['weight_decay']:<11.6f} "
                  f"{r['mean_loss']:<10.4f} {r['mean_qini']:<10.4f}")
        
        print("=" * 90)
        
        if hasattr(self, 'best_params') and self.best_params:
            print(f"\n🏆 Selected Best (by Qini): Trial with Qini={self.best_qini:.4f}, Loss={self.best_loss:.4f}")


def quick_optimize(
    input_dim: int,
    train_t_loader,
    train_c_loader,
    val_loader,
    test_loader,
    n_trials: int = 30,
    seeds: list = None,
    **kwargs
) -> dict:
    """
    Quick function to run hyperparameter optimization.
    
    This is a convenience wrapper around DragonnetOptimizer for quick usage.
    
    Parameters
    ----------
    input_dim : int
        Input dimension for covariates
    train_t_loader : DataLoader
        DataLoader for treatment group training data
    train_c_loader : DataLoader
        DataLoader for control group training data
    val_loader : DataLoader
        DataLoader for validation data
    test_loader : DataLoader
        DataLoader for test data
    n_trials : int, optional
        Number of optimization trials (default: 30)
    seeds : list, optional
        List of random seeds (default: [42, 123, 456, 789, 1024])
    **kwargs
        Additional arguments passed to DragonnetOptimizer
    
    Returns
    -------
    dict
        Optimization results including best parameters and model
    
    Example
    -------
    >>> from optimize import quick_optimize
    >>> results = quick_optimize(
    ...     input_dim=x_train.shape[1],
    ...     train_t_loader=train_t_loader,
    ...     train_c_loader=train_c_loader,
    ...     val_loader=val_loader,
    ...     test_loader=test_loader,
    ...     n_trials=20
    ... )
    >>> best_model = results['model']
    >>> print(results['best_params'])
    """
    optimizer = DragonnetOptimizer(
        input_dim=input_dim,
        train_t_loader=train_t_loader,
        train_c_loader=train_c_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        n_trials=n_trials,
        seeds=seeds,
        **kwargs
    )
    
    results = optimizer.optimize()
    
    # Train and evaluate best model
    test_results = optimizer.evaluate_best_on_test()
    results['model'] = test_results['model']
    results['test_qini'] = test_results['test_qini']
    
    return results
