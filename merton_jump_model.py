"""
Merton Jump-Diffusion Model Implementation
File: merton_jump_model.py

Merton Jump-Diffusion model for housing price dynamics.
Models discontinuous price movements and tail risk using jump-diffusion processes.
"""

import numpy as np
import pandas as pd
from scipy import optimize, stats
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore')


class MertonJumpDiffusion:
    """
    Merton Jump-Diffusion model for housing price dynamics

    Model: dS_t = μS_t dt + σS_t dW_t + S_t(e^J - 1)dN_t
    Where:
    - μ: drift parameter
    - σ: diffusion volatility
    - J: jump size (normally distributed)
    - N_t: Poisson process with intensity λ
    """

    def __init__(self, price_series, dt=1 / 12):
        """
        Initialize Merton Jump-Diffusion model

        Parameters:
        -----------
        price_series : pd.Series
            Time series of asset prices
        dt : float, default 1/12
            Time step (1/12 for monthly data)
        """
        self.price_series = price_series.dropna()
        self.returns = np.log(price_series / price_series.shift(1)).dropna()
        self.dt = dt
        self.n_obs = len(self.returns)
        self.params = None

    def calculate_moments(self):
        """
        Calculate empirical moments for parameter estimation

        Returns:
        --------
        moments : dict
            Dictionary containing empirical moments
        """
        returns = self.returns.values

        moments = {
            'mean': np.mean(returns),
            'variance': np.var(returns),
            'skewness': stats.skew(returns),
            'kurtosis': stats.kurtosis(returns),
            'min_return': np.min(returns),
            'max_return': np.max(returns)
        }

        return moments

    def identify_jumps(self, threshold_std=2.5):
        """
        Identify potential jumps using threshold method

        Parameters:
        -----------
        threshold_std : float, default 2.5
            Threshold in standard deviations for jump identification

        Returns:
        --------
        jump_times : pd.Index
            Times when jumps occurred
        jump_sizes : np.array
            Sizes of identified jumps
        jump_mask : np.array
            Boolean mask indicating jump periods
        """
        returns = self.returns.values

        # Calculate rolling volatility (excluding jumps)
        rolling_vol = pd.Series(returns).rolling(window=12, min_periods=6).std()

        # Identify jumps as returns exceeding threshold
        jump_threshold = threshold_std * rolling_vol
        jump_mask = np.abs(returns) > jump_threshold.values

        jump_times = self.returns.index[jump_mask]
        jump_sizes = returns[jump_mask]

        return jump_times, jump_sizes, jump_mask

    def negative_log_likelihood(self, params):
        """
        Negative log-likelihood function for MLE estimation

        Parameters:
        -----------
        params : list
            [mu, sigma, lambda_jump, mu_jump, sigma_jump]

        Returns:
        --------
        neg_ll : float
            Negative log-likelihood value
        """
        mu, sigma, lambda_jump, mu_jump, sigma_jump = params

        # Parameter constraints
        if sigma <= 0 or lambda_jump < 0 or sigma_jump <= 0:
            return np.inf

        returns = self.returns.values
        log_likelihood = 0

        for i, r in enumerate(returns):
            # Sum over possible number of jumps (k=0,1,2,...)
            # For computational efficiency, limit to reasonable number
            max_jumps = min(10, int(lambda_jump * self.dt * 5))

            likelihood_sum = 0

            for k in range(max_jumps + 1):
                # Poisson probability of k jumps
                poisson_prob = stats.poisson.pmf(k, lambda_jump * self.dt)

                if poisson_prob > 1e-10:  # Avoid numerical issues
                    # Mean and variance of return given k jumps
                    mean_given_k = (mu - 0.5 * sigma ** 2) * self.dt + k * mu_jump
                    var_given_k = sigma ** 2 * self.dt + k * sigma_jump ** 2

                    if var_given_k > 0:
                        # Normal density
                        normal_density = stats.norm.pdf(r, mean_given_k, np.sqrt(var_given_k))
                        likelihood_sum += poisson_prob * normal_density

            if likelihood_sum > 0:
                log_likelihood += np.log(likelihood_sum)
            else:
                return np.inf

        return -log_likelihood

    def estimate_parameters(self, method='MLE'):
        """
        Estimate Merton model parameters

        Parameters:
        -----------
        method : str, default 'MLE'
            Estimation method ('MLE' or 'moments')

        Returns:
        --------
        params : dict
            Dictionary containing estimated parameters
        """
        if method == 'MLE':
            return self._estimate_mle()
        elif method == 'moments':
            return self._estimate_moments()
        else:
            raise ValueError("Method must be 'MLE' or 'moments'")

    def _estimate_mle(self):
        """
        Maximum Likelihood Estimation

        Returns:
        --------
        params : dict
            Dictionary containing MLE parameter estimates
        """
        # Initial parameter guesses based on moments
        moments = self.calculate_moments()

        mu_init = moments['mean'] / self.dt
        sigma_init = np.sqrt(moments['variance'] / self.dt)
        lambda_init = 1.0  # 1 jump per period on average
        mu_jump_init = 0.0  # Zero mean jumps
        sigma_jump_init = sigma_init * 0.5  # Jump volatility smaller than diffusion

        initial_params = [mu_init, sigma_init, lambda_init, mu_jump_init, sigma_jump_init]

        # Parameter bounds
        bounds = [
            (-1, 1),  # mu: drift
            (0.001, 2),  # sigma: diffusion volatility
            (0, 50),  # lambda: jump intensity
            (-0.5, 0.5),  # mu_jump: jump mean
            (0.001, 1)  # sigma_jump: jump volatility
        ]

        # Optimize
        try:
            result = optimize.minimize(
                self.negative_log_likelihood,
                initial_params,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 1000}
            )

            if result.success:
                self.params = {
                    'mu': result.x[0],
                    'sigma': result.x[1],
                    'lambda': result.x[2],
                    'mu_jump': result.x[3],
                    'sigma_jump': result.x[4],
                    'log_likelihood': -result.fun,
                    'aic': 2 * len(result.x) + 2 * result.fun,
                    'success': True
                }
            else:
                self.params = {'success': False, 'message': result.message}

        except Exception as e:
            self.params = {'success': False, 'message': str(e)}

        return self.params

    def _estimate_moments(self):
        """
        Method of moments estimation (simplified approach)

        Returns:
        --------
        params : dict
            Dictionary containing moments-based parameter estimates
        """
        moments = self.calculate_moments()
        jump_times, jump_sizes, jump_mask = self.identify_jumps()

        # Estimate jump parameters from identified jumps
        if len(jump_sizes) > 0:
            lambda_est = len(jump_sizes) / (self.n_obs * self.dt)
            mu_jump_est = np.mean(jump_sizes)
            sigma_jump_est = np.std(jump_sizes) if len(jump_sizes) > 1 else 0.1
        else:
            lambda_est = 0.1
            mu_jump_est = 0.0
            sigma_jump_est = 0.1

        # Estimate diffusion parameters from non-jump returns
        non_jump_returns = self.returns.values[~jump_mask]
        if len(non_jump_returns) > 0:
            mu_est = np.mean(non_jump_returns) / self.dt
            sigma_est = np.sqrt(np.var(non_jump_returns) / self.dt)
        else:
            mu_est = moments['mean'] / self.dt
            sigma_est = np.sqrt(moments['variance'] / self.dt)

        self.params = {
            'mu': mu_est,
            'sigma': sigma_est,
            'lambda': lambda_est,
            'mu_jump': mu_jump_est,
            'sigma_jump': sigma_jump_est,
            'n_jumps_identified': len(jump_sizes),
            'success': True
        }

        return self.params

    def simulate_paths(self, n_paths=1000, n_steps=None, S0=None):
        """
        Simulate price paths using estimated parameters

        Parameters:
        -----------
        n_paths : int, default 1000
            Number of paths to simulate
        n_steps : int, optional
            Number of time steps (default: same as data length)
        S0 : float, optional
            Initial price (default: first observed price)

        Returns:
        --------
        paths : np.array
            Simulated price paths (n_paths x n_steps+1)
        """
        if self.params is None or not self.params.get('success', False):
            raise ValueError("Parameters must be estimated first")

        if n_steps is None:
            n_steps = self.n_obs
        if S0 is None:
            S0 = self.price_series.iloc[0]

        # Extract parameters
        mu = self.params['mu']
        sigma = self.params['sigma']
        lambda_jump = self.params['lambda']
        mu_jump = self.params['mu_jump']
        sigma_jump = self.params['sigma_jump']

        # Initialize paths array
        paths = np.zeros((n_paths, n_steps + 1))
        paths[:, 0] = S0

        for i in range(n_paths):
            S = S0
            for t in range(n_steps):
                # Diffusion component
                dW = np.random.normal(0, np.sqrt(self.dt))
                diffusion = (mu - 0.5 * sigma ** 2) * self.dt + sigma * dW

                # Jump component
                n_jumps = np.random.poisson(lambda_jump * self.dt)
                jump_component = 0

                if n_jumps > 0:
                    jump_sizes = np.random.normal(mu_jump, sigma_jump, n_jumps)
                    jump_component = np.sum(jump_sizes)

                # Update price
                S = S * np.exp(diffusion + jump_component)
                paths[i, t + 1] = S

        return paths

    def calculate_risk_metrics(self, confidence_level=0.05, horizon_months=12):
        """
        Calculate risk metrics using the jump-diffusion model

        Parameters:
        -----------
        confidence_level : float, default 0.05
            Confidence level for VaR calculation
        horizon_months : int, default 12
            Risk horizon in months

        Returns:
        --------
        risk_metrics : dict
            Dictionary containing various risk metrics
        """
        if self.params is None or not self.params.get('success', False):
            raise ValueError("Parameters must be estimated first")

        # Simulate many paths for risk calculation
        paths = self.simulate_paths(n_paths=10000, n_steps=horizon_months)

        # Calculate returns from paths
        final_prices = paths[:, -1]
        initial_price = paths[:, 0]
        returns = (final_prices - initial_price) / initial_price

        # Risk metrics
        var_95 = np.percentile(returns, 5)
        var_99 = np.percentile(returns, 1)
        cvar_95 = np.mean(returns[returns <= var_95])

        risk_metrics = {
            'var_95': var_95,
            'var_99': var_99,
            'cvar_95': cvar_95,
            'expected_return': np.mean(returns),
            'volatility': np.std(returns),
            'prob_large_decline': np.mean(returns < -0.2),  # Prob of >20% decline
            'max_drawdown': np.min(returns),
            'skewness': stats.skew(returns),
            'kurtosis': stats.kurtosis(returns)
        }

        return risk_metrics

    def plot_results(self, n_sim_paths=50, figsize=(15, 10)):
        """
        Plot model results and diagnostics

        Parameters:
        -----------
        n_sim_paths : int, default 50
            Number of simulated paths to plot
        figsize : tuple, default (15, 10)
            Figure size

        Returns:
        --------
        fig : matplotlib.figure.Figure
            Figure object containing plots
        """
        if self.params is None or not self.params.get('success', False):
            raise ValueError("Parameters must be estimated first")

        fig, axes = plt.subplots(2, 2, figsize=figsize)

        # 1. Original price series with jumps identified
        jump_times, jump_sizes, jump_mask = self.identify_jumps()

        axes[0, 0].plot(self.price_series.index, self.price_series.values,
                        'b-', label='Observed Prices', linewidth=1.5)
        if len(jump_times) > 0:
            axes[0, 0].scatter(jump_times, self.price_series.loc[jump_times],
                               color='red', s=50, label='Identified Jumps', zorder=5, alpha=0.8)
        axes[0, 0].set_title('Price Series with Jump Identification')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # 2. Returns with jumps
        axes[0, 1].plot(self.returns.index, self.returns.values, 'b-', alpha=0.7, linewidth=0.8)
        if len(jump_times) > 0:
            jump_returns = self.returns.loc[jump_times]
            axes[0, 1].scatter(jump_times, jump_returns,
                               color='red', s=50, zorder=5, alpha=0.8)
        axes[0, 1].set_title('Returns with Jump Events')
        axes[0, 1].set_ylabel('Returns')
        axes[0, 1].grid(True, alpha=0.3)

        # 3. Simulated paths
        sim_paths = self.simulate_paths(n_paths=n_sim_paths,
                                        n_steps=len(self.price_series))

        time_index = np.arange(len(self.price_series))
        for i in range(min(n_sim_paths, 20)):  # Plot subset for clarity
            axes[1, 0].plot(time_index, sim_paths[i, :len(self.price_series)],
                            'gray', alpha=0.3, linewidth=0.5)

        axes[1, 0].plot(time_index, self.price_series.values, 'b-',
                        linewidth=2, label='Observed')
        axes[1, 0].set_title('Simulated vs Observed Paths')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # 4. Return distribution comparison
        sim_returns = []
        for path in sim_paths[:1000]:  # Use subset for distribution
            path_returns = np.diff(np.log(path))
            sim_returns.extend(path_returns)

        axes[1, 1].hist(self.returns.values, bins=50, alpha=0.7,
                        density=True, label='Observed', color='blue')
        axes[1, 1].hist(sim_returns, bins=50, alpha=0.7,
                        density=True, label='Simulated', color='red')
        axes[1, 1].set_title('Return Distribution Comparison')
        axes[1, 1].set_xlabel('Returns')
        axes[1, 1].set_ylabel('Density')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def summary(self):
        """
        Print comprehensive model summary
        """
        if self.params is None:
            raise ValueError("Parameters must be estimated first")

        print("Merton Jump-Diffusion Model Results")
        print("=" * 40)

        if self.params.get('success', False):
            print("Parameter Estimates:")
            print(f"  Drift (μ): {self.params['mu']:.6f}")
            print(f"  Diffusion volatility (σ): {self.params['sigma']:.6f}")
            print(f"  Jump intensity (λ): {self.params['lambda']:.6f} jumps/year")
            print(f"  Jump mean (μ_J): {self.params['mu_jump']:.6f}")
            print(f"  Jump volatility (σ_J): {self.params['sigma_jump']:.6f}")

            if 'log_likelihood' in self.params:
                print(f"\nModel Fit:")
                print(f"  Log-likelihood: {self.params['log_likelihood']:.4f}")
                print(f"  AIC: {self.params['aic']:.4f}")

            # Jump analysis
            jump_times, jump_sizes, _ = self.identify_jumps()
            print(f"\nJump Analysis:")
            print(f"  Jumps identified: {len(jump_times)}")
            if len(jump_times) > 0:
                print(f"  Average jump size: {np.mean(jump_sizes):.4f}")
                print(f"  Jump frequency: {len(jump_times) / self.n_obs:.1%} of observations")
        else:
            print("Parameter estimation failed:")
            print(f"  Error: {self.params.get('message', 'Unknown error')}")


# Usage example function
def fit_merton_model_housing(price_series, estimation_method='MLE'):
    """
    Convenience function to fit Merton Jump-Diffusion model to housing prices

    Parameters:
    -----------
    price_series : pd.Series
        Housing price series
    estimation_method : str, default 'MLE'
        Estimation method ('MLE' or 'moments')

    Returns:
    --------
    model : MertonJumpDiffusion
        Fitted Merton Jump-Diffusion model instance
    """
    # Initialize model
    merton_model = MertonJumpDiffusion(price_series)

    # Estimate parameters
    params = merton_model.estimate_parameters(method=estimation_method)

    # Calculate risk metrics if estimation successful
    if params.get('success', False):
        risk_metrics = merton_model.calculate_risk_metrics()
        print("\nRisk Metrics (1-year horizon):")
        for metric, value in risk_metrics.items():
            print(f"  {metric}: {value:.4f}")

    # Print summary
    merton_model.summary()

    return merton_model


if __name__ == "__main__":
    # Example usage
    print("Merton Jump-Diffusion Model Implementation")
    print("This module provides jump-diffusion modeling capabilities for asset prices")
    print("Import this module and use MertonJumpDiffusion class or fit_merton_model_housing function")