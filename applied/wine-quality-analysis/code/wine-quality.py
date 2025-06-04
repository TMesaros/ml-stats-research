import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pymc as pm
import arviz as az
import numpy as np
from scipy.optimize import minimize
import re

if __name__ == '__main__':
    np.random.seed(42)

    # Load and prepare dataset
    data = pd.read_csv('winequality-red.csv')
    data_sample = data.sample(n=150, random_state=42)

    predictors = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
                  'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
                  'pH', 'sulphates', 'alcohol']

    # Save mean, std, min, max for later
    means = data_sample[predictors].mean()
    stds = data_sample[predictors].std()
    min_values = data_sample[predictors].min()
    max_values = data_sample[predictors].max()

    # Standardize predictors
    data_sample[predictors] = (data_sample[predictors] - means) / stds

    plt.figure(figsize=(10, 8))
    sns.heatmap(data_sample[predictors + ['quality']].corr(), annot=True, fmt=".2f", cmap='coolwarm')
    plt.title('Correlation Matrix')
    plt.show()


    # Model 1: Weak priors
    with pm.Model() as model_weak:
        coefs = pm.Normal('coefs', mu=0, sigma=1, shape=len(predictors))
        intercept = pm.Normal('intercept', mu=0, sigma=1)
        sigma = pm.HalfCauchy('sigma', beta=1)
        mu = intercept + pm.math.dot(data_sample[predictors], coefs)
        quality = pm.Normal('quality', mu=mu, sigma=sigma, observed=data_sample['quality'])
        trace_weak = pm.sample(2000, tune=1000, chains=2, target_accept=0.95, return_inferencedata=True, idata_kwargs={"log_likelihood": True})

    # Model 2: Laplace + Uniform priors
    with pm.Model() as model_laplace:
        coefs = []
        for i, pred in enumerate(predictors):
            if pred == 'pH':
                coef = pm.Uniform(f'coefs[{i}]', lower=-2, upper=2)
            elif pred == 'alcohol':
                coef = pm.Uniform(f'coefs[{i}]', lower=0, upper=3)
            else:
                coef = pm.Laplace(f'coefs[{i}]', mu=0, b=1)
            coefs.append(coef)

        intercept = pm.Normal('intercept', mu=5, sigma=2)
        sigma = pm.HalfCauchy('sigma', beta=1)
        mu = intercept + pm.math.dot(data_sample[predictors], pm.math.stack(coefs))
        quality = pm.Normal('quality', mu=mu, sigma=sigma, observed=data_sample['quality'])
        trace_laplace = pm.sample(2000, tune=1000, chains=2, target_accept=0.95, return_inferencedata=True, idata_kwargs={"log_likelihood": True})
    
    # Plot trace plot
    print("\nTrace Summary - Weakly Informative Priors:")
    weak_summary = az.summary(trace_weak, hdi_prob=0.95, var_names=["intercept", "coefs"])
    print(weak_summary[['mean', 'sd', 'hdi_2.5%', 'hdi_97.5%', 'r_hat']])
    az.plot_trace(trace_weak, var_names=["intercept", "coefs"])
    plt.suptitle("Trace Plots - Weakly Informative Priors")
    plt.tight_layout()
    plt.show()

    print("\nTrace Summary - Laplace & Uniform Priors:")
    laplace_summary = az.summary(trace_laplace, hdi_prob=0.95)
    print(laplace_summary[['mean', 'sd', 'hdi_2.5%', 'hdi_97.5%', 'r_hat']])
    az.plot_trace(trace_laplace)
    plt.suptitle("Trace Plots - Laplace Priors (Coefficients)")
    plt.tight_layout()
    plt.show()
    
    # Compute LOO for both models
    loo_weak = az.loo(trace_weak)
    loo_laplace = az.loo(trace_laplace)

    print("LOO - Weak Priors:")
    print(loo_weak)

    print("\nLOO - Laplace & Uniform Priors:")
    print(loo_laplace)

    # Predict values to reach quality = 10 
    def predict_for_quality(target_quality, trace, predictors, min_values, max_values):
    # Extract the coefficients correctly by indexing
        coefs = [trace.posterior[f'coefs[{i}]'].mean(dim=("chain", "draw")).values for i in range(len(predictors))]
        intercept = trace.posterior['intercept'].mean(dim=("chain", "draw")).values.item()

        def objective(x):
            return (intercept + np.dot(x, coefs) - target_quality) ** 2

        initial_guess = np.zeros(len(predictors))

        # Get standardized bounds
        min_vals_std = ((min_values - means) / stds).values
        max_vals_std = ((max_values - means) / stds).values
        bounds = list(zip(min_vals_std, max_vals_std))

        result = minimize(objective, x0=initial_guess, bounds=bounds)
        if not result.success:
            raise ValueError("Optimization failed")

        standardized_solution = result.x
        original_scale = standardized_solution * stds.values + means.values

        return pd.DataFrame({
            'Predictor': predictors,
            'Standardized Value': standardized_solution,
            'Original Scale': original_scale
        })


    # Run prediction on laplce model
    laplace_predictions = predict_for_quality(10, trace_laplace, predictors, min_values, max_values)
    print("\nEstimated predictor values to achieve wine quality of 10 (Laplace model):")
    print(laplace_predictions.round(3))

    # Verify predicted quality using posterior predictive samples
    def verify_prediction(opt_df, trace, means, stds):
        # Extract the original-scale values in the correct order
        x_original = opt_df['Original Scale'].values

        # Standardize using training data
        x_standardized = (x_original - means.values) / stds.values
        x_standardized = x_standardized.reshape(1, -1)

        # Extract posterior samples for each coefficient
        coefs_samples = [trace.posterior[f'coefs[{i}]'].stack(samples=("chain", "draw")).values for i in range(len(predictors))]
        intercept_samples = trace.posterior['intercept'].stack(samples=("chain", "draw")).values

        # Stack all coefficient samples together (after being flattened)
        coefs_samples = np.vstack(coefs_samples).T 
        
        # Predict using posterior samples
        mu_samples = intercept_samples + np.dot(coefs_samples, x_standardized.T).flatten()

        print("\nPosterior predictions for optimized input values:")
        print(f"Mean predicted quality: {mu_samples.mean():.2f}")
        print(f"Standard deviation: {mu_samples.std():.2f}")
        print(f"95% credible interval: ({np.percentile(mu_samples, 2.5):.2f}, {np.percentile(mu_samples, 97.5):.2f})")


    verify_prediction(laplace_predictions, trace_laplace, means, stds)

