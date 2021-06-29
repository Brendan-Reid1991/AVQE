import random
import numpy as np
from numpy import pi, exp, cos, log
import warnings
import pickle


class Alpha_VQE():
    """
    This class follows the original alpha-VQE paper
    arxiv:1802.00171

    Inputs
    ------
    REQUIRED:
        phi : float 
            phase value to estimate
        nSamples : int
            number of samples to perform 
            rejection sampling on

    OPTIONAL:
        accuracy: float : default = 0.005
            accuracy in result, exit criteria
        alpha : float : default = 0
            alpha parameter, 0 < alpha < 1
        sigma : float : default = pi/4
            initial variance
        max_shots : int : default = 10000
            maximum number of runs to do
        rescaled : int : default = 0
            two similar but different
            rejection sampling methods
            Ours is rescaled = 0

    Outputs
    -------
        error : float
            Absolute error between real and estimated
        run : int
            Number of shots taken

    Raises
    -------
    WARNINGS for poorly chosen values of accuracy, nSamples and sigma
    VALUEERROR for alpha <0 or >1
    """

    def __init__(self, phi, nSamples, accuracy=10**-3, alpha=0, sigma=pi/4, max_shots=10**4, rescaled=0):
        self.phi = phi
        self.nSamples = nSamples
        self.accuracy = accuracy
        self.alpha = alpha
        self.sigma = sigma
        self.max_shots = max_shots
        self.rescaled = rescaled

        if self.rescaled not in [0, 1]:
            raise ValueError("Rescaled keyword for rejection sampling" +
                             "process must be 0 or 1")

        if self.alpha < 0 or self.alpha > 1:
            raise ValueError("Alpha must lie between 0 and 1")

        if self.nSamples > 10**4:
            warnings.warn(f"Number of samples {nSamples} is prohibitively large" +
                          "for rejection sampling, consider reducing it.")
        if self.accuracy < 10**-5:
            warnings.warn(f"Accuracy goal set extremely low;" +
                          " will likely hit max_runs before convergence.")

        if self.sigma > 2*pi:
            warnings.warn(f"Initial variance is high," +
                          " will likely hit max_runs before convergence.")

    def get_max_shots(self):
        """Returns Eqn1 from the paper;
        maximum number of shots needed
        """
        if self.alpha < 1:
            n_max = 2/(1 - self.alpha) * \
                (1 / (self.accuracy**(2*(1 - self.alpha))) - 1)
        else:
            n_max = 4*log(1/self.accuracy)

        return(n_max)

    def probability(self, measurement_result, M, theta, phi):
        "Outcome probability, Eqn4"
        return(
            1/2 + (1 - 2*measurement_result) * cos(M * (theta - phi))/2
        )

    def update_prior(self,
                     M, theta, measurement_result, prior_samples
                     ):
        accepted = []

        for s in prior_samples:
            p = self.probability(measurement_result, M, theta, s)
            if p > random.uniform(0, 1):
                accepted.append(s)

        if len(accepted) < 2:
            mu, sigma = np.mean(prior_samples), (1 + 0.1)*np.std(prior_samples)
        else:
            mu, sigma = np.mean(accepted), np.std(accepted)

        return(mu, sigma)

    def update_prior_rescaled(self,
                                  M, theta, measurement_result, prior_samples
                                  ):
        accepted = []

        max_prob = 0
        for s in prior_samples:
            p = self.probability(measurement_result, M, theta, s)
            if p > max_prob:
                max_prob = p

        for s in prior_samples:
            p = self.probability(measurement_result, M, theta, s)
            if random.uniform(0, 1) < p/max_prob:
                accepted.append(s)

        if len(accepted) < 2:
            mu, sigma = np.mean(prior_samples), (1 + 0.1)*np.std(prior_samples)
        else:
            mu, sigma = np.mean(accepted), np.std(accepted)

        return(mu, sigma)

    def estimate_phase(self):
        mu = random.uniform(-pi, pi)

        run = 0

        while round(self.sigma, 5) > self.accuracy:
            M = max(1, int(round(1 / self.sigma**self.alpha)))
            theta = mu - self.sigma

            prior_samples = np.random.normal(mu, self.sigma, self.nSamples)

            prob_0 = self.probability(0, M, theta, self.phi)

            if random.uniform(0, 1) < prob_0:
                measurement_result = 0
            else:
                measurement_result = 1

            if self.rescaled == 0:
                mu, sigma = self.update_prior(
                    M, theta, measurement_result, prior_samples)
            else:
                mu, sigma = self.update_prior_rescaled(
                    M, theta, measurement_result, prior_samples)

            self.sigma = sigma

            run += 1
            if run > self.max_shots:
                # print(
                # f"Maximum number of runs {self.max_shots} reached; exiting routine.")
                break

        estimated = abs(cos(mu/2))

        true = abs(cos(self.phi/2))

        error = abs(estimated - true)

        return(error, run)
        # return(
        #     f"  Value estimated: {estimated:.5f}\n  True value: {true:.5f}"
        #     + f"\n  Error: {error:.5f}\n  Number of runs: {run}"
        # )
