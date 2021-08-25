import random
import numpy as np
from numpy import pi
from math import cos, sin, log, exp
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
        update : int : default = 1
            Our method is update = 0
            Rescaled method from arxiv:1506.00869
                is update = 1
            Exact is update = 2

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

    def __init__(self, phi, nSamples, 
                accuracy=5*10**-3, 
                alpha=0, 
                sigma=pi/4, 
                max_shots=10**4, 
                update=1
        ):
        
        self.phi = phi
        self.nSamples = nSamples
        self.accuracy = accuracy
        self.alpha = alpha
        self.sigma = sigma
        self.max_shots = max_shots
        self.update = update

        if self.update not in [0, 1, 2]:
            raise ValueError("Update keyword for rejection sampling" +
                             "process must be 0, 1 or 2")

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
        """Simple prior update method
        """

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
        """Prior update method from 
        arxiv:1506.00869. Same as 
        update_prior but 
        probabilities rescaled.
        """
        accepted = []

        max_prob = 0
        for s in prior_samples:
            p = self.probability(measurement_result, M, theta, s)
            if p > max_prob:
                max_prob = p

        for s in prior_samples:
            p = self.probability(measurement_result, M, theta, s)
            if random.uniform(0, 1)*max_prob < p:
                accepted.append(s)

        if len(accepted) < 2:
            mu, sigma = np.mean(prior_samples), (1 + 0.1)*np.std(prior_samples)
        else:
            mu, sigma = np.mean(accepted), np.std(accepted)

        return(mu, sigma)

    def update_exact(self,
                     mu, M, theta, measurement_result
                     ):
        "Exact update of the prior distribution"
        d = measurement_result

        Expectation = mu + ((1-2*d) * M * self.sigma**2 * sin(M*(theta - mu))) / \
            (exp(M**2 * self.sigma**2 / 2) + (1-2*d) * cos(M*(theta-mu)))

        VarNum = 2 * exp(M**2 * self.sigma**2) + (
            2 * (2*d - 1) * exp(M**2 * self.sigma**2 / 2) *
            ((M**2 * self.sigma**2) - 2) * cos(M*(theta - mu))
        ) + (
            (1 - 2*d)**2 * (1 - (2 * M**2 * self.sigma**2) + cos(2*M*(theta - mu)))
        )

        VarDenom = 2 * (
            exp(M**2 * self.sigma**2 / 2) + (1 - 2*d) * cos(M*(theta - mu))
        )**2

        Variance = self.sigma**2 * (VarNum / VarDenom)

        Std = np.sqrt(Variance)

        return(Expectation, Std)

    def estimate_phase(self):
        mu = random.uniform(-pi, pi)

        run = 0
        failed = 0

        while round(self.sigma, 5) > self.accuracy:
            M = max(1, int(round(1 / self.sigma**self.alpha)))
            theta = mu - self.sigma

            prob_0 = self.probability(0, M, theta, self.phi)

            if random.uniform(0, 1) < prob_0:
                measurement_result = 0
            else:
                measurement_result = 1

            if self.update == 0:
                prior_samples = np.random.normal(mu, self.sigma, self.nSamples)

                mu, sigma = self.update_prior(
                    M, theta, measurement_result, prior_samples)

            elif self.update == 1:
                prior_samples = np.random.normal(mu, self.sigma, self.nSamples)

                mu, sigma = self.update_prior_rescaled(
                    M, theta, measurement_result, prior_samples)

            else:
                mu, sigma = self.update_exact(mu, M, theta, measurement_result)

            self.sigma = sigma

            run += 1
            if run > self.max_shots:
                failed = 1
                # print(
                # f"Maximum number of runs {self.max_shots} reached; exiting routine.")
                break

        estimated = abs(cos(mu/2))

        true = abs(cos(self.phi/2))

        error = abs(estimated - true)

        return(error, run, failed)
        # return(
        #     f"  Value estimated: {estimated:.5f}\n  True value: {true:.5f}"
        #     + f"\n  Error: {error:.5f}\n  Number of runs: {run}"
        # )

    def run_experiment(self, experiment_number):
        mu = random.uniform(-pi, pi)

        run = 0
        data = []
        while run < experiment_number:
            if self.sigma < 10**-20:
                self.sigma = 10**-20
            M = max(1, int(np.round(1/self.sigma**self.alpha)))
            theta = mu - self.sigma

            prob_0 = self.probability(0, M, theta, self.phi)

            if random.uniform(0, 1) < prob_0:
                measurement_result = 0
            else:
                measurement_result = 1

            if self.update == 0:
                prior_samples = np.random.normal(mu, self.sigma, self.nSamples)

                mu, sigma = self.update_prior(
                    M, theta, measurement_result, prior_samples)

            elif self.update == 1:
                prior_samples = np.random.normal(mu, self.sigma, self.nSamples)

                mu, sigma = self.update_prior_rescaled(
                    M, theta, measurement_result, prior_samples)

            else:
                mu, sigma = self.update_exact(mu, M, theta, measurement_result)

            self.sigma = sigma

            

            estimated = abs(cos(mu%(2*pi)/2))

            true = abs(cos(self.phi/2))

            error = abs(estimated - true)
            data.append(error)
            run += 1

        return(data)
# print(Alpha_VQE(phi=0, nSamples = 100, alpha = 0.1).get_max_shots())