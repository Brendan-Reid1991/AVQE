import random
import numpy as np
from numpy import pi
from math import sin, cos, exp, sqrt, log
import warnings


class AVQE():
    """
    This class is the new AVQE algorithm
    - exact updates instead of rejection sampling
    - different probability function to represent
        no collapse process
    - alpha has been removed and replaced with max_depth;
        the maximum number of times 
        the untiary U can be applied


    Inputs
    ------
    REQUIRED:
        phi : float 
            phase value to estimate
        max_depth : int
            maximum value of M (unitary repetitions)

    OPTIONAL:
        accuracy: float : default = 0.005
            accuracy in result, exit criteria
        sigma : float : default = pi/4
            initial variance
        max_shots : int : default = 10000
            maximum number of circuit measurements to do

    Outputs
    -------
        error : float
            Absolute error between real and estimated
        run : int
            Number of shots taken

    Raises
    -------
    WARNINGS for poorly chosen values of accuracy and sigma

    """

    def __init__(self, phi, max_depth, accuracy=5*10**-3, sigma=pi/4, max_shots=10**4):
        self.phi = phi
        self.max_depth = max_depth
        self.accuracy = accuracy
        self.sigma = sigma
        self.max_shots = max_shots

        if self.accuracy < 10**-5:
            warnings.warn(f"Accuracy goal set extremely low;" +
                          " will likely hit max_shots before convergence.")

        if self.sigma > 2*pi:
            warnings.warn(f"Initial variance is high," +
                          " will likely hit max_shots before convergence.")

    def get_max_shots(self):
        "Get the theoretical maximum number of required shots"
        if self.max_depth < 1/self.accuracy:
            n_max = (2 / (1 - log(self.max_depth)/log(1/self.accuracy))) * \
                (1 / (self.accuracy * self.max_depth)**2 - 1)
        else:
            n_max = 4*log(1/self.accuracy)
        return(np.ceil(n_max))

    def get_alpha(self):
        "Get the value of alpha from max_depth for alpha-VQE comparison"
        return(
            min(1, -log(self.max_depth) / log(self.accuracy))
        )

    def probability(self, measurement_result, M, theta, phi):
        "Get outcome probability from circuit"
        return(
            1/2 + (1 - 2*measurement_result) * cos(M * theta) * cos(M * phi)/2
        )

    def update_prior(self,
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
        
        "Estimate the phase value by simulating the circuit"

        theory_max_shots = self.get_max_shots()
        if theory_max_shots > self.max_shots:
            warnings.warn(f"Required number of measurements for chosen accuracy is {theory_max_shots}," +
                          f" whereas maximum is currently set to {self.max_shots}."
                          )

        mu = random.uniform(-pi, pi)
        run = 0
        theta = 0
        
        while round(self.sigma, 5) > self.accuracy:
            
            M = min(
                max(1, int(round(1 / self.sigma))), self.max_depth
            )

            

            prob_0 = self.probability(0, M, theta, self.phi)

            if random.uniform(0, 1) < prob_0:
                measurement_result = 0
            else:
                measurement_result = 1

            
            mu, sigma = self.update_prior(mu, M, theta, measurement_result)
            
            self.sigma = sigma
            
            run += 1
            if run > self.max_shots:
                # print(f"Maximum number of runs {self.max_shots} reached; exiting routine.")
                break

        estimated = abs(cos(mu/2))

        true = abs(cos(self.phi/2))

        error = abs(estimated - true)

        return(
            error, run
            # f"  Value estimated: {estimated:.5f}\n  True value: {true:.5f}"
            # +f"\n  Error: {error:.5f}\n  Number of runs: {run}"
        )
