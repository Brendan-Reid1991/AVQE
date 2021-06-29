import random
import numpy as np
from numpy import pi, exp, cos, sin, log
import warnings

import time as time

class AVQE():
    """
    This class is the new AVQE algorithm
    - exact updates instead of rejection sampling
    - different probability function to represent
        no collapse process
    - alpha has been removed and replaced with m_max
    

    Inputs
    ------
    REQUIRED:
        phi : float 
            phase value to estimate
        m_max : int
            maximum value of M (unitary repetitions)
    
    OPTIONAL:
        accuracy: float : default = 0.001
            accuracy in result, exit criteria
        sigma : float : default = pi/2
            initial variance
        max_shots : int : default = 10000
            maximum number of runs to do
    
    Outputs
    -------
        cos(mu / 2) : float
            Estimated angle
        cos(self.phi / 2) : float
            True angle

    Raises
    -------
    WARNINGS for poorly chosen values of accuracy, nSamples and sigma
    VALUEERROR for alpha <0 or >1
    """
    def __init__(self, phi, max_m, accuracy=10**-3, sigma=pi/2, max_shots=10**4):
        self.phi = phi
        self.max_m = max_m
        self.accuracy = accuracy
        self.sigma = sigma
        self.max_shots = max_shots

        if self.accuracy < 10**-5:
            warnings.warn(f"Accuracy goal set extremely low;"+
            " will likely hit max_runs before convergence.")

        if self.sigma > 2*pi:
            warnings.warn(f"Initial variance is high,"+
            " will likely hit max_runs before convergence.")



    def get_max_shots(self):
        if self.max_m < 1/self.accuracy:
            n_max = (2 / (1 - log(self.max_m)/log(1/self.accuracy)))*(1 / (self.accuracy * self.max_m)**2 - 1)
        else:
            n_max = 4*log(1/self.accuracy)
        
        return(n_max)

    def get_alpha(self):
        return(
            min(1, -log(self.max_m) / log(self.accuracy))
        )


    def probability(self, measurement_result, M, theta, phi):
        return(
            1/2 + (1 - 2*measurement_result) * cos(M * theta) * cos(M * phi)/2
        )

    def update_prior(self,
                     mu, M, theta, measurement_result
                     ):
        d = measurement_result
        Expectation = mu + ((1-2*d)*M*self.sigma**2 * sin(M*(theta - mu)))/ (exp(M**2 * self.sigma**2 / 2) + (1-2*d)*cos(M*(theta-mu)))

        VarNum = 2*exp(M**2 * self.sigma**2) + (
            2*(2*d - 1)*exp(M**2 * self.sigma**2 / 2) * ((M**2 * self.sigma**2) - 2)*cos(M*(theta - mu))
            ) + (
                (1 - 2*d)**2 * (1 - (2 * M**2 * self.sigma**2) + cos(2*M*(theta - mu)))
                )

        VarDenom = 2 * (
            exp(M**2 * self.sigma**2 / 2) + (1 - 2*d)*cos(M*(theta - mu))
            )**2

        Variance = self.sigma**2 * (VarNum / VarDenom)
        
        Std = np.sqrt(Variance)

        return(Expectation, Std)


    def estimate_phase(self):
        s1 = time.time()
        theory_max_shots = self.get_max_shots()
        if theory_max_shots > self.max_shots:
            warnings.warn(f"Required number of measurements for chosen accuracy is {theory_max_shots},"+
            f" whereas maximum is currently set to {self.max_shots}."
            )

        mu = random.uniform(-pi, pi)
        run = 0
        avg_time = 0
        prior_update_time = 0
        while self.sigma > self.accuracy:
            s2 = time.time()
            M = min(
                max(1, int(round(1 / self.sigma))), self.max_m
            )
    
            theta = 0
            
            prob_0 = self.probability(0, M, theta, self.phi)

            if random.uniform(0, 1) < prob_0:
                measurement_result = 0
            else:
                measurement_result = 1
            
            s3 = time.time()
            mu, sigma = self.update_prior(mu, M, theta, measurement_result)
            self.sigma = sigma  
            e3 = time.time()

            run += 1
            e2 = time.time()
            avg_time += (e2 - s2)
            prior_update_time += (e3-s3)
            if run > self.max_shots:
                # print(f"Maximum number of runs {self.max_shots} reached; exiting routine.")
                break
            
        
        estimated = abs(cos(mu/2))

        true = abs(cos(self.phi/2))

        error = abs(estimated - true)
        e1 = time.time()
        print("Total runtime:", e1-s1)
        print("Average time for one loop:", avg_time/run)
        print("    Of that, the prior update takes:", prior_update_time/run)
        return(
            error, run
            # f"  Value estimated: {estimated:.5f}\n  True value: {true:.5f}"
            # +f"\n  Error: {error:.5f}\n  Number of runs: {run}"
        )
            


a = AVQE(phi = -.234892348, max_m = 2, max_shots = 2*10**6)
print(a.estimate_phase())