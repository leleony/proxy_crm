import numpy as np
from lmfit import Minimizer, Parameters
from numpy.typing import NDArray
from numba import jit

import matplotlib.pyplot as plt

@jit(cache=True, nopython=True, nogil=True)
def f_oil_CRMIP(f_ij, inj, alpha, beta, time):
  n_t = inj.shape[0]
  n_p = f_ij.shape[0]
  n_i = inj.shape[1]
  f_oil = np.zeros((n_t, n_p))

  for t in range(1, n_t):
    for j in range(n_p):
      temp_intgr = 0
      cum_sum = 0
      for i in range(n_i):
        if t == 1:
          cum_sum += f_ij[j,i] * inj[t,i] * 0
        else:
          cum_sum += f_ij[j,i] * inj[t,i] * (time[t] - time[t-1])

        temp_intgr += alpha[j] * (cum_sum) ** beta[j]

      f_oil[t,j] =  (1.0 + temp_intgr) ** -1

  return f_oil

class fracFlow_CRMP:
  def __init__(self, liquid: NDArray, f_ij: NDArray, inj: NDArray, oil: NDArray, time: NDArray):
    # Registering observed values
    self.liquid = liquid
    self.inj = inj
    self.oil = oil
    self.time = time

    self.n_time = self.liquid.shape[0]
    self.n_prod = self.liquid.shape[1]
    self.n_inj = self.inj.shape[1]

    if len(f_ij.shape) < 2: # Check whether f_ij is already reshaped
      self.f_ij = f_ij.reshape((self.n_prod, self.n_ij))
    elif len(f_ij.shape) == 2:
      self.f_ij = f_ij

    # Create zero values for the target of the code (the predicted data and params)
    self.f_oil = np.zeros((self.n_time, self.n_prod))
    self.q_oil = np.zeros((self.n_time, self.n_prod))
    self.conv = np.zeros((self.n_time, self.n_prod))
    self.alpha = np.zeros(self.n_prod)
    self.beta = np.zeros(self.n_prod)

  def solver(self, method='leastsq'):
    liquid = self.liquid
    f_ij = self.f_ij
    inj = self.inj
    oil = self.oil

    # Defining objective function
    def fcn_min(params, oil, liquid, f_ij, inj):
      for j in range(self.n_prod):
        self.alpha[j] = params[f'alpha_{j+1}']
        self.beta[j] = params[f'beta_{j+1}']
      
      oil_model = f_oil_CRMIP(f_ij=f_ij, inj=inj, alpha=self.alpha, beta= self.beta, time=self.time)
      return oil - (oil_model * liquid)

    # Creating parameters
    params = Parameters()
    for j in range(self.n_prod):
        params.add(f'alpha_{j+1}', value=2.0e-18, min=0, vary=True)
        params.add(f'beta_{j+1}', value=1.5, min=0, vary=True)

    solve = Minimizer(fcn_min, params, fcn_args=(oil, liquid, f_ij, inj))
    result = solve.minimize(method=method)

    final = oil + result.residual.reshape(self.n_time, self.n_prod)

    return final, self.alpha, self.beta, result
