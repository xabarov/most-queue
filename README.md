# most_queue
Software package for calculation and simulation of queuing systems and queuing network

## Authors
- [xabarov](https://github.com/xabarov)

## Installation
Install most_queue with pip
```bash
  pip install most_queue
```

## DESCRIPTION OF THE SOFTWARE PACKAGE
most_queue consists of two packages. The first package ==.theory== contains programs that implement methods for calculating queuing theory models. The second package ==.sim== contains simulation programs. 
### Package .theory
| #  | Pakage name | Description |
| ------------- | ------------- |------------- |
| 1. | diff5dots  | Numerical calculation of the derivative of a function |
| 2.  | fj_calc | Numerical calculation of the initial moments of the SW maximum distribution  | 
| 3.  | m_ph_n_prty | Numerical calculation of QS M/Ph/n with 2 classes of applications, absolute priority based on the approximation of busy periods |
| 4.  | mg1_calc | Numerical calculation of QS M/G/1 |
| 5.  | gi_m_1_calc | Numerical calculation of QS GI/M/1 |
| 6.  | gi_m_n_calc | Numerical calculation of QS GI/M/n |
| 7.  | m_d_n_calc | Numerical calculation of QS M/D/n |
| 8.  | ek_d_n_calc | Numerical calculation of QS Ek/D/n |
| 9.  | mg1_warm_calc | Numerical calculation of QS M/G/1 with warm |
| 10.  | mgn_tt | Numerical calculation of QS M/H2/n by the Takahashi-Takami method with an arbitrary value of the coefficient of variation using complex parameters when | approximating the service time by the H2 distribution |
| 11.  | mmn3_pnz_cox_approx | Calculation of QS M/M/2 with 3 classes of applications, absolute priority by the Takahashi-Takami numerical method based on the approximation of busy period by the Cox distribution of the second order |
| 12.  | mmn_prty_pnz_approx | Calculation of QS M/M/2 with 2 classes of applications, absolute priority by the Takahashi-Takami numerical method based on the approximation of the busy period by the Cox distribution of the second order |
| 13.  | mmnr_calc | Calculation of QS M/M/n/r |
| 14.  | network_calc | Calculation of queuing network with priorities in nodes |
| 15.  | passage_time | Calculation of the initial transition times between arbitrary tiers of the Markov chain |
| 16.  | prty_calc | A set of functions for calculating QS with priorities (single-channel, multi-channel). The multichannel calculation is carried out by the method of relation invariants.|
| 17.  | student_stat | Calculation of confidence probability, confidence intervals for random variable with unknown RMS based on Student's t-distribution. |
| 18.  | sv_sum_calc | Calculate the initial moments of sums of random variables |
| 19.  | weibull | Selection of Weibull distribution parameters, calculation of CDF, PDF, Tail values. |
| 20.  | flow_sum | Flow summation, numerical calculation |
### Package .sim
| #  | Pakage name | Description |
| ------------- | ------------- |------------- |
| 1.  | fj_delta_im | Simulation of QS fork-join with a delay in the start of processing between channels | 
| 2.  | fj_im | Simulation of QS fork-join | 
| 3.  | network_im_prty | Simulation of queuing network with priorities in nodes | 
| 4.  | rand_destribution | A set of functions and classes designed to generate a PRNG and select parameters for distributions H2, C2, Ek, Pa, Gamma | 
| 5.  | smo_im | Simulation of QS M/G/n | 
| 6.  | smo_prty_im | Simulation of QS M/G/n with priorities  | 
| 7.  | flow_sum_im  | Simulation of flow summation | 

## Usage
Examples package usage are presented in the folder "test"

## Requirements
* matplotlib>=3.5.2
* matplotlib-inline>=0.1.3
* numba>=0.56.0
* numpy>=1.22.4
* pandas>=1.4.3
* scipy>=1.9.0
* tqdm>=4.64.0
* tqdm-stubs>=0.2.1






