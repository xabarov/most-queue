# Queueing Systems: Simulation & Numerical Methods

![Queue](assets/queue_long.png)

A Python package for simulating and analyzing queueing systems (QS) and networks. 

![GitHub commit activity](https://img.shields.io/github/commit-activity/m/xabarov/most-queue)


## About
This repository focuses on solving steady-state problems in queueing theory.

Key Features:
- Simulate various types of queueing systems and networks.
- Numerical methods for solving queueing theory problems.
- Analyze system performance metrics such as waiting times, soujourn times, load factor and etc.


## Use Cases
- Modeling cloud computing infrastructure.
- Designing efficient call centers.
- Optimizing transportation systems.
- Network traffic analysis.

## Contributing
Contributions are welcome! If you find any issues or have suggestions, please open an [issue](https://github.com/xabarov/most-queue/issues). Your pull requests are also appreciated. You can write me at [xabarov1985@gmail.com](mailto:xabarov1985@gmail.com) 

---

## Installation
- install most-queue with pip
```bash
  pip install most-queue
```

- install most-queue from repository
```bash
  pip install -e .
```

--- 

## Description of the Project

Most_queue consists of two main parts:
 - **most_queue.theory** contains programs that implement methods for calculating queueing theory models. 
 - **most_queue.sim** contains simulation programs. 

## See examples in the tests folder:

### FIFO QS
| #   | Kendall Notations |  Description      | Example | Tutorial |
|-----|-------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------|-------------------|
| 1. | Ek/D/c           |  Numerical calculation of a multi-channel system Ek/D/n   | [link](https://github.com/xabarov/most-queue/blob/main/tests/test_ek_d_n.py) | |
| 2.  | GI/M/1          |  Solving for QS GI/M/1     | [link](https://github.com/xabarov/most-queue/blob/main/tests/test_gi_m_1_calc.py) | |
| 3.  | GI/M/c          |  Solving for QS GI/M/c      | [link](https://github.com/xabarov/most-queue/blob/main/tests/test_gi_m_n_calc.py) | |
| 4.  | M/D/c           |  Solving for QS M/D/c        | [link](https://github.com/xabarov/most-queue/blob/main/tests/test_m_d_n_calc.py) | [link](https://github.com/xabarov/most-queue/blob/main/tutorials/queue_sim.ipynb)  |
| 5.  | M/G/1           |  Solving for QS M/G/1        | [link](https://github.com/xabarov/most-queue/blob/main/tests/test_mg1_calc.py) | |
| 6.  | M/H<SUB>2</SUB>/c         |  Numerical calculation of QS M/H<SUB>2</SUB>/c by the Takahashi-Takami method with complex parameters when approximating the serving time by the H<SUB>2</SUB>-distribution    | [link](https://github.com/xabarov/most-queue/blob/main/tests/test_mgn_tt.py) | [link](https://github.com/xabarov/most-queue/blob/main/tutorials/mgn_takahasi_takami.ipynb) |
| 7.  | M/M/c/r         |  Solving for QS M/M/c/r        | [link](https://github.com/xabarov/most-queue/blob/main/tests/test_qs_sim_test.py) | [link](https://github.com/xabarov/most-queue/blob/main/tutorials/queue_sim.ipynb) |


### QS with priorities

| #   | Kendall Notations |  Description      | Example | Tutorial |
|-----|-------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------|-------------------|
| 1.  | M/Ph/c/PR     |  Numerical calculation of QS M/Ph/c with 2 classes and PR - priority. Based on the approximation of busy periods            | [link](https://github.com/xabarov/most-queue/blob/main/tests/test_m_ph_n_prty.py) | |
| 2.  | M/M/c/PR           |  Numerical calculation of QS M/M/c with 2 classes, PR - priority by the Takahashi-Takami numerical method based on the approximation of the busy period by the Cox distribution      | [link](https://github.com/xabarov/most-queue/blob/main/tests/test_mmn_prty_pnz_approx.py) | |
| 3.  | M/M/c/PR           |  Numerical calculation of QS M/M/c with 3 classes, PR - priority by the Takahashi-Takami numerical method based on the approximation of busy period by the Cox distribution     | [link](https://github.com/xabarov/most-queue/blob/main/tests/test_mmn3_pnz_cox_approx.py) | |
| 4.  | M/G/1/PR           |  Calculating QS with preemtive priorities (single-channel).     | [link](https://github.com/xabarov/most-queue/blob/main/tests/test_qs_sim_prty.py) |[link](https://github.com/xabarov/most-queue/blob/main/tutorials/priority_queue.ipynb)  |
| 5.  | M/G/1/NP           |  Calculating QS with non-preemtive priorities (single-channel).     | [link](https://github.com/xabarov/most-queue/blob/main/tests/test_qs_sim_prty.py) |[link](https://github.com/xabarov/most-queue/blob/main/tutorials/priority_queue.ipynb)  |
| 6.  | M/G/c/Priority           | Calculating QS with NP and PR (multi-channel) by method of relation      | [link](https://github.com/xabarov/most-queue/blob/main/tests/test_qs_sim_prty.py) |[link](https://github.com/xabarov/most-queue/blob/main/tutorials/priority_queue.ipynb)  |


### Fork-Join QS

| #   | Kendall Notations |  Description      | Example | Tutorial |
|-----|-------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------|-------------------|
| 1. | M/M/c/Fork-Join       |  Solving for Fork-Join queueing system      | [link](https://github.com/xabarov/most-queue/blob/main/tests/test_fj_im.py) |  |
| 1. | M/G/c/Split-Join       |  Solving for Split-Join queueing system      | [link](https://github.com/xabarov/most-queue/blob/main/tests/test_fj_im.py) |  |


### QS with Batch Arrival
| #   | Kendall Notations |  Description      | Example | Tutorial |
|-----|-------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------|-------------------|
| 1.  | M<sup>x</sup>/M/1          |  Solving for the of M<sup>x</sup>/M/1 QS with batch arrival    | [link](https://github.com/xabarov/most-queue/blob/main/tests/test_batch.py) | |

### QS with Vacations
| #   | Kendall Notations |  Description      | Example | Tutorial |
|-----|-------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------|-------------------|
| 1.  | M/H<SUB>2</SUB>/c          |  Numerical calculation of the M/H<SUB>2</SUB>/c system with H<SUB>2</SUB>-warming using the Takahasi-Takagi method.      | [link](https://github.com/xabarov/most-queue/blob/main/tests/test_m_h2_h2warm.py) | [link](https://github.com/xabarov/most-queue/blob/main/tutorials/m_h2_h2warm.ipynb)| |
| 2.  | M/G/1           | Solving for QS M/G/1 with warm-up       |  | |
| 3.  | M/Ph/c         |  Multichannel queuing system with H<SUB>2</SUB>-serving time, H<SUB>2</SUB>-warm-up, H<SUB>2</SUB>-cold delay and H<SUB>2</SUB>-cold (vacations). The system uses complex parameters, which allows you to calculate systems with arbitrary serving, warm-up, cold-delay and cold variation coefficients | [link](https://github.com/xabarov/most-queue/blob/main/tests/test_mgn_with_h2_delay_cold_warm.py) | |
| 4.  | M/M/c          |  Multichannel queuing system with exp serving time, H<SUB>2</SUB>-warm-up and H<SUB>2</SUB>-cold (vacations). The system uses complex parameters, which allows to calculate systems with arbitrary warm-up and cold variation coefficients    | [link](https://github.com/xabarov/most-queue/blob/main/tests/test_mmn_h2cold_h2warm.py) | |

### QS with Impatience
| #   | Kendall Notations |  Description      | Example | Tutorial |
|-----|-------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------|-------------------|
| 1.  | M/M/1/D         |  Solving for M/M/1 with exponential impatience     | [link](https://github.com/xabarov/most-queue/blob/main/tests/test_impatience.py) | |

### Closed QS (with finite number of sources)
| #   | Kendall Notations |  Description      | Example | Tutorial |
|-----|-------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------|-------------------|
| 1. | M/M/1/N          |  Solving for the Engset model for M/M/1 with a finite number of sources.     | [link](https://github.com/xabarov/most-queue/blob/main/tests/test_engset.py) | |

### Queuing Networks

| #   | Kendall Notations |  Description      | Example | Tutorial |
|-----|-------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------|-------------------|
| 1.  | General Network |  Numerical calculation of queuing network with priorities in nodes      | [link](https://github.com/xabarov/most-queue/blob/main/tests/test_network_im_prty.py) | |


### Usage
- Look [here](https://github.com/xabarov/most-queue/tree/main/tests) for examples
- Look [here](https://github.com/xabarov/most-queue/tree/main/tutorials) for jupyter tutorials




