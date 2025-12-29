"""
Base documentation and examples for extending Takahashi-Takami method.

This module provides documentation and examples for creating custom queueing system
calculators based on the Takahashi-Takami method implemented in MGnCalc.

For general documentation on numerical methods, see:
- docs/calculation.md - General guide on numerical calculation methods
- docs/models.md - Catalog of supported queueing system models

For examples of existing extensions, see:
- most_queue/theory/negative/mgn_rcs.py - Negative arrivals with RCS discipline
- most_queue/theory/negative/mgn_disaster.py - Negative arrivals with disasters
- most_queue/theory/vacations/m_h2_h2warm.py - Systems with warm-up periods
- most_queue/theory/priority/preemptive/m_ph_n_busy_approx.py - Priority systems

EXTENDING TAKAHASHI-TAKAMI METHOD FOR CUSTOM QUEUEING SYSTEMS
==============================================================

See also:
- Main documentation: docs/calculation.md
- Models catalog: docs/models.md
- Base class: most_queue/theory/fifo/mgn_takahasi.py

The MGnCalc class uses the Template Method pattern to allow easy extension for
custom queueing system calculations. This document explains how to create your
own calculator class.

BASIC PATTERN
-------------

To create a custom queueing system calculator:

1. Inherit from MGnCalc
2. Override matrix building methods to define your transition matrices
3. Optionally override iteration hooks for custom algorithm behavior
4. Optionally override result calculation methods

EXAMPLE 1: Simple Matrix Override
----------------------------------

The most common case is overriding transition matrices. For example, to add
negative arrivals:

    class CustomQueueCalc(MGnCalc):
        def __init__(self, n, buffer=None, calc_params=None):
            super().__init__(n, buffer, calc_params)
            self.l_neg = None  # negative arrival rate

        def set_sources(self, l_pos, l_neg):
            self.l_pos = l_pos
            self.l_neg = l_neg
            self.l = l_pos  # Set base arrival rate
            self.is_sources_set = True

        def _build_big_d_matrix(self, num):
            # Override D matrix to include negative arrivals
            # ... custom implementation ...
            return output_matrix

EXAMPLE 2: Custom Column Structure
-----------------------------------

If your state space has a different structure, override fill_cols():

    class CustomQueueCalc(MGnCalc):
        def fill_cols(self):
            # Define custom column structure
            for i in range(self.N):
                if i == 0:
                    self.cols.append(1)
                elif i < self.n + 1:
                    self.cols.append(i + 2)  # Extra column for custom state
                else:
                    self.cols.append(self.n + 2)

EXAMPLE 3: Custom Iteration Logic
----------------------------------

To customize the iteration algorithm, override iteration hooks:

    class CustomQueueCalc(MGnCalc):
        def _update_level_0(self):
            # Custom logic for level 0 (e.g., skip update in some cases)
            if self.some_condition:
                return  # Skip update
            super()._update_level_0()  # Or call parent implementation

        def _update_level_j(self, j):
            # Custom logic for level j
            if j == self.special_level:
                # Custom update for special level
                pass
            else:
                super()._update_level_j(j)  # Use default for others

EXAMPLE 4: Custom Probability Calculation
------------------------------------------

To customize how probabilities are calculated:

    class CustomQueueCalc(MGnCalc):
        def _calculate_p(self):
            # Custom probability calculation
            # ... your implementation ...
            pass

COMMON OVERRIDES SUMMARY
------------------------

Most commonly overridden methods (in order of frequency):

1. _build_big_b_matrix(num) - Downward transitions (most common)
2. _build_big_d_matrix(num) - Diagonal elements (very common)
3. fill_cols() - Column structure (when state space differs)
4. _build_big_a_matrix(num) - Upward transitions (moderately common)
5. _calculate_p() - Probability calculation (when probabilities differ)
6. _update_level_0() - Level 0 update (sometimes needed)
7. _build_big_c_matrix(num) - Horizontal transitions (rarely overridden)

Rarely overridden (but possible):

- _calculate_c(j) - Auxiliary variable calculation
- _calc_support_matrices() - Support matrix calculation
- _initial_probabilities() - Initial probability values
- _check_convergence() - Convergence criteria

COMPLETE EXAMPLE
----------------

Here's a complete example showing a typical extension:

    from most_queue.theory.fifo.mgn_takahasi import MGnCalc, TakahashiTakamiParams
    from most_queue.structs import QueueResults
    import numpy as np

    class MyCustomQueueCalc(MGnCalc):
        def __init__(self, n, buffer=None, calc_params=None):
            super().__init__(n, buffer, calc_params)
            self.custom_param = None

        def set_custom_param(self, value):
            self.custom_param = value

        def fill_cols(self):
            # Use default column structure
            super().fill_cols()
            # Or define custom structure
            # ...

        def _build_big_b_matrix(self, num):
            # Get base matrix
            base_matrix = super()._build_big_b_matrix(num)
            # Modify it
            # ... your modifications ...
            return modified_matrix

        def _build_big_d_matrix(self, num):
            # Custom diagonal matrix
            # ... your implementation ...
            return custom_matrix

        def _update_level_0(self):
            # Custom level 0 update if needed
            super()._update_level_0()  # Or implement from scratch

        def get_results(self, num_of_moments=4):
            # Customize results if needed
            results = super().get_results(num_of_moments)
            # Add custom fields
            # ...
            return results

TESTING YOUR EXTENSION
----------------------

When creating a custom extension:

1. Ensure fill_cols() sets self.cols correctly
2. Ensure matrix building methods return matrices of correct dimensions
3. Test with simple parameters first
4. Verify probabilities sum to approximately 1
5. Compare with known results if available

TROUBLESHOOTING
---------------

Common issues:

- Dimension mismatch: Check that cols[i] matches matrix dimensions
- Convergence problems: Try adjusting initial probabilities or tolerance
- Incorrect results: Verify matrix construction matches your QS model
- Performance: Consider optimizing matrix operations for large N

For more examples, see existing implementations:
- most_queue/theory/negative/mgn_rcs.py
- most_queue/theory/negative/mgn_disaster.py
- most_queue/theory/vacations/m_h2_h2warm.py
- most_queue/theory/priority/preemptive/m_ph_n_busy_approx.py

For detailed documentation, see:
- docs/calculation.md - Section "Расширение метода Такахаси-Таками"
- docs/models.md - Section "M/H₂/c (метод Такахаси-Таками)"
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from most_queue.theory.fifo.mgn_takahasi import MGnCalc

# This file serves as documentation. Actual implementation is in mgn_takahasi.py
