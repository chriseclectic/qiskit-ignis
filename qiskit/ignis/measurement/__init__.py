# This code is part of Qiskit.
#
# (C) Copyright IBM 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
=============================================
Measurement (:mod:`qiskit.ignis.measurement`)
=============================================

.. currentmodule:: qiskit.ignis.measurement

Expectation Values
==================

.. autosummary::
   :toctree: ../stubs/

   ExpectationValue
   ExpectationValueResult
   expectation_value
   pauli_diagonal


Discriminator
=============

The discriminators are used to to discriminate level one data into level two counts.

.. autosummary::
   :toctree: ../stubs/

   DiscriminationFilter
   IQDiscriminationFitter
   LinearIQDiscriminator
   QuadraticIQDiscriminator
   SklearnIQDiscriminator
"""

from .expval import (ExpectationValue,
                     ExpectationValueResult,
                     expectation_value,
                     pauli_diagonal)

from .mitigation import (MeasMitigator,
                         MeasMitigatorResult,
                         CompleteMeasMitigator,
                         TensoredMeasMitigator,
                         CTMPMeasMitigator,
                         expectation_value)

from .discriminator import (DiscriminationFilter,
                            IQDiscriminationFitter,
                            LinearIQDiscriminator,
                            QuadraticIQDiscriminator,
                            SklearnIQDiscriminator)
