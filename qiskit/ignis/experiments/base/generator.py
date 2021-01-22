# -*- coding: utf-8 -*-

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
Base Experiment Generator class.
"""

from abc import ABC, abstractmethod
from typing import  List, Dict, Optional

from qiskit import QuantumCircuit
from qiskit import transpile
from qiskit.exceptions import QiskitError


class Generator(ABC):
    """Base generator class.
    
    At minimum subclasses must implement the abstract method
    ``_generate_circuits(self, **params)`` which should return a list of
    quantum circuits containing any metadata needed for analysis in each
    ``circuit.metadata`` dictionary.

    Optionally a class may also override the ``metadata(self, **params)``
    method which returns the list of metadata dictionaries obtained from
    each circuit. If this method is not implemented calling it will extract
    metadata from the generated circuits.
    """

    def __init__(self, name: str,
                 num_qubits: int,
                 physical_qubits: Optional[List[int]] = None):
        """Initialize an experiment.

        Args:
            name: experiment name
            num_qubits: number of active qubits for the generator
            physical_qubits: the physical qubit mapping for active qubits.
        """
        # Circuit generation parameters
        self._name = str(name)
        self._num_qubits = num_qubits
        self._physical_qubits = None
        if physical_qubits:
            self.qubits = physical_qubits

    @property
    def name(self) -> str:
        """Return experiment name"""
        return self._name

    @property
    def num_qubits(self) -> int:
        """Return the number of qubits for this experiment."""
        return self._num_qubits

    @property
    def qubits(self) -> List[int]:
        """Return the qubits for this experiment."""
        if self._qubits is None:
            return list(range(self._num_qubits))
        return self._qubits

    @qubits.setter
    def qubits(self, value):
        """Set the qubits for this experiment."""
        if value is not None:
            if len(value) != self._num_qubits:
                raise QiskitError(
                    "Length of physical qubits does not match Generator qubit number.")
            value = list(value)
        self._qubits = value

    def circuits(self, **params) -> List[QuantumCircuit]:
        """Return a list of experiment circuits."""
        qubits = self.qubits
        circuits = self._generate_circuits(**params)
        if self._qubits:
            circuits = transpile(circuits,
                                 initial_layout=qubits,
                                 optimization_level=0)
        return circuits

    def metadata(self, **params) -> List[Dict[str, any]]:
        """Generate a list of experiment metadata dicts."""
        return [circ.metadata for circ in self._generate_circuits(**params)]

    @abstractmethod
    def _generate_circuits(self, **params) -> List[QuantumCircuit]:
        """Return a list of experiment circuits."""
