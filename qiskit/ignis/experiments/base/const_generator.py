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
Fixed circuit experiment Generator class.
"""

from typing import List, Dict, Optional

from qiskit import QuantumCircuit
from qiskit.exceptions import QiskitError
from .generator import Generator


class ConstantGenerator(Generator):
    """A generator class for a static list of circuits"""

    def __init__(self,
                 name: str,
                 circuits: List[QuantumCircuit],
                 metadata: Optional[List[Dict[str, any]]] = None,
                 physical_qubits: Optional[List[int]] = None):

        # Format circuits
        if isinstance(circuits, QuantumCircuit):
            circuits = [circuits]
        if not circuits:
            raise QiskitError("Input circuit list is empty")
        self._circuits = circuits

        # Format metadata
        if metadata:
            if isinstance(metadata, dict):
                metadata = [metadata]
            if len(metadata) != len(self._circuits):
                raise QiskitError("Input metadata list is not same length as circuit list")
            for i, meta in enumerate(metadata):
                self._circuits[i].metadata = meta

        # Set num qubits and physical qubits
        num_qubits = self._circuits[0].num_qubits
        for circ in self._circuits[1:]:
            if circ.num_qubits != num_qubits:
                raise QiskitError("Input circuits must all have same number of qubits.")

        if qubits is None:
            qubits = list(range(circuits[0].num_qubits))
        super().__init__(name, qubits, physical_qubits=physical_qubits)

    def _generate_circuits(self, **params) -> List[Dict[str, any]]:
        return self._circuits
