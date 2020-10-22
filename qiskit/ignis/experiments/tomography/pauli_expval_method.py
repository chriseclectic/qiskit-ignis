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
Expectation Value Experiment.
"""

from typing import Optional, List
from functools import lru_cache
import itertools as it

import numpy as np

from qiskit import QuantumCircuit
from qiskit.quantum_info import PauliTable, DensityMatrix
from qiskit.ignis.verification.tomography.fitters.lstsq_fit import make_positive_semidefinite
from qiskit.ignis.experiments.expval import pauli_diagonal, expectation_value


@lru_cache(maxsize=4)
def pauli_qst_generator(num_qubits: int,
                        labels: Optional[List[str]] = None):
    """Measurement circuit generator for Pauli basis measurements."""
    if labels is None:
        labels = [''.join(i) for i in it.product(
            ('X', 'Y', 'Z'), repeat=num_qubits)]

    circuits = []
    metadata = []
    for label in labels:
        circuits.append(pauli_meas_circuit(label))
        metadata.append({'meas_label': label})
    return circuits, metadata


def pauli_qst_analyze(data, metadata, mitigator=None, psd=True):
    """Perform linear inversion fit of Pauli QST data."""
    labels, expvals, stderrors = process_pauli_data(
        data, metadata, mitigator=mitigator)
    rho = pauli_linear_inversion(labels, expvals, stderrors)
    if psd:
        rho = make_positive_semidefinite(rho)
    return DensityMatrix(rho)


def pauli_meas_circuit(label: str):
    """Pauli measurement circuit"""
    circuit = QuantumCircuit(len(label))
    for i, val in enumerate(reversed(label)):
        if val == 'Y':
            circuit.sdg(i)
        if val in ['Y', 'X']:
            circuit.h(i)
    circuit.measure_all()
    return circuit


def pauli_linear_inversion(labels: np.ndarray,
                           expvals: np.ndarray,
                           stderrors: np.ndarray) -> np.ndarray:
    """Perform linear inversion of Pauli expectaiton value data"""
    num_qubits = len(labels[0])
    dim = 2 ** num_qubits
    rho = np.eye(dim, dtype=complex) / dim

    pbasis = PauliTable.from_labels(labels)
    for i, matrix in enumerate(pbasis.matrix_iter()):
        rho += (expvals[i] / dim) * matrix
    return rho


def process_pauli_data(data, metadata, mitigator=None):
    """Process Pauli expval measurement data for QST"""
    expval_data = {}
    stderror_data = {}
    shots_data = {}

    for counts, meta in zip(data, metadata):
        meas_basis = meta['meas_label']
        shots = sum(counts.values())
        subsets = basis_subsets(meas_basis)

        for label in subsets:
            diagonal = pauli_diagonal(label)
            expval, error = expectation_value(counts, diagonal=diagonal, mitigator=mitigator)

            if label in shots_data:
                shots_data[label].append(shots)
                expval_data[label].append(expval)
                stderror_data[label].append(error)
            else:
                shots_data[label] = [shots]
                expval_data[label] = [expval]
                stderror_data[label] = [error]

    # Combine data
    labels = np.array(list(expval_data.keys()))
    expvals = np.zeros(labels.size, dtype=float)
    stderrors = np.zeros(labels.size, dtype=float)

    for i, label in enumerate(labels):
        shots = np.sum(shots_data[label])
        expval = np.dot(shots_data[label], expval_data[label]) / shots
        var = np.dot(shots_data[label], np.array(stderror_data[label]) ** 2) / shots
        if var < 0:
            var = 0.0
        expvals[i] = expval
        stderrors[i] = np.sqrt(var / shots)

    return labels, expvals, stderrors


def basis_subsets(meas_basis):
    """Return Pauli measurement subset bases"""
    expval_bases = [meas_basis]

    num_qubits = len(meas_basis)
    for i in range(1, num_qubits):
        for elts in it.combinations(range(num_qubits), i):
            basis = list(meas_basis)
            for j in elts:
                basis[j] = 'I'
            expval_bases.append(''.join(basis))
    return expval_bases
