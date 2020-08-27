# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2019, 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""
Measurement error mitigation experiment.
"""

import logging
from typing import Optional, List, Dict

from qiskit import QuantumCircuit
from qiskit.exceptions import QiskitError

from qiskit.ignis.verification.tomography import combine_counts
from qiskit.ignis.base import Experiment

from qiskit.result import Counts
from qiskit.exceptions import QiskitError
from qiskit.ignis.base import Analysis

from qiskit.ignis.verification.tomography import combine_counts

# TODO: Move to ignis.measurement
from qiskit.ignis.mitigation.measurement.meas_mit_utils import assignment_matrix
from qiskit.ignis.mitigation.measurement.complete_method.complete_mitigator import CompleteMeasMitigator
from qiskit.ignis.mitigation.measurement.tensored_method.tensored_mitigator import TensoredMeasMitigator
from qiskit.ignis.mitigation.measurement.ctmp_method.ctmp_fitter import fit_ctmp_meas_mitigator
from qiskit.ignis.mitigation.measurement.ctmp_method.ctmp_generator_set import Generator

logger = logging.getLogger(__name__)


class MeasMitigation(Experiment):
    """Measurement error mitigator calibration experiment."""

    # pylint: disable=arguments-differ
    def __init__(self,
                 num_qubits: int,
                 method: str = 'CTMP',
                 labels: Optional[List[str]] = None):
        """Initialize measurement error mitigator calibration experiment."""
        # Base Experiment
        super().__init__(num_qubits)

        # Circuits and metadata
        self._circuits = []
        self._metadata = []
        if labels is None:
            labels = self._method_labels(method)
        for label in labels:
            self._metadata.append({
                'experiment': 'meas_mit',
                'cal': label,
            })
            self._circuits.append(self._calibration_circuit(num_qubits, label))

    def circuits(self) -> List[QuantumCircuit]:
        """Return a list of experiment circuits."""
        return self._circuits

    def metadata(self) -> List[dict]:
        """Generate a list of experiment metadata dicts."""
        return self._metadata

    def _method_labels(self, method):
        """Generate labels for initilizing via a standard method."""

        if method == 'tensored':
            return [self._num_qubits * '0', self._num_qubits * '1']

        if method in ['CTMP', 'ctmp']:
            labels = [self._num_qubits * '0', self._num_qubits * '1']
            for i in range(self._num_qubits):
                labels.append(((self._num_qubits - i - 1) * '0') + '1' +
                              (i * '0'))
            return labels

        if method == 'complete':
            labels = []
            for i in range(2**self._num_qubits):
                bits = bin(i)[2:]
                label = (self._num_qubits - len(bits)) * '0' + bits
                labels.append(label)
            return labels

        raise QiskitError("Unrecognized method {}".format(method))

    @staticmethod
    def _calibration_circuit(num_qubits: int, label: str) -> QuantumCircuit:
        """Return a calibration circuit.

        This is an N-qubit circuit where N is the length of the label.
        The circuit consists of X-gates on qubits with label bits equal to 1,
        and measurements of all qubits.
        """
        circ = QuantumCircuit(num_qubits, name='meas_mit_cal_' + label)
        for i, val in enumerate(reversed(label)):
            if val == '1':
                circ.x(i)
        circ.measure_all()
        return circ


class MeasMitigationAnalysis(Analysis):
    """Measurement error mitigator calibration experiment result."""

    # pylint: disable=arguments-differ
    def __init__(self,
                 data: Optional[any] = None,
                 metadata: Optional[Dict[str, any]] = None,
                 method: str = 'CTMP'):
        """Initialize measurement error mitigator calibration experiment."""
        # Base Experiment Result class
        super().__init__('meas_mit', data=data, metadata=metadata)

        self._method = method

        # Intermediate representation of results
        self._cal_data = {}
        self._num_qubits = None

    def _analyze(self,
                 data: List[Counts],
                 metadata: List[Dict[str, any]],
                 method: Optional[str] = None,
                 generators: Optional[List[Generator]] = None):
        """Fit and return the Mitigator object from the calibration data."""

        cal_data, num_qubits = self._calibration_data(data, metadata)

        # Run fitter for given method
        if method is None:
            method = self._method

        if method == 'complete':
            # Construct A-matrix from calibration data
            amat = assignment_matrix(cal_data, num_qubits)
            return CompleteMeasMitigator(amat)

        if method == 'tensored':
            # Construct single-qubit A-matrices from calibration data
            amats = []
            for qubit in range(num_qubits):
                amat = assignment_matrix(cal_data, num_qubits, [qubit])
                amats.append(amat)
            return TensoredMeasMitigator(amats)

        if method == 'CTMP' or method == 'ctmp':
            return fit_ctmp_meas_mitigator(cal_data, num_qubits, generators)

        raise QiskitError("Invalid analysis method {}".format(method))

    @staticmethod
    def _calibration_data(data: Counts, metadata: Dict[str, any]):
        """Process counts into calibration data"""
        cal_data = {}
        num_qubits = None
        for i, meta in enumerate(metadata):
            if num_qubits is None:
                num_qubits = len(meta['cal'])
            key = int(meta['cal'], 2)
            counts = data[i].int_outcomes()
            if key not in cal_data:
                cal_data[key] = counts
            else:
                cal_data[key] = combine_counts(cal_data[key], counts)
        return cal_data, num_qubits
