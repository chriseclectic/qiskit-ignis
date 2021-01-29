# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""
Batch Experiment class.
"""

from collections import OrderedDict
from typing import Union, Dict, List

from qiskit import QuantumCircuit
from qiskit.exceptions import QiskitError

from .experiment_data import ExperimentData, AnalysisResult
from .base_experiment import BaseExperiment, BaseAnalysis


class BatchExperimentData(ExperimentData):
    """Base ExperimentData class

    Subclasses need to implement abstract methods `circuits` and `run`.

    Note: this base class intentionally does not deal with jobs or where
    data comes from.
    """

    def __init__(self, experiment):
        """Initialize the analysis object."""
        super().__init__(experiment)

        # Initialize sub experiments
        self._sub_exp_data = [
            expr._data_class(expr) for expr in self._experiment._experiments
        ]

    def __repr__(self):
        sub_types = []
        sub_ids = []
        sub_qubits = []
        for expr in self._sub_exp_data:
            sub_types.append(expr.experiment_type)
            sub_ids.append(expr.experiment_id)
            sub_qubits.append(expr.experiment().physical_qubits)
        n_res = len(self._analysis_results)

        ret = f'Experiment: {self.experiment_type}'
        ret += f'\nExperiment ID: {self.experiment_id}'
        ret += f'\nStatus: Complete'
        ret += f'\nSub Experiments: {sub_types}'
        ret += f'\nSub Experiments IDs: {sub_ids}'
        ret += f'\nQubits: {sub_qubits}'
        ret += f'\nData: {len(self._data)}'
        ret += f'\nAnalysis Results: {n_res}'
        if n_res:
            ret += f'\nLast Analysis Result:\n{repr(self._analysis_results[-1])}'
        return ret

    def experiment_data(self, index: Union[int, slice]):
        """Return sub experiments"""
        return self._sub_exp_data[index]

    def _add_single_data(self, data: Dict[str, any]):
        """Add data to the experiment"""
        # TODO: Handle marginalizing IQ data
        metadata = data.get('metadata', {})
        if metadata.get('experiment_type') == self.experiment_type:

            # Add joint data
            self._data.append(data)

            # Add data to correct sub experiment data
            index = metadata['index']
            sub_data = {'metadata': metadata['metadata']}
            if 'counts' in data:
                sub_data['counts'] = data['counts']
            self._sub_exp_data[index].add_data(sub_data)


class BatchAnalysis(BaseAnalysis):
    """Analysis class for BatchExperiment"""

    def _run_analysis(self, experiment_data, **params) -> AnalysisResult:
        """Run analysis on ExperimentData"""
        if not isinstance(experiment_data, BatchExperimentData):
            raise QiskitError("BatchAnalysis must be run on BatchExperimentData.")

        joint_results = []
        for sub_exp_data in experiment_data._sub_exp_data:
            sub_result = sub_exp_data.analysis(**params).run()
            joint_results.append(sub_result)

        return AnalysisResult({'joint_results': joint_results})


class BatchExperiment(BaseExperiment):
    """Batch experiment class"""

    def __init__(self, experiments: List[BaseExperiment]):
        """Initialize a batch experiment."""
        self._experiments = experiments
        self._num_experiments = len(experiments)

        # Generate qubit map
        self._qubit_map = OrderedDict()
        logical_qubit = 0
        for expr in self._experiments:
            for physical_qubit in expr.physical_qubits:
                if physical_qubit not in self._qubit_map:
                    self._qubit_map[physical_qubit] = logical_qubit
                    logical_qubit += 1

        super().__init__("batch_experiment",
                         list(self._qubit_map.keys()),
                         analysis_class=BatchAnalysis,
                         data_class=BatchExperimentData)

    def circuits(self, **circuit_options) -> List[QuantumCircuit]:
        """Return a list of experiment circuits.

        .. note:
            These circuits should be on qubits ``[0, .., N-1]`` for an
            *N*-qubit experiment. The circuits mapped to physical qubits
            are obtained via the :meth:`transpiled_circuits` method.
        """
        # TODO Add option for interleaving experiments rather than
        # doing them sequentially

        batch_circuits = []

        # Generate data for combination
        for index, expr in enumerate(self._experiments):
            if self.physical_qubits == expr.physical_qubits:
                qubit_mapping = None
            else:
                qubit_mapping = [
                    self._qubit_map[qubit] for qubit in expr.physical_qubits]
            for circuit in expr.circuits(**circuit_options):
                # Update metadata
                circuit.metadata = {
                    'experiment_type': self._type,
                    'metadata': circuit.metadata,
                    'index': index
                }
                # Remap qubits if required
                if qubit_mapping:
                    circuit = self._remap_qubits(circuit, qubit_mapping)
                batch_circuits.append(circuit)
        return batch_circuits

    def _remap_qubits(self, circuit, qubit_mapping):
        """Remap qubits if physical qubit layout is different to batch layout"""
        num_qubits = self.num_qubits
        num_clbits = circuit.num_clbits
        new_circuit = QuantumCircuit(num_qubits, num_clbits,
                                     name='batch_' + circuit.name)
        new_circuit.metadata = circuit.metadata
        new_circuit.append(circuit, qubit_mapping, list(range(num_clbits)))
        return new_circuit
