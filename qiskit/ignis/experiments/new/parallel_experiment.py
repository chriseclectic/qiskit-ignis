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
Parallel Experiment class.
"""

from typing import Union, Dict, List

from qiskit import QuantumCircuit, ClassicalRegister
from qiskit.result import marginal_counts
from qiskit.exceptions import QiskitError

from .experiment_data import ExperimentData, AnalysisResult
from .base_experiment import BaseExperiment, BaseAnalysis


class ParallelExperimentData(ExperimentData):
    """Parallel experiment data class"""

    def __init__(self, experiment):
        """Initialize the experiment data.

        Args:
            experiment (ParallelExperiment): experiment object that
                                             generated the data.
        """
        super().__init__(experiment)

        # Initialize sub experiments
        self._sub_exp_data = [
            expr._data_class(expr) for expr in self._experiment._experiments
        ]

    def experiment_data(self, index: Union[int, slice]):
        """Return sub experiments"""
        return self._sub_exp_data[index]

    def _add_single_data(self, data: Dict[str, any]):
        """Add data to the experiment"""
        # TODO: Handle marginalizing IQ data
        metadata = data.get('metadata', {})
        if metadata.get('experiment_type') == self.experiment_type:

            # Add parallel data
            self._data.append(data)

            # Add marginalized data to sub experiments
            for i, index in enumerate(metadata['index']):
                clbits = metadata['clbits'][i]
                sub_data = {'metadata': metadata['metadata'][i]}
                if 'counts' in data:
                    sub_data['counts'] = marginal_counts(data['counts'], clbits)
                self._sub_exp_data[index].add_data(sub_data)


class ParallelAnalysis(BaseAnalysis):
    """Analysis class for ParallelExperiment"""

    def _run_analysis(self, experiment_data, **params) -> AnalysisResult:
        """Run analysis on ExperimentData"""
        if not isinstance(experiment_data, ParallelExperimentData):
            raise QiskitError("ParallelAnalysis must be run on ParallelExperimentData.")

        # Run analysis for sub-experiments
        for sub_exp_data in experiment_data._sub_exp_data:
            sub_exp_data.analysis(**params).run()

        # Add sub-experiment metadata as result of batch experiment
        # Note: if Analysis results had ID's these should be included here
        # rather than just the sub-experiment IDs
        sub_types = []
        sub_ids = []
        sub_qubits = []
        for expr in experiment_data._sub_exp_data:
            sub_types.append(expr.experiment_type)
            sub_ids.append(expr.experiment_id)
            sub_qubits.append(expr.experiment().physical_qubits)

        return AnalysisResult({
            'experiment_types': sub_types,
            'experiment_ids': sub_ids,
            'experiment_qubits': sub_qubits})


class ParallelExperiment(BaseExperiment):
    """Base Experiment class

    Subclasses need to implement abstract methods `circuits` and `run`.
    """

    def __init__(self, experiments: List[BaseExperiment]):
        """Initialize the analysis object."""
        self._experiments = experiments
        self._num_experiments = len(experiments)

        # Generate joint qubits and clbits
        joint_qubits = []
        for exp in self._experiments:
            joint_qubits += exp.physical_qubits

        super().__init__(type(self).__name__,
                         joint_qubits,
                         analysis_class=ParallelAnalysis,
                         data_class=ParallelExperimentData)

    def circuits(self, **circuit_options) -> List[QuantumCircuit]:
        """Return a list of experiment circuits.

        .. note:
            These circuits should be on qubits ``[0, .., N-1]`` for an
            *N*-qubit experiment. The circuits mapped to physical qubits
            are obtained via the :meth:`transpiled_circuits` method.
        """
        sub_circuits = []
        sub_qubits = []
        sub_size = []
        num_qubits = 0

        # Generate data for combination
        for expr in self._experiments:
            # Add subcircuits
            circs = expr.circuits(**circuit_options)
            sub_circuits.append(circs)
            sub_size.append(len(circs))

            # Add sub qubits
            qubits = list(range(
                num_qubits, num_qubits + expr.num_qubits))
            sub_qubits.append(qubits)
            num_qubits += expr.num_qubits

        # Generate empty joint circuits
        num_circuits = max(sub_size)
        joint_circuits = []
        for circ_idx in range(num_circuits):
            # Create joint circuit
            circuit = QuantumCircuit(self.num_qubits, name=f'parallel_exp_{circ_idx}')
            circuit.metadata = {
                "experiment_type": self._type,
                "index": [],
                "metadata": [],
                "qubits": [],
                "clbits": [],
            }
            for exp_idx in range(self._num_experiments):
                if circ_idx < sub_size[exp_idx]:
                    # Add subcircuits to joint circuit
                    sub_circ = sub_circuits[exp_idx][circ_idx]
                    num_clbits = circuit.num_clbits
                    qubits = sub_qubits[exp_idx]
                    clbits = list(range(
                        num_clbits, num_clbits + sub_circ.num_clbits))
                    circuit.add_register(ClassicalRegister(sub_circ.num_clbits))
                    circuit.append(sub_circ, qubits, clbits)
                    # Add subcircuit metadata
                    circuit.metadata['index'].append(exp_idx)
                    circuit.metadata['metadata'].append(sub_circ.metadata)
                    circuit.metadata['qubits'].append(qubits)
                    circuit.metadata['clbits'].append(clbits)

            # Add joint circuit to returned list
            joint_circuits.append(circuit)

        return joint_circuits
