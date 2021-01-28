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
Base Experiment class.
"""

from abc import ABC, abstractmethod
from numbers import Integral
from typing import Optional, Union, Dict, List, Callable

from qiskit import transpile, assemble, QuantumCircuit
from qiskit.providers import Backend
from qiskit.exceptions import QiskitError

from .experiment_data import ExperimentData, AnalysisResult


class BaseExperiment(ABC):
    """Base Experiment class

    Subclasses need to implement abstract methods `circuits` and `run`.

    Note: this base class intentionally does not deal with jobs or where
    data comes from.
    """
    def __init__(self,
                 experiment_type: str,
                 qubits: Union[int, List[int]],
                 analysis_class: Optional[object] = None,
                 data_class: Optional[object] = ExperimentData,
                 circuit_options: Optional[Dict[str, str]] = None):
        """Initialize the analysis object.

        Args:
            experiment_type: the experiment type string.
            qubits: the number of qubits or list of physical qubits for
                    the experiment.
            analysis_class: Optional, the default Analysis class to use for
                            data analysis. If None no data analysis will be
                            done on experiment data.
            data_class: Optional, a custom ExperimentData subclass that is
                        produced by the experiment.
            circuit_options: Optional, dictionary of allowed kwargs and
                             default values for the `circuit` method.
        """
        # Experiment identification metadata
        self._type = experiment_type
        self._analysis_class = analysis_class
        self._data_class = data_class

        # Circuit parameters
        if isinstance(qubits, Integral):
            self._num_qubits = qubits
            self._physical_qubits = None
        else:
            self._num_qubits = len(qubits)
            self.physical_qubits = qubits
        self._circuit_options = circuit_options if circuit_options else {}

    def run(self, backend: Backend,
            experiment_data: Optional[ExperimentData] = None,
            **kwargs) -> ExperimentData:
        """Run an experiment and perform analysis.

        Args:
            backend: The backend to run the experiment on.
            experiment_data: Optional, add results to existing experiment data.
                             If None a new ExperimentData object will be returned.
            kwargs: keyword arguments for self.circuit,
                    qiskit.transpile, and backend.run.

        Returns:
            ExperimentData: the experiment data object.

        .. note:

            This method is intended to be overriden by subclasses when required.
        """
        # Create new experiment data
        if experiment_data is None:
            experiment_data = self._data_class(self)

        # Generate and run circuits
        circuits = self.transpiled_circuits(backend, **kwargs)
        qobj = assemble(circuits)
        job = backend.run(qobj, **kwargs)

        # Add Job to ExperimentData
        experiment_data.add_data(job)

        # Queue analysis of data for when job is finished
        if self._analysis_class is not None:
            experiment_data.analysis().run()

        # Return the ExperimentData future
        return experiment_data

    @property
    def num_qubits(self) -> int:
        """Return the number of qubits for this experiment."""
        return self._num_qubits

    @property
    def physical_qubits(self) -> List[int]:
        """Return the physical qubits for this experiment."""
        if self._physical_qubits is None:
            return list(range(self.num_qubits))
        return self._physical_qubits

    @physical_qubits.setter
    def physical_qubits(self, value):
        """Set the physical qubits for this experiment."""
        if value is not None:
            # Check for duplicates
            if len(value) != len(set(value)):
                raise QiskitError("Duplicate qubits in physical qubits list.")
            if len(value) != self.num_qubits:
                raise QiskitError(
                    "Length of physical qubits does not match Generator qubit number.")
            value = list(value)
        self._physical_qubits = value

    @abstractmethod
    def circuits(self, **circuit_options) -> List[QuantumCircuit]:
        """Return a list of experiment circuits.

        .. note:
            These circuits should be on qubits ``[0, .., N-1]`` for an
            *N*-qubit experiment. The circuits mapped to physical qubits
            are obtained via the :meth:`transpiled_circuits` method.
        """
        pass

    def transpiled_circuits(self, backend: Optional[Backend] = None,
                            **kwargs) -> List[QuantumCircuit]:
        """Return a list of transpiled experiment circuits for execution."""
        # Filter kwargs to circuit and transpile options
        circuit_options = {}
        transpile_options = {}
        for key in kwargs:
            if key in self._circuit_options:
                circuit_options[key] = kwargs[key]
            else:
                transpile_options[key] = kwargs[key]

        # Generate circuits
        circuits = self.circuits(**circuit_options)

        # Transpile circuits
        if 'optimization_level' not in transpile_options:
            transpile_options['optimization_level'] = 0
        if 'initial_layout' in transpile_options:
            raise QiskitError(
                'Initial layout must be specified by the Experiement.')
        if self._physical_qubits:
            transpile_options['initial_layout'] = self.physical_qubits
        circuits = transpile(circuits, backend=backend, **transpile_options)

        return circuits


class BaseAnalysis(ABC):
    """Base Experiment result analysis class."""

    def __init__(self, experiment_data: ExperimentData):
        """Initialize the analysis object.

        Args:
            experiment_data: experiment data object.
        """
        self._experiment_data = experiment_data

    def run(self, **params) -> AnalysisResult:
        """Run analysis and update stored ExperimentData with analysis result.

        Returns:
            AnalysisResult: the output of the analysis,
        """
        result = self._run_analysis(self._experiment_data, **params)
        self._experiment_data.add_analysis_result(result)
        return result

    @abstractmethod
    def _run_analysis(self, experiment_data: ExperimentData,
                      **params) -> Dict[str, any]:
        """Run analysis on ExperimentData"""
        # Subclasses should implement this method to run analysis
        pass


class _AnalysisFunction(BaseAnalysis):
    """Analysis class defined by a function"""

    _ANALYSIS_FUNCTION = None

    @property
    def analysis_function(self):
        """Return the analysis function of the class"""
        if self._ANALYSIS_FUNCTION is None:
            raise QiskitError("Analysis function has not been specified")
        return self._ANALYSIS_FUNCTION

    def _run_analysis(self, experiment_data: ExperimentData,
                      **params) -> Dict[str, any]:
        func = self.analysis_function
        # pylint: disable = not-callable
        return func(experiment_data, **params)


# pylint: disable = invalid-name
def AnalysisFunction(func: Callable) -> type:
    """Return an AnalysisFunction.

    Args:
        func: The analysis function.

    Returns:
        type: Analysis class.

    .. note:
        ``func`` should have signature
        ``func(experiment_data: ExperimentData, **params) -> Dict[str, any]``
    """
    class Analysis(_AnalysisFunction):
        """Experiment Analysis class"""
        _ANALYSIS_FUNCTION = func

    return Analysis
