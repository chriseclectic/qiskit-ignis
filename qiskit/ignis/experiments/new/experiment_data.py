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
Experiment Data class
"""

import uuid
from typing import Union, Dict, List

from qiskit.result import Result
from qiskit.providers import Job, BaseJob
from qiskit.exceptions import QiskitError


class AnalysisResult(dict):
    """Placeholder"""


class ExperimentData:
    """Base ExperimentData class

    Subclasses need to implement abstract methods `circuits` and `run`.

    Note: this base class intentionally does not deal with jobs or where
    data comes from.
    """

    def __init__(self, experiment):
        """Initialize the analysis object.

        Args:
            experiment (BaseExperiment): experiment object that
                                         generated the data.
        """
        # Experiment identification metadata
        self._id = str(uuid.uuid4())
        self._experiment = experiment

        # Experiment Data
        self._data = []

        # Analysis
        self._analysis_results = []

    def __repr__(self):
        n_res = len(self._analysis_results)
        ret = f'Experiment: {self.experiment_type}'
        ret += f'\nExperiment ID: {self.experiment_id}'
        ret += f'\nStatus: Complete'
        ret += f'\nQubits: {self._experiment.physical_qubits}'
        ret += f'\nData: {len(self._data)}'
        ret += f'\nAnalysis Results: {n_res}'
        if n_res:
            ret += f'\nLast Analysis Result:\n{repr(self._analysis_results[-1])}'
        return ret

    @property
    def experiment_type(self) -> str:
        """Return the experiment type"""
        return self._experiment._type

    @property
    def experiment_id(self) -> str:
        """Return the experiment id"""
        return self._id

    def experiment(self):
        """Return Experiment object"""
        return self._experiment

    def analysis(self):
        """Return Analysis object for stored experiment data"""
        return self._experiment._analysis_class(self)

    def analysis_result(self, index: Union[int, slice]) -> Union[
            AnalysisResult, List[AnalysisResult]]:
        """Return stored AnalysisResult"""
        return self._analysis_results[index]

    def add_analysis_result(self, result):
        """Add an Analysis Result"""
        self._analysis_results.append(result)

    def data(self, index: Union[int, slice]) -> Union[
            Dict[str, any], List[Dict[str, any]]]:
        """Return stored experiment data"""
        return self._data[index]

    def add_data(self, data: Union[
            Result, List[Result], Job, List[Job],
            Dict[str, any], List[Dict[str, any]]]):
        """Add data to the experiment"""
        if isinstance(data, dict):
            self._add_single_data(data)
        elif isinstance(data, Result):
            self._add_result_data(data)
        elif isinstance(data, (Job, BaseJob)):
            self._add_result_data(data.result())
        elif isinstance(data, list):
            for dat in data:
                self.add_data(data)
        else:
            raise QiskitError("Invalid data format.")

    def _add_result_data(self, result: Result):
        """Add data from qiskit Result object"""
        num_data = len(result.results)
        for i in range(num_data):
            metadata = result.results[i].header.metadata
            if metadata.get("experiment_type") == self.experiment_type:
                data = result.data(i)
                data['metadata'] = metadata
                if 'counts' in data:
                    # Format to Counts object rather than hex dict
                    data['counts'] = result.get_counts(i)
                self._add_single_data(data)

    def _add_single_data(self, data: Dict[str, any]):
        """Add a single data dictionary to the experiment."""
        # This method is intended to be overriden by subclasses when necessary.
        if data.get('metadata', {}).get('experiment_type') == self._experiment._type:
            self._data.append(data)
