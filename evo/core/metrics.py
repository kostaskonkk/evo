# -*- coding: UTF8 -*-
"""
Provides metrics for the evaluation of SLAM algorithms.
author: Michael Grupp

This file is part of evo (github.com/MichaelGrupp/evo).

evo is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

evo is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with evo.  If not, see <http://www.gnu.org/licenses/>.
"""

import abc
import logging
import math
import sys
from enum import Enum  # requires enum34 in Python 2.7

import numpy as np

from evo import EvoException
from evo.core import filters
from evo.core.result import Result
from evo.core import lie_algebra as lie

if sys.version_info[0] >= 3 and sys.version_info[1] >= 4:
    ABC = abc.ABC
else:
    ABC = abc.ABCMeta('ABC', (), {})

logger = logging.getLogger(__name__)


class MetricsException(EvoException):
    pass


class StatisticsType(Enum):
    rmse = "rmse"
    mean = "mean"
    median = "median"
    std = "std"
    min = "min"
    max = "max"
    sse = "sse"


class PoseRelation(Enum):
    full_transformation = "full transformation"
    translation_part = "translation part"
    rotation_part = "rotation part"
    rotation_angle_rad = "rotation angle in radians"
    rotation_angle_deg = "rotation angle in degrees"


class Unit(Enum):
    none = "unit-less"
    meters = "m"
    seconds = "s"
    degrees = "deg"
    radians = "rad"
    frames = "frames"


class VelUnit(Enum):
    meters_per_sec = "m/s"
    rad_per_sec = "rad/s"
    degrees_per_sec = "deg/s"


class Metric(ABC):
    @abc.abstractmethod
    def reset_parameters(self, parameters):
        return

    @abc.abstractmethod
    def process_data(self, data):
        return

    @abc.abstractmethod
    def get_statistic(self, statistics_type):
        return

    @abc.abstractmethod
    def get_all_statistics(self):
        return

    @abc.abstractmethod
    def get_result(self):
        return


class PE(Metric):
    """
    Abstract base class of pose error metrics.
    """

    def __init__(self):
        self.unit = Unit.none
        self.error = []

    def __str__(self):
        return "PE metric base class"

    @abc.abstractmethod
    def reset_parameters(self, parameters):
        return

    @abc.abstractmethod
    def process_data(self, data):
        return

    def get_statistic(self, statistics_type):
        if statistics_type == StatisticsType.rmse:
            squared_errors = np.power(self.error, 2)
            return math.sqrt(np.mean(squared_errors))
        elif statistics_type == StatisticsType.sse:
            squared_errors = np.power(self.error, 2)
            return np.sum(squared_errors)
        elif statistics_type == StatisticsType.mean:
            return np.mean(self.error)
        elif statistics_type == StatisticsType.median:
            return np.median(self.error)
        elif statistics_type == StatisticsType.max:
            return np.max(self.error)
        elif statistics_type == StatisticsType.min:
            return np.min(self.error)
        elif statistics_type == StatisticsType.std:
            return np.std(self.error)
        else:
            raise MetricsException("unsupported statistics_type")

    def get_all_statistics(self):
        """
        :return: a dictionary {StatisticsType.value : float}
        """
        statistics = {}
        for s in StatisticsType:
            try:
                statistics[s.value] = self.get_statistic(s)
            except MetricsException as e:
                if "unsupported statistics_type" not in str(e):
                    raise
        return statistics

    def get_result(self, ref_name="reference", est_name="estimate"):
        """
        Wrap the result in Result object.
        :param ref_name: optional, label of the reference data
        :param est_name: optional, label of the estimated data
        :return:
        """
        result = Result()
        metric_name = self.__class__.__name__
        result.add_info({
            "title": str(self),
            "ref_name": ref_name,
            "est_name": est_name,
            "label": "{} {}".format(metric_name,
                                    "({})".format(self.unit.value))
        })
        result.add_stats(self.get_all_statistics())
        if hasattr(self, "error"):
            result.add_np_array("error_array", self.error)
        return result

class APE(PE):
    """
    APE: absolute pose error
    metric for investigating the global consistency of a SLAM trajectory
    """

    def __init__(self, pose_relation=PoseRelation.translation_part):
        self.pose_relation = pose_relation
        self.E = []
        self.error = []
        if pose_relation == PoseRelation.translation_part:
            self.unit = Unit.meters
        elif pose_relation == PoseRelation.rotation_angle_deg:
            self.unit = Unit.degrees
        elif pose_relation == PoseRelation.rotation_angle_rad:
            self.unit = Unit.radians
        else:
            self.unit = Unit.none  # dimension-less

    def __str__(self):
        title = "APE w.r.t. "
        title += (str(self.pose_relation.value) + " " +
                  ("(" + self.unit.value + ")" if self.unit else ""))
        return title

    def reset_parameters(self, pose_relation=PoseRelation.translation_part):
        """
        Resets the current parameters and results.
        :param pose_relation: PoseRelation defining how the APE is calculated
        """
        self.__init__(pose_relation)

    @staticmethod
    def ape_base(x_t, x_t_star):
        """
        Computes the absolute error pose for a single SE(3) pose pair
        following the notation of the Kümmerle paper.
        :param x_t: estimated absolute pose at t
        :param x_t_star: reference absolute pose at t
        .:return: the delta pose
        """
        return lie.relative_se3(x_t, x_t_star)

    def process_data(self, data):
        """
        Calculates the APE on a batch of SE(3) poses from trajectories.
        :param data: tuple (traj_ref, traj_est) with:
        traj_ref: reference evo.trajectory.PosePath or derived
        traj_est: estimated evo.trajectory.PosePath or derived
        """
        if len(data) != 2:
            raise MetricsException(
                "please provide data tuple as: (traj_ref, traj_est)")
        traj_ref, traj_est = data
        if traj_ref.num_poses != traj_est.num_poses:
            raise MetricsException(
                "trajectories must have same number of poses")

        if self.pose_relation == PoseRelation.translation_part:
            # don't require full SE(3) matrices for faster computation
            self.E = traj_est.positions_xyz - traj_ref.positions_xyz
        else:
            self.E = [
                self.ape_base(x_t, x_t_star) for x_t, x_t_star in zip(
                    traj_est.poses_se3, traj_ref.poses_se3)
            ]
        logger.debug("Compared {} absolute pose pairs.".format(len(self.E)))
        logger.debug("Calculating APE for {} pose relation...".format(
            (self.pose_relation.value)))

        if self.pose_relation == PoseRelation.translation_part:
            # E is an array of position vectors only in this case
            self.error = [np.linalg.norm(E_i) for E_i in self.E]
        elif self.pose_relation == PoseRelation.rotation_part:
            self.error = np.array([
                np.linalg.norm(lie.so3_from_se3(E_i) - np.eye(3))
                for E_i in self.E
            ])
        elif self.pose_relation == PoseRelation.full_transformation:
            self.error = np.array(
                [np.linalg.norm(E_i - np.eye(4)) for E_i in self.E])
        elif self.pose_relation == PoseRelation.rotation_angle_rad:
            self.error = np.array(
                [abs(lie.so3_log(E_i[:3, :3])) for E_i in self.E])
        elif self.pose_relation == PoseRelation.rotation_angle_deg:
            self.error = np.array([
                abs(lie.so3_log(E_i[:3, :3])) * 180 / np.pi for E_i in self.E
            ])
        else:
            raise MetricsException("unsupported pose_relation")
