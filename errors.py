#!/usr/bin/env python3

from evo import EvoException
from evo.core  import trajectory, sync
from enum import Enum  # requires enum34 in Python 2.7

import numpy as np

import abc
import math
import sys
from evo.core.result import Result
from evo.core import lie_algebra as lie

if sys.version_info[0] >= 3 and sys.version_info[1] >= 4:
    ABC = abc.ABC
else:
    ABC = abc.ABCMeta('ABC', (), {})

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

def ape(traj_ref, traj_est, pose_relation, align=False, correct_scale=False,
        align_origin=False, ref_name="reference", est_name="estimate"):
    ''' Based on main_ape.py
    '''
    traj_ref, traj_est = sync.associate_trajectories(traj_ref, traj_est)
    # Align the trajectories.
    only_scale = correct_scale and not align
    if align or correct_scale:
        traj_est = trajectory.align_trajectory(traj_est, traj_ref,
                                               correct_scale, only_scale)
    elif align_origin:
        traj_est = trajectory.align_trajectory_origin(traj_est, traj_ref)

    # Calculate APE.
    data = (traj_ref, traj_est)
    ape_metric = APE(pose_relation)
    ape_metric.process_data(data)

    ape_result = ape_metric.get_result(ref_name, est_name)

    ape_result.add_trajectory(ref_name, traj_ref)
    ape_result.add_trajectory(est_name, traj_est)
    if isinstance(traj_est, trajectory.PoseTrajectory3D):
        seconds_from_start = [
            t - traj_est.timestamps[0] for t in traj_est.timestamps
        ]
        ape_result.add_np_array("seconds_from_start", seconds_from_start)
        ape_result.add_np_array("timestamps", traj_est.timestamps)

    return ape_result

def results_as_dataframe(result_files, use_filenames=False, merge=False):
    import pandas as pd
    from evo.tools import pandas_bridge

    df = pd.DataFrame()
    for result_file in result_files:
        result = result_file
        # result = file_interface.load_res_file(result_file)
        name = result_file if use_filenames else None
        df = pd.concat([df, pandas_bridge.result_to_df(result, name)],
                       axis="columns")
    return df

def run(results):
    import sys

    import pandas as pd

    from evo.tools import log, user, settings
    from evo.tools.settings import SETTINGS

    pd.options.display.width = 80
    pd.options.display.max_colwidth = 20

    # log.configure_logging(args.verbose, args.silent, args.debug,
                          # local_logfile=args.logfile)

    df = results_as_dataframe(results, False, False)

    keys = df.columns.values.tolist()
    if SETTINGS.plot_usetex:
        keys = [key.replace("_", "\\_") for key in keys]
        df.columns = keys
    duplicates = [x for x in keys if keys.count(x) > 1]
    if duplicates:
        logger.error("Values of 'est_name' must be unique - duplicates: {}\n"
                     "Try using the --use_filenames option to use filenames "
                     "for labeling instead.".format(", ".join(duplicates)))
        sys.exit(1)

    # derive a common index type if possible - preferably timestamps
    common_index = None
    time_indices = ["timestamps", "seconds_from_start", "sec_from_start"]
    for idx in time_indices:
        if idx not in df.loc["np_arrays"].index:
            continue
        if df.loc["np_arrays", idx].isnull().values.any():
            continue
        else:
            common_index = idx
            break

    # build error_df (raw values) according to common_index
    if common_index is None:
        # use a non-timestamp index
        error_df = pd.DataFrame(df.loc["np_arrays", "error_array"].tolist(),
                                index=keys).T
    else:
        error_df = pd.DataFrame()
        for key in keys:
            new_error_df = pd.DataFrame({
                key: df.loc["np_arrays", "error_array"][key]
            }, index=df.loc["np_arrays", common_index][key])
            duplicates = new_error_df.index.duplicated(keep="first")
            if any(duplicates):
                logger.warning(
                    "duplicate indices in error array of {} - "
                    "keeping only first occurrence of duplicates".format(key))
                new_error_df = new_error_df[~duplicates]
            error_df = pd.concat([error_df, new_error_df], axis=1)

    # check titles
    first_title = df.loc["info", "title"][0]
    # first_file = args.result_files[0]
    first_file = results[0]

    checks = df.loc["info", "title"] != first_title
    for i, differs in enumerate(checks):
        if not differs:
            continue
        else:
            mismatching_title = df.loc["info", "title"][i]
            mismatching_file = args.result_files[i]
            logger.debug(SEP)
            logger.warning(
                CONFLICT_TEMPLATE.format(first_file, first_title,
                                         mismatching_title,
                                         mismatching_file))
            if not user.confirm(
                    "You can use --ignore_title to just aggregate data.\n"
                    "Go on anyway? - enter 'y' or any other key to exit"):
                sys.exit()

    # logger.debug(SEP)
    # logger.debug("Aggregated dataframe:\n{}".format(
        # df.to_string(line_width=80)))

    # show a statistics overview
    # logger.debug(SEP)
    # if not args.ignore_title:
        # logger.info("\n" + first_title + "\n\n")
    # logger.info(df.loc["stats"].T.to_string(line_width=80) + "\n")

    # check if data has NaN "holes" due to different indices
    inconsistent = error_df.isnull().values.any()
    if inconsistent and common_index != "timestamps" and not args.no_warnings:
        logger.debug(SEP)
        logger.warning("Data lengths/indices are not consistent, "
                       "raw value plot might not be correctly aligned")

    from evo.tools import plot
    import matplotlib.pyplot as plt
    import seaborn as sns
    import math

    # use default plot settings
    figsize = (SETTINGS.plot_figsize[0], SETTINGS.plot_figsize[1])
    use_cmap = SETTINGS.plot_multi_cmap.lower() != "none"
    colormap = SETTINGS.plot_multi_cmap if use_cmap else None
    # linestyles = ["-o" for x in args.result_files
                  # ] if args.plot_markers else None
    linestyles = ["-o" for x in results]

    # labels according to first dataset
    if "xlabel" in df.loc["info"].index and not df.loc[
            "info", "xlabel"].isnull().values.any():
        index_label = df.loc["info", "xlabel"][0]
    else:
        index_label = "$t$ (s)" if common_index else "index"
    metric_label = df.loc["info", "label"][0]

    plot_collection = plot.PlotCollection(first_title)
    # raw value plot
    fig_raw = plt.figure(figsize=figsize)
    # handle NaNs from concat() above
    error_df.interpolate(method="index").plot(
        ax=fig_raw.gca(), colormap=colormap, style=linestyles,
        title=first_title, alpha=SETTINGS.plot_trajectory_alpha)
    plt.xlabel(index_label)
    plt.ylabel(metric_label)
    plt.legend(frameon=True)
    plot_collection.add_figure("raw", fig_raw)

    # statistics plot
    if SETTINGS.plot_statistics:
        fig_stats = plt.figure(figsize=figsize)
        include = df.loc["stats"].index.isin(SETTINGS.plot_statistics)
        if any(include):
            df.loc["stats"][include].plot(kind="barh", ax=fig_stats.gca(),
                                          colormap=colormap, stacked=False)
            plt.xlabel(metric_label)
            plt.legend(frameon=True)
            plot_collection.add_figure("stats", fig_stats)

    # grid of distribution plots
    raw_tidy = pd.melt(error_df, value_vars=list(error_df.columns.values),
                       var_name="estimate", value_name=metric_label)
    col_wrap = 2 if len(results) <= 2 else math.ceil(
        len(results) / 2.0)
    dist_grid = sns.FacetGrid(raw_tidy, col="estimate", col_wrap=col_wrap)
    # TODO: see issue #98
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        dist_grid.map(sns.distplot, metric_label)  # fits=stats.gamma
    plot_collection.add_figure("histogram", dist_grid.fig)

    # box plot
    fig_box = plt.figure(figsize=figsize)
    ax = sns.boxplot(x=raw_tidy["estimate"], y=raw_tidy[metric_label],
                     ax=fig_box.gca())
    # ax.set_xticklabels(labels=[item.get_text() for item in ax.get_xticklabels()], rotation=30)
    plot_collection.add_figure("box_plot", fig_box)

    # violin plot
    fig_violin = plt.figure(figsize=figsize)
    ax = sns.violinplot(x=raw_tidy["estimate"], y=raw_tidy[metric_label],
                        ax=fig_violin.gca())
    # ax.set_xticklabels(labels=[item.get_text() for item in ax.get_xticklabels()], rotation=30)
    plot_collection.add_figure("violin_histogram", fig_violin)

    # if args.plot:
    plot_collection.show()
    if args.save_plot:
        logger.debug(SEP)
        plot_collection.export(args.save_plot,
                               confirm_overwrite=not args.no_warnings)
    if args.serialize_plot:
        logger.debug(SEP)
        plot_collection.serialize(args.serialize_plot,
                                  confirm_overwrite=not args.no_warnings)


