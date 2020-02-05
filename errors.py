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

try: #colorful errors
    import IPython.core.ultratb
except ImportError:
    # No IPython. Use default exception printing.
    pass
else:
    import sys
    sys.excepthook = IPython.core.ultratb.ColorTB()

if sys.version_info[0] >= 3 and sys.version_info[1] >= 4:
    ABC = abc.ABC
else:
    ABC = abc.ABCMeta('ABC', (), {})

class MetricsException(EvoException):
    pass

class StatisticsType(Enum):
    mean = "Mean"
    rmse = "RMSE"
    std = "STD"
    median = "Median"
    min = "Min"
    max = "Max"
    sse = "SSE"

class PoseRelation(Enum):
    full_transformation = "full transformation"
    translation_part = "translation part"
    rotation_part = "rotation part"
    rotation_angle_rad = "rotation angle in radians"
    rotation_angle_deg = "rotation angle in degrees"
    x = "x"
    y = "y"
    vx = "vx"
    vy = "vy"
    psi = "psi"
    omega = "omega"
    length = "length"
    width = "width"
    rx = "rx"
    ry = "ry"
    rlength = "rlength"
    rwidth = "rwidth"

class Unit(Enum):
    none = "unit-less"
    meters = "m"
    seconds = "s"
    degrees = "deg"
    radians = "rad"
    frames = "frames"
    speed = "m/s"
    angular_speed = "rad/s"

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
        result.add_stats(self.get_all_statistics())

        if hasattr(self, "error"):
            result.add_np_array("error_array", self.error)
            metric_name = self.pose_relation.value

        result.add_info({
            "title": str(self),
            "ref_name": ref_name,
            "est_name": est_name,
            "label": "{} {}".format(metric_name,
                                    "({})".format(self.unit.value))
        })

        return result

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

def smallestSignedAngleBetween(x, y):
    a = (x - y) % (2*math.pi)
    b = (y - x) % (2*math.pi)
    return -a if a < b else b

class APE(PE):
    """
    APE: absolute pose error
    metric for investigating the global consistency of a DATMO system
    """

    def __init__(self, pose_relation=PoseRelation.translation_part):
        self.pose_relation = pose_relation
        self.E = []
        self.error = []
        # self.error_x = []
        # self.error_y = []
        # self.error_vx = []
        # self.error_vy = []
        # self.error_psi = []
        # self.error_omega = []
        if pose_relation == PoseRelation.translation_part:
            self.unit = Unit.meters
        elif pose_relation == PoseRelation.rotation_angle_deg:
            self.unit = Unit.degrees
        elif pose_relation == PoseRelation.rotation_angle_rad:
            self.unit = Unit.radians
        elif pose_relation == PoseRelation.x:
            self.unit = Unit.meters
        elif pose_relation == PoseRelation.y:
            self.unit = Unit.meters
        elif pose_relation == PoseRelation.vx or pose_relation ==\
            PoseRelation.vy:
            self.unit = Unit.speed
        elif pose_relation == PoseRelation.psi:
            self.unit = Unit.radians
        elif pose_relation == PoseRelation.omega:
            self.unit = Unit.angular_speed
        else:
            self.unit = Unit.none  # dimension-less

    def __str__(self):
        title = "ASE "
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
        following the notation of the Kummerle paper.
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

        self.E = traj_est.positions_xyz - traj_ref.positions_xyz

        if self.pose_relation == PoseRelation.x:
            self.error = [np.linalg.norm(E_i) for E_i in self.E[:,0]]
        elif self.pose_relation == PoseRelation.y:
            self.error = [np.linalg.norm(E_i) for E_i in self.E[:,1]]
        elif self.pose_relation == PoseRelation.vx:
            dot_x = [
                trajectory.calc_velocity(traj_ref.positions_xyz[i,0],
                                      traj_ref.positions_xyz[i + 1,0],
                                      traj_ref.timestamps[i], traj_ref.timestamps[i + 1])
                for i in range(len(traj_ref.positions_xyz) - 1)]
            dot_x.append(dot_x[-1]) #last two velocities are given the same
            evx = traj_est.linear_vel[:,0] - dot_x
            self.error = [np.linalg.norm(E_i) for E_i in evx]
        elif self.pose_relation == PoseRelation.vy:
            dot_y = [
                trajectory.calc_velocity(traj_ref.positions_xyz[i,1],
                                      traj_ref.positions_xyz[i + 1,1],
                                      traj_ref.timestamps[i], traj_ref.timestamps[i + 1])
                for i in range(len(traj_ref.positions_xyz) - 1)]
            dot_y.append(dot_y[-1])
            evy = traj_est.linear_vel[:,1] - dot_y
            self.error = [np.linalg.norm(E_i) for E_i in evy]
        elif self.pose_relation == PoseRelation.vy:
            dot_y = [
                trajectory.calc_velocity(traj_ref.positions_xyz[i,1],
                                      traj_ref.positions_xyz[i + 1,1],
                                      traj_ref.timestamps[i], traj_ref.timestamps[i + 1])
                for i in range(len(traj_ref.positions_xyz) - 1)]
            dot_y.append(dot_y[-1])
        elif self.pose_relation == PoseRelation.psi:
            epsi =[]
            for i in range(0, len(traj_est.get_orientations_euler()[:,2])):
                epsi.append(smallestSignedAngleBetween(traj_est.get_orientations_euler()[i,2],
                    traj_ref.get_orientations_euler()[i,2]))

            epsi_deg = [i * 180 / math.pi for i in epsi]
            # epsi =  traj_est.get_orientations_euler()[:,2] - traj_ref.get_orientations_euler()[:,2]
            # self.error = [np.linalg.norm(E_i) for E_i in epsi]
            self.error = [np.linalg.norm(E_i) for E_i in epsi_deg]
        elif self.pose_relation == PoseRelation.omega:
            wrap = traj_ref.get_orientations_euler()[:,2]
            yaw_unwrapped = np.unwrap(wrap)
            dot_yaw = [
                trajectory.calc_angular_velocity_unwrapped(yaw_unwrapped[i],
                                      yaw_unwrapped[i + 1],
                                      traj_ref.timestamps[i], traj_ref.timestamps[i +
                                          1])
                for i in range(len(traj_ref.positions_xyz) - 1)]

            dot_yaw.append(dot_yaw[-1])

            # dot_yaw = [
                # trajectory.calc_angular_velocity(traj_ref.poses_se3[i],
                                      # traj_ref.poses_se3[i + 1],
                                      # traj_ref.timestamps[i], traj_ref.timestamps[i + 1])
                # for i in range(len(traj_ref.poses_se3) - 1)]
            # dot_yaw.append(dot_yaw[-1])

            eomega = traj_est.angular_vel[:,2] - dot_yaw
            eomega_deg = [i * 180 / math.pi for i in eomega]
            self.error = [np.linalg.norm(E_i) for E_i in eomega_deg]
        elif self.pose_relation == PoseRelation.length:
            ref_length = [0.385] * len(traj_est.length)
            elength = traj_est.length - ref_length
            self.error = [np.linalg.norm(E_i) for E_i in elength]
        elif self.pose_relation == PoseRelation.width:
            ref_width = [0.2] * len(traj_est.width)
            ewidth = traj_est.width - ref_width
            self.error = [np.linalg.norm(E_i) for E_i in ewidth]
        elif self.pose_relation == PoseRelation.rx:
            self.error = [np.linalg.norm(E_i)/0.292 for E_i in self.E[:,0]]
        elif self.pose_relation == PoseRelation.ry:
            self.error = [np.linalg.norm(E_i)/0.292 for E_i in self.E[:,1]]
        elif self.pose_relation == PoseRelation.rlength:
            ref_length = [0.385] * len(traj_est.length)
            relength = (traj_est.length - ref_length)/0.385
            self.error = [np.linalg.norm(E_i) for E_i in relength]
        elif self.pose_relation == PoseRelation.rwidth:
            ref_width = [0.2] * len(traj_est.width)
            rewidth = (traj_est.width - ref_width)/0.2
            self.error = [np.linalg.norm(E_i) for E_i in rewidth]
        else:
            raise MetricsException("unsupported pose_relation")

def stats(apes_x, apes_y, apes_vx, apes_vy, apes_psi,
        apes_omega, apes_length, apes_width, rpes_x, rpes_y, rpes_length,
        rpes_width, filename):

    import pandas as pd
    from evo.tools import pandas_bridge, plot
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import seaborn as sns


    current_palette = sns.color_palette()
    sns.set_color_codes("dark")
    # sns.palplot(current_palette)

    df_x = pd.DataFrame()
    df_y = pd.DataFrame()
    df_vx =pd.DataFrame()
    df_vy =pd.DataFrame()
    df_psi=pd.DataFrame()
    df_omega=pd.DataFrame()
    df_length=pd.DataFrame()
    df_width=pd.DataFrame()
    df_rx = pd.DataFrame()
    df_ry = pd.DataFrame()
    df_rlength=pd.DataFrame()
    df_rwidth=pd.DataFrame()

    # print(list(df_x.columns.values))

    for ape in apes_x:
        name = None
        df_x = pd.concat([df_x, pandas_bridge.result_to_df(ape, name)],
                       axis="columns")
    for ape in apes_y:
        name = None
        df_y = pd.concat([df_y, pandas_bridge.result_to_df(ape, name)],
                       axis="columns")
    for ape in apes_vx:
        name = None
        df_vx = pd.concat([df_vx, pandas_bridge.result_to_df(ape, name)],
                       axis="columns")
    for ape in apes_vy:
        name = None
        df_vy = pd.concat([df_vy, pandas_bridge.result_to_df(ape, name)],
                       axis="columns")
    for ape in apes_psi:
        name = None
        df_psi = pd.concat([df_psi, pandas_bridge.result_to_df(ape, name)],
                       axis="columns")
    for ape in apes_omega:
        name = None
        df_omega = pd.concat([df_omega, pandas_bridge.result_to_df(ape, name)],
                       axis="columns")
    for ape in apes_length:
        name = None
        df_length = pd.concat([df_length, pandas_bridge.result_to_df(ape, name)],
                       axis="columns")
    for ape in apes_width:
        name = None
        df_width = pd.concat([df_width, pandas_bridge.result_to_df(ape, name)],
                       axis="columns")
    for ape in rpes_x:
        name = None
        df_rx = pd.concat([df_rx, pandas_bridge.result_to_df(ape, name)],
                       axis="columns")
    for ape in rpes_y:
        name = None
        df_ry = pd.concat([df_ry, pandas_bridge.result_to_df(ape, name)],
                       axis="columns")
    for ape in rpes_width:
        name = None
        df_rwidth = pd.concat([df_rwidth, pandas_bridge.result_to_df(ape, name)],
                       axis="columns")
    for ape in rpes_length:
        name = None
        df_rlength = pd.concat([df_rlength, pandas_bridge.result_to_df(ape, name)],
                       axis="columns")
    # print(df_omega)


    mpl.use('pgf')
    mpl.rcParams.update({
        "text.usetex": True,
        "pgf.texsystem": "pdflatex",
    })

    setting = ["RMSE"]
    include = df_x.loc["stats"].index.isin(setting)
    # print(df_x.loc["stats"])
    # print(include)
    print("ape_x",df_x.loc["stats"][include])
    print("ape_y",df_y.loc["stats"][include])
    print("ape_vx",df_vx.loc["stats"][include])
    print("ape_vy",df_vy.loc["stats"][include])
    print("ape_psi",df_psi.loc["stats"][include])
    print("ape_omega",df_omega.loc["stats"][include])
    print("ape_length",df_length.loc["stats"][include])
    print("ape_width",df_width.loc["stats"][include])

    fig_stats, axarr = plt.subplots(4,2,figsize=(6.125,8.6))
    setting = ["Mean", "STD", "Max","Min","RMSE"]
    include = df_x.loc["stats"].index.isin(setting)
    
    # df_x.loc["stats"].reorder_levels(['STD','SSE','RMSE','Min','Median','Mean','Max'])
    # print(df_x.names)
    # df_x.reindex(['STD','SSE','RMSE','Min','Median','Mean','Max'],
            # level='stats')
    # df_x.loc["stats"][['STD','SSE','RMSE','Min','Median','Mean','Max']]
    # print(df_x.xs('stats'))

    dfx_stats = df_x.xs('stats')
    x_stats = dfx_stats.reindex(["Max","Min","STD","RMSE","SSE","Median","Mean"])
    x_stats.drop(index="SSE",inplace=True)

    dfy_stats = df_y.xs('stats')
    y_stats = dfx_stats.reindex(["Max","Min","STD","RMSE","SSE","Median","Mean"])
    y_stats.drop(index="SSE",inplace=True)

    dfvx_stats = df_vx.xs('stats')
    vx_stats = dfvx_stats.reindex(["Max","Min","STD","RMSE","SSE","Median","Mean"])
    vx_stats.drop(index="SSE",inplace=True)

    dfvy_stats = df_vy.xs('stats')
    vy_stats = dfvy_stats.reindex(["Max","Min","STD","RMSE","SSE","Median","Mean"])
    vy_stats.drop(index="SSE",inplace=True)

    dfpsi_stats = df_psi.xs('stats')
    psi_stats = dfpsi_stats.reindex(["Max","Min","STD","RMSE","SSE","Median","Mean"])
    psi_stats.drop(index="SSE",inplace=True)

    dfomega_stats = df_omega.xs('stats')
    omega_stats = dfomega_stats.reindex(["Max","Min","STD","RMSE","SSE","Median","Mean"])
    omega_stats.drop(index="SSE",inplace=True)

    dflength_stats = df_length.xs('stats')
    length_stats = dflength_stats.reindex(["Max","Min","STD","RMSE","SSE","Median","Mean"])
    length_stats.drop(index="SSE",inplace=True)

    dfwidth_stats = df_width.xs('stats')
    width_stats = dfwidth_stats.reindex(["Max","Min","STD","RMSE","SSE","Median","Mean"])
    width_stats.drop(index="SSE",inplace=True)

    dfrx_stats = df_rx.xs('stats')
    rx_stats = dfrx_stats.reindex(["Max","Min","STD","RMSE","SSE","Median","Mean"])
    rx_stats.drop(index="SSE",inplace=True)

    dfry_stats = df_ry.xs('stats')
    ry_stats = dfy_stats.reindex(["Max","Min","STD","RMSE","SSE","Median","Mean"])
    ry_stats.drop(index="SSE",inplace=True)

    dfrlength_stats = df_rlength.xs('stats')
    rlength_stats = dfrlength_stats.reindex(["Max","Min","STD","RMSE","SSE","Median","Mean"])
    rlength_stats.drop(index="SSE",inplace=True)

    dfrwidth_stats = df_rwidth.xs('stats')
    rwidth_stats = dfrwidth_stats.reindex(["Max","Min","STD","RMSE","SSE","Median","Mean"])
    rwidth_stats.drop(index="SSE",inplace=True)

    # setting = ["Mean", "STD", "Max","Min","RMSE"]
    # include = df_x.loc["stats"].index.isin(setting)
    # print(x_stats)


    # x_stats.plot(kind="barh", ax =  axarr[0,0],legend =None)
    # axarr[0,0].set_xlabel("Absolute error $x$ (m)")
    # y_stats.plot(kind="barh", ax =  axarr[0,1], legend=None)
    # axarr[0,1].set_xlabel("Absolute error $y$ (m)")
    rx_stats.plot(kind="barh", ax =  axarr[0,0],legend =None)
    axarr[0,0].set_xlabel("Relative error $x$")
    ry_stats.plot(kind="barh", ax =  axarr[0,1], legend=None)
    axarr[0,1].set_xlabel("Relative error $y$")
    vx_stats.plot(kind="barh", ax =  axarr[1,0], legend=None)
    axarr[1,0].set_xlabel("Absolute error $v_x$ (m/s)")
    vy_stats.plot(kind="barh", ax =  axarr[1,1], legend=None)
    axarr[1,1].set_xlabel("Absolute error $v_y$ (m/s)")
    psi_stats.plot(kind="barh", width =0.3, ax = axarr[2,0],
            legend=None, color='indianred')
    axarr[2,0].set_xlabel("Absolute error $\psi$ (degrees)")
    omega_stats.plot(kind="barh", width =0.3, ax =  axarr[2,1],
            legend=None, color='indianred')
    axarr[2,1].set_xlabel("Absolute error $\omega$ (degrees/s)")
    length_stats.plot(kind="barh", width =0.3, ax =  axarr[3,0],
    axarr[2,1].set_xlabel("Absolute error $\dot{\psi}$ (degrees/s)")
    # length_stats.plot(kind="barh", width =0.3, ax =  axarr[3,0],
            # legend=None, color='indianred')
    # axarr[3,0].set_xlabel("Absolute error Length (m)")
    # width_stats.plot(kind="barh", width =0.3, ax =  axarr[3,1],
            # legend=None, color='indianred')
    # axarr[3,1].set_xlabel("Absolute error Width (m)")
    rlength_stats.plot(kind="barh", width =0.3, ax =  axarr[3,0],
            legend=None, color='indianred')
    axarr[3,0].set_xlabel("Relative error Length")
    rwidth_stats.plot(kind="barh", width =0.3, ax =  axarr[3,1],
            legend=None, color='indianred')
    axarr[3,1].set_xlabel("Relative error Width")

    # handles, labels = axarr[0,0].get_legend_handles_labels()
    # lgd = fig_stats.legend(handles, labels, loc='lower center',ncol = len(labels))
    current_palette = sns.color_palette()
    sns.set_color_codes()
    red = mpatches.Patch(color='indianred', label='Shape KF')
    gray = mpatches.Patch(color='gray', label='Reference')
    green = mpatches.Patch(color='b', label='KF')
    blue = mpatches.Patch(color='g', label='UKF')
    lgd = fig_stats.legend(handles=[green,blue,red,gray],\
            loc='lower center',ncol = 4, borderpad=0.7,\
            bbox_to_anchor=(0.54,0), columnspacing=0.8)
    fig_stats.tight_layout()
    fig_stats.subplots_adjust(bottom=0.12)
    # plt.show()

    fig_stats.savefig("/home/kostas/report/figures/"+filename+"_stats.pgf")

    # error_x = pd.DataFrame(df_x.loc["np_arrays", "error_array"].tolist()).T
    # error_y = pd.DataFrame(df_y.loc["np_arrays", "error_array"].tolist()).T
    # error_vx = pd.DataFrame(df_vx.loc["np_arrays", "error_array"].tolist()).T
    # error_vy = pd.DataFrame(df_vy.loc["np_arrays", "error_array"].tolist()).T
    # error_psi = pd.DataFrame(df_psi.loc["np_arrays", "error_array"].tolist()).T
    # error_omega = pd.DataFrame(df_omega.loc["np_arrays", "error_array"].tolist()).T
    # error_length = pd.DataFrame(df_length.loc["np_arrays", "error_array"].tolist()).T
    # error_width = pd.DataFrame(df_width.loc["np_arrays", "error_array"].tolist()).T


    # print(error_x)
    # print(df_x.loc["np_arrays", "error_array"].tolist())

    # plot_collection = plot.PlotCollection(first_title)
    # raw value plot
    # fig_raw, ax_raw  = plt.subplots(3,2,figsize=(6.125,7))
    # handle NaNs from concat() above
    # error_df.interpolate(method="index").plot(
        # ax=fig_raw.gca(), colormap=colormap, style=linestyles,
        # title=first_title, alpha=SETTINGS.plot_trajectory_alpha)
    # print(df_x)

    # t = df_x.loc["np_arrays", "seconds_from_start"].tolist() 
    # print(t)
    # ax_raw[0,0].plot(t,df_x.loc["np_arrays", "error_array"].tolist(), linestyle='-')
    # plt.show()

    # error_x.interpolate(method="index").plot(ax=ax_raw[0,0], legend=None, alpha=1)
    # error_y.interpolate(method="index").plot(ax=ax_raw[0,1], legend=None, alpha=1)
    # error_vx.interpolate(method="index").plot(ax=ax_raw[1,0], legend=None, alpha=1)
    # error_vy.interpolate(method="index").plot(ax=ax_raw[1,1], legend=None, alpha=1)
    # error_psi.interpolate(method="index").plot(ax=ax_raw[2,0], legend=None, alpha=1)
    # error_omega.interpolate(method="index").plot(ax=ax_raw[2,1], legend=None, alpha=1)
    # ax_raw[0,0].set_ylabel("Absolute error $x$ (m)")
    # ax_raw[0,1].set_ylabel("Absolute error $y$ (m)")
    # ax_raw[1,0].set_ylabel("Absolute error $v_y$ (m/s)")
    # ax_raw[1,1].set_ylabel("Absolute error $v_x$ (m/s)")
    # ax_raw[2,1].set_ylabel("Absolute error $\dot{\psi}$ (rad/s)")
    # ax_raw[2,0].set_ylabel("Absolute error $\psi$ (rad)")
    # handles, labels = ax_raw[0,0].get_legend_handles_labels()
    # lgd = fig_raw.legend(handles, labels, loc='lower center',ncol = len(labels))
    # fig_raw.tight_layout()
    # fig_raw.subplots_adjust(bottom=0.13)
    # plt.show()

    # plt.legend(frameon=True)
    # plot_collection.add_figure("raw", fig_raw)
    # statistics plot
