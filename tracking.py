#!/usr/bin/env python3

from __future__ import print_function
from evo.core  import trajectory, sync, metrics
from evo.tools import file_interface, plot
from datmo.msg import Track, TrackArray
import rosbag
from pylatex import Tabular 

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.figure as fg
import numpy as np
from cycler import cycler

                                                                                                                                
def associate_segments(traj, tracks):
    """Associate segments of an object trajectory as given by a DATMO system
    with the object´s reference trajectory

    :traj: Reference trajectory
    :tracks: All the tracks that got produced by the DATMO system
    :localization: The trajectory of the self-localization
    :returns: segments: The tracks that match to the reference trajectory
    :returns: traj_ref: The part of the reference trajectory that matches with
    tracks

    """
    matches = []
    for tr in tracks: # Find the best matching tracks to the object trajectory

        traj_ref, traj_est = sync.associate_trajectories(traj, tr,
                max_diff=0.01)
        traj_est, rot, tra, _ = trajectory.align_trajectory(
                traj_est, traj_ref, correct_scale=False, return_parameters=True)
        
        # print("calculating APE for track of length", len(tr.timestamps))
        data = (traj_ref, traj_est)
        ape_metric = metrics.APE(metrics.PoseRelation.translation_part)
        ape_metric.process_data(data)
        ape_statistics = ape_metric.get_all_statistics()
        
        tra_dif = (tra - loc_tra)
        print(tra)
        abs_tra_dif = abs((tra - loc_tra)[0]) + abs((tra - loc_tra)[1])
        translation = abs(tra[0]) + abs(tra[1])
        rot_dif = (rot - loc_rot)
        abs_rot_dif = 0
        for i in range(0,len(rot_dif)): abs_rot_dif += abs(rot_dif[i][0])+ abs(rot_dif[i][1]) +\
                abs(rot_dif[i][2])
        print(abs_tra_dif,abs_rot_dif)
        mismatch = abs_tra_dif + abs_rot_dif
        tuple = [traj_est, mismatch, traj_est.get_infos()['t_start (s)']]
        matches.append(tuple)

    matches.sort(key = lambda x: x[2])
    
    segments = [] #The parts of the trajectory are added to this list
    for m in matches:
        print(m[1])
        if m[1]<0.1: # if the mismatch is smaller than 1
           # print(m[0].get_statistics()['v_avg (m/s)'])
           segments.append(m[0]) 
           # print(m[0].get_infos()['t_start (s)'],m[0].get_infos()["path length (m)"])
           # print(m[0].get_statistics()['v_avg (m/s)'])
    if len(segments)==0:
        print("No matching segments")

    whole =trajectory.merge(segments)

    traj_ref, traj_est = sync.associate_trajectories(traj, whole, max_diff=0.01)
    traj_est, rot, tra, _ = trajectory.align_trajectory(
            traj_est, traj_ref, correct_scale=False, return_parameters=True)
    print(traj_est.get_infos())

    return segments, traj_ref, translation

def associate_segments_common_frame(traj, tracks):
    """Associate segments of an object trajectory as given by a DATMO system
    with the object´s reference trajectory

    :traj: Reference trajectory
    :tracks: All the tracks that got produced by the DATMO system
    :localization: The trajectory of the self-localization
    :returns: segments: The tracks that match to the reference trajectory
    :returns: traj_ref: The part of the reference trajectory that matches with
    tracks

    """
    matches = []
    for tr in tracks: # Find the best matching tracks to the object trajectory

        traj_ref, traj_est = sync.associate_trajectories(traj, tr,
                max_diff=0.01)
        
        # print("calculating APE for track of length", len(tr.timestamps))
        data = (traj_ref, traj_est)
        ape_metric = metrics.APE(metrics.PoseRelation.translation_part)
        ape_metric.process_data(data)
        ape_statistics = ape_metric.get_all_statistics()
        # print(ape_statistics)
        
        mismatch = ape_statistics['mean']
        # print(mismatch)
        tuple = [traj_est, mismatch, traj_est.get_infos()['t_start (s)']]
        matches.append(tuple)

    matches.sort(key = lambda x: x[2])
    
    segments = [] #The parts of the trajectory are added to this list
    for m in matches:
        if m[1]<5: # if the mismatch is smaller than 1
           # print(m[0].get_statistics()['v_avg (m/s)'])
           segments.append(m[0]) 
           # print(m[0].get_infos()['t_start (s)'],m[0].get_infos()["path length (m)"])
           # print(m[0].get_statistics()['v_avg (m/s)'])
    if len(segments)==0:
        print("No matching segments")

    return segments, traj_ref

def stats_to_latex_table(traj_ref, segments, idx, table):
    """Associate segments of an object trajectory as given by a DATMO system
    with the object´s reference trajectory

    :traj_ref: Reference trajectory
    :segments: All the segments of the robot trajectory
    :table: Latex table that the statistics get added to

    """
    whole =trajectory.merge(segments)

    traj_ref, traj_est = sync.associate_trajectories(traj_ref, whole, max_diff=0.01)

    data = (traj_ref, traj_est)
    ape_metric = metrics.APE(metrics.PoseRelation.translation_part)
    ape_metric.process_data(data)
    ape_statistics = ape_metric.get_all_statistics()

    # print(traj_est.get_infos())
    table.add_row((idx+1, round(ape_statistics["rmse"],3),
        round(ape_statistics["mean"],3),
        round(ape_statistics["median"],3),
        round(ape_statistics["std"],3),
        round(ape_statistics["min"],3),
        round(ape_statistics["max"],3),
        round(ape_statistics["sse"],3),))
    table.add_hline

def four_plots(idx, b, traj_ref, segments):
    """Generates four plots into Report

    :ref: PoseTrajectory3D object that is used as reference
    :est: PoseTrajectory3D object that is plotted against reference
    :table: Tabular object that is generated by Tabular('c c')
    :name: String that is used as name for file and table entry
    :returns: translation of reference against estimation

    """
    # [ Plot ] x,y,xy,yaw 
    fig, axarr = plt.subplots(2,2,figsize=(12,8))
    # fig.suptitle('Tracking - Vehicle ' + str(idx+1), fontsize=30)
    # fig.tight_layout()
    # print(len(b.timestamps),len(traj_ref.timestamps))
    plot.traj_fourplots(axarr, b,       '--', 'gray', 'original')
    plot.traj_fourplots(axarr, traj_ref, '-', 'gray', 'reference',1 ,b.timestamps[0])
    axarr[0,1].set_prop_cycle(cycler('color', ['c', 'm', 'y', 'k']) +
           cycler('lw', [1, 2, 3, 4]))
    color=iter(plt.cm.rainbow(np.linspace(0,1,len(segments))))
    style='-'
    for i, segment in enumerate(segments):
        c=next(color)
        label = "segment" + str(i+ 1)
        plot.traj_xy(axarr[0,0:2], segment, '-', c, label,1 ,b.timestamps[0])
        # print("seg0: ",len(segment.positions_xyz[:,0]),"seg1:"
                # ,len(segment.positions_xyz[:,1]))
        axarr[1,0].plot(segment.positions_xyz[:, 0],
                segment.positions_xyz[:,1])
        # axarr[1,0].plot(segment.positions_xyz[:, 0], segment.positions_xyz[:, 1], '-', c, 1)
    handles, labels = axarr[0,0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center',ncol =
            len(segments) + 2)
    plt.savefig("/home/kostas/results/latest/tracking" + str(idx+1) +".png",
            dpi = 100, bbox_inches='tight')
    plt.waitforbuttonpress(0)
    plt.close(fig)

def plot_dimensions(segments, reference, style='-', color='black', label="", alpha=1.0,
        start_timestamp=None):
    """
    plot a path/trajectory based on xy coordinates into an axis
    :param axarr: an axis array (for Length, Width)
                  e.g. from 'fig, axarr = plt.subplots(2)'
    :param traj: trajectory.PosePath3D or trajectory.PoseTrajectory3D object
    :param style: matplotlib line style
    :param color: matplotlib color
    :param label: label (for legend)
    :param alpha: alpha value for transparency
    """
    # [ Plot ] x,y,xy,yaw 
    fig, axarr = plt.subplots(2)

    if len(axarr) != 2:
        raise PlotException("expected an axis array with 2 subplots - got " +
                            str(len(axarr)))
    ylabels = ["$Length$ [m]", "$Width$ [m]"]
    axarr[0].set_prop_cycle(cycler('color', ['c', 'm', 'y', 'k']) +
           cycler('lw', [1, 2, 3, 4]))
    color=iter(plt.cm.rainbow(np.linspace(0,1,len(segments))))

    for i, segment in enumerate(segments):
        c=next(color)
        if isinstance(segment, trajectory.PoseTrajectory3D):
            # print("it came here")
            x = segment.timestamps - (segment.timestamps[0]
                                   if start_timestamp is None else start_timestamp)
            xlabel = "$Time$ [s]"
            # print("X: ",len(x), len(segment.platos) )
            # print(x)
            # print(segment.platos)
        else:
            print("itcame here")
            x = range(0, len(segments))
            xlabel = "index"

        axarr[0].plot(x, segment.length, style, color=c,
                      label=label, alpha=alpha)
        axarr[1].plot(x, segment.width, style, color=c,
                      label=label, alpha=alpha)

    axarr[0].set_ylabel(ylabels[0])
    axarr[1].set_ylabel(ylabels[1])
    axarr[0].set_xlabel(xlabel)
    axarr[1].set_xlabel(xlabel)

    plt.waitforbuttonpress(0)
    plt.close(fig)
