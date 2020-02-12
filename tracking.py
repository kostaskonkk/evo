#!/usr/bin/env python3

from __future__ import print_function
from evo.core  import trajectory, sync, metrics
from evo.core.trajectory import PoseTrajectory3D
from evo.tools import file_interface, plot

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.figure as fg
import matplotlib.patches as mpatches
import numpy as np
import seaborn as sns
import itertools


def original_ape(traj_ref, traj_est, pose_relation, align=False, correct_scale=False,
        align_origin=False, ref_name="reference", est_name="estimate"):
    ''' Copied from main_ape.py
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
    ape_metric = metrics.APE(pose_relation)
    ape_metric.process_data(data)

    title = str(ape_metric)
    if align and not correct_scale:
        title += "\n(with SE(3) Umeyama alignment)"
    elif align and correct_scale:
        title += "\n(with Sim(3) Umeyama alignment)"
    elif only_scale:
        title += "\n(scale corrected)"
    elif align_origin:
        title += "\n(with origin alignment)"
    else:
        title += "\n(not aligned)"

    ape_result = ape_metric.get_result(ref_name, est_name)
    ape_result.info["title"] = title

    ape_result.add_trajectory(ref_name, traj_ref)
    ape_result.add_trajectory(est_name, traj_est)
    if isinstance(traj_est, trajectory.PoseTrajectory3D):
        seconds_from_start = [
            t - traj_est.timestamps[0] for t in traj_est.timestamps
        ]
        ape_result.add_np_array("seconds_from_start", seconds_from_start)
        ape_result.add_np_array("timestamps", traj_est.timestamps)

    return ape_result

def ape(traj_ref, traj_est, pose_relation, align=False, correct_scale=False,
        align_origin=False, ref_name="reference", est_name="estimate"):
    ''' Copied from main_ape.py
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
    ape_metric = metrics.APE(pose_relation)
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

def associate_segments(traj, tracks):
    """Associate segments of an object trajectory as given by a DATMO system
    with the object's reference trajectory

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
        # print(tra)
        abs_tra_dif = abs((tra - loc_tra)[0]) + abs((tra - loc_tra)[1])
        translation = abs(tra[0]) + abs(tra[1])
        rot_dif = (rot - loc_rot)
        abs_rot_dif = 0
        for i in range(0,len(rot_dif)): abs_rot_dif += abs(rot_dif[i][0])+ abs(rot_dif[i][1]) +\
                abs(rot_dif[i][2])
        # print(abs_tra_dif,abs_rot_dif)
        mismatch = abs_tra_dif + abs_rot_dif
        tuple = [traj_est, mismatch, traj_est.get_infos()['t_start (s)']]
        matches.append(tuple)

    matches.sort(key = lambda x: x[2])
    
    segments = [] #The parts of the trajectory are added to this list
    for m in matches:
        # print(m[1])
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
    # print(traj_est.get_infos())

    return segments, traj_ref, translation

def associate_segments_common_frame(traj, tracks, distance):
    """Associate segments of an object trajectory as given by a DATMO system
    with the object's reference trajectory

    :traj: Reference trajectory
    :tracks: All the tracks that got produced by the DATMO system
    :localization: The trajectory of the self-localization
    :returns: segments: The tracks that match to the reference trajectory
    :returns: traj_ref: The part of the reference trajectory that matches with
    tracks

    """
    matches = []

    for tr in tracks: # Find the best matching tracks to the object trajectory

        traj_ref, traj_est = sync.associate_trajectories(traj, tr, max_diff=0.1)
        # print("calculating APE for track of length", len(tr.timestamps))
        data = (traj_ref, traj_est)
        ape_metric = metrics.APE(metrics.PoseRelation.translation_part)
        ape_metric.process_data(data)
        ape_statistics = ape_metric.get_all_statistics()
        # print(ape_statistics)
        mismatch = ape_statistics['mean']
        # print(mismatch)
        tuple = [traj_est, mismatch, traj_est.get_infos()['t_start (s)'],
                traj_ref]
        matches.append(tuple)

    matches.sort(key = lambda x: x[2])
    segments_track = [] #The parts of the trajectory are added to this list
    segments_refer = [] #The parts of the reference trajectory are added to this list

    for m in matches:
        if m[1]<distance: # if the mismatch is smaller than 1
           # print(m[1],distance)
           # print(m[0].get_statistics()['v_avg (m/s)'])
           segments_track.append(m[0])
           segments_refer.append(m[3])
           # print(m[0].get_infos()['t_start (s)'],m[0].get_infos()["path length (m)"])
           # print(m[0].get_statistics()['v_avg (m/s)'])
    if len(segments_track)==0:
        print("No matching segments")

    traj_ref = trajectory.merge(segments_refer)
    # print(traj_ref.length)
    return segments_track, traj_ref

def stats_to_latex_table(traj_ref, segments, idx, table):
    """Associate segments of an object trajectory as given by a DATMO system
    with the object's reference trajectory

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

def four_plots(idx, b, traj_ref, segments, type_of_exp):
    """Generates four plots into Report

    :ref: PoseTrajectory3D object that is used as reference
    :est: PoseTrajectory3D object that is plotted against reference
    :table: Tabular object that is generated by Tabular('c c')
    :name: String that is used as name for file and table entry
    :returns: translation of reference against estimation

    """
    # [ Plot ] x,y,xy,yaw 
    # fig, axarr = plt.subplots(2,2,figsize=(12,8))
    fig, axarr = plt.subplots(2,2)
    # fig.suptitle('Tracking - Vehicle ' + str(idx+1), fontsize=30)
    # fig.tight_layout()
    # print(len(b.timestamps),len(traj_ref.timestamps))
    plot.traj_fourplots(axarr, b,       '*', 'gray', 'original')
    plot.traj_fourplots(axarr, traj_ref, '-', 'gray', 'reference',1 ,b.timestamps[0])
    # plot.traj_fourplots(axarr, traj_ref, '-', 'gray', 'reference')
    style='-'
    palette = itertools.cycle(sns.color_palette())
    for i, segment in enumerate(segments):
        c=next(color)
        label = "segment" + str(i+ 1)
        plot.traj_xy(axarr[0,0:2], segment, '-', c, label,1 ,b.timestamps[0])
        axarr[1,0].plot(segment.positions_xyz[:, 0],
                segment.positions_xyz[:,1])
        plot.traj_yaw(axarr[1,1],segment, style, c, None,1 ,b.timestamps[0])
    # handles, labels = axarr[0,0].get_legend_handles_labels()
    # fig.legend(handles, labels, loc='lower center',ncol =
            # len(segments) + 2)
    plt.savefig("/home/kostas/results/"+type_of_exp+"/tracking" + str(idx+1) +".png",
            dpi = 100, bbox_inches='tight')
    plt.waitforbuttonpress(0)
    plt.close(fig)

def screen(axarr, color, b, traj_ref, segments, method):
    """Generates plots for x, y and yaw onto an axarray

    :ref: PoseTrajectory3D object that is used as reference
    :est: PoseTrajectory3D object that is plotted against reference
    :table: Tabular object that is generated by Tabular('c c')
    :name: String that is used as name for file and table entry
    :returns: translation of reference against estimation

    """
    for i, segment in enumerate(segments):
        if i==0:
            plot.traj_xy(axarr[0,0:2], segment, '-', color, method,1 ,b.timestamps[0])
            plot.traj_yaw(axarr[0,2],segment, '-', color, None,1 ,b.timestamps[0])
        else:
            plot.traj_xy(axarr[0,0:2], segment, '-', color, None,1 ,b.timestamps[0])
            plot.traj_yaw(axarr[0,2],segment, '-', color, None, 1
                    ,b.timestamps[0], 6.28 )
        plot.linear_vel(axarr[1,0:2], segment, '-', color, method, 1, b.timestamps[0])
        angular_vel(axarr[1,2], segment, '-', color, method, 1, b.timestamps[0])

    axarr[0][0].set_xlim(left=0)
    axarr[0][1].set_xlim(left=0)
    axarr[0][2].set_xlim(left=0)
    axarr[1][0].set_xlim(left=0)
    axarr[1][1].set_xlim(left=0)
    axarr[1][2].set_xlim(left=0)

def merge(tracks):
    """
    Merges multiple tracks into a single, timestamp-sorted one.
    :param tracks: list of PoseTrajectory3D objects
    :return: merged PoseTrajectory3D
    """
    merged_stamps = np.concatenate([t.timestamps for t in tracks])
    merged_xyz = np.concatenate([t.positions_xyz for t in tracks])
    merged_length = np.concatenate([t.length for t in tracks])
    merged_width = np.concatenate([t.width for t in tracks])
    merged_angular_vel = np.concatenate([t.angular_vel for t in tracks])
    merged_linear_vel = np.concatenate([t.linear_vel for t in tracks])
    merged_angular_vel = np.concatenate([t.angular_vel for t in tracks])
    merged_quat = np.concatenate(
        [t.orientations_quat_wxyz for t in tracks])
    order = merged_stamps.argsort()
    merged_stamps = merged_stamps[order]
    merged_xyz = merged_xyz[order]
    merged_quat = merged_quat[order]
    merged_linear_vel = merged_linear_vel[order]
    merged_angular_vel = merged_angular_vel[order]
    merged_length = merged_length[order]
    merged_width = merged_width[order]
    return PoseTrajectory3D(merged_xyz, merged_quat, merged_stamps, linear_vel
            = merged_linear_vel, angular_vel = merged_angular_vel, length =
            merged_length, width = merged_width)

def angular_vel(ax, traj, style='-', color='black', label="", alpha=1.0,
        start_timestamp=None):
    """
    plots the angular velocity of a trajectory object 
    :param axarr: an axis array (for x, y)
                  e.g. from 'fig, axarr = plt.subplots(2)'
    :param traj: trajectory.PosePath3D or trajectory.PoseTrajectory3D object
    :param style: matplotlib line style
    :param color: matplotlib color
    :param label: label (for legend)
    :param alpha: alpha value for transparency
    :param start_timestamp: optional start time of the reference
                            (for x-axis alignment)
    """
    if isinstance(traj, trajectory.PoseTrajectory3D):
        x = traj.timestamps - (traj.timestamps[0]
                               if start_timestamp is None else start_timestamp)
        xlabel = "Time [s]"
    else:
        x = range(0, len(traj.positions_xyz - 1))
        xlabel = "index"
    ylabel = "$\omega$ (degrees/s)"
    angular_vel_degrees = [traj.angular_vel[i,2]* 57.14 
            for i in range(len(traj.angular_vel))]
    
    # ax.plot(x, traj.angular_vel[:,2], style, color=color, label=label, alpha=alpha)
    ax.plot(x, angular_vel_degrees, style, color=color, label=label, alpha=alpha)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)

def report_states(references, tracks, distance, filename):
    mpl.use('pgf')
    mpl.rcParams.update({
        "text.usetex": True,
        "pgf.texsystem": "pdflatex",})
    current_palette = sns.color_palette()
    sns.set_color_codes()
    # https://python-graph-gallery.com/100-calling-a-color-with-seaborn/
    # palette = itertools.cycle(sns.color_palette())

    for ref in references:
        fig_rep, axarr = plt.subplots(4,2,figsize=(6.125,8.6))
        # fig_rep, axarr = plt.subplots(3,2,figsize=(6.125,7))
        # fig_rep, axarr = plt.subplots(4,2,figsize=(7.14,8.8))

        for track in tracks:
            if(track[0]=='KF'):
                color = 'b'
            elif(track[0]=='UKF'):
                color = 'g'
            shape_color = 'indianred'
            segments, traj_ref = \
                associate_segments_common_frame(ref[1], track[1],distance)
            # color=next(palette)
            
            for i, segment in enumerate(segments):
                if i==0:
                    plot.traj_xy(axarr[0,0:2], segment, '-', color, track[0],1
                            ,ref[1].timestamps[0])
                    angular_vel(axarr[2,1], segment, '-', shape_color, 'Shape', 1,
                            ref[1].timestamps[0])
                else:
                    plot.traj_xy(axarr[0,0:2], segment, '-', color, None,1 ,ref[1].timestamps[0])
                    angular_vel(axarr[2,1], segment, '-', shape_color, None, 1,
                            ref[1].timestamps[0])

                plot.linear_vel(axarr[1,0:2], segment, '-', color, track[0],1
                        ,ref[1].timestamps[0])
                plot.traj_yaw(axarr[2,0],segment, '.', shape_color, None,1 ,ref[1].timestamps[0])
                plot.dimensions(axarr[3,0:2], segment, '-', shape_color, track[0], 1
                        ,ref[1].timestamps[0])

                ref_color = 'gray'
        plot.traj_xy(axarr[0,0:2], traj_ref, '-', ref_color, 'Reference', 1, ref[1].timestamps[0])
        plot.vx_vy(axarr[1,0:2], traj_ref, '-', ref_color, 'Reference', 1, ref[1].timestamps[0])
        plot.traj_yaw(axarr[2,0], traj_ref, '.',ref_color, None, 1, ref[1].timestamps[0])
        plot.angular_vel(axarr[2,1], traj_ref, '-', ref_color, None, 1, ref[1].timestamps[0])

        if filename.split('/')[0] == 'simulation':
            axarr[3,0].axhline(y=3.9, color='gray')
            axarr[3,1].axhline(y=1.78, color='gray')
        else:
            axarr[3,0].axhline(y=0.385, color='gray')
            axarr[3,1].axhline(y=0.2, color='gray')

        for i in range(0,4):
            for j in range(0,2):
                axarr[i,j].set_xlim(left=0)

        red = mpatches.Patch(color='indianred', label='Shape KF')
        gray = mpatches.Patch(color='gray', label='Reference')
        green = mpatches.Patch(color='b', label='KF')
        blue = mpatches.Patch(color='g', label='UKF')
        lgd = fig_rep.legend(handles=[green,blue,red,gray],\
                loc='lower center',ncol = 4, borderpad=0.7,\
                bbox_to_anchor=(0.54,0), columnspacing=0.8)
        fig_rep.tight_layout()
        fig_rep.subplots_adjust(bottom=0.11)
        fig_rep.savefig("/home/kostas/report/figures/"+ filename
                +ref[0]+".pgf",bbox_inches='tight')

def presentation_states(references, tracks, distance, filename):
    current_palette = sns.color_palette()
    sns.set_color_codes()

    for ref in references:
        fig_rep, axarr = plt.subplots(2,4,figsize=(19.2,10.8))

        for track in tracks:
            if(track[0]=='KF'):
                color = 'b'
            elif(track[0]=='UKF'):
                color = 'g'
            shape_color = 'indianred'
            segments, traj_ref = \
                associate_segments_common_frame(ref[1], track[1],distance)
            
            for i, segment in enumerate(segments):
                if i==0:
                    plot.traj_xy(axarr[0:2,0], segment, '-', color, track[0],1
                            ,ref[1].timestamps[0])
                    angular_vel(axarr[1,2], segment, '-', shape_color, 'Shape', 1,
                            ref[1].timestamps[0])
                else:
                    plot.traj_xy(axarr[0:2,0], segment, '-', color, None,1 ,ref[1].timestamps[0])
                    angular_vel(axarr[1,2], segment, '-', shape_color, None, 1,
                            ref[1].timestamps[0])
                plot.linear_vel(axarr[0:2,1], segment, '-', color, track[0],1
                        ,ref[1].timestamps[0])
                plot.traj_yaw(axarr[0,2],segment, '.', shape_color, None,1 ,ref[1].timestamps[0])
                plot.dimensions(axarr[0:2,3], segment, '-', shape_color, track[0], 1
                        ,ref[1].timestamps[0])

        plot.traj_xy(axarr[0:2,0], traj_ref, '-', 'gray', 'Reference', 1, ref[1].timestamps[0])
        plot.vx_vy(axarr[0:2,1], traj_ref, '-', 'gray', 'reference', 1, ref[1].timestamps[0])
        plot.traj_yaw(axarr[0,2], traj_ref, '.', 'gray', None, 1, ref[1].timestamps[0])
        plot.angular_vel(axarr[1,2], traj_ref, '-', 'gray', None, 1, ref[1].timestamps[0])

        if filename.split('/')[0] == 'simulation':
            axarr[0,3].axhline(y=3.9, color='gray')
            axarr[1,3].axhline(y=1.78, color='gray')
        else:
            axarr[0,3].axhline(y=0.4, color='gray')
            axarr[1,3].axhline(y=0.2, color='gray')

        for i in range(0,4):
            for j in range(0,2):
                axarr[j,i].set_xlim(left=0)

        red = mpatches.Patch(color='indianred', label='Shape KF')
        gray = mpatches.Patch(color='gray', label='Reference')
        green = mpatches.Patch(color='b', label='KF')
        blue = mpatches.Patch(color='g', label='UKF')
        lgd = fig_rep.legend(handles=[green,blue,red,gray],\
                loc='lower center',ncol = 4, borderpad=0.7,\
                bbox_to_anchor=(0.54,0), columnspacing=0.8)

        # handles, labels = axarr[0,0].get_legend_handles_labels()
        # lgd = fig_rep.legend(handles, labels, loc='lower center',ncol =
                # len(labels), borderpad=0.7)
        fig_rep.subplots_adjust(bottom=0.11)
        fig_rep.tight_layout()
        # plt.show()
        fig_rep.savefig("/home/kostas/Dropbox/presentation_final/figures/eight_plots.png",bbox_inches='tight',transparent=False)

def presentation_four_states(references, tracks, distance, filename):
    current_palette = sns.color_palette()
    sns.set_color_codes()

    fig_dynamic, ax_dyn = plt.subplots(2,2,figsize=(9.6,6.5))
    fig_shape,   ax_shape = plt.subplots(2,2,figsize=(9.6,6.5))
    for ref in references:

        for track in tracks:
            if(track[0]=='KF'):
                color = 'b'
            elif(track[0]=='UKF'):
                color = 'g'
            shape_color = 'indianred'
            segments, traj_ref = \
                associate_segments_common_frame(ref[1], track[1],distance)
            
            for i, segment in enumerate(segments):
                if i==0:
                    plot.traj_xy(ax_dyn[0,0:2], segment, '-', color, track[0],1
                            ,ref[1].timestamps[0])
                    angular_vel(ax_shape[0,1], segment, '-', shape_color, 'Shape', 1,
                            ref[1].timestamps[0])
                else:
                    plot.traj_xy(ax_dyn[0,0:2], segment, '-', color, None,1 ,ref[1].timestamps[0])
                    angular_vel(ax_shape[0,1], segment, '-', shape_color, None, 1,
                            ref[1].timestamps[0])
                plot.linear_vel(ax_dyn[1,0:2], segment, '-', color, track[0],1
                    ,ref[1].timestamps[0])
                plot.traj_yaw(ax_shape[0,0],segment, '-', shape_color, None,1 ,ref[1].timestamps[0])
                plot.dimensions(ax_shape[1,0:2], segment, '-', shape_color, track[0], 1
                        ,ref[1].timestamps[0])

                plot.traj_xy(ax_dyn[0,0:2], traj_ref, '-', 'gray', 'Reference', 1, ref[1].timestamps[0])
        plot.vx_vy(ax_dyn[1,0:2], traj_ref, '-', 'gray', 'reference', 1, ref[1].timestamps[0])
        plot.traj_yaw(ax_shape[0,0], traj_ref, '.', 'gray', None, 1, ref[1].timestamps[0])
        plot.angular_vel(ax_shape[0,1], traj_ref, '-', 'gray', None, 1, ref[1].timestamps[0])

        # if filename.split('/')[0] == 'simulation':
            # axarr[0,3].axhline(y=3.9, color='gray')
            # axarr[1,3].axhline(y=1.78, color='gray')
        # else:
        ax_shape[1,0].axhline(y=0.4, color='gray')
        ax_shape[1,1].axhline(y=0.2, color='gray')

        for i in range(0,2):
            for j in range(0,2):
                ax_dyn[j,i].set_xlim(left=0)
                ax_shape[j,i].set_xlim(left=0)

        red = mpatches.Patch(color='indianred', label='Shape Kalman Filter')
        gray = mpatches.Patch(color='gray', label='Reference')
        green = mpatches.Patch(color='b', label='Kalman Filter')
        blue = mpatches.Patch(color='g', label='Unscented Kalman Filter')
        lgd = fig_dynamic.legend(handles=[green,blue,gray],\
                loc='lower center',ncol = 3, borderpad=0.7,\
                bbox_to_anchor=(0.42,-0.009), columnspacing=0.8)
        # lgd = fig_dynamic.legend(handles=[green,blue,gray],\
                # loc='lower center',ncol = 3, borderpad=0.7,\
                 # columnspacing=0.8)
        lgd = fig_shape.legend(handles=[red,gray],\
                loc='lower center',ncol = 2, borderpad=0.7,\
                bbox_to_anchor=(0.42,-0.009), columnspacing=0.8)

        # fig_dynamic.subplots_adjust(bottom=0.11)
        # fig_shape.subplots_adjust(bottom=0.11)
        # fig_dynamic.tight_layout()
        # fig_shape.tight_layout()
        # plt.show()
        fig_dynamic.savefig("/home/kostas/Dropbox/final_presentation/figures/dynamic_plots.png",
                bbox_extra_artists=[lgd],bbox_inches='tight',transparent=False)
        fig_shape.savefig("/home/kostas/Dropbox/final_presentation/figures/shape_plots.png",bbox_extra_artists=[lgd], bbox_inches='tight',transparent=False)

def screen_states(references, tracks, distance):
    palette = itertools.cycle(sns.color_palette())
    for ref in references:
        fig, axarr = plt.subplots(2,3)
        # plot.traj_xyyaw(axarr[0,0:3], ref[1], '-', 'gray', 'reference',1 ,ref[1].timestamps[0])
        # plot.traj_vel  (axarr[1,0:3], ref[1], '-', 'gray')

        for track in tracks:
            segments, traj_ref = \
                associate_segments_common_frame(ref[1], track[1],distance)
            color=next(palette)
            screen(axarr, color, ref[1], traj_ref, segments, track[0])

        plot.traj_xyyaw(axarr[0,0:3], traj_ref, '-', 'gray', 'Reference',1
                ,ref[1].timestamps[0])
        plot.traj_vel  (axarr[1,0:3], traj_ref, '-', 'gray')

    fig.tight_layout()
    handles, labels = axarr[0,0].get_legend_handles_labels()
    lgd = fig.legend(handles, labels, loc='lower center',ncol = len(labels))
    plt.show()
    # fig.waitforbuttonpress(0)
