#!/usr/bin/env python3

from __future__ import print_function
from evo.core  import trajectory, sync, metrics
from evo.tools import file_interface, plot
from datmo.msg import Track, TrackArray
import local
import rosbag
from pylatex import Tabular 

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.figure as fg
# import matplotlib.animation as animation
import numpy as np
from cycler import cycler
import sys # cli arguments in sys.argv
import tikzplotlib

# Copied from main_ape.py
def ape(traj_ref, traj_est, pose_relation, align=False, correct_scale=False,
        align_origin=False, ref_name="reference", est_name="estimate"):
    
    traj_ref, traj_est = sync.associate_trajectories(traj_ref, traj_est)
    # Align the trajectories.
    only_scale = correct_scale and not align
    if align or correct_scale:
        # logger.debug(SEP)
        traj_est = trajectory.align_trajectory(traj_est, traj_ref,
                                               correct_scale, only_scale)
    elif align_origin:
        # logger.debug(SEP)
        traj_est = trajectory.align_trajectory_origin(traj_est, traj_ref)

    # Calculate APE.
    # logger.debug(SEP)
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

    # logger.debug(SEP)
    # logger.info(ape_result.pretty_str())

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

        traj_ref, traj_est = sync.associate_trajectories(traj, tr, max_diff=0.01)
        traj_est, rot, tra, _ = trajectory.align_trajectory(
                traj_est, traj_ref, correct_scale=False, return_parameters=True)
        
        # print("calculating APE for track of length", len(tr.timestamps))
        data = (traj_ref, traj_est)
        ape_metric = metrics.APE(metrics.PoseRelation.translation_part)
        ape_metric.process_data(data)
        ape_statistics = ape_metric.get_all_statistics()
        
        tra_dif = (tra - loc_tra)
        # print(tra_dif)
        abs_tra_dif = abs((tra - loc_tra)[0]) + abs((tra - loc_tra)[1])
        translation = abs(tra[0]) + abs(tra[1])
        rot_dif = (rot - loc_rot)
        abs_rot_dif = 0
        for i in range(0,len(rot_dif)): abs_rot_dif += abs(rot_dif[i][0])+ abs(rot_dif[i][1]) +\
                abs(rot_dif[i][2])
        mismatch = abs_tra_dif + abs_rot_dif
        tuple = [traj_est, mismatch, traj_est.get_infos()['t_start (s)']]
        matches.append(tuple)

    matches.sort(key = lambda x: x[2])
    
    segments = [] #The parts of the trajectory are added to this list
    for m in matches:
        if m[1]<0.8: # if the mismatch is smaller than 1
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



SMALL_SIZE = 12
MEDIUM_SIZE = 14
BIGGER_SIZE = 25
plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)    # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=MEDIUM_SIZE)   # fontsize of the tick labels
plt.rc('ytick', labelsize=MEDIUM_SIZE)   # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
# print(plt.rcParams.keys())
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['grid.color'] = 'gray'
plt.rcParams['grid.alpha'] = '0.5'
plt.rcParams['axes.edgecolor'] = 'k'
plt.rcParams['axes.facecolor'] = 'w'
plt.rcParams['legend.edgecolor'] = 'k'
plt.rcParams['legend.facecolor'] = 'w'
# print(plt.style.available)
# print (mpl.rcParams['axes.edgecolor'])

bag = rosbag.Bag(sys.argv[1])

bot= []
bot.append(file_interface.read_bag_trajectory(bag, '/robot_1'))
bot.append(file_interface.read_bag_trajectory(bag, '/robot_2'))
mean = file_interface.read_TrackArray(bag, '/tracks', 3)
filtered_tracks = file_interface.read_TrackArray(bag, '/filtered_tracks', 3)
# box_tracks = file_interface.read_TrackArray(bag, '/box_tracks', 3)
mocap= file_interface.read_bag_trajectory(bag, '/mocap_pose')
odom = file_interface.read_bag_trajectory(bag,'/odometry/wheel_imu')
slam = file_interface.read_bag_trajectory(bag,'/poseupdate')
fuse = file_interface.read_bag_trajectory(bag,'/odometry/map')
bag.close()

loc_table = Tabular('l c c c c c c c')
loc_table.add_hline()
loc_table.add_row(('method','rmse', 'mean', 'median', 'std', 'min', 'max', 'sse'))
loc_table.add_hline() 
loc_table.add_empty_row()

results = []
odom_result = ape(
    traj_ref=mocap,
    traj_est=odom,
    pose_relation=metrics.PoseRelation.translation_part,
    align=False,
    correct_scale=False,
    align_origin=True,
    ref_name="mocap",
    est_name="odom",
)
results.append(odom_result)
# file_interface.save_res_file("/home/kostas/results/res_files/odom", odom_result, True)

slam_result = ape(
    traj_ref=mocap,
    traj_est=slam,
    pose_relation=metrics.PoseRelation.translation_part,
    align=False,
    correct_scale=False,
    align_origin=True,
    ref_name="mocap",
    est_name="slam",
)
results.append(slam_result)
# file_interface.save_res_file("/home/kostas/results/res_files/slam", slam_result, True)

fuse_result = ape(
    traj_ref=mocap,
    traj_est=fuse,
    pose_relation=metrics.PoseRelation.translation_part,
    align=False,
    correct_scale=False,
    align_origin=True,
    ref_name="mocap",
    est_name="fuse",
)
results.append(fuse_result)
# file_interface.save_res_file("/home/kostas/results/res_files/fuse", fuse_result, True)
# convert_results_to_dataframe(results)

local.four_plots(mocap ,odom, loc_table, 'odometry')
local.four_plots(mocap ,slam, loc_table, 'slam')
local.four_plots(mocap ,fuse, loc_table, 'fusion')
loc_table.generate_tex('/home/kostas/report/figures/tables/loc_table')

loc_ref, loc_est = sync.associate_trajectories(mocap, fuse)
loc_est, loc_rot, loc_tra, _ = trajectory.align_trajectory(loc_est, 
        loc_ref, correct_scale=False, return_parameters=True)

table = Tabular('l c c c c c c c')
table.add_hline() 
table.add_row(('id', 'rmse', 'mean', 'median', 'std', 'min', 'max', 'sse'))
table.add_empty_row()
# print(len(filtered_tracks),len(tracks))
# for tr in filtered_tracks:
    # print(tr.linear_vel)

# plot_collection = plot.PlotCollection("System Evaluation")
def four_plots(idx, b, traj_ref, segments):
    """Generates four plots into Report

    :ref: PoseTrajectory3D object that is used as reference
    :est: PoseTrajectory3D object that is plotted against reference
    :table: Tabular object that is generated by Tabular('c c')
    :name: String that is used as name for file and table entry
    :returns: translation of reference against estimation

    """
    # [ Plot ] x,y,xy,yaw 
    fig, axarr = plt.subplots(2,2)
    fig.suptitle('Tracking - Vehicle ' + str(idx+1), fontsize=30)
    fig.tight_layout()
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
    # tikzplotlib.save(show_info = True, figure = fig,filepath = "/home/kostas/results/latest/tracking" + str(idx+1) +".tex")
    plt.savefig("/home/kostas/results/latest/tracking" + str(idx+1) +".png" , bbox_inches='tight')
    plt.waitforbuttonpress(0)
    plt.close(fig)

# for idx,b in enumerate(bot):

    # print("Calculations for track model", idx +1,"based on the mean of the cluster")
    # segments, traj_ref, mean_translation = associate_segments(b,mean)
    # # four_plots(idx, b, traj_ref, segments) 

    # whole =trajectory.merge(segments)
    # mean_result = ape(
        # traj_ref=traj_ref,
        # traj_est=whole,
        # pose_relation=metrics.PoseRelation.translation_part,
        # align=False,
        # correct_scale=False,
        # align_origin=False,
        # ref_name="mocap",
        # est_name="mean_track" + str(idx+1),
    # )
    # file_interface.save_res_file("/home/kostas/results/res_files/mean_track" +
            # str(idx+1), mean_result, False)

    # print("Calculations for track model", idx +1,"based on the l_shape")
    # segments, traj_ref, lshape_translation = associate_segments(b,filtered_tracks)
    # # four_plots(idx, b, traj_ref, segments) 

    # whole =trajectory.merge(segments)
    # mean_result = ape(
        # traj_ref=traj_ref,
        # traj_est=whole,
        # pose_relation=metrics.PoseRelation.translation_part,
        # align=False,
        # correct_scale=False,
        # align_origin=False,
        # ref_name="mocap",
        # est_name="l_shape" + str(idx+1),
    # )
    # file_interface.save_res_file("/home/kostas/results/res_files/l-shape_track" +
            # str(idx+1), mean_result, False)

    # print("Calculations for track model", idx +1,"based on the center of the bounding box")
    # segments, traj_ref, center_translation = associate_segments(b,box_tracks)
    # # four_plots(idx, b, traj_ref, segments) 

    # whole =trajectory.merge(segments)
    # center_result = ape(
        # traj_ref=traj_ref,
        # traj_est=whole,
        # pose_relation=metrics.PoseRelation.translation_part,
        # align=False,
        # correct_scale=False,
        # align_origin=False,
        # ref_name="mocap",
        # est_name="center_track" + str(idx+1),
    # )
    # file_interface.save_res_file("/home/kostas/results/res_files/center_track" +
            # str(idx+1), center_result, False)

    # print("mean_translation: ", mean_translation,"l-shape_translation:\
            # ",lshape_translation,"center_translation: ",center_translation)

    # [ Plot ] xyyaw data
    # fig, axarr = plt.subplots(3)
    # fig.suptitle('Tracking - Vehicle ' + str(idx+1), fontsize=30)
    # fig.tight_layout()
    # print(len(b.timestamps),len(traj_ref.timestamps))
    # plot.traj_xyyaw(axarr, b,       '--', 'gray', 'original')
    # plot.traj_xyyaw(axarr, traj_ref, '-', 'gray', 'reference',1 ,b.timestamps[0])
    # axarr[0].set_prop_cycle(cycler('color', ['c', 'm', 'y', 'k']) +
           # cycler('lw', [1, 2, 3, 4]))
    # color=iter(plt.cm.rainbow(np.linspace(0,1,len(segments))))
    # for i, segment in enumerate(segments):
        # c=next(color)
        # label = "segment" + str(idx + 1)
        # plot.traj_xy(axarr[0:2], segment, '-', c, label,1 ,b.timestamps[0])
    # plt.savefig("/home/kostas/results/latest/tracking" + str(idx+1) +".png" , bbox_inches='tight')
    # plt.waitforbuttonpress(0)
    # plt.close(fig)
    # # table.add_row((idx+1, round(ape_statistics["rmse"],3),
        # round(ape_statistics["mean"],3),
        # round(ape_statistics["median"],3),
        # round(ape_statistics["std"],3),
        # round(ape_statistics["min"],3),
        # round(ape_statistics["max"],3),
        # round(ape_statistics["sse"],3),))
    # table.add_hline

    # # plot velocities
    # name = 'bot1'
    # fig, axarr = plt.subplots(3)
    # fig.tight_layout()
    # fig.suptitle('Speed Estimation ' + name, fontsize=30)
    # plot.traj_vel(axarr, b, '--', 'gray', 'original')
    # plot.traj_vel(axarr, traj_ref, '-', 'gray', 'reference',1 ,b.timestamps[0])
    # axarr[0].set_prop_cycle(cycler('color', ['c', 'm', 'y', 'k']) +
           # cycler('lw', [1, 2, 3, 4]))
    # color=iter(plt.cm.rainbow(np.linspace(0,1,len(segments))))
    # for i, segment in enumerate(segments):
        # c=next(color)
        # label = "segment" + str(idx + 1)
        # plot.linear_vel(axarr[0:2], segment, '-', c, label,1 ,b.timestamps[0])
    # fig.subplots_adjust(hspace = 0.2)
    # plt.waitforbuttonpress(0)
    # plt.savefig("/home/kostas/results/latest/velocity"+name+".png",  format='png', bbox_inches='tight')

# for idx,b in enumerate(bot):
    # print("Calculations for track model", idx +1)
    # matches = []
    # for tr in box_tracks: # Find the best matching tracks to the bot trajectory

        # traj_ref, traj_est = sync.associate_trajectories(b, tr, max_diff=0.01)
        # traj_est, rot, tra, _ = trajectory.align_trajectory(
                # traj_est, traj_ref, correct_scale=False, return_parameters=True)
        
        # # print("calculating APE for track of length", len(tr.timestamps))
        # data = (traj_ref, traj_est)
        # ape_metric = metrics.APE(metrics.PoseRelation.translation_part)
        # ape_metric.process_data(data)
        # ape_statistics = ape_metric.get_all_statistics()
        
        # tra_dif = (tra - loc_tra)
        # # print(tra_dif)
        # abs_tra_dif = abs((tra - loc_tra)[0]) + abs((tra - loc_tra)[1])
        # rot_dif = (rot - loc_rot)
        # abs_rot_dif = 0
        # for i in range(0,len(rot_dif)): abs_rot_dif += abs(rot_dif[i][0])+ abs(rot_dif[i][1]) +\
                # abs(rot_dif[i][2])
        # mismatch = abs_tra_dif + abs_rot_dif
        # tuple = [traj_est, mismatch, traj_est.get_infos()['t_start (s)']]
        # matches.append(tuple)

    # matches.sort(key = lambda x: x[2])
    
    # segments = [] #The parts of the trajectory are added to this list
    # for m in matches:
        # if m[1]<0.8: # if the mismatch is smaller than 1
           # # print(m[0].get_statistics()['v_avg (m/s)'])
           # segments.append(m[0]) 
           # # print(m[0].get_infos()['t_start (s)'],m[0].get_infos()["path length (m)"])
           # # print(m[0].get_statistics()['v_avg (m/s)'])
    # if len(segments)==0:
        # print("No matching segments")
        # continue
    # whole =trajectory.merge(segments)
        
    # traj_ref, traj_est = sync.associate_trajectories(b, whole, max_diff=0.01)
    # traj_est, rot, tra, _ = trajectory.align_trajectory(
            # traj_est, traj_ref, correct_scale=False, return_parameters=True)
    # # print(traj_est.get_infos())

    # # [ Plot ] xyyaw data
    # fig, axarr = plt.subplots(3)
    # fig.suptitle('Tracking - Vehicle ' + str(idx+1), fontsize=30)
    # fig.tight_layout()
    # print(len(b.timestamps),len(traj_ref.timestamps))
    # plot.traj_xyyaw(axarr, b,       '--', 'gray', 'original')
    # # plot.traj_xyyaw(axarr, traj_ref, '-', 'gray', 'reference',1 ,b.timestamps[0])
    # axarr[0].set_prop_cycle(cycler('color', ['c', 'm', 'y', 'k']) +
           # cycler('lw', [1, 2, 3, 4]))
    # color=iter(plt.cm.rainbow(np.linspace(0,1,len(segments))))
    # for i, segment in enumerate(segments):
        # c=next(color)
        # label = "segment" + str(idx + 1)
        # plot.traj_xy(axarr[0:2], segment, '-', c, label,1 ,b.timestamps[0])
    # plt.savefig("/home/kostas/results/latest/box_tracking" + str(idx+1) +".png" , bbox_inches='tight')
    # plt.waitforbuttonpress(0)
    # # plt.close(fig)
    # table.add_row((idx+1, round(ape_statistics["rmse"],3),
        # round(ape_statistics["mean"],3),
        # round(ape_statistics["median"],3),
        # round(ape_statistics["std"],3),
        # round(ape_statistics["min"],3),
        # round(ape_statistics["max"],3),
        # round(ape_statistics["sse"],3),))
    # table.add_hline

    # fig_xy, ax_xy = plt.subplots(1)
    # plot.xy(ax_xy, b,       '--', 'gray', 'original')
    # plot.xy(ax_xy, traj_ref, '-', 'gray', 'reference')
    # # plot.xy(ax_xy, traj_est, '-', 'black', 'estimation')
    # ax_xy.set_prop_cycle(cycler('color', ['c', 'm', 'y', 'k']) +
           # cycler('lw', [1, 2, 3, 4]))
    # color=iter(plt.cm.rainbow(np.linspace(0,1,len(segments))))
    # for idx, segment in enumerate(segments):
        # c=next(color)
        # label = "segment" + str(idx + 1)
        # plot.xy(ax_xy, segment, '--', c, label)
    # plt.waitforbuttonpress(0)
    # plt.close(fig_xy)

    # fig, ax = plt.subplots(1)
    # plot.traj_yaw(ax, traj_ref)
    # plt.waitforbuttonpress(0)
    # plt.close(fig)
    # ax.plot(x, y, style, color=color, label=label, alpha=alpha)
    # plot.traj(ax, plot_mode, b, '--', 'gray', 'original')
    # plot.traj(ax, plot_mode, traj_ref, '-', 'gray', 'reference')
    # plot.traj(ax, plot_mode, traj_est, '-', 'red', 'estimation')
    # plot_collection.add_figure(str(idx),fig_2)
    # plt.grid(True)
    # plt.title('Trajectory of vehicle %i, Number of segments %i' %(idx,\
            # len(segments)))
    # plt.savefig("/home/kostas/report/figures/eval/eval.png",format='png', bbox_inches='tight')
    # plt.show()
    # table.add_row((idx, round(ape_statistics["rmse"],3),
        # round(ape_statistics["mean"],3),
        # round(ape_statistics["median"],3),
        # round(ape_statistics["std"],3),
        # round(ape_statistics["min"],3),
        # round(ape_statistics["max"],3),
        # round(ape_statistics["sse"],3),))
    # table.add_hline

# plot_collection.show()
table.generate_tex('/home/kostas/report/figures/tables/eval_table')
# plot_collection = plot.PlotCollection("Localization")
# # # metric values
# fig_1 = plt.figure(figsize=(8, 8))
# plot.error_array(fig_1, ape_metric.error, statistics=ape_statistics,
                 # name="APE", title=str(ape_metric))
# plot_collection.add_figure("raw", fig_1)
# # plt.show()
# # trajectory colormapped with error

# fig_2 = plt.figure(figsize=(8, 8))
# plot_mode = plot.PlotMode.xy
# ax = plot.prepare_axis(fig_2, plot_mode)
# plot.traj(ax, plot_mode, traj_ref, '--', 'gray', 'reference')
# plot.traj_colormap(
    # ax, traj_est, ape_metric.error, plot_mode, min_map=ape_statistics["min"],
    # max_map=ape_statistics["max"], title="APE mapped onto trajectory")
# plot_collection.add_figure("traj (error)", fig_2)

# # trajectory colormapped with speed
# fig_3 = plt.figure(figsize=(8, 8))
# plot_mode = plot.PlotMode.xy
# ax = plot.prepare_axis(fig_3, plot_mode)
# speeds = [
    # trajectory.calc_speed(traj_est.positions_xyz[i],
                          # traj_est.positions_xyz[i + 1],
                          # traj_est.timestamps[i], traj_est.timestamps[i + 1])
    # for i in range(len(traj_est.positions_xyz) - 1)
# ]
# speeds.append(0)
# plot.traj(ax, plot_mode, traj_ref, '--', 'gray', 'reference')
# plot.traj_colormap(ax, traj_est, speeds, plot_mode, min_map=min(speeds),
                   # max_map=max(speeds), title="speed mapped onto trajectory")
# fig_3.axes.append(ax)
# plot_collection.add_figure("traj (speed)", fig_3)

