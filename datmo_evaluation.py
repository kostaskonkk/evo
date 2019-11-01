#!/usr/bin/env python3

from __future__ import print_function
from evo.core  import trajectory, sync, metrics
from evo.tools import file_interface, plot
from datmo.msg import Track, TrackArray
# import local
import rosbag
from pylatex import Tabular 

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.figure as fg
# import matplotlib.animation as animation
import numpy as np
from cycler import cycler
import sys # cli arguments in sys.argv
# import tikzplotlib
import tracking

# SMALL_SIZE  = 12
# MEDIUM_SIZE = 14
# BIGGER_SIZE = 25
# plt.rc('font',  size=SMALL_SIZE)          # controls default text sizes
# plt.rc('axes',  titlesize=BIGGER_SIZE)    # fontsize of the axes title
# plt.rc('axes',  labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
# plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
# plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
# plt.rc('legend',fontsize=SMALL_SIZE)      # legend fontsize
# plt.rc('figure',titlesize=BIGGER_SIZE)    # fontsize of the figure title
# plt.rcParams['xtick.direction'] = 'in'
# plt.rcParams['ytick.direction'] = 'in'
# plt.rcParams['grid.color'] = 'gray'
# plt.rcParams['grid.alpha'] = '0.5'
# plt.rcParams['axes.edgecolor'] = 'k'
# plt.rcParams['axes.facecolor'] = 'w'
# plt.rcParams['legend.edgecolor'] = 'k'
# plt.rcParams['legend.facecolor'] = 'w'

 # print(plt.rcParams.keys())
# print(plt.style.available)
# print (mpl.rcParams['axes.edgecolor'])

# bag = rosbag.Bag(sys.argv[1])
bag = rosbag.Bag("/home/kostas/results/exp.bag")
type_of_exp = 'experiment'
distance = 0.9

# bag = rosbag.Bag("/home/kostas/results/sim.bag")
# type_of_exp = 'simulation'
# distance = 3

bot= []
bot.append(file_interface.read_bag_trajectory(bag, '/robot_1'))
bot.append(file_interface.read_bag_trajectory(bag, '/robot_2'))

tracks = []
tracks.append(('mean'   , file_interface.read_TrackArray(bag, '/tracks/mean',3)))
tracks.append(('mean_kf', file_interface.read_TrackArray(bag,'/tracks/mean_kf', 3)))
tracks.append(('box'    , file_interface.read_TrackArray(bag,'/tracks/box',3)))
tracks.append(('box_ukf', file_interface.read_TrackArray(bag, '/tracks/box_ukf', 3)))

# tracks = {
    # 'mean'     : file_interface.read_TrackArray(bag, '/tracks/mean', 3),
    # 'mean_kf'  : file_interface.read_TrackArray(bag, '/tracks/mean_kf', 3),
    # 'box'      : file_interface.read_TrackArray(bag, '/tracks/box', 3),
    # 'box_ukf'  : file_interface.read_TrackArray(bag, '/tracks/box_ukf', 3)
# }
# mean = file_interface.read_TrackArray(bag, '/tracks/mean', 3)
# mean_kf = file_interface.read_TrackArray(bag, '/tracks/mean_kf', 3)
# box = file_interface.read_TrackArray(bag, '/tracks/box', 3)
# box_ukf = file_interface.read_TrackArray(bag, '/tracks/box_ukf', 3)
# mocap= file_interface.read_bag_trajectory(bag, '/mocap_pose')

# odom = file_interface.read_bag_trajectory(bag,'/odometry/wheel_imu')
# slam = file_interface.read_bag_trajectory(bag,'/poseupdate')
# fuse = file_interface.read_bag_trajectory(bag,'/odometry/map')
bag.close()

# loc_ref, loc_est = sync.associate_trajectories(mocap, fuse)
# loc_est, loc_rot, loc_tra, _ = trajectory.align_trajectory(loc_est, 
        # loc_ref, correct_scale=False, return_parameters=True)
# print(loc_tra)

table = Tabular('l c c c c c c c')
table.add_hline() 
table.add_row(('id', 'rmse', 'mean', 'median', 'std', 'min', 'max', 'sse'))
table.add_empty_row()

for reference in bot:

    # print("here")
    for track in tracks:
        print(track[0])
        # print(track(0))
        segments, traj_reference = \
            tracking.associate_segments_common_frame(reference,track[1],distance)
        tracking.four_plots(1, reference, traj_reference, segments, type_of_exp) 
        # tracking.stats_to_latex_table(traj_reference, segments, idx, table)

# for idx,b in enumerate(bot):

    # print("Calculations for track model", idx +1,"based on the mean of the cluster")
    # segments, traj_reference = tracking.associate_segments_common_frame(b,mean,distance)
    # tracking.four_plots(idx, b, traj_reference, segments, type_of_exp) 
    # tracking.stats_to_latex_table(traj_reference, segments, idx, table)

    # segments, traj_reference = tracking.associate_segments_common_frame(b,
            # mean_kf,distance)
    # tracking.four_plots(idx, b, traj_reference, segments, type_of_exp) 
    # tracking.velocities(idx, b, traj_reference, segments, type_of_exp) 
    # tracking.stats_to_latex_table(traj_reference, segments, idx, table)

    # for segment in segments:
        # print(segment.linear_vel)
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
    # segments, traj_ref = tracking.associate_segments_common_frame(b,filtered_tracks, distance)
    # tracking.four_plots(idx, b, traj_ref, segments, type_of_exp) 
    # tracking.stats_to_latex_table(traj_ref, segments, idx, table)

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
    # segments, traj_ref = tracking.associate_segments_common_frame(b,box_tracks, distance)
    # tracking.four_plots(idx, b, traj_ref, segments, type_of_exp) 
    # tracking.stats_to_latex_table(traj_ref, segments,idx, table)
    # tracking.plot_dimensions(segments, b, start_timestamp = b.timestamps[0])

    # print("Calculations for track model", idx +1,"based on the nonlinear observer")
    # segments, traj_ref = tracking.associate_segments_common_frame(b,obs_tracks)
    # tracking.four_plots(idx, b, traj_ref, segments, type_of_exp) 
    # tracking.stats_to_latex_table(traj_ref, segments,idx, table)

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


# table.generate_tex('/home/kostas/report/figures/tables/eval_table')
