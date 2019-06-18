#!/usr/bin/env python

from __future__ import print_function
from evo.core  import trajectory, sync, metrics
from evo.tools import file_interface, plot
from datmo.msg import Track, TrackArray
import rosbag
from pylatex import Tabular 

import matplotlib.pyplot as plt
import numpy as np
from cycler import cycler


SMALL_SIZE = 12
MEDIUM_SIZE = 14
BIGGER_SIZE = 25
plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

bag = rosbag.Bag('tracks_PoseStamped.bag')
tracks = {}

bot= []
bot.append(file_interface.read_bag_trajectory(bag, '/robot_1'))
# bot.append(file_interface.read_bag_trajectory(bag, '/robot_2'))
tracks = file_interface.read_TrackArray(bag, '/tracks', 5)

loc_est = file_interface.read_bag_trajectory(bag,'/odometry/map')
loc_ref = file_interface.read_bag_trajectory(bag, '/mocap_pose')
bag.close()

loc_ref, loc_est = sync.associate_trajectories(loc_ref, loc_est)
loc_est, loc_rot, loc_tra, s = trajectory.align_trajectory(loc_est, 
        loc_ref, correct_scale=False, return_parameters=True)

table = Tabular('l c c c c c c c')
table.add_hline()
table.add_row(('id', 'rmse', 'mean', 'median', 'std', 'min', 'max', 'sse'))
table.add_empty_row()

# plot_collection = plot.PlotCollection("System Evaluation")
for idx,b in enumerate(bot):
    print("Calculations for track model", idx +1)
    matches = []
    for tr in tracks: # Find the best matching tracks to the bot trajectory

        traj_ref, traj_est = sync.associate_trajectories(b, tr, max_diff=0.01)
        traj_est, rot, tra, _ = trajectory.align_trajectory(
                traj_est, traj_ref, correct_scale=False, return_parameters=True)
        
        print("calculating APE for track of length", len(tr.timestamps))
        data = (traj_ref, traj_est)
        ape_metric = metrics.APE(metrics.PoseRelation.translation_part)
        ape_metric.process_data(data)
        ape_statistics = ape_metric.get_all_statistics()
        
        tra_dif = (tra - loc_tra)
        print(tra_dif)
        abs_tra_dif = abs((tra - loc_tra)[0]) + abs((tra - loc_tra)[1])
        rot_dif = (rot - loc_rot)
        abs_rot_dif = 0
        for i in range(0,len(rot_dif)):
            abs_rot_dif += abs(rot_dif[i][0])+ abs(rot_dif[i][1]) +\
                abs(rot_dif[i][2])
        mismatch = abs_tra_dif +abs_rot_dif
        tuple = [traj_est, mismatch, traj_est.get_infos()['t_start (s)']]
        matches.append(tuple)

    matches.sort(key = lambda x: x[2])
    
    segments = [] #The parts of the trajectory are added to this list
    for m in matches:
        if m[1]<1 and m[0].get_statistics()['v_avg (m/s)']>0.5: # if the mismatch is smaller than 1
           print(m[0].get_statistics()['v_avg (m/s)'])
           segments.append(m[0]) 
           # print(m[0].get_infos()['t_start (s)'],m[0].get_infos()["path length (m)"])
           # print(m[0].get_statistics()['v_avg (m/s)'])
    whole =trajectory.merge(segments)
        

    traj_ref, traj_est = sync.associate_trajectories(b, whole, max_diff=0.01)
    traj_est, rot, tra, _ = trajectory.align_trajectory(
            traj_est, traj_ref, correct_scale=False, return_parameters=True)

    # print(traj_est.get_infos())

    # Plot xy data
    fig_xy, ax_xy = plt.subplots(1)
    plot.xy(ax_xy, b,       '--', 'gray', 'original')
    plot.xy(ax_xy, traj_ref, '-', 'gray', 'reference')
    # plot.xy(ax_xy, traj_est, '-', 'black', 'estimation')
    ax_xy.set_prop_cycle(cycler('color', ['c', 'm', 'y', 'k']) +
           cycler('lw', [1, 2, 3, 4]))
    color=iter(plt.cm.rainbow(np.linspace(0,1,len(segments))))
    for idx, segment in enumerate(segments):
        c=next(color)
        label = "segment" + str(idx + 1)
        plot.xy(ax_xy, segment, '--', c, label)
    plt.waitforbuttonpress(0)
    plt.close(fig_xy)

    fig, ax = plt.subplots(1)
    plot.traj_yaw(ax, traj_ref)
    plt.waitforbuttonpress(0)
    plt.close(fig)
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
    table.add_row((idx, round(ape_statistics["rmse"],3),
        round(ape_statistics["mean"],3),
        round(ape_statistics["median"],3),
        round(ape_statistics["std"],3),
        round(ape_statistics["min"],3),
        round(ape_statistics["max"],3),
        round(ape_statistics["sse"],3),))
    table.add_hline

# plot_collection.show()
# table.generate_tex('/home/kostas/report/figures/tables/eval_table')
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

