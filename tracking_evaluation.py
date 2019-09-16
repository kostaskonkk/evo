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

SMALL_SIZE  = 12
MEDIUM_SIZE = 14
BIGGER_SIZE = 25
plt.rc('font',  size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes',  titlesize=BIGGER_SIZE)    # fontsize of the axes title
plt.rc('axes',  labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('legend',fontsize=SMALL_SIZE)      # legend fontsize
plt.rc('figure',titlesize=BIGGER_SIZE)    # fontsize of the figure title
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

    plot_collection.add_figure("raw", fig_raw)

    # statistics plot
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
    col_wrap = 2 if len(args.result_files) <= 2 else math.ceil(
        len(args.result_files) / 2.0)
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
    # if args.save_plot:
        # logger.debug(SEP)
        # plot_collection.export(args.save_plot,
                               # confirm_overwrite=not args.no_warnings)
    # if args.serialize_plot:
        # logger.debug(SEP)
        # plot_collection.serialize(args.serialize_plot,
                                  # confirm_overwrite=not args.no_warnings)
                                                                                                                                

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
mean = file_interface.read_TrackArray(bag, '/mean_tracks', 3)
filtered_tracks = file_interface.read_TrackArray(bag, '/filtered_tracks', 3)
box_tracks = file_interface.read_TrackArray(bag, '/box_tracks', 3)
obs_tracks = file_interface.read_TrackArray(bag, '/obs_tracks', 3)
mocap= file_interface.read_bag_trajectory(bag, '/mocap_pose')
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
# for tr in filtered_tracks:

for idx,b in enumerate(bot):

    # print("Calculations for track model", idx +1,"based on the mean of the cluster")
    # segments, traj_ref = tracking.associate_segments_common_frame(b,mean)
    # distance)
    # tracking.associate_segments_common_frame(b,mean,distance)
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

    print("Calculations for track model", idx +1,"based on the center of the bounding box")
    segments, traj_ref = tracking.associate_segments_common_frame(b,box_tracks, distance)
    tracking.four_plots(idx, b, traj_ref, segments, type_of_exp) 
    tracking.stats_to_latex_table(traj_ref, segments,idx, table)
    tracking.plot_dimensions(segments, b, start_timestamp = b.timestamps[0])

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

    # plot velocities
    print("Visualizing the velocities", idx +1,"nonlinear observer")
    name = 'bot1'
    fig, axarr = plt.subplots(3)
    fig.tight_layout()
    fig.suptitle('Speed Estimation ' + name, fontsize=30)
    plot.traj_vel(axarr, b, '--', 'gray', 'original')
    # whole =trajectory.merge(segments)
    plot.traj_vel(axarr, traj_ref, '-', 'gray', 'reference',1 ,b.timestamps[0])
    axarr[0].set_prop_cycle(cycler('color', ['c', 'm', 'y', 'k']) +
           cycler('lw', [1, 2, 3, 4]))
    color=iter(plt.cm.rainbow(np.linspace(0,1,len(segments))))
    for i, segment in enumerate(segments):
        c=next(color)
        label = "segment" + str(idx + 1)
        plot.linear_vel(axarr[0:2], segment, '-', c, label,1 ,b.timestamps[0])
        # axarr[0].plot(segment.linear_vel[0,:])
        # print(segment.linear_vel[:,1])
        # plot.linear_vel(axarr[0:2], segment, '-', c, label,1 )
    # fig.subplots_adjust(hspace = 0.2)
    plt.waitforbuttonpress(0)
    # plt.savefig("/home/kostas/results/latest/velocity"+name+".png",  format='png', bbox_inches='tight')

table.generate_tex('/home/kostas/report/figures/tables/eval_table')
