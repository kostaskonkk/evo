#!/usr/bin/env python

from __future__ import print_function
from evo.core  import trajectory, sync, metrics
from evo.tools import file_interface
from datmo.msg import Track, TrackArray
import rosbag
from pylatex import Tabular 

bag = rosbag.Bag('tracks_PoseStamped.bag')
tracks = {}

bot1 = file_interface.read_bag_trajectory(bag, '/robot_1')
bot2 = file_interface.read_bag_trajectory(bag, '/robot_2')
tracks = file_interface.read_TrackArray(bag, '/tracks')


loc_ref = file_interface.read_bag_trajectory(bag,'/odometry/map')
loc_est = file_interface.read_bag_trajectory(bag, '/mocap_pose')

bag.close()

loc_ref, loc_est = sync.associate_trajectories(loc_ref, loc_est)
loc_est, loc_rot, loc_tr, s = trajectory.align_trajectory(loc_est, 
        loc_ref, correct_scale=False, return_parameters=True)

print("registering and aligning trajectories")
pairs = []
combinations = []
table = Tabular('l c c c c c c c')
table.add_hline()
table.add_row(('id', 'rmse', 'mean', 'median', 'std', 'min', 'max', 'sse'))
# table.add_row((1, 2, 2, 8))
# table.add_hline(1, 2)
table.add_empty_row()
# table.add_row((4, 5, 3 ,8))

for tr in tracks:
    info = tr.get_infos# I should find how to use it

    traj_ref_bot1, traj_est_bot1 = sync.associate_trajectories(bot1, tr, max_diff=0.01)
    traj_ref_bot2, traj_est_bot2 = sync.associate_trajectories(bot2, tr, max_diff=0.01)
    # I could maybe use the full reference trajectory for allignment
    traj_est_bot1, r_a_bot1, t_a_bot1, s_bot1 = trajectory.align_trajectory(
            traj_est_bot1, traj_ref_bot1, correct_scale=False, return_parameters=True)
    traj_est_bot2, r_a_bot2, t_a_bot2, s_bot2 = trajectory.align_trajectory(
            traj_est_bot2, traj_ref_bot2, correct_scale=False, return_parameters=True)
    
    print("calculating APE for track of length", len(tr.timestamps))
    data_bot1 = (traj_ref_bot1, traj_est_bot1)
    data_bot2 = (traj_ref_bot2, traj_est_bot2)

    ape_metric_bot1 = metrics.APE(metrics.PoseRelation.translation_part)
    ape_metric_bot2 = metrics.APE(metrics.PoseRelation.translation_part)

    ape_metric_bot1.process_data(data_bot1)
    ape_metric_bot2.process_data(data_bot2)

    ape_statistics_bot1 = ape_metric_bot1.get_all_statistics()
    ape_statistics_bot2 = ape_metric_bot2.get_all_statistics()
    # pairs.append(inf)
    # print(info["duration (s)"])
    # print(info)
        # infos["duration (s)"] = self.timestamps[-1] - self.timestamps[0]
        # infos["t_start (s)"] = self.timestamps[0]
        # infos["t_end (s)"] = self.timestamps[-1]
    pairs.append(len(tr.timestamps))
    pairs.append(ape_statistics_bot1["mean"])
    pairs.append(t_a_bot1 + loc_tr)
    pairs.append(ape_statistics_bot2["mean"])
    pairs.append(t_a_bot2 + loc_tr)
    # pairs.append(ape_statistics_bot1["rmse"])
    # pairs.append(ape_statistics_bot2["rmse"])
    combinations.append(pairs)
            # "nr. of poses": self.num_poses,
    # print("mean:", ape_statistics["mean"])

    if len(tr.timestamps) == 179:
        from evo.tools import plot
        import matplotlib.pyplot as plt
        fig_2 = plt.figure(figsize=(8, 8))
        plot_mode = plot.PlotMode.xy
        ax = plot.prepare_axis(fig_2, plot_mode)
        plot.traj(ax, plot_mode, traj_ref_bot1, '--', 'gray', 'reference')
        plot.traj(ax, plot_mode, traj_est_bot1, '-', 'red', 'estimation')
        plt.savefig("/home/kostas/report/figures/eval/eval.png",format='png', bbox_inches='tight')
        # plt.show()
        table.add_row((1, round(ape_statistics_bot1["rmse"],3),
            round(ape_statistics_bot1["mean"],3),
            round(ape_statistics_bot1["median"],3),
            round(ape_statistics_bot1["std"],3),
            round(ape_statistics_bot1["min"],3),
            round(ape_statistics_bot1["max"],3),
            round(ape_statistics_bot1["sse"],3),))

for c in combinations:
    for p in pairs:
        print(p,)
print(len(combinations))

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

# plot_collection.show()
