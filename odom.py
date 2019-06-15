#!/usr/bin/env python

from __future__ import print_function

# print("loading required evo modules")
from evo.core  import trajectory, sync, metrics
from evo.tools import file_interface
from datmo.msg import Track, TrackArray
import rosbag

bag = rosbag.Bag('all_funny.bag')
topics = ['/odometry/wheel_imu', '/poseupdate', '/odometry/map']
trajectories = {}
for topic in topics:
    print(topic)
    trajectories[topic] = file_interface.read_bag_trajectory(bag, topic)
ref_traj = file_interface.read_bag_trajectory(bag, '/mocap_pose')
bag.close()

# print("registering and aligning trajectories")
for traj in trajectories:
    traj_ref, traj_est = sync.associate_trajectories(ref_traj,
            trajectories[traj])
    traj_est = trajectory.align_trajectory(traj_est, traj_ref, correct_scale=False)

    print("calculating APE")
    data = (traj_ref, traj_est)
    ape_metric = metrics.APE(metrics.PoseRelation.translation_part)
    ape_metric.process_data(data)
    ape_statistics = ape_metric.get_all_statistics()
    print("mean:", ape_statistics["mean"])

from evo.tools import plot
import matplotlib.pyplot as plt

plot_collection = plot.PlotCollection("Localization")
# # metric values
fig_1 = plt.figure(figsize=(8, 8))
plot.error_array(fig_1, ape_metric.error, statistics=ape_statistics,
                 name="APE", title=str(ape_metric))
plot_collection.add_figure("raw", fig_1)
# plt.show()
# trajectory colormapped with error
fig_2 = plt.figure(figsize=(8, 8))
plot_mode = plot.PlotMode.xy
ax = plot.prepare_axis(fig_2, plot_mode)
plot.traj(ax, plot_mode, traj_ref, '--', 'gray', 'reference')
plot.traj_colormap(
    ax, traj_est, ape_metric.error, plot_mode, min_map=ape_statistics["min"],
    max_map=ape_statistics["max"], title="APE mapped onto trajectory")
plot_collection.add_figure("traj (error)", fig_2)

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

plot_collection.show()
