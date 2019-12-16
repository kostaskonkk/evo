#!/usr/bin/env python3

from __future__ import print_function
from evo.core  import trajectory, sync, metrics
from evo.tools import file_interface, plot
from datmo.msg import Track, TrackArray
import rosbag

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.figure as fg
import numpy as np
from cycler import cycler
import tracking
import seaborn as sns
import itertools

def poses_vel(axarr, color, name, b, traj_ref, segments, method):
    x_y_yaws  (axarr[0,0:3], color, b, traj_ref, segments, method)
    vx_vys(axarr[1,0:3], color, b, traj_ref, segments, method)

def x_y_yaws(axarr, color, b, traj_ref, segments, method):
    palette = itertools.cycle(sns.color_palette())
    for i, segment in enumerate(segments):
        color=next(palette)
        if i==0:
            plot.traj_xy(axarr[0:2], segment, '-', color, method,1 ,b.timestamps[0])
        else:
            plot.traj_xy(axarr[0:2], segment, '-', color, None,1 ,b.timestamps[0])
        # axarr[1,0].plot(segment.positions_xyz[:, 0], segment.positions_xyz[:,1])
        plot.traj_yaw(axarr[2],segment, '-', color, None,1 ,b.timestamps[0])
        axarr[0].set_xlim(left=0)
        axarr[1].set_xlim(left=0)
        axarr[2].set_xlim(left=0)

def vx_vys(axarr, color, b, traj_ref, segments, method):
    whole =trajectory.merge(segments)
    for i, segment in enumerate(segments):
        plot.linear_vel(axarr[0:2], segment, '-', color, method, 1, b.timestamps[0])
        tracking.angular_vel(axarr[2], segment, '-', color, method, 1, b.timestamps[0])


plt.style.use(['seaborn-whitegrid', 'stylerc'])

# bag = rosbag.Bag("/home/kostas/results/experiment/test.bag")
bag = rosbag.Bag("/home/kostas/experiments/datmo.bag")
type_of_exp = 'experiment'
distance = 0.35

ref= []
ref.append(('red', file_interface.read_bag_trajectory(bag, '/red_pose')))

tracks = []
# tracks.append(('mean'   , file_interface.read_TrackArray(bag, '/tracks/mean',3)))
# tracks.append(('mean_kf', file_interface.read_TrackArray(bag,'/tracks/mean_kf', 3)))
tracks.append(('KF' , file_interface.read_TrackArray(bag,'/tracks/box_kf',1)))
# tracks.append(('UKF', file_interface.read_TrackArray(bag, '/tracks/box_ukf', 3)))
bag.close()


palette = itertools.cycle(sns.color_palette())


fig, axarr = plt.subplots(2,3)
plot.traj_xyyaw(axarr[0,0:3], ref[0][1], '-', 'gray', 'reference',1
        ,ref[0][1].timestamps[0])
plot.traj_vel(axarr[1,0:3], ref[0][1], '-', 'gray')


for track in tracks:
    segments, traj_reference = \
        tracking.associate_segments_common_frame(ref[0][1], track[1],distance)

    poses_vel(axarr, 'black', track[0]+ref[0][0], ref[0][1],
            traj_reference, segments, track[0]) 


fig.tight_layout()
handles, labels = axarr[0,0].get_legend_handles_labels()
lgd = fig.legend(handles, labels, loc='lower center',ncol = len(labels))
plt.show()
# plt.waitforbuttonpress(0)


