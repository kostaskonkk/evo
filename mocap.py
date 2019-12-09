from evo.core  import trajectory, sync, metrics
from evo.tools import plot, file_interface
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import rosbag
import seaborn as sns
import itertools

plt.style.use(['seaborn-whitegrid', 'stylerc'])

def traj_yaw(ax, traj, style='-', color='black', label="", alpha=1.0,
        start_timestamp=None):
    if isinstance(traj, trajectory.PoseTrajectory3D):
        x = traj.timestamps - (traj.timestamps[0]
                               if start_timestamp is None else start_timestamp)
        xlabel = "Time [s]"
    else:
        x = range(0, len(traj.orientations_euler))
        xlabel = "index"
    ylabel = "$\psi$ [rad/s]"

    # wrapped = np.rad2deg(traj.get_orientations_euler()[:,2])
    # wrapped = np.wrap(traj.get_orientations_euler()[:,2])
    yaw = traj.get_orientations_euler()[:,2]
    unwrapped = np.unwrap(yaw)
    ax.plot(x, unwrapped, style, markersize =1, color=color, label=label, alpha=alpha)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    ax.set_xlim(left=0)

def traj_fourplots(axarr, traj, style='-', color='black', label="", alpha=1.0,
        start_timestamp=None):

    if isinstance(traj, trajectory.PoseTrajectory3D):
        x = traj.timestamps - (traj.timestamps[0]
                               if start_timestamp is None else start_timestamp)
        xlabel = "Time [s]"
    else:
        x = range(0, len(traj.positions_xyz))
        xlabel = "index"
    ylabels = ["x [m]", "y [m]"]
    for i in range(2):
        axarr[0,i].plot(x, traj.positions_xyz[:, i], style,markersize = 1, color=color,
                      label=label, alpha=alpha)
        axarr[0,i].set_ylabel(ylabels[i])
        axarr[0,i].set_xlabel(xlabel)
    axarr[1,0].plot(traj.positions_xyz[:, 0], traj.positions_xyz[:, 1],
            style,markersize = 1, color=color, alpha=alpha)
    axarr[1,0].set(xlabel=ylabels[0], ylabel=ylabels[1])
    traj_yaw(axarr[1,1],traj, style, color,
            alpha=alpha, start_timestamp=start_timestamp)

bag = rosbag.Bag('/home/kostas/experiments/dsv/maybe2.bag')
# bag = rosbag.Bag('/home/kostas/experiments/dsv/test.bag')

ego = file_interface.read_bag_trajectory(bag, '/ego_pose')
red = file_interface.read_bag_trajectory(bag, '/red_pose')
bag.close()

fig, axarr = plt.subplots(2, 2)
traj_fourplots(axarr, ego, '-', 'gray', 'DSV')
traj_fourplots(axarr, red, '-', 'red', 'Red')
handles, labels = axarr[0,0].get_legend_handles_labels()
# fig.legend(handles, labels, loc='lower center', ncol = 2,
        # bbox_to_anchor=(0.5, 0))
fig.tight_layout()
plt.show()
# fig.waitforbuttonpress()

