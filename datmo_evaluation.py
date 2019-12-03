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
import sys # cli arguments in sys.argv
import tracking, errors, exec_time
import seaborn as sns
import itertools


plt.style.use(['seaborn-whitegrid', 'stylerc'])

# bag = rosbag.Bag(sys.argv[1])

# bag = rosbag.Bag("/home/kostas/results/experiment/test.bag")
# type_of_exp = 'experiment'
# distance = 0.5

bag = rosbag.Bag("/home/kostas/results/sim.bag")
type_of_exp = 'simulation'
distance = 3 

references= []
if type_of_exp=='simulation':
    references.append(('slow', file_interface.read_bag_trajectory(bag, '/prius_slow')))
    # references.append(('fast', file_interface.read_bag_trajectory(bag, '/prius_fast')))
else:
    references.append(('red', file_interface.read_bag_trajectory(bag, '/red_pose')))

tracks = []
# tracks.append(('mean'   , file_interface.read_TrackArray(bag, '/tracks/mean',3)))
# tracks.append(('mean_kf', file_interface.read_TrackArray(bag,'/tracks/mean_kf', 3)))
tracks.append(('KF' , file_interface.read_TrackArray(bag,'/tracks/box_kf',3)))
tracks.append(('UKF', file_interface.read_TrackArray(bag, '/tracks/box_ukf', 3)))


bag.close()

table = Tabular('l c c c c c c c')
table.add_hline() 
table.add_row(('id', 'rmse', 'mean', 'median', 'std', 'min', 'max', 'sse'))
table.add_empty_row()

results_x=[]
results_y=[]
results_vx=[]
results_vy=[]
results_psi=[]
results_omega=[]

# exec_time.whole(type_of_exp) # Make execution time plots


for ref in references:

    palette = itertools.cycle(sns.color_palette())

    fig, axarr = plt.subplots(2,3)
    plot.traj_xyyaw(axarr[0,0:3], ref[1], '-', 'gray', 'reference',1
            ,ref[1].timestamps[0])
    plot.traj_vel(axarr[1,0:3], ref[1], '-', 'gray')

    # mpl.use('pgf')
    # mpl.rcParams.update({
        # "text.usetex": True,
        # "pgf.texsystem": "pdflatex",
    # })
    # fig_rep, axarr_rep = plt.subplots(3,2,figsize=(6.125,7))
    # plot.traj_xy(axarr_rep[0,0:2], ref[1], '-', 'gray', 'reference',1
            # ,ref[1].timestamps[0])
    # plot.vx_vy(axarr_rep[1,0:2], ref[1], '-', 'gray', 'reference', 1,
            # ref[1].timestamps[0])
    # plot.traj_yaw(axarr_rep[2,0],ref[1], '-', 'gray', None, 1 ,ref[1].timestamps[0])
    # plot.angular_vel(axarr_rep[2,1], ref[1], '-', 'gray', None, 1, ref[1].timestamps[0])

    for track in tracks:
        segments, traj_reference = \
            tracking.associate_segments_common_frame(ref[1], track[1],distance)
        color=next(palette)
        # tracking.poses_vel(axarr, color, track[0]+ref[0], ref[1],
                # traj_reference, segments, track[0]) 
        # tracking.pose_vel(track[0]+ref[0], ref[1], traj_reference, segments, type_of_exp) 
        # tracking.plot_dimensions(segments, ref[1])
        # tracking.report(axarr_rep, color, track[0]+"-"+ref[0], ref[1], traj_reference, segments, type_of_exp)
        # tracking.stats_to_latex_table(traj_reference, segments, idx, table)

        # for segment in segments:
            # print(segment.linear_vel)
        whole =tracking.merge(segments)

        result_x = errors.ape(
            traj_ref=traj_reference,
            traj_est=whole,
            pose_relation=errors.PoseRelation.x,
            ref_name=ref[0],
            est_name=track[0]+" "+ref[0])
        results_x.append(result_x)

        result_y = errors.ape(
            traj_ref=traj_reference,
            traj_est=whole,
            pose_relation=errors.PoseRelation.y,
            ref_name=ref[0],
            est_name=track[0]+ref[0])
        results_y.append(result_y)
        
        result_vx = errors.ape(
            traj_ref=traj_reference,
            traj_est=whole,
            pose_relation=errors.PoseRelation.vx,
            ref_name=ref[0],
            est_name=track[0]+ref[0])
        results_vx.append(result_vx)
        
        result_vy = errors.ape(
            traj_ref=traj_reference,
            traj_est=whole,
            pose_relation=errors.PoseRelation.vy,
            ref_name=ref[0],
            est_name=track[0]+ref[0])
        results_vy.append(result_vy)

        result_psi = errors.ape(
            traj_ref=traj_reference,
            traj_est=whole,
            pose_relation=errors.PoseRelation.psi,
            ref_name=ref[0],
            est_name=track[0]+ref[0])
        results_psi.append(result_psi)
    
        result_omega = errors.ape(
            traj_ref=traj_reference,
            traj_est=whole,
            pose_relation=errors.PoseRelation.omega,
            ref_name=ref[0],
            est_name=track[0]+ref[0])
        results_omega.append(result_omega)

    # fig.tight_layout()
    # handles, labels = axarr[0,0].get_legend_handles_labels()
    # lgd = fig.legend(handles, labels, loc='lower center',ncol = len(labels))
    # plt.show()
    # plt.waitforbuttonpress(0)
    # plt.close(fig)

    # name = ref[0]
    # handles, labels = axarr_rep[0,0].get_legend_handles_labels()
    # lgd = fig_rep.legend(handles, labels, loc='lower center',ncol = len(labels))
    # fig_rep.tight_layout()
    # fig_rep.subplots_adjust(bottom=0.11)
    # fig_rep.savefig("/home/kostas/report/figures/"+type_of_exp+"/"+ name +".pgf")

errors.stats(results_x, results_y, results_vx, results_vy, results_psi,
        results_omega)
# errors.run(results)


# table.generate_tex('/home/kostas/report/figures/tables/eval_table')
