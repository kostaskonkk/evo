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

# bag = rosbag.Bag("/home/kostas/results/exp.bag")
# type_of_exp = 'experiment'
# distance = 0.9

bag = rosbag.Bag("/home/kostas/results/sim.bag")
type_of_exp = 'simulation'
distance = 3 

references= []
if type_of_exp=='simulation':
    references.append(('slow', file_interface.read_bag_trajectory(bag, '/prius_slow')))
    # references.append(('fast', file_interface.read_bag_trajectory(bag, '/prius_fast')))
else:
    references.append(('robot_1', file_interface.read_bag_trajectory(bag,
        '/robot_1')))
    # references.append(('robot_2', file_interface.read_bag_trajectory(bag,
        # '/robot_2')))

tracks = []
# tracks.append(('mean'   , file_interface.read_TrackArray(bag, '/tracks/mean',3)))
tracks.append(('mean_kf', file_interface.read_TrackArray(bag,'/tracks/mean_kf', 3)))
# tracks.append(('box_kf' , file_interface.read_TrackArray(bag,'/tracks/box_kf',3)))
tracks.append(('box_ukf', file_interface.read_TrackArray(bag, '/tracks/box_ukf', 3)))


bag.close()

table = Tabular('l c c c c c c c')
table.add_hline() 
table.add_row(('id', 'rmse', 'mean', 'median', 'std', 'min', 'max', 'sse'))
table.add_empty_row()

results_translation=[]
results_rotation=[]

# exec_time.whole(type_of_exp) # Make execution time plots

# mpl.use('pgf')
mpl.rcParams.update({
    "text.usetex": True,
    "pgf.texsystem": "pdflatex",
})
for ref in references:

    fig, axarr = plt.subplots(2,3,figsize=(6.125,7))
    fig.tight_layout()
    palette = itertools.cycle(sns.color_palette())
    plot.traj_xyyaw(axarr[0,0:3], ref[1], '-', 'gray', 'reference',1
            ,ref[1].timestamps[0])
    # plot.traj_vel(axarr[1,0:3], ref[1], '-', 'gray')

    for track in tracks:
        
        segments, traj_reference = \
            tracking.associate_segments_common_frame(ref[1], track[1],distance)
        color=next(palette)
        # axarr[0,0].legend(track[0])
        tracking.poses_vel(axarr, color, track[0]+ref[0], ref[1],
                traj_reference, segments, track[0]) 
        # tracking.pose_vel(track[0]+ref[0], ref[1], traj_reference, segments, type_of_exp) 
        # tracking.plot_dimensions(segments, ref[1])
        # tracking.report(track[0]+"-"+ref[0], ref[1], traj_reference, segments, type_of_exp)
        # tracking.stats_to_latex_table(traj_reference, segments, idx, table)

        # for segment in segments:
            # print(segment.linear_vel)
        whole =trajectory.merge(segments)

        result_translation = errors.ape(
            traj_ref=traj_reference,
            traj_est=whole,
            pose_relation=errors.PoseRelation.translation_part,
            ref_name=ref[0],
            est_name=track[0]+ref[0])
        results_translation.append(result_translation)
        # print(result.info)
        # print(result.trajectories)
        # print(result.stats)
        result_rotation = errors.ape(
            traj_ref=traj_reference,
            traj_est=whole,
            pose_relation=errors.PoseRelation.rotation_part,
            ref_name=ref[0],
            est_name=track[0]+ref[0])
        results_rotation.append(result_rotation)
        # print(result.np_arrays)
        # file_interface.save_res_file("/home/kostas/results/res_files/mean_track", mean_result, False)
    
    fig.tight_layout()
    handles, labels = axarr[0,0].get_legend_handles_labels()
    lgd = fig.legend(handles, labels, loc='lower center',ncol = len(labels))
    fig.subplots_adjust(bottom=0.1)

    name = track[0]+"-"+ref[0]
    fig.savefig("/home/kostas/report/figures/"+type_of_exp+"/"+ name +".pgf",
            bbox_extra_artists=(lgd,), bbox_inches='tight')
    # fig.savefig("/home/kostas/report/figures/"+type_of_exp+"/"+ name +".pgf", bbox_inches='tight')

    # plt.waitforbuttonpress(0)
    # plt.close(fig)
    # plt.savefig("/home/kostas/results/"+type_of_exp+"/tracking" + str(idx+1) +".png",
            # dpi = 100, bbox_inches='tight')
# errors.run(results_translation)
# errors.run(results_rotation)


# table.generate_tex('/home/kostas/report/figures/tables/eval_table')
