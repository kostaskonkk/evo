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
import tracking, errors


plt.style.use(['seaborn-whitegrid', 'stylerc'])

# bag = rosbag.Bag(sys.argv[1])
# bag = rosbag.Bag("/home/kostas/results/exp.bag")
# type_of_exp = 'experiment'
# distance = 0.9

bag = rosbag.Bag("/home/kostas/results/sim.bag")
type_of_exp = 'simulation'
distance = 15

references= []
references.append(('slow', file_interface.read_bag_trajectory(bag, '/prius_slow')))
references.append(('fast', file_interface.read_bag_trajectory(bag, '/prius_fast')))

tracks = []
# tracks.append(('mean'   , file_interface.read_TrackArray(bag, '/tracks/mean',3)))
tracks.append(('mean_kf', file_interface.read_TrackArray(bag,'/tracks/mean_kf', 3)))
# tracks.append(('box'    , file_interface.read_TrackArray(bag,'/tracks/box',3)))
tracks.append(('box_ukf', file_interface.read_TrackArray(bag, '/tracks/box_ukf', 3)))

bag.close()

table = Tabular('l c c c c c c c')
table.add_hline() 
table.add_row(('id', 'rmse', 'mean', 'median', 'std', 'min', 'max', 'sse'))
table.add_empty_row()

# for idx,b in enumerate(bot):
results=[]
for reference in references:
    for track in tracks:
        
        # if track[0]=='mean' or track[0]=='box':
            # segments, traj_reference = \
                # tracking.associate_segments_common_frame(reference,track[1],distance)
            # tracking.four_plots(1, reference, traj_reference, segments, type_of_exp) 
            # tracking.stats_to_latex_table(traj_reference, segments, idx, table)

        if track[0]=='mean_kf' or track[0]=='box_ukf':
            segments, traj_reference = \
                tracking.associate_segments_common_frame(reference[1], track[1],distance)
            # tracking.pose_vel(track[0]+reference[0], reference[1], traj_reference, segments, type_of_exp) 
            # tracking.velocities(1, reference[1], traj_reference, segments, type_of_exp) 
            # tracking.stats_to_latex_table(traj_reference, segments, idx, table)

        # for segment in segments:
            # print(segment.linear_vel)
        whole =trajectory.merge(segments)
        result = errors.ape(
            traj_ref=traj_reference,
            traj_est=whole,
            pose_relation=errors.PoseRelation.translation_part,
            ref_name=reference[0],
            est_name=track[0]+reference[0])
        print(result.info)
        print(result.trajectories)
        print(result.stats)
        results.append(result)
        # print(result.np_arrays)
        # file_interface.save_res_file("/home/kostas/results/res_files/mean_track", mean_result, False)
errors.run(results)


# table.generate_tex('/home/kostas/report/figures/tables/eval_table')
