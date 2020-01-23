#!/usr/bin/env python3

from __future__ import print_function
from evo.core  import trajectory, sync, metrics
from evo.tools import file_interface
from datmo.msg import Track, TrackArray
from pylatex import Tabular 

import rosbag
import matplotlib.pyplot as plt
import numpy as np
from cycler import cycler
import sys # cli arguments in sys.argv
import tracking, errors, exec_time
import seaborn as sns
import itertools
import os

# path = "/home/kostas/results/experiment/overtakes_new.bag"
path = "/home/kostas/results/experiment/overtakes.bag"
# path = "/home/kostas/results/experiment/parallel.bag"
# path = "/home/kostas/results/experiment/overtake_ego.bag"
# path = "/home/kostas/results/experiment/overtake_red.bag"
# path = "/home/kostas/results/experiment/intersection.bag"
# path = "/home/kostas/experiments/datmo.bag"

# path = "/home/kostas/results/simulation/double_lane_change.bag"
# path = "/home/kostas/results/simulation/lane_keeping.bag"

type_of_exp = os.path.basename(os.path.dirname(path))
scenario = os.path.splitext(os.path.basename(path))[0]

filename = type_of_exp +"/" + scenario

plt.style.use(['seaborn-whitegrid', 'stylerc'])
# bag = rosbag.Bag(sys.argv[1])

bag = rosbag.Bag(path)
# type_of_exp = 'experiment'

# bag = rosbag.Bag("/home/kostas/results/sim.bag")
# type_of_exp = 'simulation'
# distance = 3 

references= []
if type_of_exp=='simulation':
    references.append(('-slow', file_interface.read_bag_trajectory(bag, '/prius_slow')))
    references.append(('-fast', file_interface.read_bag_trajectory(bag, '/prius_fast')))
    distance = 3 
else:
    references.append(('', file_interface.read_bag_trajectory(bag, '/red_pose')))
    distance = 0.35
tracks = []
# tracks.append(('mean'   , file_interface.read_TrackArray(bag, '/tracks/mean',3)))
# tracks.append(('mean_kf', file_interface.read_TrackArray(bag,'/tracks/mean_kf', 3)))
tracks.append(('UKF', file_interface.read_TrackArray(bag, '/tracks/box_ukf', 3)))
tracks.append(('KF' , file_interface.read_TrackArray(bag,'/tracks/box_kf',3)))

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

# tracking.screen_states(references, tracks, distance)
tracking.presentation_states(references, tracks, distance, filename)
# tracking.report_states(references, tracks, distance, filename)
# exec_time.whole(type_of_exp) # Make execution time plots

palette = itertools.cycle(sns.color_palette())

for ref in references:

    for track in tracks:
        segments, traj_reference = \
            tracking.associate_segments_common_frame(ref[1], track[1],distance)
        color=next(palette)
        # tracking.stats_to_latex_table(traj_reference, segments, idx, table)

        whole =tracking.merge(segments)

        result_x = errors.ape(
            traj_ref=traj_reference,
            traj_est=whole,
            pose_relation=errors.PoseRelation.x,
            ref_name=ref[0],
            est_name=track[0]+" "+ref[0])
        results_x.append(result_x)
        # print(result_x)

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

# errors.stats(results_x, results_y, results_vx, results_vy, results_psi,
        # results_omega, filename)
print("DONE!!")
# table.generate_tex('/home/kostas/report/figures/tables/eval_table')
