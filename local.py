# -*- coding: UTF8 -*-
from evo.core  import trajectory, sync, metrics
from evo.tools import plot, file_interface
import matplotlib as mpl
import matplotlib.pyplot as plt
from golden_plots import set_size
import rosbag
from pylatex import Tabular 
# import sys # cli arguments in sys.argv
# print(plt.style.available)
plt.style.use('seaborn-whitegrid')
# plt.style.use('seaborn-paper')
# plt.rcParams['grid.color'] = 'gray'
plt.rcParams['grid.alpha'] = '0.5'
plt.rcParams['axes.edgecolor'] = 'k'
# plt.rcParams['axes.facecolor'] = 'w'
plt.rcParams['legend.edgecolor'] = 'k'
plt.rcParams['legend.facecolor'] = 'w'

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

nice_fonts = {
    # Use LaTeX to write all text
    # "text.usetex": True,
    # "font.family": "serif",
    # Use 10pt font in plots, to match 10pt font in document
    "axes.labelsize": 11,
    "font.size": 11,
    # Make the legend/label fonts a little smaller
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
}
mpl.rcParams.update(nice_fonts)

# bag = rosbag.Bag(sys.argv[1])
bag = rosbag.Bag('/home/kostas/results/local.bag')

bot= []
mocap= file_interface.read_bag_trajectory(bag, '/mocap_pose')
odom = file_interface.read_bag_trajectory(bag,'/odometry/wheel_imu')
slam = file_interface.read_bag_trajectory(bag,'/poseupdate')
fuse = file_interface.read_bag_trajectory(bag,'/odometry/map')
bag.close()

loc_table = Tabular('l c c c c c c c')
loc_table.add_hline()
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
# plt.rcParams['grid.color'] = 'gray'
# plt.rcParams['grid.alpha'] = '0.5'
plt.rcParams['axes.edgecolor'] = 'k'
# plt.rcParams['axes.facecolor'] = 'w'
plt.rcParams['legend.edgecolor'] = 'k'
plt.rcParams['legend.facecolor'] = 'w'

loc_table = Tabular('l c c c c c c c')
loc_table.add_hline()
loc_table.add_row(('method','rmse', 'mean', 'median', 'std', 'min', 'max', 'sse'))
loc_table.add_hline() 
loc_table.add_empty_row()

def three_plots(ref, est, table, name):
    """Generates plots and statistics table into Report

    :ref: PoseTrajectory3D object that is used as reference
    :est: PoseTrajectory3D object that is plotted against reference
    :table: Tabular object that is generated by Tabular('c c')
    :name: String that is used as name for file and table entry
    :returns: translation of reference against estimation

    """
    ref, est = sync.associate_trajectories(ref, est)
    est, rot, tra, s = trajectory.align_trajectory(est, 
            ref, correct_scale=False, return_parameters=True)

    data = (ref, est)
    ape_metric = metrics.APE(metrics.PoseRelation.translation_part)
    ape_metric.process_data(data)
    ape_statistics = ape_metric.get_all_statistics()


    # [ Localization ]
    fig, axarr = plt.subplots(3) #sharex=True)
    fig.suptitle('Localization', fontsize=30)
    fig.tight_layout()
    plot.traj_xyyaw(axarr, est,       '-', 'red',
            'estimation',1,ref.timestamps[0])
    plot.traj_xyyaw(axarr, ref,       '-', 'gray', 'original')
    fig.subplots_adjust(hspace = 0.2)
    plt.waitforbuttonpress(0)
    plt.savefig("/home/kostas/results/latest/"+name+".png", format='png', bbox_inches='tight')
    plt.close(fig)

    table.add_row((name,
        round(ape_statistics["rmse"],3),
        round(ape_statistics["mean"],3),
        round(ape_statistics["median"],3),
        round(ape_statistics["std"],3),
        round(ape_statistics["min"],3),
        round(ape_statistics["max"],3),
        round(ape_statistics["sse"],3),))
    table.add_hline

def four_plots(ref, est, table, name):
    """Generates plots and statistics table into Report

    :ref: PoseTrajectory3D object that is used as reference
    :est: PoseTrajectory3D object that is plotted against reference
    :table: Tabular object that is generated by Tabular('c c')
    :name: String that is used as name for file and table entry
    :returns: translation of reference against estimation

    """
    ref, est = sync.associate_trajectories(ref, est)
    # est, rot, tra, s = trajectory.align_trajectory(est, 
            # ref, correct_scale=False, return_parameters=True)

    est = trajectory.align_trajectory_origin(est, ref)

    data = (ref, est)
    ape_metric = metrics.APE(metrics.PoseRelation.translation_part)
    ape_metric.process_data(data)
    ape_statistics = ape_metric.get_all_statistics()


    # Plot x, y, xy,yaw 
    style ='-'
    if name=='slam':
        style = 'o'

    width = 442.65375
    print(set_size(width))  
    fig, axarr = plt.subplots(2, 2, figsize=(6.125,4))
    # fig, axarr = plt.subplots(2,2,figsize=(12,8))
    plot.traj_fourplots(axarr, est, style, 'red',
            'estimation',1,ref.timestamps[0])
    plot.traj_fourplots(axarr, ref, '-', 'gray', 'original')
    handles, labels = axarr[0,0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol = 2,
            bbox_to_anchor=(0.5, 0))
    plt.tight_layout()
     
    # plt.savefig("/home/kostas/results/test.png", dpi=100, format='png' )
    plt.savefig("/home/kostas/results/latest/"+name+".png", dpi=600, format='png',  bbox_inches='tight' )
    # plt.waitforbuttonpress(0)
    # plt.close(fig)

    if name=='slam':
        name = name.upper()
    else:
        name = name.capitalize()

    table.add_row((name,
        round(ape_statistics["rmse"],3),
        round(ape_statistics["mean"],3),
        round(ape_statistics["median"],3),
        round(ape_statistics["std"],3),
        round(ape_statistics["min"],3),
        round(ape_statistics["max"],3),
        round(ape_statistics["sse"],3),))
    table.add_hline

four_plots(mocap ,odom, loc_table, 'odometry')
four_plots(mocap ,slam, loc_table, 'slam')
four_plots(mocap ,fuse, loc_table, 'fusion')
loc_table.generate_tex('/home/kostas/report/figures/tables/loc_table')

# loc_ref, loc_est = sync.associate_trajectories(mocap, fuse)
# loc_est, loc_rot, loc_tra, _ = trajectory.align_trajectory(loc_est, 
        # loc_ref, correct_scale=False, return_parameters=True)
# print(loc_tra)

results = []
odom_result = ape(
    traj_ref=mocap,
    traj_est=odom,
    pose_relation=metrics.PoseRelation.translation_part,
    align=False,
    correct_scale=False,
    align_origin=True,
    ref_name="mocap",
    est_name="odom",
)
results.append(odom_result)
file_interface.save_res_file("/home/kostas/results/res_files/odom",
        odom_result, confirm_overwrite=False)

slam_result = ape(
    traj_ref=mocap,
    traj_est=slam,
    pose_relation=metrics.PoseRelation.translation_part,
    align=False,
    correct_scale=False,
    align_origin=True,
    ref_name="mocap",
    est_name="slam",
)
results.append(slam_result)
file_interface.save_res_file("/home/kostas/results/res_files/slam",
        slam_result, confirm_overwrite=False)

fuse_result = ape(
    traj_ref=mocap,
    traj_est=fuse,
    pose_relation=metrics.PoseRelation.translation_part,
    align=False,
    correct_scale=False,
    align_origin=True,
    ref_name="mocap",
    est_name="fuse",
)
results.append(fuse_result)
file_interface.save_res_file("/home/kostas/results/res_files/fuse",
        fuse_result, confirm_overwrite=False)
# convert_results_to_dataframe(results)
