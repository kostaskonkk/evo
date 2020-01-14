import tracking, errors, exec_time
import itertools
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.figure as fg
from evo.tools import plot

def report_states(references, tracks, distance, filename):
    mpl.use('pgf')
    mpl.rcParams.update({
        "text.usetex": True,
        "pgf.texsystem": "pdflatex",})
    palette = itertools.cycle(sns.color_palette())

    for ref in references:
        fig_rep, axarr_rep = plt.subplots(3,2,figsize=(6.125,7))

        for track in tracks:
            segments, traj_ref = \
                tracking.associate_segments_common_frame(ref[1], track[1],distance)
            color=next(palette)
            tracking.report(axarr_rep, color, track[0], ref[1], traj_ref, segments)
        plot.traj_xy(axarr_rep[0,0:2], traj_ref, '-', 'gray', 'reference',1 ,ref[1].timestamps[0])
        plot.vx_vy(axarr_rep[1,0:2], traj_ref, '-', 'gray', 'reference', 1, ref[1].timestamps[0])
        plot.traj_yaw(axarr_rep[2,0], traj_ref, '-', 'gray', None, 1 ,ref[1].timestamps[0])
        plot.angular_vel(axarr_rep[2,1], traj_ref, '-', 'gray', None, 1, ref[1].timestamps[0])

        handles, labels = axarr_rep[0,0].get_legend_handles_labels()
        lgd = fig_rep.legend(handles, labels, loc='lower center',ncol = len(labels))
        fig_rep.tight_layout()
        fig_rep.subplots_adjust(bottom=0.13)
        print(filename)
        fig_rep.savefig("/home/kostas/report/figures/"+ filename +ref[0]+".pgf")
        handles, labels = axarr_rep[0,0].get_legend_handles_labels()
        lgd = fig_rep.legend(handles, labels, loc='lower center',ncol = len(labels))

def screen_states(references, tracks, distance):
    palette = itertools.cycle(sns.color_palette())
    for ref in references:
        fig, axarr = plt.subplots(2,3)
        # plot.traj_xyyaw(axarr[0,0:3], ref[1], '-', 'gray', 'reference',1 ,ref[1].timestamps[0])
        # plot.traj_vel  (axarr[1,0:3], ref[1], '-', 'gray')

        for track in tracks:
            segments, traj_ref = \
                tracking.associate_segments_common_frame(ref[1], track[1],distance)
            color=next(palette)
            tracking.screen(axarr, color, ref[1], traj_ref, segments, track[0])

        plot.traj_xyyaw(axarr[0,0:3], traj_ref, '-', 'gray', 'reference',1
                ,ref[1].timestamps[0])
        plot.traj_vel  (axarr[1,0:3], traj_ref, '-', 'gray')

    fig.tight_layout()
    handles, labels = axarr[0,0].get_legend_handles_labels()
    lgd = fig.legend(handles, labels, loc='lower center',ncol = len(labels))
    plt.show()
    # fig.waitforbuttonpress(0)

def report_dimensions(references, tracks, distance, filename):
    mpl.use('pgf')
    mpl.rcParams.update({
        "text.usetex": True,
        "pgf.texsystem": "pdflatex",})
    fig_dimen, axarr_dimen = plt.subplots(2,1)
    palette = itertools.cycle(sns.color_palette())

    for ref in references:

        for track in tracks:
            segments, traj_ref = \
                tracking.associate_segments_common_frame(ref[1], track[1],distance)
            color=next(palette)
            tracking.plot_dimensions(segments, ref[1], axarr_dimen,  color, ref[0], ref[1].timestamps[0])

            whole =tracking.merge(segments)

    fig_dimen.tight_layout()
    fig_dimen.subplots_adjust(bottom=0.2)
    # handles, labels = axarr_dimen[0].get_legend_handles_labels()
    # lgd = fig_dimen.legend(handles, labels, loc='lower center',ncol = len(labels))
    fig_dimen.savefig("/home/kostas/report/figures/"+filename+"_dimensions.pgf")

