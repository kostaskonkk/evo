# from pylatex import Tabular 
import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt

def speed(type_of_exp):

    import csv
    from math import sqrt
    import pandas as pd 

    df = pd.read_csv("/home/kostas/results/exec_time/presentation.csv")
    exec_time = df['milli'].tolist()
    time = np.arange(len(df))*0.08
    plt.style.use(['seaborn-whitegrid', 'stylerc'])

    fig = plt.figure(figsize=(9.87,5.3))
    plt.plot(time, exec_time,marker = '.',c='k',lw=1)
    plt.plot(time, exec_time,marker = '.', linestyle = '',c='r',ms =8)
    # df_whole.plot(kind="barh", ax=fig_stats.gca(), colormap=colormap, stacked=False)
    # ax = df_whole['milli'].plot(marker = '.',c='k',ms =8, ax=fig_stats.gca(), colormap=colormap, stacked=False)
    # df_whole['Time'] = np.arange(len(df_whole))
    # df_whole['Time'] *= 0.08
    # ax1 = df_whole.plot(x='Time',y='milli', lw=0.4, c= 'k')
    # ax1 = df_whole.plot(x='Time',y='milli', marker = '.', linestyle = '',
            # c='r',ms =3, label=None)
    plt.ylabel('Execution Time (ms)')
    plt.xlabel('Time (s)')
        # ax_shape[1,0].axhline(y=0.385, color='gray')
    plt.xlim(left=0)
    plt.show()
    # ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    # ax2 = df_whole['objects'].plot(lw=0.2, c= 'k')
    # ax2.set_ylabel('Tracked Objects')
    # ax2.grid(False)
    # ax.set_title('Execution time of DATMO program')
    # plt.savefig("/home/kostas/report/figures/exec_time/whole.png", dpi = 300, format='png', bbox_inches='tight')
    # fig_stats.tight_layout()
    plt.savefig("/home/kostas/Dropbox/final_presentation/figures/execution_time.png",
            dpi=300)

def whole(type_of_exp):

    import pandas as pd 
    from math import sqrt
    df_whole = pd.read_csv("/home/kostas/results/exec_time/presentation.csv")
    plt.style.use(['seaborn-whitegrid', 'stylerc'])
    # mpl.use('pgf')
    # mpl.rcParams.update({"text.usetex": True})

    fig_stats = plt.figure(figsize=(6.125,3.785))
    # df_whole.plot(kind="barh", ax=fig_stats.gca(), colormap=colormap, stacked=False)
    # ax = df_whole['milli'].plot(marker = '.',c='k',ms =8, ax=fig_stats.gca(), colormap=colormap, stacked=False)
    df_whole['Time'] = np.arange(len(df_whole))
    df_whole['Time'] *= 0.08
    ax1 = df_whole.plot(x='Time',y='milli', lw=0.4, c= 'k')
    ax1 = df_whole.plot(x='Time',y='milli', marker = '.', linestyle = '',
            c='r',ms =3, label=None)
    ax1.set_ylabel('Execution Time (ms)')
    ax1.set_xlabel('Time (s)')
    print(df_whole)

    # ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    # ax2 = df_whole['objects'].plot(lw=0.2, c= 'k')
    # ax2.set_ylabel('Tracked Objects')
    # ax2.grid(False)
    # ax.set_title('Execution time of DATMO program')
    # plt.savefig("/home/kostas/report/figures/exec_time/whole.png", dpi = 300, format='png', bbox_inches='tight')
    fig_stats.tight_layout()
    plt.savefig("/home/kostas/Dropbox/final_presentation/figures/execution_time.png")

# df_whole = pd.read_csv("/home/kostas/results/exec_time/whole.csv")
# df_rect = pd.read_csv("/home/kostas/results/exec_time/rect_fitting.csv")
# df_clust =pd.read_csv("/home/kostas/results/exec_time/clustering.csv")
# df_test =pd.read_csv("/home/kostas/results/exec_time/testing.csv")
# print(df_whole.head())

# clust_table = Tabular('l c c c c c c')
# clust_table.add_hline()
# clust_table.add_row(('method', 'mean', 'max', 'min', 'median', 'std', 'var'))
# clust_table.add_hline() 
# clust_table.add_empty_row()

# rect_table = Tabular('l c c c c c c')
# rect_table.add_hline()
# rect_table.add_row(('method', 'mean', 'max', 'min', 'median', 'std', 'var'))
# rect_table.add_hline() 
# rect_table.add_empty_row()

# whole_table = Tabular('l c c c c c c')
# whole_table.add_hline()
# whole_table.add_row(('method', 'mean', 'max', 'min', 'median', 'std', 'var'))
# whole_table.add_hline() 
# whole_table.add_empty_row()

# mean_r = df_rect['dur_nano'].mean()
# max_r = df_rect['dur_nano'].max()
# min_r = df_rect['dur_nano'].min()
# median_r = df_rect['dur_nano'].median() 
# std_r = df_rect['dur_nano'].std() 
# var_r = df_rect['dur_nano'].var() 

# mean_w = df_whole['milli'].mean()
# max_w = df_whole['milli'].max()
# min_w = df_whole['milli'].min()
# median_w = df_whole['milli'].median() 
# std_w = df_whole['milli'].std() 
# var_w = df_whole['milli'].var() 

# mean_c= df_clust.mean()
# max_c= df_clust.max()
# min_c= df_clust.min()
# median_c= df_clust.median() 
# std_c= df_clust.std() 
# var_c= df_clust.var() 


# print('mean ',df_test['clusters'].mean(),'std',df_test['clusters'].std(),'min ',df_test['clusters'].min(),'max',df_test['clusters'].max())

# print('mean',df_whole['milli'].mean(),'std',df_whole['milli'].std(),'min',df_whole['milli'].min(),'max',df_whole['milli'].max())



# plt.xlabel
# plt.ylabel = 'Time [ms]'
# plt.show()
# plt.waitforbuttonpress
# df_rect.plot.scatter(x='num_points', y='dur_nano',s = 2)
# plt.show()
# plt.savefig("/home/kostas/report/figures/time_statistics.png", dpi = 300, format='png', bbox_inches='tight')
# plt.waitforbuttonpress(0)

# rect_table.add_row(('Rectangle Fitting',
    # round(mean_r),
    # round(max_r),
    # round(min_r),
    # round(median_r),
    # round(std_r),
    # round(var_r),))
# whole_table.add_hline

# clust_table.add_row(('Clustering',
    # round(mean_c),
    # round(max_c),
    # round(min_c),
    # round(median_c),
    # round(std_c),
    # round(var_c),))
# clust_table.add_hline

# whole_table.add_row(('Whole',
    # round(mean_w),
    # round(max_w),
    # round(min_w),
    # round(median_w),
    # round(std_w),
    # round(var_w),))
# whole_table.add_hline

# print(df_rect.head())
# whole_table.generate_tex('/home/kostas/report/figures/tables/exec_whole_table')
# clust_table.generate_tex('/home/kostas/report/figures/tables/exec_clust_table')
# rect_table.generate_tex('/home/kostas/report/figures/tables/exec_rect_table')
