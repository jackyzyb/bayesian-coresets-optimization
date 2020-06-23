import bokeh.plotting as bkp
import numpy as np
import sys, os
# make it so we can import models/etc from parent folder
sys.path.insert(1, os.path.join(sys.path[0], '../common'))
from plotting import *
from bokeh.io import export_svgs
from bokeh.layouts import gridplot
from bokeh.io import export_png
from bokeh.models import ColumnDataSource, Range1d, LabelSet, Label

plot_reverse_kl = True
trials = [1]
nms = [('GIGAOE', 'GIGA'), ('SVI', 'SparseVI'),  ('RAND', 'Uniform'), ('IHT', 'A-IHT'), ('IHT-2', 'A-IHT II')]

# plot the KL figure
fig = bkp.figure(y_axis_type='log', plot_width=1000, plot_height=1000, x_axis_label='iteration',
                 y_axis_label=('f' if plot_reverse_kl else 'Forward KL'))
preprocess_plot(fig, '32pt', False, True)

plot_every = 10
M = 200
max_iter = 300 + 1
marker_plot_every = 30
marker_size = 25

obj_parameter = np.load('results/obj.npz')
Phi = obj_parameter['Phi']
y = obj_parameter['y'].squeeze()
starting_obj = np.linalg.norm(y, ord=2)  # initial objective value for IHT and IHT-2

for i, nm in enumerate(nms):
    kl = []
    sz = []
    for t in trials:
        if nm[0] == 'IHT':
            obj_list = np.load('results/iht-convergence.npy')
            obj_list = np.concatenate([[starting_obj], obj_list])
        elif nm[0] == 'IHT-2':
            obj_list = np.load('results/iht-2-convergence.npy')
            obj_list = np.concatenate([[starting_obj], obj_list])
        else:
            res = np.load('results/results_' + nm[0] + '_' + str(t) + '.npz')
            w = res['w'][M, :]
            print('w sparsity: {}'.format(sum(w > 0)))
            obj_baseline = np.linalg.norm(y - Phi.dot(w), ord=2)
            obj_list = np.ones(max_iter) * obj_baseline

    iter = list(range(max_iter))
    fig.line(iter[::plot_every], obj_list[::plot_every], color=pal[i], line_width=5, legend=nm[1])
    if nm[0] == 'IHT':
      fig.circle(iter[::marker_plot_every], obj_list[::marker_plot_every], fill_color=pal[i], size=marker_size, legend=nm[1])
    elif nm[0] == 'IHT-2':
      fig.circle(iter[::marker_plot_every], obj_list[::marker_plot_every], fill_color="white",
                 size=marker_size, legend=nm[1])
    elif nm[0] == 'SVI':
      fig.square(iter[::marker_plot_every], obj_list[::marker_plot_every], fill_color="white",
                 size=marker_size, legend=nm[1])
    elif nm[0] == 'GIGAOE':
      fig.square(iter[::marker_plot_every], obj_list[::marker_plot_every], fill_color=pal[i],
                 size=marker_size, legend=nm[1])

legend_len = len(fig.legend.items)
fig.legend.items = fig.legend.items[legend_len - 2:legend_len] + fig.legend.items[0:legend_len - 2]

postprocess_plot(fig, '22pt', location='bottom_left', glyph_width=40)
fig.legend.background_fill_alpha = 0.
fig.legend.border_line_alpha = 0.

bkp.show(fig)
