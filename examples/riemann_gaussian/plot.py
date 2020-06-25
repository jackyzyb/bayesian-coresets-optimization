import os
import sys

import bokeh.plotting as bkp
import numpy as np

# make it so we can import models/etc from parent folder
sys.path.insert(1, os.path.join(sys.path[0], '../common'))
from plotting import *

plot_reverse_kl = True
trials = np.arange(1, 11)
# nms = [('IHT', 'IHT'), ('SVI', 'SparseVI'), ('GIGAO', 'GIGA (Optimal, Projected)'), ('GIGAOE', 'GIGA (Optimal, Exact)'), ('GIGAR', 'GIGA (Realistic, Projected)'), ('GIGARE', 'GIGA (Realistic, Exact)'), ('RAND', 'Uniform')]
nms = [('GIGAOE', 'GIGA'), ('SVI', 'SparseVI'), ('RAND', 'Uniform'), ('IHT', 'A-IHT'), ('IHT-2', 'A-IHT II')]

# plot the KL figure
fig = bkp.figure(y_axis_type='log', plot_width=1000, plot_height=1000, x_axis_label='K',
                 y_axis_label=('Reverse KL' if plot_reverse_kl else 'Forward KL'))
preprocess_plot(fig, '32pt', False, True)

plot_every = 8
M = 300
marker_plot_every = 5
marker_size = 25

for i, nm in enumerate(nms):
    kl = []
    sz = []
    for t in trials:
        res = np.load('results/results_' + nm[0] + '_' + str(t) + '.npz')
        if plot_reverse_kl:
            kl.append(res['fklw'][::plot_every])
        else:
            kl.append(res['fklw'][::plot_every])
        sz.append((res['w'] > 0).sum(axis=1)[::plot_every])
    x = list(range(2, M + 1))[::plot_every]
    fig.patch(x=np.hstack((x, x[::-1])),
              y=np.hstack((np.percentile(kl, 75, axis=0), np.percentile(kl, 25, axis=0)[::-1])), color=pal[i],
              fill_alpha=0.3, legend=nm[1])
    fig.line(x, np.percentile(kl, 50, axis=0), color=pal[i], line_width=5, legend=nm[1])
    if nm[0] == 'IHT':
        fig.circle(x[::marker_plot_every], np.percentile(kl, 50, axis=0)[::marker_plot_every], fill_color=pal[i],
                   size=marker_size, legend=nm[1])
    elif nm[0] == 'IHT-2':
        fig.circle(x[::marker_plot_every], np.percentile(kl, 50, axis=0)[::marker_plot_every], fill_color="white",
                   size=marker_size, legend=nm[1])
    elif nm[0] == 'SVI':
        fig.square(x[::marker_plot_every], np.percentile(kl, 50, axis=0)[::marker_plot_every], fill_color="white",
                   size=marker_size, legend=nm[1])
    elif nm[0] == 'GIGAOE':
        fig.square(x[::marker_plot_every], np.percentile(kl, 50, axis=0)[::marker_plot_every], fill_color=pal[i],
                   size=marker_size, legend=nm[1])

legend_len = len(fig.legend.items)
fig.legend.items = fig.legend.items[legend_len - 2:legend_len] + fig.legend.items[0:legend_len - 2]

postprocess_plot(fig, '22pt', location='bottom_left', glyph_width=40)
fig.legend.background_fill_alpha = 0.
fig.legend.border_line_alpha = 0.

bkp.show(fig)
