import os
import sys

import bokeh.layouts as bkl
import bokeh.plotting as bkp
import numpy as np
from bokeh.io import export_svgs
import cairosvg

# make it so we can import models/etc from parent folder
sys.path.insert(1, os.path.join(sys.path[0], '../common'))
from plotting import *
import gaussian

algs = [('GIGAR', 'GIGA', pal[0]), ('SVI', 'SparseVI', pal[1]), ('RAND', 'Uniform', pal[2]),
         ('IHT', 'A-IHT', pal[3]), ('IHT-2', 'A-IHT II', pal[4]),]
dnames = ['synth_lr', 'ds1', 'phishing', 'synth_poiss', 'biketrips', 'airportdelays']

plot_every = 5
marker_plot_every = 4
marker_size = 25
y_label = "l2 distance"
figs = []
for dnm in dnames:
    print('Plotting ' + dnm)
    fig = bkp.figure(y_axis_type='log', y_axis_label=y_label, x_axis_label='Coreset Size k', plot_width=1000, plot_height=1000,
                     x_range=(-2, 105))
    figs.append([fig])

    samples = np.load('results/' + dnm + '_samples.npy')
    samples = np.hstack((samples[:, 1:], samples[:, 0][:, np.newaxis]))
    mup = samples.mean(axis=0)

    for alg in algs:
        folder = 'results/'
        trials = [fn for fn in os.listdir(folder) if dnm + '_' + alg[0] + '_results_' in fn]
        if len(trials) == 0: continue
        Ms = np.load(folder + trials[0])['Ms'][2:][::plot_every]
        dis_all = np.zeros((len(trials), len(Ms)))
        for tridx, fn in enumerate(trials):
            res = np.load(folder + fn)
            mu = res['mus'][2:][::plot_every]
            dis_all[tridx, :] = np.sqrt(((mu - mup.reshape([1, -1])) ** 2).sum(axis=1))


        dis50 = np.percentile(dis_all, 50, axis=0)
        dis25 = np.percentile(dis_all, 35, axis=0)
        dis75 = np.percentile(dis_all, 65, axis=0)

        fig.line(Ms, dis50, color=alg[2], legend_label=alg[1], line_width=10)
        fig.patch(np.hstack((Ms, Ms[::-1])), np.hstack((dis75, dis25[::-1])), fill_color=alg[2], legend_label=alg[1],
                  alpha=0.3)
        fig.legend.location = 'bottom_left'

        if alg[0] == 'IHT':
            fig.circle(Ms[::marker_plot_every], dis50[::marker_plot_every], fill_color=alg[2],
                       size=marker_size, legend=alg[1])

        elif alg[0] == 'IHT-2':
            fig.circle(Ms[::marker_plot_every], dis50[::marker_plot_every], fill_color="white",
                       size=marker_size, legend=alg[1])

        elif alg[0] == 'SVI':
            fig.square(Ms[::marker_plot_every], dis50[::marker_plot_every], fill_color="white",
                       size=marker_size, legend=alg[1])

        elif alg[0] == 'GIGAR':
            fig.square(Ms[::marker_plot_every], dis50[::marker_plot_every], fill_color=alg[2],
                       size=marker_size, legend=alg[1])


    for f in [fig]:
        axis_font_size = '25pt'
        axis_label_size = '32pt'
        f.xaxis.axis_label_text_font_size = axis_label_size
        f.xaxis.major_label_text_font_size = axis_font_size
        f.yaxis.axis_label_text_font_size = axis_label_size
        f.yaxis.major_label_text_font_size = axis_font_size

        f.legend.label_text_font_size = '25pt'
        f.legend.glyph_width = 40
        # f.legend.glyph_height = 40
        f.legend.spacing = 15
        f.legend.visible = True
        f.legend.location = 'bottom_left'

    legend_len = len(fig.legend.items)
    fig.legend.items = fig.legend.items[legend_len - 2:legend_len] + fig.legend.items[0:legend_len - 2]


    postprocess_plot(fig, '22pt', location='bottom_left', glyph_width=40)
    fig.legend.background_fill_alpha = 0.
    fig.legend.border_line_alpha = 0.

    fig.output_backend = 'svg'

    fig_name = dnm + '_distance_'

    export_svgs(fig, filename=fig_name + '.svg')
    # cairosvg.svg2pdf(url=fig_name+'.svg', write_to=fig_name+'.pdf')
    cairosvg.svg2pdf(
        file_obj=open(fig_name + '.svg', "rb"), write_to=fig_name + '.pdf')

# bkp.show(fig2)
bkp.show(bkl.gridplot(figs))



