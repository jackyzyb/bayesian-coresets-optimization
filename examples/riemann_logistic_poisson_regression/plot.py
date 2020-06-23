import numpy as np
import bokeh.plotting as bkp
import bokeh.layouts as bkl
import time
import os, sys

# make it so we can import models/etc from parent folder
sys.path.insert(1, os.path.join(sys.path[0], '../common'))
from plotting import *
import gaussian

algs = [('GIGAR', 'GIGA', pal[0]), ('SVI', 'SparseVI', pal[1]), ('RAND', 'Uniform', pal[2]),
        ('PRIOR', 'Prior', 'black'), ('IHT', 'A-IHT', pal[3]), ('IHT-2', 'A-IHT II', pal[4])]
dnames = ['synth_lr']  # one of the dataset names in ['synth_lr', 'ds1', 'phishing', 'synth_poiss', 'biketrips', 'airportdelays']

plot_every = 5
marker_plot_every = 4
marker_size = 25
is_forward_KL = False
if is_forward_KL:
  y_label = 'Forward KL'
else:
  y_label = 'Reverse KL'

figs = []
for dnm in dnames:
    print('Plotting ' + dnm)
    fig = bkp.figure(y_axis_type='log', y_axis_label=y_label, x_axis_label='K', plot_width=1000, plot_height=1000)
    preprocess_plot(fig, '32pt', False, True)
    fig2 = bkp.figure(y_axis_type='log', y_axis_label=y_label, x_axis_type='log', x_axis_label='CPU Time (s)',
                      plot_width=1000, plot_height=1000)
    preprocess_plot(fig2, '32pt', True, True)
    fig3 = bkp.figure(y_axis_type='log', y_axis_label=y_label, x_axis_label='Coreset Size', plot_width=3000,
                      plot_height=3000)
    preprocess_plot(fig3, '32pt', False, True)

    figs.append([fig, fig2, fig3])

    # get normalizations based on the prior
    std_kls = {}
    trials = [fn for fn in os.listdir('results/') if dnm + '_PRIOR_results_' in fn]
    if len(trials) == 0:
        print('Need to run prior to establish baseline first')
        quit()
    kltot = 0.
    for tridx, fn in enumerate(trials):
        res = np.load('results/' + fn)
        assert np.all(res['kls'] == res['kls'][0])  # make sure prior doesn't change...
        kltot += res['kls'][0]
    std_kls[dnm] = kltot / len(trials)

    for alg in algs:
        if alg[0] == 'IHT-2':
            folder = 'results/results-iht-2/'
            trials = [fn for fn in os.listdir(folder) if dnm + '_' + 'IHT' + '_results_' in fn]
        else:
            folder = 'results/'
            trials = [fn for fn in os.listdir(folder) if dnm + '_' + alg[0] + '_results_' in fn]
        if len(trials) == 0: continue
        Ms = np.load(folder + trials[0])['Ms'][2:][::plot_every]
        kls = np.zeros((len(trials), len(Ms)))
        cputs = np.zeros((len(trials), len(Ms)))
        cszs = np.zeros((len(trials), len(Ms)))
        kl0 = std_kls[dnm]
        for tridx, fn in enumerate(trials):
            res = np.load(folder + fn)
            cput = res['cputs'][2:][::plot_every]
            cputs[tridx, :] = cput[:len(Ms)]
            wts = res['wts'][2:][::plot_every]
            mu = res['mus'][2:][::plot_every]
            Sig = res['Sigs'][2:][::plot_every]
            cszs[tridx, :] = (wts > 0).sum(axis=1)
            if is_forward_KL is not True:
                samples = np.load('results/' + dnm + '_samples.npy')
                samples = np.hstack((samples[:, 1:], samples[:, 0][:, np.newaxis]))
                mup = samples.mean(axis=0)
                Sigp = np.cov(samples, rowvar=False)
                M = res['mus'].shape[0] - 1
                kl = np.zeros(M + 1)
                for m in range(M + 1):
                    mul = res['mus'][m, :]
                    Sigl = res['Sigs'][m, :, :]
                    kl[m] = gaussian.gaussian_KL(mul, np.linalg.inv(Sigl), mup, Sigp)
                kl = kl[2:][::plot_every]
            else:
                kl = res['kls'][2:][::plot_every]
            kls[tridx, :] = kl[:len(Ms)] / kl0
            if 'PRIOR' in fn:
                kls[tridx, :] = np.median(kls[tridx, :])

        cput50 = np.percentile(cputs, 50, axis=0)
        cput25 = np.percentile(cputs, 35, axis=0)
        cput75 = np.percentile(cputs, 65, axis=0)

        csz50 = np.percentile(cszs, 50, axis=0)
        csz25 = np.percentile(cszs, 35, axis=0)
        csz75 = np.percentile(cszs, 65, axis=0)

        kl50 = np.percentile(kls, 50, axis=0)
        kl25 = np.percentile(kls, 35, axis=0)
        kl75 = np.percentile(kls, 65, axis=0)

        fig.line(Ms, kl50, color=alg[2], legend_label=alg[1], line_width=10)
        fig.patch(np.hstack((Ms, Ms[::-1])), np.hstack((kl75, kl25[::-1])), fill_color=alg[2], legend_label=alg[1],
                  alpha=0.3)
        fig.legend.location = 'bottom_left'
        fig2.line(cput50, kl50, color=alg[2], legend_label=alg[1], line_width=10)
        fig2.patch(np.hstack((cput50, cput50[::-1])), np.hstack((kl75, kl25[::-1])), fill_color=alg[2],
                   legend_label=alg[1], alpha=0.3)
        fig2.legend.location = 'bottom_left'
        if alg[0] == 'IHT':
            fig.circle(Ms[::marker_plot_every], kl50[::marker_plot_every], fill_color=alg[2],
                       size=marker_size, legend=alg[1])
            fig2.circle(cput50[::marker_plot_every], kl50[::marker_plot_every], fill_color=alg[2],
                        size=marker_size, legend=alg[1])
        elif alg[0] == 'IHT-2':
            fig.circle(Ms[::marker_plot_every], kl50[::marker_plot_every], fill_color="white",
                       size=marker_size, legend=alg[1])
            fig2.circle(cput50[::marker_plot_every], kl50[::marker_plot_every], fill_color="white",
                        size=marker_size, legend=alg[1])
        elif alg[0] == 'SVI':
            fig.square(Ms[::marker_plot_every], kl50[::marker_plot_every], fill_color="white",
                       size=marker_size, legend=alg[1])
            fig2.square(cput50[::marker_plot_every], kl50[::marker_plot_every], fill_color="white",
                        size=marker_size, legend=alg[1])
        elif alg[0] == 'GIGAR':
            fig.square(Ms[::marker_plot_every], kl50[::marker_plot_every], fill_color=alg[2],
                       size=marker_size, legend=alg[1])
            fig2.square(cput50[::marker_plot_every], kl50[::marker_plot_every], fill_color=alg[2],
                        size=marker_size, legend=alg[1])

    for f in [fig, fig2]:
        f.legend.label_text_font_size = '25pt'
        f.legend.glyph_width = 40
        # f.legend.glyph_height = 40
        f.legend.spacing = 15
        f.legend.visible = True
        f.legend.location = 'bottom_left'

    legend_len = len(fig.legend.items)
    fig.legend.items = fig.legend.items[legend_len - 2:legend_len] + fig.legend.items[0:legend_len - 2]

    legend_len = len(fig2.legend.items)
    fig2.legend.items = fig2.legend.items[legend_len - 2:legend_len] + fig2.legend.items[0:legend_len - 2]

    postprocess_plot(fig, '22pt', location='bottom_left', glyph_width=40)
    fig.legend.background_fill_alpha = 0.
    fig.legend.border_line_alpha = 0.

    postprocess_plot(fig2, '22pt', location='bottom_left', glyph_width=40)
    fig2.legend.background_fill_alpha = 0.
    fig2.legend.border_line_alpha = 0.

# bkp.show(fig2)
bkp.show(bkl.gridplot(figs))
