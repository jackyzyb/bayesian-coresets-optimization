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

is_stoc_grad = True

if is_stoc_grad:
    algs = [('GIGAR', 'GIGA', pal[0]), ('SVI', 'SparseVI', pal[1]), ('RAND', 'Uniform', pal[2]),
            ('PRIOR', 'Prior', 'black'), ('IHT', 'A-IHT', pal[3]), ('IHT-2', 'A-IHT II', pal[4]),
            ('IHT-stoc', 'A-IHT batch grad.', pal[5])
            ]
else:
    algs = [('GIGAR', 'GIGA', pal[0]), ('SVI', 'SparseVI', pal[1]), ('RAND', 'Uniform', pal[2]),
            ('PRIOR', 'Prior', 'black'), ('IHT', 'A-IHT', pal[3]), ('IHT-2', 'A-IHT II', pal[4])
            #('IHT-stoc', 'A-IHT batch grad.', pal[5])
            ]
dnames = ['synth_lr', 'ds1', 'phishing', 'synth_poiss', 'biketrips', 'airportdelays']  # dataset names in ['synth_lr', 'ds1', 'phishing', 'synth_poiss', 'biketrips', 'airportdelays']
                                            # or ['synth_lr_large', 'phishing_old'] (need to adjust x range)

plot_every = 5
marker_plot_every = 4
marker_size = 25
is_forward_KL = False    # if not using symmetrized KL
is_symmetrized_KL = True
if is_symmetrized_KL:
    y_label = 'Symmetrized KL'
else:
    if is_forward_KL:
        y_label = 'Forward KL'
    else:
        y_label = 'Reverse KL'



def compute_reverse_KL(dnm, res):
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
    # kl = kl[2:][::plot_every]
    return kl

figs = []
for dnm in dnames:
    print('Plotting ' + dnm)
    fig = bkp.figure(y_axis_type='log', y_axis_label=y_label, x_axis_label='Coreset Size k', plot_width=1000, plot_height=1000, x_range=(-10, 110))
    # fig = bkp.figure(y_axis_type='log', y_axis_label=y_label, x_axis_label='k', plot_width=1000, plot_height=1000,
    #                  x_range=(-10, 210))
    #preprocess_plot(fig, '32pt', True, True)
    #fig.yaxis.formatter
    fig2 = bkp.figure(y_axis_type='log', y_axis_label=y_label, x_axis_type='log', x_axis_label='CPU Time (s)',
                      plot_width=1000, plot_height=1000)
    #preprocess_plot(fig2, '32pt', True, True)
    fig3 = bkp.figure(y_axis_type='auto', y_axis_label='CPU Time (s)', x_axis_label='Coreset Size k', plot_width=1000,
                      plot_height=1000, x_range=(-10, 110))
    #preprocess_plot(fig3, '32pt', False, False)

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
        if is_symmetrized_KL:
            rkl = compute_reverse_KL(dnm, res)
        assert np.all(res['kls'] == res['kls'][0])  # make sure prior doesn't change...
        if is_symmetrized_KL:
            kltot += res['kls'][0] + rkl[0]
        else:
            kltot += res['kls'][0]
    std_kls[dnm] = kltot / len(trials)

    for alg in algs:
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
            if is_symmetrized_KL:
                rkl = compute_reverse_KL(dnm, res)[2:][::plot_every]
                fkl = res['kls'][2:][::plot_every]
                kl = rkl + fkl
            else:
                if is_forward_KL is not True:
                    kl = compute_reverse_KL(dnm, res)[2:][::plot_every]
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

        if alg[0] != 'PRIOR':
            fig2.line(cput50, kl50, color=alg[2], legend_label=alg[1], line_width=10)
            fig2.patch(np.hstack((cput50, cput50[::-1])), np.hstack((kl75, kl25[::-1])), fill_color=alg[2],
                       legend_label=alg[1], alpha=0.3)
            fig2.legend.location = 'bottom_left'

        if alg[0] != 'PRIOR':
            fig3.line(Ms, cput50, color=alg[2], legend_label=alg[1], line_width=10)
            fig3.patch(np.hstack((Ms, Ms[::-1])), np.hstack((cput75, cput25[::-1])), fill_color=alg[2], legend_label=alg[1],
                      alpha=0.3)
            fig3.legend.location = 'top_left'
        if alg[0] == 'IHT':
            fig.circle(Ms[::marker_plot_every], kl50[::marker_plot_every], fill_color=alg[2],
                       size=marker_size, legend=alg[1])
            fig2.circle(cput50[::marker_plot_every], kl50[::marker_plot_every], fill_color=alg[2],
                        size=marker_size, legend=alg[1])
            fig3.circle(Ms[::marker_plot_every], cput50[::marker_plot_every], fill_color=alg[2],
                       size=marker_size, legend=alg[1])
        elif alg[0] == 'IHT-2':
            fig.circle(Ms[::marker_plot_every], kl50[::marker_plot_every], fill_color="white",
                       size=marker_size, legend=alg[1])
            fig2.circle(cput50[::marker_plot_every], kl50[::marker_plot_every], fill_color="white",
                        size=marker_size, legend=alg[1])
            fig3.circle(Ms[::marker_plot_every], cput50[::marker_plot_every], fill_color="white",
                       size=marker_size, legend=alg[1])
        elif alg[0] == 'IHT-stoc':
            fig.circle(Ms[::marker_plot_every], kl50[::marker_plot_every], fill_color=alg[2],
                       size=marker_size, legend=alg[1])
        elif alg[0] == 'SVI':
            fig.square(Ms[::marker_plot_every], kl50[::marker_plot_every], fill_color="white",
                       size=marker_size, legend=alg[1])
            fig2.square(cput50[::marker_plot_every], kl50[::marker_plot_every], fill_color="white",
                        size=marker_size, legend=alg[1])
            fig3.square(Ms[::marker_plot_every], cput50[::marker_plot_every], fill_color="white",
                       size=marker_size, legend=alg[1])
        elif alg[0] == 'GIGAR':
            fig.square(Ms[::marker_plot_every], kl50[::marker_plot_every], fill_color=alg[2],
                       size=marker_size, legend=alg[1])
            fig2.square(cput50[::marker_plot_every], kl50[::marker_plot_every], fill_color=alg[2],
                        size=marker_size, legend=alg[1])
            fig3.square(Ms[::marker_plot_every], cput50[::marker_plot_every], fill_color=alg[2],
                       size=marker_size, legend=alg[1])

    for f in [fig, fig2, fig3]:
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

    legend_len = len(fig2.legend.items)
    fig2.legend.items = fig2.legend.items[legend_len - 2:legend_len] + fig2.legend.items[0:legend_len - 2]

    legend_len = len(fig3.legend.items)
    fig3.legend.items = fig3.legend.items[legend_len - 2:legend_len] + fig3.legend.items[0:legend_len - 2]

    postprocess_plot(fig, '22pt', location='bottom_left', glyph_width=40)
    fig.legend.background_fill_alpha = 0.
    fig.legend.border_line_alpha = 0.

    postprocess_plot(fig2, '22pt', location='bottom_left', glyph_width=40)
    fig2.legend.background_fill_alpha = 0.
    fig2.legend.border_line_alpha = 0.

    postprocess_plot(fig3, '22pt', location='top_left', glyph_width=40)
    fig3.legend.background_fill_alpha = 0.
    fig3.legend.border_line_alpha = 0.

    fig.output_backend = 'svg'
    fig2.output_backend = 'svg'
    fig3.output_backend = 'svg'

    if is_symmetrized_KL:
        y_name = 'SKL'
    else:
        if is_forward_KL:
            y_name = 'FKL'
        else:
            y_name = 'RKL'
    if is_stoc_grad:
        save_file_handle = dnm + '_stoc_'
    else:
        save_file_handle = dnm + '_'
    fig_name = save_file_handle + y_name
    fig2_name = save_file_handle + 'log_time'
    fig3_name = save_file_handle + 'linear_time'

    export_svgs(fig, filename=fig_name+'.svg')
    export_svgs(fig2, filename=fig2_name+'.svg')
    export_svgs(fig3, filename=fig3_name+'.svg')

    # cairosvg.svg2pdf(url=fig_name+'.svg', write_to=fig_name+'.pdf')
    cairosvg.svg2pdf(
        file_obj=open(fig_name+'.svg', "rb"), write_to=fig_name+'.pdf')
    cairosvg.svg2pdf(
        file_obj=open(fig2_name + '.svg', "rb"), write_to=fig2_name + '.pdf')
    cairosvg.svg2pdf(
        file_obj=open(fig3_name + '.svg', "rb"), write_to=fig3_name + '.pdf')

#bkp.show(fig)
bkp.show(bkl.gridplot(figs))




