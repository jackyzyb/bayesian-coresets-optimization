import bokeh.layouts as bkl
import bokeh.plotting as bkp
import numpy as np
import sys,os
from bokeh.models import ColumnDataSource, Range1d, LabelSet, Label
#make it so we can import models/etc from parent folder
sys.path.insert(1, os.path.join(sys.path[0], '../common'))
from plotting import *
from skimage import measure


print('Loading data')
x = np.load('../data/prices2018.npy')

latbds = (x[:, 0].min(), x[:, 0].max())
lonbds = (x[:, 1].min(), x[:, 1].max())

lats = np.linspace(latbds[0], latbds[1], 100)
lons = np.linspace(lonbds[0], lonbds[1], 100)

longrid, latgrid = np.meshgrid(lons, lats)

contour_percentiles = np.linspace(0, 100, 10)
c = contour_percentiles / contour_percentiles.max()
contour_colors = ['#%02x%02x%02x' % (int(r), int(b), int(g)) for (r, b, g) in zip(255*c, 0*np.ones(c.shape[0]), 255*(1.-c))]

#algorithm / trial + Ms to plot
#nm = ('SVI', 'SparseVI')
nm = ('IHT-2', 'A-IHT II')

trial_num = 1
Ms = [300]

#plot the sequence of coreset pts and comparison of nonopt + opt
res = np.load('results/results_'+nm[0] + '_' + str(trial_num)+'.npz')
x = res['x']
wt = res['w']
mup = res['mup']
Sigp = res['Sigp']
muwt = res['muw']
Sigwt = res['Sigw']
basis_scales = res['basis_scales']
basis_locs = res['basis_locs']
datastd = res['datastd']

figs = []

#true posterior figure
fig = bkp.figure(x_range=lonbds, y_range=latbds, plot_width=1000, plot_height=1000)
#for f in [fig, fig_opt]:
for f in [fig]:
  preprocess_plot(f, '32pt', False, False)

#plot data and coreset pts
f.scatter(x[:, 1], x[:, 0], fill_color='black', size=12, alpha=0.01, line_color=None)
#compute posterior mean regression on the grid
reg = np.zeros(longrid.shape)
for i in range(basis_scales.shape[0]):
  reg += mup[i]*np.exp(-(longrid - basis_locs[i,1])**2/(2*basis_scales[i]**2) - (latgrid - basis_locs[i,0])**2/(2*basis_scales[i]**2) )
#contour_levels
contour_levels = [np.percentile(reg, p) for p in contour_percentiles]
#plot contours
for color, level in zip(contour_colors, contour_levels):
  contours = measure.find_contours(reg, level)
  for contour in contours:
    #interpolate values
    latlons = np.hstack(( np.interp(contour[:, 0], np.arange(lats.shape[0]), lats)[:, np.newaxis], np.interp(contour[:, 1], np.arange(lons.shape[0]), lons)[:, np.newaxis]))
    f.line(latlons[:, 1], latlons[:, 0], line_width=2, line_color=color)
   
for f in [fig]:
  postprocess_plot(f, '32pt', orientation='horizontal', glyph_width=40)
  f.legend.background_fill_alpha=0.
  f.legend.border_line_alpha=0.
  #f.legend.visible=False
  f.xaxis.visible = False
  f.yaxis.visible = False
  countour_legend = Label(x=50, y=900, x_units='screen', y_units='screen',
                   text='True Posterior', text_font_size='32pt')
  f.add_layout(countour_legend)

figs.append([fig])



#coreset figures
for m in Ms:
  fig = bkp.figure(x_range=lonbds, y_range=latbds, plot_width=1000, plot_height=1000)
  preprocess_plot(fig, '32pt', False, False)

  #plot data and coreset pts
  fig.scatter(x[:, 1], x[:, 0], fill_color='black', size=12, alpha=0.01, line_color=None)
  #fig.scatter(x[:, 1], x[:, 0], fill_color='black', size=10*(wt[m, :]>0)+10*wt[m,:]/wt[m,:].max(), line_color=None)
  fig.scatter(x[:, 1], x[:, 0], fill_color='black', size=30*np.power(wt[m,:]/wt[m,:].max(), 0.4), line_color=None)
  #compute posterior mean regression on the grid
  reg = np.zeros(longrid.shape)
  for i in range(basis_scales.shape[0]):
    reg += muwt[m, i]*np.exp(-(longrid - basis_locs[i,1])**2/(2*basis_scales[i]**2) - (latgrid - basis_locs[i,0])**2/(2*basis_scales[i]**2) )
  #plot contours
  for color, level in zip(contour_colors, contour_levels):
    contours = measure.find_contours(reg, level)
    for contour in contours:
      #interpolate values
      latlons = np.hstack(( np.interp(contour[:, 0], np.arange(lats.shape[0]), lats)[:, np.newaxis], np.interp(contour[:, 1], np.arange(lons.shape[0]), lons)[:, np.newaxis]))
      fig.line(latlons[:, 1], latlons[:, 0], line_width=2, line_color=color)
     
  postprocess_plot(fig, '32pt', orientation='horizontal', glyph_width=40)
  fig.legend.background_fill_alpha=0.
  fig.legend.border_line_alpha=0.
  fig.xaxis.visible = False
  fig.yaxis.visible = False
  countour_legend = Label(x=50, y=900, x_units='screen', y_units='screen',
                          text=nm[1]+' Corset Posterior', text_font_size='32pt')
  fig.add_layout(countour_legend)

  figs.append([fig])

bkp.show(bkl.gridplot(figs))




