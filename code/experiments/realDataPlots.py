import numpy as np
import matplotlib.pyplot as plt
import pickle
import copy

f = open('realDataResults.pkl', 'rb')
results = pickle.load(f)
f.close()

datasets = ['p53', 'rcv1', 'blog']
errIJ = []
errNS = []
errIJTilde = []
errNSTilde = []

timesExact = []
timesNS = []
timesNSTilde = []
timesIJ = []
timesIJTilde = []


for dset in datasets:
  exact = results[dset]['exactCV']
  denom = results[dset]['exactCV']
  errIJ.append((np.abs(results[dset]['IJ']
                       - exact) / np.abs(denom)).mean() * 100)
  errNS.append((np.abs(results[dset]['NS']
                       - exact) / np.abs(denom)).mean() * 100)
  errIJTilde.append((np.abs(results[dset]['IJTilde']
                            - exact) / np.abs(denom)).mean() * 100)
  errNSTilde.append((np.abs(results[dset]['NSTilde']
                            - exact) / np.abs(denom)).mean() * 100)

  timesExact.append(results[dset]['timings']['exactCV'])
  timesNS.append(results[dset]['timings']['ACVExact'])
  timesNSTilde.append(results[dset]['timings']['ACVTilde'])
  timesIJ.append(results[dset]['timings']['ACVExact'])
  timesIJTilde.append(results[dset]['timings']['ACVTilde'])
  


tickFontsize = 18
linewidth = 3.0
axlabelFontsize = 20
titleFontsize = 18
legendFontsize = 14


width = 0.15
align = 'edge'
xs = np.arange(len(datasets))
plt.figure(figsize=(6,5))
barIJ = plt.bar(xs+width/2, errIJ, width, align=align,
        label=r'$\mathrm{IJ}^{\backslash n}$')
barIJTilde = plt.bar(xs+1.5*width, errIJTilde, width,
                     align=align,
                     label=r'$\widetilde \mathrm{IJ}^{\backslash n}$',
                     hatch='//')
barNS = plt.bar(xs-1.5*width, errNS, width, align=align,
                label=r'$\mathrm{NS}^{\backslash n}$')
barNSTilde = plt.bar(xs-0.5*width, errNSTilde, width, align=align,
                     label=r'$\widetilde \mathrm{NS}^{\backslash n}$',
                     hatch='//')
plt.yscale('log')
plt.gca().set_xticks(xs)
plt.gca().set_xticklabels(datasets)
plt.legend(fontsize=legendFontsize,
           bbox_to_anchor=(0.77,0.7),
           ncol=2)
plt.gca().tick_params(axis='both',
                      which='major',
                      labelsize=tickFontsize)
plt.ylabel(r'Avg. % error to $x_n^T \hat\theta_{\backslash n}$', fontsize=axlabelFontsize)
plt.tight_layout()
plt.savefig('C:YOUR_FILEPATH/real_data_accuracy.png', bbox='tight')


# BEHOLD, THE ARCANUM OF MATPLOTLIB
colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
#( what we want is the *next* color after the first four to plot
#   the exact CV bar)
plt.figure(figsize=(5,5))
plt.bar(xs-2*width, timesExact, width,
        color=colors[4],
        label='Exact')
plt.bar(xs+width/2, timesIJ, width, align=align,
        color=barIJ[0].get_facecolor(),
        label=r'$\mathrm{IJ}^{\backslash n}$',
        )
plt.bar(xs+1.5*width, timesIJTilde, width,
        align=align,
        color=barIJTilde[0].get_facecolor(),
        label=r'$\widetilde \mathrm{IJ}^{\backslash n}$',
        hatch='//')
plt.bar(xs-1.5*width, timesNS, width, align=align,
        color=barNS[0].get_facecolor(),
        label=r'$\mathrm{NS}^{\backslash n}$')
plt.bar(xs-0.5*width, timesNSTilde, width, align=align,
        color=barNSTilde[0].get_facecolor(),
        label=r'$\widetilde \mathrm{NS}^{\backslash n}$',
        hatch='//')

#plt.bar(xs-width, timesExact, width,
#        color=colors[4],
#        label='Exact')
#plt.bar(xs, timesNS, width,
#        color=barNS[0].get_facecolor(),
#        label=r'$\mathrm{NS}^{\backslash n}$')
#plt.bar(xs+width, timesNSTilde, width,
#        color=barNSTilde[0].get_facecolor(),
#        label=r'$\widetilde \mathrm{NS}^{\backslash n}$')
plt.yscale('log')
plt.gca().set_xticks(xs)
plt.gca().set_xticklabels(datasets)
plt.gca().tick_params(axis='both',
                      which='major',
                      labelsize=tickFontsize)
plt.ylabel('Time (seconds)', fontsize=axlabelFontsize)
plt.legend(fontsize=legendFontsize,
           loc='best',
           bbox_to_anchor=(0.410, 0.45))
plt.tight_layout()
plt.savefig('C:YOUR_FILEPATH/real_data_timings.png', bbox='tight')
plt.show()

