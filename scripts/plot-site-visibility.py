import sys
import pickle
import numpy as np
from astropy import units
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure


with open(sys.argv[1], 'rb') as infile:
    data = pickle.load(infile)

data2 = {}
for timespan in data:
    for obs in data[timespan]:
        times = []
        coverages = []
        for obsid, value in data[timespan][obs].items():
            times.append(value['time_since_t0'])
            coverages.append(value['coverage'])
        key = 24 if timespan is None else timespan
        data2.setdefault(obs, {})[key] = {'times': times, 'coverages': coverages}


covbins = np.arange(0, 1, 0.1)
delaybins = np.arange(0, 5, 0.2)
covfigure = Figure((12, 14))
detfigure = Figure((12, 14))
for i, name in enumerate(['SSO', 'MtBruce', 'MtMeharry', 'Mereenie', 'lapalma']):
    data = data2[name]
    coverages = []
    times = []
    labels = []
    for key in sorted(data.keys()):
        labels.append("{:.1f} hour".format(key))
        coverages.append(data[key]['coverages'])
        times.append(data[key]['times'])
    coverages = np.array(coverages)
    weights = np.ones_like(coverages)# / coverages.size
    coverages[coverages < 1e-8] = -1
    times = np.array(times) + 2.5/60
    times[np.isinf(times)] = 48
    figure = Figure((8, 6))
    canvas = FigureCanvas(figure)
    axes = figure.add_subplot(1, 1, 1)
    n, bins, patches = axes.hist(coverages.T, weights=weights.T,
                                 edgecolor='none', bins=covbins)
    axes.set_xlim(0, 1)
    axes.legend(handles=patches, labels=labels)
    axes.set_xlabel('coverage')
    axes.set_ylabel('fraction')
    axes.set_title(name)
    canvas.print_figure('coverage-{}.pdf'.format(name))

    axes1 = covfigure.add_subplot(3, 2, i+1)
    n, bins, patches1 = axes1.hist(coverages.T, weights=weights.T,
                                   edgecolor='none', bins=covbins)
    axes1.set_xlim(0, 1)
    axes1.set_xlabel('coverage (hour)')
    axes1.set_ylabel('fraction')
    axes1.set_title(name)
    
    figure = Figure((4, 6))
    canvas = FigureCanvas(figure)
    axes = figure.add_subplot(1, 1, 1)    
    n, bins, patches = axes.hist(times.T, weights=weights.T,
                                 edgecolor='none', bins=delaybins)
    axes.set_xlim(0, 5)
    axes.set_xlabel('time to source (hour)')
    axes.set_ylabel('fraction')
    axes.set_title(name)
    axes.legend(handles=patches, labels=labels)    
    canvas.print_figure('delay-{}.pdf'.format(name))

    axes2 = detfigure.add_subplot(3, 2, i+1)
    n, bins, patches2 = axes2.hist(times.T, weights=weights.T,
                                   edgecolor='none', bins=delaybins)
    axes2.set_xlim(0, 5)
    axes2.set_xlabel('time to source (hour)')
    axes2.set_ylabel('fraction')
    axes2.set_title(name)
    

axes1.legend(handles=patches1, labels=labels, loc='lower right',
             bbox_to_anchor=(2, 0.2), title='Legend')
canvas = FigureCanvas(covfigure)
canvas.print_figure('coverage.pdf')

axes2.legend(handles=patches2, labels=labels, loc='lower right',
             bbox_to_anchor=(2, 0.2), title='Legend')
canvas = FigureCanvas(detfigure)
canvas.print_figure('detections.pdf')


# # #  Fold (weigh) with number of visible nights # # #

clearnights = {'SSO': 114,
               'lapalma': 273,
               'MtBruce': 197,
               'MtMeharry': 201,
               'Mereenie': 193}

covbins = np.arange(0, 1, 0.1)
delaybins = np.arange(0, 5, 0.2)
covfigure = Figure((12, 14))
detfigure = Figure((12, 14))
for i, name in enumerate(['SSO', 'MtBruce', 'MtMeharry', 'Mereenie', 'lapalma']):
    data = data2[name]
    coverages = []
    times = []
    labels = []
    for key in sorted(data.keys()):
        labels.append("{:.1f} hour".format(key))
        coverages.append(data[key]['coverages'])
        times.append(data[key]['times'])
    coverages = np.array(coverages)
    weights = np.ones_like(coverages) / 365 * clearnights[name]
    coverages[coverages < 1e-8] = -1
    times = np.array(times) + 2.5/60
    times[np.isinf(times)] = 48
    figure = Figure((8, 6))
    canvas = FigureCanvas(figure)
    axes = figure.add_subplot(1, 1, 1)
    n, bins, patches = axes.hist(coverages.T, weights=weights.T,
                                 edgecolor='none', bins=covbins)
    axes.set_xlim(0, 1)
    axes.legend(handles=patches, labels=labels)
    axes.set_xlabel('coverage')
    axes.set_ylabel('number')
    axes.set_title(name)
    canvas.print_figure('coverage-clearnights-{}.pdf'.format(name))

    axes1 = covfigure.add_subplot(3, 2, i+1)
    n, bins, patches1 = axes1.hist(coverages.T, weights=weights.T,
                                   edgecolor='none', bins=covbins)
    axes1.set_xlim(0, 1)
    axes1.set_xlabel('coverage (hour)')
    axes1.set_ylabel('number')
    axes1.set_title(name)
    
    figure = Figure((4, 6))
    canvas = FigureCanvas(figure)
    axes = figure.add_subplot(1, 1, 1)    
    n, bins, patches = axes.hist(times.T, weights=weights.T,
                                 edgecolor='none', bins=delaybins)
    axes.set_xlim(0, 5)
    axes.set_xlabel('time to source (hour)')
    axes.set_ylabel('number')
    axes.set_title(name)
    axes.legend(handles=patches, labels=labels)    
    canvas.print_figure('delay-clearnights-{}.pdf'.format(name))

    axes2 = detfigure.add_subplot(3, 2, i+1)
    n, bins, patches2 = axes2.hist(times.T, weights=weights.T,
                                   edgecolor='none', bins=delaybins)
    axes2.set_xlim(0, 5)
    axes2.set_xlabel('time to source (hour)')
    axes2.set_ylabel('number')
    axes2.set_title(name)
    

axes1.legend(handles=patches1, labels=labels, loc='lower right',
             bbox_to_anchor=(2, 0.2), title='Legend')
canvas = FigureCanvas(covfigure)
canvas.print_figure('coverage-clearnights.pdf')

axes2.legend(handles=patches2, labels=labels, loc='lower right',
             bbox_to_anchor=(2, 0.2), title='Legend')
canvas = FigureCanvas(detfigure)
canvas.print_figure('detections-clearnights.pdf')
