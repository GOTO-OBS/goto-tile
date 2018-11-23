import os
import json
import base64
import tempfile
from django.views.generic.edit import FormView
from django.core.urlresolvers import reverse_lazy
from .forms import SkyMapForm


class SkyMapView(FormView):
    template_name = 'webtile/skymap.html'
    form_class = SkyMapForm
    success_url = '.'

    def form_valid(self, form):
        # Put the gototile and related imports here; this avoid
        # crashing all views when one of the below has a problem.
        # Set matplotlib's backend first, to a non-interactive one.
        import matplotlib
        matplotlib.use('Agg')
        from django.conf import settings
        import astropy.time
        import astropy.table
        import gototile
        from gototile.settings import GWGC_PATH, NSIDE
        from gototile.skymap import SkyMap
        from gototile.skymaptools import calculate_tiling
        from .utils import save_uploadedfile, create_grid, TiledMapError
        from gototile.utils import pointings_to_text

        telescope = form.cleaned_data['telescope']

        filename = save_uploadedfile(form.files['skymap'])
        catalog = form.cleaned_data['catalog']
        nightsky = form.cleaned_data['nightsky']
        geoplot = form.cleaned_data['geoplot']
        maxtiles = form.cleaned_data['maxtiles']
        fraction = form.cleaned_data['fraction']
        date = form.cleaned_data['date']
        title = form.cleaned_data['title']
        moon = form.cleaned_data['moon']
        sun = form.cleaned_data['sun']
        site = form.cleaned_data['site']
        fov = form.cleaned_data['fov']
        datevalue = form.cleaned_data['datevalue']
        objects = form.cleaned_data['objects']
        telescopes = form.cleaned_data['telescopes']

        date_init = form.cleaned_data['date-init']
        objects_init = form.cleaned_data['objects-init']
        site_init = form.cleaned_data['site-init']
        fov_init = form.cleaned_data['fov-init']
        self.request.session['skymap-initial'] = dict(
            telescope=telescope, catalog=catalog, nightsky=nightsky,
            geoplot=geoplot, maxtiles=maxtiles, fraction=fraction,
            moon=moon, sun=sun, title=title, date=date_init,
            datevalue=datevalue,
            objects=objects_init, site=site_init, fov=fov_init)
        self.request.session.set_expiry(7200)

        skymapimage = tempfile.NamedTemporaryFile(
            prefix='tiledmap-', suffix='.png', delete=False).name
        pointingfile = tempfile.NamedTemporaryFile(
            prefix='pointings-', suffix='.csv', delete=False).name
        self.request.session['skymap-files'] = dict(
            skymapimage=skymapimage, pointingfile=pointingfile)

        grids = create_grid(telescopes)

        if catalog is True:
            catalog = {'path': GWGC_PATH, 'key': 'weight'}
        else:
            catalog = {'path': None, 'key': None}
        coverage = {'min': 0, 'max': fraction}
        skymap = SkyMap.from_fits(filename)
        skymap.regrade(nside=NSIDE)
        try:
            pointings, _, _ = calculate_tiling(
                skymap, telescopes, date=date, maxtiles=maxtiles,
                nightsky=nightsky, catalog=catalog,
                coverage=coverage, tilespath=grids)
            if date is None:
                date = skymap.date_det
            options = {'sun': sun, 'moon': moon}
            skymap.plot(filename=skymapimage, telescopes=telescopes,
                        date=date, pointings=pointings,
                        geoplot=geoplot, catalog=catalog,
                        nightsky=nightsky, title=title,
                        objects=objects, options=options)
            table = pointings_to_text(pointings, catalog=catalog)
            table.write(pointingfile, format='ascii.csv')
        except TiledMapError as exc:
            raise
        finally:
            os.unlink(filename)

        return super().form_valid(form)

    def get_context_data(self, **kwargs):
        from django.conf import settings
        import astropy.table

        context = super().get_context_data(**kwargs)

        files = self.request.session.pop('skymap-files', {})
        skymapimage = files.get('skymapimage', '')
        pointingfile = files.get('pointingfile', '')
        try:
            with open(skymapimage, 'rb') as infile:
                image = infile.read()
            image = base64.b64encode(image)
            context['image'] = image
            os.unlink(skymapimage)
        except FileNotFoundError:
            context['image'] = None
        try:
            context['pointings'] = astropy.table.Table.read(pointingfile)
            os.unlink(pointingfile)
        except FileNotFoundError:
            context['pointings'] = None
        return context

    def get_initial(self):
        initial = self.request.session.get('skymap-initial')
        if not initial:
            initial = dict(telescope='GOTO4', maxtiles=100, fraction=0.90,
                           catalog=True, nightsky=True, geoplot=False)
        return initial
