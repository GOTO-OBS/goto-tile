from django import forms
from django.utils.translation import ugettext as _
from django.conf import settings


class SkyMapForm(forms.Form):
    skymap = forms.FileField(help_text="The sky map FITS file (Bayestar file)")
    telescope = forms.ChoiceField(choices=(
        ('GOTON4', 'GOTO 4 scope configuration / La Palma'),
        ('GOTON8', 'GOTO 8 scope configuration / La Palma'),
        ('SuperWASPN', 'SuperWASP North / La Palma'),
        ('VISTA', 'VISTA / La Silla'),
    ))
    site = forms.CharField(
        label='Alternate site', label_suffix='', required=False,
        help_text=("Latitude and longitude (in that order!), "
                   "separated by whitespace. Use degrees or "
                   "sexagesimal "
                   "notation with 'hms', 'dms' or ':' "
                   "separators. Leave this field blank to use the "
                   "site settings from the telescope option."))
    fov = forms.CharField(
        label='Alternate FoV', label_suffix='', required=False,
        help_text=("Field of view along RA and Dec, in degrees, "
                   "separated by whitespace. Leave blank to use the "
                   "default settings from the telescope option"))
    catalog = forms.BooleanField(
        label='', required=False,
        help_text="Use the GWGC galaxy catalog")
    nightsky = forms.BooleanField(
        label='', required=False,
        help_text="Use the visible night sky")
    geoplot = forms.BooleanField(
        label='', required=False,
        help_text="Create the plot in geographic "
        "coordinates (latitude & longitude instead "
        "of Right Ascension & declination)")
    fraction = forms.FloatField(
        label='', min_value=0.01, max_value=0.99,
        help_text="Fraction of the probabiliy map to cover")
    maxtiles = forms.IntegerField(
        label='', min_value=1, max_value=500,
        help_text="Maximum number of tiles to generate")
    date = forms.ChoiceField(choices=(
        ('current', 'Current'),
        ('header', 'From file header'),
        ('specify', 'Specify'),
    ))
    datevalue = forms.CharField(
        label='', required=False,
        help_text="Formats like '2015-1-1 12:00:00', "
        "'2015/1/1' or '12345 mjd' are accepted")
    moon = forms.BooleanField(
        label='', required=False,
        help_text="Overplot the moon position")
    sun = forms.BooleanField(
        label='', required=False,
        help_text="Overplot the Sun position")
    objects = forms.CharField(
        required=False,
        help_text=("List of objects to overplot. Each object consists of an "
                   "RA, Dec and name value, with their values separated by "
                   "whitespace. For RA and Dec, use degrees or sexagesimal "
                   "notation with 'hms'/'dms' or ':' separators. Separate "
                   "objects with a comma."),
        widget=forms.TextInput(attrs={'size': '40'}))
    title = forms.CharField(
        required=False,
        widget=forms.TextInput(attrs={'size': '40'}),
        help_text="Figure title. Leave blank for a generated one.")


    def clean(self):
        import astropy.time
        from astropy.coordinates import Angle, SkyCoord, EarthLocation
        from astropy import units
        from .utils  import get_telescopes

        data = super().clean()
        data['date-init'] = data['date']
        data['objects-init'] = data['objects']
        data['site-init'] = data['site']
        data['fov-init'] = data['fov']


        if data['date'] == 'specify':
            if 'datevalue' not in data:
                raise forms.ValidationError(_('Please supply a date'),
                                            code='missing')
            string = data['datevalue']
            try:
                string = string.strip()
                if string.lower().endswith('jd'):
                    if string.lower().endswith('mjd'):
                        date = astropy.time.Time(float(string[:-3]),
                                                 format='mjd')
                    else:
                        date = astropy.time.Time(float(string[:-2]),
                                                 format='jd')
                else:
                    date = astropy.time.Time(string.replace('/', '-'))
                data['date'] = date
            except ValueError:
                raise forms.ValidationError(
                    _("Invalid date format: %(value)s"),
                    code='invalid', params={'value': string})
        elif data['date'] == 'current':
            data['date'] = astropy.time.Time.now()
        elif data['date'] == 'header':
            data['date'] = None

        if len(data.get('objects', '')):
            objects = data['objects'].split(',')
            for objnr, obj in enumerate(objects):
                fields = obj.strip().split()
                if len(fields) > 3:
                    raise forms.ValidationError(
                        _("Too many entries for object #%(objnr)s. Supply "
                          "RA, dec and a name, separated by whitespace"),
                          code='too many', params={'objnr': str(objnr+1)})
                elif len(fields) < 3:
                    raise forms.ValidationError(
                        _("Too few entries for object #%(objnr)s. Supply "
                          "RA, dec and a name, separated by whitespace"),
                          code='too few', params={'objnr': str(objnr+1)})
                ra, dec, name = fields
                try:
                    if ':' in ra:
                        ra = Angle(ra, unit=units.hour)
                    else:
                        try:
                            ra = Angle(ra)
                        except units.UnitsError:
                            ra = Angle(ra, unit=units.degree)
                    if ':' in dec:
                        dec = Angle(dec, unit=units.degree)
                    else:
                        try:
                            dec = Angle(dec)
                        except units.UnitsError:
                            dec = Angle(dec, unit=units.degree)
                except (ValueError, units.UnitsError):
                    raise forms.ValidationError(
                        _("Invalid coordinate format for object %(name)s: "
                          "%(value)s"), code='invalid',
                        params={'name': name, 'value': " ".join(fields[:2])})
                coord = SkyCoord(ra, dec)
                coord.name = name
                objects[objnr] = coord
            data['objects'] = objects
        else:
            data['objects'] = None

        if not data['title']:
            data['title'] = None

        telescopes = get_telescopes()
        telescope = telescopes[data['telescope']]()

        if data['site']:
            fields = data['site'].strip().split()
            if len(fields) > 2:
                raise forms.ValidationError(
                    _("Too many entries for 'Alternate site'. Supply "
                      "latitude and longitude, separated by whitespace"),
                    code='too many')
            elif len(fields) < 2:
                raise forms.ValidationError(
                    _("Too few entries for 'Alternate site'. Supply "
                      "latitude and longitude, separated by whitespace"),
                    code='too few')
            lat, lon = fields
            try:
                if ':' in lon:
                    lon = Angle(lon, unit=units.hour)
                else:
                    try:
                        lon = Angle(lon)
                    except units.UnitsError:
                        lon = Angle(lon, unit=units.degree)
                if ':' in lat:
                    lat = Angle(lat, unit=units.degree)
                else:
                    try:
                        lat = Angle(lat)
                    except units.UnitsError:
                        lat = Angle(lat, unit=units.degree)
            except (ValueError, units.UnitsError):
                raise forms.ValidationError(
                    _("Invalid coordinate format for 'Alternate site': "
                      "%(value)s"),
                    code='invalid', params={'value': data['site']})
            location = EarthLocation.from_geodetic(lon, lat, 0)
            data['site'] = location
            telescope.location = location

        if data['fov']:
            fields = data['fov'].strip().split()
            if len(fields) > 2:
                raise forms.ValidationError(
                    _("Too many entries for 'Alternate site'. "
                      "Supply width and height, separated by whitespace"),
                    code='too many')
            elif len(fields) < 2:
                raise forms.ValidationError(
                    _("Too few entries for 'Alternate site'. "
                      "Supply width and height, separated by whitespace"),
                    code='too few')
            try:
                fov = {'ra': float(fields[0]) * units.degree,
                       'dec': float(fields[1]) * units.degree}
            except ValueError:
                raise forms.ValidationError(
                    _("Invalid format for 'Alternative FoV': %(value)s"),
                    code='invalid format', params={'value': data['fov']})
            data['fov'] = fov
            telescope.fov = fov

        # At the moment, we deal with just one telescope
        data['telescopes'] = [telescope]

        return data
