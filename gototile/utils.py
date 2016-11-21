import logging
from astropy.utils import iers


class FileNotFoundError(IOError):
    pass

class FileExistsError(IOError):
    pass


def pointings_to_text(pointings, catalog=None):
    if catalog is None:
        catalog = {'path': None, 'key': None}
    table = pointings[['fieldname', 'prob', 'cumprob', 'telescope']].copy()
    table['prob'] = ["{:.5f}".format(100 * prob)
                     for prob in table['prob']]
    table['cumprob'] = ["{:.5f}".format(100*prob)
                        for prob in  table['cumprob']]
    table['ra'] = ["{:.5f}".format(center.ra.deg)
                   for center in pointings['center']]
    table['dec'] = ["{:.5f}".format(center.dec.deg)
                    for center in pointings['center']]
    # %z was added in Python 3.3, and %Z is deprecated
    table['time'] = [time.datetime.strftime('%Y-%m-%dT%H:%M:%S%z')
                     for time in pointings['time']]
    table['dt'] = ["{:.5f}".format(dt.jd) for dt in pointings['dt']]
    columns = ['telescope', 'fieldname', 'ra', 'dec', 'time', 'dt',
               'prob', 'cumprob']
    if catalog['path']:
        table['ncatsources'] = [len(sources)
                                for sources in pointings['sources']]
        columns.append('ncatsources')
    return table[columns]



def test_iers():
    """Test whether the IERS is up and running, or whether it times out

    If the IERS-A server times out, we set the download URL to an
    empty string '': this will let the attempted download
    *immediately* fail, without timeout, and use the local IERS-B
    tables

    To get rid of further warnings, we also set the maximum allowed
    age to None.

    Note: the astropy.utils.iers documentation suggest to set
    ``iers.conf.auto_download = False``. This did not work for me.

    """
    logger = logging.getLogger(__name__)

    logger.debug("Verifying IERS-A server")
    for url in (None, 'http://toshi.nofs.navy.mil/ser7/finals2000A.all'):
        if url is not None:
            iers.conf.iers_auto_url = url
            logger.debug("Opening alternative IERS-A server: %s", url)
        else:
            logger.debug("Opening default IERS-A server: %s",
                         iers.conf.iers_auto_url)
        try:
            iers.iers.download_file(iers.conf.iers_auto_url, cache=True)
            break
        except URLError:
            pass
    else:
        logger.warning("No IERS-A server found; ignoring IERS data")
        iers.conf.auto_download = False
        iers.conf.auto_max_age = None
        iers.conf.iers_auto_url = ''
