class FileNotFoundError(IOError):
    pass

class FileExistsError(IOError):
    pass


def pointings_to_text(pointings, catalog=None):
    if catalog is None:
        catalog = {'path': None, 'key': None}
    table = pointings[['prob', 'cumprob', 'telescope']].copy()
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
    columns = ['telescope', 'ra', 'dec', 'time', 'dt', 'prob', 'cumprob']
    if catalog['path']:
        table['ncatsources'] = [len(sources)
                                for sources in pointings['sources']]
        columns.append('ncatsources')
    return table[columns]
