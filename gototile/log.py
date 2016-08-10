"""Logging settings for gototile"""

import logging
import astropy


def set_logging(verbose, quiet=False):
    loglevel = ['WARNING', 'INFO', 'DEBUG'][verbose]
    logging.basicConfig(
        level=loglevel,
        format='%(asctime)s -- %(funcName)s [%(levelname)s]: %(message)s',
        datefmt='%Y-%m-%dT%H:%M:%S')
    astropy.log.setLevel(loglevel)
    if quiet:
        logging.getLogger().setLevel(logging.ERROR)
        astropy.log.setLevel('ERROR')
        astropy.log.disable_warnings_logging()

        
