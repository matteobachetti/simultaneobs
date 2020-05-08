import os
import argparse
import copy
import socket
from dataclasses import dataclass

socket.setdefaulttimeout(600)  # set timeout to 10 minutes

import numpy as np
from astropy import log
from astropy.coordinates import SkyCoord
from astropy.table import Table, QTable, Column, unique
import astropy.units as u
from astroquery.heasarc import Heasarc, Conf
import pyvo as vo


@dataclass
class StandardTableInfo(object):
    tablename: str
    time: str = "time"
    end_time: str = 'end_time'
    obsid: str = 'obsid'
    ra: str = 'ra'
    dec: str = 'dec'
    name: str = 'name'


def set_default_mission_info():
    chandra_dict = StandardTableInfo('chanmaster', end_time=None)
    hitomi_dict = StandardTableInfo('hitomaster', end_time='stop_time')
    # integral_dict = copy.deepcopy(default)  # needs work
    nicer_dict = StandardTableInfo('nicermastr')
    nustar_dict = StandardTableInfo('numaster')
    suzaku_dict = StandardTableInfo('suzamaster', end_time='stop_time')
    swift_dict = StandardTableInfo('swiftmastr', time='start_time',
                                   end_time='stop_time')
    xmm_dict = StandardTableInfo('xmmmaster')
    xte_dict = StandardTableInfo('xtemaster',
                                 end_time=None, name='target_name')

    mission_info = {'nustar': nustar_dict,
                    'chandra': chandra_dict,
                    'nicer': nicer_dict,
                    'xmm': xmm_dict,
                    'hitomi': hitomi_dict,
                    'suzaku': suzaku_dict,
                    'swift': swift_dict,
                    'xte': xte_dict
                    }

    return mission_info


mission_info = set_default_mission_info()


def get_rows_from_times(mission_table, times):
    """

    Examples
    --------
    >>> start = np.arange(2, 4)
    >>> end = np.arange(3, 5)
    >>> labels = ['AA', 'BB']
    >>> mission_table = QTable({'mjdstart': start, 'mjdend': end, 'label': labels})
    >>> times = np.array([1.5, 2.1, 3.5, 4.5, 5.5])
    >>> table = get_rows_from_times(mission_table, times)
    >>> np.allclose(table['mjdstart'], [0, 2, 3, 0, 0])
    True
    >>> np.allclose(table['mjdend'], [0, 3, 4, 0, 0])
    True
    >>> np.all(table['label'] == np.array(['', 'AA', 'BB', '', '']))
    True
    """
    # Avoid invalid values
    mission_table = mission_table[mission_table['mjdstart'] > 0]
    start, end = mission_table['mjdstart'], mission_table['mjdend']
    idxs = np.searchsorted(start, times + 1 / 86400)

    result_table = QTable()
    good = (times >= start[0])&(times <=end[-1])
    places_to_change = mission_table[idxs[good] - 1]
    for col in mission_table.colnames:
        newarr = np.zeros(times.size, dtype=mission_table[col].dtype)
        newarr[good] = places_to_change[col]
        result_table[col] = newarr

    return result_table


def get_table_from_heasarc(mission,
        max_entries=10000000, ignore_cache=False):
    settings = mission_info[mission]
    cache_file = f'_{settings.tablename}_table_cache.hdf5'

    if not ignore_cache and os.path.exists(cache_file):
        log.info(f"Getting cached table {cache_file}...")
        table = Table.read(cache_file)
        log.info("Done")
        return table

    heasarc_tap = vo.dal.TAPService(
        "https://heasarc.gsfc.nasa.gov/xamin/vo/tap/")

    colnames = (f"{settings.time},"
                f"{settings.ra},{settings.dec},"
                f"{settings.name},{settings.obsid}")

    if settings.end_time is not None:
        colnames += f",{settings.end_time}"

    query = f"""SELECT
    TOP {max_entries}
    "__row", {colnames}
    FROM {settings.tablename}
    """

    log.info(f"Querying {settings.tablename} table...")
    table = heasarc_tap.search(query).to_table()

    for key in ['obsid', 'name']:
        col = getattr(settings, key)
        values = [f"{value}" for value in table[col]]
        table.remove_column(col)
        table[key] = values

    for col in ['__row']:
        values = [float(value) for value in table[col]]
        table.remove_column(col)
        table[col] = values

    table.rename_column(settings.time, 'mjdstart')

    table.sort('mjdstart')

    if settings.end_time is None:
        table['mjdend'] = \
            np.concatenate((table['mjdstart'][1:], table['mjdstart'][-1:] + 1))
    else:
        table.rename_column(settings.end_time, 'mjdend')

    good = table['mjdend'] > table['mjdstart']
    table = table[good]

    log.info("Writing table to cache...")
    table.write(cache_file, serialize_meta=True)

    return table


def get_all_change_times(missions=None, mjdstart=None, mjdstop=None):
    """
    Examples
    --------
    >>> table1 = Table({'mjdstart': np.arange(4), 'mjdend': np.arange(1, 5)})
    >>> table2 = Table({'mjdstart': np.arange(2.5, 5.5), 'mjdend': np.arange(3, 6)})
    >>> table3 = Table({'mjdstart': np.arange(2.5, 5.5), 'mjdend': np.arange(3, 6)})
    >>> table2['mjdstart'][:] = np.nan
    >>> all_times = get_all_change_times(table_names=[table1, table2, table3],
    ...     mjdstart=None, mjdstop=4.1)
    >>> np.allclose(all_times, [1, 2, 2.5, 3, 3.5, 4, 4.1])
    True
    """

    if missions is None or len(missions) == 0:
        missions = list(mission_info.keys())

    if mjdstart is not None:
        change_times = [[mjdstart]]
    else:
        change_times = []

    for mission in missions:
        catalog = mission_info[mission].tablename
        if isinstance(catalog, Table):  # Mainly for testing purposes
            mission_table = catalog
        else:
            mission_table = get_table_from_heasarc(mission)

        alltimes = np.transpose(np.vstack(
            (np.array(mission_table['mjdstart']),
             np.array(mission_table['mjdend']))
            )).flatten()

        good = ~np.isnan(alltimes)
        if mjdstart is not None:
            good = good & (alltimes >= mjdstart)
        if mjdstop is not None:
            good = good & (alltimes <= mjdstop)

        change_times.append(alltimes[good])

    if mjdstop is not None:
        change_times.append([mjdstop])

    change_times = np.unique(np.concatenate(change_times))
    return change_times[change_times > 0]


def sync_all_timelines(mjdstart=None, mjdend=None, missions=None,
                       ignore_cache=False):
    conf = Conf()
    conf.timeout = 600
    heasarc = Heasarc()

    if missions is None or len(missions) == 0:
        missions = list(mission_info.keys())

    # all_times = np.arange(mjdstart, mjdend, 500 / 86400)
    all_times = get_all_change_times(
        missions, mjdstart=mjdstart, mjdstop=mjdend)

    result_table = QTable({'mjd': all_times})

    for mission in missions:
        mission_table = get_table_from_heasarc(mission, ignore_cache=False)

        cols = 'mjdstart,mjdend,ra,dec,obsid,name'.split(',')
        restab = get_rows_from_times(mission_table[cols], all_times)

        restab['skycoords'] = \
            SkyCoord(np.array(restab['ra']),
                     np.array(restab['dec']), unit=('degree', 'degree'))
        for col in cols:
            result_table[f'{mission} {col}'] = restab[col]

        result_table[f'{mission} coords'] = restab['skycoords']

    return result_table


def get_all_separations(table, keyword='coords'):
    """
    Examples
    --------
    >>> a = Table({'ra': [234, 122, 0.0], 'dec': [45, 0, 0.0]})
    >>> b = Table({'ra': [234, 123, 1.], 'dec': [46, 0, 2.]})
    >>> table = QTable()
    >>> table['a coords'] = SkyCoord(a['ra'], a['dec'], unit=('degree', 'degree'))
    >>> table['b coords'] = SkyCoord(b['ra'], b['dec'], unit=('degree', 'degree'))
    >>> table = get_all_separations(table)
    >>> np.allclose(table['dist_a--b'].value[:2], [1, 1])
    True
    >>> np.isnan(table['dist_a--b'].value[2])
    True
    """
    coord_cols = [col for col in table.colnames if keyword in col]
    for i, coord1_col in enumerate(coord_cols):
        for coord2_col in coord_cols[i + 1:]:
            sep = table[coord1_col].separation(table[coord2_col])
            bad_1 = (table[coord1_col].ra.value == 0.0) & (table[coord1_col].dec.value == 0.0)
            bad_2 = (table[coord2_col].ra.value == 0.0) & (table[coord2_col].dec.value == 0.0)
            sep[bad_1 | bad_2] = np.nan * u.deg

            newcolname = \
                (f'dist_{coord1_col.replace(keyword, "").strip()}--'
                 f'{coord2_col.replace(keyword, "").strip()}')

            table[newcolname] = sep

    return table


def filter_for_low_separations(table, max_dist=7 * u.arcmin, keyword='dist'):
    """
    Examples
    --------
    >>> table = QTable({'dist_1--2': [1, 4, 0.003] * u.deg,
    ...                 'dist_2--3': [0.001, np.nan, np.nan] * u.deg,
    ...                 'dsdfasdfas': [0, 1, 3]})
    >>> newtable = filter_for_low_separations(table)
    >>> len(newtable['dist_1--2'])
    2
    >>> newtable['dist_1--2'][0].value
    1.0
    >>> newtable['dist_1--2'][1].value
    0.003
    """
    dist_cols = [col for col in table.colnames if keyword in col]
    mask = np.zeros(len(table), dtype=bool)
    for col in dist_cols:
        good = ~np.isnan(table[col])
        good = good & (table[col] <= max_dist)
        mask = mask | good
    return table[mask]


def main(args=None):
    description = 'List all (quasi-)simultaneous observations between ' \
                  'high-energy missions '
    parser = argparse.ArgumentParser(description=description)

    parser.add_argument("missions",
                        help="Mission tables. Leave "
                             "blank for all supported missions",
                        type=str, nargs='*')

    parser.add_argument("--mjdstart",
                        help="MJD start",
                        default=None, type=float)

    parser.add_argument("--mjdstop",
                        help="MJD stop",
                        default=None, type=float)

    parser.add_argument("--min-length", type=int,
                        help="Minimum length of GTIs to consider",
                        default=0)

    parser.add_argument("--ignore-cache", type=str,
                        help="Ignore cache file",
                        default=None)

    args = parser.parse_args(args)

    mjdlabel = ''
    if args.mjdstart is not None:
        mjdlabel += f'_gt{args.mjdstart}'
    if args.mjdstop is not None:
        mjdlabel += f'_lt{args.mjdstop}'

    missionlabel = '_all'
    if len(args.missions) > 0:
        missionlabel = "_" + '+'.join(args.missions)

    cache_filename = f"_timeline{missionlabel}{mjdlabel}.hdf5"

    log.info("Loading all mission tables...")
    if os.path.exists(cache_filename) and not args.ignore_cache:
        synced_table = QTable.read(cache_filename)
    else:
        synced_table = sync_all_timelines(mjdstart=args.mjdstart,
                                          mjdend=args.mjdstop,
                                          missions=args.missions,
                                          ignore_cache=args.ignore_cache)
        log.info("Calculating separations...")
        synced_table = get_all_separations(synced_table, keyword='coords')
        synced_table.write(cache_filename, overwrite=True,
                           serialize_meta=True)

    cols = [col for col in synced_table.colnames if 'dist_' in col]
    for col in cols:
        mission1, mission2 = col.replace('dist_', '').split('--')
        log.info(f"Searching for matches between {mission1} and {mission2}")
        good = ~np.isnan(synced_table[col])
        good = good&(synced_table[col] <= 30 * u.arcmin)

        res = copy.deepcopy(synced_table[good])
        if len(res) == 0:
            log.warning("No combinations here.")
            continue
        for dcol in res.colnames:
            if dcol == col:
                continue
            elif '--' in dcol:
                res.remove_column(dcol)
            elif (mission1 not in dcol) and (mission2 not in dcol):
                res.remove_column(dcol)

        o1, o2 = f'{mission1} obsid', f'{mission2} obsid'
        res['obsid_pairs'] = [
            f'{obsid1},{obsid2}'
            for obsid1, obsid2 in zip(res[o1], res[o2])]
        res = unique(res, keys=['obsid_pairs'])
        res.remove_column('obsid_pairs')
        res.write(f'{mission1}-{mission2}{mjdlabel}.hdf5', serialize_meta=True)
        res.write(f'{mission1}-{mission2}{mjdlabel}.csv', overwrite=True)
