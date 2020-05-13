import os
import argparse
import copy
import socket
from dataclasses import dataclass

import numpy as np
from astropy import log
from astropy.coordinates import SkyCoord
from astropy.table import Table, QTable, unique, vstack
import astropy.units as u
import pyvo as vo

socket.setdefaulttimeout(600)  # set timeout to 10 minutes


@dataclass
class StandardTableInfo(object):
    tablename: str
    time: str = "time"
    end_time: str = 'end_time'
    obsid: str = 'obsid'
    ra: str = 'ra'
    dec: str = 'dec'
    name: str = 'name'
    mode_entries: list = None


def set_default_mission_info():
    chandra_dict = StandardTableInfo('chanmaster', end_time=None)
    hitomi_dict = StandardTableInfo('hitomaster', end_time='stop_time')
    integral_dict = StandardTableInfo(
        'intscw', name='obs_type',
        time='start_date', end_time='end_date', obsid='scw_id',
        mode_entries=['spi_mode', 'ibis_mode',
                      'jemx1_mode', 'jemx2_mode', 'omc_mode'])
    nicer_dict = StandardTableInfo('nicermastr')
    nustar_dict = StandardTableInfo('numaster')
    suzaku_dict = StandardTableInfo('suzamaster', end_time='stop_time')
    swift_dict = StandardTableInfo('swiftmastr', time='start_time',
                                   end_time='stop_time')

    xmm_dict = StandardTableInfo(
        'xmmmaster', mode_entries=['mos1_mode', 'mos2_mode',
                                   'pn_mode', 'rgs1_mode', 'rgs2_mode'])

    xte_dict = StandardTableInfo('xtemaster',
                                 end_time=None, name='target_name')

    mission_info = {'nustar': nustar_dict,
                    'chandra': chandra_dict,
                    'nicer': nicer_dict,
                    'xmm': xmm_dict,
                    'integral': integral_dict,
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
    good = (times >= start[0]) & (times <= end[-1])
    places_to_change = mission_table[idxs[good] - 1]
    for col in mission_table.colnames:
        newarr = np.zeros(times.size, dtype=mission_table[col].dtype)
        newarr[good] = places_to_change[col]
        result_table[col] = newarr

    return result_table


def get_table_from_heasarc(
        mission, max_entries=10000000, ignore_cache=False):
    settings = mission_info[mission]
    cache_file = f'_{settings.tablename}_table_cache.hdf5'

    if mission == 'integral':
        log.warning(
            "The target name is not available in the INTEGRAL master table")
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

    if settings.mode_entries is not None:
        entries = ','.join(settings.mode_entries)
        colnames += f",{entries}"

    print(colnames)
    query = f"""SELECT
    TOP {max_entries}
    "__row", {colnames}
    FROM {settings.tablename}
    """

    log.info(f"Querying {settings.tablename} table...")
    table = heasarc_tap.search(query).to_table()
    # print(table)
    for key in ['obsid', 'name']:
        col = getattr(settings, key)
        values = [value for value in table[col].iter_str_vals()]
        table.remove_column(col)
        table[key] = values

    if settings.mode_entries is not None:
        for col in settings.mode_entries:
            values = [value for value in table[col].iter_str_vals()]
            table.remove_column(col)
            table[col] = values

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
    table.write(cache_file, serialize_meta=True, overwrite=True)

    return table


def get_all_change_times(missions=None, mjdstart=None, mjdstop=None,
                         ignore_cache=False):
    """
    Examples
    --------
    >>> table1 = Table({'mjdstart': np.arange(4), 'mjdend': np.arange(1, 5)})
    >>> table2 = Table({'mjdstart': np.arange(2.5, 5.5), 'mjdend': np.arange(3, 6)})
    >>> table3 = Table({'mjdstart': np.arange(2.5, 5.5), 'mjdend': np.arange(3, 6)})
    >>> table2['mjdstart'][:] = np.nan
    >>> all_times = get_all_change_times(missions=[table1, table2, table3],
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
        if isinstance(mission, Table):  # Mainly for testing purposes
            mission_table = mission
        else:
            mission_table = \
                get_table_from_heasarc(mission, ignore_cache=ignore_cache)

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


def filter_table_with_obsids(mission_table, obsid_list):
    """
    Examples
    --------
    >>> table = Table({'obsid': ['0101', '0101', '2345', '5656', '9090'],
    ...                'mjd': [57000, 57000, 58000, 59000, 60000]})
    >>> filt = filter_table_with_obsids(table, ['0101', '5656', '5656'])
    >>> np.allclose(filt['mjd'], [57000, 57000, 59000, 59000])
    True
    """
    tables = []
    for obsid in obsid_list:
        if obsid == "":
            log.error(f"Invalid obsid value: {obsid}")
        # print(obsid, type(obsid))
        mask = (mission_table['obsid'] == obsid)
        tables.append(mission_table[mask])

    return vstack(tables)


def sync_all_timelines(mjdstart=None, mjdend=None, missions=None,
                       ignore_cache=False):

    if missions is None or len(missions) == 0:
        missions = list(mission_info.keys())

    all_times = get_all_change_times(
        missions, mjdstart=mjdstart, mjdstop=mjdend, ignore_cache=ignore_cache)

    result_table = QTable({'mjd': all_times})

    for mission in missions:
        mission_table = get_table_from_heasarc(mission)

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

    parser.add_argument("--ignore-cache", action='store_true',
                        help="Ignore cache file",
                        default=False)

    args = parser.parse_args(args)

    mjdlabel = ''
    if args.mjdstart is not None:
        mjdlabel += f'_gt{args.mjdstart:g}'
    if args.mjdstop is not None:
        mjdlabel += f'_lt{args.mjdstop:g}'

    missionlabel = '_all'
    if len(args.missions) > 0:
        missionlabel = "_" + '+'.join(args.missions)

    cache_filename = f"_timeline{missionlabel}{mjdlabel}.hdf5"

    log.info("Loading requested mission tables...")
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
        good = good & (synced_table[col] <= 30 * u.arcmin)

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

        mission_table1 = filter_table_with_obsids(
            get_table_from_heasarc(mission1), res[o1])

        mission_table2 = filter_table_with_obsids(
            get_table_from_heasarc(mission2), res[o2])

        if mission_info[mission1].mode_entries is not None:
            for col in mission_info[mission1].mode_entries:
                res[f'{mission1} {col}'] = mission_table1[col]
        if mission_info[mission2].mode_entries is not None:
            for col in mission_info[mission2].mode_entries:
                res[f'{mission2} {col}'] = mission_table2[col]

        res.write(f'{mission1}-{mission2}{mjdlabel}.hdf5', serialize_meta=True,
                   overwrite=True)
        res.write(f'{mission1}-{mission2}{mjdlabel}.ecsv',
                  overwrite=True)


def split_missions_and_dates(fname):
    """

    Examples
    --------
    >>> fname = 'nustar-nicer_gt55000_lt58000.csv'
    >>> outdict = split_missions_and_dates(fname)
    >>> outdict['mission1']
    'nustar'
    >>> outdict['mission2']
    'nicer'
    >>> outdict['mjdstart']
    'MJD 55000'
    >>> outdict['mjdstop']
    'MJD 58000'
    >>> fname = 'nustar-nicer.csv'
    >>> outdict = split_missions_and_dates(fname)
    >>> outdict['mission1']
    'nustar'
    >>> outdict['mission2']
    'nicer'
    >>> outdict['mjdstart']
    'Mission start'
    >>> outdict['mjdstop']
    'Today'
    """
    no_ext = os.path.splitext(fname)[0]
    split_date = no_ext.split('_')
    mjdstart = 'Mission start'
    mjdstop = 'Today'
    if len(split_date) > 1:
        for date_str in split_date[1:]:
            if 'gt' in date_str:
                mjdstart = 'MJD ' + date_str.replace('gt', '')
            elif 'lt' in date_str:
                mjdstop = 'MJD ' + date_str.replace('lt', '')

    mission1, mission2 = split_date[0].split('-')
    outdict = {'mission1': mission1, 'mission2': mission2,
               'mjdstart': mjdstart, 'mjdstop': mjdstop}
    return outdict


def summary(args=None):
    description = 'Create summary page'
    parser = argparse.ArgumentParser(description=description)

    parser.add_argument("files",
                        help="List of files of kind "
                             "mission1-mission2.{hdf5,csv}",
                        type=str, nargs='+')

    parser.add_argument("-o", '--output', help='Output rst file',
                        default='outpage.rst', type=str)

    args = parser.parse_args(args)

    with open(args.output, 'w') as fobj:
        for fname in args.files:
            outdict = split_missions_and_dates(fname)
            mission1 = outdict['mission1'].capitalize()
            mission2 = outdict['mission2'].capitalize()
            start = outdict['mjdstart']
            stop = outdict['mjdstop']

            title_str = (f'{mission1} - {mission2} matches'
                         f' (between {start} and {stop})')

            print(title_str, file=fobj)
            print('-' * len(title_str) + '\n', file=fobj)

            table = QTable.read(fname)
            cols = [col for col in table.colnames if
                    'obsid' in col or 'mjd' in col or 'name' in col]

            table[cols].write(fobj, format='ascii.rst')

            print(file=fobj)
