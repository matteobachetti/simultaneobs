import time
import os
import argparse
import copy
import socket
from collections.abc import Iterable

socket.setdefaulttimeout(600)  # set timeout to 10 minutes

import numpy as np
from astropy import log
from astropy.coordinates import SkyCoord
from astropy.table import Table, QTable, Column, unique
import astropy.units as u
from astroquery.heasarc import Heasarc, Conf


def convert_coords_to_3D_cartesian(ra, dec):
    """
    Examples
    --------
    >>> # It works with arrays
    >>> pos = convert_coords_to_3D_cartesian([0, 90, 0], [0, 0, 90])
    >>> np.allclose(pos, [[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    True
    >>> # It works at other specific positions.
    >>> pos = convert_coords_to_3D_cartesian(42, 90)
    >>> np.allclose(pos, [0, 0, 1])
    True
    >>> pos = convert_coords_to_3D_cartesian(45, 0)
    >>> np.allclose(pos, [np.sqrt(2)/2, np.sqrt(2)/2, 0])
    True
    >>> pos = convert_coords_to_3D_cartesian(0, 45)
    >>> np.allclose(pos, [np.sqrt(2)/2, 0, np.sqrt(2)/2])
    True
    """
    if isinstance(ra, Iterable):
        ra = np.asarray(ra)
        dec = np.asarray(dec)
    ra = ra * np.pi / 180.
    dec = dec * np.pi / 180.
    pos = np.transpose(
        np.vstack([np.cos(ra) * np.cos(dec),
                   np.sin(ra) * np.cos(dec),
                   np.sin(dec)]))
    return pos


def distance(array, position):
    """
    Examples
    --------
    >>> np.allclose(distance([[0, 1, 2], [0, 1, 3], [0, -1, 2]], [0, 1, 2]), [0, 1, 2])
    True
    """
    return np.sqrt(np.sum((np.asarray(array) - np.asarray(position))**2, axis=1))


def select_close_observations(t, time_array, time_tolerance=7):
    # print(t, time_array)
    best_time_idx = np.searchsorted(time_array, t)
    if best_time_idx >= time_array.size:
        return []

    if best_time_idx <= 0:
        return []

    if np.abs(time_array[best_time_idx] - t) > time_tolerance:
        return []

    all_idx = [best_time_idx]
    idx = best_time_idx - 1

    while idx >= 0:
        if np.abs(time_array[idx] - t) > time_tolerance:
            break
        all_idx.append(idx)
        idx -= 1

    idx = best_time_idx + 1
    while idx < time_array.size:
        if np.abs(time_array[idx] - t) > time_tolerance:
            break
        all_idx.append(idx)
        idx += 1
    return sorted(all_idx)


def set_default_mission_dict():
    from collections import defaultdict
    default = dict(time="time", end_time='end_time', obsid='obsid', ra='ra', dec='dec', name='name')

    chandra_dict = copy.deepcopy(default)
    hitomi_dict = copy.deepcopy(default)
    integral_dict = copy.deepcopy(default)  # needs work
    nicer_dict = copy.deepcopy(default)
    nustar_dict = copy.deepcopy(default)
    suzaku_dict = copy.deepcopy(default)
    swift_dict = copy.deepcopy(default)
    xmm_dict = copy.deepcopy(default)
    xte_dict = copy.deepcopy(default)

    swift_dict['time'] = 'start_time'
    swift_dict['end_time'] = 'stop_time'
    hitomi_dict['end_time'] = 'stop_time'
    suzaku_dict['end_time'] = 'stop_time'
    xte_dict['name'] = 'target_name'
    xte_dict['end_time'] = None
    chandra_dict['end_time'] = None

    mission_time_dict = {'numaster': nustar_dict,
                         'chanmaster': chandra_dict,
                         'nicermastr': nicer_dict,
                         'xmmmaster': xmm_dict,
                         'hitomaster': hitomi_dict,
                         'suzamaster': suzaku_dict,
                         'swiftmastr': swift_dict,
                         'xtemaster': xte_dict
                         }

    return mission_time_dict


mission_time_dict = set_default_mission_dict()


def find_source_in_catalogs(coords, catalog_list):
    heasarc = Heasarc()

    tables = {}
    for catalog in catalog_list:
        time_col1 = mission_time_dict[catalog]['time']
        log.info(f"Querying online catalog {catalog}... ")
        table_mission = heasarc.query_region(
            coords, mission=catalog, radius='1 degree', sortvar=time_col1,
            fields=f'{time_col1},RA,DEC,NAME,OBSID',
            resultmax=1000000, timeout=600)
        table_mission['OBSID'] = [str(obsid) for obsid in table_mission['OBSID']]
        tables[catalog] = table_mission
    log.info('Done')
    return tables


def cross_two_tables(coords, table1_name='numaster', table2_name='swiftmastr'):
    conf = Conf()
    conf.timeout = 600
    heasarc = Heasarc()

    time_col1 = mission_time_dict[table1_name]['time']
    time_col2 = mission_time_dict[table2_name]['time']

    tables = find_source_in_catalogs(coords, [table1_name, table2_name])
    table_mission1, table_mission2 = tables[table1_name], tables[table2_name]
    time_tolerance = 7

    max_dist = 1 * u.deg
    mission1_coords = convert_coords_to_3D_cartesian(
        table_mission1['RA'], table_mission1['DEC'])
    mission2_coords = convert_coords_to_3D_cartesian(
        table_mission2['RA'], table_mission2['DEC'])
    source_coords = convert_coords_to_3D_cartesian(coords.ra, coords.dec)

    mission1_time = table_mission1[time_col1]

    all_matches = Table(names=[f'{table1_name} TIME', f'{table1_name} TARGET',
                               f'{table1_name} OBSID', f'{table1_name} distance (´)',
                               f'{table2_name} TIME', f'{table2_name} TARGET',
                               f'{table2_name} OBSID', f'{table2_name} distance (´)'],
                        dtype=[float,   'U11',     'U11',    float,
                               float,   'U11',     'U11',    float])

    for i_n, t in enumerate(mission1_time):
        mission1_row = table_mission1[i_n]
        idx = select_close_observations(
            t, table_mission2[time_col2], time_tolerance=time_tolerance)

        distance_1 = distance(mission1_coords[i_n], source_coords) * u.rad
        distances_2 = distance(mission2_coords[idx], source_coords) * u.rad

        for i, dist in zip(idx, distances_2):
            if dist > max_dist:
                continue
            all_matches.add_row(
                (mission1_row[time_col1], mission1_row['NAME'], mission1_row['OBSID'], distance_1.to(u.arcminute).value,
                 table_mission2[time_col2][i], table_mission2['NAME'][i], table_mission2['OBSID'][i],
                 dist.to(u.arcminute).value))

    return all_matches



def get_precise_position(source):
    from astroquery.simbad import Simbad
    heasarc = Heasarc()

    try:
        result_table = heasarc.query_object(source, mission='atnfpulsar',
                fields=f'RA,DEC')
        pos = result_table[0]
        ra, dec = pos['RA'] * u.deg, pos['DEC'] * u.deg
        coords = SkyCoord(ra, dec, frame='icrs')
    except (ValueError, TypeError):
        log.warning("Not found in ATNF; searching Simbad")
        result_table = Simbad.query_object(source)
        pos = result_table[0]
        ra, dec = pos['RA'], pos['DEC']
        coords = SkyCoord(ra, dec, frame='icrs', unit=(u.hourangle, u.deg))
    log.info(f"Precise position: {ra}, {dec}")

    return coords


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
    import copy
    # Avoid invalid values
    mission_table = mission_table[mission_table['mjdstart'] > 0]
    start, end = mission_table['mjdstart'], mission_table['mjdend']
    idxs = np.searchsorted(start, times + 1 / 86400)

    # print(start, end, times)
    result_table = QTable()
    # example_tab = copy.deepcopy(mission_table[:1])
    good = (times >= start[0])&(times <=end[-1])
    places_to_change = mission_table[idxs[good] - 1]
    for col in mission_table.colnames:
        newarr = np.zeros(times.size, dtype=mission_table[col].dtype)
        newarr[good] = places_to_change[col]
        result_table[col] = newarr

    return result_table


def get_table_from_heasarc(table,
        max_entries=10000000, use_cache=True):
    import pyvo as vo
    cache_file = f'_{table}_table_cache.hdf5'

    if use_cache and os.path.exists(cache_file):
        log.info(f"Getting cached table {cache_file}...")
        table = Table.read(cache_file)
        log.info("Done")
        return table

    heasarc_tap = vo.dal.TAPService(
        "https://heasarc.gsfc.nasa.gov/xamin/vo/tap/")

    settings = mission_time_dict[table]
    colnames = (f"{settings['time']},"
                f"{settings['ra']},{settings['dec']},"
                f"{settings['name']},{settings['obsid']}")

    if settings['end_time'] is not None:
        colnames += f",{settings['end_time']}"

    query = f"""SELECT
    TOP {max_entries}
    "__row", {colnames}
    FROM {table}
    """

    table = heasarc_tap.search(query).to_table()

    for key in ['obsid', 'name']:
        values = [f"{value}" for value in table[settings[key]]]
        table.remove_column(settings[key])
        table[key] = values

    for key in ['__row']:
        values = [float(value) for value in table[key]]
        table.remove_column(key)
        table[key] = values

    table.rename_column(settings['time'], 'mjdstart')

    table.sort('mjdstart')

    if settings['end_time'] is None:
        table['mjdend'] = \
            np.concatenate((table['mjdstart'][1:], table['mjdstart'][-1:] + 1))
    else:
        table.rename_column(settings['end_time'], 'mjdend')

    good = table['mjdend'] > table['mjdstart']
    table = table[good]

    if use_cache:
        log.info("Writing table to cache...")
        table.write(cache_file, serialize_meta=True)

    return table


def get_all_change_times(table_names=None, mjdstart=None, mjdstop=None):
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

    if table_names is None:
        table_names = list(mission_time_dict.keys())

    if mjdstart is not None:
        change_times = [[mjdstart]]
    else:
        change_times = []

    for catalog in table_names:
        if isinstance(catalog, Table):  # Mainly for testing purposes
            mission_table = catalog
            time_col = 'mjdstart'
            end_time_col = 'mjdend'
        else:
            mission_table = get_table_from_heasarc(catalog)

        alltimes = np.transpose(np.vstack(
            (np.array(mission_table['mjdstart']), np.array(mission_table['mjdend']))
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
    # change_times = change_times[~np.isnan(change_times)]
    return change_times[change_times > 0]


def sync_all_timelines(mjdstart=None, mjdend=None, table_names=None):
    conf = Conf()
    conf.timeout = 600
    heasarc = Heasarc()

    if table_names is None:
        table_names = list(mission_time_dict.keys())

    # all_times = np.arange(mjdstart, mjdend, 500 / 86400)
    all_times = get_all_change_times(
        table_names, mjdstart=mjdstart, mjdstop=mjdend)

    result_table = QTable({'mjd': all_times})

    time_cols_names = [mission_time_dict[name]['time'] for name in table_names]

    all_tables = {}
    for catalog in table_names:
        print(catalog)
        time_col = mission_time_dict[catalog]['time']
        end_time_col = mission_time_dict[catalog]['end_time']

        mission_table = get_table_from_heasarc(catalog)

        cols = 'mjdstart,mjdend,ra,dec,obsid,name'.split(',')
        restab = get_rows_from_times(mission_table[cols], all_times)

        restab['skycoords'] = \
            SkyCoord(np.array(restab['ra']),
                     np.array(restab['dec']), unit=('degree', 'degree'))
        for col in cols:
            result_table[f'{catalog} {col}'] = restab[col]

        result_table[f'{catalog} coords'] = restab['skycoords']

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
        mask = mask | (table[col] <= max_dist)
    return table[mask]


def pulsar_cross_match():
    targets = ['Crab', 'PSR J1824-2452A', 'PSR J1939+2134', 'HER X-1']
    missions = ['nicermastr', 'numaster', 'swiftmastr', 'xmmmaster', 'chanmaster']

    for target in targets:
        print(f"\n\nTarget: {target}\n")
        pos = get_precise_position(target)

        for i, mission1 in enumerate(missions):
            for mission2 in missions[i+1:]:
                # try:
                matches = cross_two_tables(pos, mission1, mission2)
                if len(matches) > 0:
                    print(matches)
                else:
                    print("No matches found")


def main(args=None):
    description = 'List all (quasi-)simultaneous observations between ' \
                  'high-energy missions '
    parser = argparse.ArgumentParser(description=description)

    parser.add_argument("missions",
                        help="Mission tables. Leave "
                             "blank for all supported missions",
                        type=str, nargs='+', default=None)

    parser.add_argument("--mjdstart",
                        help="MJD start",
                        default=None, type=float)

    parser.add_argument("--mjdstop",
                        help="MJD stop",
                        default=None, type=float)

    parser.add_argument("--min-length", type=int,
                        help="Minimum length of GTIs to consider",
                        default=0)

    args = parser.parse_args(args)

    log.info("Loading all mission tables...")
    if os.path.exists("full_timeline.hdf5"):
        synced_table = QTable.read("full_timeline.hdf5")
    else:
        synced_table = sync_all_timelines(mjdstart=args.mjdstart,
                                          mjdend=args.mjdstop,
                                          table_names=args.missions)
        synced_table.write("rough_timeline.ecsv", overwrite=True)
        log.info("Calculating separations...")
        synced_table = get_all_separations(synced_table, keyword='coords')
        synced_table.write("full_timeline.hdf5", overwrite=True)

    cols = [col for col in synced_table.colnames if 'dist_' in col]
    for col in cols:
        mission1, mission2 = col.replace('dist_', '').split('--')
        good = ~np.isnan(synced_table[col])
        good = good&(synced_table[col] <= 30 * u.arcmin)

        # print(np.min(synced_table[col][good]))
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
        res.write(f'{mission1}-{mission2}.hdf5')
        res.write(f'{mission1}-{mission2}.csv')

