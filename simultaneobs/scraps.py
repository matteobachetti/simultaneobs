from collections.abc import Iterable
import numpy as np
from astropy import log
from astropy.coordinates import SkyCoord
from astroquery.simbad import Simbad
from astroquery.heasarc import Heasarc  # , Conf
import astropy.units as u
# from astropy.table import Table


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


def get_precise_position(source):
    heasarc = Heasarc()

    try:
        result_table = heasarc.query_object(
            source, mission='atnfpulsar', fields=f'RA,DEC')
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


# def cross_two_tables(coords, table1_name='nustar', table2_name='nustar'):
#     conf = Conf()
#     conf.timeout = 600
#     heasarc = Heasarc()
#
#     time_col1 = mission_info[table1_name].time
#     time_col2 = mission_info[table2_name].time
#
#     tables = find_source_in_catalogs(coords, [table1_name, table2_name])
#     table_mission1, table_mission2 = tables[table1_name], tables[table2_name]
#
#     time_tolerance = 7
#
#     max_dist = 1 * u.deg
#     mission1_coords = convert_coords_to_3D_cartesian(
#         table_mission1['RA'], table_mission1['DEC'])
#     mission2_coords = convert_coords_to_3D_cartesian(
#         table_mission2['RA'], table_mission2['DEC'])
#     source_coords = convert_coords_to_3D_cartesian(coords.ra, coords.dec)
#
#     mission1_time = table_mission1[time_col1]
#
#     all_matches = Table(names=[f'{table1_name} TIME', f'{table1_name} TARGET',
#                                f'{table1_name} OBSID', f'{table1_name} distance (´)',
#                                f'{table2_name} TIME', f'{table2_name} TARGET',
#                                f'{table2_name} OBSID', f'{table2_name} distance (´)'],
#                         dtype=[float,   'U11',     'U11',    float,
#                                float,   'U11',     'U11',    float])
#
#     for i_n, t in enumerate(mission1_time):
#         mission1_row = table_mission1[i_n]
#         idx = select_close_observations(
#             t, table_mission2[time_col2], time_tolerance=time_tolerance)
#
#         distance_1 = distance(mission1_coords[i_n], source_coords) * u.rad
#         distances_2 = distance(mission2_coords[idx], source_coords) * u.rad
#
#         for i, dist in zip(idx, distances_2):
#             if dist > max_dist:
#                 continue
#             all_matches.add_row(
#                 (mission1_row[time_col1], mission1_row['NAME'],
#                  mission1_row['OBSID'], distance_1.to(u.arcminute).value,
#                  table_mission2[time_col2][i], table_mission2['NAME'][i],
#                  table_mission2['OBSID'][i],
#                  dist.to(u.arcminute).value))
#
#     return all_matches


# def find_source_in_catalogs(coords, catalog_list):
#     heasarc = Heasarc()
#
#     tables = {}
#     for catalog in catalog_list:
#         time_col1 = mission_info[catalog].time
#         log.info(f"Querying online catalog {catalog}... ")
#         table_mission = heasarc.query_region(
#             coords, mission=catalog, radius='1 degree', sortvar=time_col1,
#             fields=f'{time_col1},RA,DEC,NAME,OBSID',
#             resultmax=1000000, timeout=600)
#         table_mission['OBSID'] = [str(obsid) for obsid in table_mission['OBSID']]
#         tables[catalog] = table_mission
#     log.info('Done')
#     return tables
#
#
# def pulsar_cross_match():
#     targets = ['Crab', 'PSR J1824-2452A', 'PSR J1939+2134', 'HER X-1']
#     missions = ['nicermastr', 'numaster', 'swiftmastr', 'xmmmaster', 'chanmaster']
#
#     for target in targets:
#         print(f"\n\nTarget: {target}\n")
#         pos = get_precise_position(target)
#
#         for i, mission1 in enumerate(missions):
#             for mission2 in missions[i+1:]:
#                 # try:
#                 matches = cross_two_tables(pos, mission1, mission2)
#                 if len(matches) > 0:
#                     print(matches)
#                 else:
#                     print("No matches found")
#
