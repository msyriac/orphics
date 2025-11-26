from datetime import datetime
from zoneinfo import ZoneInfo
from timezonefinder import TimezoneFinder
from pixell.coordinates import default_site
from pixell import ephem
import numpy as np

def htime(ctime,lat=None,lng=None,el=None):
    """
    Convert a UNIX timestamp (ctime) to a human-readable local time string for a given location.

    This function determines the local time zone from the provided latitude and longitude,
    then converts the given UNIX timestamp (seconds since epoch, UTC) into the corresponding
    local date and time at that geographic location.

    Parameters
    ----------
    ctime : float or int
        UNIX timestamp (seconds since January 1, 1970, UTC).
    lat : float, optional
        Latitude in decimal degrees. If not provided, defaults to `default_site.lat`.
    lng : float, optional
        Longitude in decimal degrees. If not provided, defaults to `default_site.lon`.

    Returns
    -------
    str
        Local date and time as a string in the format ``"YYYY-MM-DD HH:MM:SS"``.

    Notes
    -----
    - Time zone is automatically determined from the coordinates using the `timezonefinder` library.
    - Handles daylight saving time (DST) automatically, if applicable.
    - Useful for converting timestamps to local civil time for observational sites or sensors.

    Examples
    --------
    >>> htime(1728140400, lat=-22.9585, lng=-67.7876)
    '2024-10-05 08:00:00'
    """
    
    # Find local timezone from coordinates
    tf = TimezoneFinder()
    tz_name = tf.timezone_at(lat=default_site.lat if lat is None else lat, lng=default_site.lon  if lng is None else lng)
    # Convert timestamp to local time
    local_time = datetime.fromtimestamp(ctime, ZoneInfo(tz_name))
    return local_time.strftime("%Y-%m-%d %H:%M:%S")

def ctime(timestr, lat=None, lng=None):
    """
    Convert a human-readable local date/time string to a UNIX timestamp (ctime) for a given location.

    This function performs the inverse of `htime()`. It takes a local civil time string and geographic
    coordinates, determines the corresponding local time zone, and computes the equivalent UNIX timestamp
    (seconds since the UTC epoch, 1970-01-01 00:00:00).

    Parameters
    ----------
    timestr : str
        Local time as a string in the format ``"YYYY-MM-DD HH:MM:SS"``.
    lat : float, optional
        Latitude in decimal degrees. If not provided, defaults to `default_site.lat`.
    lng : float, optional
        Longitude in decimal degrees. If not provided, defaults to `default_site.lon`.

    Returns
    -------
    float
        UNIX timestamp corresponding to the given local date and time (in seconds since epoch, UTC).

    Notes
    -----
    - The local time zone is determined automatically from coordinates using the `timezonefinder` library.
    - Daylight saving time (DST) transitions are handled correctly.
    - Input `timestr` must represent a valid local time for that location.
    - This function is the inverse of `htime()` when given matching coordinates.

    Examples
    --------
    >>> ctime_from_htime("2024-10-05 08:00:00", lat=-22.9585, lng=-67.7876)
    1728140400.0

    >>> # Round-trip consistency
    >>> htime(ctime_from_htime("2024-10-05 08:00:00", -22.9585, -67.7876), -22.9585, -67.7876)
    '2024-10-05 08:00:00'
    """
    tf = TimezoneFinder()
    tz_name = tf.timezone_at(
        lat=default_site.lat if lat is None else lat,
        lng=default_site.lon if lng is None else lng,
    )
    tz = ZoneInfo(tz_name)

    # Parse the time string as local time
    local_dt = datetime.strptime(timestr, "%Y-%m-%d %H:%M:%S").replace(tzinfo=tz)

    # Convert to UTC and then get the timestamp (seconds since epoch)
    return local_dt.timestamp()



def get_columns(obs, keys):
    """
    Extract multiple keys from a list of row dicts and return them
    as NumPy arrays in a dictionary.

    Parameters
    ----------
    obs : iterable of dict-like rows
    keys : list of str
        Keys to extract from each row.

    Returns
    -------
    dict : {key: np.ndarray}
    """
    # Make a list of tuples, one tuple per row
    rows = [tuple(ob[k] for k in keys) for ob in obs]
    # Transpose using zip(*) to group by column
    columns = list(zip(*rows))
    # Convert to NumPy arrays
    return {key: np.array(col) for key, col in zip(keys, columns)}

def object_path_annotation(ctime1,ctime2,objs=['Sun', 'Moon'],line_width = 3,circle_width  = 3):

    ccolor = {'Sun':'orange',
              'Moon':'black'}
    cradius  = {'Sun': 30,
                'Moon': 10}
    
    annotations = []

    for obj in objs:
        radec, dist = ephem.eval(obj, [ctime1,ctime2])
        radec = np.rad2deg(radec)
        dec1_deg = radec[0,1]
        dec2_deg = radec[1,1]
        ra1_deg = radec[0,0]
        ra2_deg = radec[1,0]
        circle_radius = cradius[obj]      # pixels

        circle_color  = ccolor[obj]

        annotations.append(
            ["circle", dec1_deg, ra1_deg, 0, 0, circle_radius, circle_width, circle_color]
        )

        annotations.append(
            ["circle", dec2_deg, ra2_deg, 0, 0, circle_radius, circle_width, circle_color]
        )


        line_color = ccolor[obj]

        annotations.append(
            ["line",
             dec1_deg, ra1_deg, 0, 0,     # start lat/lon
             dec2_deg, ra2_deg, 0, 0,     # end lat/lon
             line_width, line_color]
        )
    return annotations
