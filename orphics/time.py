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




# Sidereal orbital periods in seconds (approx)
BODY_PERIOD = {
    "Moon":    27.321661  * 86400.0,
    "Sun":    365.256     * 86400.0,   # apparent solar path (Earth's sidereal year)
    "Mercury": 87.969     * 86400.0,
    "Venus":  224.701     * 86400.0,
    "Mars":   686.980     * 86400.0,
    "Jupiter":4332.589    * 86400.0,
    "Saturn": 10759.22    * 86400.0,
    "Uranus": 30685.4     * 86400.0,
    "Neptune":60189.0     * 86400.0,
}

# Per-body styling (pixels + color); tweak to taste
BODY_STYLE = {
    "Sun":     dict(radius_pix=40, width_pix=4, color="orange"),
    "Moon":    dict(radius_pix=20, width_pix=3, color="black"),
    "Mercury": dict(radius_pix=10, width_pix=2, color="gray"),
    "Venus":   dict(radius_pix=10, width_pix=2, color="blue"),
    "Mars":    dict(radius_pix=10, width_pix=2, color="red"),
    "Jupiter": dict(radius_pix=10, width_pix=2, color="brown"),
    "Saturn":  dict(radius_pix=10, width_pix=2, color="purple"),
    "Uranus":  dict(radius_pix=10, width_pix=2, color="cyan"),
    "Neptune": dict(radius_pix=10, width_pix=2, color="green"),
}


def body_circle_annotations(
    ctime1,
    ctime2,
    bodies=None,
    points_per_orbit=20,
    min_points=3,
    max_points=200,
    default_radius=12,
    default_width=2,
    default_color="white",
    text_size = 18,
):
    """
    Build an enplot 'annotate' list of circles showing positions of
    multiple solar-system bodies between ctime1 and ctime2.

    Sampling density for each body scales with (ctime2-ctime1)/orbital_period.
    """
    if bodies is None:
        bodies = ["Sun", "Moon", "Mercury", "Venus",
                  "Mars", "Jupiter", "Saturn", "Uranus", "Neptune"]

    dt = float(ctime2 - ctime1)
    annotations = []

    for body in bodies:
        period = BODY_PERIOD.get(body, None)
        if period is None:
            # Skip unknown bodies rather than crashing
            continue

        # Choose sampling times for this body
        if dt <= 0:
            ctimes = np.array([ctime1], dtype=float)
        else:
            orbits = dt / period
            n_desired = int(np.ceil(orbits * points_per_orbit))
            n = max(min_points, min(max_points, n_desired))
            ctimes = np.linspace(ctime1, ctime2, n)

        # Evaluate ephemeris
        radec, dist = ephem.eval(body, ctimes)
        ra_deg  = np.degrees(radec[:, 0])
        dec_deg = np.degrees(radec[:, 1])

        # Get style for this body
        style      = BODY_STYLE.get(body, {})
        radius_pix = style.get("radius_pix", default_radius)
        width_pix  = style.get("width_pix",  default_width)
        color      = style.get("color",      default_color)

        # Make one circle per sample
        for ra, dec in zip(ra_deg, dec_deg):
            annotations.append(
                ["circle", float(dec), float(ra), 0, 0,
                 radius_pix, width_pix, color]
            )
        annotations.append(
            ["text", float(dec), float(ra), 0, 0,
             body, text_size, color]
        )

    return annotations
