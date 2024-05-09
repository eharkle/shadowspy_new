import numpy as np

def sph2cart(r, lat, lon):
    """
    Transform spherical (meters, degrees) to cartesian (meters)
    Args:
        r: meters
        lat: degrees
        lon: degrees

    Returns:
    Cartesian xyz (meters)
    """
    x = r * cosd(lon) * cosd(lat)
    y = r * sind(lon) * cosd(lat)
    z = r * sind(lat)

    return x, y, z

# transform cartesian to spherical (meters, radians)
def cart2sph(xyz):
    # print("cart2sph in",np.array(xyz))

    rtmp = np.linalg.norm(np.array(xyz).reshape(-1, 3), axis=1)
    lattmp = np.arcsin(np.array(xyz).reshape(-1, 3)[:, 2] / rtmp)
    lontmp = np.arctan2(np.array(xyz).reshape(-1, 3)[:, 1], np.array(xyz).reshape(-1, 3)[:, 0])

    return rtmp, lattmp, lontmp


def sind(x):
    return np.sin(np.deg2rad(x))


def cosd(x):
    return np.cos(np.deg2rad(x))


def unproject_stereographic(x, y, lon0, lat0, R):
    """
    Stereographic Coordinates unprojection
    Args:
        x: stereo coord
        y: stereo coord
        lon0: center of the projection (longitude, deg)
        lat0: center of the projection (latitude, deg)
        R: planet radius

    Returns:
    Longitude and latitude (deg) of points in cylindrical coordinates
    """
    rho = np.sqrt(np.power(x, 2) + np.power(y, 2))
    c = 2 * np.arctan2(rho, 2 * R)

    lat = np.rad2deg(np.arcsin(np.cos(c) * sind(lat0) + (cosd(lat0) * y * np.sin(c)) / rho))
    lon = np.mod(
        lon0 + np.rad2deg(np.arctan2(x * np.sin(c), cosd(lat0) * rho * np.cos(c) - sind(lat0) * y * np.sin(c))), 360)

    lat = np.where(x**2+y**2 == 0, lat0, lat)
    lon = np.where(x**2+y**2 == 0, lon0, lon)

    # if (x == 0).any() and (y == 0).any():
    #     print("coming here")
    #     #    if x == 0 and y == 0:
    #     return lon0, lat0
    # else:
    return lon, lat

def project_stereographic(lon, lat, lon0, lat0, R=1):
    """
    project cylindrical coordinates to stereographic xy from central lon0/lat0
    :param lon: array of input longitudes (deg)
    :param lat: array of input latitudes (deg)
    :param lon0: center longitude for the projection (deg)
    :param lat0: center latitude for the projection (deg)
    :param R: planetary radius (km)
    :return: stereographic projection xy coord from center (km)
    """

    cosd_lat = cosd(lat)
    cosd_lon_lon0 = cosd(lon - lon0)
    sind_lat = sind(lat)

    k = (2. * R) / (1. + sind(lat0) * sind_lat + cosd(lat0) * cosd_lat * cosd_lon_lon0)
    x = k * cosd_lat * sind(lon - lon0)
    y = k * (cosd(lat0) * sind_lat - sind(lat0) * cosd_lat * cosd_lon_lon0)

    return x, y


def compute_azimuth_elevation(observer_lat, observer_lon, observer_alt, target_pos):
    """
    Computes the azimuth and elevation angles from a given observer's location on Earth to a target position in ECEF coordinates.

    :param observer_lat: Latitude of the observer in degrees.
    :param observer_lon: Longitude of the observer in degrees.
    :param observer_alt: Altitude of the observer in meters.
    :param target_pos: Target position in ECEF coordinates (array-like of x, y, z).
    :return: Tuple of azimuth and elevation angles in degrees.

    Example:
    # >>> target_pos_input = [1.41432411e+08, -5.58812792e+07, -1.11768748e+06]
    # >>> observer_lon = 29
    # >>> observer_lat = -85.38
    # >>> observer_alt = 0
    # >>> azimuth_deg, elevation_deg = compute_azimuth_elevation(observer_lat, observer_lon, observer_alt, target_pos_input)
    # >>> print(azimuth_deg, elevation_deg)
    """

    # Convert latitude and longitude to radians
    lat_rad = np.radians(observer_lat)
    lon_rad = np.radians(observer_lon)

    # Observer's local frame transformation matrix (ECEF to ENU)
    trans_matrix = np.array([
        [-np.sin(lon_rad), np.cos(lon_rad), 0],
        [-np.sin(lat_rad) * np.cos(lon_rad), -np.sin(lat_rad) * np.sin(lon_rad), np.cos(lat_rad)],
        [np.cos(lat_rad) * np.cos(lon_rad), np.cos(lat_rad) * np.sin(lon_rad), np.sin(lat_rad)]
    ])

    # Convert target position from ECEF to ENU coordinates
    local_vector = np.dot(trans_matrix, target_pos - np.array([0, 0, observer_alt]))

    # Compute azimuth and elevation
    east = local_vector[0]
    north = local_vector[1]
    up = local_vector[2]
    azimuth_rad = np.arctan2(east, north)
    elevation_rad = np.arcsin(up / np.linalg.norm(local_vector))

    # Convert from radians to degrees
    azimuth_deg = np.degrees(azimuth_rad)
    elevation_deg = np.degrees(elevation_rad)

    return azimuth_deg, elevation_deg


def azimuth_elevation_to_cartesian(azimuth_deg, elevation_deg, distance, observer_lat, observer_lon, observer_alt):
    """
    Converts azimuth and elevation angles back to Cartesian coordinates in the ECEF system.

    :param azimuth_deg: Azimuth angle in degrees.
    :param elevation_deg: Elevation angle in degrees.
    :param distance: Distance from the observer to the target in meters.
    :param observer_lat: Latitude of the observer in degrees.
    :param observer_lon: Longitude of the observer in degrees.
    :param observer_alt: Altitude of the observer in meters.
    :return: Target position in ECEF coordinates (array-like of x, y, z).

    Example:
    # >>> azimuth_deg = 120.5
    # >>> elevation_deg = 45.0
    # >>> distance = 100000
    # >>> observer_lat = -85.38
    # >>> observer_lon = 29
    # >>> observer_alt = 0
    # >>> target_pos_output = azimuth_elevation_to_cartesian(azimuth_deg, elevation_deg, distance, observer_lat, observer_lon, observer_alt)
    # >>> print(target_pos_output)
    """

    # Convert azimuth and elevation to radians
    azimuth_rad = np.radians(azimuth_deg)
    elevation_rad = np.radians(elevation_deg)

    # Convert observer latitude and longitude from degrees to radians
    observer_lat_rad = np.radians(observer_lat)
    observer_lon_rad = np.radians(observer_lon)

    # Convert azimuth and elevation to Cartesian coordinates in ENU system
    x_enu = distance * np.cos(elevation_rad) * np.sin(azimuth_rad)
    y_enu = distance * np.cos(elevation_rad) * np.cos(azimuth_rad)
    z_enu = distance * np.sin(elevation_rad)

    # ENU to ECEF transformation matrix
    trans_matrix = np.array([
        [-np.sin(observer_lon_rad), np.cos(observer_lon_rad), 0],
        [-np.cos(observer_lon_rad) * np.sin(observer_lat_rad), -np.sin(observer_lon_rad) * np.sin(observer_lat_rad),
         np.cos(observer_lat_rad)],
        [np.cos(observer_lon_rad) * np.cos(observer_lat_rad), np.sin(observer_lon_rad) * np.cos(observer_lat_rad),
         np.sin(observer_lat_rad)]
    ])

    # Convert ENU coordinates to ECEF
    target_pos_ecef = np.dot(trans_matrix.T, np.array([x_enu, y_enu, z_enu])) + np.array([0, 0, observer_alt])

    return target_pos_ecef


