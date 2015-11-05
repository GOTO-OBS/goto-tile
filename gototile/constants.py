GOTO4 = {
    'fov-dec': 4.2,
    'fov-ra': 4.2,
    'lat': 28.7598742,
    'lon': 360.0-17.8793802,
    'height': 2396
}
GOTO8 = {
    'fov-dec': 4.2,
    'fov-ra': 8.2,
    'lat': 28.7598742,
    'lon': 360.0-17.8793802,
    'height': 2396
}
SWASPN = {
    'fov-dec': 30,
    'fov-ra': 15,
    'lat': 28.7598742,
    'lon': 360.0-17.8793802,
    'height': 2396
}
TEMP__SCOPE = {
    'fov-dec': 0,
    'fov-ra': 0,
    'lat': 0,
    'lon': 0,
    'height': 0,
}

def set_temp_scope(defaults, site=None, fov=None):
    global TEMP__SCOPE
    TEMP__SCOPE['fov-dec'] = 2 * defaults[0]
    TEMP__SCOPE['fov-ra'] = 2 * defaults[1]
    TEMP__SCOPE['lat'] = defaults[2]
    TEMP__SCOPE['lon'] = defaults[3]
    TEMP__SCOPE['height'] = defaults[4]
    if site:
        TEMP__SCOPE['lat'] = site.latitude.value
        TEMP__SCOPE['lon'] = site.longitude.value
        TEMP__SCOPE['height'] = site.height.value
    if fov:
        TEMP__SCOPE['fov-dec'] = fov[0]
        TEMP__SCOPE['fov-ra'] = fov[1]
