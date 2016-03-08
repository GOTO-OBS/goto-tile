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
        TEMP__SCOPE['fov-ra'] = fov[0]
        TEMP__SCOPE['fov-dec'] = fov[1]

def getscopename(scope):
    
    names = {'g4': 'GOTO4',
             'g8': 'GOTO8',
             'swn': 'SuperWASP-N'}
    scopename = names[scope]
    
    return scopename
    
def getscopeinfo(name):
    if name.startswith('temp__'):
        delns, delew, lat, lon, height = (
            TEMP__SCOPE['fov-dec']/2, TEMP__SCOPE['fov-ra']/2,
            TEMP__SCOPE['lat'], TEMP__SCOPE['lon'], TEMP__SCOPE['height'])
    elif name.startswith('SuperWASP'):
        delns, delew = SWASPN['fov-dec']/2, SWASPN['fov-ra']/2
        if name.endswith('N'):
            lat, lon, height = SWASPN['lat'], SWASPN['lon'], SWASPN['height']
        else:
            raise ValueError("unknown SuperWASP configuration")
    elif name.startswith('GOTO'):
        if name.endswith('4'):
            delns, delew = GOTO4['fov-dec']/2, GOTO4['fov-ra']/2
        elif name.endswith('8'):
            delns, delew = GOTO8['fov-dec']/2, GOTO8['fov-ra']/2
        else:
            raise ValueError("unknown GOTO configuration")
        lat, lon, height = GOTO4['lat'], GOTO4['lon'], GOTO4['height']
    else:
        raise ValueError("Unknown telescope")

    return delns, delew, lat, lon, height 
