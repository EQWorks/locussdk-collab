from locussdk import get_avail_poi_lists


def search_avail_poi_lists(name, **kwargs):
    # build parameters we want to pass into get_avail_poi_lists()
    params = {'list_type': 'global'}  # default search in global
    for k in ['list_type', 'whitelabel', 'customer']:
        if v := kwargs.pop(k, None):
            params[k] = v

    lists = get_avail_poi_lists(**params)

    return lists[lists['name'].str.contains(name, **kwargs)]
