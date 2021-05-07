from locussdk import get_avail_poi_lists


def search_avail_poi_lists(name, **kwargs):
    global_lists = get_avail_poi_lists(list_type='global')

    return global_lists[global_lists['name'].str.contains(name, **kwargs)]
