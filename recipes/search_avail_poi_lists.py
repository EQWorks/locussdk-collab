from pandas import DataFrame
from locussdk import get_avail_poi_lists


def search_avail_poi_lists(name: str = '', **kwargs) -> DataFrame:
    '''Search available POI lists based on locussdk.get_avail_poi_lists().

    Args:
        name (str): name to search in the available POI lists.
        **kwargs:
            All locussdk.get_avail_poi_lists() supported arguments.
            All pandas.Series.str.contains() supported arguments.

    Returns:
        pandas.DataFrame that contain the search resulting POI lists.

    Examples:
    >>> search_avail_poi_lists('pizza')
                name  poi_list_id whitelabelid customerid
    67       Bostonpizza           71         None       None
    113       Pizzapizza          118         None       None
    116       Pizzaville          121         None       None
    117     Pizzadelight          122         None       None
    139        Pizzanova          144         None       None
    157         Pizzahut          162         None       None
    175  Neworleanspizza          180         None       None
    327         241Pizza          332         None       None
    422       Royalpizza          427         None       None
    450       Ginospizza          455         None       None
    730     Mammas Pizza          735         None       None
    786      Doublepizza          791         None       None
    '''
    # build parameters we want to pass into get_avail_poi_lists()
    params = {'list_type': 'global'}  # default search in global
    for k in ['list_type', 'whitelabel', 'customer']:
        if v := kwargs.pop(k, None):
            params[k] = v

    lists = get_avail_poi_lists(**params)

    # return the full available POI lists if no search string given
    if not name:
        return lists

    return lists[lists['name'].str.contains(name, **kwargs)]
