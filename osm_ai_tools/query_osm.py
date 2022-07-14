"""
Use OSM's Overpass API to query for objects of interest
"""
from collections import namedtuple
from typing import List

import overpy
import pandas as pd

ObjectQuery = namedtuple("ObjectQuery", ["tag_key", "tag_value", "object_class"])


def get_geometry_info(nodes: List[overpy.Node]):
    lats = []
    lons = []
    for n in nodes:
        lats.append(n.lat)
        lons.append(n.lon)
    return {
        "min_lat": min(lats),
        "max_lat": max(lats),
        "min_lon": min(lons),
        "max_lon": max(lons),
        "center_lat": (min(lats) + max(lats)) / 2,
        "center_lon": (min(lons) + max(lons)) / 2,
    }


def query_objects(query_list: List[ObjectQuery] = None, include_tags: bool = False):
    api = overpy.Overpass()

    out = []
    for query in query_list:
        # fetch centre coordinates of all ways, nodes and relations
        q = f"""
            way ["{query.tag_key}"="{query.tag_value}"];
            (._;>;);
            out;
            """
        result = api.query(q)
        df = []
        for w in result.ways:
            out_dict = dict(osm_id=w.id, **get_geometry_info(w.get_nodes()))
            if include_tags:
                out_dict = {**out_dict, **w.tags}
            df.append(out_dict)
        df = pd.DataFrame(df)
        df["object_class"] = query.object_class  # map to our object names
        out.append(df)

    return pd.concat(out)
