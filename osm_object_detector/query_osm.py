"""
Use OSM's Overpass API to query for objects of interest
"""
from collections import namedtuple
from typing import List

import overpy
import pandas as pd

ObjectQuery = namedtuple("ObjectQuery", ["tag_key", "tag_value", "object_class"])


def query_objects(query_list: List[ObjectQuery] = None):
    api = overpy.Overpass()

    out = []
    for query in query_list:
        # fetch centre coordinates of all ways, nodes and relations
        q = f"""
            nwr ["{query.tag_key}"="{query.tag_value}"];
            (._;>;);
            out center;
            """
        result = api.query(q)
        df = pd.DataFrame(
            [{"lat": w.center_lat, "lon": w.center_lon} for w in result.ways] +
            [{"lat": w.center_lat, "lon": w.center_lon} for w in result.nodes if w.tags and "center_lat" in vars(w)] +
            [{"lat": w.center_lat, "lon": w.center_lon} for w in result.relations if w.tags and "center_lat" in vars(w)]
        )
        df["object_class"] = query.object_class  # map to our object names
        out.append(df)

    return pd.concat(out)
