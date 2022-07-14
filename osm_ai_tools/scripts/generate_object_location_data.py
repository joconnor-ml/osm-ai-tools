import json

from osm_ai_tools.data import query_osm


def main(query_config, output_csv, include_tags):
    with open(query_config, "rt") as f:
        conf = json.load(f)["osm_tags"]
    query_configs = [query_osm.ObjectQuery(**kwargs) for kwargs in conf]
    df = query_osm.query_objects(query_configs, include_tags).drop_duplicates(
        subset=["object_id"]
    )
    print(f"{df.shape[0]} usable results")
    df.to_csv(output_csv, index=False)
