import json
import click

from osm_ai_tools import query_osm


@click.command()
@click.option('--query-config', help='JSON config file', required=True, type=str)
@click.option('--output-csv', help='Path to output object location CSV', required=True, type=str)
def cli(query_config, output_csv):
    with open(query_config, "rt") as f:
        conf = json.load(f)
    query_configs = [query_osm.ObjectQuery(**kwargs) for kwargs in conf]
    df = query_osm.query_objects(query_configs).drop_duplicates(subset=["osm_id"])
    print(f"{df.shape[0]} usable results")
    df.to_csv(output_csv, index=False)
