import click
import pandas as pd


@click.command()
@click.option(
    "--input-csv",
    help="Must contain at least a lat lon pair and ID for each object",
    required=True,
    type=str,
)
@click.option("--output-csv", help="Path to output file", required=True, type=str)
@click.option("--id-col", help="id column", required=True, type=str)
@click.option("--lat-col", help="latitude column", default="lat", type=str)
@click.option("--lon-col", help="longitude column", default="lon", type=str)
@click.option(
    "--object-size", "approx object span in lat/lon space", default=0.001, type=float
)
@click.option("--object-class", help="Name of object class", required=True, type=str)
def cli(input_csv, id_col, lat_col, lon_col, object_size, object_class, output_csv):
    df = pd.read_csv(input_csv)
    # renaming to object_id for compatibility
    df["object_id"] = df[id_col]
    df["center_lat"] = df[lat_col]
    df["min_lat"] = df[lat_col] - object_size
    df["max_lat"] = df[lat_col] + object_size
    df["center_lon"] = df[lon_col]
    df["min_lon"] = df[lon_col] - object_size
    df["max_lon"] = df[lon_col] + object_size
    df["object_class"] = object_class
    df[
        [
            "object_id",
            "min_lat",
            "max_lat",
            "min_lon",
            "max_lon",
            "center_lat",
            "center_lon",
            "object_class",
        ]
    ].to_csv(output_csv)
