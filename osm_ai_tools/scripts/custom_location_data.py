import pandas as pd


def main(input_csv, id_col, lat_col, lon_col, object_size, object_class, output_csv):
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
