import click
import pandas as pd
from sklearn.cluster import AgglomerativeClustering


def get_span(grp):
    return max(
        abs(grp["center_lat"].max() - grp["center_lat"].min()),
        abs(grp["center_lon"].max() - grp["center_lon"].min()),
    )


def cluster_objects(df, max_distance):
    agg_clust = AgglomerativeClustering(
        n_clusters=None, distance_threshold=max_distance
    )  # euclidean distance in lat/lon
    df["cluster_id"] = agg_clust.fit_predict(df[["center_lat", "center_lon"]])
    cluster_span = df.groupby("cluster_id").apply(get_span).sort_values()
    cluster_size = df.groupby("cluster_id").size()
    lat = df.groupby("cluster_id")["center_lat"].mean()
    lon = df.groupby("cluster_id")["center_lon"].mean()
    cluster_df = pd.DataFrame(
        {
            "cluster_span": cluster_span,
            "cluster_size": cluster_size,
            "center_lat": lat,
            "center_lon": lon,
        }
    )
    return df, cluster_df


def run_clustering(input_objects, output_clusters, output_objects, max_distance):
    df = pd.read_csv(input_objects)
    df_with_cluster_id, cluster_df = cluster_objects(df, max_distance)
    df_with_cluster_id.to_csv(output_objects, index=False, float_format="%.5f")
    cluster_df.to_csv(output_clusters, float_format="%.5f")


@click.command()
@click.option(
    "--input-objects", help="CSV of object locations and IDs", required=True, type=str
)
@click.option(
    "--output-clusters", help="Path to output cluster CSV", required=True, type=str
)
@click.option(
    "--output-objects", help="Path to output object CSV", required=True, type=str
)
@click.option(
    "--max-distance", help="Max distance in lat/lon space", default=0.008, type=float
)
def cli(input_objects, output_clusters, output_objects, max_distance):
    run_clustering(input_objects, output_clusters, output_objects, max_distance)
