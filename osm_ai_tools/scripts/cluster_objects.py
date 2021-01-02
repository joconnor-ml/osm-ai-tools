import pandas as pd
from sklearn.cluster import AgglomerativeClustering


def get_span(grp):
    return max(abs(grp["center_lat"].max() - grp["center_lat"].min()),
               abs(grp["center_lon"].max() - grp["center_lon"].min()))


def cluster_objects(df):
    agg_clust = AgglomerativeClustering(n_clusters=None, distance_threshold=0.008)  # euclidean distance in lat/lon
    df["cluster_id"] = agg_clust.fit_predict(df[["center_lat", "center_lon"]])
    cluster_span = df.groupby("cluster_id").apply(get_span).sort_values()
    cluster_size = df.groupby("cluster_id").size()
    lat = df.groupby("cluster_id")["center_lat"].mean()
    lon = df.groupby("cluster_id")["center_lon"].mean()
    cluster_df = pd.DataFrame({"cluster_span": cluster_span, "cluster_size": cluster_size,
                               "center_lat": lat, "center_lon": lon})
    return df, cluster_df


def run_clustering():
    df = pd.read_csv("data/object_location_data.csv")
    df_with_cluster_id, cluster_df = cluster_objects(df)
    df_with_cluster_id.to_csv("data/object_location_data_clustered.csv", index=False, float_format="%.5f")
    cluster_df.to_csv("data/clusters.csv", float_format="%.5f")


if __name__ == "__main__":
    run_clustering()
