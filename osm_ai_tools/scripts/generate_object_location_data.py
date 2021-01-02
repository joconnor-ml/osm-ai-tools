from osm_ai_tools import query_osm

if __name__ == "__main__":
    df = query_osm.query_objects([
        query_osm.ObjectQuery(tag_key="man_made", tag_value="cooling_tower", object_class="cooling_tower"),
        query_osm.ObjectQuery(tag_key="tower:type", tag_value="cooling", object_class="cooling_tower"),
    ]).drop_duplicates(subset=["osm_id"])
    print(f"{df.shape[0]} usable results")
    df.to_csv("data/object_location_data.csv", index=False)
