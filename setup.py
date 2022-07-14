from setuptools import setup, find_packages

setup(
    name="osm_ai_tools",
    version="0.0",
    packages=find_packages(),
    include_package_data=True,
    entry_points="""
        [console_scripts]
        download_images=osm_ai_tools.scripts.download_images:main
        generate_bboxes=osm_ai_tools.scripts.generate_bboxes:main
        generate_object_location_data=osm_ai_tools.scripts.generate_object_location_data:main
        cluster_objects=osm_ai_tools.scripts.cluster_objects:main
        generate_tfrecords=osm_ai_tools.scripts.generate_tfrecords:main
        custom_location_data=osm_ai_tools.scripts.custom_location_data:main
        data_pipeline=osm_ai_tools.scripts.data_pipeline:main
    """,
)
