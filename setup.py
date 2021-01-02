from setuptools import setup, find_packages

setup(
    name='osm_ai_tools',
    version='0.0',
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'Click',
    ],
    entry_points='''
        [console_scripts]
        download_images=osm_ai_tools.scripts.download_images:cli
        generate_bboxes=osm_ai_tools.scripts.generate_bboxes:cli
        generate_object_location_data=osm_ai_tools.scripts.generate_object_location_data:cli
        cluster_objects=osm_ai_tools.scripts.cluster_objects:cli
    ''',
)
