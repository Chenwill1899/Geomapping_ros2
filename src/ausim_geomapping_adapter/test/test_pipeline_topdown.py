from launch.conditions import LaunchConfigurationEquals

from ausim_geomapping_adapter.pipeline import (
    TRAVERSABILITY_RVIZ_FILE,
    generate_ausim_scout_localmap_launch_description,
)


def _node_executable(entity):
    return getattr(entity, "_Node__node_executable", None)


def _launch_configuration_equals_value(condition):
    expected = getattr(condition, "_LaunchConfigurationEquals__expected_value", [])
    if not expected:
        return None
    return getattr(expected[0], "_TextSubstitution__text", None)


def _launch_argument_default(launch_description, name):
    for entity in launch_description.entities:
        if getattr(entity, "_DeclareLaunchArgument__name", None) != name:
            continue
        default_value = getattr(entity, "_DeclareLaunchArgument__default_value", [])
        if len(default_value) != 1:
            return None
        return getattr(default_value[0], "_TextSubstitution__text", None)
    return None


def _parameter_names(node):
    names = set()
    for parameter in getattr(node, "_Node__parameters", []):
        if not isinstance(parameter, dict):
            continue
        for key in parameter:
            if isinstance(key, tuple) and key:
                names.add(getattr(key[0], "_TextSubstitution__text", None))
    return names


def test_topdown_launch_keeps_traversability_map_for_existing_visualization():
    launch_description = generate_ausim_scout_localmap_launch_description(
        default_height_source="mujoco_topdown"
    )

    topdown_maps = [
        entity
        for entity in launch_description.entities
        if _node_executable(entity) == "traversability_map"
        and isinstance(entity.condition, LaunchConfigurationEquals)
        and _launch_configuration_equals_value(entity.condition) == "mujoco_topdown"
    ]

    assert len(topdown_maps) == 1
    assert "mapping.external_heightmap_topic" in _parameter_names(topdown_maps[0])


def test_topdown_launch_keeps_original_rviz_config_by_default():
    launch_description = generate_ausim_scout_localmap_launch_description(
        default_height_source="mujoco_topdown"
    )

    assert _launch_argument_default(launch_description, "rviz_config_file") == TRAVERSABILITY_RVIZ_FILE


def test_lidar_projection_nodes_only_run_for_lidar_height_source():
    launch_description = generate_ausim_scout_localmap_launch_description(
        default_height_source="mujoco_topdown"
    )

    lidar_executables = {"terrain_pub_node", "traversability_filter"}
    conditions = {
        _node_executable(entity): _launch_configuration_equals_value(entity.condition)
        for entity in launch_description.entities
        if _node_executable(entity) in lidar_executables
    }

    assert conditions == {
        "terrain_pub_node": "lidar",
        "traversability_filter": "lidar",
    }
