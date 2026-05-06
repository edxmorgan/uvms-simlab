cmake_minimum_required(VERSION 3.10)
project(simlab)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(std_msgs REQUIRED)

set(action_files
  "action/PlanVehicle.action"
)

set(msg_files
  "msg/ControllerPerformance.msg"
  "msg/ReferenceTargets.msg"
)

set(srv_files
  "srv/BackendRobotCommand.srv"
  "srv/BackendWorldCommand.srv"
  "srv/BackendPoseCommand.srv"
  "srv/BackendWaypointCommand.srv"
)

rosidl_generate_interfaces(${PROJECT_NAME}
  ${action_files}
  ${msg_files}
  ${srv_files}
  DEPENDENCIES geometry_msgs std_msgs
)

ament_get_python_install_dir(PYTHON_INSTALL_DIR)
ament_python_install_module(simlab/__init__.py DESTINATION_SUFFIX simlab)
install(
  DIRECTORY simlab/
  DESTINATION ${PYTHON_INSTALL_DIR}/simlab
  PATTERN "*.pyc" EXCLUDE
  PATTERN "__pycache__" EXCLUDE
)

function(simlab_python_entrypoint script_name module_name)
  set(script_path "${CMAKE_CURRENT_BINARY_DIR}/scripts/${script_name}")
  file(GENERATE
    OUTPUT "${script_path}"
    CONTENT "#!/usr/bin/env python3\nfrom ${module_name} import main\n\nif __name__ == '__main__':\n    main()\n"
  )
  install(PROGRAMS "${script_path}" DESTINATION lib/${PROJECT_NAME})
endfunction()

simlab_python_entrypoint(interactive_controller simlab.interactive_control)
simlab_python_entrypoint(joystick_controller simlab.joystick_control)
simlab_python_entrypoint(direct_thruster_controller simlab.direct_thruster_control)
simlab_python_entrypoint(mocap_publisher simlab.use_mocap)
simlab_python_entrypoint(motive_publisher simlab.sim_motive)
simlab_python_entrypoint(collision_contact_node simlab.collision_contact)
simlab_python_entrypoint(voxelviz_node simlab.voxel_viz)
simlab_python_entrypoint(bag_recorder_node simlab.bag_recorder)
simlab_python_entrypoint(mcap_to_replay_profile simlab.mcap_to_replay_profile)
simlab_python_entrypoint(env_obstacles_node simlab.env_obstacles)
simlab_python_entrypoint(planner_action_server_node simlab.planner_action_server)
simlab_python_entrypoint(camera_viewer simlab.opencv_camera_viewer)
simlab_python_entrypoint(plot_replay_session simlab.replay_session_plotter)
simlab_python_entrypoint(replay_plot_node simlab.replay_plot_node)

install(
  DIRECTORY
    resource/model_functions
    resource/whole_body
    resource/playback_profile
    resource/dynamics_profiles
    resource/world_profiles
  DESTINATION share/${PROJECT_NAME}
)

install(FILES package.xml DESTINATION share/${PROJECT_NAME})

if(BUILD_TESTING)
  find_package(ament_cmake_pytest REQUIRED)
  ament_add_pytest_test(test_world_profiles test/test_world_profiles.py)
  ament_add_pytest_test(test_planner_world test/test_planner_world.py)
  ament_add_pytest_test(test_dynamic_replanner test/test_dynamic_replanner.py)
endif()

ament_export_dependencies(rosidl_default_runtime geometry_msgs std_msgs)
ament_package()
