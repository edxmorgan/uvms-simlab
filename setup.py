from setuptools import find_packages, setup
from glob import glob

package_name = 'simlab'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (
            'share/' + package_name + '/manipulator',
            glob('resource/manipulator/*')
        ),
        (
            'share/' + package_name + '/vehicle',
            glob('resource/vehicle/*')
        ),
        (
            'share/' + package_name + '/whole_body',
            glob('resource/whole_body/*')
        ),
        ('lib/' + package_name, [package_name+'/robot.py']),
        ('lib/' + package_name, [package_name+'/alpha_reach.py']),
        ('lib/' + package_name, [package_name+'/controller_msg.py']),
        ('lib/' + package_name, [package_name+'/controllers.py']),
        ('lib/' + package_name, [package_name+'/mesh_utils.py']),
        ('lib/' + package_name, [package_name+'/se3_ompl_planner.py']),
        ('lib/' + package_name, [package_name+'/cartesian_ruckig.py']),
        ('lib/' + package_name, [package_name+'/fcl_checker.py']),
        ('lib/' + package_name, [package_name+'/interactive_utils.py']),
        ('lib/' + package_name, [package_name+'/planner_markers.py']),
        ('lib/' + package_name, [package_name+'/frame_utils.py']),
        ('lib/' + package_name, [package_name+'/uvms_backend.py']),
        ('lib/' + package_name, [package_name+'/backend_utils.py']),
        ('lib/' + package_name, [package_name+'/debug_targets.py']),
        ('lib/' + package_name, [package_name+'/planner_action_client.py']),
    ],

    install_requires=['setuptools',
                      'rclpy',
                      'std_msgs',
                      'sensor_msgs',
                      'geometry_msgs',
                      'visualization_msgs',
                      'tf2_ros',
                      'numpy',
                      'trimesh',
                      'pycollada',
                      'python-fcl',
                      ],
    zip_safe=True,
    maintainer='mr-robot',
    maintainer_email='edmorgangh@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    entry_points={
        'console_scripts': [
            'interactive_controller = simlab.interactive_control:main',
            'joystick_controller = simlab.joystick_control:main',
            'direct_thruster_controller = simlab.direct_thruster_control:main',
            'rgb2cloudpoint_publisher = simlab.rgb2cloudpoint:main',
            'mocap_publisher = simlab.use_mocap:main',
            'motive_publisher = simlab.sim_motive:main',
            'collision_contact_node = simlab.collision_contact:main',
            'voxelviz_node = simlab.voxel_viz:main',
            'bag_recorder_node = simlab.bag_recorder:main',
            'env_obscles_node = simlab.env_obstacles:main',
            'planner_action_server_node = simlab.planner_action_server:main'
        ],
    },
)
