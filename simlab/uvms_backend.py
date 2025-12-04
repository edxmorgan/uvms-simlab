# uvms_backend.py

from rclpy.duration import Duration
import rclpy
from rclpy.node import Node
from tf2_ros import TransformException, Buffer
from interactive_utils import is_point_valid, get_relative_pose, generate_rov_ellipsoid, compute_bounding_sphere_radius, visualize_min_max_coords
import numpy as np
from robot import Robot
from rclpy.qos import QoSProfile, QoSHistoryPolicy, QoSDurabilityPolicy, QoSReliabilityPolicy
from scipy.spatial.transform import Rotation as R
from geometry_msgs.msg import Pose
from scipy.spatial import ConvexHull
import os
import ament_index_python
from alpha_reach import Params as alpha
from fcl_checker import FCLWorld
from visualization_msgs.msg import Marker
from typing import List
from planner_markers import PathPlanner
from cartesian_ruckig import CartesianRuckig
from frame_utils import PoseX
from se3_ompl_planner import plan_se3_path

class UVMSBackend:
    def __init__(self, node: Node, tf_buffer: Buffer, urdf_string: str):
        package_share_directory = ament_index_python.get_package_share_directory('simlab')
        self.node = node
        self.tf_buffer = tf_buffer
        # get some parameters
        self.robots_prefix = self.node.get_parameter('robots_prefix').value
        self.record = self.node.get_parameter('record_data').value
        self.controllers = self.node.get_parameter('controllers').value
        self.base_frame = "base_link"
        self.world_frame = "world"
        self.world_bottom_frame = "world_bottom"
        self.bottom_z = None
        self.viz_frequency = 10.0       # Hz
        self.fcl_update_frequency = 50.0  # Hz
        self.control_frequency = 500.0  # Hz
        # Define a threshold error at which we start yaw blending.
        self.pos_blend_threshold = 1.1
        # Ruckig Cartesian trajectory generators, one per robot
        self.cartesian_dt = 1.0 / self.control_frequency
        self.max_cartesian_waypoints = 500
        # Simple conservative limits, tune these
        self.max_vel = np.array([0.25, 0.25, 0.20], dtype=float)
        self.max_acc = np.array([0.15, 0.15, 0.12], dtype=float)
        self.max_jerk = np.array([0.5, 0.5, 0.4], dtype=float)

        self.fcl_world = FCLWorld(urdf_string=urdf_string, world_frame=self.world_frame, vehicle_radius=0.4)

        self.node.get_logger().info(f"Minimum coordinates (min_x, min_y, min_z): {self.fcl_world.min_coords}")
        self.node.get_logger().info(f"Maximum coordinates (max_x, max_y, max_z): {self.fcl_world.max_coords}")
        self.node.get_logger().info(f"Oriented Bounding Box corners: {self.fcl_world.obb_corners}")


        viz_qos = QoSProfile(
                history=QoSHistoryPolicy.KEEP_LAST,
                depth=1,
                durability=QoSDurabilityPolicy.VOLATILE,
                reliability=QoSReliabilityPolicy.RELIABLE,
            )
        self.planner_marker_publisher = self.node.create_publisher(Marker, "planned_waypoints_marker", viz_qos)

        self.robots:List[Robot] = []
        for k, (prefix, controller) in enumerate(zip(self.robots_prefix, self.controllers)):
            robot_k = Robot(self.node, k, 4, prefix, self.record, controller)
            robot_k.cart_traj = CartesianRuckig(
                self.node,
                dofs=3,
                control_dt=self.cartesian_dt,
                max_waypoints=self.max_cartesian_waypoints,
            )

            # unique planner per robot
            robot_k.planner = PathPlanner(self.planner_marker_publisher, ns=f"planner/{prefix}", base_id=k)
            self.robots.append(robot_k)

        self.robot_selected = self.robots[0]
        self.setup_initial_robot_configuration()

        # Load workspace point cloud and hull
        workspace_pts_path = os.path.join(package_share_directory, 'manipulator/workspace.npy')
        self.workspace_pts = np.load(workspace_pts_path)
        self.workspace_hull = ConvexHull(self.workspace_pts)

        # ROV ellipsoid point cloud and hull
        self.rov_ellipsoid_cl_pts = generate_rov_ellipsoid(a=0.3, b=0.3, c=0.2, num_points=10000)
        self.vehicle_body_hull = ConvexHull(self.rov_ellipsoid_cl_pts)

        # stack clouds that represent the vehicle occupied volume
        all_pts = np.vstack([
            np.asarray(self.rov_ellipsoid_cl_pts, dtype=float),
            np.asarray(self.workspace_pts, dtype=float)
        ])

        planner_radius = compute_bounding_sphere_radius(all_pts, quantile=0.995, pad=0.03)
        self.node.get_logger().info(f"Planner robot approximation sphere radius set to {planner_radius:.3f} m")
        self.fcl_world.set_planner_radius(planner_radius)

        viz_qos = QoSProfile(
                history=QoSHistoryPolicy.KEEP_LAST,
                depth=1,
                durability=QoSDurabilityPolicy.VOLATILE,
                reliability=QoSReliabilityPolicy.RELIABLE,
            )
        
        # publisher for FCL environment AABB
        self.env_aabb_pub = self.node.create_publisher(Marker, "fcl_environment_aabb", viz_qos)

        # Timer that will try to initialize bottom_z until it succeeds
        self.bottom_z_timer = self.node.create_timer(0.1, self.init_bottom_z_once)
        self.viz_timer = self.node.create_timer(1.0 / self.viz_frequency, self.viz_timer_callback)
        self.fcl_update_timer_handle = self.node.create_timer(1.0 / self.fcl_update_frequency, self.fcl_update_timer)
        self.control_timer = self.node.create_timer(1.0 / self.control_frequency, self.control_timer_callback)

    def set_robot_selected(self, robot):
        self.robot_selected = robot

    def control_timer_callback(self):
        for k, robot in enumerate(self.robots):
            k_planner = robot.planner

            state = robot.get_state()
            if state['status'] == 'active':
                if robot.final_goal is not None and k_planner.planned_result and k_planner.planned_result['is_success']:
                    # Convert once to NumPy arrays
                    path_xyz = np.asarray(k_planner.planned_result["xyz"], dtype=float)
                    path_quat = np.asarray(k_planner.planned_result["quat_wxyz"], dtype=float)

                    # Compute current manifold errors
                    wp_err_trans, wp_err_rot, wp_err_joint, goal_err_trans, goal_err_rot = robot.compute_errors()
                    goal_xyz_error = np.linalg.norm(goal_err_trans)

                    # Calculate the blend factor.
                    # When pos_error >= pos_blend_threshold, blend_factor will be 0 (full velocity_yaw).
                    # When pos_error == 0, blend_factor will be 1 (full target_yaw).
                    robot.yaw_blend_factor = np.clip((self.pos_blend_threshold - goal_xyz_error) / self.pos_blend_threshold, 0.0, 1.0)
                    # self.get_logger().info(
                    #     f"{robot.yaw_blend_factor} yaw_blend_factor"
                    # )
                    # Get the velocity-based yaw.
                    adjusted_yaw = robot.orient_towards_velocity()

                    pos_nwu, vel_nwu, acc_nwu, res = robot.cart_traj.update(robot.yaw_blend_factor)

                    if pos_nwu is not None:
                        target_nwu = np.asarray(pos_nwu, dtype=float)

                        # Pick orientation from nearest OMPL waypoint
                        dists = np.linalg.norm(path_xyz - target_nwu, axis=1)
                        idx = int(np.argmin(dists))
                        target_quat = path_quat[idx]

                        # Convert target pose from NWU to NED
                        target_pose = PoseX.from_pose(
                            xyz=target_nwu,
                            rot=target_quat,
                            rot_rep="quat_wxyz",
                            frame="NWU",
                        )
                        p_cmd_ned, rpy_cmd_ned = target_pose.get_pose(
                            frame="NED",
                            rot_rep="euler_xyz",
                        )


                        # Blend the yaw values: more weight to target_yaw as the position error decreases.
                        rpy_cmd_ned[2] = (1 - robot.yaw_blend_factor) * adjusted_yaw + robot.yaw_blend_factor * rpy_cmd_ned[2]

                        robot.pose_command = [
                            float(p_cmd_ned[0]),
                            float(p_cmd_ned[1]),
                            float(p_cmd_ned[2]),
                            float(rpy_cmd_ned[0]),
                            float(rpy_cmd_ned[1]),
                            float(rpy_cmd_ned[2]),
                        ]


                    robot.arm.q_command = [self.q0_des, self.q1_des, self.q2_des, self.q3_des]                        

            veh_state_vec = np.array(
                list(state['pose']) + list(state['body_vel']),
                dtype=float
            )
            # log to terminal
            # self.get_logger().info(f"robot command = {robot.pose_command}")

            cmd_body_wrench = robot.ll_controllers.vehicle_controller(
                state=veh_state_vec,
                target=np.array(robot.pose_command, dtype=float),
                dt=state["dt"]
            )

            # cmd_body_wrench = np.zeros(6)
            # cmd_body_wrench = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 2.0])
            # Arm PID
            cmd_arm_tau = robot.ll_controllers.arm_controller(
                q=state["q"],
                q_dot=state["dq"],
                q_ref=robot.arm.q_command,
                Kp=alpha.Kp,
                Ki=alpha.Ki,
                Kd=alpha.Kd,
                dt=state["dt"],
                u_max=alpha.u_max,
                u_min=alpha.u_min,
                model_param=alpha.sim_p,
            )

            arm_tau_list = list(np.asarray(cmd_arm_tau, dtype=float).reshape(-1))
            # always produce 5 values, slice if longer, pad if shorter
            arm_tau_list = arm_tau_list[:5] + [0.0]

            robot.publish_commands(cmd_body_wrench, arm_tau_list)

            ref=robot.pose_command+robot.arm.q_command
            robot.write_data_to_file(ref)

    def plan_vehicle_trajectory(self):
        env_xyz_bounds = self.fcl_world._compute_env_bounds_from_fcl(z_min=self.bottom_z, pad_xy=0.0, pad_z=0.0)
        robot = self.robot_selected
        state = robot.get_state()

        # Build a pose from the current NED state, then query NWU for planning
        pose_now = PoseX.from_pose(
            xyz=np.array(state['pose'][0:3], float),
            rot=np.array(state['pose'][3:6], float),   # roll, pitch, yaw
            rot_rep="euler_xyz",
            frame="NED",
        )
        start_xyz, start_quat_wxyz = pose_now.get_pose(frame="NWU", rot_rep="quat_wxyz")

        gx = self.current_target_vehicle_marker_pose.position.x
        gy = self.current_target_vehicle_marker_pose.position.y
        gz = self.current_target_vehicle_marker_pose.position.z
        goal_xyz = np.array([gx, gy, gz], float)

        goal_quat_wxyz = np.array([
            self.current_target_vehicle_marker_pose.orientation.w,
            self.current_target_vehicle_marker_pose.orientation.x,
            self.current_target_vehicle_marker_pose.orientation.y,
            self.current_target_vehicle_marker_pose.orientation.z,
        ], float)


        goal_now = PoseX.from_pose(
            xyz=np.array(goal_xyz, float),
            rot=np.array(goal_quat_wxyz, float),   # roll, pitch, yaw
            rot_rep="quat_wxyz",
            frame="NWU",
        )

        # Goal from the UV marker is in NWU, convert that to NED for control and save it.
        robot.final_goal = goal_now.get_pose(frame="NED", rot_rep="euler_xyz")

        k_planner = robot.planner
        try:

            # self.get_logger().info(
            #     f"start_xyz start_xyz start_xyz {start_xyz}"
            # )
            k_planner.planned_result = plan_se3_path(
                self.node,
                start_xyz=start_xyz,
                start_quat_wxyz=start_quat_wxyz,
                goal_xyz=goal_xyz,
                goal_quat_wxyz=goal_quat_wxyz,
                time_limit=1.0,
                safety_margin=1e-2,
                env_bounds = env_xyz_bounds
            )

            self.node.get_logger().info(
                f"{k_planner.planned_result['message']}"
            )

            if k_planner.planned_result["is_success"]:
                path_xyz = np.asarray(
                    k_planner.planned_result["xyz"],
                    dtype=float,
                )

                # Start a smooth Cartesian trajectory in NWU
                robot.cart_traj.start_from_path(
                    current_position=start_xyz,
                    path_xyz=path_xyz,
                    max_vel=self.max_vel,
                    max_acc=self.max_acc,
                    max_jerk=self.max_jerk,
                )

                self.node.get_logger().info(
                    f"{robot.prefix} started Ruckig trajectory with "
                    f"{path_xyz.shape[0]} waypoints"
                )

        except Exception as e:
            self.node.get_logger().error(f"Planner failed, {e}")
            k_planner.planned_result = {
                        "is_success":False,
                        "message":"Planner did not find a solution"
                    }
        return k_planner.planned_result
                        
    def viz_timer_callback(self):
        stamp_now = self.node.get_clock().now().to_msg()
        min_coords = self.fcl_world.min_coords
        max_coords = self.fcl_world.max_coords
        min_marker, max_marker = visualize_min_max_coords(min_coords, max_coords, self.bottom_z, self.world_frame)
        min_marker.header.stamp = stamp_now
        max_marker.header.stamp = stamp_now
        self.env_aabb_pub.publish(min_marker)
        self.env_aabb_pub.publish(max_marker)

        if self.robot_selected:
            k_planner = self.robot_selected.planner

            if k_planner.planned_result and k_planner.planned_result['is_success']:
                k_planner.update(
                    stamp=stamp_now,
                    frame_id=self.base_frame,
                    xyz_np=k_planner.planned_result["xyz"],
                    step=3,
                    wp_size=0.08,
                    goal_size=0.14,
                )
            state = self.robot_selected.get_state()
            if state['status'] == 'active':
                self.robot_selected.publish_robot_path()


    def fcl_update_timer(self):
        self.fcl_world.update_from_tf(self.tf_buffer, rclpy.time.Time())

    def solve_whole_body_inverse_kinematics_wrt_world_frame(self, task_pose):
        pass

    def solve_inverse_kinematics_wrt_vehicle_frame(self, task_pose):
        msg = {'is_success':False,'result':None}
        task_point = np.array([task_pose.position.x,
                    task_pose.position.y,
                    task_pose.position.z])
        
        if is_point_valid(self.workspace_hull, self.vehicle_body_hull, task_point):
            relative_pose = get_relative_pose(self.arm_base_pose, task_pose)

            q_ik_sol = self.robot_selected.uvms_body_inverse_kinematics(
                np.array([relative_pose.position.x, relative_pose.position.y, relative_pose.position.z]))
            
            msg['is_success'] = True
            msg['result'] = q_ik_sol

        return msg
    

    def init_bottom_z_once(self):
        """Timer callback, tries to read world_bottom z once and then cancels itself."""
        try:
            ts = self.tf_buffer.lookup_transform(
                target_frame=self.world_frame,
                source_frame=self.world_bottom_frame,
                time=rclpy.time.Time(),
                timeout=Duration(seconds=0.1),
            )
        except TransformException as ex:
            self.node.get_logger().warn(
                f"Waiting for TF {self.world_frame} <- {self.world_bottom_frame}: {ex}"
            )
            return

        self.bottom_z = ts.transform.translation.z
        self.node.get_logger().info(
            f"Captured bottom_z from TF: {self.bottom_z:.3f}"
        )

        if self.bottom_z_timer is not None:
            self.bottom_z_timer.cancel()
            self.bottom_z_timer = None

    def setup_initial_robot_configuration(self):
        """
        Initialize arm base pose, initial vehicle marker pose,
        desired joint configuration, and the last valid task space pose.
        """
        # Arm base pose in world frame from alpha.base_T0_new
        self.arm_base_pose = Pose()
        self.arm_base_pose.position.x = float(alpha.base_T0_new[0])
        self.arm_base_pose.position.y = float(alpha.base_T0_new[1])
        self.arm_base_pose.position.z = float(alpha.base_T0_new[2])

        base_rpy = alpha.base_T0_new[3:6]
        base_rot = R.from_euler("xyz", base_rpy)
        qx, qy, qz, qw = base_rot.as_quat()
        self.arm_base_pose.orientation.x = float(qx)
        self.arm_base_pose.orientation.y = float(qy)
        self.arm_base_pose.orientation.z = float(qz)
        self.arm_base_pose.orientation.w = float(qw)

        # Vehicle marker starts at origin with identity orientation
        self.current_target_vehicle_marker_pose = Pose()
        self.current_target_vehicle_marker_pose.position.x = 0.0
        self.current_target_vehicle_marker_pose.position.y = 0.0
        self.current_target_vehicle_marker_pose.position.z = 0.0
        self.current_target_vehicle_marker_pose.orientation.x = 0.0
        self.current_target_vehicle_marker_pose.orientation.y = 0.0
        self.current_target_vehicle_marker_pose.orientation.z = 0.0
        self.current_target_vehicle_marker_pose.orientation.w = 1.0

        # Extract roll, pitch, yaw from the current target vehicle orientation
        desired_q_orientation = [
            self.current_target_vehicle_marker_pose.orientation.x,
            self.current_target_vehicle_marker_pose.orientation.y,
            self.current_target_vehicle_marker_pose.orientation.z,
            self.current_target_vehicle_marker_pose.orientation.w,
        ]
        roll, pitch, yaw = R.from_quat(desired_q_orientation).as_euler("xyz", degrees=False)

        # Initial desired joint configuration taken from robot 0
        self.q0_des, self.q1_des, self.q2_des, self.q3_des = self.robots[0].arm.q_command

        # Initial vehicle pose in world frame [x, y, z, roll, pitch, yaw]
        initial_world_pose = np.array(
            [
                self.current_target_vehicle_marker_pose.position.x,
                self.current_target_vehicle_marker_pose.position.y,
                self.current_target_vehicle_marker_pose.position.z,
                roll,
                pitch,
                yaw,
            ],
            dtype=float,
        )

        # Forward kinematics to get nominal end effector pose
        joints_and_endeffector_poses = Robot.uvms_Forward_kinematics(
            alpha.joint_home,
            alpha.base_T0_new,
            initial_world_pose,
        )

        ee = np.array(joints_and_endeffector_poses[-1].full(), dtype=float).ravel()

        # Store last valid task pose as a geometry_msgs/Pose
        self.current_target_task_pose = Pose()
        self.current_target_task_pose.position.x = float(ee[0])
        self.current_target_task_pose.position.y = float(ee[1])
        self.current_target_task_pose.position.z = float(ee[2])

        # ee[3:7] assumes [w, x, y, z]
        self.current_target_task_pose.orientation.w = float(ee[3])
        self.current_target_task_pose.orientation.x = float(ee[4])
        self.current_target_task_pose.orientation.y = float(ee[5])
        self.current_target_task_pose.orientation.z = float(ee[6])
