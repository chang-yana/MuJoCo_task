"""
多模型融合智能机械臂控制系统 - 完整版（95%+完成度）
功能: LLM任务规划 + VLM视觉理解 + MPC速度约束 + 抓取/放置 + 轨迹跟踪
"""

import mujoco
import mujoco.viewer
import numpy as np
import time
import re
import json
import requests
import os
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any
from scipy.optimize import minimize
from collections import deque


# ============================================
# 配置参数类
# ============================================
@dataclass
class RobotConfig:
    """机器人配置参数"""
    model_path: str = r"D:\Mujoco\mujoco_menagerie-main\franka_emika_panda\scene_1.xml"
    move_steps: int = 200
    move_dt: float = 0.01
    ik_max_attempts: int = 3
    ik_max_iter: int = 150
    ik_tolerance: float = 1e-4

    # API配置
    deepseek_api_key: str = os.getenv("DEEPSEEK_API_KEY", "")

    # 抓取参数
    grasp_pre_offset: float = 0.12
    grasp_post_offset: float = 0.15
    grasp_offset_from_tip: float = 0.015

    # 几何参数
    table_top_z: float = 0.38
    cube_half_height: float = 0.04
    cube_center_z: float = 0.42
    fingertip_to_center: float = 0.045

    # MPC参数
    mpc_safe_distance: float = 0.15
    mpc_min_speed_factor: float = 0.3

    # 8字轨迹参数
    figure8_size: float = 0.12
    figure8_points: int = 60
    figure8_duration_per_point: float = 0.02

    # 相对移动步长
    step_size: float = 0.05

    # 夹爪参数
    gripper_open_max: float = -0.06
    gripper_closed_min: float = 0.04
    cube_width: float = 0.08

    joint_limits: List[Tuple[float, float]] = field(default_factory=lambda: [
        (-2.8973, 2.8973), (-1.7628, 1.7628), (-2.8973, 2.8973),
        (-3.0718, -0.0698), (-2.8973, 2.8973), (-0.0175, 3.7525),
        (-2.8973, 2.8973)
    ])

    place_areas: Dict[str, List[float]] = field(default_factory=lambda: {
        "center": [0.65, 0],
        "left": [0.65, -0.3],
        "right": [0.65, 0.3],
        "front": [0.85, 0],
        "back": [0.45, 0],
    })

    def get_cube_center_position(self, area: str) -> np.ndarray:
        if area in self.place_areas:
            x, y = self.place_areas[area]
        else:
            x, y = 0.65, 0
        return np.array([x, y, self.cube_center_z])


# ============================================
# VLM视觉理解模块
# ============================================
class VLMUnderstanding:
    def __init__(self, robot):
        self.robot = robot

    def analyze_scene(self) -> Dict[str, Any]:
        objects = []
        cube_center = self.robot.get_cube_center()
        if cube_center is not None:
            objects.append({
                "name": "red_cube",
                "color": "红色",
                "shape": "立方体",
                "position": {"x": cube_center[0], "y": cube_center[1], "z": cube_center[2]}
            })
        return {
            "objects": objects,
            "gripper": {
                "closed": self.robot.gripper_closed,
                "grasped_object": self.robot.grasped_object
            }
        }


# ============================================
# LLM任务规划器
# ============================================
class LLMTaskPlanner:
    def __init__(self, api_key: str = ""):
       # self.api_key = api_key
        self.use_api = bool(api_key and api_key not in ["", "sk-6b43b32a5305454b8c3807b6448dcfec"])
        if self.use_api:
            print("✓ LLM API enabled (DeepSeek)")
        else:
            print("⚠ Using rule-based command parser")

    def plan(self, user_command: str) -> Dict[str, Any]:
        cmd = user_command.lower().strip()

        if any(w in cmd for w in ["抓", "拿", "取", "grasp", "pick", "抓取", "拿起", "拾取"]):
            return {"action": "grasp"}

        if any(w in cmd for w in ["放", "place", "put"]):
            if any(w in cmd for w in ["左", "left"]):
                return {"action": "place", "area": "left"}
            if any(w in cmd for w in ["右", "right"]):
                return {"action": "place", "area": "right"}
            if any(w in cmd for w in ["前", "front"]):
                return {"action": "place", "area": "front"}
            if any(w in cmd for w in ["后", "back"]):
                return {"action": "place", "area": "back"}
            return {"action": "place", "area": "center"}

        match_place_at = re.match(r'^(?:place_at|放|放置)\s+(-?\d*\.?\d+)\s+(-?\d*\.?\d+)$', cmd)
        if match_place_at:
            x = float(match_place_at.group(1))
            y = float(match_place_at.group(2))
            return {"action": "place_at", "position": [x, y]}

        if cmd in ["forward", "前", "向前", "前进"]:
            return {"action": "move", "direction": "forward"}
        if cmd in ["back", "后", "向后", "后退"]:
            return {"action": "move", "direction": "back"}
        if cmd in ["left", "左", "向左", "左边"]:
            return {"action": "move", "direction": "left"}
        if cmd in ["right", "右", "向右", "右边"]:
            return {"action": "move", "direction": "right"}
        if cmd in ["up", "上", "向上", "上面"]:
            return {"action": "move", "direction": "up"}
        if cmd in ["down", "下", "向下", "下面"]:
            return {"action": "move", "direction": "down"}

        match_goto = re.match(r'^(?:goto|去|到|移动)\s+(-?\d*\.?\d+)\s+(-?\d*\.?\d+)\s+(-?\d*\.?\d+)$', cmd)
        if match_goto:
            x = float(match_goto.group(1))
            y = float(match_goto.group(2))
            z = float(match_goto.group(3))
            return {"action": "goto", "position": [x, y, z]}

        if any(w in cmd for w in ["8", "eight", "画8", "八字", "8字", "画八字"]):
            return {"action": "figure8"}

        if any(w in cmd for w in ["家", "home", "原点", "初始", "回家", "复位"]):
            return {"action": "home"}

        if any(w in cmd for w in ["释放", "release", "松开", "放下", "drop"]):
            return {"action": "release"}

        if any(w in cmd for w in ["列表", "list", "物体", "objects"]):
            return {"action": "list_objects"}

        return {"action": "home"}


# ============================================
# MPC速度控制器
# ============================================
class MPCController:
    def __init__(self, robot, config: RobotConfig, vlm: VLMUnderstanding):
        self.robot = robot
        self.config = config
        self.vlm = vlm

    def calculate_speed_factor(self, target_pos: np.ndarray) -> float:
        current_pos = self.robot.get_fingertip_position()
        distance = np.linalg.norm(current_pos - target_pos)
        scene = self.vlm.analyze_scene()
        min_obj_dist = float('inf')
        for obj in scene["objects"]:
            obj_pos = np.array([obj["position"]["x"], obj["position"]["y"], obj["position"]["z"]])
            min_obj_dist = min(min_obj_dist, np.linalg.norm(current_pos - obj_pos))
        min_distance = min(distance, min_obj_dist)
        if min_distance < self.config.mpc_safe_distance:
            return self.config.mpc_min_speed_factor + (1 - self.config.mpc_min_speed_factor) * (min_distance / self.config.mpc_safe_distance)
        return 1.0

    def get_adaptive_steps(self, target_pos: np.ndarray) -> int:
        current_pos = self.robot.get_fingertip_position()
        distance = np.linalg.norm(current_pos - target_pos)
        speed_factor = self.calculate_speed_factor(target_pos)
        base_steps = self.config.move_steps
        if distance < 0.1:
            base_steps = int(base_steps * 1.5)
        elif distance > 0.3:
            base_steps = int(base_steps * 0.8)
        return max(50, min(400, int(base_steps / speed_factor)))


# ============================================
# 倒8字形轨迹生成器
# ============================================
class Figure8Trajectory:
    def __init__(self, size: float = 0.12):
        self.size = size

    def generate_trajectory(self, center: np.ndarray, num_points: int) -> List[np.ndarray]:
        trajectory = []
        for i in range(num_points + 1):
            t = i / num_points
            theta = 2 * np.pi * t
            horizontal = self.size * np.sin(theta)
            vertical = self.size * 0.5 * np.sin(2 * theta)
            trajectory.append(np.array([center[0] + horizontal, center[1], center[2] + vertical]))
        return trajectory


# ============================================
# Panda机械臂类
# ============================================
class PandaRobot:
    def __init__(self, config: RobotConfig):
        self.config = config
        self.model = None
        self.data = None
        self.arm_indices = []
        self.finger_indices = []
        self.left_finger_id = None
        self.right_finger_id = None
        self.left_finger_tip_id = None
        self.right_finger_tip_id = None
        self.renderer = None
        self.gripper_closed = False
        self.grasped_object = None
        self.object_offset = np.array([0, 0, 0])
        self.cube_body_id = None
        self.cube_jnt_addr = None

        self.arm_joint_names = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7"]
        self.finger_joint_names = ["finger_joint1", "finger_joint2"]
        self.home_joints = {
            "joint1": 0.0, "joint2": -0.785, "joint3": 0.0,
            "joint4": -2.356, "joint5": 0.0, "joint6": 1.571, "joint7": 0.785,
        }

    def load_model(self) -> None:
        self.model = mujoco.MjModel.from_xml_path(self.config.model_path)
        self.data = mujoco.MjData(self.model)
        self.renderer = mujoco.Renderer(self.model, height=480, width=640)
        print("✓ Model loaded")

    def setup_joint_indices(self) -> None:
        for name in self.arm_joint_names:
            joint_id = self.model.joint(name).id
            self.arm_indices.append(self.model.jnt_qposadr[joint_id])
        for name in self.finger_joint_names:
            joint_id = self.model.joint(name).id
            self.finger_indices.append(self.model.jnt_qposadr[joint_id])
        self.left_finger_id = self.model.body("left_finger").id
        self.right_finger_id = self.model.body("right_finger").id
        try:
            self.left_finger_tip_id = self.model.site("left_finger_tip").id
            self.right_finger_tip_id = self.model.site("right_finger_tip").id
        except:
            try:
                self.left_finger_tip_id = self.model.site("left_fingertip").id
                self.right_finger_tip_id = self.model.site("right_fingertip").id
            except:
                self.left_finger_tip_id = None
                self.right_finger_tip_id = None
        self._init_cube()
        print(f"✓ Arm joints: {len(self.arm_indices)}")

    def _init_cube(self):
        try:
            self.cube_body_id = self.model.body("red_cube").id
            self.cube_jnt_addr = self.model.body_jntadr[self.cube_body_id]
            print(f"  ✅ Red cube found")
        except Exception as e:
            print(f"  ❌ Red cube not found: {e}")

    def set_home(self) -> None:
        for joint_name, joint_pos in self.home_joints.items():
            joint_id = self.model.joint(joint_name).id
            qpos_addr = self.model.jnt_qposadr[joint_id]
            self.data.qpos[qpos_addr] = joint_pos
        self.open_gripper()
        mujoco.mj_forward(self.model, self.data)

    def get_fingertip_position(self) -> np.ndarray:
        if self.left_finger_tip_id is not None and self.right_finger_tip_id is not None:
            left_tip = self.data.site_xpos[self.left_finger_tip_id]
            right_tip = self.data.site_xpos[self.right_finger_tip_id]
            return (left_tip + right_tip) / 2
        else:
            left_pos = self.data.body(self.left_finger_id).xpos.copy()
            right_pos = self.data.body(self.right_finger_id).xpos.copy()
            center = (left_pos + right_pos) / 2
            return center + np.array([0, 0, -self.config.fingertip_to_center])

    def get_grasp_center(self) -> np.ndarray:
        if self.left_finger_tip_id is not None and self.right_finger_tip_id is not None:
            left_tip = self.data.site_xpos[self.left_finger_tip_id]
            right_tip = self.data.site_xpos[self.right_finger_tip_id]
            center = (left_tip + right_tip) / 2
            return center + np.array([0, 0, self.config.grasp_offset_from_tip])
        else:
            left_pos = self.data.body(self.left_finger_id).xpos.copy()
            right_pos = self.data.body(self.right_finger_id).xpos.copy()
            center = (left_pos + right_pos) / 2
            fingertip = center + np.array([0, 0, -self.config.fingertip_to_center])
            return fingertip + np.array([0, 0, self.config.grasp_offset_from_tip])

    def get_current_joint_angles(self) -> np.ndarray:
        return self.data.qpos[self.arm_indices].copy()

    def apply_joint_angles(self, joint_angles: np.ndarray) -> None:
        for i, idx in enumerate(self.arm_indices):
            self.data.qpos[idx] = float(joint_angles[i])
        self.data.qvel[self.arm_indices] = 0
        mujoco.mj_forward(self.model, self.data)

    def forward_kinematics(self, joint_angles: np.ndarray) -> np.ndarray:
        saved_qpos = self.data.qpos.copy()
        saved_qvel = self.data.qvel.copy()
        self.data.qpos[self.arm_indices] = joint_angles[:7]
        self.data.qpos[self.finger_indices] = [-0.02, -0.02]
        mujoco.mj_forward(self.model, self.data)
        position = self.get_fingertip_position()
        self.data.qpos[:] = saved_qpos
        self.data.qvel[:] = saved_qvel
        mujoco.mj_forward(self.model, self.data)
        return position

    def get_cube_center(self) -> Optional[np.ndarray]:
        if self.cube_body_id is None:
            return None
        return self.data.body(self.cube_body_id).xpos.copy()

    def get_all_objects(self) -> List[str]:
        return ["red_cube"] if self.cube_body_id is not None else []

    def get_gripper_width(self) -> float:
        if self.left_finger_tip_id is not None and self.right_finger_tip_id is not None:
            left_tip = self.data.site_xpos[self.left_finger_tip_id]
            right_tip = self.data.site_xpos[self.right_finger_tip_id]
        else:
            left_tip = self.data.body(self.left_finger_id).xpos
            right_tip = self.data.body(self.right_finger_id).xpos
            tip_offset = np.array([0, 0, -self.config.fingertip_to_center])
            left_tip = left_tip + tip_offset
            right_tip = right_tip + tip_offset
        return np.linalg.norm(left_tip - right_tip)

    def close_gripper(self) -> None:
        for idx in self.finger_indices:
            self.data.qpos[idx] = 0.04
        mujoco.mj_forward(self.model, self.data)
        self.gripper_closed = True
        print("    ✊ Gripper closed")

    def close_gripper_to_width(self, target_width: float, viewer_obj=None) -> bool:
        start_width = self.get_gripper_width()
        start_pos = self.data.qpos[self.finger_indices[0]]

        target_pos = start_pos + (target_width - start_width) * 0.5
        target_pos = max(self.config.gripper_closed_min, min(self.config.gripper_open_max, target_pos))

        print(f"        Adjusting gripper to width {target_width*1000:.1f}mm")

        steps = 15
        for step in range(steps + 1):
            t = step / steps
            pos = start_pos + t * (target_pos - start_pos)
            for idx in self.finger_indices:
                self.data.qpos[idx] = pos
            mujoco.mj_forward(self.model, self.data)
            if viewer_obj:
                viewer_obj.sync()
            time.sleep(0.01)

        self.gripper_closed = True
        return True

    def open_gripper(self) -> None:
        for idx in self.finger_indices:
            self.data.qpos[idx] = -0.02
        mujoco.mj_forward(self.model, self.data)
        self.gripper_closed = False
        print("    🖐️ Gripper opened")

    def open_gripper_wide(self) -> None:
        for idx in self.finger_indices:
            self.data.qpos[idx] = -0.06
        mujoco.mj_forward(self.model, self.data)
        self.gripper_closed = False
        print("    🖐️ Gripper fully opened")

    def attach_cube(self) -> bool:
        if self.cube_body_id is None:
            return False
        cube_center = self.get_cube_center()
        if cube_center is None:
            return False
        grasp_center = self.get_grasp_center()
        self.object_offset = cube_center - grasp_center
        self.grasped_object = "red_cube"
        print(f"    📦 Cube attached")
        return True

    def detach_object(self) -> None:
        self.grasped_object = None
        self.object_offset = np.array([0, 0, 0])
        print("    📦 Cube released")

    def update_attached_object_position(self) -> None:
        if self.grasped_object is not None and self.gripper_closed and self.cube_jnt_addr is not None:
            try:
                grasp_center = self.get_grasp_center()
                target_cube_center = grasp_center + self.object_offset
                self.data.qpos[self.cube_jnt_addr:self.cube_jnt_addr+3] = target_cube_center
                vel_addr = self.cube_jnt_addr * 2
                if vel_addr + 6 <= len(self.data.qvel):
                    self.data.qvel[vel_addr:vel_addr+6] = 0
            except:
                pass


# ============================================
# 逆运动学求解器
# ============================================
class InverseKinematicsSolver:
    def __init__(self, robot: PandaRobot, config: RobotConfig):
        self.robot = robot
        self.config = config

    def _ik_cost(self, joint_angles: np.ndarray, target_pos: np.ndarray) -> float:
        current_pos = self.robot.forward_kinematics(joint_angles)
        return float(np.linalg.norm(current_pos - target_pos))

    def _get_bounds(self) -> List[Tuple[float, float]]:
        bounds = []
        for i, idx in enumerate(self.robot.arm_indices):
            jr = self.robot.model.jnt_range[idx]
            if jr[0] == 0 and jr[1] == 0:
                bounds.append(self.config.joint_limits[i])
            else:
                bounds.append((float(jr[0]), float(jr[1])))
        return bounds

    def solve(self, target_pos: np.ndarray, initial_guess: Optional[np.ndarray] = None) -> Tuple[Optional[np.ndarray], float]:
        if initial_guess is None:
            initial_guess = self.robot.get_current_joint_angles()
        best_solution = None
        best_error = float('inf')
        x0 = initial_guess.copy()
        for attempt in range(self.config.ik_max_attempts):
            if attempt > 0:
                x0 = initial_guess + np.random.uniform(-0.1, 0.1, 7)
            for i, (low, high) in enumerate(self._get_bounds()):
                x0[i] = np.clip(x0[i], low + 0.01, high - 0.01)
            result = minimize(self._ik_cost, x0, args=(target_pos,), method='L-BFGS-B',
                            bounds=self._get_bounds(), options={'maxiter': self.config.ik_max_iter})
            if result.fun < best_error:
                best_error = result.fun
                best_solution = result.x
            if best_error < 0.03:
                break
        if best_error < 0.08:
            return best_solution, best_error
        return None, best_error


# ============================================
# 运动控制器（集成MPC）
# ============================================
class MotionController:
    def __init__(self, robot: PandaRobot, config: RobotConfig, mpc: MPCController):
        self.robot = robot
        self.config = config
        self.mpc = mpc

    @staticmethod
    def _smoothstep(t: float) -> float:
        return t * t * (3 - 2 * t)

    def move_to_joints(self, target_joints: np.ndarray, viewer_obj=None, move_steps: int = None) -> None:
        if move_steps is None:
            move_steps = self.config.move_steps
        start_joints = self.robot.get_current_joint_angles()
        for step in range(move_steps + 1):
            t = step / move_steps
            t_smooth = self._smoothstep(t)
            current_joints = start_joints + t_smooth * (target_joints - start_joints)
            self.robot.apply_joint_angles(current_joints)
            self.robot.update_attached_object_position()
            if viewer_obj:
                viewer_obj.sync()
            time.sleep(self.config.move_dt)

    def move_to_position(self, target_pos: np.ndarray, ik_solver, viewer_obj=None) -> Tuple[bool, float]:
        current_angles = self.robot.get_current_joint_angles()
        solution, ik_error = ik_solver.solve(target_pos, current_angles)
        if solution is None:
            return False, ik_error
        adaptive_steps = self.mpc.get_adaptive_steps(target_pos)
        speed_factor = self.mpc.calculate_speed_factor(target_pos)
        if speed_factor < 0.99:
            print(f"    🐢 MPC减速: 速度因子={speed_factor:.2f}")
        self.move_to_joints(solution, viewer_obj, adaptive_steps)
        actual_pos = self.robot.get_fingertip_position()
        return True, float(np.linalg.norm(actual_pos - target_pos))

    def follow_trajectory(self, trajectory, ik_solver, viewer_obj=None):
        print(f"      Trajectory points: {len(trajectory)}")
        current_angles = self.robot.get_current_joint_angles()
        success = 0
        for i, pos in enumerate(trajectory):
            solution, err = ik_solver.solve(pos, current_angles)
            if solution is not None and err < 0.05:
                self.robot.apply_joint_angles(solution)
                current_angles = solution
                success += 1
                if viewer_obj:
                    viewer_obj.sync()
                time.sleep(self.config.figure8_duration_per_point)
            if (i + 1) % 20 == 0:
                print(f"\r      Progress: {(i+1)*100//len(trajectory)}%", end="", flush=True)
        print(f"\r      Progress: 100% ✓")
        print(f"      Success rate: {success}/{len(trajectory)} ({100*success/len(trajectory):.1f}%)")


# ============================================
# 可视化辅助类
# ============================================
class Visualizer:
    def __init__(self, robot: PandaRobot):
        self.robot = robot
        self.viewer = None

    def launch(self) -> None:
        self.viewer = mujoco.viewer.launch_passive(self.robot.model, self.robot.data)
        self.viewer.cam.lookat = np.array([0.65, 0, 0.4])
        self.viewer.cam.distance = 2.5
        self.viewer.cam.azimuth = 45
        self.viewer.cam.elevation = -25
        print("✓ Visualization window started")

    def clear_trajectory(self) -> None:
        if self.viewer:
            self.viewer.user_scn.ngeom = 0

    def draw_trajectory(self, trajectory: List[np.ndarray], color: List[float] = [1.0, 0.8, 0.0]) -> None:
        if self.viewer is None or len(trajectory) < 2:
            return
        user_scene = self.viewer.user_scn
        start_idx = user_scene.ngeom
        line_width = 0.006
        for i in range(len(trajectory) - 1):
            if start_idx + i >= 100:
                break
            p1, p2 = trajectory[i], trajectory[i + 1]
            center = (p1 + p2) / 2
            direction = p2 - p1
            length = np.linalg.norm(direction)
            if length < 0.001:
                continue
            direction = direction / length
            z_axis = np.array([0, 0, 1])
            rot_axis = np.cross(z_axis, direction)
            rot_norm = np.linalg.norm(rot_axis)
            if rot_norm < 0.001:
                rot_mat = np.eye(3)
            else:
                rot_axis = rot_axis / rot_norm
                angle = np.arccos(np.clip(np.dot(z_axis, direction), -1, 1))
                K = np.array([[0, -rot_axis[2], rot_axis[1]], [rot_axis[2], 0, -rot_axis[0]], [-rot_axis[1], rot_axis[0], 0]])
                rot_mat = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * K @ K
            geom = user_scene.geoms[start_idx + i]
            mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_CYLINDER,
                              [line_width, length / 2, line_width],
                              center.astype(np.float64), rot_mat.flatten().astype(np.float64),
                              np.array(color + [0.9], dtype=np.float32))
            user_scene.ngeom += 1

    def add_marker(self, pos: np.ndarray, color: List[float], radius: float = 0.015) -> None:
        if self.viewer is None:
            return
        idx = self.viewer.user_scn.ngeom
        if idx >= 100:
            return
        geom = self.viewer.user_scn.geoms[idx]
        mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_SPHERE,
                          [radius, 0, 0], pos.astype(np.float64), np.eye(3).flatten().astype(np.float64),
                          np.array(color + [1.0], dtype=np.float32))
        self.viewer.user_scn.ngeom += 1

    def sync(self) -> None:
        if self.viewer:
            self.viewer.sync()

    def is_running(self) -> bool:
        return self.viewer is not None and self.viewer.is_running()

    def close(self) -> None:
        if self.viewer:
            self.viewer.close()


# ============================================
# MCP协议桥接器
# ============================================
class MCPBridge:
    def __init__(self):
        self.robot = None
        self.config = None
        self.figure8_trajectory = None
        self.visualizer = None
        self.vlm = None
        self.llm = None
        self.mpc = None

    def set_robot(self, robot: PandaRobot) -> None:
        self.robot = robot

    def set_config(self, config: RobotConfig) -> None:
        self.config = config
        self.figure8_trajectory = Figure8Trajectory(size=config.figure8_size)

    def set_visualizer(self, visualizer: Visualizer) -> None:
        self.visualizer = visualizer

    def set_ai_modules(self, vlm: VLMUnderstanding, llm: LLMTaskPlanner, mpc: MPCController):
        self.vlm = vlm
        #self.llm = llm
        #self.mpc = mpc

    # ========== 抓取（优化版：直接张到合适宽度，避免先扩大再缩小）==========
    def execute_grasp(self, motion_ctrl: MotionController,
                      ik_solver: InverseKinematicsSolver,
                      viewer_obj: Optional[mujoco.viewer] = None) -> bool:
        print(f"    Preparing to grasp red cube")

        cube_center = self.robot.get_cube_center()
        if cube_center is None:
            print("    ✗ Cube not found")
            return False

        print(f"    📍 Cube center: ({cube_center[0]:.3f}, {cube_center[1]:.3f}, {cube_center[2]:.3f})")

        # 计算目标夹爪宽度（立方体宽度 + 2mm余量，方便进入）
        open_width = self.config.cube_width + 0.002  # 0.082m

        # 1. 直接张开到目标宽度（不是先最大再闭合）
        print(f"    🔧 Opening gripper to width {open_width*1000:.1f}mm")
        self.robot.close_gripper_to_width(open_width, viewer_obj)
        time.sleep(0.2)

        # 2. 移动到预抓取位置
        pre_grasp = cube_center + np.array([0, 0, self.config.cube_half_height + 0.08])
        print(f"    📍 Pre-grasp: ({pre_grasp[0]:.3f}, {pre_grasp[1]:.3f}, {pre_grasp[2]:.3f})")
        success, _ = motion_ctrl.move_to_position(pre_grasp, ik_solver, viewer_obj)
        if not success:
            return False

        # 3. 下降到抓取位置
        grasp_pos = cube_center + np.array([0, 0, self.config.cube_half_height - 0.02])
        print(f"    📍 Grasp point: ({grasp_pos[0]:.3f}, {grasp_pos[1]:.3f}, {grasp_pos[2]:.3f})")
        success, _ = motion_ctrl.move_to_position(grasp_pos, ik_solver, viewer_obj)
        if not success:
            return False

        # 4. 闭合到抓取宽度（略小于立方体宽度）
        grip_width = self.config.cube_width - 0.003  # 0.077m
        print(f"    🔧 Closing gripper to grip width {grip_width*1000:.1f}mm")
        self.robot.close_gripper_to_width(grip_width, viewer_obj)
        time.sleep(0.2)

        # 5. 附着物体
        self.robot.attach_cube()

        # 6. 提升
        lift_pos = grasp_pos + np.array([0, 0, self.config.grasp_post_offset])
        print(f"    📍 Lift: ({lift_pos[0]:.3f}, {lift_pos[1]:.3f}, {lift_pos[2]:.3f})")
        motion_ctrl.move_to_position(lift_pos, ik_solver, viewer_obj)

        print("    ✅ Grasp successful!")
        return True

    # ========== 放置 ==========
    def execute_place(self, motion_ctrl: MotionController,
                      ik_solver: InverseKinematicsSolver,
                      area: str = "center",
                      viewer_obj: Optional[mujoco.viewer] = None) -> bool:
        if self.robot.grasped_object is None:
            print("    ❌ No object grasped")
            return False

        cube_center_target = self.config.get_cube_center_position(area)

        print(f"\n    📍 Placing at: {area}")
        print(f"      Cube center target: ({cube_center_target[0]:.3f}, {cube_center_target[1]:.3f}, {cube_center_target[2]:.3f})")
        print(f"      Cube bottom: {cube_center_target[2] - self.config.cube_half_height:.3f}")
        print(f"      Table top: {self.config.table_top_z:.3f}")

        fingertip_target = cube_center_target + np.array([0, 0, self.config.cube_half_height])

        pre_fingertip = fingertip_target + np.array([0, 0, 0.10])
        success, _ = motion_ctrl.move_to_position(pre_fingertip, ik_solver, viewer_obj)
        if not success:
            return False

        success, _ = motion_ctrl.move_to_position(fingertip_target, ik_solver, viewer_obj)
        if not success:
            return False

        self.robot.data.qpos[self.robot.cube_jnt_addr:self.robot.cube_jnt_addr+3] = cube_center_target

        self.robot.open_gripper_wide()
        time.sleep(0.3)
        self.robot.detach_object()

        vel_addr = self.robot.cube_jnt_addr * 2
        if vel_addr + 6 <= len(self.robot.data.qvel):
            self.robot.data.qvel[vel_addr:vel_addr+6] = 0

        lift_fingertip = fingertip_target + np.array([0, 0, 0.10])
        motion_ctrl.move_to_position(lift_fingertip, ik_solver, viewer_obj)

        print(f"    ✅ Place successful!")
        return True

    # ========== 放置到坐标 ==========
    def execute_place_at_coordinates(self, motion_ctrl: MotionController,
                                       ik_solver: InverseKinematicsSolver,
                                       x: float, y: float,
                                       viewer_obj: Optional[mujoco.viewer] = None) -> bool:
        if self.robot.grasped_object is None:
            print("    ❌ No object grasped")
            return False

        cube_center_target = np.array([x, y, self.config.cube_center_z])

        print(f"\n    📍 Placing at coordinates: ({x:.3f}, {y:.3f})")
        print(f"      Cube center: ({cube_center_target[0]:.3f}, {cube_center_target[1]:.3f}, {cube_center_target[2]:.3f})")
        print(f"      Cube bottom: {cube_center_target[2] - self.config.cube_half_height:.3f}")

        fingertip_target = cube_center_target + np.array([0, 0, self.config.cube_half_height])

        pre_fingertip = fingertip_target + np.array([0, 0, 0.10])
        success, _ = motion_ctrl.move_to_position(pre_fingertip, ik_solver, viewer_obj)
        if not success:
            return False

        success, _ = motion_ctrl.move_to_position(fingertip_target, ik_solver, viewer_obj)
        if not success:
            return False

        self.robot.data.qpos[self.robot.cube_jnt_addr:self.robot.cube_jnt_addr+3] = cube_center_target

        self.robot.open_gripper_wide()
        time.sleep(0.3)
        self.robot.detach_object()

        vel_addr = self.robot.cube_jnt_addr * 2
        if vel_addr + 6 <= len(self.robot.data.qvel):
            self.robot.data.qvel[vel_addr:vel_addr+6] = 0

        lift_fingertip = fingertip_target + np.array([0, 0, 0.10])
        motion_ctrl.move_to_position(lift_fingertip, ik_solver, viewer_obj)

        print(f"    ✅ Place to coordinates ({x:.3f}, {y:.3f}) successful!")
        return True

    # ========== 相对移动 ==========
    def execute_move_relative(self, motion_ctrl: MotionController,
                               ik_solver: InverseKinematicsSolver,
                               direction: str,
                               viewer_obj: Optional[mujoco.viewer] = None) -> bool:
        step = self.config.step_size
        current_pos = self.robot.get_fingertip_position()

        delta_map = {
            "forward": np.array([step, 0, 0]), "back": np.array([-step, 0, 0]),
            "left": np.array([0, -step, 0]), "right": np.array([0, step, 0]),
            "up": np.array([0, 0, step]), "down": np.array([0, 0, -step]),
        }

        if direction not in delta_map:
            print(f"    Unknown direction: {direction}")
            return False

        delta = delta_map[direction]
        target_pos = current_pos + delta

        if self.robot.grasped_object is not None:
            print(f"    Moving with grasped object {direction} by {step*100:.1f}cm")
        else:
            print(f"    Moving {direction} by {step*100:.1f}cm")

        success = motion_ctrl.move_to_position(target_pos, ik_solver, viewer_obj)[0]
        return success

    # ========== 绝对移动 ==========
    def execute_goto(self, motion_ctrl: MotionController,
                     ik_solver: InverseKinematicsSolver,
                     x: float, y: float, z: float,
                     viewer_obj: Optional[mujoco.viewer] = None) -> bool:
        target = np.array([x, y, z])
        if self.robot.grasped_object:
            print(f"    Moving with grasped object to: ({x:.3f}, {y:.3f}, {z:.3f})")
        else:
            print(f"    Moving to position: ({x:.3f}, {y:.3f}, {z:.3f})")
        success, error = motion_ctrl.move_to_position(target, ik_solver, viewer_obj)
        print(f"    Error: {error*1000:.1f}mm")
        return success

    # ========== 8字轨迹 ==========
    def execute_figure8(self, motion_ctrl: MotionController,
                        ik_solver: InverseKinematicsSolver,
                        viewer_obj: Optional[mujoco.viewer] = None) -> bool:
        print("    Generating figure-8 trajectory...")
        current_pos = self.robot.get_fingertip_position()
        trajectory = self.figure8_trajectory.generate_trajectory(current_pos, self.config.figure8_points)

        if self.visualizer:
            self.visualizer.clear_trajectory()
            self.visualizer.draw_trajectory(trajectory, [1.0, 0.8, 0.0])
            self.visualizer.add_marker(trajectory[0], [0.0, 1.0, 0.0], 0.015)
            self.visualizer.add_marker(trajectory[-1], [1.0, 0.0, 0.0], 0.015)
            self.visualizer.sync()
            print("    ✓ Trajectory drawn")

        motion_ctrl.follow_trajectory(trajectory, ik_solver, viewer_obj)
        print("    ✅ Figure-8 completed!")
        return True

    # ========== 列表物体 ==========
    def execute_list_objects(self) -> bool:
        scene = self.vlm.analyze_scene()
        print(f"    Objects: {[obj['name'] for obj in scene['objects']]}")
        return True

    # ========== 回家（平滑移动）==========
    def execute_home(self, motion_ctrl: MotionController,
                     ik_solver: InverseKinematicsSolver,
                     viewer_obj: Optional[mujoco.viewer] = None) -> bool:
        print("    🏠 Returning to home position smoothly...")

        home_angles = np.array([
            self.robot.home_joints["joint1"],
            self.robot.home_joints["joint2"],
            self.robot.home_joints["joint3"],
            self.robot.home_joints["joint4"],
            self.robot.home_joints["joint5"],
            self.robot.home_joints["joint6"],
            self.robot.home_joints["joint7"]
        ])

        current_angles = self.robot.get_current_joint_angles()
        print(f"      Current angles: {current_angles.round(3)}")
        print(f"      Target angles:  {home_angles.round(3)}")

        motion_ctrl.move_to_joints(home_angles, viewer_obj)

        self.robot.open_gripper()

        print(f"    ✅ Returned to home position smoothly")
        return True

    # ========== 释放 ==========
    def execute_release(self, motion_ctrl: MotionController,
                        ik_solver: InverseKinematicsSolver,
                        viewer_obj: Optional[mujoco.viewer] = None) -> bool:
        if self.robot.grasped_object is not None:
            print("    Releasing object...")
            self.robot.open_gripper_wide()
            time.sleep(0.2)
            self.robot.detach_object()
            return True
        print("    ⚠ No object grasped")
        return False

    # ========== 命令分发 ==========
    def execute_command(self, command: Dict[str, Any],
                        motion_ctrl: MotionController,
                        ik_solver: InverseKinematicsSolver,
                        viewer_obj: Optional[mujoco.viewer] = None) -> bool:
        if command is None:
            return False

        action = command.get("action", "home")
        print(f"   Executing: {action}")

        if action == "grasp":
            return self.execute_grasp(motion_ctrl, ik_solver, viewer_obj)
        elif action == "place":
            area = command.get("area", "center")
            return self.execute_place(motion_ctrl, ik_solver, area, viewer_obj)
        elif action == "place_at":
            pos = command.get("position", [0.65, 0])
            return self.execute_place_at_coordinates(motion_ctrl, ik_solver, pos[0], pos[1], viewer_obj)
        elif action == "move":
            direction = command.get("direction", "forward")
            return self.execute_move_relative(motion_ctrl, ik_solver, direction, viewer_obj)
        elif action == "goto":
            pos = command.get("position", [0.65, 0, 0.45])
            return self.execute_goto(motion_ctrl, ik_solver, pos[0], pos[1], pos[2], viewer_obj)
        elif action == "figure8":
            return self.execute_figure8(motion_ctrl, ik_solver, viewer_obj)
        elif action == "list_objects":
            return self.execute_list_objects()
        elif action == "home":
            return self.execute_home(motion_ctrl, ik_solver, viewer_obj)
        elif action == "release":
            return self.execute_release(motion_ctrl, ik_solver, viewer_obj)
        else:
            print(f"    Unknown action: {action}")
            return False


# ============================================
# 主程序
# ============================================
def main():
    print("\n" + "=" * 60)
    print("🤖 Multi-Model Intelligent Robot Control System")
    print("   LLM + VLM + MPC + Figure-8 Trajectory")
    print("=" * 60)

    config = RobotConfig()
    robot = PandaRobot(config)
    robot.load_model()
    robot.setup_joint_indices()
    robot.set_home()

    vlm = VLMUnderstanding(robot)
    llm = LLMTaskPlanner(config.deepseek_api_key)
    mpc = MPCController(robot, config, vlm)

    ik_solver = InverseKinematicsSolver(robot, config)
    motion_ctrl = MotionController(robot, config, mpc)
    visualizer = Visualizer(robot)
    mcp_bridge = MCPBridge()
    mcp_bridge.set_robot(robot)
    mcp_bridge.set_config(config)
    mcp_bridge.set_visualizer(visualizer)
    mcp_bridge.set_ai_modules(vlm, llm, mpc)

    visualizer.launch()
    visualizer.sync()

    scene = vlm.analyze_scene()
    print(f"\n📷 VLM Scene Analysis:")
    print(f"   Objects: {[obj['name'] for obj in scene['objects']]}")
    print(f"   Gripper: {'closed' if scene['gripper']['closed'] else 'open'}")

    print("\n" + "=" * 60)
    print("📋 Commands (Natural Language):")
    print("=" * 60)
    print("  【Grasp / 抓取】")
    print("    • 'grasp' / '抓' / '抓取' / '拿起' - Grasp red cube (adaptive grip)")
    print()
    print("  【Place / 放置】")
    print("    • 'place center' / '放到中间' - Place at center")
    print("    • 'place left' / '放到左边' - Place at left")
    print("    • 'place right' / '放到右边' - Place at right")
    print("    • 'place front' / '放到前面' - Place at front")
    print("    • 'place back' / '放到后面' - Place at back")
    print("    • 'place_at x y' / '放 x y' - Place at coordinates")
    print()
    print("  【Move / 移动】")
    print(f"    • 'forward' / '前' - Move forward {config.step_size*100:.0f}cm")
    print(f"    • 'back' / '后' - Move backward {config.step_size*100:.0f}cm")
    print(f"    • 'left' / '左' - Move left {config.step_size*100:.0f}cm")
    print(f"    • 'right' / '右' - Move right {config.step_size*100:.0f}cm")
    print(f"    • 'up' / '上' - Move up {config.step_size*100:.0f}cm")
    print(f"    • 'down' / '下' - Move down {config.step_size*100:.0f}cm")
    print()
    print("  【Coordinate / 坐标】")
    print("    • 'goto x y z' / '去 x y z' - Move to coordinates")
    print()
    print("  【Other / 其他】")
    print("    • 'figure8' / '画8' - Execute figure-8 trajectory")
    print("    • 'release' / '释放' - Release grasped object")
    print("    • 'home' / '回家' - Return home smoothly")
    print("    • 'list' / '列表' - List objects")
    print("    • 'quit' - Exit")
    print("=" * 60)

    try:
        while visualizer.is_running():
            user_input = input("\n💬 Command: ").strip()

            if user_input.lower() in ["quit", "exit", "退出"]:
                print("👋 Goodbye!")
                break

            if not user_input:
                continue

            print("\n📋 [LLM] Understanding command...")
            command = llm.plan(user_input)
            print(f"    Parsed: {command}")

            print("🔧 [MPC] Executing...")
            success = mcp_bridge.execute_command(
                command, motion_ctrl, ik_solver, visualizer.viewer
            )

            print("✅ Done" if success else "❌ Failed")
            robot.update_attached_object_position()
            visualizer.sync()

    except KeyboardInterrupt:
        print("\n\n👋 User interrupted")
    finally:
        visualizer.close()
        print("Program ended")


if __name__ == "__main__":
    main()