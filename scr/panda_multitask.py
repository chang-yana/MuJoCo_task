"""
多模型融合智能机械臂控制系统

主要实现了一个完整的智能机械臂控制系统，集成了以下功能：
- LLM任务规划（基于规则匹配的自然语言指令解析）
- VLM视觉感知（场景分析和物体识别）
- MPC速度约束（靠近物体自动减速）
- 自适应抓取/精确放置
- 轨迹跟踪（8字形轨迹）
- 可视化界面

"""

import mujoco
import mujoco.viewer
import numpy as np
import time
import re
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any
from scipy.optimize import minimize
from collections import deque


# ============================================================================
# 配置参数类
# ============================================================================

@dataclass
class RobotConfig:
    """
    机器人配置参数类

    包含所有可配置的参数：模型路径、运动参数、几何参数、控制参数等。
    使用 @dataclass 装饰器自动生成 __init__ 方法。

    Attributes:
        model_path: MuJoCo模型文件路径
        move_steps: 运动插值步数
        move_dt: 每步时间间隔（秒）
        ik_max_attempts: IK求解最大尝试次数
        ik_max_iter: IK求解最大迭代次数
        ik_tolerance: IK收敛精度（米）
        deepseek_api_key: DeepSeek API密钥（当前未使用）
        grasp_pre_offset: 预抓取高度偏移（米）
        grasp_post_offset: 抓取后提升高度（米）
        grasp_offset_from_tip: 指尖偏移量（米）
        table_top_z: 桌面上表面高度（米）
        cube_half_height: 立方体半高（米）
        cube_center_z: 立方体中心高度（米）
        fingertip_to_center: 指尖到夹爪中心偏移（米）
        mpc_safe_distance: MPC安全距离（米）
        mpc_min_speed_factor: MPC最小速度因子
        figure8_size: 8字形轨迹大小（米）
        figure8_points: 8字形轨迹点数
        figure8_duration_per_point: 每点停留时间（秒）
        step_size: 相对移动步长（米）
        gripper_open_max: 夹爪最大张开位置
        gripper_closed_min: 夹爪最小闭合位置
        cube_width: 立方体边长（米）
        joint_limits: 关节限位列表
        place_areas: 预设放置区域字典
    """
    model_path: str = r"D:\Mujoco\mujoco_menagerie-main\franka_emika_panda\scene_1.xml"
    move_steps: int = 200
    move_dt: float = 0.01
    ik_max_attempts: int = 3
    ik_max_iter: int = 150
    ik_tolerance: float = 1e-4

    # API配置（当前使用规则匹配，保留接口以备扩展）
    deepseek_api_key: str = ""

    # 抓取参数
    grasp_pre_offset: float = 0.12
    grasp_post_offset: float = 0.15
    grasp_offset_from_tip: float = 0.015

    # 几何参数
    table_top_z: float = 0.38
    cube_half_height: float = 0.04
    cube_center_z: float = 0.42
    fingertip_to_center: float = 0.045

    # MPC（模型预测控制）参数
    mpc_safe_distance: float = 0.15
    mpc_min_speed_factor: float = 0.3

    # 8字形轨迹参数
    figure8_size: float = 0.12
    figure8_points: int = 60
    figure8_duration_per_point: float = 0.02

    # 运动参数
    step_size: float = 0.05

    # 夹爪参数
    gripper_open_max: float = -0.06
    gripper_closed_min: float = 0.04
    cube_width: float = 0.08

    # 关节限位（Panda机械臂官方参数）
    joint_limits: List[Tuple[float, float]] = field(default_factory=lambda: [
        (-2.8973, 2.8973),  # joint1: 腰部旋转
        (-1.7628, 1.7628),  # joint2: 肩部
        (-2.8973, 2.8973),  # joint3: 肘部
        (-3.0718, -0.0698),  # joint4: 第一前臂（特殊：全负值）
        (-2.8973, 2.8973),  # joint5: 第二前臂
        (-0.0175, 3.7525),  # joint6: 腕部
        (-2.8973, 2.8973)  # joint7: 手部旋转
    ])

    # 预设放置区域（坐标：x, y）
    place_areas: Dict[str, List[float]] = field(default_factory=lambda: {
        "center": [0.65, 0],
        "left": [0.65, -0.3],
        "right": [0.65, 0.3],
        "front": [0.85, 0],
        "back": [0.45, 0],
    })

    def get_cube_center_position(self, area: str) -> np.ndarray:
        """
        获取立方体中心的目标放置位置

        Args:
            area: 放置区域名称（center/left/right/front/back）

        Returns:
            目标位置数组 [x, y, z]
        """
        if area in self.place_areas:
            x, y = self.place_areas[area]
        else:
            x, y = 0.65, 0
        return np.array([x, y, self.cube_center_z])


# ============================================================================
# VLM视觉理解模块
# ============================================================================

class VLMUnderstanding:
    """
    VLM（视觉语言模型）视觉理解模块

    负责分析当前场景，识别物体位置和夹爪状态。
    当前实现从MuJoCo仿真中直接读取物体位置。
    """

    def __init__(self, robot: 'PandaRobot') -> None:
        """
        初始化VLM模块

        Args:
            robot: PandaRobot实例
        """
        self.robot = robot

    def analyze_scene(self) -> Dict[str, Any]:
        """
        分析当前场景

        Returns:
            场景信息字典，包含：
            - objects: 物体列表（名称、颜色、形状、位置）
            - gripper: 夹爪状态（闭合状态、抓取的物体）
        """
        objects = []
        cube_center = self.robot.get_cube_center()

        if cube_center is not None:
            objects.append({
                "name": "red_cube",
                "color": "红色",
                "shape": "立方体",
                "position": {
                    "x": cube_center[0],
                    "y": cube_center[1],
                    "z": cube_center[2]
                }
            })

        return {
            "objects": objects,
            "gripper": {
                "closed": self.robot.gripper_closed,
                "grasped_object": self.robot.grasped_object
            }
        }


# ============================================================================
# LLM任务规划器（规则匹配版）
# ============================================================================

class LLMTaskPlanner:
    """
    LLM（大语言模型）任务规划器（规则匹配版）

    负责将用户的自然语言指令解析为机器可执行的命令。
    当前使用关键词匹配实现，无需API调用，响应速度快。

    支持的命令类型：
    - 抓取: grasp / 抓 / 拿 / 取
    - 放置: place / 放 / 放到
    - 移动: left / right / up / down / forward / back
    - 坐标移动: goto / 去 / 到
    - 8字轨迹: figure8 / 画8
    - 回家: home / 回家
    - 释放: release / 释放
    - 列表: list / 列表
    """

    def __init__(self, api_key: str = "") -> None:
        """
        初始化LLM规划器

        Args:
            api_key: API密钥（当前未使用，保留接口）
        """
        print("⚠ Using rule-based command parser")

    def plan(self, user_command: str) -> Dict[str, Any]:
        """
        解析用户自然语言指令

        Args:
            user_command: 用户输入的命令字符串

        Returns:
            命令字典，包含 action 和相应参数
        """
        cmd = user_command.lower().strip()

        # 抓取命令
        if any(w in cmd for w in ["抓", "拿", "取", "grasp", "pick", "抓取", "拿起", "拾取"]):
            return {"action": "grasp"}

        # 放置命令
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

        # 放置到坐标
        match_place_at = re.match(r'^(?:place_at|放|放置)\s+(-?\d*\.?\d+)\s+(-?\d*\.?\d+)$', cmd)
        if match_place_at:
            x = float(match_place_at.group(1))
            y = float(match_place_at.group(2))
            return {"action": "place_at", "position": [x, y]}

        # 相对移动命令
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

        # 绝对移动命令
        match_goto = re.match(r'^(?:goto|去|到|移动)\s+(-?\d*\.?\d+)\s+(-?\d*\.?\d+)\s+(-?\d*\.?\d+)$', cmd)
        if match_goto:
            x = float(match_goto.group(1))
            y = float(match_goto.group(2))
            z = float(match_goto.group(3))
            return {"action": "goto", "position": [x, y, z]}

        # 8字形轨迹命令
        if any(w in cmd for w in ["8", "eight", "画8", "八字", "8字", "画八字"]):
            return {"action": "figure8"}

        # 回家命令
        if any(w in cmd for w in ["家", "home", "原点", "初始", "回家", "复位"]):
            return {"action": "home"}

        # 释放命令
        if any(w in cmd for w in ["释放", "release", "松开", "放下", "drop"]):
            return {"action": "release"}

        # 列表命令
        if any(w in cmd for w in ["列表", "list", "物体", "objects"]):
            return {"action": "list_objects"}

        # 默认返回home
        return {"action": "home"}


# ============================================================================
# MPC速度控制器
# ============================================================================

class MPCController:
    """
    MPC（模型预测控制）速度控制器

    实现靠近物体时自动减速的功能，提高操作安全性。
    速度因子根据距离目标的距离线性变化。
    """

    def __init__(self, robot: 'PandaRobot', config: RobotConfig, vlm: VLMUnderstanding) -> None:
        """
        初始化MPC控制器

        Args:
            robot: PandaRobot实例
            config: 机器人配置
            vlm: VLM视觉模块
        """
        self.robot = robot
        self.config = config
        self.vlm = vlm

    def calculate_speed_factor(self, target_pos: np.ndarray) -> float:
        """
        计算速度因子

        根据当前位置到目标的距离，计算速度因子。距离越近，速度越慢。

        Args:
            target_pos: 目标位置

        Returns:
            速度因子（0.3 ~ 1.0）
        """
        current_pos = self.robot.get_fingertip_position()
        distance = np.linalg.norm(current_pos - target_pos)

        # 获取最近的物体距离
        scene = self.vlm.analyze_scene()
        min_obj_dist = float('inf')

        for obj in scene["objects"]:
            obj_pos = np.array([obj["position"]["x"], obj["position"]["y"], obj["position"]["z"]])
            min_obj_dist = min(min_obj_dist, np.linalg.norm(current_pos - obj_pos))

        min_distance = min(distance, min_obj_dist)

        # 线性插值计算速度因子
        if min_distance < self.config.mpc_safe_distance:
            speed_factor = self.config.mpc_min_speed_factor + \
                           (1 - self.config.mpc_min_speed_factor) * (min_distance / self.config.mpc_safe_distance)
            return max(self.config.mpc_min_speed_factor, min(1.0, speed_factor))

        return 1.0

    def get_adaptive_steps(self, target_pos: np.ndarray) -> int:
        """
        根据距离自适应调整运动步数

        近距离使用更多步数（精细运动），远距离使用更少步数（快速运动）。

        Args:
            target_pos: 目标位置

        Returns:
            自适应步数（50 ~ 400）
        """
        current_pos = self.robot.get_fingertip_position()
        distance = np.linalg.norm(current_pos - target_pos)
        speed_factor = self.calculate_speed_factor(target_pos)

        base_steps = self.config.move_steps
        if distance < 0.1:
            base_steps = int(base_steps * 1.5)  # 近距离精细运动
        elif distance > 0.3:
            base_steps = int(base_steps * 0.8)  # 远距离快速运动

        return max(50, min(400, int(base_steps / speed_factor)))


# ============================================================================
# 轨迹生成器
# ============================================================================

class Figure8Trajectory:
    """
    8字形轨迹生成器

    生成立式的倒8字形轨迹（无穷符号 ∞）。
    """

    def __init__(self, size: float = 0.12) -> None:
        """
        初始化轨迹生成器

        Args:
            size: 轨迹大小（米）
        """
        self.size = size

    def generate_trajectory(self, center: np.ndarray, num_points: int) -> List[np.ndarray]:
        """
        生成离散轨迹点

        Args:
            center: 轨迹中心点
            num_points: 轨迹点数

        Returns:
            轨迹点列表，每个点为 [x, y, z]
        """
        trajectory = []
        for i in range(num_points + 1):
            t = i / num_points
            theta = 2 * np.pi * t

            # 立式8字形：左右摆动（X轴）+ 上下画8（Z轴）
            horizontal = self.size * np.sin(theta)
            vertical = self.size * 0.5 * np.sin(2 * theta)

            trajectory.append(np.array([
                center[0] + horizontal,
                center[1],
                center[2] + vertical
            ]))

        return trajectory


# ============================================================================
# Panda机械臂类
# ============================================================================

class PandaRobot:
    """
    Panda机械臂控制类

    负责机器人的底层控制，包括：
    - 模型加载
    - 关节控制
    - 正向/逆向运动学
    - 夹爪控制
    - 物体抓取/释放
    """

    def __init__(self, config: RobotConfig) -> None:
        """
        初始化Panda机械臂

        Args:
            config: 机器人配置
        """
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

        # Home位置关节角度
        self.home_joints = {
            "joint1": 0.0,
            "joint2": -0.785,
            "joint3": 0.0,
            "joint4": -2.356,
            "joint5": 0.0,
            "joint6": 1.571,
            "joint7": 0.785,
        }

    # ------------------------------------------------------------------------
    # 初始化方法
    # ------------------------------------------------------------------------

    def load_model(self) -> None:
        """加载MuJoCo模型"""
        self.model = mujoco.MjModel.from_xml_path(self.config.model_path)
        self.data = mujoco.MjData(self.model)
        self.renderer = mujoco.Renderer(self.model, height=480, width=640)
        print("✓ Model loaded")

    def setup_joint_indices(self) -> None:
        """设置关节索引"""
        # 手臂关节
        for name in self.arm_joint_names:
            joint_id = self.model.joint(name).id
            self.arm_indices.append(self.model.jnt_qposadr[joint_id])

        # 手指关节
        for name in self.finger_joint_names:
            joint_id = self.model.joint(name).id
            self.finger_indices.append(self.model.jnt_qposadr[joint_id])

        # 夹爪body ID
        self.left_finger_id = self.model.body("left_finger").id
        self.right_finger_id = self.model.body("right_finger").id

        # 指尖site
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

    def _init_cube(self) -> None:
        """初始化立方体物体"""
        try:
            self.cube_body_id = self.model.body("red_cube").id
            self.cube_jnt_addr = self.model.body_jntadr[self.cube_body_id]
            print(f"  ✅ Red cube found")
        except Exception as e:
            print(f"  ❌ Red cube not found: {e}")

    def set_home(self) -> None:
        """设置机械臂到Home位置"""
        for joint_name, joint_pos in self.home_joints.items():
            joint_id = self.model.joint(joint_name).id
            qpos_addr = self.model.jnt_qposadr[joint_id]
            self.data.qpos[qpos_addr] = joint_pos
        self.open_gripper()
        mujoco.mj_forward(self.model, self.data)

    # ------------------------------------------------------------------------
    # 运动学方法
    # ------------------------------------------------------------------------

    def get_fingertip_position(self) -> np.ndarray:
        """
        获取指尖中心位置

        Returns:
            指尖中心坐标 [x, y, z]
        """
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
        """
        获取手指内侧中心（抓取时使用）

        Returns:
            手指内侧中心坐标
        """
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
        """获取当前关节角度"""
        return self.data.qpos[self.arm_indices].copy()

    def apply_joint_angles(self, joint_angles: np.ndarray) -> None:
        """应用关节角度到机器人"""
        for i, idx in enumerate(self.arm_indices):
            self.data.qpos[idx] = float(joint_angles[i])
        self.data.qvel[self.arm_indices] = 0
        mujoco.mj_forward(self.model, self.data)

    def forward_kinematics(self, joint_angles: np.ndarray) -> np.ndarray:
        """
        正向运动学：计算给定关节角度下的末端位置

        Args:
            joint_angles: 7个关节角度

        Returns:
            末端位置 [x, y, z]
        """
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

    # ------------------------------------------------------------------------
    # 物体操作方法
    # ------------------------------------------------------------------------

    def get_cube_center(self) -> Optional[np.ndarray]:
        """获取立方体中心位置"""
        if self.cube_body_id is None:
            return None
        return self.data.body(self.cube_body_id).xpos.copy()

    def get_all_objects(self) -> List[str]:
        """获取场景中所有物体名称"""
        return ["red_cube"] if self.cube_body_id is not None else []

    def get_gripper_width(self) -> float:
        """
        获取当前夹爪宽度（两指尖距离）

        Returns:
            夹爪宽度（米）
        """
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

    # ------------------------------------------------------------------------
    # 夹爪控制方法
    # ------------------------------------------------------------------------

    def close_gripper(self) -> None:
        """闭合夹爪"""
        for idx in self.finger_indices:
            self.data.qpos[idx] = 0.04
        mujoco.mj_forward(self.model, self.data)
        self.gripper_closed = True
        print("    ✊ Gripper closed")

    def close_gripper_to_width(self, target_width: float, viewer_obj=None) -> bool:
        """
        自适应闭合夹爪到指定宽度

        Args:
            target_width: 目标宽度（米）
            viewer_obj: 可视化对象

        Returns:
            是否成功
        """
        start_width = self.get_gripper_width()
        start_pos = self.data.qpos[self.finger_indices[0]]

        target_pos = start_pos + (target_width - start_width) * 0.5
        target_pos = max(self.config.gripper_closed_min, min(self.config.gripper_open_max, target_pos))

        print(f"        Adjusting gripper to width {target_width * 1000:.1f}mm")

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
        """张开夹爪（常规张开）"""
        for idx in self.finger_indices:
            self.data.qpos[idx] = -0.02
        mujoco.mj_forward(self.model, self.data)
        self.gripper_closed = False
        print("    🖐️ Gripper opened")

    def open_gripper_wide(self) -> None:
        """最大张开夹爪"""
        for idx in self.finger_indices:
            self.data.qpos[idx] = -0.06
        mujoco.mj_forward(self.model, self.data)
        self.gripper_closed = False
        print("    🖐️ Gripper fully opened")

    # ------------------------------------------------------------------------
    # 抓取/释放方法
    # ------------------------------------------------------------------------

    def attach_cube(self) -> bool:
        """
        将立方体附着到夹爪上

        Returns:
            是否成功
        """
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
        """释放物体"""
        self.grasped_object = None
        self.object_offset = np.array([0, 0, 0])
        print("    📦 Cube released")

    def update_attached_object_position(self) -> None:
        """每帧更新附着物体的位置"""
        if self.grasped_object is not None and self.gripper_closed and self.cube_jnt_addr is not None:
            try:
                grasp_center = self.get_grasp_center()
                target_cube_center = grasp_center + self.object_offset
                self.data.qpos[self.cube_jnt_addr:self.cube_jnt_addr + 3] = target_cube_center

                vel_addr = self.cube_jnt_addr * 2
                if vel_addr + 6 <= len(self.data.qvel):
                    self.data.qvel[vel_addr:vel_addr + 6] = 0
            except:
                pass


# ============================================================================
# 逆运动学求解器
# ============================================================================

class InverseKinematicsSolver:
    """
    逆运动学求解器

    使用数值优化方法（L-BFGS-B）求解给定目标位置的关节角度。
    支持多初始值和随机扰动以提高求解成功率。
    """

    def __init__(self, robot: PandaRobot, config: RobotConfig) -> None:
        """
        初始化IK求解器

        Args:
            robot: PandaRobot实例
            config: 机器人配置
        """
        self.robot = robot
        self.config = config

    def _ik_cost(self, joint_angles: np.ndarray, target_pos: np.ndarray) -> float:
        """
        IK代价函数

        Args:
            joint_angles: 候选关节角度
            target_pos: 目标位置

        Returns:
            位置误差（欧氏距离）
        """
        current_pos = self.robot.forward_kinematics(joint_angles)
        return float(np.linalg.norm(current_pos - target_pos))

    def _get_bounds(self) -> List[Tuple[float, float]]:
        """
        获取关节边界约束

        Returns:
            关节限位列表
        """
        bounds = []
        for i, idx in enumerate(self.robot.arm_indices):
            jr = self.robot.model.jnt_range[idx]
            if jr[0] == 0 and jr[1] == 0:
                bounds.append(self.config.joint_limits[i])
            else:
                bounds.append((float(jr[0]), float(jr[1])))
        return bounds

    def solve(self, target_pos: np.ndarray,
              initial_guess: Optional[np.ndarray] = None) -> Tuple[Optional[np.ndarray], float]:
        """
        求解逆运动学

        Args:
            target_pos: 目标位置 [x, y, z]
            initial_guess: 初始猜测关节角度（可选）

        Returns:
            (关节角度解, 位置误差)，求解失败时关节角度为None
        """
        if initial_guess is None:
            initial_guess = self.robot.get_current_joint_angles()

        best_solution = None
        best_error = float('inf')
        x0 = initial_guess.copy()

        for attempt in range(self.config.ik_max_attempts):
            if attempt > 0:
                x0 = initial_guess + np.random.uniform(-0.1, 0.1, 7)

            # 确保初始值在限位内
            for i, (low, high) in enumerate(self._get_bounds()):
                x0[i] = np.clip(x0[i], low + 0.01, high - 0.01)

            result = minimize(
                self._ik_cost, x0, args=(target_pos,),
                method='L-BFGS-B', bounds=self._get_bounds(),
                options={'maxiter': self.config.ik_max_iter}
            )

            if result.fun < best_error:
                best_error = result.fun
                best_solution = result.x

            if best_error < 0.03:
                break

        if best_error < 0.08:
            return best_solution, best_error
        return None, best_error


# ============================================================================
# 运动控制器
# ============================================================================

class MotionController:
    """
    运动控制器

    实现平滑的关节空间运动控制，集成MPC速度约束。
    """

    def __init__(self, robot: PandaRobot, config: RobotConfig, mpc: MPCController) -> None:
        """
        初始化运动控制器

        Args:
            robot: PandaRobot实例
            config: 机器人配置
            mpc: MPC控制器
        """
        self.robot = robot
        self.config = config
        self.mpc = mpc

    @staticmethod
    def _smoothstep(t: float) -> float:
        """
        S曲线平滑函数

        Args:
            t: 插值参数 [0, 1]

        Returns:
            平滑后的值
        """
        return t * t * (3 - 2 * t)

    def move_to_joints(self, target_joints: np.ndarray, viewer_obj=None,
                       move_steps: int = None) -> None:
        """
        平滑移动到目标关节角度

        Args:
            target_joints: 目标关节角度
            viewer_obj: 可视化对象
            move_steps: 运动步数（可选）
        """
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

    def move_to_position(self, target_pos: np.ndarray, ik_solver,
                         viewer_obj=None) -> Tuple[bool, float]:
        """
        移动到目标空间位置

        Args:
            target_pos: 目标位置
            ik_solver: IK求解器
            viewer_obj: 可视化对象

        Returns:
            (是否成功, 最终误差)
        """
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

    def follow_trajectory(self, trajectory: List[np.ndarray], ik_solver,
                          viewer_obj=None) -> None:
        """
        跟踪轨迹点序列

        Args:
            trajectory: 轨迹点列表
            ik_solver: IK求解器
            viewer_obj: 可视化对象
        """
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
                percent = (i + 1) * 100 // len(trajectory)
                print(f"\r      Progress: {percent}%", end="", flush=True)

        print(f"\r      Progress: 100% ✓")
        print(f"      Success rate: {success}/{len(trajectory)} ({100 * success / len(trajectory):.1f}%)")


# ============================================================================
# 可视化辅助类
# ============================================================================

class Visualizer:
    """
    可视化辅助类

    负责MuJoCo可视化窗口的管理，包括：
    - 窗口启动与关闭
    - 相机视角设置
    - 轨迹绘制
    - 标记点添加
    """

    def __init__(self, robot: PandaRobot) -> None:
        """
        初始化可视化器

        Args:
            robot: PandaRobot实例
        """
        self.robot = robot
        self.viewer = None

    def launch(self) -> None:
        """启动可视化窗口"""
        self.viewer = mujoco.viewer.launch_passive(self.robot.model, self.robot.data)
        self._setup_camera()
        print("✓ Visualization window started")

    def _setup_camera(self) -> None:
        """设置相机视角"""
        self.viewer.cam.lookat = np.array([0.65, 0, 0.4])
        self.viewer.cam.distance = 2.5
        self.viewer.cam.azimuth = 45
        self.viewer.cam.elevation = -25

    def clear_trajectory(self) -> None:
        """清除轨迹"""
        if self.viewer:
            self.viewer.user_scn.ngeom = 0

    def draw_trajectory(self, trajectory: List[np.ndarray],
                        color: List[float] = None) -> None:
        """
        绘制轨迹线

        Args:
            trajectory: 轨迹点列表
            color: 颜色 [r, g, b]，默认金色
        """
        if color is None:
            color = [1.0, 0.8, 0.0]

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

            # 计算旋转矩阵
            z_axis = np.array([0, 0, 1])
            rot_axis = np.cross(z_axis, direction)
            rot_norm = np.linalg.norm(rot_axis)

            if rot_norm < 0.001:
                rot_mat = np.eye(3)
            else:
                rot_axis = rot_axis / rot_norm
                angle = np.arccos(np.clip(np.dot(z_axis, direction), -1, 1))
                K = np.array([
                    [0, -rot_axis[2], rot_axis[1]],
                    [rot_axis[2], 0, -rot_axis[0]],
                    [-rot_axis[1], rot_axis[0], 0]
                ])
                rot_mat = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * K @ K

            geom = user_scene.geoms[start_idx + i]
            mujoco.mjv_initGeom(
                geom, mujoco.mjtGeom.mjGEOM_CYLINDER,
                [line_width, length / 2, line_width],
                center.astype(np.float64), rot_mat.flatten().astype(np.float64),
                np.array(color + [0.9], dtype=np.float32)
            )
            user_scene.ngeom += 1

    def add_marker(self, pos: np.ndarray, color: List[float], radius: float = 0.015) -> None:
        """
        添加标记点（彩色小球）

        Args:
            pos: 位置 [x, y, z]
            color: 颜色 [r, g, b]
            radius: 半径（米）
        """
        if self.viewer is None:
            return

        idx = self.viewer.user_scn.ngeom
        if idx >= 100:
            return

        geom = self.viewer.user_scn.geoms[idx]
        mujoco.mjv_initGeom(
            geom, mujoco.mjtGeom.mjGEOM_SPHERE,
            [radius, 0, 0], pos.astype(np.float64),
            np.eye(3).flatten().astype(np.float64),
            np.array(color + [1.0], dtype=np.float32)
        )
        self.viewer.user_scn.ngeom += 1

    def sync(self) -> None:
        """同步视图"""
        if self.viewer:
            self.viewer.sync()

    def is_running(self) -> bool:
        """检查窗口是否运行中"""
        return self.viewer is not None and self.viewer.is_running()

    def close(self) -> None:
        """关闭窗口"""
        if self.viewer:
            self.viewer.close()


# ============================================================================
# MCP协议桥接器（命令执行器）
# ============================================================================

class MCPBridge:
    """
    MCP协议桥接器

    负责执行解析后的命令，是控制系统的核心执行模块。
    相当于"模型上下文协议"（Model Context Protocol）中的工具执行层。

    支持的操作：
    - grasp: 抓取物体
    - place: 放置到预设区域
    - place_at: 放置到指定坐标
    - move: 相对移动
    - goto: 绝对移动
    - figure8: 执行8字形轨迹
    - home: 回家
    - release: 释放物体
    - list_objects: 列出物体
    """

    def __init__(self) -> None:
        """初始化MCP桥接器"""
        self.robot = None
        self.config = None
        self.figure8_trajectory = None
        self.visualizer = None
        self.vlm = None

    def set_robot(self, robot: PandaRobot) -> None:
        """设置机器人引用"""
        self.robot = robot

    def set_config(self, config: RobotConfig) -> None:
        """设置配置引用"""
        self.config = config
        self.figure8_trajectory = Figure8Trajectory(size=config.figure8_size)

    def set_visualizer(self, visualizer: Visualizer) -> None:
        """设置可视化器引用"""
        self.visualizer = visualizer

    def set_ai_modules(self, vlm: VLMUnderstanding, llm: LLMTaskPlanner, mpc: MPCController) -> None:
        """设置AI模块引用"""
        self.vlm = vlm

    # ------------------------------------------------------------------------
    # 抓取执行
    # ------------------------------------------------------------------------

    def execute_grasp(self, motion_ctrl: MotionController,
                      ik_solver: InverseKinematicsSolver,
                      viewer_obj: Optional[mujoco.viewer] = None) -> bool:
        """
        执行抓取操作

        流程：
        1. 张开夹爪到合适宽度（略大于立方体）
        2. 移动到预抓取位置
        3. 下降到抓取位置
        4. 闭合夹爪到抓取宽度
        5. 附着物体
        6. 提升
        """
        print(f"    Preparing to grasp red cube")

        cube_center = self.robot.get_cube_center()
        if cube_center is None:
            print("    ✗ Cube not found")
            return False

        print(f"    📍 Cube center: ({cube_center[0]:.3f}, {cube_center[1]:.3f}, {cube_center[2]:.3f})")

        # 张开到略大于立方体的宽度
        open_width = self.config.cube_width + 0.002  # 0.082m
        print(f"    🔧 Opening gripper to width {open_width * 1000:.1f}mm")
        self.robot.close_gripper_to_width(open_width, viewer_obj)
        time.sleep(0.2)

        # 移动到预抓取位置
        pre_grasp = cube_center + np.array([0, 0, self.config.cube_half_height + 0.08])
        print(f"    📍 Pre-grasp: ({pre_grasp[0]:.3f}, {pre_grasp[1]:.3f}, {pre_grasp[2]:.3f})")
        success, _ = motion_ctrl.move_to_position(pre_grasp, ik_solver, viewer_obj)
        if not success:
            return False

        # 下降到抓取位置（下移2cm，增加接触面积）
        grasp_pos = cube_center + np.array([0, 0, self.config.cube_half_height - 0.02])
        print(f"    📍 Grasp point: ({grasp_pos[0]:.3f}, {grasp_pos[1]:.3f}, {grasp_pos[2]:.3f})")
        success, _ = motion_ctrl.move_to_position(grasp_pos, ik_solver, viewer_obj)
        if not success:
            return False

        # 闭合到抓取宽度
        grip_width = self.config.cube_width - 0.003  # 0.077m
        print(f"    🔧 Closing gripper to grip width {grip_width * 1000:.1f}mm")
        self.robot.close_gripper_to_width(grip_width, viewer_obj)
        time.sleep(0.2)

        # 附着物体
        self.robot.attach_cube()

        # 提升
        lift_pos = grasp_pos + np.array([0, 0, self.config.grasp_post_offset])
        print(f"    📍 Lift: ({lift_pos[0]:.3f}, {lift_pos[1]:.3f}, {lift_pos[2]:.3f})")
        motion_ctrl.move_to_position(lift_pos, ik_solver, viewer_obj)

        print("    ✅ Grasp successful!")
        return True

    # ------------------------------------------------------------------------
    # 放置执行
    # ------------------------------------------------------------------------

    def execute_place(self, motion_ctrl: MotionController,
                      ik_solver: InverseKinematicsSolver,
                      area: str = "center",
                      viewer_obj: Optional[mujoco.viewer] = None) -> bool:
        """
        执行放置操作（预设区域）

        Args:
            motion_ctrl: 运动控制器
            ik_solver: IK求解器
            area: 放置区域
            viewer_obj: 可视化对象
        """
        if self.robot.grasped_object is None:
            print("    ❌ No object grasped")
            return False

        # 计算目标位置
        cube_center_target = self.config.get_cube_center_position(area)

        print(f"\n    📍 Placing at: {area}")
        print(
            f"      Cube center target: ({cube_center_target[0]:.3f}, {cube_center_target[1]:.3f}, {cube_center_target[2]:.3f})")
        print(f"      Cube bottom: {cube_center_target[2] - self.config.cube_half_height:.3f}")
        print(f"      Table top: {self.config.table_top_z:.3f}")

        fingertip_target = cube_center_target + np.array([0, 0, self.config.cube_half_height])

        # 移动到放置位置
        pre_fingertip = fingertip_target + np.array([0, 0, 0.10])
        success, _ = motion_ctrl.move_to_position(pre_fingertip, ik_solver, viewer_obj)
        if not success:
            return False

        success, _ = motion_ctrl.move_to_position(fingertip_target, ik_solver, viewer_obj)
        if not success:
            return False

        # 精确设置立方体位置
        self.robot.data.qpos[self.robot.cube_jnt_addr:self.robot.cube_jnt_addr + 3] = cube_center_target

        # 释放
        self.robot.open_gripper_wide()
        time.sleep(0.3)
        self.robot.detach_object()

        # 重置速度
        vel_addr = self.robot.cube_jnt_addr * 2
        if vel_addr + 6 <= len(self.robot.data.qvel):
            self.robot.data.qvel[vel_addr:vel_addr + 6] = 0

        # 抬起
        lift_fingertip = fingertip_target + np.array([0, 0, 0.10])
        motion_ctrl.move_to_position(lift_fingertip, ik_solver, viewer_obj)

        print(f"    ✅ Place successful!")
        return True

    def execute_place_at_coordinates(self, motion_ctrl: MotionController,
                                     ik_solver: InverseKinematicsSolver,
                                     x: float, y: float,
                                     viewer_obj: Optional[mujoco.viewer] = None) -> bool:
        """
        执行放置操作（指定坐标）

        Args:
            motion_ctrl: 运动控制器
            ik_solver: IK求解器
            x: X坐标
            y: Y坐标
            viewer_obj: 可视化对象
        """
        if self.robot.grasped_object is None:
            print("    ❌ No object grasped")
            return False

        cube_center_target = np.array([x, y, self.config.cube_center_z])

        print(f"\n    📍 Placing at coordinates: ({x:.3f}, {y:.3f})")
        print(
            f"      Cube center: ({cube_center_target[0]:.3f}, {cube_center_target[1]:.3f}, {cube_center_target[2]:.3f})")
        print(f"      Cube bottom: {cube_center_target[2] - self.config.cube_half_height:.3f}")

        fingertip_target = cube_center_target + np.array([0, 0, self.config.cube_half_height])

        pre_fingertip = fingertip_target + np.array([0, 0, 0.10])
        success, _ = motion_ctrl.move_to_position(pre_fingertip, ik_solver, viewer_obj)
        if not success:
            return False

        success, _ = motion_ctrl.move_to_position(fingertip_target, ik_solver, viewer_obj)
        if not success:
            return False

        # 精确设置立方体位置
        self.robot.data.qpos[self.robot.cube_jnt_addr:self.robot.cube_jnt_addr + 3] = cube_center_target

        self.robot.open_gripper_wide()
        time.sleep(0.3)
        self.robot.detach_object()

        vel_addr = self.robot.cube_jnt_addr * 2
        if vel_addr + 6 <= len(self.robot.data.qvel):
            self.robot.data.qvel[vel_addr:vel_addr + 6] = 0

        lift_fingertip = fingertip_target + np.array([0, 0, 0.10])
        motion_ctrl.move_to_position(lift_fingertip, ik_solver, viewer_obj)

        print(f"    ✅ Place to coordinates ({x:.3f}, {y:.3f}) successful!")
        return True

    # ------------------------------------------------------------------------
    # 移动执行
    # ------------------------------------------------------------------------

    def execute_move_relative(self, motion_ctrl: MotionController,
                              ik_solver: InverseKinematicsSolver,
                              direction: str,
                              viewer_obj: Optional[mujoco.viewer] = None) -> bool:
        """
        执行相对移动

        Args:
            motion_ctrl: 运动控制器
            ik_solver: IK求解器
            direction: 移动方向
            viewer_obj: 可视化对象
        """
        step = self.config.step_size
        current_pos = self.robot.get_fingertip_position()

        delta_map = {
            "forward": np.array([step, 0, 0]),
            "back": np.array([-step, 0, 0]),
            "left": np.array([0, -step, 0]),
            "right": np.array([0, step, 0]),
            "up": np.array([0, 0, step]),
            "down": np.array([0, 0, -step]),
        }

        if direction not in delta_map:
            print(f"    Unknown direction: {direction}")
            return False

        delta = delta_map[direction]
        target_pos = current_pos + delta

        if self.robot.grasped_object is not None:
            print(f"    Moving with grasped object {direction} by {step * 100:.1f}cm")
        else:
            print(f"    Moving {direction} by {step * 100:.1f}cm")

        success = motion_ctrl.move_to_position(target_pos, ik_solver, viewer_obj)[0]
        return success

    def execute_goto(self, motion_ctrl: MotionController,
                     ik_solver: InverseKinematicsSolver,
                     x: float, y: float, z: float,
                     viewer_obj: Optional[mujoco.viewer] = None) -> bool:
        """
        执行绝对移动

        Args:
            motion_ctrl: 运动控制器
            ik_solver: IK求解器
            x: X坐标
            y: Y坐标
            z: Z坐标
            viewer_obj: 可视化对象
        """
        target = np.array([x, y, z])

        if self.robot.grasped_object:
            print(f"    Moving with grasped object to: ({x:.3f}, {y:.3f}, {z:.3f})")
        else:
            print(f"    Moving to position: ({x:.3f}, {y:.3f}, {z:.3f})")

        success, error = motion_ctrl.move_to_position(target, ik_solver, viewer_obj)
        print(f"    Error: {error * 1000:.1f}mm")
        return success

    # ------------------------------------------------------------------------
    # 轨迹执行
    # ------------------------------------------------------------------------

    def execute_figure8(self, motion_ctrl: MotionController,
                        ik_solver: InverseKinematicsSolver,
                        viewer_obj: Optional[mujoco.viewer] = None) -> bool:
        """
        执行8字形轨迹

        Args:
            motion_ctrl: 运动控制器
            ik_solver: IK求解器
            viewer_obj: 可视化对象
        """
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

    # ------------------------------------------------------------------------
    # 其他执行
    # ------------------------------------------------------------------------

    def execute_list_objects(self) -> bool:
        """列出场景中的物体"""
        scene = self.vlm.analyze_scene()
        print(f"    Objects: {[obj['name'] for obj in scene['objects']]}")
        return True

    def execute_home(self, motion_ctrl: MotionController,
                     ik_solver: InverseKinematicsSolver,
                     viewer_obj: Optional[mujoco.viewer] = None) -> bool:
        """
        平滑回家

        Args:
            motion_ctrl: 运动控制器
            ik_solver: IK求解器
            viewer_obj: 可视化对象
        """
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

    def execute_release(self, motion_ctrl: MotionController,
                        ik_solver: InverseKinematicsSolver,
                        viewer_obj: Optional[mujoco.viewer] = None) -> bool:
        """
        释放物体

        Args:
            motion_ctrl: 运动控制器
            ik_solver: IK求解器
            viewer_obj: 可视化对象
        """
        if self.robot.grasped_object is not None:
            print("    Releasing object...")
            self.robot.open_gripper_wide()
            time.sleep(0.2)
            self.robot.detach_object()
            return True
        print("    ⚠ No object grasped")
        return False

    # ------------------------------------------------------------------------
    # 命令分发
    # ------------------------------------------------------------------------

    def execute_command(self, command: Dict[str, Any],
                        motion_ctrl: MotionController,
                        ik_solver: InverseKinematicsSolver,
                        viewer_obj: Optional[mujoco.viewer] = None) -> bool:
        """
        执行解析后的命令

        Args:
            command: 命令字典
            motion_ctrl: 运动控制器
            ik_solver: IK求解器
            viewer_obj: 可视化对象

        Returns:
            是否执行成功
        """
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


# ============================================================================
# 主程序
# ============================================================================

def main() -> None:
    """
    主函数

    程序入口，初始化所有组件并进入交互循环。
    """
    print("\n" + "=" * 60)
    print("🤖 Multi-Model Intelligent Robot Control System")
    print("   LLM + VLM + MPC + Figure-8 Trajectory")
    print("=" * 60)

    # 初始化配置
    config = RobotConfig()

    # 初始化机器人
    robot = PandaRobot(config)
    robot.load_model()
    robot.setup_joint_indices()
    robot.set_home()

    # 初始化AI模块
    vlm = VLMUnderstanding(robot)
    llm = LLMTaskPlanner(config.deepseek_api_key)
    mpc = MPCController(robot, config, vlm)

    # 初始化控制器
    ik_solver = InverseKinematicsSolver(robot, config)
    motion_ctrl = MotionController(robot, config, mpc)
    visualizer = Visualizer(robot)
    mcp_bridge = MCPBridge()

    mcp_bridge.set_robot(robot)
    mcp_bridge.set_config(config)
    mcp_bridge.set_visualizer(visualizer)
    mcp_bridge.set_ai_modules(vlm, llm, mpc)

    # 启动可视化
    visualizer.launch()
    visualizer.sync()

    # 显示场景分析
    scene = vlm.analyze_scene()
    print(f"\n📷 VLM Scene Analysis:")
    print(f"   Objects: {[obj['name'] for obj in scene['objects']]}")
    print(f"   Gripper: {'closed' if scene['gripper']['closed'] else 'open'}")

    # 显示帮助信息
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
    print(f"    • 'forward' / '前' - Move forward {config.step_size * 100:.0f}cm")
    print(f"    • 'back' / '后' - Move backward {config.step_size * 100:.0f}cm")
    print(f"    • 'left' / '左' - Move left {config.step_size * 100:.0f}cm")
    print(f"    • 'right' / '右' - Move right {config.step_size * 100:.0f}cm")
    print(f"    • 'up' / '上' - Move up {config.step_size * 100:.0f}cm")
    print(f"    • 'down' / '下' - Move down {config.step_size * 100:.0f}cm")
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

    # 交互循环
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