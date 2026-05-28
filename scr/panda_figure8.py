"""
Panda机械臂倒8字形轨迹跟踪控制程序（立式倒8字形）
功能: 实现机械臂末端执行器跟踪立式的倒8字形轨迹（无穷符号）
"""

import os
import time
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple, cast

import mujoco
import mujoco.viewer
import numpy as np
from scipy.optimize import OptimizeResult, minimize


os.environ["MUJOCO_GL"] = "glfw"


# ============================================
# 配置参数
# ============================================
@dataclass
class RobotConfig:
    """机器人配置参数。"""

    model_path: str = r"D:\Mujoco\mujoco_menagerie-main\franka_emika_panda\scene.xml"

    move_steps: int = 200
    move_dt: float = 0.01
    pause_time: float = 1.0

    ik_max_attempts: int = 3
    ik_max_iter: int = 200
    ik_tolerance: float = 1e-4

    trajectory_duration_per_point: float = 0.025
    trajectory_display_points: int = 60
    trajectory_control_points: int = 200

    trajectory_center: Optional[np.ndarray] = None
    trajectory_size: float = 0.12
    trajectory_height: float = 0.00
    trajectory_orientation: str = "vertical_xz"

    joint_limits: Optional[List[Tuple[float, float]]] = None

    def __post_init__(self) -> None:
        if self.joint_limits is None:
            self.joint_limits = [
                (-2.8973, 2.8973),
                (-1.7628, 1.7628),
                (-2.8973, 2.8973),
                (-3.0718, -0.0698),
                (-2.8973, 2.8973),
                (-0.0175, 3.7525),
                (-2.8973, 2.8973),
            ]

        if self.trajectory_center is None:
            self.trajectory_center = np.array([0.45, 0.0, 0.45], dtype=float)


# ============================================
# 工具函数
# ============================================
def format_pos(pos: np.ndarray) -> str:
    """将三维位置向量格式化为字符串。"""
    return (
        f"({float(pos[0]):.3f}, "
        f"{float(pos[1]):.3f}, "
        f"{float(pos[2]):.3f})"
    )


# ============================================
# 倒 8 字形轨迹生成器
# ============================================
class Figure8Trajectory:
    """倒 8 字形轨迹生成器。"""

    def __init__(
        self,
        center: np.ndarray,
        size: float = 0.12,
        height: float = 0.04,
        orientation: str = "horizontal_xy",
    ) -> None:
        self.center = np.asarray(center, dtype=float)
        self.size = float(size)
        self.height = float(height)
        self.orientation = orientation

    def get_position(self, t: float) -> np.ndarray:
        """根据参数 t 获取轨迹上的位置，t 的范围为 [0, 1]。"""
        theta = 2.0 * np.pi * float(t)

        horizontal = self.size * np.sin(theta)
        cross = self.size * 0.5 * np.sin(2.0 * theta)
        vertical = self.height * np.sin(theta)

        if self.orientation == "horizontal_xy":
            x_pos = self.center[0] + horizontal
            y_pos = self.center[1] + cross
            z_pos = self.center[2] + vertical
        elif self.orientation == "vertical_xz":
            x_pos = self.center[0] + horizontal
            y_pos = self.center[1] + vertical
            z_pos = self.center[2] + cross
        elif self.orientation == "vertical_yz":
            x_pos = self.center[0] + vertical
            y_pos = self.center[1] + horizontal
            z_pos = self.center[2] + cross
        else:
            x_pos = self.center[0] + horizontal
            y_pos = self.center[1] + cross
            z_pos = self.center[2] + vertical

        return np.array([x_pos, y_pos, z_pos], dtype=float)

    def generate_trajectory(self, num_points: int = 200) -> List[np.ndarray]:
        """生成离散轨迹点。"""
        points: List[np.ndarray] = []

        for index in range(num_points + 1):
            t = float(index) / float(num_points)
            points.append(self.get_position(t))

        return points


# ============================================
# Panda 机械臂
# ============================================
class PandaRobot:
    """Panda 机械臂控制类。"""

    def __init__(self, config: RobotConfig) -> None:
        self.config = config

        self.model: Optional[mujoco.MjModel] = None
        self.data: Optional[mujoco.MjData] = None

        self.arm_joint_names = [
            "joint1",
            "joint2",
            "joint3",
            "joint4",
            "joint5",
            "joint6",
            "joint7",
        ]

        self.finger_joint_names = [
            "finger_joint1",
            "finger_joint2",
        ]

        self.arm_indices: List[int] = []
        self.finger_indices: List[int] = []

        self.left_finger_id: Optional[int] = None
        self.right_finger_id: Optional[int] = None

        self.home_joints = {
            "joint1": 0.0,
            "joint2": -0.785,
            "joint3": 0.0,
            "joint4": -2.356,
            "joint5": 0.0,
            "joint6": 1.571,
            "joint7": 0.785,
        }

    def _require_model_data(self) -> Tuple[mujoco.MjModel, mujoco.MjData]:
        """检查模型和数据是否已加载，并返回 model 和 data。"""
        if self.model is None or self.data is None:
            raise RuntimeError("MuJoCo 模型尚未加载，请先调用 load_model()。")
        return self.model, self.data

    def _require_finger_ids(self) -> Tuple[int, int]:
        """返回左右手指的 body ID。"""
        if self.left_finger_id is None or self.right_finger_id is None:
            raise RuntimeError("手指 body ID 尚未初始化。")
        return self.left_finger_id, self.right_finger_id

    def load_model(self) -> None:
        """加载 MuJoCo 模型。"""
        self.model = mujoco.MjModel.from_xml_path(self.config.model_path)
        self.data = mujoco.MjData(self.model)
        print("✓ 模型加载成功")

    def setup_joint_indices(self) -> None:
        """设置手臂关节和手指关节的索引。"""
        model, _ = self._require_model_data()

        self.arm_indices.clear()
        self.finger_indices.clear()

        for name in self.arm_joint_names:
            joint_id = model.joint(name).id
            self.arm_indices.append(int(model.jnt_qposadr[joint_id]))

        for name in self.finger_joint_names:
            joint_id = model.joint(name).id
            self.finger_indices.append(int(model.jnt_qposadr[joint_id]))

        self.left_finger_id = int(model.body("left_finger").id)
        self.right_finger_id = int(model.body("right_finger").id)

        print(f"✓ 手臂关节数: {len(self.arm_indices)}")
        print(f"✓ 找到左手指: left_finger, id={self.left_finger_id}")
        print(f"✓ 找到右手指: right_finger, id={self.right_finger_id}")

    def set_home_position(self) -> None:
        """将机械臂移动到 Home 初始姿态。"""
        model, data = self._require_model_data()

        for joint_name, joint_pos in self.home_joints.items():
            joint_id = model.joint(joint_name).id
            position_address = model.jnt_qposadr[joint_id]
            data.qpos[position_address] = float(joint_pos)

        for index in self.finger_indices:
            data.qpos[index] = 0.0

        mujoco.mj_forward(model, data)
        print("✓ 设置 Home 位置完成")

    def get_end_effector_position(self) -> np.ndarray:
        """返回末端执行器中心位置。"""
        _, data = self._require_model_data()
        left_finger_id, right_finger_id = self._require_finger_ids()

        left_pos = data.body(left_finger_id).xpos.copy()
        right_pos = data.body(right_finger_id).xpos.copy()

        return np.asarray((left_pos + right_pos) / 2.0, dtype=float)

    def forward_kinematics(self, joint_angles: np.ndarray) -> np.ndarray:
        """根据关节角计算末端执行器位置。"""
        model, data = self._require_model_data()
        joint_angles = np.asarray(joint_angles, dtype=float)

        saved_positions = data.qpos.copy()
        saved_velocities = data.qvel.copy()

        data.qpos[self.arm_indices] = joint_angles[:7]
        data.qpos[self.finger_indices] = [0.0, 0.0]

        mujoco.mj_forward(model, data)

        position = self.get_end_effector_position()

        data.qpos[:] = saved_positions
        data.qvel[:] = saved_velocities

        mujoco.mj_forward(model, data)

        return position

    def get_current_joint_angles(self) -> np.ndarray:
        """返回当前机械臂关节角。"""
        _, data = self._require_model_data()
        return np.asarray(data.qpos[self.arm_indices].copy(), dtype=float)

    def apply_joint_angles(self, joint_angles: np.ndarray) -> None:
        """将关节角应用到机械臂模型中。"""
        model, data = self._require_model_data()
        joint_angles = np.asarray(joint_angles, dtype=float)

        for index, joint_index in enumerate(self.arm_indices):
            data.qpos[joint_index] = float(joint_angles[index])

        data.qpos[self.finger_indices] = [0.0, 0.0]
        data.qvel[self.arm_indices] = 0.0

        mujoco.mj_forward(model, data)


# ============================================
# 逆运动学求解器
# ============================================
class InverseKinematicsSolver:
    """数值逆运动学求解器。"""

    def __init__(self, robot: PandaRobot, config: RobotConfig) -> None:
        self.robot = robot
        self.config = config

    def _ik_cost(self, joint_angles: np.ndarray, target_pos: np.ndarray) -> float:
        """逆运动学优化的代价函数。"""
        joint_angles = np.asarray(joint_angles, dtype=float)
        target_pos = np.asarray(target_pos, dtype=float)

        current_pos = self.robot.forward_kinematics(joint_angles)
        return float(np.linalg.norm(current_pos - target_pos))

    def _get_bounds(self) -> List[Tuple[float, float]]:
        """返回各关节的角度约束范围。"""
        model, _ = self.robot._require_model_data()

        if self.config.joint_limits is None:
            raise RuntimeError("关节限位尚未初始化。")

        bounds: List[Tuple[float, float]] = []

        for index, joint_name in enumerate(self.robot.arm_joint_names):
            joint_id = model.joint(joint_name).id
            joint_range = model.jnt_range[joint_id]

            if float(joint_range[0]) == 0.0 and float(joint_range[1]) == 0.0:
                bounds.append(self.config.joint_limits[index])
            else:
                bounds.append((float(joint_range[0]), float(joint_range[1])))

        return bounds

    def solve(
        self,
        target_pos: np.ndarray,
        initial_guess: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, float]:
        """求解逆运动学。"""
        target_pos = np.asarray(target_pos, dtype=float)

        if initial_guess is None:
            initial_guess = self.robot.get_current_joint_angles()
        else:
            initial_guess = np.asarray(initial_guess, dtype=float)

        best_solution: Optional[np.ndarray] = None
        best_error = float("inf")

        for attempt in range(self.config.ik_max_attempts):
            if attempt > 0:
                noise = np.random.uniform(-0.1, 0.1, size=len(initial_guess))
                x0 = initial_guess + noise
            else:
                x0 = initial_guess.copy()

            # 忽略 PyCharm 对 scipy.optimize.minimize 参数类型的误报
            # noinspection PyTypeChecker
            result: OptimizeResult = minimize(
                fun=self._ik_cost,
                x0=np.asarray(x0, dtype=float),
                args=(target_pos,),
                method="L-BFGS-B",
                bounds=self._get_bounds(),
                options={
                    "maxiter": int(self.config.ik_max_iter),
                    "ftol": 1e-6,
                },
            )

            result_error = float(result.fun)

            if result_error < best_error:
                best_error = result_error
                best_solution = np.asarray(result.x, dtype=float)

            if best_error < self.config.ik_tolerance:
                break

        if best_solution is None:
            return initial_guess.copy(), best_error

        return best_solution, best_error


# ============================================
# 运动控制器
# ============================================
class MotionController:
    """带平滑插值的运动控制器。"""

    def __init__(self, robot: PandaRobot, config: RobotConfig) -> None:
        self.robot = robot
        self.config = config

    @staticmethod
    def _normalize_angle(angle: float) -> float:
        """将角度归一化到 [-pi, pi] 范围内。"""
        angle = float(angle)

        while angle > np.pi:
            angle -= 2.0 * np.pi

        while angle < -np.pi:
            angle += 2.0 * np.pi

        return angle

    @staticmethod
    def _smoothstep(t: float) -> float:
        """平滑插值函数。"""
        return t * t * (3.0 - 2.0 * t)

    def move_to_joints(
        self,
        target_joints: np.ndarray,
        viewer_obj: Optional[Any] = None,
    ) -> None:
        """移动到目标关节角。"""
        target_joints = np.asarray(target_joints, dtype=float)
        start_joints = self.robot.get_current_joint_angles()

        print(f"      运动步数: {self.config.move_steps}")

        for step in range(self.config.move_steps + 1):
            t = float(step) / float(self.config.move_steps)
            t_smooth = self._smoothstep(t)

            current_joints = start_joints + t_smooth * (target_joints - start_joints)

            for index in range(len(current_joints)):
                current_joints[index] = self._normalize_angle(float(current_joints[index]))

            self.robot.apply_joint_angles(current_joints)

            if viewer_obj is not None:
                cast(Any, viewer_obj).sync()

            time.sleep(self.config.move_dt)

            if step % 40 == 0:
                percent = float(step) / float(self.config.move_steps) * 100.0
                print(f"\r      运动进度: {percent:.0f}%", end="", flush=True)

        print("\r      运动进度: 100% ✓")

    def move_to_position(
        self,
        target_pos: np.ndarray,
        ik_solver: InverseKinematicsSolver,
        viewer_obj: Optional[Any] = None,
    ) -> Tuple[bool, float]:
        """移动到目标笛卡尔空间位置。"""
        target_pos = np.asarray(target_pos, dtype=float)

        current_angles = self.robot.get_current_joint_angles()
        solution, ik_error = ik_solver.solve(target_pos, current_angles)

        if ik_error >= 0.02:
            return False, float(ik_error)

        self.move_to_joints(solution, viewer_obj)

        actual_pos = self.robot.get_end_effector_position()
        position_error = float(np.linalg.norm(actual_pos - target_pos))

        return True, position_error

    def follow_trajectory(
        self,
        trajectory: List[np.ndarray],
        ik_solver: InverseKinematicsSolver,
        viewer_obj: Optional[Any] = None,
    ) -> None:
        """逐点跟踪轨迹。"""
        print(f"      轨迹点数: {len(trajectory)}")

        success_count = 0
        current_angles = self.robot.get_current_joint_angles()

        for index, target_pos in enumerate(trajectory):
            target_pos = np.asarray(target_pos, dtype=float)

            solution, ik_error = ik_solver.solve(target_pos, current_angles)

            if ik_error < 0.02:
                self.robot.apply_joint_angles(solution)
                current_angles = solution
                success_count += 1

            if viewer_obj is not None:
                cast(Any, viewer_obj).sync()

            time.sleep(self.config.trajectory_duration_per_point)

            if (index + 1) % 50 == 0:
                percent = float(index + 1) / float(len(trajectory)) * 100.0
                print(f"\r      轨迹跟踪进度: {percent:.0f}%", end="", flush=True)

        success_rate = 100.0 * float(success_count) / float(len(trajectory))

        print("\r      轨迹跟踪进度: 100% ✓")
        print(f"      IK 成功率: {success_count}/{len(trajectory)} ({success_rate:.1f}%)")


# ============================================
# 可视化辅助类
# ============================================
class Visualizer:
    """MuJoCo 可视化辅助类。"""

    def __init__(self, robot: PandaRobot) -> None:
        self.robot = robot
        self.viewer: Optional[Any] = None

    def launch(self) -> None:
        """启动可视化窗口；如果启动失败，则继续以无可视化模式运行。"""
        model, data = self.robot._require_model_data()

        try:
            self.viewer = mujoco.viewer.launch_passive(model, data)
            self._setup_camera()
            print("✓ 可视化窗口已启动")
        except Exception as exc:
            self.viewer = None
            print("⚠ 可视化窗口启动失败，程序将以无可视化模式继续运行")
            print(f"  失败原因: {exc}")

    def _setup_camera(self) -> None:
        """设置相机视角。"""
        if self.viewer is None:
            return

        viewer_obj = cast(Any, self.viewer)
        viewer_obj.cam.lookat = np.array([0.2, 0, 0.4], dtype=float)
        viewer_obj.cam.distance = 2.5
        viewer_obj.cam.azimuth = 90
        viewer_obj.cam.elevation = -20

    def add_marker(
        self,
        position: np.ndarray,
        color: List[float],
        radius: float = 0.025,
    ) -> None:
        """添加球形标记点。"""
        if self.viewer is None:
            return

        viewer_obj = cast(Any, self.viewer)
        user_scene = viewer_obj.user_scn
        marker_index = int(user_scene.ngeom)

        if marker_index >= 100:
            print("⚠ 标记点数量已达上限")
            return

        geom = user_scene.geoms[marker_index]

        # 忽略 PyCharm 对 mjv_initGeom 参数列表的误报
        # noinspection PyArgumentList
        mujoco.mjv_initGeom(  # type: ignore[call-arg]
            geom,
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=np.array([radius, 0.0, 0.0], dtype=np.float64),
            pos=np.asarray(position, dtype=np.float64),
            mat=np.eye(3, dtype=np.float64).flatten(),
            rgba=np.array(color + [1.0], dtype=np.float32),
        )

        user_scene.ngeom += 1

    def add_trajectory_line(
        self,
        trajectory: List[np.ndarray],
        color: List[float],
        line_width: float = 0.006,
    ) -> None:
        """使用短圆柱线段绘制轨迹曲线。"""
        if self.viewer is None or len(trajectory) < 2:
            return

        viewer_obj = cast(Any, self.viewer)
        user_scene = viewer_obj.user_scn
        start_index = int(user_scene.ngeom)

        added_count = 0

        for index in range(len(trajectory) - 1):
            geom_index = start_index + index

            if geom_index >= 100:
                print(f"  ⚠ 线段数量已达上限: {geom_index}/100")
                break

            p1 = np.asarray(trajectory[index], dtype=float)
            p2 = np.asarray(trajectory[index + 1], dtype=float)

            center = (p1 + p2) / 2.0
            direction = p2 - p1
            length = float(np.linalg.norm(direction))

            if length < 0.001:
                continue

            direction = direction / length

            z_axis = np.array([0.0, 0.0, 1.0], dtype=float)
            rotation_axis = np.cross(z_axis, direction)
            rotation_axis_norm = float(np.linalg.norm(rotation_axis))

            if rotation_axis_norm < 0.001:
                rotation_matrix = np.eye(3, dtype=float)
            else:
                rotation_axis = rotation_axis / rotation_axis_norm
                angle = float(np.arccos(np.clip(np.dot(z_axis, direction), -1.0, 1.0)))

                k_matrix = np.array(
                    [
                        [0.0, -rotation_axis[2], rotation_axis[1]],
                        [rotation_axis[2], 0.0, -rotation_axis[0]],
                        [-rotation_axis[1], rotation_axis[0], 0.0],
                    ],
                    dtype=float,
                )

                rotation_matrix = (
                    np.eye(3, dtype=float)
                    + np.sin(angle) * k_matrix
                    + (1.0 - np.cos(angle)) * k_matrix @ k_matrix
                )

            geom = user_scene.geoms[geom_index]

            # 忽略 PyCharm 对 mjv_initGeom 参数列表的误报
            # noinspection PyArgumentList
            mujoco.mjv_initGeom(  # type: ignore[call-arg]
                geom,
                type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                size=np.array([line_width, length / 2.0, line_width], dtype=np.float64),
                pos=np.asarray(center, dtype=np.float64),
                mat=np.asarray(rotation_matrix.flatten(), dtype=np.float64),
                rgba=np.array(color + [0.9], dtype=np.float32),
            )

            user_scene.ngeom += 1
            added_count += 1

        print(f"  ✓ 已绘制 {added_count} 条轨迹线段")

    def sync(self) -> None:
        """同步可视化窗口。"""
        if self.viewer is not None:
            cast(Any, self.viewer).sync()

    def is_running(self) -> bool:
        """检查可视化窗口是否仍在运行。"""
        if self.viewer is None:
            return False

        return bool(cast(Any, self.viewer).is_running())

    def close(self) -> None:
        """关闭可视化窗口。"""
        if self.viewer is not None:
            cast(Any, self.viewer).close()
            self.viewer = None


# ============================================
# 主程序
# ============================================
def main() -> None:
    """主函数。"""
    print("\n" + "=" * 60)
    print("Panda 机械臂立式倒 8 字形轨迹跟踪控制程序")
    print("=" * 60)

    config = RobotConfig()
    config.trajectory_orientation = "vertical_xz"

    robot = PandaRobot(config)
    robot.load_model()
    robot.setup_joint_indices()
    robot.set_home_position()

    start_pos = robot.get_end_effector_position()
    print(f"\n初始末端位置: {format_pos(start_pos)}")

    ik_solver = InverseKinematicsSolver(robot, config)
    motion_controller = MotionController(robot, config)
    visualizer = Visualizer(robot)

    visualizer.launch()

    if config.trajectory_center is None:
        raise RuntimeError("轨迹中心尚未初始化。")

    trajectory_generator = Figure8Trajectory(
        center=config.trajectory_center,
        size=config.trajectory_size,
        height=config.trajectory_height,
        orientation=config.trajectory_orientation,
    )

    trajectory_display = trajectory_generator.generate_trajectory(
        config.trajectory_display_points
    )

    trajectory_control = trajectory_generator.generate_trajectory(
        config.trajectory_control_points
    )

    orientation_names = {
        "horizontal_xy": "水平倒 8 字形，XY 平面",
        "vertical_xz": "立式倒 8 字形，XZ 平面",
        "vertical_yz": "侧立式倒 8 字形，YZ 平面",
    }

    print("\n【立式倒 8 字形轨迹参数】")
    print(
        "  轨迹方向: "
        f"{orientation_names.get(config.trajectory_orientation, config.trajectory_orientation)}"
    )
    print(f"  轨迹中心: {format_pos(config.trajectory_center)}")
    print(f"  轨迹大小: {config.trajectory_size:.3f} m")
    print(f"  高度波动: {config.trajectory_height:.3f} m")
    print(f"  显示点数: {config.trajectory_display_points}")
    print(f"  控制点数: {config.trajectory_control_points}")

    print("\n【添加轨迹标记】")
    visualizer.add_trajectory_line(
        trajectory=trajectory_display,
        color=[1.0, 0.8, 0.0],
        line_width=0.006,
    )

    visualizer.add_marker(
        position=config.trajectory_center,
        color=[1.0, 0.0, 0.0],
        radius=0.012,
    )

    visualizer.sync()

    print("\n  金色曲线 = 立式倒 8 字形轨迹路径")
    print("  红色小球 = 轨迹中心")

    print("\n【步骤 1】移动到轨迹起点...")
    success, error = motion_controller.move_to_position(
        target_pos=trajectory_control[0],
        ik_solver=ik_solver,
        viewer_obj=visualizer.viewer,
    )

    if success:
        print(f"  ✓ 到达轨迹起点，误差: {error * 1000.0:.2f} mm")
    else:
        print("  ⚠ 移动到起点失败")

    time.sleep(1.0)

    print("\n【步骤 2】跟踪立式倒 8 字形轨迹...")
    print("-" * 60)

    motion_controller.follow_trajectory(
        trajectory=trajectory_control,
        ik_solver=ik_solver,
        viewer_obj=visualizer.viewer,
    )

    print("\n" + "=" * 60)
    print("✓ 立式倒 8 字形轨迹跟踪完成")
    print("=" * 60)

    final_pos = robot.get_end_effector_position()
    print(f"\n最终末端位置: {format_pos(final_pos)}")

    if visualizer.viewer is not None:
        print("\n关闭窗口退出...")

        try:
            while visualizer.is_running():
                visualizer.sync()
                time.sleep(0.01)
        except KeyboardInterrupt:
            print("\n程序退出")
        finally:
            visualizer.close()
    else:
        print("\n无可视化模式运行结束。")


if __name__ == "__main__":
    main()