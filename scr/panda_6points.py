"""
Panda机械臂6点目标点位控制程序
功能: 实现机械臂末端执行器的6个空间目标点位的逆运动学求解和运动控制
"""

import os
import time
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

import mujoco
import mujoco.viewer
import numpy as np
from scipy.optimize import OptimizeResult, minimize


# 设置 MuJoCo 渲染后端
os.environ["MUJOCO_GL"] = "glfw"


# ============================================
# 配置参数类
# ============================================
@dataclass
class RobotConfig:
    """机器人配置参数"""

    # 模型路径
    model_path: str = r"D:\Mujoco\mujoco_menagerie-main\franka_emika_panda\scene.xml"

    # 运动控制参数
    move_steps: int = 200
    move_dt: float = 0.01
    pause_time: float = 1.0

    # IK 求解参数
    ik_max_attempts: int = 3
    ik_max_iter: int = 200
    ik_tolerance: float = 1e-4

    # Panda 机械臂关节限位，单位为 rad
    joint_limits: Optional[List[Tuple[float, float]]] = None

    def __post_init__(self) -> None:
        """初始化默认关节限位"""
        if self.joint_limits is None:
            self.joint_limits = [
                (-2.8973, 2.8973),    # joint1
                (-1.7628, 1.7628),    # joint2
                (-2.8973, 2.8973),    # joint3
                (-3.0718, -0.0698),   # joint4
                (-2.8973, 2.8973),    # joint5
                (-0.0175, 3.7525),    # joint6
                (-2.8973, 2.8973),    # joint7
            ]


@dataclass
class TargetPoint:
    """目标点数据结构"""

    name: str
    pos: np.ndarray
    color: List[float]


# ============================================
# 工具函数
# ============================================
def format_pos(pos: np.ndarray) -> str:
    """
    将三维位置格式化为字符串。

    Args:
        pos: 位置向量 [x, y, z]

    Returns:
        格式化字符串
    """
    return (
        f"({float(pos[0]):.3f}, "
        f"{float(pos[1]):.3f}, "
        f"{float(pos[2]):.3f})"
    )


# ============================================
# Panda 机械臂类
# ============================================
class PandaRobot:
    """Panda机械臂控制类"""

    def __init__(self, config: RobotConfig) -> None:
        self.config = config

        self.model: Optional[mujoco.MjModel] = None
        self.data: Optional[mujoco.MjData] = None

        # 关节名称
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

        # 关节索引
        self.arm_indices: List[int] = []
        self.finger_indices: List[int] = []

        # 手指 body ID
        self.left_finger_id: Optional[int] = None
        self.right_finger_id: Optional[int] = None

        # Home 位置配置
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
        """
        确保模型和数据已经加载。

        Returns:
            model, data
        Raises:
            RuntimeError: 如果模型尚未加载
        """
        if self.model is None or self.data is None:
            raise RuntimeError("MuJoCo 模型尚未加载，请先调用 load_model()。")
        return self.model, self.data

    def _require_finger_ids(self) -> Tuple[int, int]:
        """
        确保左右手指 body ID 已经初始化。
        Returns:
            left_finger_id, right_finger_id
        """
        if self.left_finger_id is None or self.right_finger_id is None:
            raise RuntimeError("手指 body ID 尚未初始化，请先调用 setup_joint_indices()。")
        return self.left_finger_id, self.right_finger_id

    def load_model(self) -> None:
        """加载 MuJoCo 模型"""
        self.model = mujoco.MjModel.from_xml_path(self.config.model_path)
        self.data = mujoco.MjData(self.model)
        print("✓ 模型加载成功")

    def setup_joint_indices(self) -> None:
        """设置手臂和手指关节索引"""
        model, _ = self._require_model_data()

        self.arm_indices.clear()
        self.finger_indices.clear()

        # 手臂关节索引
        for name in self.arm_joint_names:
            joint_id = model.joint(name).id
            self.arm_indices.append(int(model.jnt_qposadr[joint_id]))

        # 手指关节索引
        for name in self.finger_joint_names:
            joint_id = model.joint(name).id
            self.finger_indices.append(int(model.jnt_qposadr[joint_id]))

        # 手指 body ID
        self.left_finger_id = int(model.body("left_finger").id)
        self.right_finger_id = int(model.body("right_finger").id)

        print(f"✓ 手臂关节数: {len(self.arm_indices)}")
        print(f"✓ 找到左手指: left_finger (id={self.left_finger_id})")
        print(f"✓ 找到右手指: right_finger (id={self.right_finger_id})")

    def set_home_position(self) -> None:
        """设置机械臂到 Home 位置"""
        model, data = self._require_model_data()

        for joint_name, joint_pos in self.home_joints.items():
            joint_id = model.joint(joint_name).id
            qpos_addr = model.jnt_qposadr[joint_id]
            data.qpos[qpos_addr] = float(joint_pos)

        # 手指初始闭合，张度为 0
        for idx in self.finger_indices:
            data.qpos[idx] = 0.0

        mujoco.mj_forward(model, data)
        print("✓ 设置 Home 位置完成")

    def get_end_effector_position(self) -> np.ndarray:
        """
        获取末端执行器位置。
        取左右手指 body 中心位置的平均值，作为末端执行器中心点。
        Returns:
            末端位置 [x, y, z]
        """
        _, data = self._require_model_data()
        left_finger_id, right_finger_id = self._require_finger_ids()

        left_pos = data.body(left_finger_id).xpos.copy()
        right_pos = data.body(right_finger_id).xpos.copy()

        return np.asarray((left_pos + right_pos) / 2.0, dtype=float)

    def forward_kinematics(self, joint_angles: np.ndarray) -> np.ndarray:
        """
        正向运动学。

        Args:
            joint_angles: 7 个关节角度

        Returns:
            末端执行器位置 [x, y, z]
        """
        model, data = self._require_model_data()

        joint_angles = np.asarray(joint_angles, dtype=float)

        # 保存当前状态
        saved_qpos = data.qpos.copy()
        saved_qvel = data.qvel.copy()

        # 设置测试关节角度
        data.qpos[self.arm_indices] = joint_angles[:7]
        data.qpos[self.finger_indices] = [0.0, 0.0]

        # 更新正向运动学
        mujoco.mj_forward(model, data)

        # 获取末端位置
        position = self.get_end_effector_position()

        # 恢复状态
        data.qpos[:] = saved_qpos
        data.qvel[:] = saved_qvel
        mujoco.mj_forward(model, data)

        return position

    def get_current_joint_angles(self) -> np.ndarray:
        """
        获取当前手臂关节角度。

        Returns:
            当前 7 个关节角度
        """
        _, data = self._require_model_data()
        return np.asarray(data.qpos[self.arm_indices].copy(), dtype=float)

    def apply_joint_angles(self, joint_angles: np.ndarray) -> None:
        """
        应用关节角度到机器人。

        Args:
            joint_angles: 目标关节角度
        """
        model, data = self._require_model_data()

        joint_angles = np.asarray(joint_angles, dtype=float)

        for i, idx in enumerate(self.arm_indices):
            data.qpos[idx] = float(joint_angles[i])

        # 本实验不控制夹爪开合，保持闭合状态
        data.qpos[self.finger_indices] = [0.0, 0.0]

        # 清零手臂速度
        data.qvel[self.arm_indices] = 0.0

        mujoco.mj_forward(model, data)


# ============================================
# 逆运动学求解器类
# ============================================
class InverseKinematicsSolver:
    """逆运动学求解器，使用数值优化方法"""

    def __init__(self, robot: PandaRobot, config: RobotConfig) -> None:
        self.robot = robot
        self.config = config

    def _ik_cost(self, joint_angles: np.ndarray, target_pos: np.ndarray) -> float:
        """
        IK 成本函数。

        成本定义为当前末端位置与目标位置之间的欧氏距离。

        Args:
            joint_angles: 当前优化变量，即 7 个关节角
            target_pos: 目标末端位置

        Returns:
            位置误差
        """
        joint_angles = np.asarray(joint_angles, dtype=float)
        target_pos = np.asarray(target_pos, dtype=float)

        current_pos = self.robot.forward_kinematics(joint_angles)
        return float(np.linalg.norm(current_pos - target_pos))

    def _get_bounds(self) -> List[Tuple[float, float]]:
        """
        获取关节边界约束。

        Returns:
            每个关节的上下限
        """
        model, _ = self.robot._require_model_data()

        bounds: List[Tuple[float, float]] = []

        if self.config.joint_limits is None:
            raise RuntimeError("joint_limits 未初始化。")

        for i, qpos_idx in enumerate(self.robot.arm_indices):

            joint_name = self.robot.arm_joint_names[i]
            joint_id = model.joint(joint_name).id
            joint_range = model.jnt_range[joint_id]

            # 如果模型没有定义有效限位，则使用配置中的 Panda 默认限位
            if float(joint_range[0]) == 0.0 and float(joint_range[1]) == 0.0:
                bounds.append(self.config.joint_limits[i])
            else:
                bounds.append((float(joint_range[0]), float(joint_range[1])))

        return bounds

    def solve(
        self,
        target_pos: np.ndarray,
        initial_guess: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, float]:
        """
        求解逆运动学。

        Args:
            target_pos: 目标位置 [x, y, z]
            initial_guess: 初始猜测关节角度，为 None 时使用当前关节角度

        Returns:
            关节角度解和位置误差
        """
        target_pos = np.asarray(target_pos, dtype=float)

        if initial_guess is None:
            initial_guess = self.robot.get_current_joint_angles()
        else:
            initial_guess = np.asarray(initial_guess, dtype=float)

        best_solution: Optional[np.ndarray] = None
        best_error: float = float("inf")

        for attempt in range(self.config.ik_max_attempts):
            # 第一次使用当前角度，后续尝试添加随机扰动
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
# 运动控制器类
# ============================================
class MotionController:
    """运动控制器，负责平滑插值运动"""

    def __init__(self, robot: PandaRobot, config: RobotConfig) -> None:
        self.robot = robot
        self.config = config

    @staticmethod
    def _normalize_angle(angle: float) -> float:
        """
        将角度归一化到 [-pi, pi]。

        Args:
            angle: 输入角度

        Returns:
            归一化后的角度
        """
        angle = float(angle)

        while angle > np.pi:
            angle -= 2.0 * np.pi

        while angle < -np.pi:
            angle += 2.0 * np.pi

        return angle

    @staticmethod
    def _smoothstep(t: float) -> float:
        """
        平滑步进函数。

        Args:
            t: 归一化时间，范围 [0, 1]

        Returns:
            平滑插值系数
        """
        return t * t * (3.0 - 2.0 * t)

    def move_to_joints(
        self,
        target_joints: np.ndarray,
        viewer_obj: Optional[Any] = None,
    ) -> None:
        """
        平滑移动到目标关节角度。

        Args:
            target_joints: 目标关节角度
            viewer_obj: MuJoCo 可视化对象
        """
        target_joints = np.asarray(target_joints, dtype=float)
        start_joints = self.robot.get_current_joint_angles()

        print(f"      运动步数: {self.config.move_steps}")

        for step in range(self.config.move_steps + 1):
            t = float(step) / float(self.config.move_steps)
            t_smooth = self._smoothstep(t)

            # 线性插值 + S 曲线时间缩放
            current_joints = start_joints + t_smooth * (target_joints - start_joints)

            # 角度归一化
            for i in range(len(current_joints)):
                current_joints[i] = self._normalize_angle(float(current_joints[i]))

            # 应用到机器人
            self.robot.apply_joint_angles(current_joints)

            # 同步视图
            if viewer_obj is not None:
                viewer_obj.sync()

            # 控制运动速度
            time.sleep(self.config.move_dt)

            # 打印进度
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
        """
        移动到目标空间位置。

        Args:
            target_pos: 目标位置 [x, y, z]
            ik_solver: 逆运动学求解器
            viewer_obj: 可视化对象

        Returns:
            是否成功，以及最终定位误差
        """
        target_pos = np.asarray(target_pos, dtype=float)

        # 求解逆运动学
        current_angles = self.robot.get_current_joint_angles()
        solution, ik_error = ik_solver.solve(target_pos, current_angles)

        # 误差大于 2 cm 认为 IK 失败
        if ik_error >= 0.02:
            return False, float(ik_error)

        # 执行运动
        self.move_to_joints(solution, viewer_obj)

        # 验证最终位置
        actual_pos = self.robot.get_end_effector_position()
        position_error = float(np.linalg.norm(actual_pos - target_pos))

        return True, position_error


# ============================================
# 可视化辅助类
# ============================================
class Visualizer:
    """MuJoCo 可视化辅助类"""

    def __init__(self, robot: PandaRobot) -> None:
        self.robot = robot
        self.viewer: Optional[Any] = None

    def launch(self) -> None:
        """启动可视化窗口"""
        model, data = self.robot._require_model_data()
        self.viewer = mujoco.viewer.launch_passive(model, data)
        self._setup_camera()
        print("✓ 可视化窗口已启动")

    def _setup_camera(self) -> None:
        """设置相机视角"""
        if self.viewer is None:
            return

        self.viewer.cam.lookat = np.array([0.4, 0.0, 0.4], dtype=float)
        self.viewer.cam.distance = 2.5
        self.viewer.cam.azimuth = 145
        self.viewer.cam.elevation = -15

    def add_marker(
        self,
        position: np.ndarray,
        color: List[float],
        radius: float = 0.025,
    ) -> None:
        """
        添加 3D 标记点。

        Args:
            position: 目标点位置 [x, y, z]
            color: RGB 颜色 [r, g, b]
            radius: 标记点半径
        """
        if self.viewer is None:
            return

        position = np.asarray(position, dtype=np.float64)

        user_scene = self.viewer.user_scn
        marker_idx = int(user_scene.ngeom)

        if marker_idx >= 100:
            print("⚠ 标记点数量已达上限 100")
            return

        geom = user_scene.geoms[marker_idx]

        # 这里的写法是 MuJoCo Python API 中常用的方式。
        mujoco.mjv_initGeom(  # type: ignore[call-arg]
            geom,
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=np.array([radius, 0.0, 0.0], dtype=np.float64),
            pos=position,
            mat=np.eye(3, dtype=np.float64).flatten(),
            rgba=np.array(color + [1.0], dtype=np.float32),
        )

        user_scene.ngeom += 1

    def sync(self) -> None:
        """同步可视化窗口"""
        if self.viewer is not None:
            self.viewer.sync()

    def is_running(self) -> bool:
        """
        检查可视化窗口是否仍在运行。

        Returns:
            True 表示窗口仍在运行
        """
        return self.viewer is not None and bool(self.viewer.is_running())

    def close(self) -> None:
        """关闭可视化窗口"""
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None


# ============================================
# 主程序
# ============================================
def main() -> None:
    """主函数"""
    print("\n" + "=" * 60)
    print("Panda机械臂6点目标点位控制程序")
    print("=" * 60)

    # 1. 初始化配置
    config = RobotConfig()

    # 2. 初始化机器人
    robot = PandaRobot(config)
    robot.load_model()
    robot.setup_joint_indices()
    robot.set_home_position()

    # 3. 获取初始位置
    start_pos = robot.get_end_effector_position()
    print(f"\n初始末端位置: {format_pos(start_pos)}")

    # 4. 初始化组件
    ik_solver = InverseKinematicsSolver(robot, config)
    motion_ctrl = MotionController(robot, config)
    visualizer = Visualizer(robot)

    # 5. 启动可视化
    visualizer.launch()

    # 6. 定义 6 个目标点
    target_points = [
        TargetPoint(
            name="点1-前方",
            pos=np.array([0.50, 0.00, 0.40], dtype=float),
            color=[1.0, 0.0, 0.0],
        ),
        TargetPoint(
            name="点2-右前方",
            pos=np.array([0.45, 0.25, 0.45], dtype=float),
            color=[0.0, 1.0, 0.0],
        ),
        TargetPoint(
            name="点3-左前方",
            pos=np.array([0.45, -0.25, 0.45], dtype=float),
            color=[0.0, 0.0, 1.0],
        ),
        TargetPoint(
            name="点4-上方",
            pos=np.array([0.40, 0.00, 0.55], dtype=float),
            color=[1.0, 1.0, 0.0],
        ),
        TargetPoint(
            name="点5-右下方",
            pos=np.array([0.35, 0.20, 0.35], dtype=float),
            color=[1.0, 0.0, 1.0],
        ),
        TargetPoint(
            name="点6-左下方",
            pos=np.array([0.35, -0.20, 0.35], dtype=float),
            color=[0.0, 1.0, 1.0],
        ),
    ]

    # 7. 添加标记点
    print("\n【添加目标点标记】")
    for i, point in enumerate(target_points, 1):
        visualizer.add_marker(point.pos, point.color, radius=0.025)
        print(f"  目标点{i}: {point.name} at {format_pos(point.pos)}")

    visualizer.add_marker(start_pos, [0.5, 0.5, 0.5], radius=0.02)
    print(f"  起点: 初始位置 at {format_pos(start_pos)}")

    visualizer.sync()

    print("\n等待 2 秒后开始运动...")
    time.sleep(2.0)

    # 8. 依次运动到每个目标点
    print("\n" + "=" * 60)
    print("开始执行运动控制")
    print("=" * 60)

    for i, point in enumerate(target_points, 1):
        print(f"\n--- 目标点 {i}: {point.name} ---")
        print(f"  目标位置: {format_pos(point.pos)}")

        success, error = motion_ctrl.move_to_position(
            target_pos=point.pos,
            ik_solver=ik_solver,
            viewer_obj=visualizer.viewer,
        )

        if success:
            actual_pos = robot.get_end_effector_position()
            print(f"  ✓ 到达目标点 {i}")
            print(f"    实际位置: {format_pos(actual_pos)}")
            print(f"    定位误差: {error * 1000.0:.2f} mm")
        else:
            print(f"  ⚠ 运动失败，IK误差: {error * 1000.0:.2f} mm")

        # 点间暂停
        if i < len(target_points):
            print(f"  暂停 {config.pause_time} 秒...")
            time.sleep(config.pause_time)

    # 9. 返回 Home 位置
    print("\n" + "=" * 60)
    print("✓ 所有目标点运动完成!")
    print("=" * 60)

    time.sleep(1.0)

    print("\n返回 Home 位置...")
    home_target = np.array(
        [robot.home_joints[name] for name in robot.arm_joint_names],
        dtype=float,
    )

    motion_ctrl.move_to_joints(
        target_joints=home_target,
        viewer_obj=visualizer.viewer,
    )

    print("✓ 已返回 Home 位置")

    # 10. 保持窗口打开
    print("\n关闭窗口退出...")

    try:
        while visualizer.is_running():
            visualizer.sync()
            time.sleep(0.01)
    except KeyboardInterrupt:
        print("\n程序退出")
    finally:
        visualizer.close()


if __name__ == "__main__":
    main()