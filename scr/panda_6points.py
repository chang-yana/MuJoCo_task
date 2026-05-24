"""
Panda机械臂6点目标点位控制程序
功能: 实现机械臂末端执行器的6个空间目标点位的逆运动学求解和运动控制
"""
import os
os.environ['MUJOCO_GL'] = 'glfw'

import mujoco
import mujoco.viewer
import numpy as np
import time
from scipy.optimize import minimize
from dataclasses import dataclass
from typing import List, Tuple, Optional


# ============================================
# 配置参数类
# ============================================
@dataclass
class RobotConfig:
    """机器人配置参数"""
    # 模型路径
    model_path: str = r"D:\Mujoco\mujoco_menagerie-main\franka_emika_panda\scene.xml"

    # 运动控制参数
    move_steps: int = 200  # 运动插值步数
    move_dt: float = 0.01  # 每步时间间隔(秒)
    pause_time: float = 1.0  # 点间暂停时间(秒)

    # IK求解参数
    ik_max_attempts: int = 3  # 最大尝试次数
    ik_max_iter: int = 200  # 最大迭代次数
    ik_tolerance: float = 1e-4  # IK收敛精度(m)

    # Panda机械臂关节限位(弧度)
    joint_limits: List[Tuple[float, float]] = None

    def __post_init__(self):
        if self.joint_limits is None:
            self.joint_limits = [
                (-2.8973, 2.8973),  # joint1
                (-1.7628, 1.7628),  # joint2
                (-2.8973, 2.8973),  # joint3
                (-3.0718, -0.0698),  # joint4
                (-2.8973, 2.8973),  # joint5
                (-0.0175, 3.7525),  # joint6
                (-2.8973, 2.8973)  # joint7
            ]


# ============================================
# Panda机械臂类
# ============================================
class PandaRobot:
    """Panda机械臂控制类"""

    def __init__(self, config: RobotConfig):
        self.config = config
        self.model = None
        self.data = None

        # 关节信息
        self.arm_joint_names = ["joint1", "joint2", "joint3", "joint4",
                                "joint5", "joint6", "joint7"]
        self.finger_joint_names = ["finger_joint1", "finger_joint2"]

        self.arm_indices = []
        self.finger_indices = []
        self.left_finger_id = None
        self.right_finger_id = None

        # Home位置配置(标准工作姿态)
        self.home_joints = {
            "joint1": 0.0,
            "joint2": -0.785,  # -45度
            "joint3": 0.0,
            "joint4": -2.356,  # -135度
            "joint5": 0.0,
            "joint6": 1.571,  # 90度
            "joint7": 0.785,  # 45度
        }

    def load_model(self) -> None:
        """加载URDF模型"""
        self.model = mujoco.MjModel.from_xml_path(self.config.model_path)
        self.data = mujoco.MjData(self.model)
        print("✓ 模型加载成功")

    def setup_joint_indices(self) -> None:
        """设置关节索引"""
        # 手臂关节索引
        for name in self.arm_joint_names:
            joint_id = self.model.joint(name).id
            self.arm_indices.append(self.model.jnt_qposadr[joint_id])

        # 手指关节索引
        for name in self.finger_joint_names:
            joint_id = self.model.joint(name).id
            self.finger_indices.append(self.model.jnt_qposadr[joint_id])

        # 手指body ID
        self.left_finger_id = self.model.body("left_finger").id
        self.right_finger_id = self.model.body("right_finger").id

        print(f"✓ 手臂关节数: {len(self.arm_indices)}")
        print(f"✓ 找到左手指: left_finger (id={self.left_finger_id})")
        print(f"✓ 找到右手指: right_finger (id={self.right_finger_id})")

    def set_home_position(self) -> None:
        """设置机械臂到Home位置(标准工作姿态)"""
        for joint_name, joint_pos in self.home_joints.items():
            joint_id = self.model.joint(joint_name).id
            qpos_addr = self.model.jnt_qposadr[joint_id]
            self.data.qpos[qpos_addr] = joint_pos

        # 手指微张
        for idx in self.finger_indices:
            self.data.qpos[idx] = -0.02

        mujoco.mj_forward(self.model, self.data)
        print("✓ 设置Home位置完成")

    def get_end_effector_position(self) -> np.ndarray:
        """获取末端执行器位置(两手指中心点)"""
        left_pos = self.data.body(self.left_finger_id).xpos.copy()
        right_pos = self.data.body(self.right_finger_id).xpos.copy()
        return (left_pos + right_pos) / 2

    def forward_kinematics(self, joint_angles: np.ndarray) -> np.ndarray:
        """
        正向运动学: 给定关节角度,返回末端位置
        Args:
            joint_angles: 7个关节角度
        Returns:
            末端执行器位置 [x, y, z]
        """
        # 保存当前状态
        saved_qpos = self.data.qpos.copy()
        saved_qvel = self.data.qvel.copy()

        # 设置测试关节角度
        self.data.qpos[self.arm_indices] = joint_angles[:7]
        self.data.qpos[self.finger_indices] = [-0.02, -0.02]

        # 正向动力学更新
        mujoco.mj_forward(self.model, self.data)

        # 获取末端位置
        position = self.get_end_effector_position()

        # 恢复状态
        self.data.qpos[:] = saved_qpos
        self.data.qvel[:] = saved_qvel
        mujoco.mj_forward(self.model, self.data)

        return position

    def get_current_joint_angles(self) -> np.ndarray:
        """获取当前关节角度"""
        return self.data.qpos[self.arm_indices].copy()

    def apply_joint_angles(self, joint_angles: np.ndarray) -> None:
        """应用关节角度到机器人"""
        for i, idx in enumerate(self.arm_indices):
            self.data.qpos[idx] = float(joint_angles[i])
        self.data.qpos[self.finger_indices] = [-0.02, -0.02]
        self.data.qvel[self.arm_indices] = 0
        mujoco.mj_forward(self.model, self.data)


# ============================================
# 逆运动学求解器类
# ============================================
class InverseKinematicsSolver:
    """逆运动学求解器(使用数值优化)"""

    def __init__(self, robot: PandaRobot, config: RobotConfig):
        self.robot = robot
        self.config = config

    def _ik_cost(self, joint_angles: np.ndarray, target_pos: np.ndarray) -> float:
        """IK成本函数: 末端位置与目标位置的欧氏距离"""
        current_pos = self.robot.forward_kinematics(joint_angles)
        return float(np.linalg.norm(current_pos - target_pos))

    def _get_bounds(self) -> List[Tuple[float, float]]:
        """获取关节边界约束"""
        bounds = []
        for i, idx in enumerate(self.robot.arm_indices):
            joint_range = self.robot.model.jnt_range[idx]
            # 如果模型未定义限位,使用配置中的限位
            if joint_range[0] == 0 and joint_range[1] == 0:
                bounds.append(self.config.joint_limits[i])
            else:
                bounds.append((float(joint_range[0]), float(joint_range[1])))
        return bounds

    def solve(self, target_pos: np.ndarray,
              initial_guess: Optional[np.ndarray] = None) -> Tuple[np.ndarray, float]:
        """
        求解逆运动学
        Args:
            target_pos: 目标位置 [x, y, z]
            initial_guess: 初始猜测关节角度,为None时使用当前关节角度
        Returns:
            (关节角度解, 位置误差)
        """
        if initial_guess is None:
            initial_guess = self.robot.get_current_joint_angles()

        best_solution = None
        best_error = float('inf')

        for attempt in range(self.config.ik_max_attempts):
            # 添加随机扰动(帮助跳出局部最优)
            if attempt > 0:
                noise = np.random.uniform(-0.1, 0.1, len(initial_guess))
                x0 = initial_guess + noise
            else:
                x0 = initial_guess.copy()

            # 优化求解
            result = minimize(
                self._ik_cost,
                x0,
                args=(target_pos,),
                method='L-BFGS-B',
                bounds=self._get_bounds(),
                options={'maxiter': self.config.ik_max_iter, 'ftol': 1e-6}
            )

            if result.fun < best_error:
                best_error = result.fun
                best_solution = result.x

            if best_error < self.config.ik_tolerance:
                break

        return best_solution, best_error


# ============================================
# 运动控制器类
# ============================================
class MotionController:
    """运动控制器(处理平滑插值运动)"""

    def __init__(self, robot: PandaRobot, config: RobotConfig):
        self.robot = robot
        self.config = config

    @staticmethod
    def _normalize_angle(angle: float) -> float:
        """归一化角度到 [-π, π]"""
        angle = float(angle)
        while angle > np.pi:
            angle -= 2.0 * np.pi
        while angle < -np.pi:
            angle += 2.0 * np.pi
        return angle

    @staticmethod
    def _smoothstep(t: float) -> float:
        """平滑步进函数(S曲线)"""
        return t * t * (3 - 2 * t)

    def move_to_joints(self, target_joints: np.ndarray,
                       viewer_obj: Optional[mujoco.viewer] = None) -> None:
        """
        平滑移动到目标关节角度
        Args:
            target_joints: 目标关节角度
            viewer_obj: MuJoCo可视化对象
        """
        start_joints = self.robot.get_current_joint_angles()

        print(f"      运动步数: {self.config.move_steps}")

        for step in range(self.config.move_steps + 1):
            t = step / self.config.move_steps
            t_smooth = self._smoothstep(t)

            # 线性插值
            current_joints = start_joints + t_smooth * (target_joints - start_joints)

            # 角度归一化
            for i in range(len(current_joints)):
                current_joints[i] = self._normalize_angle(current_joints[i])

            # 应用到机器人
            self.robot.apply_joint_angles(current_joints)

            # 同步视图
            if viewer_obj is not None:
                viewer_obj.sync()

            # 控制速度
            time.sleep(self.config.move_dt)

            # 打印进度
            if step % 40 == 0:
                percent = (step / self.config.move_steps) * 100
                print(f"\r      运动进度: {percent:.0f}%", end="", flush=True)

        print(f"\r      运动进度: 100% ✓")

    def move_to_position(self, target_pos: np.ndarray,
                         ik_solver: InverseKinematicsSolver,
                         viewer_obj: Optional[mujoco.viewer] = None) -> Tuple[bool, float]:
        """
        移动到目标空间位置
        Args:
            target_pos: 目标位置
            ik_solver: 逆运动学求解器
            viewer_obj: 可视化对象
        Returns:
            (是否成功, 定位误差)
        """
        # 求解逆运动学
        current_angles = self.robot.get_current_joint_angles()
        solution, ik_error = ik_solver.solve(target_pos, current_angles)

        if ik_error >= 0.02:  # 误差大于2cm认为失败
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
    """可视化辅助类"""

    def __init__(self, robot: PandaRobot):
        self.robot = robot
        self.viewer = None

    def launch(self) -> None:
        """启动可视化窗口"""
        self.viewer = mujoco.viewer.launch_passive(self.robot.model, self.robot.data)
        self._setup_camera()
        print("✓ 可视化窗口已启动")

    def _setup_camera(self) -> None:
        """设置相机视角"""
        self.viewer.cam.lookat = np.array([0.4, 0, 0.4])
        self.viewer.cam.distance = 2.5
        self.viewer.cam.azimuth = 45
        self.viewer.cam.elevation = -30

    def add_marker(self, position: np.ndarray, color: List[float], radius: float = 0.025) -> None:
        """
        添加3D标记点(彩色小球)
        Args:
            position: 位置 [x, y, z]
            color: 颜色 [r, g, b]
            radius: 半径
        """
        if self.viewer is None:
            return

        user_scene = self.viewer.user_scn
        marker_idx = user_scene.ngeom

        if marker_idx >= 100:
            print("⚠ 标记点数量已达上限(100)")
            return

        # 修复: 移除 assets 和 filename 参数，使用正确的参数格式
        geom = user_scene.geoms[marker_idx]
        mujoco.mjv_initGeom(
            geom,
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=[radius, 0.0, 0.0],
            pos=position.astype(np.float64),
            mat=np.eye(3).flatten().astype(np.float64),
            rgba=np.array(color + [1.0], dtype=np.float32)
        )
        user_scene.ngeom += 1

    def sync(self) -> None:
        """同步视图"""
        if self.viewer is not None:
            self.viewer.sync()

    def is_running(self) -> bool:
        """检查窗口是否运行中"""
        return self.viewer is not None and self.viewer.is_running()

    def close(self) -> None:
        """关闭窗口"""
        if self.viewer is not None:
            self.viewer.close()


# ============================================
# 主程序
# ============================================
def main():
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
    print(f"\n初始末端位置: ({start_pos[0]:.3f}, {start_pos[1]:.3f}, {start_pos[2]:.3f})")

    # 4. 初始化组件
    ik_solver = InverseKinematicsSolver(robot, config)
    motion_ctrl = MotionController(robot, config)
    visualizer = Visualizer(robot)

    # 5. 启动可视化
    visualizer.launch()

    # 6. 定义7个目标点(6个不同位置+返回起点)
    target_points = [
        {"name": "点1-前方", "pos": np.array([0.5, 0.0, 0.4]), "color": [1.0, 0.0, 0.0]},
        {"name": "点2-右前方", "pos": np.array([0.45, 0.25, 0.45]), "color": [0.0, 1.0, 0.0]},
        {"name": "点3-左前方", "pos": np.array([0.45, -0.25, 0.45]), "color": [0.0, 0.0, 1.0]},
        {"name": "点4-上方", "pos": np.array([0.4, 0.0, 0.55]), "color": [1.0, 1.0, 0.0]},
        {"name": "点5-右下方", "pos": np.array([0.35, 0.2, 0.35]), "color": [1.0, 0.0, 1.0]},
        {"name": "点6-左下方", "pos": np.array([0.35, -0.2, 0.35]), "color": [0.0, 1.0, 1.0]},
        {"name": "点7-返回起点", "pos": np.array([0.5, 0.0, 0.4]), "color": [1.0, 0.5, 0.0]},
    ]

    # 7. 添加标记点
    print("\n【添加目标点标记】")
    for i, point in enumerate(target_points, 1):
        visualizer.add_marker(point["pos"], point["color"], radius=0.025)
        print(f"  目标点{i}: {point['name']} at {point['pos']}")

    visualizer.add_marker(start_pos, [0.5, 0.5, 0.5], radius=0.02)
    print(f"  起点: 初始位置 at {start_pos}")

    visualizer.sync()
    print("\n等待 2 秒后开始运动...")
    time.sleep(2)

    # 8. 依次运动到每个目标点
    print("\n" + "=" * 60)
    print("开始执行运动控制")
    print("=" * 60)

    for i, point in enumerate(target_points, 1):
        print(f"\n--- 目标点 {i}: {point['name']} ---")
        print(f"  目标位置: ({point['pos'][0]:.3f}, {point['pos'][1]:.3f}, {point['pos'][2]:.3f})")

        # 执行运动
        success, error = motion_ctrl.move_to_position(
            point["pos"], ik_solver, visualizer.viewer
        )

        if success:
            actual_pos = robot.get_end_effector_position()
            print(f"  ✓ 到达目标点 {i}")
            print(f"    实际位置: ({actual_pos[0]:.3f}, {actual_pos[1]:.3f}, {actual_pos[2]:.3f})")
            print(f"    定位误差: {error * 1000:.2f} mm")
        else:
            print(f"  ⚠ 运动失败, IK误差: {error * 1000:.2f} mm")

        # 点间暂停
        if i < len(target_points):
            print(f"  暂停 {config.pause_time} 秒...")
            time.sleep(config.pause_time)

    # 9. 完成
    print("\n" + "=" * 60)
    print("✓ 所有目标点运动完成!")
    print("=" * 60)

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