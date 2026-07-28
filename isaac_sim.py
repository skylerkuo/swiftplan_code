#!/usr/bin/env python3
"""
isaac_manual_record_server_2s.py

Isaac Sim 人工標註與 Franka 控制 Server。

控制規則：
1. 每一筆新指令都先拍照並寫入一筆 JSONL。
2. JSONL 格式保持不變：
   {
       "image_path": "...png",
       "instruction": "tidy up the properties",
       "annotation": {
           "action_label": "pickup apple"
       }
   }
3. pickup / putdown tray / putdown box：
   - 一般情況每次執行 2 秒模擬時間。
   - 若 2 秒到達時正在閉合夾爪，會繼續完成整個閉合階段。
   - 夾爪閉合完成後立即暫停，不會直接進入後續抬升。
   - 一旦完成閉合，暫停、抬升與移動期間都持續維持閉合目標。
   - 直到 putdown 進入開爪階段，才解除閉合目標。
   - 相同指令從原 generator 暫停位置繼續。
   - 不同指令取消原 generator，開始新動作。
4. home：
   - 先記錄一筆資料。
   - 不受 2 秒限制，直接執行到完成。
5. finished：
   - 記錄 "tidy up the properties finished"。
   - 取消目前動作並保持手臂目前姿態。
   - 不執行其他機械手臂動作。
6. 每個 TCP 連線只處理一筆 JSON 指令，回覆後立即關閉。
"""

from __future__ import annotations

import json
import os
import re
import socket
import traceback
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Generator, Optional, Tuple

import numpy as np
from PIL import Image
from isaacsim import SimulationApp


# SimulationApp 必須先於多數 Isaac Sim 模組建立。
simulation_app = SimulationApp({"headless": False})

import omni.usd
from isaacsim.core.api import World
from isaacsim.robot.manipulators.examples.franka import Franka
from isaacsim.robot.manipulators.examples.franka.controllers.rmpflow_controller import (
    RMPFlowController,
)
from isaacsim.sensors.camera import Camera
from pxr import Usd, UsdGeom

try:
    from isaacsim.core.utils.types import ArticulationAction
except ImportError:
    from omni.isaac.core.utils.types import ArticulationAction


# =========================
# Config
# =========================

HOST = "127.0.0.1"
PORT = 6547
LISTEN_BACKLOG = 8
MAX_REQUEST_BYTES = 64 * 1024

DATA_ROOT = Path("/home/skyler/Desktop/isaac_python_v2")
SAVE_DIR = DATA_ROOT / "captured_images"
JSONL_PATH = DATA_ROOT / "data_gradu.jsonl"
USD_PATH = str(DATA_ROOT / "graduation.usd")

# Server 可接受的兩種完整任務指令。
TASK_INSTRUCTIONS = {
    "tray": "tidy up the properties into the tray",
    "box": "tidy up the properties into the box",
}

DEFAULT_TASK_DESTINATION = "tray"
DEFAULT_TASK_INSTRUCTION = TASK_INSTRUCTIONS[
    DEFAULT_TASK_DESTINATION
]

CAMERA_PRIM_PATH = "/World/Camera"
FRANKA_PRIM_PATH = "/World/Franka"
CAMERA_RESOLUTION = (640, 480)

ACTION_SLICE_SECONDS = 2.0
DEFAULT_PHYSICS_DT = 1.0 / 60.0

TARGET_PRIM_PATHS: Dict[str, str] = {
    # "apple": "/World/Apple",
    # "orange": "/World/Orange",
    # "cube": "/World/Cube",
}

TARGET_GRASP_OFFSETS: Dict[str, np.ndarray] = {
    # "apple": np.array([0.0, 0.0, 0.01]),
}

VERTICAL_Q = np.array([0.0, 1.0, 0.0, 0.0])
HIGH_HOME_POS = np.array([0.4, 0.0, 0.6])

# 兩個固定放置位置。
# 依照目前需求：tray 位於正 Y 側，box 位於負 Y 側。
# 若實際場景座標不同，只需修改這兩個陣列。
TRAY_PLACE_POS = np.array(
    [-0.12, -0.65, 0.45],
    dtype=np.float64,
)
BOX_PLACE_POS = np.array(
    [-0.12, 0.65, 0.45],
    dtype=np.float64,
)

APPROACH_HEIGHT = 0.10
GRASP_Z_OFFSET = 0.01
# 抓取後先垂直抬升，避免水平分量使物體從指間滑落。
LIFT_OFFSET = np.array([0.0, 0.0, 0.12])
LIFT_FRAMES = 150

# Franka Panda 兩根手指的關節位置（公尺）。
# 每根手指 0.04 為張開；0.0 為完全閉合目標。
GRIPPER_OPEN_POSITIONS = np.array(
    [0.035, 0.035],
    dtype=np.float64,
)
GRIPPER_CLOSED_POSITIONS = np.array(
    [0.0, 0.0],
    dtype=np.float64,
)

MOVE_FRAMES = 100
GRIPPER_CLOSE_FRAMES = 90
GRIPPER_OPEN_FRAMES = 30
HOME_FRAMES = 30
CAMERA_WARMUP_FRAMES = 30
CAPTURE_SETTLE_FRAMES = 3

COMPLETED_REQUEST_CACHE_SIZE = 200
DEBUG = True

SAVE_DIR.mkdir(parents=True, exist_ok=True)
JSONL_PATH.parent.mkdir(parents=True, exist_ok=True)


# =========================
# Logging
# =========================

def log(message: str) -> None:
    print(message, flush=True)


def debug(message: str) -> None:
    if DEBUG:
        print(f"[DEBUG] {message}", flush=True)


def log_exception(prefix: str, exc: BaseException) -> None:
    log(f"{prefix}: {type(exc).__name__}: {exc}")
    if DEBUG:
        traceback.print_exc()


# =========================
# Scene initialization
# =========================

def open_stage_and_wait() -> None:
    log(f"[INIT] Opening USD stage: {USD_PATH}")
    result = omni.usd.get_context().open_stage(USD_PATH)
    debug(f"open_stage result={result}")

    for _ in range(100):
        simulation_app.update()

    stage = omni.usd.get_context().get_stage()

    if stage is None:
        raise RuntimeError("USD stage 載入失敗。")


def require_prim(prim_path: str) -> None:
    stage = omni.usd.get_context().get_stage()
    prim = stage.GetPrimAtPath(prim_path)

    if not prim or not prim.IsValid():
        raise RuntimeError(f"找不到必要 Prim：{prim_path}")


open_stage_and_wait()
require_prim(CAMERA_PRIM_PATH)
require_prim(FRANKA_PRIM_PATH)

world = World(stage_units_in_meters=1.0)

franka = world.scene.add(
    Franka(
        prim_path=FRANKA_PRIM_PATH,
        name="franka",
    )
)

camera = Camera(
    prim_path=CAMERA_PRIM_PATH,
    resolution=CAMERA_RESOLUTION,
)

world.reset()
camera.initialize()

rmpflow_controller = RMPFlowController(
    name="manual_record_rmpflow",
    robot_articulation=franka,
)

for _ in range(CAMERA_WARMUP_FRAMES):
    world.step(render=True)

try:
    physics_dt = float(world.get_physics_dt())

    if physics_dt <= 0.0:
        raise ValueError("physics_dt 必須大於 0。")

except Exception:
    physics_dt = DEFAULT_PHYSICS_DT

ACTION_SLICE_STEPS = max(
    1,
    int(round(ACTION_SLICE_SECONDS / physics_dt)),
)

log("[INIT] Isaac Sim scene, Franka and camera are ready.")
log(
    f"[INIT] physics_dt={physics_dt:.6f}, "
    f"slice_steps={ACTION_SLICE_STEPS}"
)


# =========================
# Persistent action state
# =========================

current_target_name: Optional[str] = None
last_picked_target_name: Optional[str] = None

# 未完成、可在下一次相同指令時繼續的 generator。
active_task: Optional[Generator[None, None, None]] = None
active_action_key: Optional[Tuple[str, Optional[str]]] = None
active_action_label: Optional[str] = None
active_command: Optional[str] = None
active_target_name: Optional[str] = None

# 目前這一筆 request 的執行狀態。
execution_running = False
execution_mode: Optional[str] = None  # "SLICE" 或 "FULL"
execution_steps_remaining = 0
execution_request_id: Optional[str] = None
execution_image_path: Optional[str] = None

# 保存目前非同步執行中的完整任務指令。
# start_request() 結束後，finish_paused() / finish_completed() /
# fail_execution() 仍需要使用此值。
execution_task_instruction: Optional[str] = None

execution_continued = False
execution_switched = False

# 目前 generator 正在執行的細部階段。
# 兩秒到達時若為 GRIPPER_CLOSING，會繼續到 GRIPPER_CLOSED_BOUNDARY。
current_motion_phase = "IDLE"
slice_time_expired = False

# 持續夾持目標。
# 一旦 close_gripper() 設為閉合，即使 generator 暫停或手臂正在移動，
# 每一個 simulation step 都會重新送出閉合位置，直到 open_gripper()。
desired_gripper_positions: Optional[np.ndarray] = (
    GRIPPER_OPEN_POSITIONS.copy()
)

# 單次 TCP 連線狀態。
client_conn: Optional[socket.socket] = None
client_addr: Optional[Tuple[str, int]] = None
request_buffer = bytearray()

# 相同 request_id 不可重複記錄或執行。
completed_requests: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()


# =========================
# Socket setup
# =========================

server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
server_socket.bind((HOST, PORT))
server_socket.listen(LISTEN_BACKLOG)
server_socket.setblocking(False)

log(f"[SOCKET] Listening on {HOST}:{PORT}")
log("[SOCKET] One connection carries exactly one command.")


# =========================
# Socket helpers
# =========================

def close_client(reason: str = "") -> None:
    global client_conn, client_addr, request_buffer

    if reason:
        debug(f"Closing client {client_addr}: {reason}")

    if client_conn is not None:
        try:
            client_conn.close()
        except OSError:
            pass

    client_conn = None
    client_addr = None
    request_buffer = bytearray()


def send_reply_and_close(reply: Dict[str, Any]) -> None:
    if client_conn is not None:
        encoded = (
            json.dumps(reply, ensure_ascii=False) + "\n"
        ).encode("utf-8")

        try:
            client_conn.setblocking(True)
            client_conn.settimeout(5.0)
            client_conn.sendall(encoded)

        except OSError as exc:
            log_exception("[SOCKET] Failed to send reply", exc)

    close_client("reply completed")


def make_reply(
    request_id: Optional[str],
    status: str,
    **extra: Any,
) -> Dict[str, Any]:
    reply: Dict[str, Any] = {
        "request_id": request_id,
        "status": status,
    }
    reply.update(extra)
    return reply


def cache_completed_reply(
    request_id: str,
    reply: Dict[str, Any],
) -> None:
    completed_requests[request_id] = dict(reply)
    completed_requests.move_to_end(request_id)

    while len(completed_requests) > COMPLETED_REQUEST_CACHE_SIZE:
        completed_requests.popitem(last=False)


# =========================
# Command parsing
# =========================

def parse_task_instruction(
    instruction: Any,
) -> Tuple[str, str]:
    """
    驗證 Client 傳入的完整任務指令。

    回傳：
        task_instruction, task_destination
    """
    if instruction is None:
        task_instruction = DEFAULT_TASK_INSTRUCTION
    elif isinstance(instruction, str):
        task_instruction = instruction.strip()
    else:
        raise ValueError("instruction 必須是字串。")

    for destination, allowed_instruction in TASK_INSTRUCTIONS.items():
        if task_instruction.lower() == allowed_instruction.lower():
            return allowed_instruction, destination

    allowed_text = "、".join(
        repr(value)
        for value in TASK_INSTRUCTIONS.values()
    )
    raise ValueError(
        f"不支援的 instruction：{task_instruction!r}。"
        f"只接受 {allowed_text}。"
    )


def parse_action_label(
    action_label: str,
    task_instruction: str,
    task_destination: str,
) -> Tuple[str, Optional[str]]:
    text = str(action_label).strip()
    lower = text.lower()

    if not text:
        raise ValueError("action_label 不可為空。")

    expected_finished_label = (
        f"{task_instruction} finished"
    )

    # Client 可只送出 finished；Server 依固定 instruction 自動展開。
    if lower in {
        "finished",
        expected_finished_label.lower(),
    }:
        return "finished", None

    if lower == "putdown":
        return f"putdown_{task_destination}", None

    if lower in {"putdown tray", "putdown box"}:
        requested_destination = lower.split(maxsplit=1)[1]

        if requested_destination != task_destination:
            raise ValueError(
                f"目前 instruction 的目的地為 "
                f"{task_destination}，不可使用 "
                f"putdown {requested_destination}。"
            )

        return f"putdown_{requested_destination}", None

    if lower == "home":
        return "home", None

    if lower.startswith("pickup"):
        parts = text.split(maxsplit=1)

        if len(parts) != 2 or not parts[1].strip():
            raise ValueError("pickup 後面必須有物品名稱。")

        return "pickup", parts[1].strip()

    raise ValueError(
        "不支援的 action_label。只接受 pickup <物品名稱>、"
        f"putdown {task_destination}、finished 或 home。"
    )


def make_action_key(
    command: str,
    target_name: Optional[str],
) -> Tuple[str, Optional[str]]:
    normalized_target = (
        target_name.strip().lower()
        if target_name is not None
        else None
    )

    return command, normalized_target


# =========================
# USD helpers
# =========================

def find_target_prim_path(target_name: str) -> str:
    stage = omni.usd.get_context().get_stage()
    lookup_key = target_name.lower()

    configured_path = TARGET_PRIM_PATHS.get(lookup_key)

    if configured_path:
        prim = stage.GetPrimAtPath(configured_path)

        if prim and prim.IsValid():
            return configured_path

        raise ValueError(
            f"TARGET_PRIM_PATHS 指定的 Prim 不存在：{configured_path}"
        )

    direct_path = f"/World/{target_name}"
    direct_prim = stage.GetPrimAtPath(direct_path)

    if direct_prim and direct_prim.IsValid():
        return direct_path

    matches = []

    for prim in stage.Traverse():
        if prim.GetName().lower() == lookup_key:
            matches.append(str(prim.GetPath()))

    if len(matches) == 1:
        return matches[0]

    if len(matches) > 1:
        raise ValueError(
            f"物品名稱 {target_name!r} 對應到多個 Prim：{matches}。"
            "請在 TARGET_PRIM_PATHS 指定完整路徑。"
        )

    raise ValueError(
        f"找不到物品 {target_name!r}。已嘗試 {direct_path}，"
        "也找不到同名 Prim。"
    )


def get_world_position(prim_path: str) -> np.ndarray:
    stage = omni.usd.get_context().get_stage()
    prim = stage.GetPrimAtPath(prim_path)

    if not prim or not prim.IsValid():
        raise ValueError(f"Prim 不存在：{prim_path}")

    xform = UsdGeom.Xformable(prim)
    matrix = xform.ComputeLocalToWorldTransform(
        Usd.TimeCode.Default()
    )
    translation = matrix.ExtractTranslation()

    return np.array(
        [translation[0], translation[1], translation[2]],
        dtype=np.float64,
    )


# =========================
# Image and JSONL
# =========================

def rgba_to_uint8_rgb(rgba: np.ndarray) -> np.ndarray:
    if rgba.ndim != 3 or rgba.shape[2] < 3:
        raise ValueError(
            f"Camera frame shape 不正確：{rgba.shape}"
        )

    rgb = np.asarray(rgba[:, :, :3])

    if rgb.dtype == np.uint8:
        return rgb

    rgb = rgb.astype(np.float32)

    if rgb.size > 0 and float(np.nanmax(rgb)) <= 1.0:
        rgb = rgb * 255.0

    rgb = np.nan_to_num(
        rgb,
        nan=0.0,
        posinf=255.0,
        neginf=0.0,
    )

    return np.clip(rgb, 0.0, 255.0).astype(np.uint8)


def safe_filename_component(text: str) -> str:
    cleaned = re.sub(
        r"[^A-Za-z0-9._-]+",
        "_",
        text.strip(),
    )
    cleaned = cleaned.strip("._-")
    return cleaned or "unknown"


def resolve_episode_target_name(
    command_target: Optional[str],
) -> str:
    if command_target:
        return command_target

    if current_target_name:
        return current_target_name

    if last_picked_target_name:
        return last_picked_target_name

    return "unknown"


def capture_and_append_jsonl(
    action_label: str,
    task_instruction: str,
    command_target: Optional[str],
    request_id: str,
) -> str:
    """
    每一筆新 request 都拍照並寫入一筆 JSONL。
    JSONL 欄位與原格式完全相同。
    """
    for _ in range(CAPTURE_SETTLE_FRAMES):
        # 拍照等待期間若正在拿著物體，仍持續維持夾爪閉合。
        maintain_gripper_target()
        world.step(render=True)

    rgba = camera.get_rgba()

    if rgba is None:
        raise RuntimeError(
            "camera.get_rgba() returned None，Camera 尚未準備完成。"
        )

    rgb = rgba_to_uint8_rgb(np.asarray(rgba))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

    target_name = resolve_episode_target_name(command_target)
    target_component = safe_filename_component(target_name)
    action_component = safe_filename_component(action_label)

    final_path = SAVE_DIR / (
        f"capture_{target_component}_"
        f"{action_component}_{timestamp}.png"
    )
    temp_path = final_path.with_suffix(".tmp.png")

    Image.fromarray(rgb).save(temp_path)
    os.replace(temp_path, final_path)

    entry = {
        "image_path": str(final_path),
        "instruction": task_instruction,
        "annotation": {
            "action_label": action_label,
        },
    }

    with JSONL_PATH.open("a", encoding="utf-8") as file:
        file.write(
            json.dumps(entry, ensure_ascii=False) + "\n"
        )
        file.flush()
        os.fsync(file.fileno())

    log(f"[DATA] Image saved : {final_path}")
    log(f"[DATA] Action label : {action_label!r}")
    debug(f"[DATA] request_id={request_id}")

    return str(final_path)


# =========================
# Hold control
# =========================

def hold_current_pose() -> None:
    """
    暫停手臂姿態，但不要把夾爪改成當下的半閉合位置。

    Franka 一般包含 7 個手臂關節與 2 個手指關節。此處優先只
    Hold 手臂關節；夾爪則由 maintain_gripper_target() 持續控制。
    """
    joint_positions = franka.get_joint_positions()

    if joint_positions is None:
        log("[HOLD] 無法取得目前關節位置，略過 Hold。")
        maintain_gripper_target()
        return

    joint_positions = np.asarray(
        joint_positions,
        dtype=np.float64,
    )

    # 最後兩個 DOF 視為手指關節。
    arm_dof_count = max(0, joint_positions.size - 2)

    try:
        if arm_dof_count > 0:
            arm_indices = np.arange(
                arm_dof_count,
                dtype=np.int64,
            )

            hold_action = ArticulationAction(
                joint_positions=joint_positions[:arm_dof_count].copy(),
                joint_velocities=np.zeros(
                    arm_dof_count,
                    dtype=np.float64,
                ),
                joint_indices=arm_indices,
            )
        else:
            hold_action = ArticulationAction(
                joint_positions=joint_positions.copy(),
                joint_velocities=np.zeros_like(joint_positions),
            )

        franka.apply_action(hold_action)

    except TypeError:
        # 相容部分不支援 joint_indices 的舊版 Isaac Sim。
        # 即使暫時對全部關節 Hold，後面也會立刻重新送出夾爪目標。
        fallback_action = ArticulationAction(
            joint_positions=joint_positions.copy(),
            joint_velocities=np.zeros_like(joint_positions),
        )
        franka.apply_action(fallback_action)

    maintain_gripper_target()
    debug("[HOLD] Arm pose held; gripper target preserved.")


# =========================
# Franka generators
# =========================

def set_motion_phase(phase: str) -> str:
    """更新並回傳目前細部動作階段。"""
    global current_motion_phase
    current_motion_phase = phase
    return phase


def apply_gripper_target(target_positions: np.ndarray) -> None:
    """
    每個 simulation step 都重新送出夾爪目標。

    這可避免兩秒暫停時 hold_current_pose() 固定半閉合位置後，
    下一次續跑卻沒有重新送出閉合命令。
    """
    target_positions = np.asarray(
        target_positions,
        dtype=np.float64,
    )

    if target_positions.shape != (2,):
        raise ValueError(
            "夾爪目標必須包含兩個手指關節位置，"
            f"目前 shape={target_positions.shape}"
        )

    gripper_action = ArticulationAction(
        joint_positions=target_positions.copy(),
    )
    franka.gripper.apply_action(gripper_action)


def set_desired_gripper_positions(
    target_positions: np.ndarray,
) -> None:
    """設定之後每一個 simulation step 都要維持的夾爪目標。"""
    global desired_gripper_positions

    target_positions = np.asarray(
        target_positions,
        dtype=np.float64,
    )

    if target_positions.shape != (2,):
        raise ValueError(
            "desired gripper target 必須包含兩個手指關節位置，"
            f"目前 shape={target_positions.shape}"
        )

    desired_gripper_positions = target_positions.copy()
    apply_gripper_target(desired_gripper_positions)


def maintain_gripper_target() -> None:
    """
    持續施加夾爪目標。

    即使 pickup generator 暫停、手臂正在抬升或移動到放置區，
    只要 desired_gripper_positions 仍是 CLOSED，就會持續夾緊。
    """
    if desired_gripper_positions is None:
        return

    apply_gripper_target(desired_gripper_positions)


def get_finger_positions() -> Optional[np.ndarray]:
    """讀取最後兩個手指關節位置，供除錯顯示。"""
    joint_positions = franka.get_joint_positions()

    if joint_positions is None:
        return None

    joint_positions = np.asarray(
        joint_positions,
        dtype=np.float64,
    )

    if joint_positions.size < 2:
        return None

    return joint_positions[-2:].copy()


def move_to(
    target_position: np.ndarray,
    frames: int = MOVE_FRAMES,
    phase: str = "ARM_MOVING",
) -> Generator[str, None, None]:
    rmpflow_controller.reset()

    for _ in range(frames):
        action = rmpflow_controller.forward(
            target_end_effector_position=target_position,
            target_end_effector_orientation=VERTICAL_Q,
        )
        franka.apply_action(action)
        yield set_motion_phase(phase)


def open_gripper(
    frames: int = GRIPPER_OPEN_FRAMES,
) -> Generator[str, None, None]:
    """
    設定持續張開目標。

    一旦進入此階段，之後每個 simulation step 都會維持 OPEN，
    直到下一次 close_gripper()。
    """
    set_desired_gripper_positions(GRIPPER_OPEN_POSITIONS)

    for frame_index in range(frames):
        maintain_gripper_target()

        if DEBUG and frame_index % 10 == 0:
            finger_positions = get_finger_positions()
            if finger_positions is not None:
                debug(
                    "[GRIPPER OPEN] "
                    f"frame={frame_index}, "
                    f"positions={finger_positions}"
                )

        yield set_motion_phase("GRIPPER_OPENING")

    yield set_motion_phase("GRIPPER_OPENED_BOUNDARY")


def close_gripper(
    frames: int = GRIPPER_CLOSE_FRAMES,
) -> Generator[str, None, None]:
    """
    設定持續閉合目標。

    閉合完成後不會解除此目標。暫停、抬升、移動到放置位置期間
    都會持續送出 CLOSED target，直到 putdown 執行 open_gripper()。
    """
    set_desired_gripper_positions(GRIPPER_CLOSED_POSITIONS)

    for frame_index in range(frames):
        maintain_gripper_target()

        if DEBUG and frame_index % 10 == 0:
            finger_positions = get_finger_positions()
            if finger_positions is not None:
                debug(
                    "[GRIPPER CLOSE] "
                    f"frame={frame_index}, "
                    f"positions={finger_positions}"
                )

        yield set_motion_phase("GRIPPER_CLOSING")

    # 只代表閉合階段完成；desired target 仍保持 CLOSED。
    yield set_motion_phase("GRIPPER_CLOSED_BOUNDARY")


def go_home(
    frames: int = HOME_FRAMES,
) -> Generator[str, None, None]:
    yield from move_to(
        HIGH_HOME_POS,
        frames=frames,
        phase="HOME_MOVING",
    )

    yield from open_gripper(frames)


def pick_sequence(
    target_name: str,
    target_position: np.ndarray,
) -> Generator[str, None, None]:
    grasp_offset = TARGET_GRASP_OFFSETS.get(
        target_name.lower(),
        np.zeros(3, dtype=np.float64),
    )

    grasp_position = (
        target_position
        + grasp_offset
        + np.array([0.0, 0.0, GRASP_Z_OFFSET])
    )
    approach_position = (
        grasp_position
        + np.array([0.0, 0.0, APPROACH_HEIGHT])
    )
    lift_position = grasp_position + LIFT_OFFSET

    debug(f"target_position={target_position}")
    debug(f"grasp_offset={grasp_offset}")
    debug(f"grasp_position={grasp_position}")

    yield from open_gripper()

    yield from move_to(
        approach_position,
        phase="APPROACHING_TARGET",
    )

    yield from move_to(
        grasp_position,
        phase="DESCENDING_TO_GRASP",
    )

    yield from close_gripper()

    # 只有下一次相同 pickup 指令續跑後，才會從此處開始抬升。
    yield from move_to(
        lift_position,
        frames=LIFT_FRAMES,
        phase="LIFTING_OBJECT",
    )


def place_sequence(
    destination_name: str,
    destination_position: np.ndarray,
) -> Generator[str, None, None]:
    """
    將目前夾取物移動到指定目的地。

    移動期間會持續維持夾爪閉合；到達目的地後才開爪。
    """
    yield from move_to(
        destination_position,
        phase=f"MOVING_TO_{destination_name.upper()}",
    )
    yield from open_gripper()
    yield from go_home()


def build_new_task(
    command: str,
    target_name: Optional[str],
) -> Generator[str, None, None]:
    global current_target_name
    global last_picked_target_name

    if command == "pickup":
        assert target_name is not None

        prim_path = find_target_prim_path(target_name)
        target_position = get_world_position(prim_path)

        current_target_name = target_name
        last_picked_target_name = target_name

        log(
            f"[ROBOT] New pickup: target={target_name!r}, "
            f"prim={prim_path}, position={target_position}"
        )

        return pick_sequence(
            target_name=target_name,
            target_position=target_position,
        )

    if command in {"putdown_tray", "putdown_box"}:
        if current_target_name is None:
            current_target_name = last_picked_target_name

        if command == "putdown_tray":
            destination_name = "tray"
            destination_position = TRAY_PLACE_POS
        else:
            destination_name = "box"
            destination_position = BOX_PLACE_POS

        log(
            f"[ROBOT] New putdown destination={destination_name}, "
            f"position={destination_position}"
        )
        return place_sequence(
            destination_name=destination_name,
            destination_position=destination_position,
        )

    if command == "home":
        log(f"[ROBOT] Home directly to {HIGH_HOME_POS}")
        return go_home()

    raise ValueError(f"無法建立 command={command!r} 的動作。")


# =========================
# Action state helpers
# =========================

def clear_active_action() -> None:
    global active_task
    global active_action_key
    global active_action_label
    global active_command
    global active_target_name

    active_task = None
    active_action_key = None
    active_action_label = None
    active_command = None
    active_target_name = None


def clear_execution_state() -> None:
    global execution_running
    global execution_mode
    global execution_steps_remaining
    global execution_request_id
    global execution_image_path
    global execution_task_instruction
    global execution_continued
    global execution_switched
    global slice_time_expired

    execution_running = False
    execution_mode = None
    execution_steps_remaining = 0
    execution_request_id = None
    execution_image_path = None
    execution_task_instruction = None
    execution_continued = False
    execution_switched = False
    slice_time_expired = False


def cancel_active_action() -> None:
    if active_task is not None:
        debug(
            f"[ACTION] Cancel active action: "
            f"{active_action_label!r}"
        )

    clear_active_action()
    hold_current_pose()


# =========================
# Request execution
# =========================

def start_request(request: Dict[str, Any]) -> None:
    global active_task
    global active_action_key
    global active_action_label
    global active_command
    global active_target_name

    global execution_running
    global execution_mode
    global execution_steps_remaining
    global execution_request_id
    global execution_image_path
    global execution_task_instruction
    global execution_continued
    global execution_switched
    global slice_time_expired

    global current_target_name

    request_id_raw = request.get("request_id")
    action_label_raw = request.get("action_label")
    instruction_raw = request.get("instruction")

    if not isinstance(request_id_raw, str) or not request_id_raw.strip():
        send_reply_and_close(
            make_reply(
                None,
                "ERROR",
                message="request_id 必須是非空字串。",
            )
        )
        return

    request_id = request_id_raw.strip()

    if request_id in completed_requests:
        cached_reply = dict(completed_requests[request_id])
        cached_reply["execution_state"] = "DUPLICATE"
        cached_reply["duplicate_request"] = True

        log(
            f"[REQUEST] Duplicate request_id={request_id}; "
            "skip record and motion."
        )
        send_reply_and_close(cached_reply)
        return

    if not isinstance(action_label_raw, str):
        send_reply_and_close(
            make_reply(
                request_id,
                "ERROR",
                message="action_label 必須是字串。",
            )
        )
        return

    action_label = action_label_raw.strip()

    try:
        task_instruction, task_destination = (
            parse_task_instruction(instruction_raw)
        )
        command, target_name = parse_action_label(
            action_label=action_label,
            task_instruction=task_instruction,
            task_destination=task_destination,
        )

        # JSONL 中仍保留完整完成標註，不把簡短的 "finished" 寫入資料集。
        if command == "finished":
            action_label = f"{task_instruction} finished"

    except ValueError as exc:
        send_reply_and_close(
            make_reply(
                request_id,
                "ERROR",
                message=str(exc),
            )
        )
        return

    new_action_key = make_action_key(
        command=command,
        target_name=target_name,
    )

    log(
        f"[REQUEST] id={request_id} | "
        f"instruction={task_instruction!r} | "
        f"action_label={action_label!r}"
    )

    # pickup 先驗證物品存在，避免錯誤物品名稱被寫入資料集。
    if command == "pickup":
        try:
            assert target_name is not None
            find_target_prim_path(target_name)

        except Exception as exc:
            log_exception(
                "[REQUEST] Cannot resolve pickup target",
                exc,
            )
            send_reply_and_close(
                make_reply(
                    request_id,
                    "ERROR",
                    message=str(exc),
                    recorded=False,
                )
            )
            return

    # 每一筆新 request 都記錄一次資料，包含 home 與 finished。
    try:
        image_path = capture_and_append_jsonl(
            action_label=action_label,
            task_instruction=task_instruction,
            command_target=target_name,
            request_id=request_id,
        )

    except Exception as exc:
        log_exception(
            "[DATA] Capture or JSONL append failed",
            exc,
        )
        send_reply_and_close(
            make_reply(
                request_id,
                "ERROR",
                message=(
                    "資料記錄失敗："
                    f"{type(exc).__name__}: {exc}"
                ),
                recorded=False,
            )
        )
        return

    # finished：取消未完成動作，記錄後立即完成。
    if command == "finished":
        cancel_active_action()
        current_target_name = None

        reply = make_reply(
            request_id,
            "SUCCESS",
            execution_state="FINISHED",
            command=command,
            action_label=action_label,
            instruction=task_instruction,
            recorded=True,
            image_path=image_path,
            motion_executed=False,
            action_completed=True,
            continued=False,
            switched=True,
            slice_seconds=0.0,
        )

        cache_completed_reply(request_id, reply)
        send_reply_and_close(reply)
        return

    # home：取消未完成動作，建立 Home generator，直接跑到完成。
    if command == "home":
        switched = active_task is not None

        if active_task is not None:
            cancel_active_action()

        try:
            active_task = build_new_task(
                command="home",
                target_name=None,
            )

        except Exception as exc:
            log_exception("[ROBOT] Cannot build home task", exc)

            reply = make_reply(
                request_id,
                "ERROR",
                execution_state="BUILD_FAILED",
                command=command,
                action_label=action_label,
                instruction=task_instruction,
                message=(
                    "建立 Home 動作失敗："
                    f"{type(exc).__name__}: {exc}"
                ),
                recorded=True,
                image_path=image_path,
                motion_executed=False,
            )

            cache_completed_reply(request_id, reply)
            send_reply_and_close(reply)
            return

        active_action_key = new_action_key
        active_action_label = action_label
        active_command = command
        active_target_name = None

        execution_running = True
        execution_mode = "FULL"
        execution_steps_remaining = 0
        execution_request_id = request_id
        execution_image_path = image_path
        execution_task_instruction = task_instruction
        execution_continued = False
        execution_switched = switched
        slice_time_expired = False

        log("[EXECUTION] Home will run directly until completion.")
        return

    # pickup / putdown tray / putdown box：
    # 相同動作與相同目的地續跑；不同目的地會切換成新動作。
    if (
        active_task is not None
        and active_action_key == new_action_key
    ):
        continued = True
        switched = False

        log(
            f"[ACTION] Continue existing action: "
            f"{action_label!r}"
        )

    else:
        switched = active_task is not None

        if active_task is not None:
            log(
                f"[ACTION] Switch from "
                f"{active_action_label!r} to {action_label!r}"
            )
            cancel_active_action()

        try:
            active_task = build_new_task(
                command=command,
                target_name=target_name,
            )

        except Exception as exc:
            log_exception(
                "[ROBOT] Cannot build new task",
                exc,
            )

            reply = make_reply(
                request_id,
                "ERROR",
                execution_state="BUILD_FAILED",
                command=command,
                action_label=action_label,
                instruction=task_instruction,
                message=(
                    "建立機械手臂動作失敗："
                    f"{type(exc).__name__}: {exc}"
                ),
                recorded=True,
                image_path=image_path,
                motion_executed=False,
            )

            cache_completed_reply(request_id, reply)
            send_reply_and_close(reply)
            return

        active_action_key = new_action_key
        active_action_label = action_label
        active_command = command
        active_target_name = target_name
        continued = False

    execution_running = True
    execution_mode = "SLICE"
    execution_steps_remaining = ACTION_SLICE_STEPS
    execution_request_id = request_id
    execution_image_path = image_path
    execution_task_instruction = task_instruction
    execution_continued = continued
    execution_switched = switched
    slice_time_expired = False

    log(
        f"[EXECUTION] Start 2-second slice: "
        f"steps={ACTION_SLICE_STEPS}, "
        f"continued={continued}, switched={switched}"
    )


def finish_paused() -> None:
    assert execution_request_id is not None
    assert active_command is not None
    assert active_action_label is not None
    assert execution_task_instruction is not None

    request_id = execution_request_id
    image_path = execution_image_path
    instruction = execution_task_instruction
    command = active_command
    action_label = active_action_label
    continued = execution_continued
    switched = execution_switched

    hold_current_pose()

    reply = make_reply(
        request_id,
        "SUCCESS",
        execution_state="PAUSED",
        command=command,
        action_label=action_label,
        instruction=instruction,
        recorded=True,
        image_path=image_path,
        motion_executed=True,
        action_completed=False,
        continued=continued,
        switched=switched,
        slice_seconds=ACTION_SLICE_SECONDS,
        remaining_action=active_action_label,
    )

    cache_completed_reply(request_id, reply)
    clear_execution_state()
    send_reply_and_close(reply)


def finish_completed() -> None:
    global current_target_name

    assert execution_request_id is not None
    assert active_command is not None
    assert active_action_label is not None
    assert execution_task_instruction is not None

    request_id = execution_request_id
    image_path = execution_image_path
    instruction = execution_task_instruction
    command = active_command
    action_label = active_action_label
    continued = execution_continued
    switched = execution_switched
    mode = execution_mode

    if command in {"putdown_tray", "putdown_box", "home"}:
        current_target_name = None

    hold_current_pose()
    clear_active_action()

    state = (
        "HOME_COMPLETED"
        if command == "home"
        else "COMPLETED"
    )

    reply = make_reply(
        request_id,
        "SUCCESS",
        execution_state=state,
        command=command,
        action_label=action_label,
        instruction=instruction,
        recorded=True,
        image_path=image_path,
        motion_executed=True,
        action_completed=True,
        continued=continued,
        switched=switched,
        slice_seconds=(
            None if mode == "FULL" else ACTION_SLICE_SECONDS
        ),
        remaining_action=None,
    )

    cache_completed_reply(request_id, reply)
    clear_execution_state()
    send_reply_and_close(reply)


def fail_execution(exc: BaseException) -> None:
    request_id = execution_request_id
    image_path = execution_image_path
    instruction = execution_task_instruction
    command = active_command
    action_label = active_action_label

    log_exception("[ROBOT] Execution failed", exc)

    hold_current_pose()
    clear_active_action()

    reply = make_reply(
        request_id,
        "ERROR",
        execution_state="FAILED",
        command=command,
        action_label=action_label,
        instruction=instruction,
        message=(
            "機械手臂動作失敗："
            f"{type(exc).__name__}: {exc}"
        ),
        recorded=image_path is not None,
        image_path=image_path,
        motion_executed=False,
        action_completed=False,
    )

    if request_id is not None:
        cache_completed_reply(request_id, reply)

    clear_execution_state()
    send_reply_and_close(reply)


# =========================
# Connection handling
# =========================

def accept_client_if_idle() -> None:
    global client_conn
    global client_addr
    global request_buffer

    if client_conn is not None or execution_running:
        return

    try:
        conn, addr = server_socket.accept()

    except BlockingIOError:
        return

    conn.setblocking(False)
    client_conn = conn
    client_addr = addr
    request_buffer = bytearray()

    debug(f"Accepted client: {addr}")


def read_single_request_if_available() -> None:
    global request_buffer

    if client_conn is None or execution_running:
        return

    peer_closed = False

    try:
        while True:
            try:
                chunk = client_conn.recv(4096)

            except BlockingIOError:
                break

            if not chunk:
                peer_closed = True
                break

            request_buffer.extend(chunk)

            if len(request_buffer) > MAX_REQUEST_BYTES:
                send_reply_and_close(
                    make_reply(
                        None,
                        "ERROR",
                        message="Request 過大。",
                    )
                )
                return

            if b"\n" in request_buffer:
                break

    except OSError as exc:
        log_exception("[SOCKET] Receive failed", exc)
        close_client("receive error")
        return

    newline_index = request_buffer.find(b"\n")

    if newline_index >= 0:
        raw_line = bytes(
            request_buffer[:newline_index]
        ).strip()

        trailing = bytes(
            request_buffer[newline_index + 1:]
        ).strip()

        if trailing:
            send_reply_and_close(
                make_reply(
                    None,
                    "ERROR",
                    message="每個連線只能傳送一筆 JSON 指令。",
                )
            )
            return

    elif peer_closed:
        raw_line = bytes(request_buffer).strip()

    else:
        return

    if not raw_line:
        send_reply_and_close(
            make_reply(
                None,
                "ERROR",
                message="收到空白 Request。",
            )
        )
        return

    try:
        decoded = raw_line.decode("utf-8")
        request = json.loads(decoded)

    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        send_reply_and_close(
            make_reply(
                None,
                "ERROR",
                message=f"Request 不是有效 JSON：{exc}",
            )
        )
        return

    if not isinstance(request, dict):
        send_reply_and_close(
            make_reply(
                None,
                "ERROR",
                message="Request 必須是 JSON object。",
            )
        )
        return

    request_buffer = bytearray()
    start_request(request)


# =========================
# Main loop
# =========================

def tick_execution() -> None:
    """
    執行一個 generator step。

    一般階段：
        2 秒到達後立即暫停。

    夾爪閉合階段：
        即使超過 2 秒，也繼續執行到 GRIPPER_CLOSED_BOUNDARY，
        然後立即暫停，不會開始抬升。
    """
    global execution_steps_remaining
    global slice_time_expired

    if not execution_running or active_task is None:
        return

    try:
        phase = next(active_task)

        if execution_mode != "SLICE":
            return

        if not slice_time_expired:
            execution_steps_remaining -= 1

            if execution_steps_remaining <= 0:
                slice_time_expired = True
                log(
                    f"[EXECUTION] Reached "
                    f"{ACTION_SLICE_SECONDS:.1f} simulated seconds "
                    f"during phase={phase}."
                )

        if not slice_time_expired:
            return

        # 時間已到，但正在閉合：允許閉合階段完整跑完。
        if phase == "GRIPPER_CLOSING":
            debug(
                "[EXECUTION] Time expired during gripper closing; "
                "continue until fully closed."
            )
            return

        # close_gripper() 額外產生此邊界，確保閉合完成後
        # 暫停於此，不會在同一個 next() 中直接開始抬升。
        if phase == "GRIPPER_CLOSED_BOUNDARY":
            log(
                "[EXECUTION] Gripper closing completed after time limit; "
                "pause before lifting."
            )
            finish_paused()
            return

        # 其他任何階段在時間到達後照常暫停。
        log(
            f"[EXECUTION] Paused after "
            f"{ACTION_SLICE_SECONDS:.1f} simulated seconds, "
            f"phase={phase}."
        )
        finish_paused()

    except StopIteration:
        log("[EXECUTION] Action completed.")
        finish_completed()

    except Exception as exc:
        fail_execution(exc)


try:
    log("=" * 76)
    log("Isaac Sim Manual Record Server — 2 Second Action Slices")
    log(
        "Allowed instructions: "
        f"{list(TASK_INSTRUCTIONS.values())}"
    )
    log(f"Action slice      : {ACTION_SLICE_SECONDS:.1f} simulated seconds")
    log("Close behavior    : finish gripper closing even after time limit")
    log("Home behavior     : run directly until completion")
    log(f"Tray place pos    : {TRAY_PLACE_POS}")
    log(f"Box place pos     : {BOX_PLACE_POS}")
    log(f"Gripper open pos  : {GRIPPER_OPEN_POSITIONS}")
    log(f"Gripper close pos : {GRIPPER_CLOSED_POSITIONS}")
    log("Grip hold         : CLOSED target maintained during pause/lift/move")
    log(f"Lift offset       : {LIFT_OFFSET}")
    log(f"Lift frames       : {LIFT_FRAMES}")
    log(f"Image directory  : {SAVE_DIR}")
    log(f"JSONL file       : {JSONL_PATH}")
    log("=" * 76)

    while simulation_app.is_running():
        accept_client_if_idle()
        read_single_request_if_available()

        # 先更新本幀的手臂與夾爪控制命令。
        tick_execution()

        # 不論 generator 是否暫停，夾爪都持續維持目標。
        maintain_gripper_target()

        # 套用本幀控制命令並推進物理模擬。
        world.step(render=True)

except KeyboardInterrupt:
    log("[MAIN] Interrupted by user.")

except Exception as exc:
    log_exception("[MAIN] Server crashed", exc)

finally:
    log("[MAIN] Closing server.")

    try:
        hold_current_pose()
    except Exception:
        pass

    close_client("server shutdown")

    try:
        server_socket.close()
    except OSError:
        pass

    simulation_app.close()
