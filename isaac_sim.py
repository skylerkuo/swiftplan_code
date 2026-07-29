#!/usr/bin/env python3
"""
isaac_manual_record_server_dynamic.py

Isaac Sim 人工標註與 Franka 控制 Server（動態任務指令版本）。

主要改動：
1. instruction 不再限制為固定白名單，只要是非空字串即可，例如：
       put the apple into the tray
       put the cube into the box
       tidy up the properties into the tray
       prepare the fruits
2. 目前機器人控制技能保留：
       pickup <物品名稱>
       putdown tray
       putdown box
       finished
       clear
       home
3. putdown 可只傳 "putdown"，Server 會使用 request 中的
   default_destination；若未提供，會嘗試從 instruction 推定 tray/box，
   最後才使用 DEFAULT_DESTINATION。
4. JSONL 格式維持不變：
       {
           "image_path": "...png",
           "instruction": "put the apple into the box",
           "annotation": {
               "action_label": "pickup apple"
           }
       }
5. finished 仍寫成完整標註：
       "<instruction> finished"
6. pickup / putdown 每次最多執行 2 秒模擬時間：
   - 相同 instruction 與相同動作：從原 generator 暫停位置繼續。
   - instruction、目標物或目的地不同：取消舊 generator，建立新動作。
   - 若時間到達時正在閉合夾爪，會完成閉合後再暫停。
7. home 不受 2 秒限制，直接執行到完成。
8. clear 不拍照、不寫入 JSONL，直接重設場景。
9. 每個 TCP 連線只處理一筆 JSON 指令。

Request 範例：
    {
        "request_id": "abc123",
        "instruction": "put the cube into the box",
        "default_destination": "box",
        "action_label": "pickup cube"
    }
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

DEFAULT_TASK_INSTRUCTION = "tidy up the properties into the tray"
DEFAULT_DESTINATION = "tray"

CAMERA_PRIM_PATH = "/World/Camera"
FRANKA_PRIM_PATH = "/World/Franka"
CAMERA_RESOLUTION = (640, 480)

# 與原始程式註解一致，設定為 2 秒模擬時間。
ACTION_SLICE_SECONDS = 2.0
DEFAULT_PHYSICS_DT = 1.0 / 60.0

# 物品別名與 USD Prim path。
# 未列出的物品仍會依名稱自動搜尋：
#   /World/<target_name>
#   /World/<target_name 將空白轉底線>
#   Stage 中同名 Prim
TARGET_PRIM_PATHS: Dict[str, str] = {
    # "apple": "/World/Apple",
    # "orange": "/World/Orange",
    # "red cube": "/World/RedCube",
}

TARGET_GRASP_OFFSETS: Dict[str, np.ndarray] = {
    # "apple": np.array([0.0, 0.0, 0.01]),
}

VERTICAL_Q = np.array([0.0, 1.0, 0.0, 0.0])
HIGH_HOME_POS = np.array([0.4, 0.0, 0.6])

# 目前場景仍支援 tray 與 box；之後新增 basket、shelf 等目的地時，
# 只需在此新增設定，不必修改解析與執行主流程。
#
# prim_path：
#   - 設為有效 USD Prim path 時，會使用該 Prim 的即時世界座標。
#   - 設為 None 時，使用 fixed_position。
# place_offset：
#   - 若使用 prim_path，可用來把末端位置抬高到容器上方。
DESTINATION_CONFIGS: Dict[str, Dict[str, Any]] = {
    "tray": {
        "prim_path": None,
        "fixed_position": np.array(
            [-0.12, -0.65, 0.45],
            dtype=np.float64,
        ),
        "place_offset": np.zeros(3, dtype=np.float64),
    },
    "box": {
        "prim_path": None,
        "fixed_position": np.array(
            [-0.12, 0.65, 0.45],
            dtype=np.float64,
        ),
        "place_offset": np.zeros(3, dtype=np.float64),
    },
}

if DEFAULT_DESTINATION not in DESTINATION_CONFIGS:
    raise ValueError(
        "DEFAULT_DESTINATION 必須存在於 DESTINATION_CONFIGS。"
    )

APPROACH_HEIGHT = 0.10
GRASP_Z_OFFSET = 0.014
LIFT_OFFSET = np.array([0.0, 0.0, 0.12])
LIFT_FRAMES = 150

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
CLEAR_WARMUP_FRAMES = 10

COMPLETED_REQUEST_CACHE_SIZE = 200
MAX_INSTRUCTION_CHARS = 1000
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

# Action key：command, target, destination, normalized instruction
ActiveActionKey = Tuple[
    str,
    Optional[str],
    Optional[str],
    str,
]

active_task: Optional[Generator[str, None, None]] = None
active_action_key: Optional[ActiveActionKey] = None
active_action_label: Optional[str] = None
active_command: Optional[str] = None
active_target_name: Optional[str] = None
active_destination_name: Optional[str] = None

execution_running = False
execution_mode: Optional[str] = None  # "SLICE" 或 "FULL"
execution_steps_remaining = 0
execution_request_id: Optional[str] = None
execution_image_path: Optional[str] = None
execution_task_instruction: Optional[str] = None
execution_default_destination: Optional[str] = None
execution_continued = False
execution_switched = False

current_motion_phase = "IDLE"
slice_time_expired = False

desired_gripper_positions: Optional[np.ndarray] = (
    GRIPPER_OPEN_POSITIONS.copy()
)

client_conn: Optional[socket.socket] = None
client_addr: Optional[Tuple[str, int]] = None
request_buffer = bytearray()

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

def normalize_instruction(instruction: Any) -> str:
    """接受任意非空自然語言任務指令，不使用固定 instruction 白名單。"""
    if instruction is None:
        text = DEFAULT_TASK_INSTRUCTION
    elif isinstance(instruction, str):
        text = instruction.strip()
    else:
        raise ValueError("instruction 必須是字串。")

    if not text:
        raise ValueError("instruction 不可為空。")

    if len(text) > MAX_INSTRUCTION_CHARS:
        raise ValueError(
            f"instruction 過長，最多 {MAX_INSTRUCTION_CHARS} 個字元。"
        )

    return text


def normalize_destination(destination: Any) -> str:
    if not isinstance(destination, str):
        raise ValueError("目的地必須是字串。")

    normalized = destination.strip().lower()

    if normalized not in DESTINATION_CONFIGS:
        allowed = "、".join(sorted(DESTINATION_CONFIGS))
        raise ValueError(
            f"不支援的目的地 {normalized!r}；目前只支援：{allowed}。"
        )

    return normalized


def infer_destination_from_instruction(
    instruction: str,
) -> Optional[str]:
    lower = instruction.lower()
    matches = []

    for destination in DESTINATION_CONFIGS:
        if re.search(rf"\b{re.escape(destination)}\b", lower):
            matches.append(destination)

    if len(matches) == 1:
        return matches[0]

    return None


def resolve_default_destination(
    request: Dict[str, Any],
    task_instruction: str,
) -> str:
    """
    目的地解析優先順序：
    1. request["default_destination"]
    2. request["task"]["destination"]（相容未來結構化 TaskSpec）
    3. 從 instruction 中推定 tray 或 box
    4. DEFAULT_DESTINATION
    """
    raw_destination = request.get("default_destination")

    if raw_destination is not None:
        return normalize_destination(raw_destination)

    task_spec = request.get("task")
    if isinstance(task_spec, dict):
        task_destination = task_spec.get("destination")
        if task_destination is not None:
            return normalize_destination(task_destination)

    inferred = infer_destination_from_instruction(task_instruction)
    if inferred is not None:
        return inferred

    return DEFAULT_DESTINATION


def parse_action_label(
    action_label: str,
    task_instruction: str,
    default_destination: str,
) -> Tuple[
    str,
    Optional[str],
    Optional[str],
    str,
]:
    """
    回傳：
        command, target_name, destination_name, canonical_action_label
    """
    text = str(action_label).strip()
    lower = text.lower()

    if not text:
        raise ValueError("action_label 不可為空。")

    expected_finished_label = f"{task_instruction} finished"

    if lower in {
        "finished",
        expected_finished_label.lower(),
    }:
        return (
            "finished",
            None,
            None,
            expected_finished_label,
        )

    if lower == "clear":
        return "clear", None, None, "clear"

    if lower == "home":
        return "home", None, None, "home"

    if lower == "putdown":
        destination = normalize_destination(default_destination)
        return (
            "putdown",
            None,
            destination,
            f"putdown {destination}",
        )

    if lower.startswith("putdown"):
        parts = text.split(maxsplit=1)

        if len(parts) != 2 or not parts[1].strip():
            raise ValueError(
                "putdown 後面必須提供目的地，例如：putdown box"
            )

        destination = normalize_destination(parts[1])
        return (
            "putdown",
            None,
            destination,
            f"putdown {destination}",
        )

    if lower.startswith("pickup"):
        parts = text.split(maxsplit=1)

        if len(parts) != 2 or not parts[1].strip():
            raise ValueError("pickup 後面必須有物品名稱。")

        target_name = parts[1].strip()
        return (
            "pickup",
            target_name,
            None,
            f"pickup {target_name}",
        )

    raise ValueError(
        "不支援的 action_label。只接受 pickup <物品名稱>、putdown、"
        "putdown tray、putdown box、finished、clear 或 home。"
    )


def make_action_key(
    command: str,
    target_name: Optional[str],
    destination_name: Optional[str],
    task_instruction: str,
) -> ActiveActionKey:
    normalized_target = (
        target_name.strip().lower()
        if target_name is not None
        else None
    )
    normalized_destination = (
        destination_name.strip().lower()
        if destination_name is not None
        else None
    )
    normalized_instruction = " ".join(
        task_instruction.strip().lower().split()
    )

    return (
        command,
        normalized_target,
        normalized_destination,
        normalized_instruction,
    )


# =========================
# USD helpers
# =========================

def find_target_prim_path(target_name: str) -> str:
    stage = omni.usd.get_context().get_stage()
    lookup_key = target_name.strip().lower()

    configured_path = TARGET_PRIM_PATHS.get(lookup_key)

    if configured_path:
        prim = stage.GetPrimAtPath(configured_path)

        if prim and prim.IsValid():
            return configured_path

        raise ValueError(
            f"TARGET_PRIM_PATHS 指定的 Prim 不存在：{configured_path}"
        )

    candidate_names = [
        target_name.strip(),
        target_name.strip().replace(" ", "_"),
        "".join(part.capitalize() for part in target_name.split()),
    ]

    tried_paths = []
    for candidate_name in dict.fromkeys(candidate_names):
        direct_path = f"/World/{candidate_name}"
        tried_paths.append(direct_path)
        direct_prim = stage.GetPrimAtPath(direct_path)

        if direct_prim and direct_prim.IsValid():
            return direct_path

    matches = []

    for prim in stage.Traverse():
        prim_name = prim.GetName().strip().lower()
        normalized_prim_name = prim_name.replace("_", " ")

        if prim_name == lookup_key or normalized_prim_name == lookup_key:
            matches.append(str(prim.GetPath()))

    if len(matches) == 1:
        return matches[0]

    if len(matches) > 1:
        raise ValueError(
            f"物品名稱 {target_name!r} 對應到多個 Prim：{matches}。"
            "請在 TARGET_PRIM_PATHS 指定完整路徑。"
        )

    raise ValueError(
        f"找不到物品 {target_name!r}。已嘗試 {tried_paths}，"
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


def get_destination_position(destination_name: str) -> np.ndarray:
    destination = normalize_destination(destination_name)
    config = DESTINATION_CONFIGS[destination]

    prim_path = config.get("prim_path")
    place_offset = np.asarray(
        config.get("place_offset", np.zeros(3)),
        dtype=np.float64,
    )

    if place_offset.shape != (3,):
        raise ValueError(
            f"{destination} 的 place_offset 必須是長度 3 的向量。"
        )

    if prim_path:
        stage = omni.usd.get_context().get_stage()
        prim = stage.GetPrimAtPath(str(prim_path))

        if prim and prim.IsValid():
            return get_world_position(str(prim_path)) + place_offset

        raise ValueError(
            f"目的地 {destination!r} 的 Prim 不存在：{prim_path}"
        )

    fixed_position = np.asarray(
        config.get("fixed_position"),
        dtype=np.float64,
    )

    if fixed_position.shape != (3,):
        raise ValueError(
            f"{destination} 的 fixed_position 必須是長度 3 的向量。"
        )

    return fixed_position.copy() + place_offset


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

    JSONL 欄位維持：image_path、instruction、annotation.action_label。
    """
    for _ in range(CAPTURE_SETTLE_FRAMES):
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
    log(f"[DATA] Instruction : {task_instruction!r}")
    log(f"[DATA] Action label: {action_label!r}")
    debug(f"[DATA] request_id={request_id}")

    return str(final_path)


# =========================
# Hold control
# =========================

def hold_current_pose() -> None:
    """
    暫停手臂姿態，但不把夾爪改成當下的半閉合位置。
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
    global current_motion_phase
    current_motion_phase = phase
    return phase


def apply_gripper_target(target_positions: np.ndarray) -> None:
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
    if desired_gripper_positions is None:
        return

    apply_gripper_target(desired_gripper_positions)


def get_finger_positions() -> Optional[np.ndarray]:
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

    yield from move_to(
        lift_position,
        frames=LIFT_FRAMES,
        phase="LIFTING_OBJECT",
    )


def place_sequence(
    destination_name: str,
    destination_position: np.ndarray,
) -> Generator[str, None, None]:
    yield from move_to(
        destination_position,
        phase=f"MOVING_TO_{destination_name.upper()}",
    )
    yield from open_gripper()
    yield from go_home()


def build_new_task(
    command: str,
    target_name: Optional[str],
    destination_name: Optional[str],
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

    if command == "putdown":
        assert destination_name is not None

        if current_target_name is None:
            current_target_name = last_picked_target_name

        destination_position = get_destination_position(
            destination_name
        )

        log(
            f"[ROBOT] New putdown: destination={destination_name!r}, "
            f"position={destination_position}, "
            f"held_target={current_target_name!r}"
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
    global active_destination_name

    active_task = None
    active_action_key = None
    active_action_label = None
    active_command = None
    active_target_name = None
    active_destination_name = None


def clear_execution_state() -> None:
    global execution_running
    global execution_mode
    global execution_steps_remaining
    global execution_request_id
    global execution_image_path
    global execution_task_instruction
    global execution_default_destination
    global execution_continued
    global execution_switched
    global slice_time_expired

    execution_running = False
    execution_mode = None
    execution_steps_remaining = 0
    execution_request_id = None
    execution_image_path = None
    execution_task_instruction = None
    execution_default_destination = None
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


def reset_scene_to_usd_state() -> None:
    global current_target_name
    global last_picked_target_name
    global desired_gripper_positions
    global current_motion_phase

    log("[CLEAR] Resetting scene to the USD initial state.")

    clear_active_action()
    clear_execution_state()

    current_target_name = None
    last_picked_target_name = None
    current_motion_phase = "IDLE"
    desired_gripper_positions = GRIPPER_OPEN_POSITIONS.copy()

    world.reset()
    camera.initialize()
    rmpflow_controller.reset()

    set_desired_gripper_positions(GRIPPER_OPEN_POSITIONS)

    for _ in range(CLEAR_WARMUP_FRAMES):
        maintain_gripper_target()
        world.step(render=True)

    log("[CLEAR] Scene reset completed.")


# =========================
# Request execution
# =========================

def start_request(request: Dict[str, Any]) -> None:
    global active_task
    global active_action_key
    global active_action_label
    global active_command
    global active_target_name
    global active_destination_name

    global execution_running
    global execution_mode
    global execution_steps_remaining
    global execution_request_id
    global execution_image_path
    global execution_task_instruction
    global execution_default_destination
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

    try:
        task_instruction = normalize_instruction(instruction_raw)
        default_destination = resolve_default_destination(
            request=request,
            task_instruction=task_instruction,
        )
        (
            command,
            target_name,
            destination_name,
            action_label,
        ) = parse_action_label(
            action_label=action_label_raw,
            task_instruction=task_instruction,
            default_destination=default_destination,
        )

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
        destination_name=destination_name,
        task_instruction=task_instruction,
    )

    log(
        f"[REQUEST] id={request_id} | "
        f"instruction={task_instruction!r} | "
        f"default_destination={default_destination!r} | "
        f"action_label={action_label!r}"
    )

    # clear：不拍照、不寫入 JSONL。
    if command == "clear":
        try:
            reset_scene_to_usd_state()

        except Exception as exc:
            log_exception("[CLEAR] Scene reset failed", exc)

            reply = make_reply(
                request_id,
                "ERROR",
                execution_state="CLEAR_FAILED",
                command=command,
                action_label=action_label,
                instruction=task_instruction,
                default_destination=default_destination,
                message=(
                    "場景恢復失敗："
                    f"{type(exc).__name__}: {exc}"
                ),
                recorded=False,
                motion_executed=False,
                action_completed=False,
            )

            cache_completed_reply(request_id, reply)
            send_reply_and_close(reply)
            return

        reply = make_reply(
            request_id,
            "SUCCESS",
            execution_state="CLEAR_COMPLETED",
            command=command,
            action_label=action_label,
            instruction=task_instruction,
            default_destination=default_destination,
            destination=None,
            recorded=False,
            image_path=None,
            motion_executed=True,
            action_completed=True,
            scene_reset=True,
        )

        cache_completed_reply(request_id, reply)
        send_reply_and_close(reply)
        return

    # 在寫資料前先驗證 pickup 物品或 putdown 目的地。
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
                    instruction=task_instruction,
                    action_label=action_label,
                    message=str(exc),
                    recorded=False,
                )
            )
            return

    if command == "putdown":
        try:
            assert destination_name is not None
            get_destination_position(destination_name)

        except Exception as exc:
            log_exception(
                "[REQUEST] Cannot resolve putdown destination",
                exc,
            )
            send_reply_and_close(
                make_reply(
                    request_id,
                    "ERROR",
                    instruction=task_instruction,
                    action_label=action_label,
                    destination=destination_name,
                    message=str(exc),
                    recorded=False,
                )
            )
            return

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
                instruction=task_instruction,
                action_label=action_label,
                recorded=False,
            )
        )
        return

    # finished：只記錄，不執行機械手臂動作。
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
            default_destination=default_destination,
            destination=None,
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

    # home：取消未完成動作，直接跑到完成。
    if command == "home":
        switched = active_task is not None

        if active_task is not None:
            cancel_active_action()

        try:
            active_task = build_new_task(
                command="home",
                target_name=None,
                destination_name=None,
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
                default_destination=default_destination,
                destination=None,
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
        active_destination_name = None

        execution_running = True
        execution_mode = "FULL"
        execution_steps_remaining = 0
        execution_request_id = request_id
        execution_image_path = image_path
        execution_task_instruction = task_instruction
        execution_default_destination = default_destination
        execution_continued = False
        execution_switched = switched
        slice_time_expired = False

        log("[EXECUTION] Home will run directly until completion.")
        return

    # pickup / putdown：同一 instruction、物品與目的地才續跑。
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
                destination_name=destination_name,
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
                default_destination=default_destination,
                destination=destination_name,
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
        active_destination_name = destination_name
        continued = False

    execution_running = True
    execution_mode = "SLICE"
    execution_steps_remaining = ACTION_SLICE_STEPS
    execution_request_id = request_id
    execution_image_path = image_path
    execution_task_instruction = task_instruction
    execution_default_destination = default_destination
    execution_continued = continued
    execution_switched = switched
    slice_time_expired = False

    log(
        f"[EXECUTION] Start {ACTION_SLICE_SECONDS:.1f}-second slice: "
        f"steps={ACTION_SLICE_STEPS}, "
        f"continued={continued}, switched={switched}"
    )


def finish_paused() -> None:
    assert execution_request_id is not None
    assert active_command is not None
    assert active_action_label is not None
    assert execution_task_instruction is not None
    assert execution_default_destination is not None

    request_id = execution_request_id
    image_path = execution_image_path
    instruction = execution_task_instruction
    default_destination = execution_default_destination
    command = active_command
    action_label = active_action_label
    destination = active_destination_name
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
        default_destination=default_destination,
        destination=destination,
        recorded=True,
        image_path=image_path,
        motion_executed=True,
        action_completed=False,
        continued=continued,
        switched=switched,
        slice_seconds=ACTION_SLICE_SECONDS,
        remaining_action=active_action_label,
        motion_phase=current_motion_phase,
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
    assert execution_default_destination is not None

    request_id = execution_request_id
    image_path = execution_image_path
    instruction = execution_task_instruction
    default_destination = execution_default_destination
    command = active_command
    action_label = active_action_label
    destination = active_destination_name
    continued = execution_continued
    switched = execution_switched
    mode = execution_mode

    if command in {"putdown", "home"}:
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
        default_destination=default_destination,
        destination=destination,
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
        motion_phase=current_motion_phase,
    )

    cache_completed_reply(request_id, reply)
    clear_execution_state()
    send_reply_and_close(reply)


def fail_execution(exc: BaseException) -> None:
    request_id = execution_request_id
    image_path = execution_image_path
    instruction = execution_task_instruction
    default_destination = execution_default_destination
    command = active_command
    action_label = active_action_label
    destination = active_destination_name

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
        default_destination=default_destination,
        destination=destination,
        message=(
            "機械手臂動作失敗："
            f"{type(exc).__name__}: {exc}"
        ),
        recorded=image_path is not None,
        image_path=image_path,
        motion_executed=False,
        action_completed=False,
        motion_phase=current_motion_phase,
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

    一般階段：2 秒到達後立即暫停。
    夾爪閉合階段：時間到達後仍完成閉合，再停在
    GRIPPER_CLOSED_BOUNDARY，不會直接開始抬升。
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

        if phase == "GRIPPER_CLOSING":
            debug(
                "[EXECUTION] Time expired during gripper closing; "
                "continue until fully closed."
            )
            return

        if phase == "GRIPPER_CLOSED_BOUNDARY":
            log(
                "[EXECUTION] Gripper closing completed after time limit; "
                "pause before lifting."
            )
            finish_paused()
            return

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
    log("=" * 80)
    log("Isaac Sim Manual Record Server — Dynamic Task Instructions")
    log("Instruction policy  : any non-empty natural-language string")
    log(
        "Destinations       : "
        f"{sorted(DESTINATION_CONFIGS.keys())}"
    )
    log(
        f"Default destination: {DEFAULT_DESTINATION!r}"
    )
    log(
        f"Action slice       : {ACTION_SLICE_SECONDS:.1f} "
        "simulated seconds"
    )
    log("Close behavior     : finish gripper closing after time limit")
    log("Home behavior      : run directly until completion")
    log("Clear behavior     : reset scene; do not record JSONL")

    for name in sorted(DESTINATION_CONFIGS):
        try:
            position = get_destination_position(name)
            log(f"Destination {name:<5}: {position}")
        except Exception as exc:
            log(f"Destination {name:<5}: CONFIG ERROR — {exc}")

    log(f"Gripper open pos   : {GRIPPER_OPEN_POSITIONS}")
    log(f"Gripper close pos  : {GRIPPER_CLOSED_POSITIONS}")
    log("Grip hold          : maintain target during pause/lift/move")
    log(f"Lift offset        : {LIFT_OFFSET}")
    log(f"Lift frames        : {LIFT_FRAMES}")
    log(f"Image directory    : {SAVE_DIR}")
    log(f"JSONL file         : {JSONL_PATH}")
    log("=" * 80)

    while simulation_app.is_running():
        accept_client_if_idle()
        read_single_request_if_available()

        tick_execution()
        maintain_gripper_target()
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
