#!/usr/bin/env python3
"""
isaac_manual_record_server_three_step_1s.py

三步驟動作流程：
    1. pick <object>
       快速打開夾爪，移動到物品抓取位置，以原本速度關閉夾爪。
    2. up
       將上一個 pick 的物品垂直抬升到原物品位置上方。
    3. putdown tray / putdown box
       移動到目的地，到達後自動快速打開夾爪。

速度設定：
- GRIPPER_OPEN_FRAMES = 10：由原本 30 frames 改快。
- GRIPPER_CLOSE_FRAMES = 90：保持原本速度不變。

資料收集：
- 每一筆新動作先拍照並寫入 JSONL，再執行或續跑 1 秒。
- 相同動作再次送入時，從 generator 暫停位置續跑。
- 關閉或開啟夾爪時，不會停在半開或半閉狀態。
- clear 不記錄；finished 只記錄；home 直接執行到完成。
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

DEFAULT_TASK_INSTRUCTION = "put the orange into the box"
DEFAULT_DESTINATION = "box"

CAMERA_PRIM_PATH = "/World/Camera"
FRANKA_PRIM_PATH = "/World/Franka"
CAMERA_RESOLUTION = (640, 480)

ACTION_SLICE_SECONDS = 1.0
DEFAULT_PHYSICS_DT = 1.0 / 60.0

TARGET_PRIM_PATHS: Dict[str, str] = {
    # "orange": "/World/Orange",
    # "bottle": "/World/Bottle",
    # "red cube": "/World/RedCube",
}

TARGET_GRASP_OFFSETS: Dict[str, np.ndarray] = {
    # "orange": np.array([0.0, 0.0, 0.01]),
}

VERTICAL_Q = np.array([0.0, 1.0, 0.0, 0.0])
HIGH_HOME_POS = np.array([0.4, 0.0, 0.6], dtype=np.float64)

DESTINATION_CONFIGS: Dict[str, Dict[str, Any]] = {
    "tray": {
        "prim_path": None,
        "fixed_position": np.array([-0.12, -0.7, 0.45], dtype=np.float64),
        "place_offset": np.zeros(3, dtype=np.float64),
    },
    "box": {
        "prim_path": None,
        "fixed_position": np.array([-0.12, 0.7, 0.45], dtype=np.float64),
        "place_offset": np.zeros(3, dtype=np.float64),
    },
}

if DEFAULT_DESTINATION not in DESTINATION_CONFIGS:
    raise ValueError("DEFAULT_DESTINATION 必須存在於 DESTINATION_CONFIGS。")

APPROACH_HEIGHT = 0.10
GRASP_Z_OFFSET = 0.014
LIFT_OFFSET = np.array([0.0, 0.0, 0.12], dtype=np.float64)

MOVE_APPROACH_FRAMES = 100
MOVE_DESCEND_FRAMES = 100
UP_LIFT_FRAMES = 150
PUTDOWN_MOVE_FRAMES = 100
HOME_FRAMES = 30

# 開夾爪加快：原本 30 frames，現在 10 frames。
GRIPPER_OPEN_FRAMES = 10

# 關夾爪保持原本速度。
GRIPPER_CLOSE_FRAMES = 90

GRIPPER_OPEN_POSITIONS = np.array([0.035, 0.035], dtype=np.float64)
GRIPPER_CLOSED_POSITIONS = np.array([0.0, 0.0], dtype=np.float64)

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

    if omni.usd.get_context().get_stage() is None:
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
franka = world.scene.add(Franka(prim_path=FRANKA_PRIM_PATH, name="franka"))
camera = Camera(prim_path=CAMERA_PRIM_PATH, resolution=CAMERA_RESOLUTION)

world.reset()
camera.initialize()

rmpflow_controller = RMPFlowController(
    name="three_step_flow_rmpflow",
    robot_articulation=franka,
)

for _ in range(CAMERA_WARMUP_FRAMES):
    world.step(render=True)

try:
    physics_dt = float(world.get_physics_dt())
    if physics_dt <= 0.0:
        raise ValueError
except Exception:
    physics_dt = DEFAULT_PHYSICS_DT

ACTION_SLICE_STEPS = max(1, int(round(ACTION_SLICE_SECONDS / physics_dt)))

log("[INIT] Isaac Sim scene, Franka and camera are ready.")
log(f"[INIT] physics_dt={physics_dt:.6f}, slice_steps={ACTION_SLICE_STEPS}")


# =========================
# Persistent workflow state
# =========================

current_target_name: Optional[str] = None
current_target_prim_path: Optional[str] = None
target_original_position: Optional[np.ndarray] = None
target_grasp_position: Optional[np.ndarray] = None
target_lift_position: Optional[np.ndarray] = None

pick_completed = False
up_completed = False
held_target_name: Optional[str] = None

ActiveActionKey = Tuple[str, Optional[str], Optional[str], str]

active_task: Optional[Generator[str, None, None]] = None
active_action_key: Optional[ActiveActionKey] = None
active_action_label: Optional[str] = None
active_command: Optional[str] = None
active_target_name: Optional[str] = None
active_destination_name: Optional[str] = None

execution_running = False
execution_mode: Optional[str] = None
execution_steps_remaining = 0
execution_request_id: Optional[str] = None
execution_image_path: Optional[str] = None
execution_task_instruction: Optional[str] = None
execution_default_destination: Optional[str] = None
execution_continued = False
execution_switched = False

current_motion_phase = "IDLE"
slice_time_expired = False
desired_gripper_positions: Optional[np.ndarray] = GRIPPER_OPEN_POSITIONS.copy()

client_conn: Optional[socket.socket] = None
client_addr: Optional[Tuple[str, int]] = None
request_buffer = bytearray()

completed_requests: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()


# =========================
# Socket setup / helpers
# =========================

server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
server_socket.bind((HOST, PORT))
server_socket.listen(LISTEN_BACKLOG)
server_socket.setblocking(False)

log(f"[SOCKET] Listening on {HOST}:{PORT}")


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
        encoded = (json.dumps(reply, ensure_ascii=False) + "\n").encode("utf-8")
        try:
            client_conn.setblocking(True)
            client_conn.settimeout(5.0)
            client_conn.sendall(encoded)
        except OSError as exc:
            log_exception("[SOCKET] Failed to send reply", exc)
    close_client("reply completed")


def make_reply(request_id: Optional[str], status: str, **extra: Any) -> Dict[str, Any]:
    reply: Dict[str, Any] = {"request_id": request_id, "status": status}
    reply.update(extra)
    return reply


def cache_completed_reply(request_id: str, reply: Dict[str, Any]) -> None:
    completed_requests[request_id] = dict(reply)
    completed_requests.move_to_end(request_id)
    while len(completed_requests) > COMPLETED_REQUEST_CACHE_SIZE:
        completed_requests.popitem(last=False)


# =========================
# Parsing
# =========================

def normalize_instruction(instruction: Any) -> str:
    if instruction is None:
        text = DEFAULT_TASK_INSTRUCTION
    elif isinstance(instruction, str):
        text = instruction.strip()
    else:
        raise ValueError("instruction 必須是字串。")

    if not text:
        raise ValueError("instruction 不可為空。")
    if len(text) > MAX_INSTRUCTION_CHARS:
        raise ValueError(f"instruction 過長，最多 {MAX_INSTRUCTION_CHARS} 個字元。")
    return text


def normalize_destination(destination: Any) -> str:
    if not isinstance(destination, str):
        raise ValueError("目的地必須是字串。")
    normalized = destination.strip().lower()
    if normalized not in DESTINATION_CONFIGS:
        allowed = "、".join(sorted(DESTINATION_CONFIGS))
        raise ValueError(f"不支援的目的地 {normalized!r}；目前只支援：{allowed}。")
    return normalized


def infer_destination_from_instruction(instruction: str) -> Optional[str]:
    lower = instruction.lower()
    matches = [
        destination
        for destination in DESTINATION_CONFIGS
        if re.search(rf"\b{re.escape(destination)}\b", lower)
    ]
    return matches[0] if len(matches) == 1 else None


def resolve_default_destination(request: Dict[str, Any], task_instruction: str) -> str:
    raw_destination = request.get("default_destination")
    if raw_destination is not None:
        return normalize_destination(raw_destination)
    inferred = infer_destination_from_instruction(task_instruction)
    return inferred if inferred is not None else DEFAULT_DESTINATION


def parse_action_label(
    action_label: str,
    task_instruction: str,
    default_destination: str,
) -> Tuple[str, Optional[str], Optional[str], str]:
    text = str(action_label).strip()
    lower = text.lower()

    if not text:
        raise ValueError("action_label 不可為空。")

    expected_finished_label = f"{task_instruction} finished"

    if lower in {"finished", expected_finished_label.lower()}:
        return "finished", None, None, expected_finished_label
    if lower == "clear":
        return "clear", None, None, "clear"
    if lower == "home":
        return "home", None, None, "home"
    if lower == "up":
        return "up", None, None, "up"

    if lower.startswith("pickup"):
        raise ValueError("新流程不使用 pickup。請改用 pick <物品名稱>，下一步使用 up。")

    if lower.startswith("pick"):
        parts = text.split(maxsplit=1)
        if lower == "pick" or len(parts) != 2 or not parts[1].strip():
            raise ValueError("pick 後面必須提供物品名稱，例如：pick orange")
        target = parts[1].strip()
        return "pick", target, None, f"pick {target}"

    if lower == "putdown":
        destination = normalize_destination(default_destination)
        return "putdown", None, destination, f"putdown {destination}"

    if lower.startswith("putdown"):
        parts = text.split(maxsplit=1)
        if len(parts) != 2 or not parts[1].strip():
            raise ValueError("putdown 後面必須提供目的地，例如：putdown box")
        destination = normalize_destination(parts[1])
        return "putdown", None, destination, f"putdown {destination}"

    raise ValueError(
        "只接受 pick <物品名稱>、up、putdown、putdown tray、"
        "putdown box、finished、clear 或 home。"
    )


def make_action_key(
    command: str,
    internal_target_name: Optional[str],
    destination_name: Optional[str],
    task_instruction: str,
) -> ActiveActionKey:
    target = internal_target_name.strip().lower() if internal_target_name else None
    destination = destination_name.strip().lower() if destination_name else None
    instruction = " ".join(task_instruction.strip().lower().split())
    return command, target, destination, instruction


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
        raise ValueError(f"TARGET_PRIM_PATHS 指定的 Prim 不存在：{configured_path}")

    candidate_names = [
        target_name.strip(),
        target_name.strip().replace(" ", "_"),
        "".join(part.capitalize() for part in target_name.split()),
    ]

    tried_paths = []
    for candidate_name in dict.fromkeys(candidate_names):
        direct_path = f"/World/{candidate_name}"
        tried_paths.append(direct_path)
        prim = stage.GetPrimAtPath(direct_path)
        if prim and prim.IsValid():
            return direct_path

    matches = []
    for prim in stage.Traverse():
        prim_name = prim.GetName().strip().lower()
        if prim_name == lookup_key or prim_name.replace("_", " ") == lookup_key:
            matches.append(str(prim.GetPath()))

    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise ValueError(
            f"物品名稱 {target_name!r} 對應到多個 Prim：{matches}。"
            "請在 TARGET_PRIM_PATHS 指定完整路徑。"
        )
    raise ValueError(f"找不到物品 {target_name!r}。已嘗試 {tried_paths}。")


def get_world_position(prim_path: str) -> np.ndarray:
    stage = omni.usd.get_context().get_stage()
    prim = stage.GetPrimAtPath(prim_path)
    if not prim or not prim.IsValid():
        raise ValueError(f"Prim 不存在：{prim_path}")

    matrix = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(
        Usd.TimeCode.Default()
    )
    translation = matrix.ExtractTranslation()
    return np.array(
        [translation[0], translation[1], translation[2]], dtype=np.float64
    )


def get_destination_position(destination_name: str) -> np.ndarray:
    destination = normalize_destination(destination_name)
    config = DESTINATION_CONFIGS[destination]
    place_offset = np.asarray(config.get("place_offset", np.zeros(3)), dtype=np.float64)

    if place_offset.shape != (3,):
        raise ValueError(f"{destination} 的 place_offset 必須是長度 3 的向量。")

    prim_path = config.get("prim_path")
    if prim_path:
        return get_world_position(str(prim_path)) + place_offset

    fixed_position = np.asarray(config.get("fixed_position"), dtype=np.float64)
    if fixed_position.shape != (3,):
        raise ValueError(f"{destination} 的 fixed_position 必須是長度 3 的向量。")
    return fixed_position.copy() + place_offset


# =========================
# Image and JSONL
# =========================

def rgba_to_uint8_rgb(rgba: np.ndarray) -> np.ndarray:
    if rgba.ndim != 3 or rgba.shape[2] < 3:
        raise ValueError(f"Camera frame shape 不正確：{rgba.shape}")

    rgb = np.asarray(rgba[:, :, :3])
    if rgb.dtype == np.uint8:
        return rgb

    rgb = rgb.astype(np.float32)
    if rgb.size > 0 and float(np.nanmax(rgb)) <= 1.0:
        rgb *= 255.0
    rgb = np.nan_to_num(rgb, nan=0.0, posinf=255.0, neginf=0.0)
    return np.clip(rgb, 0.0, 255.0).astype(np.uint8)


def safe_filename_component(text: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", text.strip()).strip("._-")
    return cleaned or "unknown"


def resolve_episode_target_name(parsed_target_name: Optional[str]) -> str:
    if parsed_target_name:
        return parsed_target_name
    if held_target_name:
        return held_target_name
    if current_target_name:
        return current_target_name
    return "unknown"


def capture_and_append_jsonl(
    action_label: str,
    task_instruction: str,
    parsed_target_name: Optional[str],
    request_id: str,
) -> str:
    for _ in range(CAPTURE_SETTLE_FRAMES):
        maintain_gripper_target()
        world.step(render=True)

    rgba = camera.get_rgba()
    if rgba is None:
        raise RuntimeError("camera.get_rgba() returned None。")

    rgb = rgba_to_uint8_rgb(np.asarray(rgba))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    target_component = safe_filename_component(
        resolve_episode_target_name(parsed_target_name)
    )
    action_component = safe_filename_component(action_label)

    final_path = SAVE_DIR / (
        f"capture_{target_component}_{action_component}_{timestamp}.png"
    )
    temp_path = final_path.with_suffix(".tmp.png")
    Image.fromarray(rgb).save(temp_path)
    os.replace(temp_path, final_path)

    entry = {
        "image_path": str(final_path),
        "instruction": task_instruction,
        "annotation": {"action_label": action_label},
    }

    with JSONL_PATH.open("a", encoding="utf-8") as file:
        file.write(json.dumps(entry, ensure_ascii=False) + "\n")
        file.flush()
        os.fsync(file.fileno())

    log(f"[DATA] Image saved : {final_path}")
    log(f"[DATA] Action label: {action_label!r}")
    debug(f"[DATA] request_id={request_id}")
    return str(final_path)


# =========================
# Gripper / motion control
# =========================

def set_motion_phase(phase: str) -> str:
    global current_motion_phase
    current_motion_phase = phase
    return phase


def apply_gripper_target(target_positions: np.ndarray) -> None:
    target_positions = np.asarray(target_positions, dtype=np.float64)
    if target_positions.shape != (2,):
        raise ValueError("夾爪目標必須包含兩個手指關節位置。")
    franka.gripper.apply_action(
        ArticulationAction(joint_positions=target_positions.copy())
    )


def set_desired_gripper_positions(target_positions: np.ndarray) -> None:
    global desired_gripper_positions
    desired_gripper_positions = np.asarray(target_positions, dtype=np.float64).copy()
    apply_gripper_target(desired_gripper_positions)


def maintain_gripper_target() -> None:
    if desired_gripper_positions is not None:
        apply_gripper_target(desired_gripper_positions)


def hold_current_pose() -> None:
    joint_positions = franka.get_joint_positions()
    if joint_positions is None:
        maintain_gripper_target()
        return

    joint_positions = np.asarray(joint_positions, dtype=np.float64)
    arm_dof_count = max(0, joint_positions.size - 2)

    try:
        if arm_dof_count > 0:
            arm_indices = np.arange(arm_dof_count, dtype=np.int64)
            action = ArticulationAction(
                joint_positions=joint_positions[:arm_dof_count].copy(),
                joint_velocities=np.zeros(arm_dof_count, dtype=np.float64),
                joint_indices=arm_indices,
            )
        else:
            action = ArticulationAction(
                joint_positions=joint_positions.copy(),
                joint_velocities=np.zeros_like(joint_positions),
            )
        franka.apply_action(action)
    except TypeError:
        franka.apply_action(
            ArticulationAction(
                joint_positions=joint_positions.copy(),
                joint_velocities=np.zeros_like(joint_positions),
            )
        )

    maintain_gripper_target()


def move_to(
    target_position: np.ndarray,
    frames: int,
    phase: str,
) -> Generator[str, None, None]:
    rmpflow_controller.reset()
    for _ in range(frames):
        action = rmpflow_controller.forward(
            target_end_effector_position=target_position,
            target_end_effector_orientation=VERTICAL_Q,
        )
        franka.apply_action(action)
        yield set_motion_phase(phase)


def open_gripper(frames: int = GRIPPER_OPEN_FRAMES) -> Generator[str, None, None]:
    set_desired_gripper_positions(GRIPPER_OPEN_POSITIONS)
    for _ in range(frames):
        maintain_gripper_target()
        yield set_motion_phase("GRIPPER_OPENING")
    yield set_motion_phase("GRIPPER_OPENED_BOUNDARY")


def close_gripper(frames: int = GRIPPER_CLOSE_FRAMES) -> Generator[str, None, None]:
    set_desired_gripper_positions(GRIPPER_CLOSED_POSITIONS)
    for _ in range(frames):
        maintain_gripper_target()
        yield set_motion_phase("GRIPPER_CLOSING")
    yield set_motion_phase("GRIPPER_CLOSED_BOUNDARY")


def pick_sequence(grasp_position: np.ndarray) -> Generator[str, None, None]:
    approach_position = grasp_position + np.array([0.0, 0.0, APPROACH_HEIGHT])

    # 快速完整開夾爪。
    yield from open_gripper()
    yield from move_to(
        approach_position,
        frames=MOVE_APPROACH_FRAMES,
        phase="PICK_APPROACHING_TARGET",
    )
    yield from move_to(
        grasp_position,
        frames=MOVE_DESCEND_FRAMES,
        phase="PICK_DESCENDING_TO_GRASP",
    )

    # 關閉速度維持 90 frames。
    yield from close_gripper()


def up_sequence(lift_position: np.ndarray) -> Generator[str, None, None]:
    set_desired_gripper_positions(GRIPPER_CLOSED_POSITIONS)
    yield from move_to(
        lift_position,
        frames=UP_LIFT_FRAMES,
        phase="UP_LIFTING_OBJECT",
    )


def putdown_sequence(
    destination_name: str,
    destination_position: np.ndarray,
) -> Generator[str, None, None]:
    set_desired_gripper_positions(GRIPPER_CLOSED_POSITIONS)
    yield from move_to(
        destination_position,
        frames=PUTDOWN_MOVE_FRAMES,
        phase=f"PUTDOWN_MOVING_TO_{destination_name.upper()}",
    )

    # 到達目的地後自動快速開夾爪。
    yield from open_gripper()


def home_sequence() -> Generator[str, None, None]:
    yield from move_to(HIGH_HOME_POS, frames=HOME_FRAMES, phase="HOME_MOVING")
    yield from open_gripper()


# =========================
# Workflow helpers
# =========================

def clear_active_action() -> None:
    global active_task, active_action_key, active_action_label
    global active_command, active_target_name, active_destination_name

    active_task = None
    active_action_key = None
    active_action_label = None
    active_command = None
    active_target_name = None
    active_destination_name = None


def clear_object_state() -> None:
    global current_target_name, current_target_prim_path
    global target_original_position, target_grasp_position, target_lift_position
    global pick_completed, up_completed, held_target_name

    current_target_name = None
    current_target_prim_path = None
    target_original_position = None
    target_grasp_position = None
    target_lift_position = None
    pick_completed = False
    up_completed = False
    held_target_name = None


def clear_execution_state() -> None:
    global execution_running, execution_mode, execution_steps_remaining
    global execution_request_id, execution_image_path, execution_task_instruction
    global execution_default_destination, execution_continued, execution_switched
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
    clear_active_action()
    hold_current_pose()


def reset_scene_to_usd_state() -> None:
    global desired_gripper_positions, current_motion_phase

    clear_active_action()
    clear_execution_state()
    clear_object_state()

    current_motion_phase = "IDLE"
    desired_gripper_positions = GRIPPER_OPEN_POSITIONS.copy()

    world.reset()
    camera.initialize()
    rmpflow_controller.reset()
    set_desired_gripper_positions(GRIPPER_OPEN_POSITIONS)

    for _ in range(CLEAR_WARMUP_FRAMES):
        maintain_gripper_target()
        world.step(render=True)


def validate_workflow_before_record(
    command: str,
    parsed_target_name: Optional[str],
    destination_name: Optional[str],
) -> None:
    if command == "pick":
        assert parsed_target_name is not None
        find_target_prim_path(parsed_target_name)
        return

    if command == "up":
        if current_target_name is None or target_lift_position is None:
            raise ValueError("尚未選定物品。請先完成 pick <物品名稱>。")
        if not pick_completed:
            raise ValueError(
                "pick 尚未完成。請繼續輸入相同 pick 指令，"
                "直到 execution_state=COMPLETED，再輸入 up。"
            )
        return

    if command == "putdown":
        assert destination_name is not None
        get_destination_position(destination_name)
        if held_target_name is None or not up_completed:
            raise ValueError(
                "物品尚未完成 up。請依序完成 pick <物品名稱>、up，"
                "再輸入 putdown tray 或 putdown box。"
            )


def prepare_new_pick(target_name: str) -> Generator[str, None, None]:
    global current_target_name, current_target_prim_path
    global target_original_position, target_grasp_position, target_lift_position
    global pick_completed, up_completed, held_target_name

    prim_path = find_target_prim_path(target_name)
    original_position = get_world_position(prim_path)
    grasp_offset = TARGET_GRASP_OFFSETS.get(
        target_name.strip().lower(), np.zeros(3, dtype=np.float64)
    )
    grasp_position = (
        original_position
        + grasp_offset
        + np.array([0.0, 0.0, GRASP_Z_OFFSET])
    )
    lift_position = grasp_position + LIFT_OFFSET

    current_target_name = target_name
    current_target_prim_path = prim_path
    target_original_position = original_position.copy()
    target_grasp_position = grasp_position.copy()
    target_lift_position = lift_position.copy()
    pick_completed = False
    up_completed = False
    held_target_name = None

    log(
        f"[ROBOT] New pick: target={target_name!r}, "
        f"grasp={grasp_position}, lift={lift_position}"
    )
    return pick_sequence(grasp_position)


def build_new_task(
    command: str,
    parsed_target_name: Optional[str],
    destination_name: Optional[str],
) -> Generator[str, None, None]:
    if command == "pick":
        assert parsed_target_name is not None
        return prepare_new_pick(parsed_target_name)
    if command == "up":
        assert target_lift_position is not None
        return up_sequence(target_lift_position.copy())
    if command == "putdown":
        assert destination_name is not None
        return putdown_sequence(
            destination_name,
            get_destination_position(destination_name),
        )
    if command == "home":
        return home_sequence()
    raise ValueError(f"無法建立 command={command!r} 的動作。")


# =========================
# Request execution
# =========================

def start_request(request: Dict[str, Any]) -> None:
    global active_task, active_action_key, active_action_label
    global active_command, active_target_name, active_destination_name
    global execution_running, execution_mode, execution_steps_remaining
    global execution_request_id, execution_image_path, execution_task_instruction
    global execution_default_destination, execution_continued, execution_switched
    global slice_time_expired

    request_id_raw = request.get("request_id")
    action_label_raw = request.get("action_label")
    instruction_raw = request.get("instruction")

    if not isinstance(request_id_raw, str) or not request_id_raw.strip():
        send_reply_and_close(make_reply(None, "ERROR", message="request_id 必須是非空字串。"))
        return

    request_id = request_id_raw.strip()

    if request_id in completed_requests:
        cached = dict(completed_requests[request_id])
        cached["execution_state"] = "DUPLICATE"
        cached["duplicate_request"] = True
        send_reply_and_close(cached)
        return

    if not isinstance(action_label_raw, str):
        send_reply_and_close(
            make_reply(request_id, "ERROR", message="action_label 必須是字串。")
        )
        return

    try:
        task_instruction = normalize_instruction(instruction_raw)
        default_destination = resolve_default_destination(request, task_instruction)
        command, parsed_target_name, destination_name, action_label = parse_action_label(
            action_label_raw,
            task_instruction,
            default_destination,
        )
    except ValueError as exc:
        send_reply_and_close(make_reply(request_id, "ERROR", message=str(exc)))
        return

    internal_target_name = (
        parsed_target_name if command == "pick" else current_target_name
    )
    new_action_key = make_action_key(
        command,
        internal_target_name,
        destination_name,
        task_instruction,
    )

    log(
        f"[REQUEST] id={request_id} | instruction={task_instruction!r} | "
        f"action_label={action_label!r}"
    )

    if command == "clear":
        try:
            reset_scene_to_usd_state()
            reply = make_reply(
                request_id,
                "SUCCESS",
                execution_state="CLEAR_COMPLETED",
                command=command,
                action_label=action_label,
                instruction=task_instruction,
                recorded=False,
                image_path=None,
                motion_executed=True,
                action_completed=True,
                scene_reset=True,
            )
        except Exception as exc:
            reply = make_reply(
                request_id,
                "ERROR",
                execution_state="CLEAR_FAILED",
                message=f"場景恢復失敗：{type(exc).__name__}: {exc}",
                recorded=False,
            )
        cache_completed_reply(request_id, reply)
        send_reply_and_close(reply)
        return

    if command in {"pick", "up", "putdown"}:
        try:
            validate_workflow_before_record(
                command,
                parsed_target_name,
                destination_name,
            )
        except Exception as exc:
            send_reply_and_close(
                make_reply(
                    request_id,
                    "ERROR",
                    command=command,
                    action_label=action_label,
                    instruction=task_instruction,
                    message=str(exc),
                    recorded=False,
                )
            )
            return

    try:
        image_path = capture_and_append_jsonl(
            action_label,
            task_instruction,
            parsed_target_name,
            request_id,
        )
    except Exception as exc:
        send_reply_and_close(
            make_reply(
                request_id,
                "ERROR",
                message=f"資料記錄失敗：{type(exc).__name__}: {exc}",
                recorded=False,
            )
        )
        return

    if command == "finished":
        cancel_active_action()
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
            slice_seconds=0.0,
        )
        cache_completed_reply(request_id, reply)
        send_reply_and_close(reply)
        return

    if command == "home":
        if active_task is not None:
            cancel_active_action()
        active_task = build_new_task("home", None, None)
        active_action_key = new_action_key
        active_action_label = action_label
        active_command = command
        active_target_name = None
        active_destination_name = None

        execution_running = True
        execution_mode = "FULL"
        execution_request_id = request_id
        execution_image_path = image_path
        execution_task_instruction = task_instruction
        execution_default_destination = default_destination
        execution_continued = False
        execution_switched = False
        return

    if active_task is not None and active_action_key == new_action_key:
        continued = True
        switched = False
    else:
        switched = active_task is not None
        if active_task is not None:
            cancel_active_action()
        try:
            active_task = build_new_task(
                command,
                parsed_target_name,
                destination_name,
            )
        except Exception as exc:
            reply = make_reply(
                request_id,
                "ERROR",
                execution_state="BUILD_FAILED",
                message=f"建立動作失敗：{type(exc).__name__}: {exc}",
                recorded=True,
                image_path=image_path,
            )
            cache_completed_reply(request_id, reply)
            send_reply_and_close(reply)
            return

        active_action_key = new_action_key
        active_action_label = action_label
        active_command = command
        active_target_name = internal_target_name
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


def finish_paused() -> None:
    assert execution_request_id is not None
    assert active_command is not None
    assert active_action_label is not None
    assert execution_task_instruction is not None

    request_id = execution_request_id
    hold_current_pose()

    reply = make_reply(
        request_id,
        "SUCCESS",
        execution_state="PAUSED",
        command=active_command,
        action_label=active_action_label,
        instruction=execution_task_instruction,
        destination=active_destination_name,
        target=current_target_name,
        recorded=True,
        image_path=execution_image_path,
        motion_executed=True,
        action_completed=False,
        continued=execution_continued,
        switched=execution_switched,
        slice_seconds=ACTION_SLICE_SECONDS,
        remaining_action=active_action_label,
        motion_phase=current_motion_phase,
        pick_completed=pick_completed,
        up_completed=up_completed,
    )

    cache_completed_reply(request_id, reply)
    clear_execution_state()
    send_reply_and_close(reply)


def finish_completed() -> None:
    global pick_completed, up_completed, held_target_name

    assert execution_request_id is not None
    assert active_command is not None
    assert active_action_label is not None
    assert execution_task_instruction is not None

    request_id = execution_request_id
    command = active_command
    mode = execution_mode
    action_label = active_action_label
    destination = active_destination_name
    instruction = execution_task_instruction
    image_path = execution_image_path
    continued = execution_continued
    switched = execution_switched
    completed_target = current_target_name

    if command == "pick":
        pick_completed = True
        up_completed = False
        held_target_name = current_target_name
    elif command == "up":
        up_completed = True
        held_target_name = current_target_name
    elif command in {"putdown", "home"}:
        clear_object_state()

    hold_current_pose()
    clear_active_action()

    reply = make_reply(
        request_id,
        "SUCCESS",
        execution_state="HOME_COMPLETED" if command == "home" else "COMPLETED",
        command=command,
        action_label=action_label,
        instruction=instruction,
        destination=destination,
        target=completed_target,
        recorded=True,
        image_path=image_path,
        motion_executed=True,
        action_completed=True,
        continued=continued,
        switched=switched,
        slice_seconds=None if mode == "FULL" else ACTION_SLICE_SECONDS,
        remaining_action=None,
        motion_phase=current_motion_phase,
        pick_completed=pick_completed,
        up_completed=up_completed,
    )

    cache_completed_reply(request_id, reply)
    clear_execution_state()
    send_reply_and_close(reply)


def fail_execution(exc: BaseException) -> None:
    request_id = execution_request_id
    command = active_command
    action_label = active_action_label
    instruction = execution_task_instruction
    destination = active_destination_name
    image_path = execution_image_path

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
        destination=destination,
        message=f"機械手臂動作失敗：{type(exc).__name__}: {exc}",
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
    global client_conn, client_addr, request_buffer
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
                send_reply_and_close(make_reply(None, "ERROR", message="Request 過大。"))
                return
            if b"\n" in request_buffer:
                break
    except OSError as exc:
        log_exception("[SOCKET] Receive failed", exc)
        close_client("receive error")
        return

    newline_index = request_buffer.find(b"\n")
    if newline_index >= 0:
        raw_line = bytes(request_buffer[:newline_index]).strip()
        trailing = bytes(request_buffer[newline_index + 1:]).strip()
        if trailing:
            send_reply_and_close(
                make_reply(None, "ERROR", message="每個連線只能傳送一筆 JSON 指令。")
            )
            return
    elif peer_closed:
        raw_line = bytes(request_buffer).strip()
    else:
        return

    if not raw_line:
        send_reply_and_close(make_reply(None, "ERROR", message="收到空白 Request。"))
        return

    try:
        request = json.loads(raw_line.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        send_reply_and_close(make_reply(None, "ERROR", message=f"Request 不是有效 JSON：{exc}"))
        return

    if not isinstance(request, dict):
        send_reply_and_close(make_reply(None, "ERROR", message="Request 必須是 JSON object。"))
        return

    request_buffer = bytearray()
    start_request(request)


# =========================
# Main execution loop
# =========================

def tick_execution() -> None:
    global execution_steps_remaining, slice_time_expired

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
                    f"[EXECUTION] Reached {ACTION_SLICE_SECONDS:.1f} simulated "
                    f"seconds during phase={phase}."
                )

        if not slice_time_expired:
            return

        # 不允許停在半閉或半開狀態。
        if phase in {"GRIPPER_CLOSING", "GRIPPER_OPENING"}:
            return

        # pick 關閉完成即視為完成，不需再多送一次 pick。
        if phase == "GRIPPER_CLOSED_BOUNDARY" and active_command == "pick":
            finish_completed()
            return

        # putdown 開啟完成即視為完成。
        if phase == "GRIPPER_OPENED_BOUNDARY" and active_command == "putdown":
            finish_completed()
            return

        finish_paused()

    except StopIteration:
        finish_completed()
    except Exception as exc:
        fail_execution(exc)


try:
    log("=" * 80)
    log("Isaac Sim Manual Record Server — Three-Step Flow / 1 Second")
    log("Action flow         : pick <object> -> up -> putdown <destination>")
    log(f"Action slice        : {ACTION_SLICE_SECONDS:.1f} simulated seconds")
    log(f"Gripper open frames : {GRIPPER_OPEN_FRAMES} (faster)")
    log(f"Gripper close frames: {GRIPPER_CLOSE_FRAMES} (unchanged)")
    log(f"Destinations        : {sorted(DESTINATION_CONFIGS.keys())}")
    log(f"Image directory     : {SAVE_DIR}")
    log(f"JSONL file          : {JSONL_PATH}")
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
