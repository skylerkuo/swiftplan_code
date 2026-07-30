#!/usr/bin/env python3
"""
manual_command_client_three_step_1s.py

固定任務、三步驟動作流程：
    1. pick <物品名稱>
       自動快速打開夾爪，移動到物品位置，再以原本速度關閉夾爪。
    2. up
       將上一個 pick 的物品垂直抬升到原物品位置上方。
    3. putdown tray / putdown box
       移動到目的地，到達後自動快速打開夾爪。

每筆動作由 Server 先拍照、寫入 JSONL，再執行或續跑 1 秒。
直接按 Enter 可重複上一個標籤並建立新 request_id。
"""

from __future__ import annotations

import json
import socket
import sys
import uuid
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple


# =========================
# Config：收集不同任務時，只修改這裡
# =========================

HOST = "127.0.0.1"
PORT = 6547

CONNECT_TIMEOUT_SECONDS = 10.0
REPLY_TIMEOUT_SECONDS = 300.0
MAX_REPLY_BYTES = 1024 * 1024

TASK_INSTRUCTION = "put the orange into the box"
TASK_DESTINATION = "box"

SUPPORTED_DESTINATIONS: Tuple[str, ...] = ("tray", "box")
ACTION_SLICE_SECONDS = 1.0


def validate_config() -> None:
    instruction = str(TASK_INSTRUCTION).strip()
    destination = str(TASK_DESTINATION).strip().lower()

    if not instruction:
        raise ValueError("TASK_INSTRUCTION 不可為空。")
    if len(instruction) > 1000:
        raise ValueError("TASK_INSTRUCTION 最多 1000 個字元。")
    if destination not in SUPPORTED_DESTINATIONS:
        allowed = "、".join(SUPPORTED_DESTINATIONS)
        raise ValueError(
            f"TASK_DESTINATION={destination!r} 不支援；目前只支援：{allowed}。"
        )


validate_config()
FIXED_TASK_INSTRUCTION = str(TASK_INSTRUCTION).strip()
FIXED_TASK_DESTINATION = str(TASK_DESTINATION).strip().lower()
FINISHED_LABEL = f"{FIXED_TASK_INSTRUCTION} finished"


@dataclass(frozen=True)
class RequestPayload:
    request_id: str
    action_label: str
    instruction: str
    default_destination: str

    def to_dict(self) -> Dict[str, str]:
        return {
            "request_id": self.request_id,
            "action_label": self.action_label,
            "instruction": self.instruction,
            "default_destination": self.default_destination,
        }


def normalize_destination(raw_text: str) -> str:
    destination = str(raw_text).strip().lower()
    if destination not in SUPPORTED_DESTINATIONS:
        allowed = "、".join(SUPPORTED_DESTINATIONS)
        raise ValueError(
            f"不支援的目的地 {destination!r}；目前只支援：{allowed}。"
        )
    return destination


def normalize_manual_label(raw_text: str) -> str:
    text = str(raw_text).strip()
    lower = text.lower()

    if not text:
        raise ValueError("動作指令不可為空。")

    if lower == "finished" or lower == FINISHED_LABEL.lower():
        return "finished"

    if lower == "up":
        return "up"

    if lower.startswith("pickup"):
        raise ValueError(
            "新流程不使用 pickup。請改用 pick <物品名稱>，下一步再輸入 up。"
        )

    if lower.startswith("pick"):
        parts = text.split(maxsplit=1)
        if lower == "pick" or len(parts) != 2 or not parts[1].strip():
            raise ValueError("pick 後面必須提供物品名稱，例如：pick orange")
        return f"pick {parts[1].strip()}"

    if lower == "putdown":
        return f"putdown {FIXED_TASK_DESTINATION}"

    if lower.startswith("putdown"):
        parts = text.split(maxsplit=1)
        if len(parts) != 2 or not parts[1].strip():
            raise ValueError("putdown 後面必須提供目的地，例如：putdown box")
        destination = normalize_destination(parts[1])
        return f"putdown {destination}"

    if lower == "clear":
        return "clear"
    if lower == "home":
        return "home"

    raise ValueError(
        "不支援的動作。請輸入 pick <物品名稱>、up、putdown、"
        "putdown tray、putdown box、finished、clear 或 home。"
    )


def receive_one_json_line(sock: socket.socket) -> Dict[str, Any]:
    buffer = bytearray()
    raw_line: Optional[bytes] = None

    while True:
        chunk = sock.recv(4096)
        if not chunk:
            break
        buffer.extend(chunk)

        if len(buffer) > MAX_REPLY_BYTES:
            raise RuntimeError("Server 回覆超過允許大小。")

        newline_index = buffer.find(b"\n")
        if newline_index >= 0:
            raw_line = bytes(buffer[:newline_index]).strip()
            break

    if raw_line is None:
        raw_line = bytes(buffer).strip()
    if not raw_line:
        raise ConnectionError("Server 關閉連線，但沒有回傳 JSON。")

    try:
        reply = json.loads(raw_line.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Server 回覆不是有效 JSON：{exc}") from exc

    if not isinstance(reply, dict):
        raise ValueError("Server 回覆必須是 JSON object。")
    return reply


def send_request_once(payload: RequestPayload) -> Dict[str, Any]:
    request_line = json.dumps(payload.to_dict(), ensure_ascii=False) + "\n"

    with socket.create_connection(
        (HOST, PORT), timeout=CONNECT_TIMEOUT_SECONDS
    ) as sock:
        sock.settimeout(REPLY_TIMEOUT_SECONDS)
        sock.sendall(request_line.encode("utf-8"))
        sock.shutdown(socket.SHUT_WR)
        return receive_one_json_line(sock)


def print_fixed_task() -> None:
    print("\n[FIXED TASK]")
    print(f"  instruction         : {FIXED_TASK_INSTRUCTION}")
    print(f"  default destination : {FIXED_TASK_DESTINATION}")
    print(f"  action slice        : {ACTION_SLICE_SECONDS:.1f} simulated second")


def print_help() -> None:
    print(
        "\n三步驟動作流程：\n"
        "  pick <物品名稱>                 快速開夾爪、移動到物品、關閉夾爪\n"
        "  up                              將目前物品垂直抬升到原位置上方\n"
        f"  putdown                          預設移動到 {FIXED_TASK_DESTINATION}，到達後快速開夾爪\n"
        "  putdown tray                    移動到 tray，到達後快速開夾爪\n"
        "  putdown box                     移動到 box，到達後快速開夾爪\n"
        "  finished                        記錄完整任務完成標籤\n"
        "  clear                           重設場景，不拍照、不寫入 JSONL\n"
        "  home                            記錄目前畫面，直接回 Home 並開夾爪\n"
        "  retry                           重送上一筆相同 request_id\n"
        "  show                            顯示固定任務設定\n"
        "  help                            顯示說明\n"
        "  quit                            結束程式\n"
        "\n範例：pick orange -> up -> putdown box -> finished\n"
        "\n直接按 Enter：重複上一個 action label，從暫停位置續跑下一個 1 秒。\n"
    )


def print_reply(reply: Dict[str, Any]) -> None:
    print("\n[SERVER REPLY]")
    print(json.dumps(reply, ensure_ascii=False, indent=2))

    status = reply.get("status", "UNKNOWN")
    if status == "ERROR":
        print(f"❌ Server 執行失敗：{reply.get('message', 'unknown error')}")
        return
    if status != "SUCCESS":
        print(f"⚠️ 未預期的 Server 狀態：{status}")
        return

    state = reply.get("execution_state", "UNKNOWN")
    if reply.get("recorded"):
        print(f"✅ 已記錄一筆資料：{reply.get('image_path')}")

    if state == "PAUSED":
        print("⏸️ 動作已推進 1 秒，目前暫停。")
    elif state == "COMPLETED":
        print("🏁 此動作已完成，可輸入下一步。")
    elif state == "HOME_COMPLETED":
        print("🏠 Home 動作已完成。")
    elif state == "FINISHED":
        print("🏁 已記錄整體任務完成標籤。")
    elif state == "CLEAR_COMPLETED":
        print("🧹 場景已恢復為 USD 初始狀態。")
    elif state == "DUPLICATE":
        print("ℹ️ 相同 request_id 已處理過。")


def main() -> int:
    print("=" * 80)
    print("Isaac Sim Manual Annotation Client — Three-Step Flow / 1 Second")
    print(f"Server                  : {HOST}:{PORT}")
    print(f"Task instruction        : {FIXED_TASK_INSTRUCTION}")
    print(f"Default destination     : {FIXED_TASK_DESTINATION}")
    print("Action flow             : pick <object> -> up -> putdown <destination>")
    print("=" * 80)
    print_help()

    last_payload: Optional[RequestPayload] = None
    last_action_label: Optional[str] = None

    while True:
        try:
            raw = input("action> ")
        except (EOFError, KeyboardInterrupt):
            print("\n結束程式。")
            return 0

        stripped = raw.strip()
        lower = stripped.lower()

        if lower in {"quit", "exit", "q"}:
            print("結束程式。")
            return 0
        if lower in {"help", "h", "?"}:
            print_help()
            continue
        if lower == "show":
            print_fixed_task()
            continue

        if lower == "retry":
            if last_payload is None:
                print("⚠️ 目前沒有可 retry 的上一筆要求。")
                continue
            payload = last_payload
            print(f"♻️ 重送 request_id={payload.request_id}")
        else:
            if not stripped:
                if last_action_label is None:
                    print("⚠️ 尚未輸入過任何動作，無法重複。")
                    continue
                action_label = last_action_label
                print(f"↻ 重複上一個標註：{action_label}")
            else:
                try:
                    action_label = normalize_manual_label(stripped)
                except ValueError as exc:
                    print(f"⚠️ {exc}")
                    continue

            payload = RequestPayload(
                request_id=uuid.uuid4().hex,
                action_label=action_label,
                instruction=FIXED_TASK_INSTRUCTION,
                default_destination=FIXED_TASK_DESTINATION,
            )
            last_payload = payload
            if action_label != "clear":
                last_action_label = action_label

        try:
            reply = send_request_once(payload)
        except ConnectionRefusedError:
            print(f"❌ 無法連線到 {HOST}:{PORT}。請先啟動 Server。")
            continue
        except socket.timeout:
            print("❌ 等待 Server 回覆逾時。確認後可輸入 retry。")
            continue
        except (OSError, ConnectionError, RuntimeError, ValueError) as exc:
            print(f"❌ 通訊失敗：{type(exc).__name__}: {exc}")
            continue

        if reply.get("request_id") != payload.request_id:
            print("⚠️ Server 回覆的 request_id 與本次要求不同。")

        print_reply(reply)


if __name__ == "__main__":
    sys.exit(main())
