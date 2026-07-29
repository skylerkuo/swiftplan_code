#!/usr/bin/env python3
"""
manual_command_client_fixed_task_1s.py

Isaac Sim 外部人工標註控制端：固定任務、每次推進 1 秒。

使用方式：
1. 先在下方 Config 區修改 TASK_INSTRUCTION 與 TASK_DESTINATION。
2. 啟動 Server。
3. 執行本 Client 後，直接輸入高階動作標註。
4. 每送出一筆 pickup / putdown：
   - Server 先拍攝目前畫面並寫入一筆 JSONL。
   - 再執行或續跑該高階動作 1 秒模擬時間。
5. 直接按 Enter 會重複上一個 action label，建立新的 request_id，
   因此可連續收集同一高階動作每隔 1 秒的狀態資料。

保留指令：
    pickup <物品名稱>
    putdown
    putdown tray
    putdown box
    finished
    clear
    home
    retry
    show
    help
    quit

注意：
- TASK_DESTINATION 只決定簡寫 `putdown` 的預設目的地。
- 明確輸入 `putdown tray` 或 `putdown box` 仍然都可使用。
- `clear` 不拍照、不寫入 JSONL。
- `home` 會先記錄一筆資料，再直接執行到完成。
- `finished` 只記錄完成標註，不執行機械手臂動作。
"""

from __future__ import annotations

import json
import socket
import sys
import uuid
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple


# =========================
# Config：每次收集不同任務時，只需修改這裡
# =========================

HOST = "127.0.0.1"
PORT = 6547

CONNECT_TIMEOUT_SECONDS = 10.0
REPLY_TIMEOUT_SECONDS = 300.0
MAX_REPLY_BYTES = 1024 * 1024

# 這一輪資料收集要使用的自然語言任務指令。
# 範例：
#   "put the apple into the tray"
#   "put the cube into the box"
#   "tidy up the properties into the tray"
#   "tidy up the fruits into the box"
TASK_INSTRUCTION = "put the apple into the box"

# 輸入簡寫 `putdown` 時使用的預設目的地。
# 目前支援 "tray" 與 "box"。
TASK_DESTINATION = "box"

SUPPORTED_DESTINATIONS: Tuple[str, ...] = ("tray", "box")
ACTION_SLICE_SECONDS = 1.0


# =========================
# Config validation
# =========================

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


# =========================
# Request payload
# =========================

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


# =========================
# Input normalization
# =========================

def normalize_destination(raw_text: str) -> str:
    destination = str(raw_text).strip().lower()

    if destination not in SUPPORTED_DESTINATIONS:
        allowed = "、".join(SUPPORTED_DESTINATIONS)
        raise ValueError(
            f"不支援的目的地 {destination!r}；目前只支援：{allowed}。"
        )

    return destination


def normalize_manual_label(raw_text: str) -> str:
    """
    驗證並正規化人工輸入。

    回傳值會送往 Server，並成為 JSONL 中的 annotation.action_label。
    """
    text = str(raw_text).strip()
    lower = text.lower()

    if not text:
        raise ValueError("動作指令不可為空。")

    if lower == "finished" or lower == FINISHED_LABEL.lower():
        return "finished"

    if lower == "putdown":
        return f"putdown {FIXED_TASK_DESTINATION}"

    if lower.startswith("putdown"):
        parts = text.split(maxsplit=1)

        if len(parts) != 2 or not parts[1].strip():
            raise ValueError(
                "putdown 後面必須提供目的地，例如：putdown box"
            )

        destination = normalize_destination(parts[1])
        return f"putdown {destination}"

    if lower == "clear":
        return "clear"

    if lower == "home":
        return "home"

    if lower.startswith("pickup"):
        parts = text.split(maxsplit=1)

        if len(parts) != 2 or not parts[1].strip():
            raise ValueError(
                "pickup 後面必須提供物品名稱，例如：pickup apple"
            )

        target_name = parts[1].strip()
        return f"pickup {target_name}"

    raise ValueError(
        "不支援的動作。請輸入 pickup <物品名稱>、putdown、"
        "putdown tray、putdown box、finished、clear 或 home。"
    )


# =========================
# TCP request
# =========================

def receive_one_json_line(sock: socket.socket) -> Dict[str, Any]:
    """接收 Server 回傳的一行 JSON。"""
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
        decoded = raw_line.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("Server 回覆不是有效 UTF-8。") from exc

    try:
        reply = json.loads(decoded)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Server 回覆不是有效 JSON：{decoded!r}"
        ) from exc

    if not isinstance(reply, dict):
        raise ValueError(
            "Server 回覆必須是 JSON object，"
            f"目前為：{type(reply).__name__}"
        )

    return reply


def send_request_once(payload: RequestPayload) -> Dict[str, Any]:
    """建立一次 TCP 連線，只傳送一筆 JSON 指令。"""
    request_line = json.dumps(
        payload.to_dict(),
        ensure_ascii=False,
    ) + "\n"

    with socket.create_connection(
        (HOST, PORT),
        timeout=CONNECT_TIMEOUT_SECONDS,
    ) as sock:
        sock.settimeout(REPLY_TIMEOUT_SECONDS)
        sock.sendall(request_line.encode("utf-8"))
        sock.shutdown(socket.SHUT_WR)

        return receive_one_json_line(sock)


# =========================
# Console UI
# =========================

def print_fixed_task() -> None:
    print("\n[FIXED TASK]")
    print(f"  instruction         : {FIXED_TASK_INSTRUCTION}")
    print(f"  default destination : {FIXED_TASK_DESTINATION}")
    print(f"  action slice        : {ACTION_SLICE_SECONDS:.1f} simulated second")


def print_help() -> None:
    print(
        "\n可用指令：\n"
        "  pickup <物品名稱>                記錄目前畫面，執行或續跑 pickup 1 秒\n"
        f"  putdown                           預設放到 {FIXED_TASK_DESTINATION}\n"
        "  putdown tray                     記錄目前畫面，放到 tray 1 秒\n"
        "  putdown box                      記錄目前畫面，放到 box 1 秒\n"
        "  finished                         記錄完整任務完成標註\n"
        "  clear                            重設場景，不拍照、不寫入 JSONL\n"
        "  home                             記錄目前畫面，直接執行到 Home 完成\n"
        "  retry                            重送上一筆相同 request_id\n"
        "  show                             顯示程式中固定的任務設定\n"
        "  help                             顯示說明\n"
        "  quit                             結束程式\n"
        "\n快捷操作：\n"
        "  直接按 Enter                    重複上一個 action label，"
        "建立新 request_id，收集下一個 1 秒狀態\n"
    )


def print_reply(reply: Dict[str, Any]) -> None:
    status = reply.get("status", "UNKNOWN")
    request_id = reply.get("request_id")

    print("\n[SERVER REPLY]")
    print(json.dumps(reply, ensure_ascii=False, indent=2))

    if status == "ERROR":
        print(
            "❌ Server 執行失敗："
            f"{reply.get('message', 'unknown error')}"
        )
        return

    if status != "SUCCESS":
        print(f"⚠️ 未預期的 Server 狀態：{status}")
        return

    execution_state = reply.get("execution_state", "UNKNOWN")
    image_path = reply.get("image_path")
    recorded = bool(reply.get("recorded", False))

    print(f"✅ request_id={request_id}")

    if recorded:
        print(f"✅ 已記錄一筆資料：{image_path}")

    if execution_state == "PAUSED":
        if reply.get("continued"):
            print("⏸️ 相同動作已續跑 1 秒，目前暫停。")
        elif reply.get("switched"):
            print("⏸️ 已切換新動作並執行 1 秒，目前暫停。")
        else:
            print("⏸️ 動作已執行 1 秒，目前暫停。")

    elif execution_state == "COMPLETED":
        print("🏁 此高階動作已完成。")

    elif execution_state == "HOME_COMPLETED":
        print("🏠 Home 動作已直接執行完成。")

    elif execution_state == "FINISHED":
        print("🏁 已記錄整體任務完成標註。")

    elif execution_state == "CLEAR_COMPLETED":
        print("🧹 場景已恢復為 USD 初始狀態。")

    elif execution_state == "DUPLICATE":
        print("ℹ️ 相同 request_id 已處理過，未重複記錄或執行。")

    else:
        print(f"ℹ️ execution_state={execution_state}")


def main() -> int:
    print("=" * 76)
    print("Isaac Sim Manual Annotation Client — Fixed Task / 1 Second")
    print(f"Server                  : {HOST}:{PORT}")
    print(f"Task instruction        : {FIXED_TASK_INSTRUCTION}")
    print(f"Default destination     : {FIXED_TASK_DESTINATION}")
    print(f"Action slice            : {ACTION_SLICE_SECONDS:.1f} simulated second")
    print("Supported destinations  : tray, box")
    print("Task changes             : edit TASK_INSTRUCTION in this file")
    print("Destination changes      : edit TASK_DESTINATION in this file")
    print("=" * 76)
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

        print(
            f"\n[SEND] request_id={payload.request_id}, "
            f"action_label={payload.action_label!r}, "
            f"instruction={payload.instruction!r}, "
            f"default_destination={payload.default_destination!r}"
        )

        try:
            reply = send_request_once(payload)

        except ConnectionRefusedError:
            print(
                f"❌ 無法連線到 {HOST}:{PORT}。"
                "請先啟動 Isaac Sim Server。"
            )
            continue

        except socket.timeout:
            print(
                "❌ 等待 Server 回覆逾時。請先查看 Isaac Sim 狀態；"
                "確認後輸入 retry，沿用相同 request_id，"
                "避免重複記錄或重複執行。"
            )
            continue

        except (OSError, ConnectionError, RuntimeError, ValueError) as exc:
            print(f"❌ 通訊失敗：{type(exc).__name__}: {exc}")
            print(
                "確認 Server 狀態後，可輸入 retry 重送相同 request_id。"
            )
            continue

        if reply.get("request_id") != payload.request_id:
            print(
                "⚠️ Server 回覆的 request_id 與本次要求不同："
                f"sent={payload.request_id}, "
                f"got={reply.get('request_id')}"
            )

        print_reply(reply)


if __name__ == "__main__":
    sys.exit(main())
