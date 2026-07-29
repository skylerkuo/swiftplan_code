#!/usr/bin/env python3
"""
manual_command_client_dynamic.py

Isaac Sim 外部人工標註控制端（動態任務指令版本）。

設計目標：
1. 任務 instruction 不再綁定單一固定句型，可在執行期間切換，例如：
       put the apple into the tray
       put the cube into the box
       tidy up the properties into the tray
       prepare the fruits
2. 目前控制技能仍保留：
       pickup <物品名稱>
       putdown tray
       putdown box
       finished
       clear
       home
3. putdown 可省略目的地，使用目前設定的預設目的地。
4. 每一筆新動作建立新的 request_id；retry 會沿用上一筆 request_id。
5. 直接按 Enter 會重複上一個 action label，但建立新的 request_id。

建議操作流程：
    task put the apple into the box
    dest box
    pickup apple
    <直接按 Enter，續跑 pickup>
    putdown box
    <直接按 Enter，續跑 putdown box>
    finished
    clear
"""

from __future__ import annotations

import json
import re
import socket
import sys
import uuid
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple


# =========================
# Config
# =========================

HOST = "127.0.0.1"
PORT = 6547

CONNECT_TIMEOUT_SECONDS = 10.0
REPLY_TIMEOUT_SECONDS = 300.0
MAX_REPLY_BYTES = 1024 * 1024

SUPPORTED_DESTINATIONS: Tuple[str, ...] = ("tray", "box")

DEFAULT_TASK_INSTRUCTION = "tidy up the properties into the tray"
DEFAULT_TASK_DESTINATION = "tray"

if DEFAULT_TASK_DESTINATION not in SUPPORTED_DESTINATIONS:
    raise ValueError(
        "DEFAULT_TASK_DESTINATION 必須存在於 SUPPORTED_DESTINATIONS。"
    )


# =========================
# Data models
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


@dataclass
class TaskContext:
    instruction: str = DEFAULT_TASK_INSTRUCTION
    default_destination: str = DEFAULT_TASK_DESTINATION

    def validate(self) -> None:
        self.instruction = normalize_instruction(self.instruction)
        self.default_destination = normalize_destination(
            self.default_destination
        )


# =========================
# Task and action normalization
# =========================

def normalize_instruction(raw_text: str) -> str:
    """驗證自然語言任務指令；不限制固定句型。"""
    text = str(raw_text).strip()

    if not text:
        raise ValueError("任務 instruction 不可為空。")

    if len(text) > 1000:
        raise ValueError("任務 instruction 過長，最多 1000 個字元。")

    return text


def normalize_destination(raw_text: str) -> str:
    destination = str(raw_text).strip().lower()

    if destination not in SUPPORTED_DESTINATIONS:
        allowed = "、".join(SUPPORTED_DESTINATIONS)
        raise ValueError(
            f"不支援的目的地 {destination!r}；目前只支援：{allowed}。"
        )

    return destination


def infer_destination_from_instruction(
    instruction: str,
) -> Optional[str]:
    """
    若 instruction 中只出現一個已支援目的地，就回傳該目的地。

    這只是 Client 端便利功能；Server 仍會接收明確的
    default_destination，因此不依賴自然語言解析來控制機器人。
    """
    lower = instruction.lower()
    matches = []

    for destination in SUPPORTED_DESTINATIONS:
        if re.search(rf"\b{re.escape(destination)}\b", lower):
            matches.append(destination)

    if len(matches) == 1:
        return matches[0]

    return None


def normalize_manual_label(
    raw_text: str,
    current_destination: str,
) -> str:
    """
    驗證並正規化人工輸入。

    回傳值會傳給 Server；Server 會再次驗證並將 canonical action label
    寫入 JSONL。
    """
    text = str(raw_text).strip()
    lower = text.lower()

    if not text:
        raise ValueError("動作指令不可為空。")

    if lower == "finished":
        return "finished"

    if lower == "putdown":
        destination = normalize_destination(current_destination)
        return f"putdown {destination}"

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
    """建立一次連線、傳送一筆 JSON、接收一筆 JSON，然後關閉。"""
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

def print_task_context(context: TaskContext) -> None:
    print("\n[CURRENT TASK]")
    print(f"  instruction         : {context.instruction}")
    print(f"  default destination : {context.default_destination}")


def print_examples() -> None:
    print(
        "\n操作範例：\n"
        "\n"
        "範例 1：單一物件放入 tray\n"
        "  task put the apple into the tray\n"
        "  dest tray\n"
        "  pickup apple\n"
        "  <Enter，直到 pickup 完成>\n"
        "  putdown tray\n"
        "  <Enter，直到 putdown 完成>\n"
        "  finished\n"
        "\n"
        "範例 2：單一物件放入 box\n"
        "  task put the cube into the box\n"
        "  dest box\n"
        "  pickup cube\n"
        "  putdown box\n"
        "  finished\n"
        "\n"
        "範例 3：多物件整理\n"
        "  task tidy up the properties into the box\n"
        "  dest box\n"
        "  pickup bottle\n"
        "  putdown box\n"
        "  pickup cube\n"
        "  putdown box\n"
        "  finished\n"
    )


def print_help(context: TaskContext) -> None:
    print(
        "\n任務設定指令：\n"
        "  task <自然語言任務>              設定目前任務 instruction\n"
        "  dest <tray|box>                  設定 putdown 的預設目的地\n"
        "  show                             顯示目前任務設定\n"
        "  examples                         顯示操作範例\n"
        "\n"
        "機器人與標註指令：\n"
        "  pickup <物品名稱>                執行或續跑 pickup 2 秒\n"
        f"  putdown                           放到目前預設目的地 "
        f"({context.default_destination})\n"
        "  putdown tray                     明確放到 tray\n"
        "  putdown box                      明確放到 box\n"
        "  finished                          記錄完整任務完成標註\n"
        "  clear                             重設場景，不寫入 JSONL\n"
        "  home                              記錄資料並直接回 Home\n"
        "  retry                             重送上一筆相同 request_id\n"
        "  help                              顯示說明\n"
        "  quit                              結束程式\n"
        "\n"
        "快捷操作：\n"
        "  直接按 Enter                     重複上一個 action label，"
        "但建立新的 request_id\n"
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
        print(f"✅ 已記錄資料：{image_path}")

    if execution_state == "PAUSED":
        if reply.get("continued"):
            print("⏸️ 相同動作已續跑，目前暫停。")
        elif reply.get("switched"):
            print("⏸️ 已切換新動作並執行，目前暫停。")
        else:
            print("⏸️ 動作已執行一個時間片，目前暫停。")

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


def handle_task_command(
    raw_command: str,
    context: TaskContext,
) -> bool:
    """
    處理不會送往 Server 的本地任務設定指令。

    回傳 True 表示已處理；False 表示這是機器人動作指令。
    """
    stripped = raw_command.strip()
    lower = stripped.lower()

    if lower == "show":
        print_task_context(context)
        return True

    if lower == "examples":
        print_examples()
        return True

    if lower == "task":
        print_task_context(context)
        print("使用方式：task <自然語言任務>")
        return True

    if lower.startswith("task "):
        instruction = normalize_instruction(stripped[5:])
        context.instruction = instruction

        inferred = infer_destination_from_instruction(instruction)
        if inferred is not None:
            context.default_destination = inferred
            print(
                f"✅ 已設定 instruction，並從句子推定預設目的地為 "
                f"{inferred}。"
            )
        else:
            print("✅ 已設定 instruction。")

        print_task_context(context)
        return True

    if lower in {"dest", "destination"}:
        print_task_context(context)
        print("使用方式：dest tray 或 dest box")
        return True

    if lower.startswith("dest "):
        destination = normalize_destination(stripped[5:])
        context.default_destination = destination
        print(f"✅ 預設目的地已設為 {destination}。")
        return True

    if lower.startswith("destination "):
        destination = normalize_destination(stripped[12:])
        context.default_destination = destination
        print(f"✅ 預設目的地已設為 {destination}。")
        return True

    return False


def main() -> int:
    context = TaskContext()
    context.validate()

    print("=" * 76)
    print("Isaac Sim Manual Annotation Client — Dynamic Task Instructions")
    print(f"Server                  : {HOST}:{PORT}")
    print(f"Initial instruction     : {context.instruction}")
    print(f"Initial destination     : {context.default_destination}")
    print("Action slice            : 2 simulated seconds")
    print("Supported destinations  : tray, box")
    print("Instruction syntax      : arbitrary non-empty natural language")
    print("=" * 76)
    print_help(context)

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
            print_help(context)
            continue

        try:
            if stripped and handle_task_command(stripped, context):
                # 任務條件改變後，不直接沿用上一個 action label，避免誤標。
                if lower.startswith(("task ", "dest ", "destination ")):
                    last_action_label = None
                continue
        except ValueError as exc:
            print(f"⚠️ {exc}")
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
                    action_label = normalize_manual_label(
                        stripped,
                        context.default_destination,
                    )
                except ValueError as exc:
                    print(f"⚠️ {exc}")
                    continue

            try:
                context.validate()
            except ValueError as exc:
                print(f"⚠️ 任務設定無效：{exc}")
                continue

            payload = RequestPayload(
                request_id=uuid.uuid4().hex,
                action_label=action_label,
                instruction=context.instruction,
                default_destination=context.default_destination,
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
