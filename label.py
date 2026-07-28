#!/usr/bin/env python3
"""
manual_command_client_2s.py

Isaac Sim 外部人工標註控制端。

控制規則：
1. 每一筆指令建立一次 TCP 連線。
2. 每一筆指令都會由 Server 拍照並寫入一筆 JSONL。
3. pickup / putdown 每次最多執行 2 秒：
   - 相同動作：從上次暫停位置繼續。
   - 不同動作：取消上次未完成動作，開始新動作。
4. home 不受 2 秒限制，會直接執行到完成。
5. finished 只記錄完成標註，不執行機械手臂動作。
6. retry 沿用相同 request_id，避免通訊失敗後重複記錄或重複執行。

可用指令：
    pickup apple
    putdown
    putdown tray
    putdown box
    finished
    home
    retry
    help
    quit

快捷操作：
    直接按 Enter：再次送出上一個 action label，但使用新的 request_id。
"""

from __future__ import annotations

import json
import socket
import sys
import uuid
from dataclasses import dataclass
from typing import Any, Dict, Optional


# =========================
# Config
# =========================

HOST = "127.0.0.1"
PORT = 6547

CONNECT_TIMEOUT_SECONDS = 10.0
REPLY_TIMEOUT_SECONDS = 300.0
MAX_REPLY_BYTES = 1024 * 1024

# 本次資料收集任務的目的地，只能設定為 "tray" 或 "box"。
TASK_DESTINATION = "tray"

if TASK_DESTINATION not in {"tray", "box"}:
    raise ValueError(
        'TASK_DESTINATION 只能是 "tray" 或 "box"。'
    )

# 不要在字串尾端加空白。
TASK_INSTRUCTION = (
    f"tidy up the properties into the {TASK_DESTINATION}"
)
FINISHED_LABEL = f"{TASK_INSTRUCTION} finished"


# =========================
# Request payload
# =========================

@dataclass(frozen=True)
class RequestPayload:
    request_id: str
    action_label: str
    instruction: str

    def to_dict(self) -> Dict[str, str]:
        return {
            "request_id": self.request_id,
            "action_label": self.action_label,
            "instruction": self.instruction,
        }


# =========================
# Input normalization
# =========================

def normalize_manual_label(raw_text: str) -> str:
    """
    驗證並正規化人工輸入。

    回傳值會直接寫入 JSONL 的 annotation.action_label。
    pickup 後面的物品名稱保留原始大小寫。
    """
    text = str(raw_text).strip()
    lower = text.lower()

    if not text:
        raise ValueError("指令不可為空。")

    if lower == "finished":
        return "finished"

    # 仍保留完整句子的相容性，但實際送給 Server 時統一簡化成 finished。
    if lower == FINISHED_LABEL.lower():
        return "finished"

    if lower == "putdown":
        return f"putdown {TASK_DESTINATION}"

    if lower in {"putdown tray", "putdown box"}:
        requested_destination = lower.split(maxsplit=1)[1]

        if requested_destination != TASK_DESTINATION:
            raise ValueError(
                f"目前任務目的地為 {TASK_DESTINATION}，"
                f"不可輸入 putdown {requested_destination}。"
            )

        return f"putdown {requested_destination}"

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
        "不支援的指令。請輸入 pickup <物品名稱>、putdown tray、"
        "putdown box、finished 或 home。"
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
    """
    建立一次連線、傳送一筆 JSON、等待一筆 JSON 回覆，然後關閉。
    """
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

        # 本次連線不再傳送其他資料。
        sock.shutdown(socket.SHUT_WR)

        return receive_one_json_line(sock)


# =========================
# Console UI
# =========================

def print_help() -> None:
    print(
        "\n可用指令：\n"
        "  pickup <物品名稱>                 記錄資料，執行或續跑 pickup 2 秒\n"
        f"  putdown                            預設放到 {TASK_DESTINATION}\n"
        f"  putdown {TASK_DESTINATION:<25} 記錄資料並放到目前目的地\n"
        "  finished                           結束任務並記錄完整完成標註\n"
        "  home                               記錄資料，直接執行到 Home 完成\n"
        "  retry                              重送相同 request_id\n"
        "  help                               顯示說明\n"
        "  quit                               結束程式\n"
        "\n快捷操作：\n"
        "  直接按 Enter                      重複上一個標註並建立新資料\n"
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
            print("⏸️ 相同動作已續跑 2 秒，目前暫停。")
        elif reply.get("switched"):
            print("⏸️ 已切換新動作並執行 2 秒，目前暫停。")
        else:
            print("⏸️ 動作已執行 2 秒，目前暫停。")

    elif execution_state == "COMPLETED":
        print("🏁 此高階動作已完成。")

    elif execution_state == "HOME_COMPLETED":
        print("🏠 Home 動作已直接執行完成。")

    elif execution_state == "FINISHED":
        print("🏁 已記錄整體任務完成標註。")

    elif execution_state == "DUPLICATE":
        print("ℹ️ 相同 request_id 已處理過，未重複記錄或執行。")

    else:
        print(f"ℹ️ execution_state={execution_state}")


def main() -> int:
    print("=" * 72)
    print("Isaac Sim Manual Annotation Client — 2 Second Action Slices")
    print(f"Server: {HOST}:{PORT}")
    print(f"Task destination: {TASK_DESTINATION}")
    print(f"Fixed instruction: {TASK_INSTRUCTION}")
    print(
        f"pickup / putdown {TASK_DESTINATION} 每次執行 2 秒。"
    )
    print("home 直接執行到完成。")
    print("每一筆新指令都記錄一筆資料。")
    print("=" * 72)
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
                instruction=TASK_INSTRUCTION,
            )
            last_payload = payload
            last_action_label = action_label

        print(
            f"\n[SEND] request_id={payload.request_id}, "
            f"action_label={payload.action_label!r}, "
            f"instruction={payload.instruction!r}"
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
