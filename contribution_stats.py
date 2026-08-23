from __future__ import annotations

import json
import os
import re
import time
from datetime import datetime
from typing import Any


_SUCCESS_RE = re.compile(
    r"处理完毕[！!]?\s*已将\s*(\d+)\s*张加密图片转发至\s*(\d+)\s*个安全群聊"
)
HISTORY_SCAN_VERSION = 2


def _message_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "".join(_message_text(item) for item in value)
    if not isinstance(value, dict):
        return ""

    segment_type = str(value.get("type", "")).lower()
    data = value.get("data")
    if segment_type == "text" and isinstance(data, dict):
        return str(data.get("text", ""))

    for key in ("message", "content", "raw_message"):
        if key in value:
            text = _message_text(value[key])
            if text:
                return text
    return ""


def _sender_id(message: dict) -> str:
    sender = message.get("sender")
    if isinstance(sender, dict) and sender.get("user_id") is not None:
        return str(sender["user_id"])
    if message.get("user_id") is not None:
        return str(message["user_id"])
    return ""


def _message_timestamp(message: dict) -> int:
    for key in ("time", "timestamp"):
        try:
            value = int(message.get(key, 0))
            if value > 0:
                return value
        except (TypeError, ValueError):
            pass
    return 0


def _message_identity(message: dict, fallback_index: int) -> str:
    for key in ("message_id", "message_seq", "real_id", "id"):
        value = message.get(key)
        if value not in (None, ""):
            return str(value)
    return f"fallback:{_message_timestamp(message)}:{fallback_index}"


def extract_history_contributions(
    messages: list,
    bot_id: str,
    before_timestamp: int | None = None,
) -> list[dict]:
    """Extract successful /加密2 receipts sent by the bot."""
    records = []
    seen_ids = set()
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            continue
        if bot_id and _sender_id(message) != str(bot_id):
            continue

        timestamp = _message_timestamp(message)
        if before_timestamp and timestamp and timestamp >= before_timestamp:
            continue

        match = _SUCCESS_RE.search(_message_text(message))
        if not match:
            continue
        image_count = int(match.group(1))
        group_count = int(match.group(2))
        if image_count <= 0 or group_count <= 0:
            continue

        message_id = _message_identity(message, index)
        if message_id in seen_ids:
            continue
        seen_ids.add(message_id)
        records.append(
            {
                "message_id": message_id,
                "count": image_count,
                "timestamp": timestamp,
            }
        )
    return records


class ContributionStore:
    def __init__(self, path: str):
        self.path = path
        self.data = self._load()

    def _load(self) -> dict:
        default = {"version": 2, "users": {}}
        if not os.path.exists(self.path):
            return default
        try:
            with open(self.path, "r", encoding="utf-8") as file:
                data = json.load(file)
            if not isinstance(data, dict) or not isinstance(data.get("users"), dict):
                return default
            data.setdefault("version", 1)
            return data
        except (OSError, ValueError, TypeError):
            return default

    def save(self) -> None:
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        temp_path = f"{self.path}.tmp"
        with open(temp_path, "w", encoding="utf-8") as file:
            json.dump(self.data, file, ensure_ascii=False, indent=2)
        os.replace(temp_path, self.path)

    def _user(self, user_id: str, name: str = "") -> dict:
        user_id = str(user_id)
        user = self.data["users"].setdefault(
            user_id,
            {
                "name": name or user_id,
                "total": 0,
                "months": {},
                "history_imported": False,
                "history_cutoff": None,
                "history_scan_version": 0,
            },
        )
        user.setdefault("name", name or user_id)
        user.setdefault("total", 0)
        user.setdefault("months", {})
        user.setdefault("history_imported", False)
        user.setdefault("history_cutoff", None)
        user.setdefault("history_scan_version", 0)
        if name:
            user["name"] = name
        return user

    def record_live(
        self,
        user_id: str,
        name: str,
        image_count: int,
        timestamp: int | None = None,
    ) -> None:
        if image_count <= 0:
            return
        timestamp = int(timestamp or time.time())
        user = self._user(user_id, name)
        if not user["history_imported"] and not user.get("history_cutoff"):
            user["history_cutoff"] = timestamp
        self._add_count(user, image_count, timestamp)
        self.save()

    def history_state(self, user_id: str, name: str = "") -> tuple[bool, int | None]:
        user = self._user(user_id, name)
        cutoff = user.get("history_cutoff")
        scan_version = int(user.get("history_scan_version", 0))
        imported = bool(user.get("history_imported")) and (
            scan_version >= HISTORY_SCAN_VERSION
        )
        return imported, int(cutoff) if cutoff else None

    def import_history(
        self,
        user_id: str,
        name: str,
        records: list[dict],
    ) -> None:
        user = self._user(user_id, name)
        if (
            user.get("history_imported")
            and int(user.get("history_scan_version", 0)) >= HISTORY_SCAN_VERSION
        ):
            return

        seen_ids = set()
        history_total = 0
        history_months = {}
        for record in records:
            message_id = str(record.get("message_id", ""))
            if not message_id or message_id in seen_ids:
                continue
            seen_ids.add(message_id)
            timestamp = int(record.get("timestamp", 0))
            image_count = int(record.get("count", 0))
            if image_count <= 0:
                continue
            if timestamp <= 0:
                timestamp = int(time.time())
            month_key = datetime.fromtimestamp(timestamp).strftime("%Y-%m")
            history_total += image_count
            history_months[month_key] = (
                int(history_months.get(month_key, 0)) + image_count
            )

        # Version 1 may already contain a partial history import. Keep whichever
        # value is larger so a corrective rescan never lowers existing totals.
        user["total"] = max(int(user.get("total", 0)), history_total)
        months = user.setdefault("months", {})
        for month_key, image_count in history_months.items():
            months[month_key] = max(int(months.get(month_key, 0)), image_count)
        user["history_imported"] = True
        user["history_scan_version"] = HISTORY_SCAN_VERSION
        self.data["version"] = 2
        self.save()

    def mark_history_imported(self, user_id: str, name: str) -> None:
        user = self._user(user_id, name)
        user["history_imported"] = True
        user["history_scan_version"] = HISTORY_SCAN_VERSION
        self.data["version"] = 2
        self.save()

    @staticmethod
    def _add_count(user: dict, image_count: int, timestamp: int) -> None:
        if image_count <= 0:
            return
        if timestamp <= 0:
            timestamp = int(time.time())
        month_key = datetime.fromtimestamp(timestamp).strftime("%Y-%m")
        user["total"] = int(user.get("total", 0)) + image_count
        months = user.setdefault("months", {})
        months[month_key] = int(months.get(month_key, 0)) + image_count

    def user_ids(self) -> set[str]:
        return set(self.data.get("users", {}).keys())

    def ranking(self, month_key: str | None = None, limit: int = 10) -> list[dict]:
        ranking = []
        for user_id, user in self.data.get("users", {}).items():
            count = (
                int(user.get("months", {}).get(month_key, 0))
                if month_key
                else int(user.get("total", 0))
            )
            if count <= 0:
                continue
            ranking.append(
                {
                    "uid": str(user_id),
                    "name": str(user.get("name") or user_id),
                    "count": count,
                }
            )
        ranking.sort(key=lambda item: (-item["count"], item["uid"]))
        return ranking[:limit]
