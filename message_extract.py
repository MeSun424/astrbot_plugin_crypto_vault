import html
import json
import re
from dataclasses import dataclass, field
from typing import Any


_CQ_IMAGE_RE = re.compile(r"\[CQ:image,([^\]]+)\]", re.IGNORECASE)
_CQ_FORWARD_RE = re.compile(r"\[CQ:forward,([^\]]+)\]", re.IGNORECASE)
_CQ_REPLY_RE = re.compile(r"\[CQ:reply,([^\]]+)\]", re.IGNORECASE)
_CQ_PARAM_RE = re.compile(r"(?:^|,)([a-zA-Z_][\w]*)=([^,\]]*)")
_FORWARD_ID_RE = re.compile(
    r"""(?:resid|m_resid|forward_id|forwardId)\s*["']?\s*[:=]\s*["']?([a-zA-Z0-9_+\-/=]+)""",
    re.IGNORECASE,
)
_EXPIRED_IMAGE_RE = re.compile(r"(?:\[图片\]\s*)?已过期|图片已过期")
_FORWARD_TYPES = {"forward", "nodes", "node"}


@dataclass
class PayloadScan:
    image_refs: list[Any] = field(default_factory=list)
    forward_ids: list[str] = field(default_factory=list)
    reply_ids: list[str] = field(default_factory=list)
    forward_segments: int = 0
    expired_count: int = 0


def _append_unique(items: list, value: Any, seen: set) -> None:
    if value is None:
        return
    marker = repr(value)
    if marker not in seen:
        seen.add(marker)
        items.append(value)


def _parse_cq_params(raw_params: str) -> dict[str, str]:
    return {
        match.group(1).lower(): html.unescape(match.group(2))
        for match in _CQ_PARAM_RE.finditer(raw_params)
    }


def _looks_like_forward_id(value: Any) -> bool:
    if value is None:
        return False
    text = str(value).strip()
    return bool(text and text != "0" and len(text) <= 512)


def scan_message_payload(payload: Any) -> PayloadScan:
    """Extract image references and nested forward IDs from OneBot/NapCat payloads."""
    result = PayloadScan()
    seen_images: set[str] = set()
    seen_forward_ids: set[str] = set()
    seen_reply_ids: set[str] = set()
    seen_text: set[str] = set()
    visited_containers: set[int] = set()

    def add_forward_id(value: Any) -> None:
        if _looks_like_forward_id(value):
            _append_unique(result.forward_ids, str(value).strip(), seen_forward_ids)

    def add_reply_id(value: Any) -> None:
        if _looks_like_forward_id(value):
            _append_unique(result.reply_ids, str(value).strip(), seen_reply_ids)

    def scan_text(text: str) -> None:
        nonlocal result
        decoded = html.unescape(text)
        if decoded in seen_text:
            return
        seen_text.add(decoded)

        result.expired_count += len(_EXPIRED_IMAGE_RE.findall(decoded))

        for match in _CQ_IMAGE_RE.finditer(decoded):
            params = _parse_cq_params(match.group(1))
            if params:
                _append_unique(result.image_refs, params, seen_images)

        for match in _CQ_FORWARD_RE.finditer(decoded):
            params = _parse_cq_params(match.group(1))
            result.forward_segments += 1
            add_forward_id(params.get("id") or params.get("resid"))

        for match in _CQ_REPLY_RE.finditer(decoded):
            params = _parse_cq_params(match.group(1))
            add_reply_id(params.get("id") or params.get("message_id"))

        for match in _FORWARD_ID_RE.finditer(decoded):
            add_forward_id(match.group(1))

        stripped = decoded.strip()
        if stripped.startswith(("{", "[")):
            try:
                parsed = json.loads(stripped)
            except (TypeError, ValueError):
                return
            visit(parsed)

    def visit(value: Any) -> None:
        if value is None:
            return
        if isinstance(value, str):
            scan_text(value)
            return
        if isinstance(value, (bytes, bytearray, int, float, bool)):
            return
        if isinstance(value, (dict, list, tuple)):
            object_id = id(value)
            if object_id in visited_containers:
                return
            visited_containers.add(object_id)

        if isinstance(value, (list, tuple)):
            for item in value:
                visit(item)
            return
        if not isinstance(value, dict):
            return

        segment_type = str(value.get("type", "")).lower()
        segment_data = value.get("data")
        if not isinstance(segment_data, dict):
            segment_data = value

        if segment_type == "image":
            _append_unique(result.image_refs, segment_data, seen_images)
        elif segment_type in _FORWARD_TYPES:
            result.forward_segments += 1
            for key in ("id", "resid", "message_id", "forward_id"):
                add_forward_id(segment_data.get(key))
        elif segment_type == "reply":
            add_reply_id(segment_data.get("id") or segment_data.get("message_id"))

        for key, item in value.items():
            lower_key = str(key).lower()
            if lower_key in {"resid", "m_resid", "forward_id"}:
                add_forward_id(item)
            visit(item)

    visit(payload)
    return result


def image_reference_locations(reference: Any) -> tuple[list[str], list[str]]:
    """Return candidate HTTP URLs and OneBot file identifiers for an image."""
    urls: list[str] = []
    files: list[str] = []
    seen_urls: set[str] = set()
    seen_files: set[str] = set()

    def add_url(value: Any) -> None:
        if isinstance(value, str):
            value = html.unescape(value).strip()
            if value.startswith(("http://", "https://")) and value not in seen_urls:
                seen_urls.add(value)
                urls.append(value)

    def add_file(value: Any) -> None:
        if isinstance(value, str):
            value = html.unescape(value).strip()
            if not value:
                return
            if value.startswith(("http://", "https://")):
                add_url(value)
            elif value not in seen_files:
                seen_files.add(value)
                files.append(value)

    if isinstance(reference, str):
        add_url(reference)
        add_file(reference)
        return urls, files
    if not isinstance(reference, dict):
        return urls, files

    for key in ("url", "src"):
        add_url(reference.get(key))

    image_url = reference.get("image_url")
    if isinstance(image_url, dict):
        add_url(image_url.get("url"))
    else:
        add_url(image_url)

    for key in ("file", "file_id", "filename", "path"):
        add_file(reference.get(key))

    return urls, files
