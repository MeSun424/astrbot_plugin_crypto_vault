import os
import io
import asyncio
import functools
import uuid
import aiohttp
import numpy as np
from datetime import datetime
from PIL import Image as PILImage

import astrbot.api.message_components as Comp
from astrbot.api import logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.star import Context, Star, register
from astrbot.api.star import StarTools

from .message_extract import image_reference_locations, scan_message_payload
from .contribution_stats import ContributionStore, extract_history_contributions
from .tomato_scramble import TomatoScramble
from . import forward_utils

_MAX_DOWNLOAD_SIZE = 20 * 1024 * 1024
_MAX_PIXEL_COUNT = 4096 * 4096
_MAX_FORWARD_DEPTH = 6
_MAX_FORWARD_RECORDS = 32
_HISTORY_PAGE_SIZE = 200
_MAX_HISTORY_PAGES = 500


@register("crypto_vault", "MeSun", "跨群合并转发版图片加密/解密插件", "1.0")
class CryptoVaultPlugin(Star):

    def __init__(self, context: Context, config=None):
        super().__init__(context, config)
        self.data_dir = str(StarTools.get_data_dir("crypto_vault"))
        os.makedirs(self.data_dir, exist_ok=True)
        self._session = None
        # 2 GB 内存环境下串行处理，避免多个大图同时构建曲线和像素数组。
        self._process_semaphore = asyncio.Semaphore(1)
        self._stats_lock = asyncio.Lock()
        self._history_backfill_announced = False
        self.contribution_store = ContributionStore(
            os.path.join(self.data_dir, "contribution_stats.json")
        )
        
        self.target_groups = []
        if config:
            vault_cfg = config.get("vault_settings", {})
            self.target_groups = vault_cfg.get("target_group_ids", [])

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=15))
        return self._session

    async def _download_image(self, url: str) -> bytes:
        session = await self._get_session()
        async with session.get(url) as resp:
            if resp.status != 200:
                raise Exception(f"HTTP {resp.status}")
            if resp.content_length and resp.content_length > _MAX_DOWNLOAD_SIZE:
                raise ValueError("图片文件超过 20 MB")

            content = bytearray()
            async for chunk in resp.content.iter_chunked(64 * 1024):
                if len(content) + len(chunk) > _MAX_DOWNLOAD_SIZE:
                    raise ValueError("图片文件超过 20 MB")
                content.extend(chunk)
            return bytes(content)

    @staticmethod
    def _find_http_url(payload) -> str | None:
        if isinstance(payload, str):
            return payload if payload.startswith(("http://", "https://")) else None
        if isinstance(payload, list):
            for item in payload:
                url = CryptoVaultPlugin._find_http_url(item)
                if url:
                    return url
            return None
        if not isinstance(payload, dict):
            return None

        for key in ("url", "image_url", "src"):
            url = CryptoVaultPlugin._find_http_url(payload.get(key))
            if url:
                return url
        for value in payload.values():
            url = CryptoVaultPlugin._find_http_url(value)
            if url:
                return url
        return None

    @staticmethod
    def _component_payload(component) -> dict:
        component_type = type(component).__name__.lower()
        data = {}
        raw_data = getattr(component, "data", None)
        if isinstance(raw_data, dict):
            data.update(raw_data)
        elif raw_data is not None:
            data["data"] = raw_data
        for name in (
            "id", "resid", "message_id", "forward_id", "content",
            "message", "raw_message", "url", "file", "image_url", "text"
        ):
            value = getattr(component, name, None)
            if value is not None:
                data[name] = value
        return {"type": component_type, "data": data}

    async def _extract_all_images(self, event: AstrMessageEvent) -> tuple[list, list, int]:
        """Extract image URLs from direct, replied, forwarded and nested messages."""
        images = []
        errors = []
        expired_count = 0
        visited_forward_ids = set()
        visited_reply_ids = set()
        seen_image_references = set()
        resolved_file_ids = set()
        unresolved_file_ids = set()

        client = None
        if hasattr(event, "bot") and hasattr(event.bot, "api"):
            client = event.bot

        def add_image_url(url: str) -> None:
            if url.startswith(("http://", "https://")) and url not in images:
                images.append(url)

        async def resolve_image_reference(reference) -> None:
            marker = repr(reference)
            if marker in seen_image_references:
                return
            seen_image_references.add(marker)

            urls, file_ids = image_reference_locations(reference)
            for url in urls:
                add_image_url(url)
            if urls:
                # 同一图片可能在另一份载荷中只剩 file 字段。只要任意一份
                # 载荷带有 URL，就不能再把这个 file 计为不可读取。
                resolved_file_ids.update(file_ids)
                unresolved_file_ids.difference_update(file_ids)
                return

            for file_id in file_ids:
                if file_id in resolved_file_ids:
                    return
                if not client:
                    unresolved_file_ids.add(file_id)
                    continue
                try:
                    image_data = await client.api.call_action("get_image", file=file_id)
                    url = self._find_http_url(image_data)
                    if url:
                        add_image_url(url)
                        resolved_file_ids.add(file_id)
                        unresolved_file_ids.discard(file_id)
                        return
                except Exception as e:
                    logger.warning(f"[CryptoVault] 通过 get_image 补取图片 {file_id} 失败: {e}")
                unresolved_file_ids.add(file_id)

            # NapCat 有时会额外返回一个只有 [图片] 预览信息、没有 URL/file
            # 的 image 段。这不是一张额外的失效图片，不参与数量统计。

        async def consume_payload(payload, depth: int = 0) -> None:
            nonlocal expired_count
            scan = scan_message_payload(payload)
            expired_count += scan.expired_count
            for reference in scan.image_refs:
                await resolve_image_reference(reference)
            for forward_id in scan.forward_ids:
                await parse_forward(forward_id, depth + 1)

        async def parse_forward(forward_id: str, depth: int) -> None:
            if not client or not forward_id or forward_id in visited_forward_ids:
                return
            if depth > _MAX_FORWARD_DEPTH:
                errors.append("合并转发嵌套层级过深")
                return
            if len(visited_forward_ids) >= _MAX_FORWARD_RECORDS:
                errors.append("单次读取的合并转发记录过多")
                return

            visited_forward_ids.add(forward_id)
            forward_data = None
            last_error = None
            for params in ({"id": forward_id}, {"message_id": forward_id}):
                try:
                    candidate = await client.api.call_action("get_forward_msg", **params)
                    if candidate:
                        forward_data = candidate
                        break
                except Exception as e:
                    last_error = e

            if forward_data:
                await consume_payload(forward_data, depth)
                return

            logger.error(f"[CryptoVault] 提取合并转发消息 {forward_id} 失败: {last_error}")
            error_text = str(last_error or "")
            if "1200" in error_text or "找不到相关" in error_text:
                errors.append("获取聊天记录失败（记录未同步、已过期或嵌套记录不可用）")
            else:
                errors.append("获取聊天记录失败（NapCat 未返回记录内容）")

        async def parse_replied_message(message_id) -> None:
            if not client or message_id is None:
                return
            message_id = str(message_id).strip()
            if not message_id or message_id in visited_reply_ids:
                return
            visited_reply_ids.add(message_id)

            image_count_before = len(images)
            needs_history_fallback = True
            try:
                original_msg = await client.api.call_action(
                    "get_msg", message_id=message_id
                )
                original_scan = scan_message_payload(original_msg)
                needs_history_fallback = (
                    original_scan.forward_segments > 0
                    or not original_scan.image_refs
                )
                await consume_payload(original_msg)
            except Exception as e:
                logger.error(f"[CryptoVault] 底层获取引用消息 {message_id} 失败: {e}")

            # NapCat 的 get_msg 默认不展开合并转发。群历史接口可通过
            # parse_mult_msg 将跨会话转发后的记录直接展开为节点内容。
            if needs_history_fallback or len(images) == image_count_before:
                group_id = event.get_group_id()
                if group_id:
                    try:
                        history = await client.api.call_action(
                            "get_group_msg_history",
                            group_id=str(group_id),
                            message_seq=message_id,
                            count=1,
                            reverse_order=False,
                            parse_mult_msg=True,
                        )
                        await consume_payload(history)
                    except Exception as e:
                        logger.error(
                            f"[CryptoVault] 从群历史展开引用消息 {message_id} 失败: {e}"
                        )
                        errors.append("无法从群历史展开被引用的合并转发记录")

        async def consume_component(component, depth: int = 0) -> None:
            if isinstance(component, Comp.Image):
                await resolve_image_reference({
                    "url": getattr(component, "url", None),
                    "file": getattr(component, "file", None),
                    "image_url": getattr(component, "image_url", None),
                })
                return
            await consume_payload(self._component_payload(component), depth)

        for comp in event.message_obj.message:
            if isinstance(comp, Comp.Reply):
                for chain_comp in getattr(comp, "chain", None) or []:
                    await consume_component(chain_comp)

                await parse_replied_message(getattr(comp, "id", None))
            else:
                await consume_component(comp)

        # AstrBot 有时会把跨会话转发降级为纯文本，但原始 OneBot 事件中
        # 仍保留 reply 段及被引用消息 ID。
        raw_event = getattr(event.message_obj, "raw_message", None)
        if raw_event is not None:
            if not isinstance(raw_event, (dict, list, tuple, str)):
                try:
                    raw_event = dict(raw_event)
                except (TypeError, ValueError):
                    raw_event = getattr(raw_event, "__dict__", None)
            if raw_event:
                raw_scan = scan_message_payload(raw_event)
                await consume_payload(raw_event)
                for reply_id in raw_scan.reply_ids:
                    await parse_replied_message(reply_id)

        unavailable_count = max(
            len(unresolved_file_ids - resolved_file_ids),
            expired_count,
        )
        if expired_count:
            errors.append("聊天记录中存在 QQ 已标记为过期的图片")
        return images, errors, unavailable_count

    def _create_chunked_forward_nodes(
        self,
        sender_id: str,
        sender_name: str,
        saved_paths: list,
        chunk_size: int = 6,
    ):
        """将大量图片分块打包成多个合并转发节点，每个节点（气泡）最多包含 chunk_size 张图"""
        nodes = []
        for i in range(0, len(saved_paths), chunk_size):
            chunk_paths = saved_paths[i:i+chunk_size]
            segments = [forward_utils.local_image_to_segment(p) for p in chunk_paths]
            nodes.append(
                forward_utils.create_forward_node(sender_id, sender_name, segments)
            )
        return nodes

    @staticmethod
    def _get_forward_sender(event: AstrMessageEvent) -> tuple[str, str]:
        """Use the command sender's QQ identity for forward avatar and nickname."""
        sender_id = str(event.get_sender_id() or "").strip()
        get_sender_name = getattr(event, "get_sender_name", None)
        sender_name = ""
        if callable(get_sender_name):
            sender_name = str(get_sender_name() or "").strip()
        if not sender_name:
            sender = getattr(event.message_obj, "sender", None)
            sender_name = str(getattr(sender, "nickname", "") or "").strip()
        if not sender_id:
            sender_id = str(event.message_obj.self_id)
        if not sender_name:
            sender_name = sender_id
        return sender_id, sender_name

    @staticmethod
    def _unwrap_action_list(response, key: str) -> list:
        if isinstance(response, list):
            return response
        if not isinstance(response, dict):
            return []
        value = response.get(key)
        if isinstance(value, list):
            return value
        data = response.get("data")
        if isinstance(data, list):
            return data
        if isinstance(data, dict) and isinstance(data.get(key), list):
            return data[key]
        return []

    @staticmethod
    def _history_cursor(messages: list) -> str | None:
        candidates = []
        for index, message in enumerate(messages):
            if not isinstance(message, dict):
                continue
            try:
                timestamp = int(message.get("time", 0))
            except (TypeError, ValueError):
                timestamp = 0
            message_id = (
                message.get("message_id")
                or message.get("message_seq")
                or message.get("real_id")
                or message.get("id")
            )
            if message_id not in (None, ""):
                candidates.append((timestamp, index, str(message_id)))
        if not candidates:
            return None
        return min(candidates, key=lambda item: (item[0] or 2**63, item[1]))[2]

    async def _fetch_private_contribution_history(
        self,
        bot,
        user_id: str,
        bot_id: str,
        cutoff: int | None,
    ) -> list[dict]:
        records = []
        seen_messages = set()
        cursor = None

        for page_index in range(_MAX_HISTORY_PAGES):
            params = {
                "user_id": str(user_id),
                "count": _HISTORY_PAGE_SIZE,
                "disable_get_url": True,
                "parse_mult_msg": False,
            }
            if cursor:
                params["message_seq"] = cursor
                # NapCat uses reverseOrder. Keep the snake_case variant for
                # protocol implementations that normalize OneBot parameters.
                params["reverseOrder"] = True
                params["reverse_order"] = True
            else:
                params["reverseOrder"] = False
                params["reverse_order"] = False

            try:
                response = await bot.api.call_action(
                    "get_friend_msg_history", **params
                )
            except Exception as e:
                if not cursor and ("不存在" in str(e) or "not exist" in str(e).lower()):
                    return []
                raise

            messages = self._unwrap_action_list(response, "messages")
            if not messages:
                break

            new_messages = []
            for index, message in enumerate(messages):
                if not isinstance(message, dict):
                    continue
                message_id = (
                    message.get("message_id")
                    or message.get("message_seq")
                    or message.get("real_id")
                    or f"{message.get('time', 0)}:{index}:{repr(message.get('message'))}"
                )
                marker = str(message_id)
                if marker in seen_messages:
                    continue
                seen_messages.add(marker)
                new_messages.append(message)

            if not new_messages:
                break
            records.extend(
                extract_history_contributions(new_messages, bot_id, cutoff)
            )

            next_cursor = self._history_cursor(messages)
            if (
                not next_cursor
                or next_cursor == cursor
            ):
                break
            cursor = next_cursor

            if (page_index + 1) % 25 == 0:
                logger.info(
                    f"[CryptoVault] 正在回填用户 {user_id} 的私聊历史，"
                    f"已扫描 {len(seen_messages)} 条消息"
                )
        else:
            logger.warning(
                f"[CryptoVault] 用户 {user_id} 私聊历史超过扫描上限，"
                f"仅检查最近 {_HISTORY_PAGE_SIZE * _MAX_HISTORY_PAGES} 条消息"
            )

        return records

    async def _backfill_contribution_history(self, event: AstrMessageEvent) -> None:
        bot = getattr(event, "bot", None)
        if not bot or not hasattr(bot, "api"):
            return

        candidates = {}
        try:
            response = await bot.api.call_action("get_friend_list", no_cache=False)
            for friend in self._unwrap_action_list(response, "friends"):
                if not isinstance(friend, dict):
                    continue
                user_id = str(friend.get("user_id") or "").strip()
                if not user_id:
                    continue
                candidates[user_id] = (
                    friend.get("remark") or friend.get("nickname") or user_id
                )
        except Exception as e:
            logger.warning(f"[CryptoVault] 获取好友列表用于贡献榜回填失败: {e}")

        async with self._stats_lock:
            for user_id in self.contribution_store.user_ids():
                user = self.contribution_store.data["users"].get(user_id, {})
                candidates.setdefault(user_id, user.get("name") or user_id)

        get_self_id = getattr(event, "get_self_id", None)
        bot_id = str(
            (get_self_id() if callable(get_self_id) else "")
            or event.message_obj.self_id
        )
        for user_id, name in candidates.items():
            async with self._stats_lock:
                imported, _ = self.contribution_store.history_state(
                    user_id, str(name)
                )
            if imported:
                continue

            try:
                records = await self._fetch_private_contribution_history(
                    bot, user_id, bot_id, None
                )
            except Exception as e:
                logger.warning(
                    f"[CryptoVault] 回填用户 {user_id} 的私聊贡献记录失败: {e}"
                )
                continue

            async with self._stats_lock:
                self.contribution_store.import_history(
                    user_id, str(name), records
                )

    async def _render_contribution_ranking(
        self,
        event: AstrMessageEvent,
        month_only: bool,
    ):
        if not self._history_backfill_announced:
            self._history_backfill_announced = True
            yield event.plain_result("首次整理贡献记录，正在读取可用的私聊历史，请稍候...")

        await self._backfill_contribution_history(event)

        now = datetime.now()
        month_key = now.strftime("%Y-%m") if month_only else None
        async with self._stats_lock:
            ranking = self.contribution_store.ranking(month_key, limit=10)

        if not ranking:
            period = "本月" if month_only else "目前"
            yield event.plain_result(
                f"{period}还没有通过 /加密2 成功分享的图片记录。"
            )
            return

        template_path = os.path.join(
            os.path.dirname(__file__), "contribution_ranking.html"
        )
        try:
            with open(template_path, "r", encoding="utf-8") as file:
                template_content = file.read()
        except OSError as e:
            logger.error(f"[CryptoVault] 读取贡献榜模板失败: {e}")
            yield event.plain_result("贡献榜模板读取失败。")
            return

        if month_only:
            title = f"{now.year}年{now.month}月 涩图英雄月榜"
            subtitle = "本月为群友无私分享的加密图片"
            footer = f"统计周期：{now.year}年{now.month}月"
        else:
            title = "涩图英雄总榜"
            subtitle = "每一张分享，都是给群友的珍贵贡献"
            footer = "累计统计成功通过 /加密2 分享的图片"

        compact_layout = len(ranking) > 6
        if compact_layout:
            dynamic_height = 118 + len(ranking) * 52 + 42
        else:
            dynamic_height = 142 + len(ranking) * 76 + 58
        try:
            url = await self.html_render(
                template_content,
                {
                    "ranking": ranking,
                    "title": title,
                    "subtitle": subtitle,
                    "footer": footer,
                    "canvas_height": dynamic_height,
                    "compact_layout": compact_layout,
                },
                options={
                    "viewport_width": 520,
                    "viewport_height": dynamic_height,
                    "type": "jpeg",
                    "quality": 100,
                    "full_page": True,
                    "scale": "device",
                    "device_scale_factor_level": "ultra",
                },
            )
            yield event.image_result(url)
        except Exception as e:
            logger.error(f"[CryptoVault] 渲染贡献榜失败: {e}")
            yield event.plain_result("贡献榜绘制失败，请查看 AstrBot 日志。")

    @staticmethod
    def _extraction_failure_message(errors: list, unavailable_count: int, default: str) -> str:
        if unavailable_count:
            reason = errors[0] if errors else "QQ 未提供可下载的图片文件"
            return (
                f"未能读取可用图片：{reason}。"
                "如果记录中显示“[图片]已过期”，原图数据已不在记录中，无法恢复。"
            )
        if errors:
            return f"未能提取到图片：{errors[0]}，请稍后重试。"
        return default

    @staticmethod
    def _cleanup_files(paths: list) -> None:
        for path in paths:
            try:
                os.remove(path)
            except OSError:
                pass

    def _process_image_sync(self, img_bytes: bytes, mode: str, key: float) -> str:
        if mode not in {"encrypt", "decrypt"}:
            raise ValueError("不支持的图片处理模式")

        with PILImage.open(io.BytesIO(img_bytes)) as source:
            width, height = source.size
            if width <= 0 or height <= 0 or width * height > _MAX_PIXEL_COUNT:
                raise ValueError("图片尺寸过大")
            # 必须先检查尺寸再 convert；convert 会实际解码并分配整张图的内存。
            img = source.convert("RGB")
            
        pixels = np.asarray(img, dtype=np.uint8).reshape(-1, 3)
        scrambler = TomatoScramble(pixels, width, height, key)
        
        new_pixels = scrambler.encrypt() if mode == "encrypt" else scrambler.decrypt()
        new_img = PILImage.fromarray(new_pixels.reshape(height, width, 3), "RGB")

        filename = f"{mode}_{uuid.uuid4().hex}.png"
        save_path = os.path.join(self.data_dir, filename)
        new_img.save(save_path, "PNG")
        return save_path

    async def _process_image_async(self, img_url: str, mode: str, key: float) -> str:
        async with self._process_semaphore:
            img_bytes = await self._download_image(img_url)
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(
                None, functools.partial(self._process_image_sync, img_bytes, mode, key)
            )

    @filter.command("加密", alias=["混淆"])
    @filter.platform_adapter_type(filter.PlatformAdapterType.AIOCQHTTP)
    async def encrypt_cmd(self, event: AstrMessageEvent, key: float = 1.0):
        event.stop_event() # 拦截事件，防止大图进入 LLM 导致崩溃
        
        image_urls, errors, unavailable_count = await self._extract_all_images(event)
        if not image_urls:
            message = self._extraction_failure_message(
                errors,
                unavailable_count,
                "请在指令中附带或引用需要加密的图片。",
            )
            yield event.plain_result(message)
            return

        yield event.plain_result("收到，正在混淆...（使用“/加密2”会将结果转发到安全群聊）")
        saved_paths = []

        for url in image_urls:
            try:
                path = await self._process_image_async(url, "encrypt", key)
                saved_paths.append(path)
            except Exception as e:
                logger.error(f"加密失败: {e}")

        if saved_paths:
            if len(saved_paths) <= 6:
                yield event.chain_result([Comp.Image.fromFileSystem(p) for p in saved_paths])
            else:
                sender_id, sender_name = self._get_forward_sender(event)
                nodes = self._create_chunked_forward_nodes(
                    sender_id, sender_name, saved_paths, chunk_size=6
                )
                group_id = event.get_group_id()
                if group_id:
                    await forward_utils.send_group_forward_message_by_api(event.bot, int(group_id), nodes)
                else:
                    await forward_utils.send_private_forward_message_by_api(
                        event.bot, int(event.get_sender_id()), nodes
                    )

            self._cleanup_files(saved_paths)
        else:
            yield event.plain_result("图片处理失败，请检查图片格式、文件大小或尺寸。")

    @filter.command("加密2", alias=["混淆2"])
    @filter.platform_adapter_type(filter.PlatformAdapterType.AIOCQHTTP)
    async def encrypt2_cmd(self, event: AstrMessageEvent, key: float = 1.0):
        event.stop_event() # 拦截事件，防止大图进入 LLM 导致崩溃
        
        if not self.target_groups:
            yield event.plain_result("错误：管理员尚未配置加密图片的接收群聊。")
            return

        image_urls, errors, unavailable_count = await self._extract_all_images(event)
        if not image_urls:
            message = self._extraction_failure_message(
                errors,
                unavailable_count,
                "请在指令中附带或引用需要加密的图片。",
            )
            yield event.plain_result(message)
            return

        yield event.plain_result("正在加密并存入安全图库...")
        saved_paths = []

        for url in image_urls:
            try:
                path = await self._process_image_async(url, "encrypt", key)
                saved_paths.append(path)
            except Exception as e:
                logger.error(f"加密2失败: {e}")

        if saved_paths:
            # 安全图库本质是合并转发，直接分块（每节点6张），防止安全群因为节点过大而掉线
            sender_id, sender_name = self._get_forward_sender(event)
            nodes = self._create_chunked_forward_nodes(
                sender_id, sender_name, saved_paths, chunk_size=6
            )
            
            success_count = 0
            for group_id in self.target_groups:
                if await forward_utils.send_group_forward_message_by_api(event.bot, int(group_id), nodes):
                    success_count += 1

            if success_count > 0:
                try:
                    async with self._stats_lock:
                        self.contribution_store.record_live(
                            sender_id,
                            sender_name,
                            len(saved_paths),
                        )
                except Exception as e:
                    logger.error(f"[CryptoVault] 保存分享贡献统计失败: {e}")

            self._cleanup_files(saved_paths)

            yield event.plain_result(
                f"处理完毕！已将 {len(saved_paths)} 张加密图片转发至 "
                f"{success_count} 个安全群聊。"
            )
        else:
            yield event.plain_result("图片处理失败，请检查图片格式、文件大小或尺寸。")

    @filter.command("涩图榜")
    @filter.platform_adapter_type(filter.PlatformAdapterType.AIOCQHTTP)
    async def contribution_total_ranking_cmd(self, event: AstrMessageEvent):
        event.stop_event()
        async for result in self._render_contribution_ranking(
            event, month_only=False
        ):
            yield result

    @filter.command("涩图月榜")
    @filter.platform_adapter_type(filter.PlatformAdapterType.AIOCQHTTP)
    async def contribution_month_ranking_cmd(self, event: AstrMessageEvent):
        event.stop_event()
        async for result in self._render_contribution_ranking(
            event, month_only=True
        ):
            yield result

    @filter.command("解密", alias=["解析"])
    @filter.platform_adapter_type(filter.PlatformAdapterType.AIOCQHTTP)
    async def decrypt_cmd(self, event: AstrMessageEvent, key: float = 1.0):
        event.stop_event() # 拦截事件，防止大图进入 LLM 导致崩溃
        
        image_urls, errors, unavailable_count = await self._extract_all_images(event)
        if not image_urls:
            message = self._extraction_failure_message(
                errors,
                unavailable_count,
                "请引用需要解密的图片或合并转发记录。",
            )
            yield event.plain_result(message)
            return

        bot_id = str(event.message_obj.self_id)
        user_id = int(event.get_sender_id())

        # 【修改点】改为在私聊发送“正在解密”提示，群内完全静默
        try:
            await event.bot.api.call_action('send_private_msg', user_id=user_id, message="正在为您解密，请稍候...")
        except Exception as e:
            logger.warning(f"[CryptoVault] 尝试发送私聊状态提示失败: {e}")

        saved_paths = []

        for url in image_urls:
            try:
                path = await self._process_image_async(url, "decrypt", key)
                saved_paths.append(path)
            except Exception as e:
                logger.error(f"解密失败: {e}")

        if saved_paths:
            # 解密同样采用分块机制，每6张图合并成一个节点发出
            nodes = self._create_chunked_forward_nodes(bot_id, "解密结果", saved_paths, chunk_size=6)
            
            success = await forward_utils.send_private_forward_message_by_api(event.bot, user_id, nodes)
            
            self._cleanup_files(saved_paths)
                
            if success:
                try:
                    await event.bot.api.call_action(
                        "send_private_msg",
                        user_id=user_id,
                        message=f"解密完成，共 {len(saved_paths)} 张。",
                    )
                except Exception:
                    pass
            else:
                yield event.plain_result("私聊发送失败，请确保您已添加机器人为好友或允许临时会话。")
        else:
            yield event.plain_result("图片解密失败，请检查密钥、图片格式、文件大小或尺寸。")

    async def terminate(self):
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None
