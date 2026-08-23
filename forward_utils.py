from typing import List, Dict, Any
import os
from astrbot.api import logger

async def send_group_forward_message_by_api(
    bot_client: Any,
    group_id: int,
    nodes: List[Dict]
) -> bool:
    """向指定群聊发送合并转发消息"""
    if not bot_client or not hasattr(bot_client, 'api'):
        return False
    try:
        await bot_client.api.call_action(
            'send_group_forward_msg',
            group_id=group_id,
            messages=nodes
        )
        return True
    except Exception as e:
        logger.error(f"[CryptoVault] 向群聊 {group_id} 发送合并转发失败: {e}")
        return False

async def send_private_forward_message_by_api(
    bot_client: Any,
    user_id: int,
    nodes: List[Dict]
) -> bool:
    """向指定用户私聊发送合并转发消息 (OneBot v11 支持)"""
    if not bot_client or not hasattr(bot_client, 'api'):
        return False
    try:
        await bot_client.api.call_action(
            'send_private_forward_msg',
            user_id=user_id,
            messages=nodes
        )
        return True
    except Exception as e:
        logger.error(f"[CryptoVault] 向用户 {user_id} 私聊发送合并转发失败: {e}")
        return False

def create_forward_node(user_id: str, nickname: str, content_segments: List[Dict]) -> Dict:
    return {
        "type": "node",
        "data": {
            "user_id": user_id,
            "nickname": nickname,
            "content": content_segments
        }
    }

def local_image_to_segment(file_path: str) -> Dict:
    absolute_path = os.path.abspath(file_path)
    correct_uri = f"file://{absolute_path}"
    return {"type": "image", "data": {"file": correct_uri}}

def text_to_segment(text: str) -> Dict:
    return {"type": "text", "data": {"text": text}}