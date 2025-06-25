"""
Configuration settings for the unified video downloader
"""

import os
from typing import Dict, Any

class Config:
    """Configuration class for the downloader"""
    
    # Default download settings
    DEFAULT_DOWNLOAD_PATH = os.path.join(os.getcwd(), "downloads")
    DEFAULT_VIDEO_QUALITY = "best"
    DEFAULT_AUDIO_QUALITY = "best"
    
    # User agent for requests
    USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    
    # Bilibili specific settings
    BILIBILI_HEADERS = {
        "User-Agent": USER_AGENT,
        "Referer": "https://www.bilibili.com/",
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
        "Accept-Encoding": "gzip, deflate, br",
        "Connection": "keep-alive",
        "Sec-Fetch-Dest": "empty",
        "Sec-Fetch-Mode": "cors",
        "Sec-Fetch-Site": "same-site"
    }
    
    # YouTube specific settings  
    YOUTUBE_HEADERS = {
        "User-Agent": USER_AGENT,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
        "Accept-Encoding": "gzip, deflate",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1"
    }
    
    # Timeout settings
    REQUEST_TIMEOUT = 30
    DOWNLOAD_TIMEOUT = 300
    
    # Retry settings
    MAX_RETRIES = 3
    RETRY_DELAY = 1
    
    # Bilibili authentication (optional)
    # Add your bilibili cookies here for high-quality downloads
    BILIBILI_COOKIES = {
        # Example format:
        # 'SESSDATA': 'your_sessdata_here',
        # 'bili_jct': 'your_bili_jct_here',
        # 'DedeUserID': 'your_userid_here',
        # ... other cookies
    }

    # PushPlus notification (optional)
    # Get your token from http://www.pushplus.plus/
    PUSHPLUS_TOKEN = '2a6b548aa1ba47c6ad2d2de0c3861167'  # 测试token，请替换为您的token

    @classmethod
    def get_download_path(cls, custom_path: str = None) -> str:
        """Get the download path, creating it if it doesn't exist"""
        path = custom_path or cls.DEFAULT_DOWNLOAD_PATH
        os.makedirs(path, exist_ok=True)
        return path

    @classmethod
    def get_bilibili_cookies(cls) -> dict:
        """Get bilibili cookies for authentication"""
        return cls.BILIBILI_COOKIES
