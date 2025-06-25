"""
统一视频下载器
支持YouTube和Bilibili的高质量视频下载
"""

__version__ = "1.0.0"
__author__ = "Video Downloader Team"

from core.downloader import UnifiedDownloader
from core.url_detector import URLDetector

__all__ = ['UnifiedDownloader', 'URLDetector']
