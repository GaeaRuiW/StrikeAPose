"""
URL detection and routing system for different video platforms
"""

import re
from enum import Enum
from typing import Optional, Dict, Any
from urllib.parse import urlparse, parse_qs

class VideoSource(Enum):
    """Supported video sources"""
    BILIBILI = "bilibili"
    YOUTUBE = "youtube"
    DIRECT_URL = "direct_url"  # 直接视频文件链接
    UNKNOWN = "unknown"

class URLDetector:
    """Detects video source from URL and extracts relevant information"""
    
    # Bilibili URL patterns
    BILIBILI_PATTERNS = [
        r'bilibili\.com',
        r'b23\.tv',
        r'bili2233\.cn',
        r'BV\w+',
        r'av\d+',
        r'ep\d+',
        r'ss\d+'
    ]
    
    # YouTube URL patterns  
    YOUTUBE_PATTERNS = [
        r'youtube\.com',
        r'youtu\.be',
        r'm\.youtube\.com',
        r'www\.youtube\.com'
    ]
    
    @classmethod
    def is_direct_video_url(cls, url: str) -> bool:
        """
        Check if URL is a direct video file link

        Args:
            url: URL to check

        Returns:
            bool: True if URL appears to be a direct video file
        """
        if not url:
            return False

        # 常见视频文件扩展名
        video_extensions = [
            '.mp4', '.avi', '.mkv', '.mov', '.wmv', '.flv', '.webm',
            '.m4v', '.3gp', '.ogv', '.ts', '.m3u8', '.f4v'
        ]

        # 检查URL是否以视频扩展名结尾
        url_lower = url.lower()
        for ext in video_extensions:
            if url_lower.endswith(ext):
                return True

        # 检查URL中是否包含视频扩展名（带参数的情况）
        from urllib.parse import urlparse
        parsed = urlparse(url_lower)
        path = parsed.path
        for ext in video_extensions:
            if ext in path:
                return True

        # 检查常见的视频流媒体模式
        streaming_patterns = [
            '/video/', '/stream/', '/media/', '/download/',
            'video=', 'stream=', 'media=', 'file='
        ]
        for pattern in streaming_patterns:
            if pattern in url_lower:
                return True

        return False

    @classmethod
    def detect_source(cls, url: str) -> VideoSource:
        """
        Detect the video source from URL

        Args:
            url: Video URL to analyze

        Returns:
            VideoSource enum indicating the detected source
        """
        url_lower = url.lower()

        # Check for bilibili patterns
        for pattern in cls.BILIBILI_PATTERNS:
            if re.search(pattern, url_lower):
                return VideoSource.BILIBILI

        # Check for YouTube patterns
        for pattern in cls.YOUTUBE_PATTERNS:
            if re.search(pattern, url_lower):
                return VideoSource.YOUTUBE

        # Check for direct video URL
        if cls.is_direct_video_url(url):
            return VideoSource.DIRECT_URL

        return VideoSource.UNKNOWN
    
    @classmethod
    def extract_bilibili_info(cls, url: str) -> Dict[str, Any]:
        """
        Extract information from bilibili URL
        
        Args:
            url: Bilibili URL
            
        Returns:
            Dictionary containing extracted information
        """
        info = {
            'source': VideoSource.BILIBILI,
            'original_url': url,
            'video_id': None,
            'episode_id': None,
            'season_id': None,
            'part': None,
            'url_type': None
        }
        
        # Extract BV ID
        bv_match = re.search(r'BV\w+', url)
        if bv_match:
            info['video_id'] = bv_match.group()
            info['url_type'] = 'video'
            
        # Extract AV ID
        av_match = re.search(r'av(\d+)', url)
        if av_match:
            info['video_id'] = f"av{av_match.group(1)}"
            info['url_type'] = 'video'
            
        # Extract episode ID
        ep_match = re.search(r'ep(\d+)', url)
        if ep_match:
            info['episode_id'] = ep_match.group(1)
            info['url_type'] = 'episode'
            
        # Extract season ID
        ss_match = re.search(r'ss(\d+)', url)
        if ss_match:
            info['season_id'] = ss_match.group(1)
            info['url_type'] = 'season'
            
        # Extract part number
        part_match = re.search(r'p=(\d+)', url)
        if part_match:
            info['part'] = int(part_match.group(1))
            
        return info
    
    @classmethod
    def extract_youtube_info(cls, url: str) -> Dict[str, Any]:
        """
        Extract information from YouTube URL
        
        Args:
            url: YouTube URL
            
        Returns:
            Dictionary containing extracted information
        """
        info = {
            'source': VideoSource.YOUTUBE,
            'original_url': url,
            'video_id': None,
            'playlist_id': None,
            'timestamp': None,
            'url_type': 'video'
        }
        
        parsed = urlparse(url)
        
        # Handle youtu.be short URLs
        if 'youtu.be' in parsed.netloc:
            video_id = parsed.path.lstrip('/')
            if video_id:
                info['video_id'] = video_id
                
        # Handle youtube.com URLs
        elif 'youtube.com' in parsed.netloc:
            query_params = parse_qs(parsed.query)
            
            # Extract video ID
            if 'v' in query_params:
                info['video_id'] = query_params['v'][0]
                
            # Extract playlist ID
            if 'list' in query_params:
                info['playlist_id'] = query_params['list'][0]
                info['url_type'] = 'playlist'
                
            # Extract timestamp
            if 't' in query_params:
                info['timestamp'] = query_params['t'][0]
                
        return info

    @classmethod
    def extract_direct_url_info(cls, url: str) -> Dict[str, Any]:
        """
        Extract information from direct video URL

        Args:
            url: Direct video URL

        Returns:
            Dictionary containing extracted information
        """
        from urllib.parse import urlparse
        import os

        parsed = urlparse(url)
        filename = os.path.basename(parsed.path)

        # 提取文件扩展名
        name, ext = os.path.splitext(filename)
        if not ext:
            ext = '.mp4'  # 默认扩展名

        info = {
            'source': VideoSource.DIRECT_URL,
            'original_url': url,
            'filename': filename or 'video',
            'extension': ext,
            'domain': parsed.netloc,
            'path': parsed.path,
            'url_type': 'direct'
        }

        return info

    @classmethod
    def analyze_url(cls, url: str) -> Dict[str, Any]:
        """
        Analyze URL and extract all relevant information
        
        Args:
            url: Video URL to analyze
            
        Returns:
            Dictionary containing source type and extracted information
        """
        source = cls.detect_source(url)
        
        if source == VideoSource.BILIBILI:
            return cls.extract_bilibili_info(url)
        elif source == VideoSource.YOUTUBE:
            return cls.extract_youtube_info(url)
        elif source == VideoSource.DIRECT_URL:
            return cls.extract_direct_url_info(url)
        else:
            return {
                'source': VideoSource.UNKNOWN,
                'original_url': url,
                'error': 'Unsupported video source'
            }
