"""
Bilibili video downloader implementation
"""

import asyncio
import aiohttp
import json
import os
import re
from typing import Dict, Any, Optional, List, Callable
from urllib.parse import urlencode
import hashlib
import time

from config.config import Config
from core.url_detector import URLDetector, VideoSource

class BilibiliDownloader:
    """Async Bilibili video downloader"""
    
    def __init__(self):
        self.session: Optional[aiohttp.ClientSession] = None
        self.headers = Config.BILIBILI_HEADERS.copy()
        
    async def __aenter__(self):
        """Async context manager entry"""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=Config.REQUEST_TIMEOUT),
            headers=self.headers
        )

        # 自动检查Cookie状态
        try:
            from auth.cookie_monitor import integrate_cookie_monitor
            integrate_cookie_monitor()
        except:
            # 静默处理，不影响主功能
            pass

        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
            
    async def _request_get(self, url: str, **kwargs) -> Dict[str, Any]:
        """Make async GET request"""
        if not self.session:
            raise RuntimeError("Session not initialized. Use async context manager.")
            
        async with self.session.get(url, **kwargs) as response:
            response.raise_for_status()
            return await response.json()
            
    async def _get_video_info(self, video_id: str) -> Dict[str, Any]:
        """Get video information from bilibili API"""
        # Convert AV to BV if needed
        if video_id.startswith('av'):
            aid = int(video_id[2:])
            bvid = self._aid_to_bvid(aid)
        else:
            bvid = video_id
            
        url = f"https://api.bilibili.com/x/web-interface/view?bvid={bvid}"
        
        try:
            data = await self._request_get(url)
            if data.get('code') != 0:
                raise Exception(f"API Error: {data.get('message', 'Unknown error')}")
            return data['data']
        except Exception as e:
            raise Exception(f"Failed to get video info: {str(e)}")
            
    def _get_quality_value(self, quality: str) -> int:
        """Convert quality string to bilibili quality value"""
        quality_map = {
            'best': 120,      # 4K
            '1080p': 80,      # 1080P
            '720p': 64,       # 720P
            '480p': 32,       # 480P
            '360p': 16,       # 360P
            'worst': 6        # 240P
        }
        return quality_map.get(quality, 80)  # Default to 1080P

    async def _get_video_stream_urls(self, bvid: str, cid: int, quality: int = 80) -> Dict[str, Any]:
        """Get video stream URLs"""
        params = {
            'bvid': bvid,
            'cid': cid,
            'qn': quality,
            'fnver': 0,
            'fnval': 4048,
            'fourk': 1,
            'platform': 'pc',
            'high_quality': 1
        }

        # 添加更多认证相关的headers
        headers = self.headers.copy()
        headers.update({
            'Origin': 'https://www.bilibili.com',
            'Sec-Fetch-Dest': 'empty',
            'Sec-Fetch-Mode': 'cors',
            'Sec-Fetch-Site': 'same-site'
        })

        # 添加Cookie认证支持
        try:
            from auth.bilibili_cookies import get_cookies
            cookies = get_cookies()
            if cookies:
                cookie_str = '; '.join([f'{k}={v}' for k, v in cookies.items()])
                headers['Cookie'] = cookie_str
        except ImportError:
            # 如果没有bilibili_cookies模块，尝试从Config获取
            cookies = Config.get_bilibili_cookies()
            if cookies:
                cookie_str = '; '.join([f'{k}={v}' for k, v in cookies.items()])
                headers['Cookie'] = cookie_str

        url = f"https://api.bilibili.com/x/player/wbi/playurl?{urlencode(params)}"
        
        try:
            # 使用自定义headers发送请求
            async with self.session.get(url, headers=headers) as response:
                response.raise_for_status()
                data = await response.json()
                if data.get('code') != 0:
                    raise Exception(f"Stream API Error: {data.get('message', 'Unknown error')}")
                return data['data']
        except Exception as e:
            raise Exception(f"Failed to get stream URLs: {str(e)}")
            
    async def _download_file_chunk(self, url: str, start: int, end: int) -> bytes:
        """Download a chunk of file"""
        headers = self.headers.copy()
        headers['Range'] = f'bytes={start}-{end}'
        
        async with self.session.get(url, headers=headers) as response:
            response.raise_for_status()
            return await response.read()
            
    async def _download_stream(self, url: str, filepath: str, 
                             progress_callback: Optional[Callable[[int, int], None]] = None) -> None:
        """Download video stream with progress tracking"""
        # Get file size
        async with self.session.head(url) as response:
            response.raise_for_status()
            total_size = int(response.headers.get('Content-Length', 0))
            
        if total_size == 0:
            raise Exception("Could not determine file size")
            
        # Download in chunks
        chunk_size = 1024 * 1024  # 1MB chunks
        downloaded = 0
        
        with open(filepath, 'wb') as f:
            while downloaded < total_size:
                end = min(downloaded + chunk_size - 1, total_size - 1)
                chunk = await self._download_file_chunk(url, downloaded, end)
                f.write(chunk)
                downloaded += len(chunk)
                
                if progress_callback:
                    progress_callback(downloaded, total_size)
                    
    def _aid_to_bvid(self, aid: int) -> str:
        """Convert AV ID to BV ID"""
        XOR_CODE = 23442827791579
        MAX_AID = 1 << 51
        ALPHABET = "FcwAPNKTMug3GV5Lj7EJnHpWsx4tb8haYeviqBz6rkCy12mUSDQX9RdoZf"
        ENCODE_MAP = (8, 7, 0, 5, 1, 3, 2, 4, 6)
        
        bvid = [""] * 9
        tmp = (MAX_AID | aid) ^ XOR_CODE
        
        for i in range(len(ENCODE_MAP)):
            bvid[ENCODE_MAP[i]] = ALPHABET[tmp % len(ALPHABET)]
            tmp //= len(ALPHABET)
            
        return "BV1" + "".join(bvid)
        
    def _get_legal_filename(self, filename: str) -> str:
        """Get legal filename by removing invalid characters"""
        return re.sub(r'[<>:"/\\|?*]', '_', filename)

    def _select_best_stream(self, stream_data: Dict[str, Any], quality: str) -> str:
        """Select the best stream URL based on quality preference"""
        if 'dash' in stream_data:
            # DASH format - select best video stream
            video_streams = stream_data['dash']['video']
            if not video_streams:
                raise Exception("No video streams available")

            # Sort streams by quality (higher id usually means higher quality)
            sorted_streams = sorted(video_streams, key=lambda x: x.get('id', 0), reverse=True)

            # For bilibili, select based on quality preference
            if quality == 'best':
                # Select highest quality
                selected_stream = sorted_streams[0]
            elif quality == 'worst':
                # Select lowest quality
                selected_stream = sorted_streams[-1]
            else:
                # Select best stream that matches or is lower than requested quality
                quality_value = self._get_quality_value(quality)
                selected_stream = sorted_streams[-1]  # Default to lowest

                for stream in sorted_streams:
                    stream_quality = stream.get('id', 0)
                    if stream_quality <= quality_value:
                        selected_stream = stream
                        break

            return selected_stream['baseUrl']

        elif 'durl' in stream_data:
            # FLV format - usually single stream
            return stream_data['durl'][0]['url']
        else:
            raise Exception("No supported stream format found")
        
    async def download(self, url: str, output_path: str = None, quality: str = "best", info_only: bool = False) -> Dict[str, Any]:
        """
        Download bilibili video

        Args:
            url: Bilibili video URL
            output_path: Output file path (can be directory or full file path)
            quality: Video quality preference
            info_only: If True, only return video info without downloading

        Returns:
            Dictionary with download result information
        """
        # Analyze URL
        url_info = URLDetector.analyze_url(url)
        if url_info['source'] != VideoSource.BILIBILI:
            raise ValueError("Not a valid bilibili URL")
            
        video_id = url_info.get('video_id')
        if not video_id:
            raise ValueError("Could not extract video ID from URL")
            
        # Get video information
        video_info = await self._get_video_info(video_id)
        
        # Get first page if no specific part requested
        page_info = video_info['pages'][0]
        cid = page_info['cid']
        
        # If info_only, return video info without downloading
        if info_only:
            return {
                'title': video_info['title'],
                'duration': video_info.get('duration', 0),
                'views': video_info.get('stat', {}).get('view', 0),
                'description': video_info.get('desc', ''),
                'uploader': video_info.get('owner', {}).get('name', ''),
                'upload_date': video_info.get('pubdate', 0)
            }

        # Get stream URLs with quality preference
        quality_value = self._get_quality_value(quality)
        stream_data = await self._get_video_stream_urls(video_id, cid, quality_value)

        # Handle output path - support both directory and full file path
        if output_path:
            if output_path.endswith('.mp4') or '.' in os.path.basename(output_path):
                # output_path is a full file path
                filepath = os.path.abspath(output_path)
                # Ensure directory exists
                dir_path = os.path.dirname(filepath)
                if dir_path:  # Only create if directory path is not empty
                    os.makedirs(dir_path, exist_ok=True)
            else:
                # output_path is a directory
                title = self._get_legal_filename(video_info['title'])
                filename = f"{title}_{video_id}.mp4"
                filepath = os.path.abspath(os.path.join(output_path, filename))
                os.makedirs(output_path, exist_ok=True)
        else:
            # Use default download path
            download_dir = Config.get_download_path()
            title = self._get_legal_filename(video_info['title'])
            filename = f"{title}_{video_id}.mp4"
            filepath = os.path.abspath(os.path.join(download_dir, filename))
            os.makedirs(download_dir, exist_ok=True)
        
        # Get best quality stream URL based on quality preference
        stream_url = self._select_best_stream(stream_data, quality)
            
        # Download the video
        await self._download_stream(stream_url, filepath)
        
        return {
            'success': True,
            'title': video_info['title'],
            'video_id': video_id,
            'file_path': filepath,
            'duration': video_info.get('duration', 0),
            'views': video_info.get('stat', {}).get('view', 0),
            'quality': quality,
            'source': 'bilibili'
        }
