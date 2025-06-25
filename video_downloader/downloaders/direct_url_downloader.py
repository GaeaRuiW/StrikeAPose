#!/usr/bin/env python3
"""
Direct URL video downloader
Downloads video files directly from URLs
"""

import os
import aiohttp
import asyncio
import logging
from typing import Dict, Any, Optional
from urllib.parse import urlparse
import time

# Add parent directory to path for imports
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from config.config import Config

class DirectURLDownloader:
    """Downloads videos directly from URLs"""
    
    def __init__(self):
        self.session = None
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'video/webm,video/ogg,video/*;q=0.9,application/ogg;q=0.7,audio/*;q=0.6,*/*;q=0.5',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'identity',
            'Range': 'bytes=0-'
        }
    
    async def __aenter__(self):
        """Async context manager entry"""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=Config.REQUEST_TIMEOUT),
            headers=self.headers
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
    
    def _get_filename_from_url(self, url: str) -> str:
        """Generate filename from URL"""
        # 从URL提取文件名
        parsed = urlparse(url)
        filename = os.path.basename(parsed.path)

        if not filename or '.' not in filename:
            # 如果没有文件名或扩展名，生成一个
            timestamp = int(time.time())
            filename = f"video_{timestamp}.mp4"

        return filename
    
    def _get_output_path(self, filename: str) -> str:
        """Get full output path"""
        if os.path.isabs(filename):
            return filename
        
        # 使用默认下载目录
        download_dir = Config.get_download_path()
        return os.path.join(download_dir, filename)
    
    async def _get_video_info(self, url: str) -> Dict[str, Any]:
        """Get video information from URL"""
        try:
            async with self.session.head(url) as response:
                if response.status in [200, 206]:  # 206 = Partial Content (正常)
                    headers = response.headers
                    
                    # 获取文件大小
                    content_length = headers.get('Content-Length')
                    file_size = int(content_length) if content_length else None
                    
                    # 获取内容类型
                    content_type = headers.get('Content-Type', '')
                    
                    # 从URL提取文件名
                    parsed = urlparse(url)
                    filename = os.path.basename(parsed.path) or 'video'
                    
                    return {
                        'title': filename,
                        'filename': filename,
                        'file_size': file_size,
                        'content_type': content_type,
                        'url': url,
                        'duration': None,  # 无法从直接URL获取时长
                        'views': None      # 无法从直接URL获取观看数
                    }
                else:
                    raise Exception(f"无法访问视频URL: HTTP {response.status}")
                    
        except Exception as e:
            logging.error(f"获取视频信息失败: {e}")
            # 返回基本信息
            parsed = urlparse(url)
            filename = os.path.basename(parsed.path) or 'video'
            return {
                'title': filename,
                'filename': filename,
                'file_size': None,
                'content_type': 'video/mp4',
                'url': url,
                'duration': None,
                'views': None
            }
    
    async def _download_with_progress(self, url: str, output_path: str) -> Dict[str, Any]:
        """Download video with progress tracking"""
        try:
            async with self.session.get(url) as response:
                if response.status != 200:
                    raise Exception(f"下载失败: HTTP {response.status}")
                
                # 获取文件大小
                total_size = response.headers.get('Content-Length')
                total_size = int(total_size) if total_size else None
                
                # 确保输出目录存在
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                
                downloaded = 0
                chunk_size = 8192
                
                with open(output_path, 'wb') as f:
                    async for chunk in response.content.iter_chunked(chunk_size):
                        f.write(chunk)
                        downloaded += len(chunk)
                        
                        # 显示进度
                        if total_size:
                            progress = (downloaded / total_size) * 100
                            print(f"\r下载进度: {progress:.1f}% ({downloaded}/{total_size} bytes)", end='', flush=True)
                        else:
                            print(f"\r已下载: {downloaded} bytes", end='', flush=True)
                
                print()  # 换行
                
                return {
                    'file_path': output_path,
                    'file_size': downloaded,
                    'success': True
                }
                
        except Exception as e:
            logging.error(f"下载过程出错: {e}")
            raise
    
    async def download(self, url: str, output_path: str = None, quality: str = "best", info_only: bool = False) -> Dict[str, Any]:
        """
        Download video from direct URL
        
        Args:
            url: Direct video URL
            output_path: Output file path
            quality: Quality setting (ignored for direct URLs)
            info_only: If True, only return video info without downloading
            
        Returns:
            Dictionary containing download result
        """
        logging.info(f"开始处理直接视频URL: {url}")
        
        try:
            # 获取视频信息
            video_info = await self._get_video_info(url)
            
            if info_only:
                return video_info
            
            # 处理输出路径 - 与其他下载器保持一致
            if output_path:
                if output_path.endswith('.mp4') or '.' in os.path.basename(output_path):
                    # output_path是完整文件路径
                    full_output_path = os.path.abspath(output_path)
                    # 确保目录存在
                    dir_path = os.path.dirname(full_output_path)
                    if dir_path:
                        os.makedirs(dir_path, exist_ok=True)
                else:
                    # output_path是目录路径
                    filename = self._get_filename_from_url(url)
                    full_output_path = os.path.abspath(os.path.join(output_path, filename))
                    os.makedirs(output_path, exist_ok=True)
            else:
                # 使用默认路径
                filename = self._get_filename_from_url(url)
                download_dir = Config.get_download_path()
                full_output_path = os.path.join(download_dir, filename)
                os.makedirs(download_dir, exist_ok=True)
            
            logging.info(f"开始下载到: {full_output_path}")
            
            # 下载视频
            download_result = await self._download_with_progress(url, full_output_path)
            
            # 合并结果
            result = {
                **video_info,
                **download_result,
                'quality': 'direct',  # 直接下载没有质量选择
                'source': 'direct_url'
            }
            
            logging.info(f"下载完成: {result['title']}")
            return result
            
        except Exception as e:
            error_msg = f"直接URL下载失败: {str(e)}"
            logging.error(error_msg)
            raise Exception(error_msg)

# 为了兼容性，创建一个同步版本的下载函数
async def download_direct_url(url: str, output_path: str = None, quality: str = "best", info_only: bool = False) -> Dict[str, Any]:
    """
    Standalone function to download from direct URL
    
    Args:
        url: Direct video URL
        output_path: Output file path
        quality: Quality setting (ignored)
        info_only: If True, only return info
        
    Returns:
        Dictionary containing result
    """
    async with DirectURLDownloader() as downloader:
        return await downloader.download(url, output_path, quality, info_only)

if __name__ == "__main__":
    # 测试直接URL下载
    import asyncio
    
    async def test():
        test_url = "https://sample-videos.com/zip/10/mp4/SampleVideo_1280x720_1mb.mp4"
        
        print("测试直接URL下载...")
        try:
            result = await download_direct_url(test_url, "test_direct.mp4")
            print(f"下载成功: {result}")
        except Exception as e:
            print(f"测试失败: {e}")
    
    asyncio.run(test())
