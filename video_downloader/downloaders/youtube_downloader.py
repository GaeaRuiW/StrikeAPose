"""
YouTube video downloader implementation using yt-dlp
"""

import asyncio
import os
import subprocess
import json
from typing import Dict, Any, Optional, Callable
import tempfile
import shutil

from config.config import Config
from core.url_detector import URLDetector, VideoSource

class YouTubeDownloader:
    """Async YouTube video downloader using yt-dlp"""
    
    def __init__(self):
        self.yt_dlp_path = self._find_yt_dlp()
        
    def _find_yt_dlp(self) -> str:
        """Find yt-dlp executable"""
        # Try common locations
        possible_paths = ['yt-dlp', 'yt-dlp.exe']
        
        for path in possible_paths:
            if shutil.which(path):
                return path
                
        raise RuntimeError("yt-dlp not found. Please install yt-dlp: pip install yt-dlp")
        
    async def _run_command(self, cmd: list) -> tuple[int, str, str]:
        """Run command asynchronously"""
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        stdout, stderr = await process.communicate()
        return process.returncode, stdout.decode(), stderr.decode()
        
    async def _get_video_info(self, url: str) -> Dict[str, Any]:
        """Get video information using yt-dlp"""
        cmd = [
            self.yt_dlp_path,
            '--dump-json',
            '--no-download',
            url
        ]
        
        returncode, stdout, stderr = await self._run_command(cmd)
        
        if returncode != 0:
            raise Exception(f"Failed to get video info: {stderr}")
            
        try:
            return json.loads(stdout)
        except json.JSONDecodeError as e:
            raise Exception(f"Failed to parse video info: {str(e)}")
            
    async def _download_with_progress(self, url: str, output_path: str, filename: str,
                                    quality: str = "best",
                                    progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None) -> str:
        """Download video with progress tracking"""
        
        # Create progress hook file
        progress_file = None
        if progress_callback:
            progress_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json')
            progress_file.close()
            
        # Build yt-dlp command
        cmd = [
            self.yt_dlp_path,
            '--format', self._get_format_selector(quality),
            '--output', os.path.join(output_path, filename),
            '--merge-output-format', 'mp4',
            '--no-playlist'
        ]
        
        if progress_file:
            cmd.extend(['--progress-template', f'{{\"progress\": %(progress)j}}'])
            
        cmd.append(url)
        
        # Run download
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        # Monitor progress if callback provided
        if progress_callback and progress_file:
            asyncio.create_task(self._monitor_progress(progress_file.name, progress_callback))
            
        stdout, stderr = await process.communicate()
        
        # Clean up progress file
        if progress_file:
            try:
                os.unlink(progress_file.name)
            except:
                pass
                
        if process.returncode != 0:
            raise Exception(f"Download failed: {stderr.decode()}")
            
        return stdout.decode()
        
    async def _monitor_progress(self, progress_file: str, callback: Callable[[Dict[str, Any]], None]):
        """Monitor download progress"""
        last_size = 0
        
        while True:
            try:
                if os.path.exists(progress_file):
                    current_size = os.path.getsize(progress_file)
                    if current_size > last_size:
                        with open(progress_file, 'r') as f:
                            f.seek(last_size)
                            new_content = f.read()
                            last_size = current_size
                            
                        # Parse progress updates
                        for line in new_content.strip().split('\n'):
                            if line.strip():
                                try:
                                    progress_data = json.loads(line)
                                    callback(progress_data)
                                except json.JSONDecodeError:
                                    continue
                                    
                await asyncio.sleep(0.5)
                
            except Exception:
                break
                
    def _get_format_selector(self, quality: str) -> str:
        """Get yt-dlp format selector based on quality preference"""
        quality_map = {
            # 优先选择最高质量的视频+音频组合
            'best': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/bestvideo+bestaudio/best[ext=mp4]/best',
            'worst': 'worstvideo[ext=mp4]+worstaudio[ext=m4a]/worstvideo+worstaudio/worst[ext=mp4]/worst',
            '1080p': 'bestvideo[height<=1080][ext=mp4]+bestaudio[ext=m4a]/bestvideo[height<=1080]+bestaudio/best[height<=1080][ext=mp4]/best[height<=1080]',
            '720p': 'bestvideo[height<=720][ext=mp4]+bestaudio[ext=m4a]/bestvideo[height<=720]+bestaudio/best[height<=720][ext=mp4]/best[height<=720]',
            '480p': 'bestvideo[height<=480][ext=mp4]+bestaudio[ext=m4a]/bestvideo[height<=480]+bestaudio/best[height<=480][ext=mp4]/best[height<=480]',
            '360p': 'bestvideo[height<=360][ext=mp4]+bestaudio[ext=m4a]/bestvideo[height<=360]+bestaudio/best[height<=360][ext=mp4]/best[height<=360]'
        }

        return quality_map.get(quality, 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/bestvideo+bestaudio/best[ext=mp4]/best')
        
    def _get_legal_filename(self, filename: str) -> str:
        """Get legal filename by removing invalid characters"""
        import re
        return re.sub(r'[<>:"/\\|?*]', '_', filename)
        
    async def download(self, url: str, output_path: str = None, quality: str = "best", info_only: bool = False) -> Dict[str, Any]:
        """
        Download YouTube video

        Args:
            url: YouTube video URL
            output_path: Output file path (can be directory or full file path)
            quality: Video quality preference
            info_only: If True, only return video info without downloading

        Returns:
            Dictionary with download result information
        """
        # Analyze URL
        url_info = URLDetector.analyze_url(url)
        if url_info['source'] != VideoSource.YOUTUBE:
            raise ValueError("Not a valid YouTube URL")
            
        # Get video information
        video_info = await self._get_video_info(url)

        # Extract common info
        title = self._get_legal_filename(video_info.get('title', 'Unknown'))
        video_id = video_info.get('id', 'unknown')

        # If info_only, return video info without downloading
        if info_only:
            return {
                'title': video_info.get('title', 'Unknown'),
                'duration': video_info.get('duration', 0),
                'views': video_info.get('view_count', 0),
                'description': video_info.get('description', ''),
                'uploader': video_info.get('uploader', ''),
                'upload_date': video_info.get('upload_date', '')
            }

        # Handle output path - support both directory and full file path
        if output_path:
            if output_path.endswith('.mp4') or '.' in os.path.basename(output_path):
                # output_path is a full file path
                filepath = os.path.abspath(output_path)
                download_dir = os.path.dirname(filepath)
                base_name = os.path.splitext(os.path.basename(filepath))[0]
                filename = f"{base_name}.%(ext)s"
                # Ensure directory exists
                if download_dir:
                    os.makedirs(download_dir, exist_ok=True)
            else:
                # output_path is a directory
                download_dir = output_path
                filename = f"{title}_{video_id}.%(ext)s"
                filepath = os.path.join(download_dir, f"{title}_{video_id}.mp4")
                os.makedirs(download_dir, exist_ok=True)
        else:
            # Use default download path
            download_dir = Config.get_download_path()
            filename = f"{title}_{video_id}.%(ext)s"
            filepath = os.path.join(download_dir, f"{title}_{video_id}.mp4")
            os.makedirs(download_dir, exist_ok=True)

        # Download the video
        await self._download_with_progress(url, download_dir, filename, quality)
        
        return {
            'success': True,
            'title': video_info.get('title', 'Unknown'),
            'video_id': video_id,
            'file_path': filepath,
            'duration': video_info.get('duration', 0),
            'views': video_info.get('view_count', 0),
            'uploader': video_info.get('uploader', 'Unknown'),
            'quality': quality,
            'source': 'youtube'
        }
