"""
Unified video downloader - main interface
"""

import asyncio
import argparse
import sys
from typing import Dict, Any, Optional, Callable, Union
import logging

from core.url_detector import URLDetector, VideoSource
from downloaders.bilibili_downloader import BilibiliDownloader
from downloaders.youtube_downloader import YouTubeDownloader
from config.config import Config

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DownloadError(Exception):
    """Custom exception for download errors"""
    pass

def parse_output_path(output_path: Optional[str], default_filename: str = None) -> tuple[str, Optional[str]]:
    """
    Parse output path to determine directory and filename

    Args:
        output_path: User provided output path
        default_filename: Default filename to use if not specified

    Returns:
        Tuple of (directory_path, filename_or_None)
        If filename_or_None is None, use auto-generated filename
    """
    import os

    if not output_path:
        # No path provided, use default directory
        return Config.get_download_path(), None

    # Check if path ends with a file extension
    _, ext = os.path.splitext(output_path)

    if ext:
        # Has extension, treat as full file path
        directory = os.path.dirname(output_path) or "."
        filename = os.path.basename(output_path)

        # Create directory if it doesn't exist
        os.makedirs(directory, exist_ok=True)

        return directory, filename
    else:
        # No extension, treat as directory
        os.makedirs(output_path, exist_ok=True)
        return output_path, None

async def download_video(
    url: str,
    output_path: Optional[str] = None,
    quality: str = "best",
    progress_callback: Optional[Callable[[Union[int, Dict[str, Any]], Union[int, None]], None]] = None
) -> Dict[str, Any]:
    """
    Unified video download function that automatically detects video source
    and routes to the appropriate downloader.

    Args:
        url: Video URL (bilibili or YouTube)
        output_path: Output file path (directory + filename) or directory path (optional)
                    If ends with file extension, treated as full file path
                    If directory, auto-generates filename
                    If None, uses default directory with auto-generated filename
        quality: Video quality preference ("best", "worst", "1080p", "720p", "480p", "360p")
        progress_callback: Progress callback function
                          - For bilibili: callback(downloaded_bytes: int, total_bytes: int)
                          - For YouTube: callback(progress_data: Dict[str, Any], None)
    
    Returns:
        Dictionary containing download result information:
        {
            'success': bool,
            'source': str,  # 'bilibili' or 'youtube'
            'title': str,
            'video_id': str,
            'filepath': str,
            'duration': int,  # in seconds
            'view_count': int,
            'uploader': str,  # YouTube only
            'error': str  # if success is False
        }
    
    Raises:
        DownloadError: If download fails
        ValueError: If URL is invalid or unsupported
    """
    
    try:
        # Detect video source
        logger.info(f"Analyzing URL: {url}")
        url_info = URLDetector.analyze_url(url)
        source = url_info['source']
        
        if source == VideoSource.UNKNOWN:
            raise ValueError(f"Unsupported video source: {url}")
            
        logger.info(f"Detected source: {source.value}")
        
        # Parse output path to determine directory and filename
        output_dir, custom_filename = parse_output_path(output_path)

        # Route to appropriate downloader
        result = None

        if source == VideoSource.BILIBILI:
            logger.info("Using Bilibili downloader")
            async with BilibiliDownloader() as downloader:
                result = await downloader.download(
                    url=url,
                    output_path=output_dir,
                    custom_filename=custom_filename,
                    quality=quality,
                    progress_callback=progress_callback
                )

        elif source == VideoSource.YOUTUBE:
            logger.info("Using YouTube downloader")
            downloader = YouTubeDownloader()
            result = await downloader.download(
                url=url,
                output_path=output_dir,
                custom_filename=custom_filename,
                quality=quality,
                progress_callback=progress_callback
            )
            
        if result:
            result['source'] = source.value
            logger.info(f"Download completed: {result.get('title', 'Unknown')}")
            return result
        else:
            raise DownloadError("Download failed - no result returned")
            
    except ValueError as e:
        logger.error(f"Invalid input: {str(e)}")
        return {
            'success': False,
            'source': 'unknown',
            'error': str(e)
        }
        
    except Exception as e:
        logger.error(f"Download failed: {str(e)}")
        return {
            'success': False,
            'source': url_info.get('source', VideoSource.UNKNOWN).value if 'url_info' in locals() else 'unknown',
            'error': str(e)
        }

def download_video_sync(
    url: str,
    output_path: Optional[str] = None,
    quality: str = "best",
    progress_callback: Optional[Callable[[Union[int, Dict[str, Any]], Union[int, None]], None]] = None
) -> Dict[str, Any]:
    """
    Synchronous wrapper for the async download_video function.
    
    Args:
        url: Video URL (bilibili or YouTube)
        output_path: Output directory path (optional)
        quality: Video quality preference
        progress_callback: Progress callback function
    
    Returns:
        Dictionary containing download result information
    """
    try:
        # Get or create event loop
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If we're already in an async context, we need to use a new loop
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(
                        asyncio.run,
                        download_video(url, output_path, quality, progress_callback)
                    )
                    return future.result()
            else:
                return loop.run_until_complete(
                    download_video(url, output_path, quality, progress_callback)
                )
        except RuntimeError:
            # No event loop exists, create a new one
            return asyncio.run(
                download_video(url, output_path, quality, progress_callback)
            )
            
    except Exception as e:
        logger.error(f"Sync download failed: {str(e)}")
        return {
            'success': False,
            'source': 'unknown',
            'error': str(e)
        }

# Convenience functions for specific sources
async def download_bilibili_video(
    url: str,
    output_path: Optional[str] = None,
    quality: str = "best",
    progress_callback: Optional[Callable[[int, int], None]] = None
) -> Dict[str, Any]:
    """Download bilibili video specifically"""
    url_info = URLDetector.analyze_url(url)
    if url_info['source'] != VideoSource.BILIBILI:
        raise ValueError("Not a bilibili URL")
        
    return await download_video(url, output_path, quality, progress_callback)

async def download_youtube_video(
    url: str,
    output_path: Optional[str] = None,
    quality: str = "best",
    progress_callback: Optional[Callable[[Dict[str, Any], None], None]] = None
) -> Dict[str, Any]:
    """Download YouTube video specifically"""
    url_info = URLDetector.analyze_url(url)
    if url_info['source'] != VideoSource.YOUTUBE:
        raise ValueError("Not a YouTube URL")

    return await download_video(url, output_path, quality, progress_callback)

def create_progress_callback():
    """Create a progress callback for command line usage"""
    def progress_callback(downloaded: Union[int, Dict[str, Any]], total: Union[int, None] = None):
        if isinstance(downloaded, int) and total is not None:
            # Bilibili progress (bytes)
            percent = (downloaded / total) * 100 if total > 0 else 0
            print(f"\rProgress: {percent:.1f}% ({downloaded:,}/{total:,} bytes)", end="", flush=True)
        elif isinstance(downloaded, dict):
            # YouTube progress (yt-dlp format)
            if 'progress' in downloaded:
                progress_info = downloaded['progress']
                if progress_info.get('status') == 'downloading':
                    percent = progress_info.get('_percent_str', 'N/A')
                    speed = progress_info.get('_speed_str', 'N/A')
                    print(f"\rProgress: {percent} Speed: {speed}", end="", flush=True)
    return progress_callback

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Unified video downloader for bilibili and YouTube",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s "https://www.bilibili.com/video/BV1xx411c7mD"
  %(prog)s "https://www.youtube.com/watch?v=dQw4w9WgXcQ" -o ./downloads/
  %(prog)s "https://youtu.be/dQw4w9WgXcQ" -o ./my_video.mp4
  %(prog)s "https://www.bilibili.com/video/BV1xx411c7mD" -o /path/to/custom_name.mp4 --quality 720p
        """
    )

    parser.add_argument(
        "url",
        help="Video URL to download (bilibili or YouTube)"
    )

    parser.add_argument(
        "-o", "--output",
        dest="output_path",
        help="Output file path (directory + filename) or directory path"
    )

    parser.add_argument(
        "-q", "--quality",
        choices=["best", "worst", "1080p", "720p", "480p", "360p"],
        default="best",
        help="Video quality preference (default: best)"
    )

    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable progress display"
    )

    parser.add_argument(
        "--info",
        action="store_true",
        help="Show video information without downloading"
    )

    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )

    return parser.parse_args()

async def main_cli():
    """Main CLI function"""
    args = parse_args()

    # Set up logging
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Analyze URL first
    logger.info(f"Analyzing URL: {args.url}")
    url_info = URLDetector.analyze_url(args.url)

    if url_info['source'] == VideoSource.UNKNOWN:
        print(f"Error: Unsupported video source: {args.url}", file=sys.stderr)
        sys.exit(1)

    print(f"Detected source: {url_info['source'].value}")

    # Show info only if requested
    if args.info:
        print(f"URL: {args.url}")
        print(f"Source: {url_info['source'].value}")
        print(f"Details: {url_info}")
        return

    # Set up progress callback
    progress_callback = None if args.no_progress else create_progress_callback()

    try:
        # Download video
        print(f"Starting download...")
        result = await download_video(
            url=args.url,
            output_path=args.output_path,
            quality=args.quality,
            progress_callback=progress_callback
        )

        if result['success']:
            print(f"\n✓ Download completed successfully!")
            print(f"Title: {result['title']}")
            print(f"File: {result['filepath']}")
            if result.get('duration'):
                print(f"Duration: {result['duration']} seconds")
            if result.get('view_count'):
                print(f"Views: {result['view_count']:,}")
        else:
            print(f"\n✗ Download failed: {result['error']}", file=sys.stderr)
            sys.exit(1)

    except KeyboardInterrupt:
        print("\n\nDownload cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Error: {str(e)}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main_cli())
