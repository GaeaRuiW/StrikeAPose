#!/usr/bin/env python3
"""
Command line interface for the unified video downloader
"""

import sys
import os
import asyncio
import argparse
import logging

from core.url_detector import URLDetector
from downloaders.youtube_downloader import YouTubeDownloader
from downloaders.bilibili_downloader import BilibiliDownloader
from downloaders.direct_url_downloader import DirectURLDownloader
from config.config import Config

class UnifiedDownloader:
    """Unified downloader that automatically detects video source"""

    def __init__(self):
        self.url_detector = URLDetector()

    async def download(self, url: str, output_path: str = None, quality: str = "best", info_only: bool = False) -> dict:
        """Download video from URL"""

        # Detect video source
        url_info = self.url_detector.analyze_url(url)
        source = url_info.get('source')

        # Handle VideoSource enum
        if hasattr(source, 'value'):
            source_str = source.value
        else:
            source_str = str(source)

        logging.info(f"Detected source: {source_str}")

        # Choose appropriate downloader
        if source_str == 'youtube':
            downloader = YouTubeDownloader()
        elif source_str == 'bilibili':
            downloader = BilibiliDownloader()
        elif source_str == 'direct_url':
            downloader = DirectURLDownloader()
        else:
            raise ValueError(f"Unsupported video source: {source_str}")

        # Download video
        if source_str == 'bilibili':
            async with downloader:
                return await downloader.download(url, output_path, quality, info_only)
        elif source_str == 'direct_url':
            async with downloader:
                return await downloader.download(url, output_path, quality, info_only)
        else:
            # YouTube downloader doesn't need async context
            return await downloader.download(url, output_path, quality, info_only)

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Unified Video Downloader - Download videos from YouTube, Bilibili, and direct URLs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s "https://www.youtube.com/watch?v=dQw4w9WgXcQ" -o video.mp4 -q 1080p
  %(prog)s "https://www.bilibili.com/video/BV1xx411c7mD" -q best
  %(prog)s "https://example.com/video.mp4" -o my_video.mp4
  %(prog)s "https://www.youtube.com/watch?v=dQw4w9WgXcQ" --info

Supported sources:
  - YouTube (youtube.com, youtu.be)
  - Bilibili (bilibili.com, b23.tv)
  - Direct video URLs (.mp4, .avi, .mkv, .mov, .webm, etc.)
        """
    )

    parser.add_argument(
        "url",
        help="Video URL to download"
    )

    parser.add_argument(
        "-o", "--output",
        help="Output file path (default: auto-generated)"
    )

    parser.add_argument(
        "-q", "--quality",
        choices=["best", "worst", "1080p", "720p", "480p", "360p"],
        default="best",
        help="Video quality (default: best)"
    )

    parser.add_argument(
        "--info",
        action="store_true",
        help="Show video info only, don't download"
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
        logging.basicConfig(level=logging.DEBUG, format='%(levelname)s:%(name)s:%(message)s')
    else:
        logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')

    try:
        # Create downloader
        downloader = UnifiedDownloader()

        # Analyze URL
        logging.info(f"Analyzing URL: {args.url}")

        # Download or show info
        result = await downloader.download(
            url=args.url,
            output_path=args.output,
            quality=args.quality,
            info_only=args.info
        )

        if args.info:
            print(f"\n📋 Video Information:")
            print(f"Title: {result.get('title', 'Unknown')}")
            print(f"Duration: {result.get('duration', 0)} seconds")
            print(f"Views: {result.get('views', 'Unknown')}")
            if result.get('description'):
                print(f"Description: {result['description'][:100]}...")
        else:
            print(f"\n✓ Download completed successfully!")
            print(f"Title: {result.get('title', 'Unknown')}")
            print(f"File: {result.get('file_path', 'Unknown')}")
            print(f"Duration: {result.get('duration', 0)} seconds")
            print(f"Views: {result.get('views', 'Unknown')}")

    except KeyboardInterrupt:
        print("\n❌ Download cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main_cli())
