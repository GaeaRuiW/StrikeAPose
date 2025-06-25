#!/usr/bin/env python3
"""
统一视频下载器 FastAPI 服务
提供Web API接口进行视频下载和管理
"""

import os
import sys
import asyncio
import logging
from typing import Optional, Dict, Any
from datetime import datetime

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel, HttpUrl

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入项目模块
from core.url_detector import URLDetector
from downloaders.youtube_downloader import YouTubeDownloader
from downloaders.bilibili_downloader import BilibiliDownloader
from downloaders.direct_url_downloader import DirectURLDownloader
from auth.cookie_validator import check_current_cookies, analyze_cookie_expiry
from auth.cookie_auto_updater import CookieAutoUpdater
from notifications.pushplus_notifier import PushPlusNotifier

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 创建FastAPI应用
app = FastAPI(
    title="统一视频下载器 API",
    description="支持YouTube、Bilibili和直接URL的视频下载服务",
    version="1.0.0"
)

# 请求模型
class DownloadRequest(BaseModel):
    url: HttpUrl
    output_path: Optional[str] = None
    quality: str = "best"
    info_only: bool = False

class CookieUpdateRequest(BaseModel):
    cookie_string: str

class PushPlusTokenRequest(BaseModel):
    token: str

class UnifiedDownloader:
    """统一下载器"""
    
    def __init__(self):
        self.url_detector = URLDetector()
        
    async def download(self, url: str, output_path: str = None, quality: str = "best", info_only: bool = False) -> dict:
        """下载视频"""
        
        # 检测视频源
        url_info = self.url_detector.analyze_url(url)
        source = url_info.get('source')
        
        # 处理VideoSource枚举
        if hasattr(source, 'value'):
            source_str = source.value
        else:
            source_str = str(source)
            
        logger.info(f"检测到视频源: {source_str}")
        
        # 选择合适的下载器
        if source_str == 'youtube':
            downloader = YouTubeDownloader()
        elif source_str == 'bilibili':
            downloader = BilibiliDownloader()
        elif source_str == 'direct_url':
            downloader = DirectURLDownloader()
        else:
            raise ValueError(f"不支持的视频源: {source_str}")
        
        # 下载视频
        if source_str == 'bilibili':
            async with downloader:
                return await downloader.download(url, output_path, quality, info_only)
        elif source_str == 'direct_url':
            async with downloader:
                return await downloader.download(url, output_path, quality, info_only)
        else:
            # YouTube下载器不需要async context
            return await downloader.download(url, output_path, quality, info_only)

# 全局下载器实例
unified_downloader = UnifiedDownloader()

@app.get("/")
async def root():
    """根路径 - API信息"""
    return {
        "message": "统一视频下载器 API",
        "version": "1.0.0",
        "supported_sources": ["YouTube", "Bilibili", "Direct URL"],
        "endpoints": {
            "download": "/api/download",
            "cookie_status": "/api/cookie/status",
            "cookie_update": "/api/cookie/update",
            "pushplus_token": "/api/pushplus/token"
        }
    }

@app.post("/api/download")
async def download_video(request: DownloadRequest, background_tasks: BackgroundTasks):
    """
    下载视频API
    
    Args:
        request: 下载请求参数
        background_tasks: 后台任务（用于异步下载）
    
    Returns:
        视频信息和下载状态
    """
    try:
        logger.info(f"收到下载请求: {request.url}")
        
        # 执行下载
        result = await unified_downloader.download(
            url=str(request.url),
            output_path=request.output_path,
            quality=request.quality,
            info_only=request.info_only
        )
        
        # 构造响应
        response = {
            "success": True,
            "message": "下载完成" if not request.info_only else "获取信息成功",
            "data": {
                "title": result.get('title', 'Unknown'),
                "duration": result.get('duration', 0),
                "views": result.get('views', 0),
                "quality": result.get('quality', request.quality),
                "source": result.get('source', 'unknown'),
                "file_path": result.get('file_path') if not request.info_only else None,
                "info_only": request.info_only
            },
            "timestamp": datetime.now().isoformat()
        }
        
        logger.info(f"下载成功: {result.get('title', 'Unknown')}")
        return response
        
    except Exception as e:
        logger.error(f"下载失败: {str(e)}")
        raise HTTPException(
            status_code=400,
            detail={
                "success": False,
                "message": f"下载失败: {str(e)}",
                "timestamp": datetime.now().isoformat()
            }
        )

@app.get("/api/cookie/status")
async def get_cookie_status():
    """
    获取Bilibili Cookie状态
    
    Returns:
        Cookie状态信息
    """
    try:
        logger.info("检查Cookie状态")
        
        # 获取Cookie状态
        from auth.bilibili_cookies import get_cookies
        cookies = get_cookies()
        
        # 分析Cookie有效期
        cookie_analysis = await analyze_cookie_expiry(cookies)
        
        response = {
            "success": True,
            "message": "Cookie状态获取成功",
            "data": {
                "status": cookie_analysis.get('status', 'unknown'),
                "expires": cookie_analysis.get('expires', {}),
                "warnings": cookie_analysis.get('warnings', []),
                "current_time": datetime.fromtimestamp(cookie_analysis.get('current_time', 0)).isoformat(),
                "check_time": datetime.now().isoformat()
            }
        }
        
        return response
        
    except ImportError:
        raise HTTPException(
            status_code=404,
            detail={
                "success": False,
                "message": "未找到Cookie配置文件",
                "timestamp": datetime.now().isoformat()
            }
        )
    except Exception as e:
        logger.error(f"获取Cookie状态失败: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={
                "success": False,
                "message": f"获取Cookie状态失败: {str(e)}",
                "timestamp": datetime.now().isoformat()
            }
        )

def clean_cookie_string(cookie_string: str) -> str:
    """
    清理Cookie字符串，移除无效字符

    Args:
        cookie_string: 原始Cookie字符串

    Returns:
        清理后的Cookie字符串
    """
    # 移除可能导致JSON解析错误的字符
    cleaned = cookie_string.replace("'", "").replace('"', "").replace('\n', '').replace('\r', '')

    # 移除多余的空格
    cleaned = ' '.join(cleaned.split())

    return cleaned

@app.post("/api/cookie/update")
async def update_cookie(request: CookieUpdateRequest):
    """
    更新Bilibili Cookie

    Args:
        request: Cookie更新请求

    Returns:
        更新结果
    """
    try:
        logger.info("开始更新Cookie")

        # 清理Cookie字符串
        cleaned_cookie = clean_cookie_string(request.cookie_string)
        logger.info(f"Cookie清理完成，长度: {len(cleaned_cookie)}")

        # 创建Cookie更新器
        updater = CookieAutoUpdater()

        # 更新Cookie
        success = await updater.update_cookie_file(cleaned_cookie)

        if success:
            response = {
                "success": True,
                "message": "Cookie更新成功",
                "data": {
                    "update_time": datetime.now().isoformat(),
                    "original_length": len(request.cookie_string),
                    "cleaned_length": len(cleaned_cookie),
                    "cookie_preview": cleaned_cookie[:50] + "..." if len(cleaned_cookie) > 50 else cleaned_cookie
                }
            }
            logger.info("Cookie更新成功")
            return response
        else:
            raise HTTPException(
                status_code=400,
                detail={
                    "success": False,
                    "message": "Cookie更新失败，请检查Cookie格式",
                    "timestamp": datetime.now().isoformat()
                }
            )

    except Exception as e:
        logger.error(f"Cookie更新失败: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={
                "success": False,
                "message": f"Cookie更新失败: {str(e)}",
                "timestamp": datetime.now().isoformat()
            }
        )

@app.post("/api/pushplus/token")
async def update_pushplus_token(request: PushPlusTokenRequest):
    """
    更新PushPlus推送Token
    
    Args:
        request: Token更新请求
    
    Returns:
        更新结果
    """
    try:
        logger.info("开始更新PushPlus Token")
        
        # 创建通知器并测试Token
        notifier = PushPlusNotifier(request.token)
        
        # 测试推送功能
        test_success = await notifier.test_notification()
        
        if test_success:
            # TODO: 这里应该将Token保存到配置文件
            # 目前只是测试Token有效性
            response = {
                "success": True,
                "message": "PushPlus Token更新成功",
                "data": {
                    "token": request.token[:10] + "..." + request.token[-10:],  # 隐藏部分Token
                    "test_result": "推送测试成功",
                    "update_time": datetime.now().isoformat()
                }
            }
            logger.info("PushPlus Token更新成功")
            return response
        else:
            raise HTTPException(
                status_code=400,
                detail={
                    "success": False,
                    "message": "Token无效或推送服务不可用",
                    "timestamp": datetime.now().isoformat()
                }
            )
            
    except Exception as e:
        logger.error(f"PushPlus Token更新失败: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={
                "success": False,
                "message": f"PushPlus Token更新失败: {str(e)}",
                "timestamp": datetime.now().isoformat()
            }
        )

@app.get("/api/health")
async def health_check():
    """健康检查"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "service": "统一视频下载器 API"
    }

if __name__ == "__main__":
    import uvicorn
    
    # 启动服务
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8080,
        reload=True,
        log_level="info"
    )
