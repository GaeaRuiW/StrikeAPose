#!/usr/bin/env python3
"""
Bilibili认证下载示例
演示如何通过Cookie获取高质量流
"""

import asyncio
import aiohttp
from urllib.parse import urlencode

class BilibiliAuthDownloader:
    """带认证的Bilibili下载器示例"""
    
    def __init__(self, cookies=None):
        """
        初始化下载器
        
        Args:
            cookies: 登录后的Cookie字典，例如：
                {
                    'SESSDATA': 'your_sessdata_here',
                    'bili_jct': 'your_bili_jct_here', 
                    'DedeUserID': 'your_userid_here'
                }
        """
        self.cookies = cookies or {}
        self.session = None
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
            
    async def get_video_info(self, bvid):
        """获取视频信息"""
        url = f"https://api.bilibili.com/x/web-interface/view?bvid={bvid}"
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Referer': 'https://www.bilibili.com/'
        }
        
        if self.cookies:
            cookie_str = '; '.join([f'{k}={v}' for k, v in self.cookies.items()])
            headers['Cookie'] = cookie_str
            
        async with self.session.get(url, headers=headers) as response:
            response.raise_for_status()
            data = await response.json()
            if data.get('code') != 0:
                raise Exception(f"API Error: {data.get('message')}")
            return data['data']
            
    async def get_stream_urls(self, bvid, cid, quality=80):
        """获取视频流URL"""
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
        
        url = f"https://api.bilibili.com/x/player/wbi/playurl?{urlencode(params)}"
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Referer': 'https://www.bilibili.com/',
            'Origin': 'https://www.bilibili.com'
        }
        
        if self.cookies:
            cookie_str = '; '.join([f'{k}={v}' for k, v in self.cookies.items()])
            headers['Cookie'] = cookie_str
            
        async with self.session.get(url, headers=headers) as response:
            response.raise_for_status()
            data = await response.json()
            if data.get('code') != 0:
                raise Exception(f"Stream API Error: {data.get('message')}")
            return data['data']
            
    async def test_quality_access(self, bvid):
        """测试质量访问能力"""
        print(f"测试视频: {bvid}")
        print(f"认证状态: {'已登录' if self.cookies else '未登录'}")
        print()
        
        try:
            # 获取视频信息
            video_info = await self.get_video_info(bvid)
            print(f"视频标题: {video_info['title']}")
            
            cid = video_info['pages'][0]['cid']
            print(f"CID: {cid}")
            print()
            
            # 测试不同质量
            qualities = [
                (120, '4K'),
                (80, '1080P'),
                (64, '720P'),
                (32, '480P'),
                (16, '360P')
            ]
            
            for qn, desc in qualities:
                try:
                    stream_data = await self.get_stream_urls(bvid, cid, qn)
                    
                    if 'dash' in stream_data and stream_data['dash'].get('video'):
                        videos = stream_data['dash']['video']
                        max_height = max(v.get('height', 0) for v in videos)
                        resolutions = set(f"{v.get('width')}x{v.get('height')}" for v in videos)
                        
                        print(f"{desc:6} (qn={qn:3}): 最高{max_height}p, 可用分辨率: {sorted(resolutions)}")
                    else:
                        print(f"{desc:6} (qn={qn:3}): 无可用流")
                        
                except Exception as e:
                    print(f"{desc:6} (qn={qn:3}): 错误 - {e}")
                    
        except Exception as e:
            print(f"测试失败: {e}")

async def main():
    """主函数 - 演示认证和非认证的区别"""
    
    bvid = "BV1QEjYzTEYZ"  # 测试视频
    
    print("=" * 60)
    print("Bilibili认证下载测试")
    print("=" * 60)
    
    # 测试1：未登录状态
    print("\n1. 未登录状态测试:")
    print("-" * 30)
    async with BilibiliAuthDownloader() as downloader:
        await downloader.test_quality_access(bvid)
    
    # 测试2：模拟登录状态（需要用户提供真实Cookie）
    print("\n2. 登录状态测试:")
    print("-" * 30)
    print("注意：需要真实的Cookie才能测试登录状态")
    print("如何获取Cookie：")
    print("1. 在浏览器中登录bilibili.com")
    print("2. 打开开发者工具 (F12)")
    print("3. 在Network标签中找到API请求")
    print("4. 复制Cookie值")
    print()
    
    # 示例Cookie格式（用户需要替换为真实值）
    example_cookies = {
        # 'SESSDATA': 'your_sessdata_here',
        # 'bili_jct': 'your_bili_jct_here',
        # 'DedeUserID': 'your_userid_here'
    }
    
    if any(example_cookies.values()):
        async with BilibiliAuthDownloader(example_cookies) as downloader:
            await downloader.test_quality_access(bvid)
    else:
        print("未提供Cookie，跳过登录状态测试")
    
    print("\n" + "=" * 60)
    print("测试完成")
    print("=" * 60)

if __name__ == "__main__":
    asyncio.run(main())
