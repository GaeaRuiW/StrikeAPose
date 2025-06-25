#!/usr/bin/env python3
# encoding:utf-8
"""
PushPlus推送通知服务
用于Bilibili Cookie过期提醒
"""

import aiohttp
import json
import asyncio
import aiofiles
from datetime import datetime
from typing import Optional

class PushPlusNotifier:
    """PushPlus推送通知器"""
    
    def __init__(self, token: str = None):
        """
        初始化推送器
        
        Args:
            token: PushPlus的token，如果不提供则从配置文件读取
        """
        self.token = token or self.get_token_from_config()
        self.url = 'http://www.pushplus.plus/send'
        
    def get_token_from_config(self) -> Optional[str]:
        """从配置文件获取token"""
        try:
            from config import Config
            return getattr(Config, 'PUSHPLUS_TOKEN', None)
        except:
            # 如果没有配置，使用测试token
            return '2a6b548aa1ba47c6ad2d2de0c3861167'
    
    async def send_notification(self, title: str, content: str, template: str = 'html') -> bool:
        """
        发送推送通知

        Args:
            title: 通知标题
            content: 通知内容
            template: 模板类型 ('html', 'txt', 'json', 'markdown')

        Returns:
            bool: 发送是否成功
        """
        if not self.token:
            print("❌ PushPlus token未配置，无法发送通知")
            return False

        data = {
            "token": self.token,
            "title": title,
            "content": content,
            "template": template
        }

        try:
            headers = {'Content-Type': 'application/json'}

            # 尝试多个URL
            urls = [
                'http://www.pushplus.plus/send',
                'https://www.pushplus.plus/send',
                'http://pushplus.plus/send',
                'https://pushplus.plus/send'
            ]

            async with aiohttp.ClientSession() as session:
                for url in urls:
                    try:
                        async with session.post(url, json=data, headers=headers, timeout=10) as response:
                            if response.status == 200:
                                result = await response.json()
                                if result.get('code') == 200:
                                    print(f"✅ 推送通知发送成功: {title}")
                                    return True
                                else:
                                    print(f"❌ 推送失败: {result.get('msg', '未知错误')}")
                                    continue
                            else:
                                print(f"❌ 推送请求失败: HTTP {response.status}")
                                continue

                    except aiohttp.ClientError as e:
                        print(f"⚠️  尝试URL {url} 失败: {e}")
                        continue
                    except asyncio.TimeoutError:
                        print(f"⚠️  URL {url} 请求超时")
                        continue

            # 如果所有URL都失败，保存到本地文件
            await self.save_notification_to_file(title, content)
            print("📝 推送服务暂时不可用，通知已保存到本地文件")
            return False

        except Exception as e:
            print(f"❌ 发送推送通知时出错: {e}")
            await self.save_notification_to_file(title, content)
            return False

    async def save_notification_to_file(self, title: str, content: str):
        """将通知保存到本地文件"""
        try:
            import os
            if not os.path.exists('notifications'):
                os.makedirs('notifications')

            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'notifications/notification_{timestamp}.html'

            html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>{title}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        h1 {{ color: #333; }}
        .timestamp {{ color: #666; font-size: 0.9em; }}
    </style>
</head>
<body>
    <h1>{title}</h1>
    <div class="timestamp">生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</div>
    <hr>
    {content}
</body>
</html>
"""

            async with aiofiles.open(filename, 'w', encoding='utf-8') as f:
                await f.write(html_content)

            print(f"📝 通知已保存到: {filename}")

        except Exception as e:
            print(f"❌ 保存通知文件失败: {e}")
    
    async def send_cookie_expiry_warning(self, days_left: float, cookie_name: str = 'Bilibili Cookie') -> bool:
        """
        发送Cookie过期警告
        
        Args:
            days_left: 剩余天数
            cookie_name: Cookie名称
            
        Returns:
            bool: 发送是否成功
        """
        if days_left <= 0:
            # Cookie已过期
            title = "🚨 Bilibili Cookie已过期"
            content = f"""
<h2>🚨 紧急通知：Cookie已过期</h2>
<p><strong>状态：</strong> ❌ 已过期</p>
<p><strong>影响：</strong> 无法下载高质量视频，已降级到480p</p>
<p><strong>建议：</strong> 立即更新Cookie</p>

<h3>📋 更新步骤：</h3>
<ol>
<li>在浏览器中登录 <a href="https://www.bilibili.com">bilibili.com</a></li>
<li>按F12打开开发者工具 → Network标签</li>
<li>刷新页面，找到API请求</li>
<li>复制Cookie值</li>
<li>运行: <code>python3 cookie_auto_updater.py update</code></li>
</ol>

<p><strong>检查时间：</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
"""
        elif days_left < 1:
            # 24小时内过期
            hours_left = days_left * 24
            title = f"🚨 Bilibili Cookie将在{hours_left:.1f}小时内过期"
            content = f"""
<h2>🚨 紧急警告：Cookie即将过期</h2>
<p><strong>剩余时间：</strong> ⏰ {hours_left:.1f}小时</p>
<p><strong>状态：</strong> 🚨 紧急</p>
<p><strong>建议：</strong> 立即更新Cookie</p>

<h3>📋 快速更新：</h3>
<p>运行命令: <code>python3 cookie_auto_updater.py update</code></p>

<h3>📱 或者手动更新：</h3>
<ol>
<li>登录 <a href="https://www.bilibili.com">bilibili.com</a></li>
<li>F12 → Network → 刷新页面</li>
<li>复制Cookie → 更新配置文件</li>
</ol>

<p><strong>检查时间：</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
"""
        elif days_left <= 3:
            # 3天内过期
            title = f"⚠️ Bilibili Cookie将在{days_left:.1f}天内过期"
            content = f"""
<h2>⚠️ Cookie过期提醒</h2>
<p><strong>剩余时间：</strong> ⏰ {days_left:.1f}天</p>
<p><strong>状态：</strong> ⚠️ 需要关注</p>
<p><strong>建议：</strong> 准备更新Cookie</p>

<h3>📋 更新方法：</h3>
<p>运行命令: <code>python3 cookie_auto_updater.py update</code></p>

<h3>📊 当前状态：</h3>
<ul>
<li>✅ 高质量下载仍然可用</li>
<li>⚠️ 建议在过期前更新</li>
<li>📱 可以继续正常使用</li>
</ul>

<p><strong>检查时间：</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
"""
        elif days_left <= 7:
            # 7天内过期
            title = f"📅 Bilibili Cookie将在{days_left:.1f}天内过期"
            content = f"""
<h2>📅 Cookie过期提醒</h2>
<p><strong>剩余时间：</strong> ⏰ {days_left:.1f}天</p>
<p><strong>状态：</strong> 📅 提前提醒</p>
<p><strong>建议：</strong> 可以开始准备更新</p>

<h3>📊 当前状态：</h3>
<ul>
<li>✅ 所有功能正常</li>
<li>📅 建议一周内更新</li>
<li>🔄 系统会继续监控</li>
</ul>

<h3>📋 更新方法：</h3>
<p>当需要时运行: <code>python3 cookie_auto_updater.py update</code></p>

<p><strong>检查时间：</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
"""
        else:
            # 不需要发送通知
            return True

        return await self.send_notification(title, content, 'html')
    
    async def send_cookie_update_success(self, old_days: float, new_days: float) -> bool:
        """
        发送Cookie更新成功通知
        
        Args:
            old_days: 旧Cookie剩余天数
            new_days: 新Cookie剩余天数
            
        Returns:
            bool: 发送是否成功
        """
        title = "✅ Bilibili Cookie更新成功"
        content = f"""
<h2>✅ Cookie更新完成</h2>
<p><strong>更新时间：</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>

<h3>📊 更新前后对比：</h3>
<table border="1" style="border-collapse: collapse;">
<tr><th>项目</th><th>更新前</th><th>更新后</th></tr>
<tr><td>剩余时间</td><td>{old_days:.1f}天</td><td><strong>{new_days:.1f}天</strong></td></tr>
<tr><td>状态</td><td>⚠️ 即将过期</td><td><strong>✅ 正常</strong></td></tr>
</table>

<h3>🎉 更新效果：</h3>
<ul>
<li>✅ 高质量下载功能已恢复</li>
<li>✅ 可以正常下载1080p视频</li>
<li>✅ 系统将继续自动监控</li>
</ul>

<p><strong>下次检查：</strong> 系统会在适当时间自动检查Cookie状态</p>
"""
        return await self.send_notification(title, content, 'html')
    
    async def test_notification(self) -> bool:
        """测试推送功能"""
        title = "🧪 Bilibili下载器推送测试"
        content = f"""
<h2>🧪 推送功能测试</h2>
<p><strong>测试时间：</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
<p><strong>状态：</strong> ✅ 推送服务正常工作</p>

<h3>📋 功能说明：</h3>
<ul>
<li>🔍 自动监控Cookie状态</li>
<li>⚠️ 过期前自动提醒</li>
<li>📱 支持手机推送通知</li>
<li>🔄 智能更新引导</li>
</ul>

<p>如果您收到这条消息，说明推送功能已正确配置！</p>
"""
        return await self.send_notification(title, content, 'html')

async def main():
    """测试推送功能"""
    import sys

    # 使用测试token
    notifier = PushPlusNotifier('2a6b548aa1ba47c6ad2d2de0c3861167')

    if len(sys.argv) > 1:
        command = sys.argv[1]
        if command == 'test':
            # 测试推送
            success = await notifier.test_notification()
            if success:
                print("✅ 测试推送发送成功！请检查您的手机或微信")
            else:
                print("❌ 测试推送发送失败")
        elif command == 'warning':
            # 测试警告推送
            days = float(sys.argv[2]) if len(sys.argv) > 2 else 2.5
            success = await notifier.send_cookie_expiry_warning(days)
            if success:
                print(f"✅ 警告推送发送成功！(剩余{days}天)")
            else:
                print("❌ 警告推送发送失败")
        else:
            print("用法: python3 pushplus_notifier.py [test|warning [days]]")
    else:
        # 默认测试
        await notifier.test_notification()

if __name__ == "__main__":
    asyncio.run(main())
