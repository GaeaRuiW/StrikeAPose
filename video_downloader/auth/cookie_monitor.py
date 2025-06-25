#!/usr/bin/env python3
"""
Bilibili Cookie智能监控系统
自动检测Cookie状态并在需要时提醒用户更新
"""

import os
import json
import time
import asyncio
from datetime import datetime, timedelta

class CookieMonitor:
    """Cookie监控器"""
    
    def __init__(self):
        self.status_file = "cookie_status.json"
        self.last_check_file = "last_cookie_check.json"
        self.last_notification_file = "last_notification.json"
        
    async def load_status(self):
        """加载Cookie状态"""
        if os.path.exists(self.status_file):
            try:
                with open(self.status_file, 'r') as f:
                    return json.load(f)
            except:
                pass
        return {}
    
    async def save_status(self, status):
        """保存Cookie状态"""
        try:
            with open(self.status_file, 'w') as f:
                json.dump(status, f, indent=2, default=str)
        except Exception as e:
            print(f"保存状态失败: {e}")
    
    def should_check_cookies(self):
        """判断是否需要检查Cookie"""
        if not os.path.exists(self.last_check_file):
            return True
            
        try:
            with open(self.last_check_file, 'r') as f:
                data = json.load(f)
                last_check = datetime.fromisoformat(data['last_check'])
                
                # 根据上次状态决定检查频率
                status = data.get('status', 'unknown')
                if status == 'critical':
                    interval = timedelta(hours=6)  # 危险状态：6小时检查一次
                elif status == 'warning':
                    interval = timedelta(hours=12)  # 警告状态：12小时检查一次
                else:
                    interval = timedelta(days=1)   # 正常状态：每天检查一次
                
                return datetime.now() - last_check > interval
        except:
            return True
    
    def update_last_check(self, status):
        """更新最后检查时间"""
        data = {
            'last_check': datetime.now().isoformat(),
            'status': status
        }
        try:
            with open(self.last_check_file, 'w') as f:
                json.dump(data, f)
        except Exception as e:
            print(f"更新检查时间失败: {e}")
    
    async def check_cookie_validity(self):
        """检查Cookie有效性"""
        try:
            from auth.cookie_validator import analyze_cookie_expiry
            from auth.bilibili_cookies import get_cookies
            
            cookies = get_cookies()
            results = analyze_cookie_expiry(cookies)
            
            # 保存状态
            status_data = {
                'check_time': datetime.now().isoformat(),
                'status': results['status'],
                'warnings': results['warnings'],
                'expires': {}
            }
            
            # 转换时间戳为可序列化格式
            for name, info in results['expires'].items():
                status_data['expires'][name] = {
                    'timestamp': info['timestamp'],
                    'datetime': info['datetime'].isoformat(),
                    'days_left': info['days_left']
                }
            
            self.save_status(status_data)
            self.update_last_check(results['status'])
            
            return results
            
        except Exception as e:
            print(f"检查Cookie有效性失败: {e}")
            return None
    
    def should_send_notification(self, status, days_left):
        """判断是否应该发送推送通知"""
        if not os.path.exists(self.last_notification_file):
            return True

        try:
            with open(self.last_notification_file, 'r') as f:
                data = json.load(f)
                last_notification = datetime.fromisoformat(data['last_notification'])
                last_status = data.get('status', 'unknown')
                last_days = data.get('days_left', 999)

                # 状态变化时发送通知
                if status != last_status:
                    return True

                # 根据状态决定通知频率
                now = datetime.now()
                if status == 'critical':
                    # 危险状态：每6小时通知一次
                    return now - last_notification > timedelta(hours=6)
                elif status == 'warning':
                    # 警告状态：每天通知一次
                    return now - last_notification > timedelta(days=1)
                elif status == 'expired':
                    # 过期状态：每天通知一次
                    return now - last_notification > timedelta(days=1)
                else:
                    # 正常状态：不重复通知
                    return False

        except:
            return True

    def update_last_notification(self, status, days_left):
        """更新最后通知时间"""
        data = {
            'last_notification': datetime.now().isoformat(),
            'status': status,
            'days_left': days_left
        }
        try:
            with open(self.last_notification_file, 'w') as f:
                json.dump(data, f)
        except Exception as e:
            print(f"更新通知时间失败: {e}")

    def send_push_notification(self, status, warnings, days_left):
        """发送推送通知"""
        try:
            from notifications.pushplus_notifier import PushPlusNotifier

            notifier = PushPlusNotifier()

            # 找出最短的剩余天数
            min_days = days_left
            for warning in warnings:
                if '天内过期' in warning:
                    # 提取天数
                    import re
                    match = re.search(r'(\d+\.?\d*)天内过期', warning)
                    if match:
                        days = float(match.group(1))
                        min_days = min(min_days, days)
                elif '小时内过期' in warning:
                    # 提取小时数并转换为天数
                    import re
                    match = re.search(r'(\d+\.?\d*)小时内过期', warning)
                    if match:
                        hours = float(match.group(1))
                        days = hours / 24
                        min_days = min(min_days, days)

            # 发送推送通知
            success = notifier.send_cookie_expiry_warning(min_days)
            if success:
                print(f"📱 推送通知已发送 (剩余{min_days:.1f}天)")
                self.update_last_notification(status, min_days)
            else:
                print("📱 推送通知发送失败")

        except Exception as e:
            print(f"📱 推送通知出错: {e}")

    def show_notification(self, status, warnings):
        """显示通知"""
        # 计算最短剩余天数
        min_days = 999
        for warning in warnings:
            if '天内过期' in warning:
                import re
                match = re.search(r'(\d+\.?\d*)天内过期', warning)
                if match:
                    days = float(match.group(1))
                    min_days = min(min_days, days)
            elif '小时内过期' in warning:
                import re
                match = re.search(r'(\d+\.?\d*)小时内过期', warning)
                if match:
                    hours = float(match.group(1))
                    days = hours / 24
                    min_days = min(min_days, days)

        # 发送推送通知（如果需要）
        if self.should_send_notification(status, min_days):
            self.send_push_notification(status, warnings, min_days)

        # 显示控制台通知
        if status == 'critical':
            print("\n" + "🚨" * 20)
            print("🚨 COOKIE紧急警告 🚨")
            print("🚨" * 20)
            print("您的Bilibili Cookie即将在24小时内过期!")
            print("请立即更新Cookie以继续使用高质量下载功能")
            print("\n快速更新命令:")
            print("python3 cookie_auto_updater.py update")
            print("🚨" * 20)

        elif status == 'warning':
            print("\n" + "⚠️ " * 15)
            print("⚠️  Cookie过期提醒")
            print("⚠️ " * 15)
            print("您的Bilibili Cookie即将过期")
            for warning in warnings:
                print(f"   - {warning}")
            print("\n建议尽快更新Cookie:")
            print("python3 cookie_auto_updater.py update")
            print("⚠️ " * 15)

        elif status == 'expired':
            print("\n" + "❌" * 20)
            print("❌ Cookie已过期!")
            print("❌" * 20)
            print("您的Bilibili Cookie已过期，现在只能下载480p质量")
            print("请更新Cookie以恢复高质量下载:")
            print("python3 cookie_auto_updater.py update")
            print("❌" * 20)
    
    async def auto_monitor(self):
        """自动监控Cookie状态"""
        if not self.should_check_cookies():
            return  # 不需要检查
        
        print("🔍 正在检查Cookie状态...")
        results = await self.check_cookie_validity()
        
        if results:
            status = results['status']
            warnings = results['warnings']
            
            # 显示通知
            if status in ['critical', 'warning', 'expired']:
                self.show_notification(status, warnings)
                
                # 询问是否立即更新
                if status in ['critical', 'expired']:
                    try:
                        response = input("\n是否立即更新Cookie? (y/N): ").strip().lower()
                        if response == 'y':
                            from auth.cookie_auto_updater import CookieAutoUpdater
                            updater = CookieAutoUpdater()
                            updater.interactive_update()
                    except KeyboardInterrupt:
                        print("\n操作取消")
            else:
                print("✅ Cookie状态正常")

# 集成到主下载器中
def integrate_cookie_monitor():
    """将Cookie监控集成到主下载器"""
    monitor = CookieMonitor()
    
    # 异步检查Cookie状态
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # 如果事件循环已在运行，创建任务
            asyncio.create_task(monitor.auto_monitor())
        else:
            # 如果没有运行的事件循环，直接运行
            asyncio.run(monitor.auto_monitor())
    except Exception as e:
        # 静默处理错误，不影响主程序
        pass

if __name__ == "__main__":
    monitor = CookieMonitor()
    asyncio.run(monitor.auto_monitor())
