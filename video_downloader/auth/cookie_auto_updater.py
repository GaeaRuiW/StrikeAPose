#!/usr/bin/env python3
"""
Bilibili Cookie自动更新工具
提供多种自动化程度的Cookie更新方案
"""

import os
import time
import json
import asyncio
import aiohttp
from datetime import datetime
from urllib.parse import unquote

class CookieAutoUpdater:
    """Cookie自动更新器"""
    
    def __init__(self):
        self.cookie_file = "bilibili_cookies.py"
        self.backup_dir = "cookie_backups"
        
    async def backup_current_cookies(self):
        """备份当前Cookie"""
        if not os.path.exists(self.backup_dir):
            os.makedirs(self.backup_dir)
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_file = os.path.join(self.backup_dir, f"cookies_backup_{timestamp}.py")
        
        if os.path.exists(self.cookie_file):
            import shutil
            shutil.copy2(self.cookie_file, backup_file)
            print(f"✅ Cookie已备份到: {backup_file}")
            return backup_file
        return None
    
    def parse_cookie_string(self, cookie_string):
        """解析Cookie字符串"""
        cookies = {}
        for item in cookie_string.strip().split('; '):
            if '=' in item:
                key, value = item.split('=', 1)
                cookies[key.strip()] = value.strip()
        return cookies
    
    async def update_cookie_file(self, new_cookie_string):
        """更新Cookie文件"""
        # 备份当前Cookie
        await self.backup_current_cookies()
        
        # 解析新Cookie
        cookies = self.parse_cookie_string(new_cookie_string)
        
        # 生成新的Cookie文件内容
        content = f'''"""
Bilibili Cookie配置
自动更新于: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
"""

# Cookie字符串
COOKIE_STRING = """{new_cookie_string}"""

def parse_cookies(cookie_string):
    """解析Cookie字符串为字典"""
    cookies = {{}}
    for item in cookie_string.split('; '):
        if '=' in item:
            key, value = item.split('=', 1)
            cookies[key] = value
    return cookies

# 解析后的Cookie字典
BILIBILI_COOKIES = parse_cookies(COOKIE_STRING)

def get_cookies():
    """获取Bilibili认证Cookie"""
    return BILIBILI_COOKIES

if __name__ == "__main__":
    # 测试Cookie解析
    cookies = get_cookies()
    print(f"解析的Cookie数量: {{len(cookies)}}")
    print(f"用户ID: {{cookies.get('DedeUserID', '未找到')}}")
    print(f"SESSDATA: {{cookies.get('SESSDATA', '未找到')[:20]}}...")
    print("Cookie解析成功！")
'''
        
        # 写入新文件
        with open(self.cookie_file, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print(f"✅ Cookie文件已更新: {self.cookie_file}")
        
        # 验证新Cookie
        return await self.validate_new_cookies()
    
    async def validate_new_cookies(self):
        """验证新Cookie是否有效"""
        try:
            from auth.bilibili_cookies import get_cookies
            cookies = get_cookies()
            
            # 测试API调用
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                'Referer': 'https://www.bilibili.com/'
            }
            
            if cookies:
                cookie_str = '; '.join([f'{k}={v}' for k, v in cookies.items()])
                headers['Cookie'] = cookie_str
            
            async with aiohttp.ClientSession() as session:
                # 测试用户信息API
                async with session.get('https://api.bilibili.com/x/web-interface/nav', headers=headers) as response:
                    if response.status == 200:
                        data = await response.json()
                        if data.get('code') == 0:
                            user_info = data.get('data', {})
                            print(f"✅ Cookie验证成功!")
                            print(f"   用户: {user_info.get('uname', '未知')}")
                            print(f"   等级: Lv{user_info.get('level_info', {}).get('current_level', 0)}")
                            return True
                        else:
                            print(f"❌ Cookie验证失败: {data.get('message', '未知错误')}")
                            return False
                    else:
                        print(f"❌ API请求失败: {response.status}")
                        return False
                        
        except Exception as e:
            print(f"❌ Cookie验证出错: {e}")
            return False
    
    async def interactive_update(self):
        """交互式Cookie更新"""
        print("=" * 60)
        print("Bilibili Cookie交互式更新工具")
        print("=" * 60)
        
        print("\n📋 获取新Cookie的步骤:")
        print("1. 在浏览器中打开 https://www.bilibili.com")
        print("2. 确保已登录到您的账户")
        print("3. 按F12打开开发者工具")
        print("4. 切换到 Network(网络) 标签")
        print("5. 刷新页面")
        print("6. 在请求列表中找到任意API请求")
        print("7. 点击请求，在Headers中找到Cookie")
        print("8. 复制完整的Cookie值")
        
        print("\n" + "="*60)
        print("请粘贴新的Cookie字符串 (按Ctrl+C取消):")
        print("="*60)
        
        try:
            new_cookie = input().strip()
            if not new_cookie:
                print("❌ 未输入Cookie，操作取消")
                return False
                
            print(f"\n📝 收到Cookie，长度: {len(new_cookie)} 字符")
            
            # 简单验证Cookie格式
            if 'SESSDATA=' not in new_cookie:
                print("⚠️  警告: Cookie中未找到SESSDATA，可能不完整")
                confirm = input("是否继续? (y/N): ").strip().lower()
                if confirm != 'y':
                    print("操作取消")
                    return False
            
            # 更新Cookie文件
            print("\n🔄 正在更新Cookie文件...")
            success = await self.update_cookie_file(new_cookie)
            
            if success:
                print("\n🎉 Cookie更新成功!")
                return True
            else:
                print("\n❌ Cookie更新失败，请检查Cookie是否有效")
                return False
                
        except KeyboardInterrupt:
            print("\n\n操作已取消")
            return False
        except Exception as e:
            print(f"\n❌ 更新过程出错: {e}")
            return False
    
    async def auto_check_and_prompt(self):
        """自动检查Cookie状态并提示更新"""
        try:
            from auth.cookie_validator import check_current_cookies
            results = await check_current_cookies()
            
            if results and results['status'] in ['critical', 'expired']:
                print("\n🚨 检测到Cookie即将过期或已过期!")
                print("建议立即更新Cookie以继续使用高质量下载功能")
                
                update = input("\n是否现在更新Cookie? (y/N): ").strip().lower()
                if update == 'y':
                    return await self.interactive_update()
            elif results and results['status'] == 'warning':
                print("\n⚠️  Cookie即将在一周内过期")
                print("建议准备更新Cookie")

                update = input("\n是否现在更新Cookie? (y/N): ").strip().lower()
                if update == 'y':
                    return await self.interactive_update()
            else:
                print("\n✅ Cookie状态正常，无需更新")
                
        except Exception as e:
            print(f"❌ 检查Cookie状态时出错: {e}")
            
        return False

async def main():
    """主函数"""
    updater = CookieAutoUpdater()

    import sys
    if len(sys.argv) > 1:
        command = sys.argv[1]
        if command == 'check':
            # 仅检查状态
            await updater.auto_check_and_prompt()
        elif command == 'update':
            # 强制更新
            await updater.interactive_update()
        elif command == 'backup':
            # 仅备份
            await updater.backup_current_cookies()
        else:
            print("用法: python3 cookie_auto_updater.py [check|update|backup]")
    else:
        # 默认：检查并提示
        await updater.auto_check_and_prompt()

if __name__ == "__main__":
    asyncio.run(main())
