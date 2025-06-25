#!/usr/bin/env python3
"""
Bilibili Cookie有效性检测工具 - 异步版本
"""

import asyncio
import time
from datetime import datetime
from urllib.parse import unquote

async def analyze_cookie_expiry(cookies):
    """分析Cookie有效期"""
    current_time = time.time()
    results = {
        'current_time': current_time,
        'expires': {},
        'warnings': [],
        'status': 'unknown'
    }
    
    # 分析bili_ticket_expires
    if 'bili_ticket_expires' in cookies:
        try:
            expires = int(cookies['bili_ticket_expires'])
            days_left = (expires - current_time) / (24 * 3600)
            results['expires']['bili_ticket'] = {
                'timestamp': expires,
                'datetime': datetime.fromtimestamp(expires),
                'days_left': days_left
            }
            
            if days_left < 0:
                results['warnings'].append('bili_ticket已过期')
            elif days_left < 1:
                results['warnings'].append(f'bili_ticket将在{days_left*24:.1f}小时内过期')
            elif days_left < 7:
                results['warnings'].append(f'bili_ticket将在{days_left:.1f}天内过期')
                
        except ValueError:
            results['warnings'].append('bili_ticket_expires格式错误')
    
    # 分析SESSDATA
    if 'SESSDATA' in cookies:
        try:
            sessdata = unquote(cookies['SESSDATA'])
            parts = sessdata.split(',')
            if len(parts) >= 2:
                expires = int(parts[1])
                days_left = (expires - current_time) / (24 * 3600)
                results['expires']['SESSDATA'] = {
                    'timestamp': expires,
                    'datetime': datetime.fromtimestamp(expires),
                    'days_left': days_left
                }
                
                if days_left < 0:
                    results['warnings'].append('SESSDATA已过期')
                elif days_left < 7:
                    results['warnings'].append(f'SESSDATA将在{days_left:.1f}天内过期')
                elif days_left < 30:
                    results['warnings'].append(f'SESSDATA将在{days_left:.1f}天内过期，建议准备更新')
        except (ValueError, IndexError):
            results['warnings'].append('SESSDATA格式错误')
    
    # 确定整体状态
    if not results['expires']:
        results['status'] = 'no_auth'
    elif any('已过期' in w for w in results['warnings']):
        results['status'] = 'expired'
    elif any('小时内过期' in w for w in results['warnings']):
        results['status'] = 'critical'
    elif any('天内过期' in w for w in results['warnings']):
        results['status'] = 'warning'
    else:
        results['status'] = 'good'
    
    return results

async def print_cookie_status(cookies):
    """打印Cookie状态报告"""
    results = await analyze_cookie_expiry(cookies)
    
    print("=" * 60)
    print("Bilibili Cookie状态报告")
    print("=" * 60)
    
    current_dt = datetime.fromtimestamp(results['current_time'])
    print(f"检查时间: {current_dt.strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # 显示各个Cookie的有效期
    if results['expires']:
        print("Cookie有效期:")
        print("-" * 30)
        for name, info in results['expires'].items():
            dt_str = info['datetime'].strftime('%Y-%m-%d %H:%M:%S')
            days = info['days_left']
            if days > 0:
                print(f"{name:12}: {dt_str} ({days:.1f}天后)")
            else:
                print(f"{name:12}: {dt_str} (已过期 {abs(days):.1f}天)")
        print()
    
    # 显示警告
    if results['warnings']:
        print("⚠️  警告:")
        for warning in results['warnings']:
            print(f"   - {warning}")
        print()
    
    # 显示状态和建议
    status_messages = {
        'good': '✅ Cookie状态良好',
        'warning': '⚠️  Cookie即将过期，建议关注',
        'critical': '🚨 Cookie即将过期，建议立即更新',
        'expired': '❌ Cookie已过期，需要重新登录',
        'no_auth': '❌ 未找到认证Cookie',
        'unknown': '❓ 无法确定Cookie状态'
    }
    
    print(f"状态: {status_messages.get(results['status'], '未知')}")
    
    # 提供建议
    print("\n建议:")
    if results['status'] == 'good':
        print("- Cookie状态正常，可以正常使用高质量下载")
    elif results['status'] == 'warning':
        print("- 继续使用，但建议准备更新Cookie")
        print("- 定期检查Cookie状态")
    elif results['status'] == 'critical':
        print("- 建议立即重新登录获取新Cookie")
        print("- 备份当前Cookie以防万一")
    elif results['status'] in ['expired', 'no_auth']:
        print("- 需要重新登录bilibili.com获取新Cookie")
        print("- 没有有效Cookie时将限制为480p下载")
    
    print("\n如何更新Cookie:")
    print("1. 在浏览器中登录bilibili.com")
    print("2. 打开开发者工具(F12) → Network标签")
    print("3. 刷新页面，找到API请求")
    print("4. 复制新的Cookie值")
    print("5. 更新bilibili_cookies.py文件")
    
    return results

async def check_current_cookies():
    """检查当前配置的Cookie"""
    try:
        from auth.bilibili_cookies import get_cookies
        cookies = get_cookies()
        return await print_cookie_status(cookies)
    except ImportError:
        print("❌ 未找到bilibili_cookies.py文件")
        print("请先配置Cookie文件")
        return None

if __name__ == "__main__":
    asyncio.run(check_current_cookies())
