"""
Bilibili Cookie配置
用于高质量下载的认证信息
"""

# 将您的Cookie字符串解析为字典
COOKIE_STRING = """enable_web_push=DISABLE; header_theme_version=CLOSE; home_feed_column=5; browser_resolution=2560-1271; DedeUserID=27503984; DedeUserID__ckMd5=d1d309fd72bd0e6d; enable_feed_channel=ENABLE; CURRENT_QUALITY=80; buvid3=625EBAF4-D0A7-FBA6-3647-3A8C85C04D9386423infoc; b_nut=1750689186; b_lsid=DF41A1DD_1979D34FFA0; _uuid=D874510F6-E416-58D9-FF36-29E434107C5FE85706infoc; bili_ticket=eyJhbGciOiJIUzI1NiIsImtpZCI6InMwMyIsInR5cCI6IkpXVCJ9.eyJleHAiOjE3NTA5NDgzODcsImlhdCI6MTc1MDY4OTEyNywicGx0IjotMX0.le6gk-HqIHwIPI6e4avnFiVk4nyH6A0reLQn00zCIFc; bili_ticket_expires=1750948327; buvid_fp=1bc59b4ef1449c16d4ff2a5c8bb32feb; SESSDATA=c30f4f80%2C1766241188%2Ce5ce9%2A62CjCQbsKs5hcYbIHl0jjk31p4NzG9kSjrLma2cFj1VHdM5fPlEGa85zhbkhrpO9kYr-gSVkxOMGdRYUdfUHdZSHZodV83TFJueEZ0VXRKOFgtY1J1NTJEQU9ZS3Y2ZmdETVBjb3JPbkNmMVhIS0d5QkZkLUQxdG9wd2I3UTJzYlJhX0ZZX3AtanRRIIEC; bili_jct=3c381402534bd4828e9970d2ff2e3e93; buvid4=09EEFFE8-843D-B877-97BF-2C2048DD940283878-024051809-kM+ut9HDKfLMmkwR9NCDGA%3D%3D; rpdid=|(J|)kmu|ml~0J'u~l|Y)mumu; sid=8eu403g9; CURRENT_FNVAL=4048"""

def parse_cookies(cookie_string):
    """解析Cookie字符串为字典"""
    cookies = {}
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
    print(f"解析的Cookie数量: {len(cookies)}")
    print(f"用户ID: {cookies.get('DedeUserID', '未找到')}")
    print(f"SESSDATA: {cookies.get('SESSDATA', '未找到')[:20]}...")
    print("Cookie解析成功！")
