# 统一视频下载器

一个支持YouTube和Bilibili的高质量视频下载工具，具有智能质量选择、Cookie认证和推送通知功能。

## 📁 项目结构

```
video_downloader/
├── core/                    # 核心模块
│   ├── downloader.py       # 统一下载器
│   └── url_detector.py     # URL检测器
├── downloaders/            # 下载器实现
│   ├── youtube_downloader.py
│   └── bilibili_downloader.py
├── auth/                   # 认证模块
│   ├── bilibili_cookies.py
│   ├── cookie_validator.py
│   ├── cookie_auto_updater.py
│   └── cookie_monitor.py
├── notifications/          # 通知模块
│   └── pushplus_notifier.py
├── config/                 # 配置模块
│   └── config.py
├── docs/                   # 文档
│   ├── README.md
│   ├── USAGE_EXAMPLES.md
│   └── ...
├── tests/                  # 测试
│   └── test_downloader.py
├── examples/               # 示例
│   └── example.py
├── cli.py                  # 命令行接口
├── main.py                 # 主程序入口
└── requirements.txt        # 依赖列表
```

## 🚀 快速开始

```bash
# 安装依赖
pip install -r requirements.txt

# 下载视频
python cli.py "视频URL" -o output.mp4 -q 1080p
```

## 📚 详细文档

请查看 `docs/` 目录中的详细文档。

## ✨ 主要功能

- 🎯 统一接口支持多平台
- 🔍 智能质量选择和自动降级
- 🍪 Cookie认证支持高质量下载
- 📱 推送通知和状态监控
- 🛡️ 自动备份和错误恢复

