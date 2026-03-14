#!/usr/bin/env python3
"""
东方财富模拟炒股 - 登录模块

登录方式：cookie 持久化
- cookie 文件: /opt/quant-llm/.eastmoney_cookies.json
- 首次使用：浏览器手动登录后，在 Console 执行 document.cookie 获取 cookie
- 刷新方式：python3 eastmoney_login.py --update "cookie字符串"
- 验证方式：python3 eastmoney_login.py --check
"""

import json
import os
import sys
import time
import asyncio
from playwright.async_api import async_playwright

COOKIE_FILE = "/opt/quant-llm/.eastmoney_cookies.json"
UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"


def parse_raw_cookie(raw: str) -> list[dict]:
    """解析 document.cookie 字符串为 Playwright cookie 格式"""
    cookies = []
    for pair in raw.strip().split("; "):
        if "=" not in pair:
            continue
        name, value = pair.split("=", 1)
        cookies.append({
            "name": name.strip(),
            "value": value.strip(),
            "domain": ".eastmoney.com",
            "path": "/",
            "httpOnly": False,
            "secure": False,
            "sameSite": "Lax",
        })
    return cookies


def save_cookies(cookies: list[dict]):
    """保存 cookie 到文件"""
    state = {"cookies": cookies, "origins": []}
    with open(COOKIE_FILE, "w") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)
    os.chmod(COOKIE_FILE, 0o600)


def load_cookies() -> dict | None:
    """加载已保存的 cookie storage_state"""
    if not os.path.exists(COOKIE_FILE):
        return None
    with open(COOKIE_FILE) as f:
        return json.load(f)


def cookie_age_hours() -> float:
    """cookie 文件的年龄（小时）"""
    if not os.path.exists(COOKIE_FILE):
        return float("inf")
    return (time.time() - os.path.getmtime(COOKIE_FILE)) / 3600


async def create_authenticated_context(playwright_browser):
    """创建已认证的 browser context（供其他模块调用）"""
    state = load_cookies()
    if state is None:
        raise RuntimeError("Cookie 文件不存在，请先运行: python3 eastmoney_login.py --update 'cookie字符串'")

    context = await playwright_browser.new_context(
        storage_state=COOKIE_FILE,
        user_agent=UA,
    )
    return context


async def verify_login(context) -> bool:
    """验证 cookie 是否有效"""
    page = await context.new_page()
    try:
        await page.goto("https://i.eastmoney.com/", timeout=15000, wait_until="domcontentloaded")
        await page.wait_for_timeout(2000)
        url = page.url
        return "passport" not in url.lower() and "login" not in url.lower()
    except Exception:
        return False
    finally:
        await page.close()


async def check_login():
    """检查当前 cookie 状态"""
    state = load_cookies()
    if state is None:
        print("✗ Cookie 文件不存在")
        return False

    age = cookie_age_hours()
    cookie_names = [c["name"] for c in state.get("cookies", [])]
    has_auth = all(k in cookie_names for k in ("ct", "ut"))

    print(f"Cookie 文件: {COOKIE_FILE}")
    print(f"Cookie 数量: {len(cookie_names)}")
    print(f"关键 cookie: ct={'✓' if 'ct' in cookie_names else '✗'}, ut={'✓' if 'ut' in cookie_names else '✗'}, pi={'✓' if 'pi' in cookie_names else '✗'}")
    print(f"文件年龄: {age:.1f} 小时")

    if not has_auth:
        print("✗ 缺少关键认证 cookie")
        return False

    # 在线验证
    print("在线验证中...")
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await create_authenticated_context(browser)
        valid = await verify_login(context)
        await browser.close()

    if valid:
        print("✓ 登录状态有效")
    else:
        print("✗ Cookie 已失效，请更新")
    return valid


def update_cookies(raw_cookie: str):
    """更新 cookie"""
    cookies = parse_raw_cookie(raw_cookie)
    if not cookies:
        print("✗ 解析失败，cookie 为空")
        return False

    cookie_names = [c["name"] for c in cookies]
    if "ct" not in cookie_names or "ut" not in cookie_names:
        print(f"⚠ 缺少关键 cookie (ct/ut)，已有: {cookie_names}")

    save_cookies(cookies)
    print(f"✓ 已保存 {len(cookies)} 个 cookie 到 {COOKIE_FILE}")
    return True


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(0)

    cmd = sys.argv[1]

    if cmd == "--check":
        asyncio.run(check_login())

    elif cmd == "--update":
        if len(sys.argv) < 3:
            print("用法: python3 eastmoney_login.py --update 'cookie字符串'")
            sys.exit(1)
        raw = sys.argv[2]
        if update_cookies(raw):
            print("验证中...")
            asyncio.run(check_login())

    else:
        print(f"未知命令: {cmd}")
        print("可用: --check | --update 'cookie'")
