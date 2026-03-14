#!/usr/bin/env python3
"""
东方财富 session 保活守护
- 每 5 分钟访问一次东方财富页面刷新 cookie
- 检测到过期时输出告警
- 用法: python3 eastmoney_keepalive.py
"""

import json
import os
import sys
import time
import asyncio
import logging
from datetime import datetime
from playwright.async_api import async_playwright

sys.path.insert(0, os.path.dirname(__file__))
from eastmoney_login import COOKIE_FILE, UA, load_cookies, save_cookies

INTERVAL = 300  # 5 分钟
LOG_FILE = "/opt/quant-llm/output/keepalive.log"
REFRESH_URLS = [
    "https://i.eastmoney.com/",
    "https://group.eastmoney.com/room/index.html",
]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger("keepalive")


async def refresh_session():
    """访问页面刷新 session，并保存更新后的 cookie"""
    state = load_cookies()
    if not state:
        log.error("Cookie 文件不存在")
        return False

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(storage_state=COOKIE_FILE, user_agent=UA)
        page = await context.new_page()

        valid = False
        for url in REFRESH_URLS:
            try:
                resp = await page.goto(url, timeout=15000, wait_until="domcontentloaded")
                await page.wait_for_timeout(2000)
                final_url = page.url

                if "passport" in final_url.lower() or "login" in final_url.lower():
                    log.warning(f"被重定向到登录页: {final_url}")
                    continue

                log.info(f"刷新成功: {url} -> {final_url} (status={resp.status})")
                valid = True
                break
            except Exception as e:
                log.warning(f"访问失败 {url}: {e}")

        if valid:
            # 保存更新后的 cookie
            new_cookies = await context.cookies()
            if new_cookies:
                save_cookies(new_cookies)
                log.info(f"Cookie 已更新 ({len(new_cookies)} 个)")

        await browser.close()
        return valid


async def daemon():
    """守护循环"""
    log.info("=" * 40)
    log.info("东方财富 session 保活启动")
    log.info(f"刷新间隔: {INTERVAL}s")
    log.info(f"Cookie 文件: {COOKIE_FILE}")
    log.info("=" * 40)

    fail_count = 0
    while True:
        try:
            ok = await refresh_session()
            if ok:
                fail_count = 0
            else:
                fail_count += 1
                log.error(f"Session 刷新失败 (连续 {fail_count} 次)")
                if fail_count >= 3:
                    log.critical("Session 已过期！请手动更新 cookie: python3 eastmoney_login.py --update 'cookie'")
        except Exception as e:
            fail_count += 1
            log.exception(f"保活异常: {e}")

        await asyncio.sleep(INTERVAL)


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--once":
        # 单次刷新模式
        ok = asyncio.run(refresh_session())
        sys.exit(0 if ok else 1)
    else:
        asyncio.run(daemon())


if __name__ == "__main__":
    main()
