#!/usr/bin/env python3
"""查看东方财富模拟盘持仓和资金情况"""

import asyncio
from playwright.async_api import async_playwright
from eastmoney_login import create_authenticated_context, COOKIE_FILE

OUTPUT = "/opt/quant-llm/output"


async def main():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await create_authenticated_context(browser)
        page = await context.new_page()

        print("访问模拟组合交易页面...")
        await page.goto("https://group.eastmoney.com/room/index.html",
                         timeout=20000, wait_until="domcontentloaded")
        await page.wait_for_timeout(5000)

        # 提取账户信息
        account_info = await page.evaluate("""() => {
            const text = document.body.innerText;
            const result = {};

            // 总资产
            const totalMatch = text.match(/总资产[：:：]?\\s*([\\d,.]+)/);
            if (totalMatch) result.totalAsset = totalMatch[1];

            // 收益
            const dayMatch = text.match(/日收益[：:：]?\\s*([\\-\\d.]+%?)/);
            if (dayMatch) result.dayReturn = dayMatch[1];
            const weekMatch = text.match(/周收益[：:：]?\\s*([\\-\\d.]+%?)/);
            if (weekMatch) result.weekReturn = weekMatch[1];
            const monthMatch = text.match(/月收益[：:：]?\\s*([\\-\\d.]+%?)/);
            if (monthMatch) result.monthReturn = monthMatch[1];
            const yearMatch = text.match(/年收益[：:：]?\\s*([\\-\\d.]+%?)/);
            if (yearMatch) result.yearReturn = yearMatch[1];

            return result;
        }""")

        print("\n========== 模拟盘账户概览 ==========")
        print(f"  总资产: {account_info.get('totalAsset', 'N/A')} 元")
        print(f"  日收益: {account_info.get('dayReturn', 'N/A')}")
        print(f"  周收益: {account_info.get('weekReturn', 'N/A')}")
        print(f"  月收益: {account_info.get('monthReturn', 'N/A')}")
        print(f"  年收益: {account_info.get('yearReturn', 'N/A')}")

        # 提取持仓信息 - 点击持仓 tab
        print("\n========== 持仓明细 ==========")
        try:
            # 找到并点击"持仓"tab
            await page.click("text=持仓", timeout=3000)
            await page.wait_for_timeout(2000)
        except:
            pass

        positions = await page.evaluate("""() => {
            // 查找持仓表格
            const tables = document.querySelectorAll('table');
            const positions = [];
            for (const table of tables) {
                const rows = table.querySelectorAll('tr');
                for (const row of rows) {
                    const cells = row.querySelectorAll('td');
                    if (cells.length >= 4) {
                        const text = Array.from(cells).map(c => c.textContent.trim());
                        // 过滤掉表头和空行
                        if (text[0] && !text[0].includes('代码') && text[0].match(/\\d/)) {
                            positions.push(text);
                        }
                    }
                }
            }
            return positions;
        }""")

        if positions:
            for pos in positions:
                print(f"  {' | '.join(pos[:6])}")
        else:
            print("  (未检测到持仓数据，可能需要在交易页面手动查看)")

        # 提取自选股列表
        print("\n========== 自选股 ==========")
        watchlist = await page.evaluate("""() => {
            const items = [];
            const links = document.querySelectorAll('.stockList a, .stock-item, [class*="stock"] a');
            for (const link of links) {
                const text = link.textContent.trim();
                if (text && text.length < 20 && !text.includes('添加')) {
                    items.push(text);
                }
            }
            // 备用：从左侧面板提取
            if (items.length === 0) {
                const tds = document.querySelectorAll('td');
                for (const td of tds) {
                    const text = td.textContent.trim();
                    if (text.match(/^[\\u4e00-\\u9fa5A-Za-z]+$/) && text.length <= 6) {
                        items.push(text);
                    }
                }
            }
            return [...new Set(items)].slice(0, 20);
        }""")

        if watchlist:
            print(f"  {', '.join(watchlist)}")

        # 截图
        await page.screenshot(path=f"{OUTPUT}/trade_page.png", full_page=False)
        print(f"\n截图: {OUTPUT}/trade_page.png")
        print("=" * 40)

        await browser.close()


asyncio.run(main())
