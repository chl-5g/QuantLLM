#!/usr/bin/env python3
"""
东方财富模拟盘 HTTP API 客户端
逆向自 group.eastmoney.com JS (2026-04-02)

域名:
  查询: https://simqry2.eastmoney.com/qry_tzzh_v2
  操作: https://simoper.eastmoney.com/oper_tzzh_v2

认证: utToken(cookie ut) + ctToken(cookie ct)
"""

from __future__ import annotations

import json
import random
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import requests


@dataclass
class EastMoneyConfig:
    """从浏览器cookie中提取的认证信息"""
    ut_token: str       # cookie 'ut'
    ct_token: str       # cookie 'ct'
    userid: str         # 从 cookie 'pi' 提取的用户ID
    cookies: str        # 完整cookie字符串（用于请求头）
    zjzh: str = ''      # 组合账号（可后续设置）

    @classmethod
    def from_cookies(cls, cookie_str: str, zjzh: str = '') -> 'EastMoneyConfig':
        """从完整cookie字符串解析配置"""
        def _get(name: str) -> str:
            m = re.search(rf'(?:^|;\s*){re.escape(name)}=([^;]+)', cookie_str)
            return m.group(1) if m else ''

        ut = _get('ut')
        ct = _get('ct')
        pi_raw = _get('pi')
        # userid是pi值的第一个分隔符前的数字
        userid = ''
        if pi_raw:
            from urllib.parse import unquote
            pi_decoded = unquote(pi_raw)
            m = re.match(r'(\d+)', pi_decoded)
            if m:
                userid = m.group(1)

        return cls(ut_token=ut, ct_token=ct, userid=userid, cookies=cookie_str, zjzh=zjzh)


class EastMoneyAPI:
    """东方财富模拟盘 HTTP API"""

    QRY_URL = 'https://simqry2.eastmoney.com/qry_tzzh_v2'
    OPER_URL = 'https://simoper.eastmoney.com/oper_tzzh_v2'

    def __init__(self, config: EastMoneyConfig, timeout: int = 20, max_retries: int = 3):
        self.config = config
        self.timeout = timeout
        self.max_retries = max_retries
        self.session = requests.Session()
        # 强制直连：忽略 HTTP(S)_PROXY / ALL_PROXY 等环境变量。
        self.session.trust_env = False
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36',
            'Cookie': config.cookies,
            'Referer': 'https://group.eastmoney.com/Trade/Buy',
            'Accept': '*/*',
        })

    def _base_params(self) -> Dict[str, str]:
        return {
            'plat': '2',
            'ver': 'web20',
            'utToken': self.config.ut_token,
            'ctToken': self.config.ct_token,
        }

    def _request(self, url: str, extra_params: Dict[str, str]) -> Dict[str, Any]:
        """发送GET请求，带重试"""
        params = {**self._base_params(), **extra_params}
        last_err = None
        for attempt in range(self.max_retries):
            try:
                r = self.session.get(url, params=params, timeout=self.timeout)
                text = r.text.strip()
                if not text:
                    return {'result': -1, 'message': 'empty_response'}
                return json.loads(text)
            except (requests.ConnectionError, requests.Timeout, OSError) as e:
                last_err = e
                if attempt < self.max_retries - 1:
                    time.sleep(1 * (attempt + 1))
                continue
            except json.JSONDecodeError:
                return {'result': -1, 'message': f'json_parse_error', 'raw': text[:500]}
        return {'result': -1, 'message': f'request_failed_after_{self.max_retries}_retries: {last_err}'}

    def _qry(self, params: Dict[str, str]) -> Dict[str, Any]:
        return self._request(self.QRY_URL, params)

    def _oper(self, params: Dict[str, str]) -> Dict[str, Any]:
        return self._request(self.OPER_URL, params)

    # ========== 查询类 ==========

    def get_user_info(self) -> Dict[str, Any]:
        """获取用户基本信息"""
        return self._qry({'type': 'spo_user_basic', 'userid': self.config.userid})

    def list_portfolios(self) -> List[Dict[str, Any]]:
        """获取我的组合列表"""
        r = self._qry({'type': 'spo_zuhe_preview', 'userid': self.config.userid})
        return r.get('data', []) if r.get('result') == 0 else []

    def get_portfolio_detail(self, zjzh: str = '') -> Dict[str, Any]:
        """获取组合详情"""
        return self._qry({'type': 'spo_zuhe_detail_basic', 'zjzh': zjzh or self.config.zjzh})

    def get_hold_detail(self, zjzh: str = '', page: int = 1, size: int = 50) -> Dict[str, Any]:
        """获取持仓明细（公开查询）"""
        return self._qry({
            'type': 'spo_hold_detail', 'zjzh': zjzh or self.config.zjzh,
            'recIdx': str(page), 'recCnt': str(size),
        })

    # ========== 操作类 ==========

    def get_balance(self, zjzh: str = '') -> Dict[str, Any]:
        """查询资金（总资产、可用余额等）"""
        r = self._oper({
            'type': 'spo_bal_info',
            'zjzh': zjzh or self.config.zjzh,
            'userid': self.config.userid,
        })
        data = r.get('data', [])
        if isinstance(data, list) and data:
            return data[0]
        return data if isinstance(data, dict) else {}

    def get_positions(self, zjzh: str = '') -> List[Dict[str, Any]]:
        """查询持仓"""
        r = self._oper({
            'type': 'spo_hold',
            'zjzh': zjzh or self.config.zjzh,
            'userid': self.config.userid,
            'reqUserid': self.config.userid,
            'recIdx': '0', 'recCnt': '0',
        })
        return r.get('data', []) if r.get('result') == 0 else []

    def get_today_orders(self, zjzh: str = '') -> List[Dict[str, Any]]:
        """查询当日委托"""
        r = self._oper({
            'type': 'spo_orders_all',
            'zjzh': zjzh or self.config.zjzh,
            'userid': self.config.userid,
            'reqUserid': self.config.userid,
            'recIdx': '0', 'recCnt': '0',
        })
        return r.get('data', []) if r.get('result') == 0 else []

    def get_today_deals(self, zjzh: str = '') -> List[Dict[str, Any]]:
        """查询当日成交"""
        r = self._oper({
            'type': 'spo_deal_today',
            'zjzh': zjzh or self.config.zjzh,
            'userid': self.config.userid,
            'reqUserid': self.config.userid,
            'recIdx': '0', 'recCnt': '0',
        })
        return r.get('data', []) if r.get('result') == 0 else []

    def get_cancelable_orders(self, zjzh: str = '') -> List[Dict[str, Any]]:
        """查询可撤委托"""
        r = self._oper({
            'type': 'spo_orders_cancel',
            'zjzh': zjzh or self.config.zjzh,
            'userid': self.config.userid,
            'reqUserid': self.config.userid,
        })
        return r.get('data', []) if r.get('result') == 0 else []

    def get_max_buy(self, stk_code: str, price: str, zjzh: str = '') -> Dict[str, Any]:
        """查询最大可买"""
        mkt = '1' if stk_code.startswith(('6', '9')) else '0'
        return self._oper({
            'type': 'spo_order_limit',
            'zjzh': zjzh or self.config.zjzh,
            'userid': self.config.userid,
            'mmfx': '1',  # 买入
            'mktCode': mkt,
            'stkCode': stk_code,
            'price': price,
            'wtsl': '0',
        })

    def get_max_sell(self, stk_code: str, price: str, zjzh: str = '') -> Dict[str, Any]:
        """查询最大可卖"""
        mkt = '1' if stk_code.startswith(('6', '9')) else '0'
        return self._oper({
            'type': 'spo_order_limit',
            'zjzh': zjzh or self.config.zjzh,
            'userid': self.config.userid,
            'mmfx': '2',  # 卖出
            'mktCode': mkt,
            'stkCode': stk_code,
            'price': price,
            'wtsl': '-1',
        })

    def place_order(self, stk_code: str, price: str, quantity: int, side: str = 'buy', zjzh: str = '') -> Dict[str, Any]:
        """
        下单（买入/卖出）

        Args:
            stk_code: 股票代码，如 "000001", "600519"
            price: 委托价格
            quantity: 委托数量（股）
            side: "buy" 或 "sell"
            zjzh: 组合账号（不传则用默认）

        Returns:
            API响应dict，result=0表示成功
        """
        mkt = '1' if stk_code.startswith(('6', '9')) else '0'
        mmfx = '1' if side == 'buy' else '2'
        r = self._oper({
            'type': 'spo_order',
            'zjzh': zjzh or self.config.zjzh,
            'userid': self.config.userid,
            'mmfx': mmfx,
            'mktCode': mkt,
            'stkCode': stk_code,
            'price': str(price),
            'wtsl': str(quantity),
            'r': str(random.randint(10000000, 99999999)),
        })
        return r

    def cancel_order(self, stk_code: str, wth: str, mmfx: str, zjzh: str = '') -> Dict[str, Any]:
        """
        撤单

        Args:
            stk_code: 股票代码
            wth: 委托号
            mmfx: 买卖方向 "1"=买 "2"=卖
            zjzh: 组合账号
        """
        mkt = '1' if stk_code.startswith(('6', '9')) else '0'
        return self._oper({
            'type': 'spo_cancel',
            'zjzh': zjzh or self.config.zjzh,
            'mmfx': mmfx,
            'mktCode': mkt,
            'stkCode': stk_code,
            'wth': wth,
        })

    def create_portfolio(self, name: str, comment: str = '', public: bool = True) -> Dict[str, Any]:
        """创建新组合"""
        return self._oper({
            'type': 'spo_create_zuhe',
            'zuheName': name,
            'comment': comment,
            'authority': '1' if public else '0',
        })

    def delete_portfolio(self, zjzh: str) -> Dict[str, Any]:
        """删除组合"""
        return self._oper({'type': 'spo_delete_zuhe', 'zjzh': zjzh})

    # ========== 快照（兼容旧接口） ==========

    def fetch_snapshot(self, zjzh: str = '') -> Dict[str, Any]:
        """
        获取组合快照（替代 Playwright 版本）
        返回格式与旧 eastmoney_portfolio_snapshot.py 兼容
        """
        _zjzh = zjzh or self.config.zjzh
        out = {
            'total_asset': 0.0,
            'available_cash': 0.0,
            'positions_pct': {},
            'ok': False,
            'error': '',
        }
        try:
            bal = self.get_balance(_zjzh)
            out['total_asset'] = float(bal.get('zzc', 0) or 0)
            out['available_cash'] = float(bal.get('kyye', 0) or 0)

            positions = self.get_positions(_zjzh)
            total = out['total_asset'] or 1.0
            for pos in positions:
                code = str(pos.get('stkCode', ''))
                mkt = str(pos.get('mktCode', ''))
                if not code or len(code) != 6:
                    continue
                # 市值
                mv = float(pos.get('mktVal', 0) or pos.get('stkMktVal', 0) or 0)
                if mv <= 0:
                    # 尝试 数量*现价
                    qty = float(pos.get('stkQty', 0) or pos.get('currentQty', 0) or 0)
                    price = float(pos.get('lastPrice', 0) or pos.get('currentPrice', 0) or 0)
                    mv = qty * price
                if mv > 0:
                    prefix = 'sh' if code[0] in ('6', '9') else 'sz'
                    sym = prefix + code
                    out['positions_pct'][sym] = round(mv / total, 6)

            out['ok'] = True
        except Exception as e:
            out['error'] = str(e)[:500]
        return out


# ========== CLI 测试 ==========

if __name__ == '__main__':
    import sys

    COOKIE_STR = 'qgqp_b_id=ed40fc1fcadbbf1c7c3ed7317c6143fb; mtp=1; ct=mmvv8ioGtzMkjZU1ZA8TD-Jwb4rH3PwG3rep9TaqLFY9pZbJdLR_2izkTyEyVHSS0UtwPmoQQZRXN-jaEm1n1b39RUvFWuWP9o2GwivbKFt9kGYua8kv8xjx97xVfuFRK6yfy3xiUuvpzqt-SU6P7VRbuOA1R37fT2JWkfQYJiE; ut=FobyicMgeV4gpb_T1OEssf3bMqTSF5vMramNZh-vF0K8egC0lexgfza65-Jp9R_rByTW0l38aV35H9MiiTCVeluxuEDuAv6HNdeecjM2k0u9jXr5YFSzIottZCvcjnLJbbNOGC9nzld6IB6Pwi0FzdqumltI1ApzCJXC8dZBtDvmDmYzF2SZDTg4oM2VCP6WE4Io2Mu3mZHIufdqGSZnKr5wKozy3NUHcaA0uLtTGJzdduDATsGp-zhYjfMkyIOZwAPP1eZclMP1banKC4htwXkhDsgmGPY3cyEssfpoUIksXXhhreVsc-8nez68UYDLao0LTfNu_wWHwQshnZ9L8nOpTrG--5YSXNpHOlNuOJb7Frn22X-xsFmh1rKPryxYKwoAal_JbITLWbH2ayu8vpXMDSAzUFQrNWvauxXeRQuhBhP-2mAAHnItBkOWqR8BnE7v7Om2OLDAN_UjgNUGy27vBTe0t54DDKPfC85ZMCEPcwWrwiK-BCLRhSqsVVoI1uiv385OXog; pi=1541025369701016%3Bt1541025369701016%3B%E8%82%A1%E5%8F%8B0Q880Q8005%3BqezFG2F0ht4141k1s0R0OwLwvajnJ86waYujhtugHtHPIQBMaShE8pr2ScjV1WL2cmPU8dbeybCSBfF%2BdFJlfu579P2Bexsaf9ZmMIMEEFXf3%2FGpajWaHl6WSNYcUby99Z43wIaladpxrBQ%2BfoduQ1bp89GMBwaR8e09TVtPeS%2BswQ43r6HXygrtrtpVH8I%2B8jWIGUpL%3B3RsdkLFOHsPp7NA%2B3oiGDzUI1bFly%2FvOzt9lBIfYhxiYH8mKjlBpstycvIQqwm87IUu%2FP5gHasJVry69bQTmOA3bqQkvC1FIzvRgZd6kMOH50R%2FNtK%2BXc%2BeRkym9XCxkSyegXmVkdRZ8RYhu9cg8K0gsdsYeKA%3D%3D; uidal=1541025369701016%e8%82%a1%e5%8f%8b0Q880Q8005; sid=128177011'

    cfg = EastMoneyConfig.from_cookies(COOKIE_STR, zjzh='260914300000052248')
    api = EastMoneyAPI(cfg)

    print("=== 用户信息 ===")
    print(json.dumps(api.get_user_info(), ensure_ascii=False, indent=2))

    print("\n=== 组合列表 ===")
    portfolios = api.list_portfolios()
    print(json.dumps(portfolios, ensure_ascii=False, indent=2))

    print("\n=== 资金查询 ===")
    bal = api.get_balance()
    print(json.dumps(bal, ensure_ascii=False, indent=2))

    print("\n=== 持仓查询 ===")
    positions = api.get_positions()
    print(json.dumps(positions, ensure_ascii=False, indent=2))

    print("\n=== 快照（兼容旧接口） ===")
    snap = api.fetch_snapshot()
    print(json.dumps(snap, ensure_ascii=False, indent=2))

    print("\n=== 最大可买（平安银行 12元） ===")
    print(json.dumps(api.get_max_buy('000001', '12.00'), ensure_ascii=False, indent=2))

    # 如果传了 --buy 参数，实际下单
    if '--buy' in sys.argv:
        print("\n=== 买入 平安银行 100股 @ 12.00 ===")
        r = api.place_order('000001', '12.00', 100, 'buy')
        print(json.dumps(r, ensure_ascii=False, indent=2))

        print("\n=== 当日委托 ===")
        print(json.dumps(api.get_today_orders(), ensure_ascii=False, indent=2))
    else:
        print("\n(加 --buy 参数实际下单测试)")
