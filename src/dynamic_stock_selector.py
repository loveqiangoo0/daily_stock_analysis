# -*- coding: utf-8 -*-
"""
===================================
动态股票选择器 - 基于市场数据自动选股
===================================

职责：
1. 根据市场数据自动选择股票（成交额、涨幅等）
2. 支持多种选股策略
3. 提供容错机制，选股失败时返回空列表
"""
import logging
import requests
import time
from typing import List, Optional
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

logger = logging.getLogger(__name__)


# 重试装饰器：网络错误时最多重试3次，指数退避
_retry_decorator = retry(
    retry=retry_if_exception_type((requests.exceptions.RequestException, ConnectionError)),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    reraise=True
)


@_retry_decorator
def get_top_stocks_by_volume(n: int = 10) -> List[str]:
    """
    获取A股当日成交额最大的N只股票
    
    数据来源：东方财富网 API（直接HTTP请求，不依赖第三方库）
    
    Args:
        n: 返回股票数量，默认10只
        
    Returns:
        股票代码列表，例如 ['600519', '000001', '300750', ...]
        失败时返回空列表 []
        
    示例:
        >>> codes = get_top_stocks_by_volume(10)
        >>> print(codes)
        ['600519', '000001', '300750', ...]
    """
    try:
        logger.info(f"🔍 正在获取A股成交额前{n}只股票...")
        
        # 东方财富行情 API
        # pz: 每页数量
        # po: 1=降序排列
        # fid: f6=成交额排序
        # fields: f12=代码, f14=名称, f2=最新价, f6=成交额, f3=涨跌幅
        url = (
            f"http://push2.eastmoney.com/api/qt/clist/get"
            f"?pn=1&pz={n}&po=1&np=1"
            f"&ut=bd1d9ddb04089700cf9c27f6f7426281"
            f"&fltt=2&invt=2&fid=f6"
            f"&fs=m:0+t:6,m:0+t:80,m:1+t:2,m:1+t:23"
            f"&fields=f12,f14,f2,f3,f6"
        )
        
        # 直接请求，显式禁用代理（国内数据源）
        proxies = {
            'http': None,
            'https': None
        }
        # 模拟真实浏览器请求头
        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36',
            'Accept': 'application/json, text/plain, */*',
            'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Referer': 'http://quote.eastmoney.com/'
        }
        
        # 使用 session 保持连接
        session = requests.Session()
        response = session.get(url, timeout=15, proxies=proxies, headers=headers)
        response.raise_for_status()
        
        # 添加延迟，避免请求过快
        time.sleep(0.5)
        
        data = response.json()
        
        # 检查返回数据
        if data.get('rc') != 0 or not data.get('data', {}).get('diff'):
            logger.warning("⚠️ 未获取到A股行情数据")
            return []
        
        stocks = data['data']['diff']
        
        if not stocks:
            logger.warning("⚠️ 行情数据为空")
            return []
        
        # 提取股票代码和信息
        stock_codes = []
        logger.info(f"✅ 成功获取成交额前{len(stocks)}只股票:")
        
        for stock in stocks:
            code = stock.get('f12', '')  # 股票代码
            name = stock.get('f14', '')  # 股票名称
            volume = stock.get('f6', 0)  # 成交额
            
            if code:
                stock_codes.append(code)
                
                # 格式化成交额显示
                if volume >= 1e8:
                    amount_str = f"{volume / 1e8:.2f}亿"
                else:
                    amount_str = f"{volume / 1e4:.2f}万"
                
                logger.info(f"  {code} {name:8s} 成交额: {amount_str}")
        
        return stock_codes
        
    except requests.exceptions.RequestException as e:
        logger.error(f"❌ 网络请求失败: {e}")
        return []
    except Exception as e:
        logger.error(f"❌ 获取动态选股失败: {e}", exc_info=True)
        return []


@_retry_decorator
def get_top_stocks_by_change(n: int = 10, exclude_st: bool = True) -> List[str]:
    """
    获取A股当日涨幅最大的N只股票（备用策略）
    
    数据来源：东方财富网 API
    
    Args:
        n: 返回股票数量
        exclude_st: 是否排除ST股票（高风险）
        
    Returns:
        股票代码列表
    """
    try:
        logger.info(f"🔍 正在获取A股涨幅前{n * 2}只股票（将过滤ST后取前{n}只）...")
        
        # 东方财富行情 API
        # fid: f3=涨跌幅排序
        url = (
            f"http://push2.eastmoney.com/api/qt/clist/get"
            f"?pn=1&pz={n * 2}&po=1&np=1"
            f"&ut=bd1d9ddb04089700cf9c27f6f7426281"
            f"&fltt=2&invt=2&fid=f3"
            f"&fs=m:0+t:6,m:0+t:80,m:1+t:2,m:1+t:23"
            f"&fields=f12,f14,f2,f3,f6"
        )
        
        # 显式禁用代理（国内数据源）
        proxies = {
            'http': None,
            'https': None
        }
        # 模拟真实浏览器请求头
        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36',
            'Accept': 'application/json, text/plain, */*',
            'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Referer': 'http://quote.eastmoney.com/'
        }
        
        # 使用 session 保持连接
        session = requests.Session()
        response = session.get(url, timeout=15, proxies=proxies, headers=headers)
        response.raise_for_status()
        
        # 添加延迟，避免请求过快
        time.sleep(0.5)
        
        data = response.json()
        
        if data.get('rc') != 0 or not data.get('data', {}).get('diff'):
            logger.warning("⚠️ 未获取到A股行情数据")
            return []
        
        stocks = data['data']['diff']
        
        if not stocks:
            logger.warning("⚠️ 行情数据为空")
            return []
        
        # 提取股票代码，过滤ST
        stock_codes = []
        logger.info(f"✅ 成功获取涨幅前{n}只股票:")
        
        for stock in stocks:
            code = stock.get('f12', '')
            name = stock.get('f14', '')
            change = stock.get('f3', 0)  # 涨跌幅
            
            # 排除ST股票
            if exclude_st and name and 'ST' in name:
                continue
            
            if code:
                stock_codes.append(code)
                logger.info(f"  {code} {name:8s} 涨跌幅: {change:.2f}%")
                
                # 达到目标数量就停止
                if len(stock_codes) >= n:
                    break
        
        return stock_codes
        
    except requests.exceptions.RequestException as e:
        logger.error(f"❌ 网络请求失败: {e}")
        return []
    except Exception as e:
        logger.error(f"❌ 获取涨幅排名失败: {e}", exc_info=True)
        return []


if __name__ == "__main__":
    # 测试选股功能
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("=" * 60)
    print("测试动态选股功能 - 东方财富API直连")
    print("=" * 60)
    print()
    
    # 测试成交额选股
    print("【测试1】成交额排名")
    stocks_volume = get_top_stocks_by_volume(5)
    print(f"\n结果: {stocks_volume}")
    print()
    
    # 测试涨幅选股
    print("【测试2】涨幅排名")
    stocks_change = get_top_stocks_by_change(5)
    print(f"\n结果: {stocks_change}")
    print()
    
    print("=" * 60)
    print("测试完成")
    print("=" * 60)
