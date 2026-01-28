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
import akshare as ak
from typing import List, Optional

logger = logging.getLogger(__name__)


def get_top_stocks_by_volume(n: int = 10) -> List[str]:
    """
    获取A股当日成交额最大的N只股票
    
    数据来源：akshare - 东方财富实时行情
    
    Args:
        n: 返回股票数量，默认10只
        
    Returns:
        股票代码列表，例如 ['600519', '000001', '300750', ...]
        失败时返回空列表 []
        
    示例：
        >>> codes = get_top_stocks_by_volume(10)
        >>> print(codes)
        ['600519', '000001', '300750', ...]
    """
    try:
        logger.info(f"🔍 正在获取A股成交额前{n}只股票...")
        
        # 获取沪深A股实时行情（包含成交额）
        # 数据列：代码、名称、最新价、涨跌幅、涨跌额、成交量、成交额、振幅、最高、最低、今开、昨收
        df = ak.stock_zh_a_spot_em()
        
        if df.empty:
            logger.warning("⚠️ 未获取到A股行情数据")
            return []
        
        # 按成交额降序排序，取前N只
        df_sorted = df.sort_values(by='成交额', ascending=False)
        top_stocks = df_sorted.head(n)
        
        # 提取股票代码
        stock_codes = top_stocks['代码'].tolist()
        
        # 打印选中的股票信息（代码、名称、成交额）
        logger.info(f"✅ 成功获取成交额前{n}只股票:")
        for idx, row in top_stocks.iterrows():
            amount_str = f"{row['成交额'] / 1e8:.2f}亿" if row['成交额'] >= 1e8 else f"{row['成交额'] / 1e4:.2f}万"
            logger.info(f"  {row['代码']} {row['名称']:8s} 成交额: {amount_str}")
        
        return stock_codes
        
    except Exception as e:
        logger.error(f"❌ 获取动态选股失败: {e}", exc_info=True)
        # 失败时返回空列表，由调用方决定后续处理（使用备选列表或退出）
        return []


def get_top_stocks_by_change(n: int = 10, exclude_st: bool = True) -> List[str]:
    """
    获取A股当日涨幅最大的N只股票（备用策略）
    
    Args:
        n: 返回股票数量
        exclude_st: 是否排除ST股票（高风险）
        
    Returns:
        股票代码列表
    """
    try:
        logger.info(f"🔍 正在获取A股涨幅前{n}只股票...")
        
        df = ak.stock_zh_a_spot_em()
        
        if df.empty:
            logger.warning("⚠️ 未获取到A股行情数据")
            return []
        
        # 排除ST股票
        if exclude_st:
            df = df[~df['名称'].str.contains('ST', na=False)]
        
        # 按涨跌幅降序排序
        df_sorted = df.sort_values(by='涨跌幅', ascending=False)
        top_stocks = df_sorted.head(n)
        
        stock_codes = top_stocks['代码'].tolist()
        
        logger.info(f"✅ 成功获取涨幅前{n}只股票:")
        for idx, row in top_stocks.iterrows():
            logger.info(f"  {row['代码']} {row['名称']:8s} 涨跌幅: {row['涨跌幅']:.2f}%")
        
        return stock_codes
        
    except Exception as e:
        logger.error(f"❌ 获取涨幅排名失败: {e}", exc_info=True)
        return []


if __name__ == "__main__":
    # 测试选股功能
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("=== 测试动态选股 ===\n")
    
    # 测试成交额选股
    stocks_volume = get_top_stocks_by_volume(5)
    print(f"\n成交额前5: {stocks_volume}\n")
    
    # 测试涨幅选股
    stocks_change = get_top_stocks_by_change(5)
    print(f"\n涨幅前5: {stocks_change}\n")
