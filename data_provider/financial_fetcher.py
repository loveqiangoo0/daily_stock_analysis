# -*- coding: utf-8 -*-
"""
===================================
财务数据获取器
===================================

职责：
1. 获取ROE（净资产收益率）
2. 获取营收/利润增长率
3. 为综合投资分析提供基本面数据

数据来源：
- 东方财富（AkShare）
- 新浪财经（备选）
"""

import logging
import time
import random
from typing import Optional, Dict, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class FinancialIndicators:
    """财务指标数据类"""
    code: str
    name: str = ""
    
    # 盈利能力
    roe: Optional[float] = None  # 净资产收益率 (%)
    roa: Optional[float] = None  # 总资产收益率 (%)
    gross_profit_margin: Optional[float] = None  # 毛利率 (%)
    net_profit_margin: Optional[float] = None  # 净利率 (%)
    
    # 增长能力
    revenue_growth: Optional[float] = None  # 营收增长率 (%)
    profit_growth: Optional[float] = None  # 净利润增长率 (%)
    revenue_growth_3y: Optional[float] = None  # 3年营收复合增长率 (%)
    profit_growth_3y: Optional[float] = None  # 3年利润复合增长率 (%)
    
    # 估值指标（可能在实时行情中已有）
    pe_ttm: Optional[float] = None  # 市盈率TTM
    pb: Optional[float] = None  # 市净率
    ps_ttm: Optional[float] = None  # 市销率TTM
    
    # 财务健康
    debt_to_asset: Optional[float] = None  # 资产负债率 (%)
    current_ratio: Optional[float] = None  # 流动比率
    
    # 元数据
    report_date: Optional[str] = None  # 财报日期
    update_time: Optional[str] = None  # 数据更新时间
    data_source: str = "unknown"
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'code': self.code,
            'name': self.name,
            'roe': self.roe,
            'roa': self.roa,
            'gross_profit_margin': self.gross_profit_margin,
            'net_profit_margin': self.net_profit_margin,
            'revenue_growth': self.revenue_growth,
            'profit_growth': self.profit_growth,
            'revenue_growth_3y': self.revenue_growth_3y,
            'profit_growth_3y': self.profit_growth_3y,
            'pe_ttm': self.pe_ttm,
            'pb': self.pb,
            'ps_ttm': self.ps_ttm,
            'debt_to_asset': self.debt_to_asset,
            'current_ratio': self.current_ratio,
            'report_date': self.report_date,
            'update_time': self.update_time,
            'data_source': self.data_source,
        }


class FinancialFetcher:
    """财务数据获取器"""
    
    def __init__(self, sleep_min: float = 1.0, sleep_max: float = 3.0):
        """
        初始化
        
        Args:
            sleep_min: 最小休眠时间（秒）
            sleep_max: 最大休眠时间（秒）
        """
        self.sleep_min = sleep_min
        self.sleep_max = sleep_max
    
    def _random_sleep(self):
        """随机休眠（防封禁）"""
        sleep_time = random.uniform(self.sleep_min, self.sleep_max)
        time.sleep(sleep_time)
    
    def get_financial_indicators(self, stock_code: str) -> Optional[FinancialIndicators]:
        """
        获取财务指标（优先使用东方财富数据）
        
        Args:
            stock_code: 股票代码
            
        Returns:
            FinancialIndicators 对象，失败返回 None
        """
        # 尝试东方财富数据源
        result = self._get_indicators_from_eastmoney(stock_code)
        
        if result is None:
            # 备选：尝试新浪财经
            logger.debug(f"[财务数据] 东财数据获取失败，尝试新浪财经...")
            result = self._get_indicators_from_sina(stock_code)
        
        return result
    
    def _get_indicators_from_eastmoney(self, stock_code: str) -> Optional[FinancialIndicators]:
        """
        从东方财富获取财务指标
        
        使用 akshare 的多个接口组合获取数据：
        1. stock_financial_abstract_ths - 同花顺财务摘要（ROE、增长率、毛利率等）
        2. stock_financial_report_sina - 利润表（备选）
        """
        try:
            import akshare as ak
            import pandas as pd
            
            self._random_sleep()
            
            logger.info(f"[财务数据] 获取 {stock_code} 的财务指标...")
            
            # 优先方法：同花顺财务摘要接口（数据最全）
            try:
                logger.debug(f"[API调用] ak.stock_financial_abstract_ths(symbol={stock_code})")
                df_abstract = ak.stock_financial_abstract_ths(symbol=stock_code)
                
                if df_abstract is not None and not df_abstract.empty:
                    logger.debug(f"[API返回] 同花顺财务摘要: {len(df_abstract)} 条记录")
                    logger.debug(f"[API返回] 列名: {df_abstract.columns.tolist()}")
                    
                    # 注意：数据是倒序的（从旧到新），取最后一条是最新数据
                    latest = df_abstract.iloc[-1]
                    
                    # 提取数据
                    indicators = FinancialIndicators(
                        code=stock_code,
                        data_source="ths_abstract",
                        report_date=str(latest.get('报告期', '')),
                    )
                    
                    # 解析各项指标
                    roe_str = latest.get('净资产收益率', None)
                    if roe_str and roe_str != False:
                        indicators.roe = self._parse_percent(str(roe_str))
                    
                    revenue_growth_str = latest.get('营业总收入同比增长率', None)
                    if revenue_growth_str and revenue_growth_str != False:
                        indicators.revenue_growth = self._parse_percent(str(revenue_growth_str))
                    
                    profit_growth_str = latest.get('净利润同比增长率', None)
                    if profit_growth_str and profit_growth_str != False:
                        indicators.profit_growth = self._parse_percent(str(profit_growth_str))
                    
                    # 额外指标
                    gross_margin_str = latest.get('销售毛利率', None)
                    if gross_margin_str and gross_margin_str != False:
                        indicators.gross_profit_margin = self._parse_percent(str(gross_margin_str))
                    
                    net_margin_str = latest.get('销售净利率', None)
                    if net_margin_str and net_margin_str != False:
                        indicators.net_profit_margin = self._parse_percent(str(net_margin_str))
                    
                    logger.info(f"[财务数据] {stock_code} 同花顺数据: ROE={indicators.roe}%, "
                              f"营收增长={indicators.revenue_growth}%, 利润增长={indicators.profit_growth}%, "
                              f"毛利率={indicators.gross_profit_margin}%")
                    return indicators
                    
            except Exception as e:
                logger.debug(f"[财务数据] 财务摘要接口失败: {e}")
            
            # 尝试方法2：财务分析指标接口
            try:
                self._random_sleep()
                logger.debug(f"[API调用] ak.stock_financial_analysis_indicator(symbol={stock_code})")
                df_indicator = ak.stock_financial_analysis_indicator(symbol=stock_code)
                
                if df_indicator is not None and not df_indicator.empty:
                    logger.debug(f"[API返回] 财务分析指标: {len(df_indicator)} 条记录")
                    logger.debug(f"[API返回] 列名: {df_indicator.columns.tolist()}")
                    
                    latest = df_indicator.iloc[0]
                    
                    indicators = FinancialIndicators(
                        code=stock_code,
                        data_source="eastmoney_indicator",
                        report_date=str(latest.get('日期', '')),
                    )
                    
                    # 尝试多种可能的列名
                    roe_keys = ['净资产收益率', 'ROE', '加权平均净资产收益率']
                    for key in roe_keys:
                        if key in latest:
                            indicators.roe = self._safe_float(latest[key])
                            break
                    
                    revenue_keys = ['营业总收入同比增长', '营业收入同比增长率', '营业收入增长率']
                    for key in revenue_keys:
                        if key in latest:
                            indicators.revenue_growth = self._safe_float(latest[key])
                            break
                    
                    profit_keys = ['净利润同比增长', '净利润同比增长率', '净利润增长率']
                    for key in profit_keys:
                        if key in latest:
                            indicators.profit_growth = self._safe_float(latest[key])
                            break
                    
                    if indicators.roe or indicators.revenue_growth or indicators.profit_growth:
                        logger.info(f"[财务数据] {stock_code} 获取成功: ROE={indicators.roe}, "
                                  f"营收增长={indicators.revenue_growth}, 利润增长={indicators.profit_growth}")
                        return indicators
                    
            except Exception as e:
                logger.debug(f"[财务数据] 财务分析指标接口失败: {e}")
            
            # 尝试方法3：利润表接口（计算增长率）
            try:
                self._random_sleep()
                logger.debug(f"[API调用] ak.stock_financial_report_sina(stock={stock_code}, symbol=利润表)")
                df_income = ak.stock_financial_report_sina(stock=stock_code, symbol="利润表")
                
                if df_income is not None and len(df_income) >= 2:
                    logger.debug(f"[API返回] 利润表数据: {len(df_income)} 条记录")
                    
                    # 取最近两期数据计算增长率
                    current = df_income.iloc[0]
                    previous = df_income.iloc[1]
                    
                    indicators = FinancialIndicators(
                        code=stock_code,
                        data_source="sina_income",
                        report_date=str(current.get('报告期', '')),
                    )
                    
                    # 计算营收增长率
                    revenue_keys = ['营业总收入', '营业收入']
                    for key in revenue_keys:
                        if key in current and key in previous:
                            current_revenue = self._safe_float(current[key])
                            previous_revenue = self._safe_float(previous[key])
                            if current_revenue and previous_revenue and previous_revenue > 0:
                                indicators.revenue_growth = ((current_revenue - previous_revenue) / previous_revenue) * 100
                                break
                    
                    # 计算利润增长率
                    profit_keys = ['净利润', '归属于母公司股东的净利润']
                    for key in profit_keys:
                        if key in current and key in previous:
                            current_profit = self._safe_float(current[key])
                            previous_profit = self._safe_float(previous[key])
                            if current_profit and previous_profit and previous_profit > 0:
                                indicators.profit_growth = ((current_profit - previous_profit) / previous_profit) * 100
                                break
                    
                    if indicators.revenue_growth or indicators.profit_growth:
                        logger.info(f"[财务数据] {stock_code} 从利润表计算: "
                                  f"营收增长={indicators.revenue_growth}, 利润增长={indicators.profit_growth}")
                        return indicators
                        
            except Exception as e:
                logger.debug(f"[财务数据] 利润表接口失败: {e}")
            
            logger.warning(f"[财务数据] {stock_code} 所有东财接口均失败")
            return None
            
        except Exception as e:
            logger.error(f"[财务数据] {stock_code} 获取失败: {e}")
            return None
    
    def _get_indicators_from_sina(self, stock_code: str) -> Optional[FinancialIndicators]:
        """从新浪财经获取财务指标（备选）"""
        # TODO: 实现新浪财经接口
        logger.debug(f"[财务数据] 新浪财经接口暂未实现")
        return None
    
    def _safe_float(self, value: Any) -> Optional[float]:
        """安全转换为float"""
        try:
            if value is None or value == '' or value == '--':
                return None
            # 去除百分号
            if isinstance(value, str):
                value = value.strip().replace('%', '').replace(',', '')
            return float(value)
        except:
            return None
    
    def _parse_percent(self, value_str: str) -> Optional[float]:
        """解析百分比字符串，如 '15.8%' -> 15.8"""
        try:
            if not value_str or value_str == '--':
                return None
            # 去除百分号和空格
            cleaned = str(value_str).strip().replace('%', '').replace(',', '')
            return float(cleaned)
        except:
            return None


if __name__ == "__main__":
    # 测试
    logging.basicConfig(level=logging.DEBUG)
    
    fetcher = FinancialFetcher()
    
    # 测试贵州茅台
    print("=" * 50)
    print("测试贵州茅台财务指标获取")
    print("=" * 50)
    
    indicators = fetcher.get_financial_indicators('600519')
    if indicators:
        print(f"获取成功:")
        print(f"  ROE: {indicators.roe}%")
        print(f"  营收增长率: {indicators.revenue_growth}%")
        print(f"  利润增长率: {indicators.profit_growth}%")
        print(f"  数据来源: {indicators.data_source}")
        print(f"  报告期: {indicators.report_date}")
    else:
        print("获取失败")
