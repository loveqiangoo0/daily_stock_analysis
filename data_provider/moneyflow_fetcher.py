# -*- coding: utf-8 -*-
"""
===================================
资金流数据获取器
===================================

职责：
1. 获取个股资金流向（主力、大单、中单、小单）
2. 获取北向资金流向（沪深港通）
3. 为综合投资分析的"资金面"提供数据

数据来源：
- Tushare Pro（主力）
- 东方财富（备选）
"""

import logging
import time
import random
from typing import Optional, Dict, Any
from dataclasses import dataclass
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


@dataclass
class MoneyFlowData:
    """资金流向数据类"""
    code: str
    name: str = ""
    trade_date: str = ""  # 交易日期 YYYYMMDD
    
    # 个股资金流向（单位：万元）
    buy_sm_amount: Optional[float] = None  # 小单买入金额
    sell_sm_amount: Optional[float] = None  # 小单卖出金额
    buy_md_amount: Optional[float] = None  # 中单买入金额
    sell_md_amount: Optional[float] = None  # 中单卖出金额
    buy_lg_amount: Optional[float] = None  # 大单买入金额
    sell_lg_amount: Optional[float] = None  # 大单卖出金额
    buy_elg_amount: Optional[float] = None  # 特大单买入金额
    sell_elg_amount: Optional[float] = None  # 特大单卖出金额
    
    # 净流入（单位：万元）
    net_mf_amount: Optional[float] = None  # 净流入金额（总计）
    net_mf_sm: Optional[float] = None  # 小单净流入
    net_mf_md: Optional[float] = None  # 中单净流入
    net_mf_lg: Optional[float] = None  # 大单净流入
    net_mf_elg: Optional[float] = None  # 特大单净流入
    
    # 主力资金（特大单+大单）
    main_net_inflow: Optional[float] = None  # 主力净流入（万元）
    main_net_inflow_rate: Optional[float] = None  # 主力净流入占比（%）
    
    # 北向资金（港资通）
    north_net_inflow: Optional[float] = None  # 北向资金净流入（万元）
    north_buy: Optional[float] = None  # 北向资金买入
    north_sell: Optional[float] = None  # 北向资金卖出
    
    # 元数据
    data_source: str = "unknown"
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'code': self.code,
            'name': self.name,
            'trade_date': self.trade_date,
            'net_mf_amount': self.net_mf_amount,
            'main_net_inflow': self.main_net_inflow,
            'main_net_inflow_rate': self.main_net_inflow_rate,
            'net_mf_lg': self.net_mf_lg,
            'net_mf_md': self.net_mf_md,
            'net_mf_sm': self.net_mf_sm,
            'north_net_inflow': self.north_net_inflow,
            'data_source': self.data_source,
        }
    
    def get_main_flow_summary(self) -> str:
        """获取主力资金流向摘要"""
        if self.main_net_inflow is None:
            return "数据缺失"
        
        # 转换为亿元
        inflow_yi = self.main_net_inflow / 10000
        
        if inflow_yi > 1:
            return f"主力净流入 {inflow_yi:.2f}亿元"
        elif inflow_yi > 0:
            return f"主力小幅流入 {self.main_net_inflow:.0f}万元"
        elif inflow_yi > -1:
            return f"主力小幅流出 {abs(self.main_net_inflow):.0f}万元"
        else:
            return f"主力净流出 {abs(inflow_yi):.2f}亿元"
    
    def get_trend_label(self) -> str:
        """获取资金流向趋势标签"""
        if self.main_net_inflow is None:
            return "未知"
        
        inflow_yi = self.main_net_inflow / 10000
        
        if inflow_yi > 0.5:
            return "持续流入"
        elif inflow_yi > 0:
            return "小幅流入"
        elif inflow_yi > -0.5:
            return "小幅流出"
        else:
            return "持续流出"


class MoneyFlowFetcher:
    """资金流数据获取器"""
    
    def __init__(self, sleep_min: float = 0.5, sleep_max: float = 2.0):
        """
        初始化
        
        Args:
            sleep_min: 最小休眠时间（秒）
            sleep_max: 最大休眠时间（秒）
        """
        self.sleep_min = sleep_min
        self.sleep_max = sleep_max
        self._tushare_api = None
        self._init_tushare()
    
    def _init_tushare(self):
        """初始化 Tushare API"""
        try:
            import sys
            import os
            
            # 添加项目根目录到 path
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            if project_root not in sys.path:
                sys.path.insert(0, project_root)
            
            from src.config import get_config
            config = get_config()
            
            if not config.tushare_token:
                logger.debug("[资金流] Tushare Token 未配置")
                return
            
            import tushare as ts
            ts.set_token(config.tushare_token)
            self._tushare_api = ts.pro_api()
            logger.debug("[资金流] Tushare API 初始化成功")
            
        except Exception as e:
            logger.debug(f"[资金流] Tushare API 初始化失败: {e}")
    
    def _random_sleep(self):
        """随机休眠（防封禁）"""
        sleep_time = random.uniform(self.sleep_min, self.sleep_max)
        time.sleep(sleep_time)
    
    def get_moneyflow(self, stock_code: str, trade_date: Optional[str] = None) -> Optional[MoneyFlowData]:
        """
        获取个股资金流向数据
        
        优先级：Tushare Pro（高质量，需600积分）> AkShare（免费）
        
        Args:
            stock_code: 股票代码（6位数字）
            trade_date: 交易日期 YYYYMMDD（默认最近交易日）
            
        Returns:
            MoneyFlowData 对象，失败返回 None
        """
        # 优先使用 Tushare（如果配置了Token且有权限）
        result = self._get_moneyflow_from_tushare(stock_code, trade_date)
        
        if result is None:
            # 备选：AkShare（免费，无需配置）
            logger.debug(f"[资金流] Tushare 获取失败，尝试 AkShare...")
            result = self._get_moneyflow_from_akshare(stock_code)
        
        return result
    
    def _get_moneyflow_from_tushare(
        self, 
        stock_code: str, 
        trade_date: Optional[str] = None
    ) -> Optional[MoneyFlowData]:
        """
        从 Tushare Pro 获取资金流数据
        
        接口：moneyflow（个股资金流向）
        文档：https://tushare.pro/document/2?doc_id=170
        
        权限要求：Tushare Pro 600积分以上
        """
        if not self._tushare_api:
            logger.debug("[资金流] Tushare API 未初始化")
            return None
        
        try:
            self._random_sleep()
            
            # 转换股票代码格式：600519 -> 600519.SH
            if stock_code.startswith(('6', '9', '5')):
                ts_code = f"{stock_code}.SH"
            else:
                ts_code = f"{stock_code}.SZ"
            
            # 如果未指定日期，使用最近3个交易日（确保能获取到数据）
            if trade_date is None:
                end_date = datetime.now().strftime('%Y%m%d')
                start_date = (datetime.now() - timedelta(days=7)).strftime('%Y%m%d')
                
                logger.debug(f"[API调用] tushare.moneyflow(ts_code={ts_code}, start_date={start_date}, end_date={end_date})")
                df = self._tushare_api.moneyflow(
                    ts_code=ts_code,
                    start_date=start_date,
                    end_date=end_date
                )
            else:
                logger.debug(f"[API调用] tushare.moneyflow(ts_code={ts_code}, trade_date={trade_date})")
                df = self._tushare_api.moneyflow(
                    ts_code=ts_code,
                    trade_date=trade_date
                )
            
            if df is None or df.empty:
                logger.warning(f"[资金流] {stock_code} 无资金流数据")
                return None
            
            logger.debug(f"[API返回] Tushare moneyflow: {len(df)} 条记录")
            logger.debug(f"[API返回] 列名: {df.columns.tolist()}")
            
            # 取最新一天的数据
            latest = df.iloc[0]
            
            # 计算主力资金（特大单+大单）
            net_lg = latest.get('net_mf_lg', 0) or 0
            net_elg = latest.get('net_mf_elg', 0) or 0
            main_net_inflow = net_lg + net_elg
            
            # 计算主力净流入占比（如果有成交额）
            main_net_inflow_rate = None
            if 'amount' in latest and latest['amount'] and latest['amount'] > 0:
                # amount 单位是千元，main_net_inflow 单位是万元
                main_net_inflow_rate = (main_net_inflow * 10) / latest['amount'] * 100
            
            data = MoneyFlowData(
                code=stock_code,
                trade_date=str(latest.get('trade_date', '')),
                buy_sm_amount=latest.get('buy_sm_amount'),
                sell_sm_amount=latest.get('sell_sm_amount'),
                buy_md_amount=latest.get('buy_md_amount'),
                sell_md_amount=latest.get('sell_md_amount'),
                buy_lg_amount=latest.get('buy_lg_amount'),
                sell_lg_amount=latest.get('sell_lg_amount'),
                buy_elg_amount=latest.get('buy_elg_amount'),
                sell_elg_amount=latest.get('sell_elg_amount'),
                net_mf_amount=latest.get('net_mf_amount'),
                net_mf_sm=latest.get('net_mf_sm'),
                net_mf_md=latest.get('net_mf_md'),
                net_mf_lg=latest.get('net_mf_lg'),
                net_mf_elg=latest.get('net_mf_elg'),
                main_net_inflow=main_net_inflow,
                main_net_inflow_rate=main_net_inflow_rate,
                data_source="tushare_moneyflow",
            )
            
            logger.info(f"[资金流] {stock_code} 获取成功: {data.get_main_flow_summary()}")
            return data
            
        except Exception as e:
            error_msg = str(e)
            
            # 检查是否是权限不足
            if '没有权限' in error_msg or '权限' in error_msg or '积分' in error_msg:
                logger.warning(f"[资金流] Tushare 权限不足（需600积分）: {e}")
            else:
                logger.error(f"[资金流] {stock_code} Tushare 获取失败: {e}")
            
            return None
    
    def _get_moneyflow_from_akshare(self, stock_code: str) -> Optional[MoneyFlowData]:
        """
        从 AkShare 获取资金流数据（免费备选方案）
        
        接口：stock_individual_fund_flow（个股资金流向）
        文档：https://akshare.akfamily.xyz/data/stock/stock.html#id170
        
        特点：免费、无需配置、数据来自东方财富网
        """
        try:
            import akshare as ak
            
            self._random_sleep()
            
            # 判断市场（沪市/深市）
            if stock_code.startswith(('6', '9', '5')):
                market = 'sh'
            else:
                market = 'sz'
            
            logger.debug(f"[API调用] ak.stock_individual_fund_flow(stock={stock_code}, market={market})")
            df = ak.stock_individual_fund_flow(stock=stock_code, market=market)
            
            if df is None or df.empty:
                logger.warning(f"[资金流] {stock_code} AkShare 返回空数据")
                return None
            
            logger.debug(f"[API返回] AkShare stock_individual_fund_flow: {len(df)} 条记录")
            logger.debug(f"[API返回] 列名: {df.columns.tolist()}")
            
            # 取最新一天的数据（第一行）
            latest = df.iloc[0]
            
            # AkShare 列名映射：
            # '主力净流入-净额' - 主力资金净流入（元）
            # '主力净流入-净占比' - 主力资金净流入占比（%）
            # '超大单净流入-净额', '大单净流入-净额', '中单净流入-净额', '小单净流入-净额'
            
            # 将元转换为万元
            main_net_inflow = latest.get('主力净流入-净额', 0) or 0
            main_net_inflow_wan = main_net_inflow / 10000  # 元 -> 万元
            
            net_elg = (latest.get('超大单净流入-净额', 0) or 0) / 10000
            net_lg = (latest.get('大单净流入-净额', 0) or 0) / 10000
            net_md = (latest.get('中单净流入-净额', 0) or 0) / 10000
            net_sm = (latest.get('小单净流入-净额', 0) or 0) / 10000
            
            data = MoneyFlowData(
                code=stock_code,
                trade_date=str(latest.get('日期', '')).replace('-', ''),  # 转换为 YYYYMMDD 格式
                net_mf_amount=main_net_inflow_wan,
                net_mf_elg=net_elg,
                net_mf_lg=net_lg,
                net_mf_md=net_md,
                net_mf_sm=net_sm,
                main_net_inflow=main_net_inflow_wan,
                main_net_inflow_rate=latest.get('主力净流入-净占比'),
                data_source="akshare_individual_flow",
            )
            
            logger.info(f"[资金流] {stock_code} AkShare获取成功: {data.get_main_flow_summary()}")
            return data
            
        except Exception as e:
            logger.error(f"[资金流] {stock_code} AkShare 获取失败: {e}")
            return None
    
    def get_north_moneyflow(self, stock_code: str, days: int = 5) -> Optional[Dict[str, Any]]:
        """
        获取北向资金流向（最近N日汇总）
        
        优先级：Tushare Pro（详细数据）> AkShare（免费）
        
        Args:
            stock_code: 股票代码
            days: 统计天数
            
        Returns:
            包含北向资金数据的字典，失败返回 None
        """
        # 优先使用 Tushare
        result = self._get_north_moneyflow_from_tushare(stock_code, days)
        
        if result is None:
            # 备选：AkShare
            logger.debug(f"[北向资金] Tushare 获取失败，尝试 AkShare...")
            result = self._get_north_moneyflow_from_akshare(stock_code, days)
        
        return result
    
    def _get_north_moneyflow_from_tushare(self, stock_code: str, days: int = 5) -> Optional[Dict[str, Any]]:
        """从 Tushare 获取北向资金（原实现）"""
        if not self._tushare_api:
            logger.debug("[北向资金] Tushare API 未初始化")
            return None
        
        try:
            self._random_sleep()
            
            # 转换股票代码格式
            if stock_code.startswith(('6', '9', '5')):
                ts_code = f"{stock_code}.SH"
            else:
                ts_code = f"{stock_code}.SZ"
            
            # 获取最近N天的北向资金数据
            end_date = datetime.now().strftime('%Y%m%d')
            start_date = (datetime.now() - timedelta(days=days*2)).strftime('%Y%m%d')
            
            logger.debug(f"[API调用] tushare.moneyflow_hsgt(ts_code={ts_code}, start_date={start_date}, end_date={end_date})")
            df = self._tushare_api.moneyflow_hsgt(
                ts_code=ts_code,
                start_date=start_date,
                end_date=end_date
            )
            
            if df is None or df.empty:
                logger.debug(f"[北向资金] {stock_code} 无北向资金数据")
                return None
            
            logger.debug(f"[API返回] Tushare moneyflow_hsgt: {len(df)} 条记录")
            
            # 取最近N条有效数据
            recent_df = df.head(min(days, len(df)))
            
            # 汇总统计
            total_net_amount = recent_df['net_amount'].sum() if 'net_amount' in recent_df else 0
            avg_net_amount = recent_df['net_amount'].mean() if 'net_amount' in recent_df else 0
            
            # 判断趋势
            if total_net_amount > 10000:  # 1亿元以上
                trend = "持续流入"
            elif total_net_amount > 0:
                trend = "小幅流入"
            elif total_net_amount > -10000:
                trend = "小幅流出"
            else:
                trend = "持续流出"
            
            result = {
                'code': stock_code,
                'days': days,
                'total_net_amount': total_net_amount,  # 单位：万元
                'avg_net_amount': avg_net_amount,  # 单位：万元
                'trend': trend,
                'data_source': 'tushare_hsgt',
            }
            
            logger.info(f"[北向资金] {stock_code} 最近{days}日: {trend}, 累计{total_net_amount/10000:.2f}亿元")
            return result
            
        except Exception as e:
            error_msg = str(e)
            
            if '没有权限' in error_msg or '权限' in error_msg:
                logger.warning(f"[北向资金] Tushare 权限不足: {e}")
            else:
                logger.debug(f"[北向资金] {stock_code} 获取失败: {e}")
            
            return None
    
    def _get_north_moneyflow_from_akshare(self, stock_code: str, days: int = 5) -> Optional[Dict[str, Any]]:
        """
        从 AkShare 获取北向资金（免费备选方案）
        
        接口：stock_hsgt_individual_em（个股北向资金历史数据）
        文档：https://akshare.akfamily.xyz/data/stock/stock.html#id219
        
        特点：免费、无需配置、数据来自东方财富网
        注意：只能获取到在沪深港通范围内的股票数据
        """
        try:
            import akshare as ak
            
            self._random_sleep()
            
            logger.debug(f"[API调用] ak.stock_hsgt_individual_em(symbol={stock_code})")
            df = ak.stock_hsgt_individual_em(symbol=stock_code)
            
            if df is None or df.empty:
                logger.debug(f"[北向资金] {stock_code} 可能不在沪深港通范围内")
                return None
            
            logger.debug(f"[API返回] AkShare stock_hsgt_individual_em: {len(df)} 条记录")
            logger.debug(f"[API返回] 列名: {df.columns.tolist()}")
            
            # 取最近N条记录
            recent_df = df.head(min(days, len(df)))
            
            # AkShare 列名：'日期', '收盘价', '涨跌幅', '北上资金-持股数', '北上资金-持股数变化', 
            #             '北上资金-持股数变化率', '北上资金-持股市值', '北上资金-占流通股比'
            
            # 计算北向资金净流入（基于持股数变化和股价）
            # 注意：这里是近似值，精确值需要成交数据
            total_change = 0
            for _, row in recent_df.iterrows():
                share_change = row.get('北上资金-持股数变化', 0) or 0
                close_price = row.get('收盘价', 0) or 0
                if share_change and close_price:
                    # 持股数变化（股）* 股价 = 资金变化（元）
                    value_change = share_change * close_price
                    total_change += value_change
            
            # 转换为万元
            total_net_amount = total_change / 10000
            avg_net_amount = total_net_amount / len(recent_df) if len(recent_df) > 0 else 0
            
            # 判断趋势
            if total_net_amount > 10000:  # 1亿元以上
                trend = "持续流入"
            elif total_net_amount > 0:
                trend = "小幅流入"
            elif total_net_amount > -10000:
                trend = "小幅流出"
            else:
                trend = "持续流出"
            
            result = {
                'code': stock_code,
                'days': days,
                'total_net_amount': total_net_amount,  # 单位：万元
                'avg_net_amount': avg_net_amount,  # 单位：万元
                'trend': trend,
                'data_source': 'akshare_hsgt',
            }
            
            logger.info(f"[北向资金] {stock_code} AkShare获取成功: {trend}, 累计{total_net_amount/10000:.2f}亿元")
            return result
            
        except Exception as e:
            logger.debug(f"[北向资金] {stock_code} AkShare 获取失败: {e}")
            return None


if __name__ == "__main__":
    # 测试
    logging.basicConfig(level=logging.DEBUG)
    
    fetcher = MoneyFlowFetcher()
    
    print("=" * 50)
    print("测试贵州茅台资金流数据获取")
    print("=" * 50)
    
    # 测试个股资金流
    data = fetcher.get_moneyflow('600519')
    if data:
        print(f"获取成功:")
        print(f"  交易日期: {data.trade_date}")
        print(f"  主力净流入: {data.main_net_inflow/10000:.2f}亿元")
        print(f"  主力净流入占比: {data.main_net_inflow_rate:.2f}%" if data.main_net_inflow_rate else "  主力净流入占比: N/A")
        print(f"  大单净流入: {data.net_mf_lg/10000:.2f}亿元" if data.net_mf_lg else "  大单净流入: N/A")
        print(f"  中单净流入: {data.net_mf_md/10000:.2f}亿元" if data.net_mf_md else "  中单净流入: N/A")
        print(f"  小单净流入: {data.net_mf_sm/10000:.2f}亿元" if data.net_mf_sm else "  小单净流入: N/A")
        print(f"  资金流向: {data.get_main_flow_summary()}")
        print(f"  数据来源: {data.data_source}")
    else:
        print("获取失败（可能是 Tushare 权限不足，需600积分）")
    
    # 测试北向资金
    print("\n" + "=" * 50)
    print("测试北向资金数据")
    print("=" * 50)
    
    north = fetcher.get_north_moneyflow('600519', days=5)
    if north:
        print(f"最近5日北向资金:")
        print(f"  累计净流入: {north['total_net_amount']/10000:.2f}亿元")
        print(f"  日均净流入: {north['avg_net_amount']/10000:.2f}亿元")
        print(f"  趋势: {north['trend']}")
    else:
        print("获取失败（可能是 Tushare 权限不足或该股不在港股通范围）")
