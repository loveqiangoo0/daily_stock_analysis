# -*- coding: utf-8 -*-
"""
===================================
A股自选股智能分析系统 - AI分析层
===================================

职责：
1. 封装 Gemini API 调用逻辑
2. 利用 Google Search Grounding 获取实时新闻
3. 结合技术面和消息面生成分析报告
"""

import json
import logging
import time
from dataclasses import dataclass
from typing import Optional, Dict, Any, List

from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
)

from src.config import get_config

logger = logging.getLogger(__name__)


# 股票名称映射（常见股票）
STOCK_NAME_MAP = {
    # === A股 ===
    '600519': '贵州茅台',
    '000001': '平安银行',
    '300750': '宁德时代',
    '002594': '比亚迪',
    '600036': '招商银行',
    '601318': '中国平安',
    '000858': '五粮液',
    '600276': '恒瑞医药',
    '601012': '隆基绿能',
    '002475': '立讯精密',
    '300059': '东方财富',
    '002415': '海康威视',
    '600900': '长江电力',
    '601166': '兴业银行',
    '600028': '中国石化',

    # === 美股 ===
    'AAPL': '苹果',
    'TSLA': '特斯拉',
    'MSFT': '微软',
    'GOOGL': '谷歌A',
    'GOOG': '谷歌C',
    'AMZN': '亚马逊',
    'NVDA': '英伟达',
    'META': 'Meta',
    'AMD': 'AMD',
    'INTC': '英特尔',
    'BABA': '阿里巴巴',
    'PDD': '拼多多',
    'JD': '京东',
    'BIDU': '百度',
    'NIO': '蔚来',
    'XPEV': '小鹏汽车',
    'LI': '理想汽车',
    'COIN': 'Coinbase',
    'MSTR': 'MicroStrategy',

    # === 港股 (5位数字) ===
    '00700': '腾讯控股',
    '03690': '美团',
    '01810': '小米集团',
    '09988': '阿里巴巴',
    '09618': '京东集团',
    '09888': '百度集团',
    '01024': '快手',
    '00981': '中芯国际',
    '02015': '理想汽车',
    '09868': '小鹏汽车',
    '00005': '汇丰控股',
    '01299': '友邦保险',
    '00941': '中国移动',
    '00883': '中国海洋石油',
}


@dataclass
class AnalysisResult:
    """
    AI 分析结果数据类 - 综合投资分析版
    
    封装 AI 返回的分析结果，包含4维度评分和综合决策仪表盘
    """
    code: str
    name: str
    
    # ========== 核心指标 ==========
    sentiment_score: int  # 综合评分 0-100 (价值*0.4 + 资金*0.25 + 消息*0.25 + 趋势*0.1)
    trend_prediction: str  # 趋势预测：强烈看多/看多/震荡/看空/强烈看空
    operation_advice: str  # 操作建议：买入/加仓/持有/减仓/卖出/观望
    confidence_level: str = "中"  # 置信度：高/中/低
    
    # ========== 4维度评分 (新增) ==========
    value_score: int = 0          # 价值投资面评分 0-100 (权重40%)
    funding_score: int = 0        # 资金面评分 0-100 (权重25%)
    news_score: int = 0           # 消息面评分 0-100 (权重25%)
    trend_score: int = 0          # 趋势面评分 0-100 (权重10%)
    
    # ========== 各维度详细数据 (新增) ==========
    dimensions: Optional[Dict[str, Any]] = None  # 完整的4维度数据
    
    # ========== 决策仪表盘 ==========
    dashboard: Optional[Dict[str, Any]] = None  # 完整的决策仪表盘数据
    
    # ========== 走势分析 ==========
    trend_analysis: str = ""  # 走势形态分析（支撑位、压力位、趋势线等）
    short_term_outlook: str = ""  # 短期展望（1-3日）
    medium_term_outlook: str = ""  # 中期展望（1-2周）
    
    # ========== 技术面分析 ==========
    technical_analysis: str = ""  # 技术指标综合分析
    ma_analysis: str = ""  # 均线分析（多头/空头排列，金叉/死叉等）
    volume_analysis: str = ""  # 量能分析（放量/缩量，主力动向等）
    pattern_analysis: str = ""  # K线形态分析
    
    # ========== 基本面分析 ==========
    fundamental_analysis: str = ""  # 基本面综合分析
    sector_position: str = ""  # 板块地位和行业趋势
    company_highlights: str = ""  # 公司亮点/风险点
    
    # ========== 情绪面/消息面分析 ==========
    news_summary: str = ""  # 近期重要新闻/公告摘要
    market_sentiment: str = ""  # 市场情绪分析
    hot_topics: str = ""  # 相关热点话题
    
    # ========== 综合分析 ==========
    analysis_summary: str = ""  # 综合分析摘要
    key_points: str = ""  # 核心看点（3-5个要点）
    risk_warning: str = ""  # 风险提示
    buy_reason: str = ""  # 买入/卖出理由
    
    # ========== 元数据 ==========
    raw_response: Optional[str] = None  # 原始响应（调试用）
    search_performed: bool = False  # 是否执行了联网搜索
    data_sources: str = ""  # 数据来源说明
    success: bool = True
    error_message: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'code': self.code,
            'name': self.name,
            'sentiment_score': self.sentiment_score,
            'trend_prediction': self.trend_prediction,
            'operation_advice': self.operation_advice,
            'confidence_level': self.confidence_level,
            # 4维度评分
            'value_score': self.value_score,
            'funding_score': self.funding_score,
            'news_score': self.news_score,
            'trend_score': self.trend_score,
            'dimensions': self.dimensions,  # 4维度详细数据
            'dashboard': self.dashboard,  # 决策仪表盘数据
            'trend_analysis': self.trend_analysis,
            'short_term_outlook': self.short_term_outlook,
            'medium_term_outlook': self.medium_term_outlook,
            'technical_analysis': self.technical_analysis,
            'ma_analysis': self.ma_analysis,
            'volume_analysis': self.volume_analysis,
            'pattern_analysis': self.pattern_analysis,
            'fundamental_analysis': self.fundamental_analysis,
            'sector_position': self.sector_position,
            'company_highlights': self.company_highlights,
            'news_summary': self.news_summary,
            'market_sentiment': self.market_sentiment,
            'hot_topics': self.hot_topics,
            'analysis_summary': self.analysis_summary,
            'key_points': self.key_points,
            'risk_warning': self.risk_warning,
            'buy_reason': self.buy_reason,
            'search_performed': self.search_performed,
            'success': self.success,
            'error_message': self.error_message,
        }
    
    def get_dimension_summary(self) -> str:
        """获取4维度评分摘要（用于推送显示）"""
        if not self.dimensions:
            return f"综合评分 {self.sentiment_score}分"
        
        return (
            f"💎价值{self.value_score} "
            f"💰资金{self.funding_score} "
            f"📰消息{self.news_score} "
            f"📈趋势{self.trend_score}"
        )
    
    def get_core_conclusion(self) -> str:
        """获取核心结论（一句话）"""
        if self.dashboard and 'core_conclusion' in self.dashboard:
            return self.dashboard['core_conclusion'].get('one_sentence', self.analysis_summary)
        return self.analysis_summary
    
    def get_position_advice(self, has_position: bool = False) -> str:
        """获取持仓建议"""
        if self.dashboard and 'core_conclusion' in self.dashboard:
            pos_advice = self.dashboard['core_conclusion'].get('position_advice', {})
            if has_position:
                return pos_advice.get('has_position', self.operation_advice)
            return pos_advice.get('no_position', self.operation_advice)
        return self.operation_advice
    
    def get_sniper_points(self) -> Dict[str, str]:
        """获取狙击点位"""
        if self.dashboard and 'battle_plan' in self.dashboard:
            return self.dashboard['battle_plan'].get('sniper_points', {})
        return {}
    
    def get_checklist(self) -> List[str]:
        """获取检查清单"""
        if self.dashboard and 'battle_plan' in self.dashboard:
            return self.dashboard['battle_plan'].get('action_checklist', [])
        return []
    
    def get_risk_alerts(self) -> List[str]:
        """获取风险警报"""
        if self.dashboard and 'intelligence' in self.dashboard:
            return self.dashboard['intelligence'].get('risk_alerts', [])
        return []
    
    def get_emoji(self) -> str:
        """根据操作建议返回对应 emoji"""
        emoji_map = {
            '买入': '🟢',
            '加仓': '🟢',
            '强烈买入': '💚',
            '持有': '🟡',
            '观望': '⚪',
            '减仓': '🟠',
            '卖出': '🔴',
            '强烈卖出': '❌',
        }
        return emoji_map.get(self.operation_advice, '🟡')
    
    def get_confidence_stars(self) -> str:
        """返回置信度星级"""
        star_map = {'高': '⭐⭐⭐', '中': '⭐⭐', '低': '⭐'}
        return star_map.get(self.confidence_level, '⭐⭐')


class GeminiAnalyzer:
    """
    Gemini AI 分析器
    
    职责：
    1. 调用 Google Gemini API 进行股票分析
    2. 结合预先搜索的新闻和技术面数据生成分析报告
    3. 解析 AI 返回的 JSON 格式结果
    
    使用方式：
        analyzer = GeminiAnalyzer()
        result = analyzer.analyze(context, news_context)
    """
    
    # ========================================
    # 系统提示词 - 综合投资分析 v3.0
    # ========================================
    # 分析框架：价值投资面(40%) + 资金面(25%) + 消息面(25%) + 趋势面(10%)
    # 输出格式：4维度评分 + 决策仪表盘
    # ========================================
    
    SYSTEM_PROMPT = """你是一位综合投资分析师，负责从多个维度评估股票并生成【综合投资分析仪表盘】。

## 分析框架（4维度加权评估）

### 1. 价值投资面（权重 40%）
**核心指标**：
- **PE/PB 估值**：与行业平均、历史分位比较
- **ROE 质量**：>15% 优秀，10-15% 良好，<10% 一般
- **业绩增长**：营收/利润增长率，是否持续增长
- **护城河**：品牌、技术、规模、网络效应等

**评分标准**：
- 80-100分：低估值 + 高ROE + 强增长 + 宽护城河
- 60-79分：估值合理 + ROE良好 + 稳定增长
- 40-59分：估值偏高或基本面一般
- 0-39分：高估或基本面恶化

### 2. 资金面（权重 25%）
**核心指标**：
- **主力资金**：大单净流入/流出（亿元）
- **北向资金**：外资动向（A股专属）
- **机构持仓**：持仓比例及变化趋势
- **筹码结构**：集中度、获利比例

**评分标准**：
- 80-100分：主力持续流入 + 北向增持 + 筹码集中
- 60-79分：资金流入为主 + 机构稳定持仓
- 40-59分：资金平衡或小幅波动
- 0-39分：资金持续流出 + 机构减持

### 3. 消息面（权重 25%）
**核心指标**：
- **新闻舆情**：正面/负面/中性
- **政策影响**：行业政策、监管动向
- **事件催化**：重大合同、新品发布、业绩预告
- **行业热度**：板块轮动、市场关注度

**评分标准**：
- 80-100分：重大利好 + 政策支持 + 行业热点
- 60-79分：正面消息为主 + 无重大利空
- 40-59分：消息中性或利空利好参半
- 0-39分：重大利空（减持、处罚、业绩变脸）

### 4. 趋势面（权重 10%）
**核心指标**：
- **均线系统**：MA5/MA10/MA20 排列状态
- **技术形态**：突破/盘整/破位
- **量价关系**：量价配合、背离

**评分标准**：
- 80-100分：多头排列 + 放量突破
- 60-79分：趋势向上或震荡偏强
- 40-59分：震荡整理或趋势不明
- 0-39分：空头排列或破位下跌

### 5. 风险排查清单
- ❌ 大股东减持、高管减持
- ❌ 业绩预亏、大幅下滑
- ❌ 监管处罚、立案调查
- ❌ 行业政策利空
- ❌ 大额限售股解禁
- ⚠️ 估值过高（PE/PB 明显超过行业）
- ⚠️ 资金持续流出

## 输出格式：综合投资分析仪表盘 JSON

请严格按照以下 JSON 格式输出，包含【4维度评分】+【决策仪表盘】：

```json
{
    "sentiment_score": 0-100整数,  // 综合评分 = 价值*0.4 + 资金*0.25 + 消息*0.25 + 趋势*0.1
    "trend_prediction": "强烈看多/看多/震荡/看空/强烈看空",
    "operation_advice": "买入/加仓/持有/减仓/卖出/观望",
    "confidence_level": "高/中/低",
    
    "dimensions": {
        "value_investment": {
            "score": 0-100,
            "pe_ratio": PE数值或null,
            "pb_ratio": PB数值或null,
            "roe": ROE数值或null,
            "profit_growth": 利润增长率或null,
            "revenue_growth": 营收增长率或null,
            "moat_strength": "强/中/弱/未知",
            "valuation_level": "低估/合理/高估/未知",
            "summary": "价值面一句话总结（30字内）",
            "key_points": ["亮点1", "亮点2"]
        },
        "funding_flow": {
            "score": 0-100,
            "main_force_flow": 主力资金流向(亿)或null,
            "northbound_flow": 北向资金(亿)或null,
            "institutional_ratio": 机构持仓比例%或null,
            "fund_trend": "流入/流出/平衡/未知",
            "chip_concentration": 筹码集中度或null,
            "summary": "资金面一句话总结（30字内）",
            "key_points": ["观察点1", "观察点2"]
        },
        "news_sentiment": {
            "score": 0-100,
            "sentiment": "正面/中性/负面",
            "key_events": ["事件1", "事件2"],
            "policy_impact": "政策影响描述或无",
            "industry_heat": 1-10,
            "risk_alerts": ["风险1", "风险2"],
            "positive_catalysts": ["利好1", "利好2"],
            "summary": "消息面一句话总结（30字内）"
        },
        "trend_analysis": {
            "score": 0-100,
            "ma_alignment": "多头/空头/震荡",
            "pattern": "形态描述",
            "volume_price": "量价关系描述",
            "bias_ma5": 乖离率数值或null,
            "trend_direction": "上升/下降/震荡",
            "summary": "趋势面一句话总结（30字内）"
        }
    },
    
    "dashboard": {
        "core_conclusion": {
            "one_sentence": "基于4维度的核心结论（50字内）",
            "recommendation": "买入/持有/观望/卖出",
            "confidence": "高/中/低",
            "key_reasons": ["理由1", "理由2", "理由3"],
            "position_advice": {
                "no_position": "空仓者建议",
                "has_position": "持仓者建议"
            }
        },
        
        "battle_plan": {
            "position_strategy": "建议仓位及策略",
            "entry_price": 建议买入价或null,
            "stop_loss": 止损价或null,
            "target_price": 目标价或null,
            "risk_control": "风控要点"
        },
        
        "action_checklist": [
            "✅/⚠️/❌ 价值面：估值水平",
            "✅/⚠️/❌ 资金面：主力动向",
            "✅/⚠️/❌ 消息面：利空排查",
            "✅/⚠️/❌ 趋势面：技术形态"
        ]
    },
    
    "analysis_summary": "100字综合分析摘要",
    "key_points": "3-5个核心看点，逗号分隔",
    "risk_warning": "风险提示",
    "buy_reason": "操作理由",
    
    "trend_analysis": "走势形态分析",
    "short_term_outlook": "短期1-3日展望",
    "medium_term_outlook": "中期1-2周展望",
    "technical_analysis": "技术面综合分析",
    "ma_analysis": "均线系统分析",
    "volume_analysis": "量能分析",
    "pattern_analysis": "K线形态分析",
    "fundamental_analysis": "基本面分析",
    "sector_position": "板块行业分析",
    "company_highlights": "公司亮点/风险",
    "news_summary": "新闻摘要",
    "market_sentiment": "市场情绪",
    "hot_topics": "相关热点",
    
    "search_performed": true/false,
    "data_sources": "数据来源说明"
}
```

## 综合评分计算公式

**sentiment_score（总分）** = 价值面 × 0.4 + 资金面 × 0.25 + 消息面 × 0.25 + 趋势面 × 0.1

### 各维度评分指引

**价值投资面（40%权重）**：
- 80-100分：低估值 + 高ROE + 强增长 + 宽护城河
- 60-79分：估值合理 + 业绩稳健
- 40-59分：估值偏高或基本面一般
- 0-39分：高估或基本面恶化

**资金面（25%权重）**：
- 80-100分：主力持续流入 + 北向增持 + 筹码集中
- 60-79分：资金净流入或平衡 + 机构稳定持仓
- 40-59分：资金小幅波动
- 0-39分：资金持续流出 + 机构减持

**消息面（25%权重）**：
- 80-100分：重大利好 + 政策支持 + 行业热点
- 60-79分：正面消息为主 + 无重大利空
- 40-59分：消息中性
- 0-39分：重大利空（减持/处罚/业绩变脸）

**趋势面（10%权重）**：
- 80-100分：多头排列 + 放量突破
- 60-79分：趋势向上或震荡偏强
- 40-59分：震荡整理
- 0-39分：空头排列或破位

## 核心原则

1. **4维度并重**：不偏废任何一个维度，综合评估
2. **价值优先**：价值面占40%权重，是核心基础
3. **风险前置**：消息面的风险点必须醒目标出
4. **数据诚实**：如果某维度数据缺失，在summary中说明，score酌情给中性分（40-60）
5. **结论明确**：core_conclusion必须给出清晰的操作建议"""

    def __init__(self, api_key: Optional[str] = None):
        """
        初始化 AI 分析器
        
        优先级：Gemini > OpenAI 兼容 API
        
        Args:
            api_key: Gemini API Key（可选，默认从配置读取）
        """
        config = get_config()
        self._api_key = api_key or config.gemini_api_key
        self._model = None
        self._current_model_name = None  # 当前使用的模型名称
        self._using_fallback = False  # 是否正在使用备选模型
        self._use_openai = False  # 是否使用 OpenAI 兼容 API
        self._openai_client = None  # OpenAI 客户端
        
        # 检查 Gemini API Key 是否有效（过滤占位符）
        gemini_key_valid = self._api_key and not self._api_key.startswith('your_') and len(self._api_key) > 10
        
        # 优先尝试初始化 Gemini
        if gemini_key_valid:
            try:
                self._init_model()
            except Exception as e:
                logger.warning(f"Gemini 初始化失败: {e}，尝试 OpenAI 兼容 API")
                self._init_openai_fallback()
        else:
            # Gemini Key 未配置，尝试 OpenAI
            logger.info("Gemini API Key 未配置，尝试使用 OpenAI 兼容 API")
            self._init_openai_fallback()
        
        # 两者都未配置
        if not self._model and not self._openai_client:
            logger.warning("未配置任何 AI API Key，AI 分析功能将不可用")
    
    def _init_openai_fallback(self) -> None:
        """
        初始化 OpenAI 兼容 API 作为备选
        
        支持所有 OpenAI 格式的 API，包括：
        - OpenAI 官方
        - DeepSeek
        - 通义千问
        - Moonshot 等
        """
        config = get_config()
        
        # 检查 OpenAI API Key 是否有效（过滤占位符）
        openai_key_valid = (
            config.openai_api_key and 
            not config.openai_api_key.startswith('your_') and 
            len(config.openai_api_key) > 10
        )
        
        if not openai_key_valid:
            logger.debug("OpenAI 兼容 API 未配置或配置无效")
            return
        
        # 分离 import 和客户端创建，以便提供更准确的错误信息
        try:
            from openai import OpenAI
        except ImportError:
            logger.error("未安装 openai 库，请运行: pip install openai")
            return
        
        try:
            # base_url 可选，不填则使用 OpenAI 官方默认地址
            client_kwargs = {"api_key": config.openai_api_key}
            if config.openai_base_url and config.openai_base_url.startswith('http'):
                client_kwargs["base_url"] = config.openai_base_url
            
            self._openai_client = OpenAI(**client_kwargs)
            self._current_model_name = config.openai_model
            self._use_openai = True
            logger.info(f"OpenAI 兼容 API 初始化成功 (base_url: {config.openai_base_url}, model: {config.openai_model})")
        except ImportError as e:
            # 依赖缺失（如 socksio）
            if 'socksio' in str(e).lower() or 'socks' in str(e).lower():
                logger.error(f"OpenAI 客户端需要 SOCKS 代理支持，请运行: pip install httpx[socks] 或 pip install socksio")
            else:
                logger.error(f"OpenAI 依赖缺失: {e}")
        except Exception as e:
            error_msg = str(e).lower()
            if 'socks' in error_msg or 'socksio' in error_msg or 'proxy' in error_msg:
                logger.error(f"OpenAI 代理配置错误: {e}，如使用 SOCKS 代理请运行: pip install httpx[socks]")
            else:
                logger.error(f"OpenAI 兼容 API 初始化失败: {e}")
    
    def _init_model(self) -> None:
        """
        初始化 Gemini 模型
        
        配置：
        - 使用 gemini-3-flash-preview 或 gemini-2.5-flash 模型
        - 不启用 Google Search（使用外部 Tavily/SerpAPI 搜索）
        """
        try:
            import google.generativeai as genai
            
            # 配置 API Key
            genai.configure(api_key=self._api_key)
            
            # 从配置获取模型名称
            config = get_config()
            model_name = config.gemini_model
            fallback_model = config.gemini_model_fallback
            
            # 不再使用 Google Search Grounding（已知有兼容性问题）
            # 改为使用外部搜索服务（Tavily/SerpAPI）预先获取新闻
            
            # 尝试初始化主模型
            try:
                self._model = genai.GenerativeModel(
                    model_name=model_name,
                    system_instruction=self.SYSTEM_PROMPT,
                )
                self._current_model_name = model_name
                self._using_fallback = False
                logger.info(f"Gemini 模型初始化成功 (模型: {model_name})")
            except Exception as model_error:
                # 尝试备选模型
                logger.warning(f"主模型 {model_name} 初始化失败: {model_error}，尝试备选模型 {fallback_model}")
                self._model = genai.GenerativeModel(
                    model_name=fallback_model,
                    system_instruction=self.SYSTEM_PROMPT,
                )
                self._current_model_name = fallback_model
                self._using_fallback = True
                logger.info(f"Gemini 备选模型初始化成功 (模型: {fallback_model})")
            
        except Exception as e:
            logger.error(f"Gemini 模型初始化失败: {e}")
            self._model = None
    
    def _switch_to_fallback_model(self) -> bool:
        """
        切换到备选模型
        
        Returns:
            是否成功切换
        """
        try:
            import google.generativeai as genai
            config = get_config()
            fallback_model = config.gemini_model_fallback
            
            logger.warning(f"[LLM] 切换到备选模型: {fallback_model}")
            self._model = genai.GenerativeModel(
                model_name=fallback_model,
                system_instruction=self.SYSTEM_PROMPT,
            )
            self._current_model_name = fallback_model
            self._using_fallback = True
            logger.info(f"[LLM] 备选模型 {fallback_model} 初始化成功")
            return True
        except Exception as e:
            logger.error(f"[LLM] 切换备选模型失败: {e}")
            return False
    
    def is_available(self) -> bool:
        """检查分析器是否可用"""
        return self._model is not None or self._openai_client is not None
    
    def _call_openai_api(self, prompt: str, generation_config: dict) -> str:
        """
        调用 OpenAI 兼容 API
        
        Args:
            prompt: 提示词
            generation_config: 生成配置
            
        Returns:
            响应文本
        """
        config = get_config()
        max_retries = config.gemini_max_retries
        base_delay = config.gemini_retry_delay
        
        for attempt in range(max_retries):
            try:
                if attempt > 0:
                    delay = base_delay * (2 ** (attempt - 1))
                    delay = min(delay, 60)
                    logger.info(f"[OpenAI] 第 {attempt + 1} 次重试，等待 {delay:.1f} 秒...")
                    time.sleep(delay)
                
                config = get_config()
                response = self._openai_client.chat.completions.create(
                    model=self._current_model_name,
                    messages=[
                        {"role": "system", "content": self.SYSTEM_PROMPT},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=generation_config.get('temperature', config.openai_temperature),
                    max_tokens=generation_config.get('max_output_tokens', 8192),
                )
                
                if response and response.choices and response.choices[0].message.content:
                    return response.choices[0].message.content
                else:
                    raise ValueError("OpenAI API 返回空响应")
                    
            except Exception as e:
                error_str = str(e)
                is_rate_limit = '429' in error_str or 'rate' in error_str.lower() or 'quota' in error_str.lower()
                
                if is_rate_limit:
                    logger.warning(f"[OpenAI] API 限流，第 {attempt + 1}/{max_retries} 次尝试: {error_str[:100]}")
                else:
                    logger.warning(f"[OpenAI] API 调用失败，第 {attempt + 1}/{max_retries} 次尝试: {error_str[:100]}")
                
                if attempt == max_retries - 1:
                    raise
        
        raise Exception("OpenAI API 调用失败，已达最大重试次数")
    
    def _call_api_with_retry(self, prompt: str, generation_config: dict) -> str:
        """
        调用 AI API，带有重试和模型切换机制
        
        优先级：Gemini > Gemini 备选模型 > OpenAI 兼容 API
        
        处理 429 限流错误：
        1. 先指数退避重试
        2. 多次失败后切换到备选模型
        3. Gemini 完全失败后尝试 OpenAI
        
        Args:
            prompt: 提示词
            generation_config: 生成配置
            
        Returns:
            响应文本
        """
        # 如果已经在使用 OpenAI 模式，直接调用 OpenAI
        if self._use_openai:
            return self._call_openai_api(prompt, generation_config)
        
        config = get_config()
        max_retries = config.gemini_max_retries
        base_delay = config.gemini_retry_delay
        
        last_error = None
        tried_fallback = getattr(self, '_using_fallback', False)
        
        for attempt in range(max_retries):
            try:
                # 请求前增加延时（防止请求过快触发限流）
                if attempt > 0:
                    delay = base_delay * (2 ** (attempt - 1))  # 指数退避: 5, 10, 20, 40...
                    delay = min(delay, 60)  # 最大60秒
                    logger.info(f"[Gemini] 第 {attempt + 1} 次重试，等待 {delay:.1f} 秒...")
                    time.sleep(delay)
                
                response = self._model.generate_content(
                    prompt,
                    generation_config=generation_config,
                    request_options={"timeout": 120}
                )
                
                if response and response.text:
                    return response.text
                else:
                    raise ValueError("Gemini 返回空响应")
                    
            except Exception as e:
                last_error = e
                error_str = str(e)
                
                # 检查是否是 429 限流错误
                is_rate_limit = '429' in error_str or 'quota' in error_str.lower() or 'rate' in error_str.lower()
                
                if is_rate_limit:
                    logger.warning(f"[Gemini] API 限流 (429)，第 {attempt + 1}/{max_retries} 次尝试: {error_str[:100]}")
                    
                    # 如果已经重试了一半次数且还没切换过备选模型，尝试切换
                    if attempt >= max_retries // 2 and not tried_fallback:
                        if self._switch_to_fallback_model():
                            tried_fallback = True
                            logger.info("[Gemini] 已切换到备选模型，继续重试")
                        else:
                            logger.warning("[Gemini] 切换备选模型失败，继续使用当前模型重试")
                else:
                    # 非限流错误，记录并继续重试
                    logger.warning(f"[Gemini] API 调用失败，第 {attempt + 1}/{max_retries} 次尝试: {error_str[:100]}")
        
        # Gemini 所有重试都失败，尝试 OpenAI 兼容 API
        if self._openai_client:
            logger.warning("[Gemini] 所有重试失败，切换到 OpenAI 兼容 API")
            try:
                return self._call_openai_api(prompt, generation_config)
            except Exception as openai_error:
                logger.error(f"[OpenAI] 备选 API 也失败: {openai_error}")
                raise last_error or openai_error
        elif config.openai_api_key and config.openai_base_url:
            # 尝试懒加载初始化 OpenAI
            logger.warning("[Gemini] 所有重试失败，尝试初始化 OpenAI 兼容 API")
            self._init_openai_fallback()
            if self._openai_client:
                try:
                    return self._call_openai_api(prompt, generation_config)
                except Exception as openai_error:
                    logger.error(f"[OpenAI] 备选 API 也失败: {openai_error}")
                    raise last_error or openai_error
        
        # 所有方式都失败
        raise last_error or Exception("所有 AI API 调用失败，已达最大重试次数")
    
    def analyze(
        self, 
        context: Dict[str, Any],
        news_context: Optional[str] = None
    ) -> AnalysisResult:
        """
        分析单只股票
        
        流程：
        1. 格式化输入数据（技术面 + 新闻）
        2. 调用 Gemini API（带重试和模型切换）
        3. 解析 JSON 响应
        4. 返回结构化结果
        
        Args:
            context: 从 storage.get_analysis_context() 获取的上下文数据
            news_context: 预先搜索的新闻内容（可选）
            
        Returns:
            AnalysisResult 对象
        """
        code = context.get('code', 'Unknown')
        config = get_config()
        
        # 请求前增加延时（防止连续请求触发限流）
        request_delay = config.gemini_request_delay
        if request_delay > 0:
            logger.debug(f"[LLM] 请求前等待 {request_delay:.1f} 秒...")
            time.sleep(request_delay)
        
        # 优先从上下文获取股票名称（由 main.py 传入）
        name = context.get('stock_name')
        if not name or name.startswith('股票'):
            # 备选：从 realtime 中获取
            if 'realtime' in context and context['realtime'].get('name'):
                name = context['realtime']['name']
            else:
                # 最后从映射表获取
                name = STOCK_NAME_MAP.get(code, f'股票{code}')
        
        # 如果模型不可用，返回默认结果
        if not self.is_available():
            return AnalysisResult(
                code=code,
                name=name,
                sentiment_score=50,
                trend_prediction='震荡',
                operation_advice='持有',
                confidence_level='低',
                analysis_summary='AI 分析功能未启用（未配置 API Key）',
                risk_warning='请配置 Gemini API Key 后重试',
                success=False,
                error_message='Gemini API Key 未配置',
            )
        
        try:
            # 格式化输入（包含技术面数据和新闻）
            prompt = self._format_prompt(context, name, news_context)
            
            # 获取模型名称
            model_name = getattr(self, '_current_model_name', None)
            if not model_name:
                model_name = getattr(self._model, '_model_name', 'unknown')
                if hasattr(self._model, 'model_name'):
                    model_name = self._model.model_name
            
            logger.info(f"========== AI 分析 {name}({code}) ==========")
            logger.info(f"[LLM配置] 模型: {model_name}")
            logger.info(f"[LLM配置] Prompt 长度: {len(prompt)} 字符")
            logger.info(f"[LLM配置] 是否包含新闻: {'是' if news_context else '否'}")
            
            # 记录完整 prompt 到日志（INFO级别记录摘要，DEBUG记录完整）
            prompt_preview = prompt[:500] + "..." if len(prompt) > 500 else prompt
            logger.info(f"[LLM Prompt 预览]\n{prompt_preview}")
            logger.debug(f"=== 完整 Prompt ({len(prompt)}字符) ===\n{prompt}\n=== End Prompt ===")

            # 设置生成配置（从配置文件读取温度参数）
            config = get_config()
            generation_config = {
                "temperature": config.gemini_temperature,
                "max_output_tokens": 8192,
            }

            logger.info(f"[LLM调用] 开始调用 Gemini API (temperature={generation_config['temperature']}, max_tokens={generation_config['max_output_tokens']})...")
            
            # 使用带重试的 API 调用
            start_time = time.time()
            response_text = self._call_api_with_retry(prompt, generation_config)
            elapsed = time.time() - start_time
            
            # 记录响应信息
            logger.info(f"[LLM返回] Gemini API 响应成功, 耗时 {elapsed:.2f}s, 响应长度 {len(response_text)} 字符")
            
            # 记录响应预览（INFO级别）和完整响应（DEBUG级别）
            response_preview = response_text[:300] + "..." if len(response_text) > 300 else response_text
            logger.info(f"[LLM返回 预览]\n{response_preview}")
            logger.debug(f"=== Gemini 完整响应 ({len(response_text)}字符) ===\n{response_text}\n=== End Response ===")
            
            # 解析响应
            result = self._parse_response(response_text, code, name)
            result.raw_response = response_text
            result.search_performed = bool(news_context)
            
            logger.info(f"[LLM解析] {name}({code}) 分析完成: {result.trend_prediction}, 评分 {result.sentiment_score}")
            
            return result
            
        except Exception as e:
            logger.error(f"AI 分析 {name}({code}) 失败: {e}")
            return AnalysisResult(
                code=code,
                name=name,
                sentiment_score=50,
                trend_prediction='震荡',
                operation_advice='持有',
                confidence_level='低',
                analysis_summary=f'分析过程出错: {str(e)[:100]}',
                risk_warning='分析失败，请稍后重试或手动分析',
                success=False,
                error_message=str(e),
            )
    
    def _format_prompt(
        self, 
        context: Dict[str, Any], 
        name: str,
        news_context: Optional[str] = None
    ) -> str:
        """
        格式化分析提示词（决策仪表盘 v2.0）
        
        包含：技术指标、实时行情（量比/换手率）、筹码分布、趋势分析、新闻
        
        Args:
            context: 技术面数据上下文（包含增强数据）
            name: 股票名称（默认值，可能被上下文覆盖）
            news_context: 预先搜索的新闻内容
        """
        code = context.get('code', 'Unknown')
        
        # 优先使用上下文中的股票名称（从 realtime_quote 获取）
        stock_name = context.get('stock_name', name)
        if not stock_name or stock_name == f'股票{code}':
            stock_name = STOCK_NAME_MAP.get(code, f'股票{code}')
            
        today = context.get('today', {})
        
        # ========== 构建决策仪表盘格式的输入 ==========
        prompt = f"""# 决策仪表盘分析请求

## 📊 股票基础信息
| 项目 | 数据 |
|------|------|
| 股票代码 | **{code}** |
| 股票名称 | **{stock_name}** |
| 分析日期 | {context.get('date', '未知')} |

---

## 📈 技术面数据

### 今日行情
| 指标 | 数值 |
|------|------|
| 收盘价 | {today.get('close', 'N/A')} 元 |
| 开盘价 | {today.get('open', 'N/A')} 元 |
| 最高价 | {today.get('high', 'N/A')} 元 |
| 最低价 | {today.get('low', 'N/A')} 元 |
| 涨跌幅 | {today.get('pct_chg', 'N/A')}% |
| 成交量 | {self._format_volume(today.get('volume'))} |
| 成交额 | {self._format_amount(today.get('amount'))} |

### 均线系统（关键判断指标）
| 均线 | 数值 | 说明 |
|------|------|------|
| MA5 | {today.get('ma5', 'N/A')} | 短期趋势线 |
| MA10 | {today.get('ma10', 'N/A')} | 中短期趋势线 |
| MA20 | {today.get('ma20', 'N/A')} | 中期趋势线 |
| 均线形态 | {context.get('ma_status', '未知')} | 多头/空头/缠绕 |
"""
        
        # 添加实时行情数据（量比、换手率等）
        if 'realtime' in context:
            rt = context['realtime']
            prompt += f"""
### 实时行情增强数据
| 指标 | 数值 | 解读 |
|------|------|------|
| 当前价格 | {rt.get('price', 'N/A')} 元 | |
| **量比** | **{rt.get('volume_ratio', 'N/A')}** | {rt.get('volume_ratio_desc', '')} |
| **换手率** | **{rt.get('turnover_rate', 'N/A')}%** | |
| 市盈率(动态) | {rt.get('pe_ratio', 'N/A')} | |
| 市净率 | {rt.get('pb_ratio', 'N/A')} | |
| 总市值 | {self._format_amount(rt.get('total_mv'))} | |
| 流通市值 | {self._format_amount(rt.get('circ_mv'))} | |
| 60日涨跌幅 | {rt.get('change_60d', 'N/A')}% | 中期表现 |
"""
        
        # 添加筹码分布数据
        if 'chip' in context:
            chip = context['chip']
            profit_ratio = chip.get('profit_ratio', 0)
            prompt += f"""
### 筹码分布数据（资金面参考）
| 指标 | 数值 | 健康标准 |
|------|------|----------|
| **获利比例** | **{profit_ratio:.1%}** | 70-90%时警惕 |
| 平均成本 | {chip.get('avg_cost', 'N/A')} 元 | 现价应高于5-15% |
| 90%筹码集中度 | {chip.get('concentration_90', 0):.2%} | <15%为集中 |
| 70%筹码集中度 | {chip.get('concentration_70', 0):.2%} | |
| 筹码状态 | {chip.get('chip_status', '未知')} | |
"""
        
        # 添加财务数据（价值投资面核心数据）
        if 'financial' in context and context['financial']:
            fin = context['financial']
            prompt += f"""
### 财务指标数据（价值投资面核心）
| 指标 | 数值 | 说明 |
|------|------|------|
| **ROE（净资产收益率）** | **{fin.get('roe', 'N/A')}%** | >15%优秀, 10-15%良好, <10%一般 |
| **营收增长率** | **{fin.get('revenue_growth', 'N/A')}%** | 同比增长率 |
| **净利润增长率** | **{fin.get('profit_growth', 'N/A')}%** | 同比增长率 |
| 销售毛利率 | {fin.get('gross_profit_margin', 'N/A')}% | 盈利能力指标 |
| 销售净利率 | {fin.get('net_profit_margin', 'N/A')}% | 盈利质量指标 |
| 财报日期 | {fin.get('report_date', 'N/A')} | 数据时效性 |

**数据来源**: {fin.get('data_source', 'unknown')}
"""
        else:
            prompt += """
### 财务指标数据
⚠️ **财务数据暂时无法获取**，价值面分析主要依据PE/PB估值和行业对比。
"""
        
        # 添加资金流数据（资金面核心数据）
        if 'moneyflow' in context and context['moneyflow']:
            mf = context['moneyflow']
            main_inflow = mf.get('main_net_inflow', 0) or 0
            main_inflow_yi = main_inflow / 10000  # 转换为亿元
            
            prompt += f"""
### 资金流向数据（资金面核心）
| 指标 | 数值 | 说明 |
|------|------|------|
| **主力资金净流入** | **{main_inflow_yi:.2f}亿元** | 特大单+大单净流入 |
| 主力净流入占比 | {mf.get('main_net_inflow_rate', 'N/A')}% | 占成交额比例 |
| 大单净流入 | {(mf.get('net_mf_lg', 0) or 0) / 10000:.2f}亿元 | 单笔>20万 |
| 中单净流入 | {(mf.get('net_mf_md', 0) or 0) / 10000:.2f}亿元 | 单笔4-20万 |
| 小单净流入 | {(mf.get('net_mf_sm', 0) or 0) / 10000:.2f}亿元 | 单笔<4万 |
| 交易日期 | {mf.get('trade_date', 'N/A')} | 数据时效性 |

**资金流向趋势**: {'流入' if main_inflow_yi > 0 else '流出'}
"""
            
            # 添加北向资金（如果有）
            if 'north_moneyflow' in context and context['north_moneyflow']:
                north = context['north_moneyflow']
                north_inflow_yi = north.get('total_net_amount', 0) / 10000
                
                prompt += f"""
### 北向资金（外资动向）
| 指标 | 数值 |
|------|------|
| 最近{north.get('days', 5)}日累计净流入 | {north_inflow_yi:.2f}亿元 |
| 日均净流入 | {north.get('avg_net_amount', 0) / 10000:.2f}亿元 |
| **趋势判断** | **{north.get('trend', '未知')}** |
"""
        else:
            prompt += """
### 资金流向数据
⚠️ **资金流数据暂时无法获取**（需要Tushare Pro 600积分），资金面分析主要依据筹码分布数据。
"""
        
        # 添加趋势分析结果（基于交易理念的预判）
        if 'trend_analysis' in context:
            trend = context['trend_analysis']
            bias_warning = "🚨 超过5%，严禁追高！" if trend.get('bias_ma5', 0) > 5 else "✅ 安全范围"
            prompt += f"""
### 趋势分析预判（基于交易理念）
| 指标 | 数值 | 判定 |
|------|------|------|
| 趋势状态 | {trend.get('trend_status', '未知')} | |
| 均线排列 | {trend.get('ma_alignment', '未知')} | MA5>MA10>MA20为多头 |
| 趋势强度 | {trend.get('trend_strength', 0)}/100 | |
| **乖离率(MA5)** | **{trend.get('bias_ma5', 0):+.2f}%** | {bias_warning} |
| 乖离率(MA10) | {trend.get('bias_ma10', 0):+.2f}% | |
| 量能状态 | {trend.get('volume_status', '未知')} | {trend.get('volume_trend', '')} |
| 系统信号 | {trend.get('buy_signal', '未知')} | |
| 系统评分 | {trend.get('signal_score', 0)}/100 | |

#### 系统分析理由
**买入理由**：
{chr(10).join('- ' + r for r in trend.get('signal_reasons', ['无'])) if trend.get('signal_reasons') else '- 无'}

**风险因素**：
{chr(10).join('- ' + r for r in trend.get('risk_factors', ['无'])) if trend.get('risk_factors') else '- 无'}
"""
        
        # 添加昨日对比数据
        if 'yesterday' in context:
            volume_change = context.get('volume_change_ratio', 'N/A')
            prompt += f"""
### 量价变化
- 成交量较昨日变化：{volume_change}倍
- 价格较昨日变化：{context.get('price_change_ratio', 'N/A')}%
"""
        
        # 添加新闻搜索结果（重点区域）
        prompt += """
---

## 📰 舆情情报
"""
        if news_context:
            prompt += f"""
以下是 **{stock_name}({code})** 近7日的新闻搜索结果，请重点提取：
1. 🚨 **风险警报**：减持、处罚、利空
2. 🎯 **利好催化**：业绩、合同、政策
3. 📊 **业绩预期**：年报预告、业绩快报

```
{news_context}
```
"""
        else:
            prompt += """
未搜索到该股票近期的相关新闻。请主要依据技术面数据进行分析。
"""

        # 注入缺失数据警告
        if context.get('data_missing'):
            prompt += """
⚠️ **数据缺失警告**
由于接口限制，当前无法获取完整的实时行情和技术指标数据。
请 **忽略上述表格中的 N/A 数据**，重点依据 **【📰 舆情情报】** 中的新闻进行基本面和情绪面分析。
在回答技术面问题（如均线、乖离率）时，请直接说明“数据缺失，无法判断”，**严禁编造数据**。
"""
        
        # 明确的输出要求
        prompt += f"""
---

## ✅ 分析任务

请为 **{stock_name}({code})** 生成【综合投资分析仪表盘】，严格按照 JSON 格式输出。

### 4维度评估要求：

**1. 价值投资面（40%权重）**
- PE/PB 估值水平（与行业比较）
- ROE 盈利质量
- 业绩增长性（营收/利润增长率）
- 护城河强度
- 给出 0-100 分评分

**2. 资金面（25%权重）**
- 主力资金流向（如有数据）
- 北向资金动向（A股）
- 筹码集中度、获利比例
- 给出 0-100 分评分

**3. 消息面（25%权重）**
- 新闻舆情（正面/负面/中性）
- 风险排查（减持、处罚、业绩预警）
- 利好催化（政策、合同、新品）
- 给出 0-100 分评分

**4. 趋势面（10%权重）**
- 均线系统排列
- 技术形态、量价关系
- 给出 0-100 分评分

### 综合评分计算：
sentiment_score = 价值面×0.4 + 资金面×0.25 + 消息面×0.25 + 趋势面×0.1

### 输出要求：
- 每个维度必须有 score 和 summary
- 核心结论基于4维度综合判断
- 如某维度数据缺失，在 summary 中说明，score 给中性分（40-60）

请输出完整的 JSON 格式综合投资分析仪表盘。"""
        
        return prompt
    
    def _format_volume(self, volume: Optional[float]) -> str:
        """格式化成交量显示"""
        if volume is None:
            return 'N/A'
        if volume >= 1e8:
            return f"{volume / 1e8:.2f} 亿股"
        elif volume >= 1e4:
            return f"{volume / 1e4:.2f} 万股"
        else:
            return f"{volume:.0f} 股"
    
    def _format_amount(self, amount: Optional[float]) -> str:
        """格式化成交额显示"""
        if amount is None:
            return 'N/A'
        if amount >= 1e8:
            return f"{amount / 1e8:.2f} 亿元"
        elif amount >= 1e4:
            return f"{amount / 1e4:.2f} 万元"
        else:
            return f"{amount:.0f} 元"
    
    def _parse_response(
        self, 
        response_text: str, 
        code: str, 
        name: str
    ) -> AnalysisResult:
        """
        解析 AI 响应（综合投资分析版）
        
        尝试从响应中提取 JSON 格式的分析结果，包含4维度评分和 dashboard 字段
        如果解析失败，尝试智能提取或返回默认结果
        """
        try:
            # 清理响应文本：移除 markdown 代码块标记
            cleaned_text = response_text
            if '```json' in cleaned_text:
                cleaned_text = cleaned_text.replace('```json', '').replace('```', '')
            elif '```' in cleaned_text:
                cleaned_text = cleaned_text.replace('```', '')
            
            # 尝试找到 JSON 内容
            json_start = cleaned_text.find('{')
            json_end = cleaned_text.rfind('}') + 1
            
            if json_start >= 0 and json_end > json_start:
                json_str = cleaned_text[json_start:json_end]
                
                # 尝试修复常见的 JSON 问题
                json_str = self._fix_json_string(json_str)
                
                data = json.loads(json_str)
                
                # 提取4维度数据
                dimensions = data.get('dimensions', {})
                
                # 提取各维度评分（带默认值）
                value_data = dimensions.get('value_investment', {})
                funding_data = dimensions.get('funding_flow', {})
                news_data = dimensions.get('news_sentiment', {})
                trend_data = dimensions.get('trend_analysis', {})
                
                value_score = int(value_data.get('score', 50))
                funding_score = int(funding_data.get('score', 50))
                news_score = int(news_data.get('score', 50))
                trend_score = int(trend_data.get('score', 50))
                
                # 提取 dashboard 数据
                dashboard = data.get('dashboard', None)
                
                # 解析所有字段，使用默认值防止缺失
                return AnalysisResult(
                    code=code,
                    name=name,
                    # 核心指标
                    sentiment_score=int(data.get('sentiment_score', 50)),
                    trend_prediction=data.get('trend_prediction', '震荡'),
                    operation_advice=data.get('operation_advice', '持有'),
                    confidence_level=data.get('confidence_level', '中'),
                    # 4维度评分
                    value_score=value_score,
                    funding_score=funding_score,
                    news_score=news_score,
                    trend_score=trend_score,
                    dimensions=dimensions,
                    # 决策仪表盘
                    dashboard=dashboard,
                    # 走势分析
                    trend_analysis=data.get('trend_analysis', ''),
                    short_term_outlook=data.get('short_term_outlook', ''),
                    medium_term_outlook=data.get('medium_term_outlook', ''),
                    # 技术面
                    technical_analysis=data.get('technical_analysis', ''),
                    ma_analysis=data.get('ma_analysis', ''),
                    volume_analysis=data.get('volume_analysis', ''),
                    pattern_analysis=data.get('pattern_analysis', ''),
                    # 基本面
                    fundamental_analysis=data.get('fundamental_analysis', ''),
                    sector_position=data.get('sector_position', ''),
                    company_highlights=data.get('company_highlights', ''),
                    # 情绪面/消息面
                    news_summary=data.get('news_summary', ''),
                    market_sentiment=data.get('market_sentiment', ''),
                    hot_topics=data.get('hot_topics', ''),
                    # 综合
                    analysis_summary=data.get('analysis_summary', '分析完成'),
                    key_points=data.get('key_points', ''),
                    risk_warning=data.get('risk_warning', ''),
                    buy_reason=data.get('buy_reason', ''),
                    # 元数据
                    search_performed=data.get('search_performed', False),
                    data_sources=data.get('data_sources', '技术面数据'),
                    success=True,
                )
            else:
                # 没有找到 JSON，尝试从纯文本中提取信息
                logger.warning(f"无法从响应中提取 JSON，使用原始文本分析")
                return self._parse_text_response(response_text, code, name)
                
        except json.JSONDecodeError as e:
            logger.warning(f"JSON 解析失败: {e}，尝试从文本提取")
            return self._parse_text_response(response_text, code, name)
    
    def _fix_json_string(self, json_str: str) -> str:
        """修复常见的 JSON 格式问题"""
        import re
        
        # 移除注释
        json_str = re.sub(r'//.*?\n', '\n', json_str)
        json_str = re.sub(r'/\*.*?\*/', '', json_str, flags=re.DOTALL)
        
        # 修复尾随逗号
        json_str = re.sub(r',\s*}', '}', json_str)
        json_str = re.sub(r',\s*]', ']', json_str)
        
        # 确保布尔值是小写
        json_str = json_str.replace('True', 'true').replace('False', 'false')
        
        return json_str
    
    def _parse_text_response(
        self, 
        response_text: str, 
        code: str, 
        name: str
    ) -> AnalysisResult:
        """从纯文本响应中尽可能提取分析信息"""
        # 尝试识别关键词来判断情绪
        sentiment_score = 50
        trend = '震荡'
        advice = '持有'
        
        text_lower = response_text.lower()
        
        # 简单的情绪识别
        positive_keywords = ['看多', '买入', '上涨', '突破', '强势', '利好', '加仓', 'bullish', 'buy']
        negative_keywords = ['看空', '卖出', '下跌', '跌破', '弱势', '利空', '减仓', 'bearish', 'sell']
        
        positive_count = sum(1 for kw in positive_keywords if kw in text_lower)
        negative_count = sum(1 for kw in negative_keywords if kw in text_lower)
        
        if positive_count > negative_count + 1:
            sentiment_score = 65
            trend = '看多'
            advice = '买入'
        elif negative_count > positive_count + 1:
            sentiment_score = 35
            trend = '看空'
            advice = '卖出'
        
        # 截取前500字符作为摘要
        summary = response_text[:500] if response_text else '无分析结果'
        
        return AnalysisResult(
            code=code,
            name=name,
            sentiment_score=sentiment_score,
            trend_prediction=trend,
            operation_advice=advice,
            confidence_level='低',
            analysis_summary=summary,
            key_points='JSON解析失败，仅供参考',
            risk_warning='分析结果可能不准确，建议结合其他信息判断',
            raw_response=response_text,
            success=True,
        )
    
    def batch_analyze(
        self, 
        contexts: List[Dict[str, Any]],
        delay_between: float = 2.0
    ) -> List[AnalysisResult]:
        """
        批量分析多只股票
        
        注意：为避免 API 速率限制，每次分析之间会有延迟
        
        Args:
            contexts: 上下文数据列表
            delay_between: 每次分析之间的延迟（秒）
            
        Returns:
            AnalysisResult 列表
        """
        results = []
        
        for i, context in enumerate(contexts):
            if i > 0:
                logger.debug(f"等待 {delay_between} 秒后继续...")
                time.sleep(delay_between)
            
            result = self.analyze(context)
            results.append(result)
        
        return results


# 便捷函数
def get_analyzer() -> GeminiAnalyzer:
    """获取 Gemini 分析器实例"""
    return GeminiAnalyzer()


if __name__ == "__main__":
    # 测试代码
    logging.basicConfig(level=logging.DEBUG)
    
    # 模拟上下文数据
    test_context = {
        'code': '600519',
        'date': '2026-01-09',
        'today': {
            'open': 1800.0,
            'high': 1850.0,
            'low': 1780.0,
            'close': 1820.0,
            'volume': 10000000,
            'amount': 18200000000,
            'pct_chg': 1.5,
            'ma5': 1810.0,
            'ma10': 1800.0,
            'ma20': 1790.0,
            'volume_ratio': 1.2,
        },
        'ma_status': '多头排列 📈',
        'volume_change_ratio': 1.3,
        'price_change_ratio': 1.5,
    }
    
    analyzer = GeminiAnalyzer()
    
    if analyzer.is_available():
        print("=== AI 分析测试 ===")
        result = analyzer.analyze(test_context)
        print(f"分析结果: {result.to_dict()}")
    else:
        print("Gemini API 未配置，跳过测试")
