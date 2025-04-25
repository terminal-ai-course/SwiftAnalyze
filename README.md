# SwiftAnalyze：基于DeepResearch的极速行业分析智能引擎

## 项目简介

基于网络搜索和大语言模型（LLM）的智能行业分析报告生成工具，支持金融市场、科技行业等多个领域的深度分析。

面对信息爆炸和快速变化的市场环境，传统的手动信息收集、筛选和整合过程耗时耗力。**此工具通过深度融合网络搜索引擎的实时数据获取能力**与大型语言模型的强大推理、理解和生成能力，为用户提供一站式、智能化的行业分析解决方案。

## 功能特性

- 🚀 自动化行业分析流程：查询分解→智能搜索→信息整合→报告生成
- 🔍 支持多源网络数据抓取
- 🧠 基于Qwen和Deepseek大模型的智能推理与内容生成
- 📊 内置基础数据分析功能（关键词统计、数值提取）
- 🏭 多行业支持，灵活配置分析策略
- 📑 自动生成带引用来源的专业级报告

## 模型架构

![img](https://uvoialj5w0z.feishu.cn/space/api/box/stream/download/asynccode/?code=YmVhZTVhODk2NDg0ODAxNWE0NThiN2M2ZmU0YmMxZWVfbExOU2l1Q2J3TmwweUtMRWhrZTFzMUdSY2dydW5BbW1fVG9rZW46SzJZeGJvVXUyb0VDa3V4ZGNKMmNQV2xUblpmXzE3NDU1ODE1MDU6MTc0NTU4NTEwNV9WNA)

**本项目核心逻辑是迭代式、自我优化的研究****范式****：**

1. **智能规划 (****Query** **Decomposition):** 接收用户关于特定行业（如金融、科技）的初始查询后，LLM首先将宽泛的问题分解为一系列具体的、可操作的子问题，形成结构化的研究计划。
2. **动态网络搜索 (Intelligent Search):** 针对分解后的子问题，工具自动调用网络搜索API，抓取最新的相关新闻、文章摘要、数据点等信息。
3. **信息整合与去重 (Integration & Deduplication):** 收集到的信息被系统地整理、存储，并通过URL进行去重，确保信息来源的多样性和有效性。
4. **LLM****驱动的反思与评估 (Reflection & Evaluation):** 在每一轮信息收集后，LLM会对当前掌握的信息进行“反思”，评估其是否足以回答原始问题、识别冗余或无关信息，并**智能生成新的、更深入的****子查询**以弥补信息缺口或澄清疑点。
5. **循环迭代 (Iterative Refinement):** “搜索-整合-反思”的循环会进行多轮（可配置），直至信息被认为足够全面或达到预设的迭代次数。
6. **基础数据扫描 (Basic** **Data Analysis****):** 在合成报告前，工具可对收集到的文本执行初步的数据扫描，提取关键术语频率、数值、百分比等，为报告提供量化视角（可选）。
7. **报告自动合成 (Report Generation):** 最后，利用专门配置的LLM（如Deepseek），基于所有筛选、整合后的信息和数据扫描结果，自动撰写一篇结构清晰、逻辑连贯、观点客观的行业分析报告。
8. **严谨引用 (Citation):** 最重要的是，报告在引用信息时会自动标注其来源URL (`[来源: URL]`)，确保内容的可追溯性和专业性。

## 使用方法

### 快速使用

1. 填写API密钥：

```TOML
BOCHAAI_API_KEY=your_bochaai_key
DASHSCOPE_API_KEY=your_dashscope_key
```

1. 修改行业：

```Python
# 修改行业配置（支持 finance/tech）
SELECTED_INDUSTRY = "tech" 

# 设置分析问题
user_query = "2024年人工智能芯片领域的最新发展动态"
```

1. 运行脚本：

```Bash
python industry_analyst.py
```

### 行业配置

通过`INDUSTRY_CONFIGS`字典自定义分析策略：

```Python
INDUSTRY_CONFIGS = {
    "finance": {
        "name": "金融市场",
        "filename_prefix": "金融分析_",
        "analyzer_keywords": ["股票", "指数", "涨跌幅"],
        # ...其他配置参数
    },
    "tech": {
        "name": "科技行业",
        # ...科技行业专属配置
    }
}
```

### 输出示例

```Plain
==================== 最终科技行业分析报告 ====================

根据收集的信息，2024年人工智能芯片领域呈现以下发展趋势...

[核心数据]
- 全球AI芯片市场规模同比增长35% [来源: https://example.com/market-report]
- 国产芯片厂商出货量突破百万级 [来源: https://example.com/tech-news]

[技术突破]
寒武纪最新发布第五代MLU架构...
[来源: https://example.com/chip-release]

==================== 报告结束 ====================
```

### 高级功能

#### 自定义搜索策略

```Python
# 修改搜索参数
def websearch(query, count=8):  # 增加搜索结果数量
    # ...
```

#### 数据分析配置

```Python
# 添加行业关键词
"analyzer_keywords": ["量子计算", "神经形态芯片", "存算一体"]
```

## 注意事项

1. API服务可用性依赖Bochaai和Dashscale的运营状态
2. 建议在虚拟环境中运行
3. 默认搜索次数限制为3次迭代（可通过max_iterations参数调整）
