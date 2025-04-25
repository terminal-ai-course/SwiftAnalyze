# -*- coding: utf-8 -*-
"""
行业分析报告生成器（使用网络搜索和LLM）

此脚本自动化生成行业分析报告的过程，通过以下步骤：
1.  接收用户关于特定行业的初始查询。
2.  使用 LLM 将查询分解为可搜索的子查询（规划）。
3.  使用 Bochaai API 对这些子查询执行网络搜索。
4.  整合和去重搜索结果。
5.  使用 LLM 评估收集到的信息，并在需要时建议进一步的子查询（反思）。
6.  重复搜索和反思步骤，直到达到最大迭代次数。
7.  （可选）对收集到的文本片段执行基本数据分析（关键词频率、数字提取）。
8.  使用流式 LLM（Deepseek）基于所有收集的信息合成最终报告，并适当地引用来源。

脚本可以通过 `INDUSTRY_CONFIGS` 字典为不同行业进行配置，
允许自定义提示、关键词和输出文件名。
"""

import requests # 用于发出 HTTP 请求到 API
import json     # 用于处理 JSON 数据（API 请求/响应）
from openai import OpenAI # 官方 OpenAI 库，用于与兼容 API（如 DashScope）交互
import os       # 用于访问环境变量（API 密钥）
import time     # 用于暂停执行（例如，在 API 调用之间）
import re       # 用于正则表达式（数据分析、文件名清理）
from urllib.parse import urlparse # 用于解析 URL，对去重有用
from collections import Counter # 用于在数据分析中统计关键词频率
import logging # 用于结构化地记录信息和错误

# --- 配置 ---

# API 密钥:
# 建议使用环境变量以确保安全。
# 仅在不使用环境变量时才替换占位符字符串。
SEARCH_API_KEY = os.getenv("BOCHAAI_API_KEY", "") # Bochaai 网络搜索 API 密钥
LLM_API_KEY = os.getenv("DASHSCOPE_API_KEY", "")      # DashScope (或兼容的 LLM 提供商) API 密钥
LLM_BASE_URL = os.getenv("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1") # LLM 提供商的 API 端点

# 设置基本的日志记录配置
# 将 INFO 级别及以上的日志消息记录到控制台。
# 日志包含时间戳、日志级别和消息本身。
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- 行业配置 ---
# 这个字典存储了脚本可以分析的每个行业的特定设置。
# 在这里添加新行业，遵循相同的结构。
INDUSTRY_CONFIGS = {
    "finance": {
        "name": "金融市场 (Financial Markets)", # 行业的显示名称
        "filename_prefix": "金融市场分析_",       # 输出报告文件名的前缀
        # 用于规划和反思阶段的 LLM 系统提示
        "llm_system_prompt_assistant": "你是一位专门研究金融市场行情的资深分析助理。请仔细遵循指示并以要求的格式回答。",
        # 用于最终报告合成阶段的 LLM 系统提示
        "llm_system_prompt_synthesizer": "你是一位专业的金融市场分析师，负责根据收集的信息生成客观的市场分析报告。请严格遵循指示，并使用 `[来源: URL]` 格式引用来源。",
        # simple_data_analyzer 函数用于此行业的关键词
        "analyzer_keywords": [
            '股票', 'A股', '港股', '美股', '上证指数', '深证成指', '创业板指', '恒生指数', '纳斯达克', '道琼斯',
            '涨', '跌', '上涨', '下跌', '涨幅', '跌幅', '成交额', '成交量', '换手率',
            '市盈率', '市净率', '概念', '板块', '行业', '龙头',
            '宏观经济', '利率', '通胀', '加息', '降息', '财报', '业绩',
            '利好', '利空', '风险', '机会', '预期', '预测', '分析', '行情', 'IPO', '并购', '央行'
        ],
        # 用于生成初始子查询的 LLM 提示模板（规划）
        # 使用 f-string 格式化占位符 {industry_name} 和 {initial_query}
        "plan_prompt_template": """
            请将以下用户的关于【{industry_name}】的主要问题分解为具体的、可搜索的子问题列表，以全面分析该主题。
            请关注市场概览、关键指标/指数表现、重要板块/公司动态、相关新闻事件、宏观经济/政策影响、技术趋势（如果信息允许）等方面。
            请严格按照以下 JSON 格式输出，不要包含任何额外的解释或评论：
            {{
              "subqueries": [
                "子问题1: {industry_name}整体表现如何？",
                "子问题2: 主要基准（如指数、利率）情况怎样？",
                "子问题3: 有哪些值得关注的热点领域或概念？",
                "子问题4: 有哪些重要的行业新闻或政策发布？",
                "子问题5: （可选，如适用）特定公司或资产的表现和相关信息？"
              ]
            }}

            用户主要问题："{initial_query}"
        """,
        # 用于评估收集到的数据的 LLM 提示模板（反思）
        # 使用占位符 {industry_name}, {initial_query}, {memory_context_for_llm}
        "reflection_prompt_template": """
            作为【{industry_name}】分析评估员，请评估为回答以下用户原始问题而收集的信息摘要。

            用户原始问题："{initial_query}"

            目前收集到的信息摘要（可能部分截断）：
            {memory_context_for_llm}

            请评估：
            1.  `can_answer`: 这些信息是否**足够全面**地回答用户的原始【{industry_name}】问题？(true/false)
            2.  `irrelevant_urls`: 当前摘要中，是否有与回答原始问题**明显无关或信息价值低**的条目？（仅列出这些条目的来源 URL 列表，如果没有则为空列表 []）
            3.  `new_subqueries`: 基于当前信息和原始问题，还需要提出哪些**具体的、新的**子问题来**获取关键行业数据**或**澄清当前状况**？（例如：需要特定指标的最新数据？需要某领域更详细的动态？需要某事件的最新进展？如果信息已足够，则返回空列表 []）

            请严格按照以下 JSON 格式进行响应，不要包含任何额外的解释或评论：
            {{
                "can_answer": boolean,
                "irrelevant_urls": ["url1", "url2", ...],
                "new_subqueries": ["新问题1", "新问题2", ...]
            }}
        """,
        # 用于生成最终报告的 LLM 提示模板（合成）
        # 使用占位符 {industry_name}, {initial_query}, {final_memory_context}, {analysis_section}
         "synthesis_prompt_template": """
            您是一位专业的【{industry_name}】分析师。您的任务是基于以下通过网络搜索收集到的信息片段，为用户生成一份全面、结构清晰、客观中立的行业分析报告，以回答他们的原始问题。

            用户的原始问题是："{initial_query}"

            以下是收集到的相关行业信息（主要来自新闻、网页摘要）：
            --- 开始信息 ---
            {final_memory_context}
            --- 结束信息 ---
            {analysis_section}
            请严格遵守以下要求撰写报告：
            1.  **完全基于**上面提供的信息片段撰写报告。不得添加任何外部知识、个人观点、未经证实的精确数据（除非信息片段中明确提到）。
            2.  清晰、有条理地组织报告内容，直接回答用户的原始问题。可适当使用标题和小标题（例如：行业概览、主要趋势、关键参与者、重要新闻、总结与展望等）。
            3.  在报告中**必须**引用信息来源。当您使用某条信息时，请在其后用方括号注明来源 URL，格式为 `[来源: URL]`。例如：XX 公司发布了新产品 [来源: http://example.com/news1]。**确保URL完整且在方括号内**。
            4.  如果提供了“数据扫描摘要”，请将扫描结果（如关键词频率、发现的数值或百分比）适当地融入报告内容中，并指明这只是基于所提供文本的初步扫描。**不要将扫描到的数字当作精确的实时数据**。
            5.  语言专业、客观、中立。避免使用过度乐观或悲观的词语，避免给出直接的商业建议。专注于总结和呈现收集到的信息。
            6.  如果信息片段之间存在矛盾或不一致之处，请客观地指出。
            7.  如果收集到的信息不足以回答问题的某些方面，请在报告中明确说明。
            8.  报告结尾可以根据信息做一个简要的总结或展望，但必须基于已提供的信息，并保持客观。

            请开始撰写您的【{industry_name}】分析报告：
        """
    },
    "tech": {
        "name": "科技行业 (Technology Industry)",
        "filename_prefix": "科技行业分析_",
        "llm_system_prompt_assistant": "你是一位专门研究科技行业动态的资深分析助理。请仔细遵循指示并以要求的格式回答。",
        "llm_system_prompt_synthesizer": "你是一位专业的科技行业分析师，负责根据收集的信息生成客观的行业分析报告。请严格遵循指示，并使用 `[来源: URL]` 格式引用来源。",
        "analyzer_keywords": [
            '人工智能', 'AI', '机器学习', '芯片', '半导体', '云计算', '大数据', '软件', '硬件',
            '互联网', 'SaaS', 'PaaS', 'IaaS', '物联网', 'IoT', '5G', '6G', '元宇宙', 'VR', 'AR',
            '初创公司', '融资', '风险投资', 'VC', 'PE', '上市', 'IPO', '裁员', '并购', 'M&A',
            '科技巨头', '苹果', '谷歌', '微软', '亚马逊', 'Meta', '腾讯', '阿里巴巴', '华为', '字节跳动',
            '创新', '研发', '专利', '趋势', '法规', '监管', '数据隐私', '网络安全'
        ],
        "plan_prompt_template": """
            请将以下用户的关于【{industry_name}】的主要问题分解为具体的、可搜索的子问题列表，以全面分析该主题。
            请关注市场规模与增长、关键技术领域、主要公司动态（产品、战略、财报）、投融资活动、最新行业新闻、政策法规影响等方面。
            请严格按照以下 JSON 格式输出，不要包含任何额外的解释或评论：
            {{
              "subqueries": [
                "子问题1: {industry_name}整体发展趋势如何？",
                "子问题2: 主要技术领域（如AI、云计算）有哪些新进展？",
                "子问题3: 重点科技公司最近有哪些重要动态？",
                "子问题4: {industry_name}最近的投融资情况怎样？",
                "子问题5: 有哪些值得关注的行业新闻或政策发布？"
              ]
            }}

            用户主要问题："{initial_query}"
        """,
        "reflection_prompt_template": """
            作为【{industry_name}】分析评估员，请评估为回答以下用户原始问题而收集的信息摘要。

            用户原始问题："{initial_query}"

            目前收集到的信息摘要（可能部分截断）：
            {memory_context_for_llm}

            请评估：
            1.  `can_answer`: 这些信息是否**足够全面**地回答用户的原始【{industry_name}】问题？(true/false)
            2.  `irrelevant_urls`: 当前摘要中，是否有与回答原始问题**明显无关或信息价值低**的条目？（仅列出这些条目的来源 URL 列表，如果没有则为空列表 []）
            3.  `new_subqueries`: 基于当前信息和原始问题，还需要提出哪些**具体的、新的**子问题来**获取关键行业数据**或**澄清当前状况**？（例如：需要某项技术的最新应用？需要某公司更详细的战略分析？需要某事件的最新进展？如果信息已足够，则返回空列表 []）

            请严格按照以下 JSON 格式进行响应，不要包含任何额外的解释或评论：
            {{
                "can_answer": boolean,
                "irrelevant_urls": ["url1", "url2", ...],
                "new_subqueries": ["新问题1", "新问题2", ...]
            }}
        """,
        "synthesis_prompt_template": """
            您是一位专业的【{industry_name}】分析师。您的任务是基于以下通过网络搜索收集到的信息片段，为用户生成一份全面、结构清晰、客观中立的行业分析报告，以回答他们的原始问题。

            用户的原始问题是："{initial_query}"

            以下是收集到的相关行业信息（主要来自新闻、网页摘要）：
            --- 开始信息 ---
            {final_memory_context}
            --- 结束信息 ---
            {analysis_section}
            请严格遵守以下要求撰写报告：
            1.  **完全基于**上面提供的信息片段撰写报告。不得添加任何外部知识、个人观点、未经证实的精确数据（除非信息片段中明确提到）。
            2.  清晰、有条理地组织报告内容，直接回答用户的原始问题。可适当使用标题和小标题（例如：行业概览、技术趋势、主要公司动态、投融资情况、总结与展望等）。
            3.  在报告中**必须**引用信息来源。当您使用某条信息时，请在其后用方括号注明来源 URL，格式为 `[来源: URL]`。例如：XX 公司发布了新AI模型 [来源: http://example.com/news1]。**确保URL完整且在方括号内**。
            4.  如果提供了“数据扫描摘要”，请将扫描结果（如关键词频率、发现的数值或百分比）适当地融入报告内容中，并指明这只是基于所提供文本的初步扫描。**不要将扫描到的数字当作精确的实时数据**。
            5.  语言专业、客观、中立。避免使用过度乐观或悲观的词语，避免给出直接的商业建议。专注于总结和呈现收集到的信息。
            6.  如果信息片段之间存在矛盾或不一致之处，请客观地指出。
            7.  如果收集到的信息不足以回答问题的某些方面，请在报告中明确说明。
            8.  报告结尾可以根据信息做一个简要的总结或展望，但必须基于已提供的信息，并保持客观。

            请开始撰写您的【{industry_name}】分析报告：
        """
    },
    # --- 在这里添加更多行业 ---
    # 新行业的示例结构：
    # "healthcare": {
    #     "name": "医疗健康行业 (Healthcare Industry)",
    #     "filename_prefix": "医疗健康分析_",
    #     "llm_system_prompt_assistant": "...",
    #     "llm_system_prompt_synthesizer": "...",
    #     "analyzer_keywords": [...],
    #     "plan_prompt_template": """...""",
    #     "reflection_prompt_template": """...""",
    #     "synthesis_prompt_template": """..."""
    # },
    # "energy": { ... },
}

# --- 选择要分析的行业 ---
# 将此值更改为您想要分析的行业的键（例如，"finance", "tech"）。
SELECTED_INDUSTRY = "tech"

# --- 核心函数 ---

def websearch(query, count=5):
    """
    使用配置的 Bochaai API 端点执行网络搜索。

    Args:
        query (str): 搜索查询字符串。
        count (int): 期望的搜索结果数量（默认值：5）。

    Returns:
        list: 包含字典的列表，每个字典代表一个搜索结果，
              包含 'url', 'name', 'summary', 'snippet' 等键。
              如果搜索失败或没有返回有效结果，则返回空列表。
    """
    logging.info(f"正在执行网络搜索: '{query}'")
    url = "https://api.bochaai.com/v1/web-search" # Bochaai API 端点
    # 根据 Bochaai API 文档准备请求负载
    payload = json.dumps({
        "query": query,
        "summary": True, # 请求结果的摘要（对获取上下文有用）
        "count": count,  # 要获取的结果数量
        "page": 1       # 搜索结果页码
    })
    # 设置必需的请求头，包括 API 密钥
    headers = {
        'Authorization': SEARCH_API_KEY,
        'Content-Type': 'application/json'
    }

    try:
        # 向 API 发送 POST 请求
        # 增加超时时间以处理新闻/复杂站点可能较慢的响应
        response = requests.post(url, headers=headers, data=payload, timeout=45)
        response.raise_for_status() # 对错误的响应（4xx 或 5xx）抛出 HTTPError
        data = response.json()      # 解析 JSON 响应
    except requests.exceptions.Timeout:
        # 特别处理请求超时错误
        logging.error(f"查询 '{query}' 的网络搜索请求超时")
        return []
    except requests.exceptions.RequestException as e:
        # 处理其他可能的网络或 HTTP 错误
        logging.error(f"查询 '{query}' 的网络搜索请求期间出错: {e}")
        return []
    except json.JSONDecodeError as e:
        # 处理响应不是有效 JSON 的错误
        logging.error(f"解码网络搜索 '{query}' 的 JSON 时出错: {e}。响应文本: {response.text[:500]}...") # 记录响应的前 500 个字符
        return []
    except Exception as e:
        # 处理搜索期间的任何其他意外错误
        logging.error(f"查询 '{query}' 的网络搜索期间发生意外错误: {e}", exc_info=True) # 记录完整的追溯信息
        return []

    # 从嵌套的 JSON 结构中提取网页结果列表
    webpages_data = data.get('data', {}).get('webPages', {})
    value_list = webpages_data.get('value')

    # 验证预期的 'value' 列表是否存在且确实是一个列表
    if value_list is None or not isinstance(value_list, list):
        logging.warning(f"在查询 '{query}' 的响应中找不到 'value' 列表或它不是一个列表。")
        return []

    # 过滤结果以确保它们有用：
    # 每个结果必须有 'url' 和 'summary' 或 'snippet'。
    filtered_results = [
        item for item in value_list
        if item.get('url') and (item.get('summary') or item.get('snippet'))
    ]
    logging.info(f"查询 '{query}' 的网络搜索返回了 {len(filtered_results)} 个有效结果。")
    return filtered_results

def qwen_llm(prompt, industry_config, model="qwen-max", response_format=None):
    """
    使用非流式请求调用 Qwen LLM（通过 DashScope 或兼容 API）。
    此函数通常用于结构化任务，如规划和反思，通常期望得到 JSON 响应。

    Args:
        prompt (str): 提供给 LLM 的用户提示。
        industry_config (dict): 所选行业的配置字典，用于检索适当的系统提示。
        model (str): 要使用的特定 LLM 模型（默认值："qwen-max"）。
        response_format (dict, optional): 指定期望的响应格式，例如 {"type": "json_object"}。默认为 None。

    Returns:
        str | None: LLM 响应的内容（字符串形式），如果发生错误则返回 None。
                    如果 response_format 是 JSON，此字符串将包含 JSON。
    """
    logging.info(f"正在调用 Qwen LLM (模型: {model}) 处理: {prompt[:100]}...")
    # 从行业配置中获取适当的系统提示
    system_message_content = industry_config.get("llm_system_prompt_assistant", "你是一个有用的助手。") # 如果找不到则使用默认值

    try:
        # 初始化 OpenAI 客户端，指向 DashScope 基础 URL
        client = OpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL)

        # 准备 Chat Completions API 调用的参数
        completion_args = {
            "model": model,
            "messages": [
                {'role': 'system', 'content': system_message_content}, # 设置系统角色/身份
                {'role': 'user', 'content': prompt}                    # 提供用户的请求
            ],
            "temperature": 0.2, # 较低的温度以获得更确定性、创造性较低的输出（适用于 JSON）
            # 如果需要，可以在此处添加 max_tokens: "max_tokens": 1024
        }
        # 如果请求了特定的响应格式（如 JSON），则将其添加到参数中
        if response_format:
            completion_args["response_format"] = response_format
            logging.info("正在请求 LLM 返回 JSON 格式。")

        # 进行 API 调用
        completion = client.chat.completions.create(**completion_args)
        # 提取响应内容
        content = completion.choices[0].message.content
        logging.info("LLM 调用成功。")
        return content
    except Exception as e:
        # 处理 LLM API 调用期间的任何错误
        logging.error(f"调用 Qwen LLM 时出错: {e}", exc_info=True) # 记录完整的追溯信息
        return None # 指示失败

def deepseek_stream(prompt, industry_config, model_name="deepseek-r1"):
    """
    使用流式请求调用 Deepseek LLM（通过 DashScope 或兼容 API）。
    此函数主要用于生成最终的、可能较长的报告，允许逐步显示输出。

    Args:
        prompt (str): 提供给 LLM 的用户提示（包含合成指令和收集的信息）。
        industry_config (dict): 所选行业的配置字典，用于检索适当的系统提示。
        model_name (str): 要使用的特定 LLM 模型（默认值："deepseek-r1"）。

    Returns:
        str: LLM 生成的完整文本，由所有流式块连接而成。
             如果流式处理失败，则返回错误消息字符串。
    """
    logging.info(f"正在调用 Deepseek LLM 流 (模型: {model_name}) 进行最终合成...")
    # 从行业配置中获取用于合成的适当系统提示
    system_message_content = industry_config.get(
        "llm_system_prompt_synthesizer",
        "你是一个有用的助手，负责将信息合成为最终报告，并使用 [来源: URL] 格式仔细引用来源。"
    ) # 如果找不到则使用默认值

    try:
        # 初始化 OpenAI 客户端，指向 DashScope 基础 URL
        client = OpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL)
        # 使用 stream=True 进行 API 调用
        stream = client.chat.completions.create(
            model=model_name,
            messages=[
                {'role': 'system', 'content': system_message_content}, # 设置系统角色/身份
                {'role': 'user', 'content': prompt}                    # 提供用户的请求（合成提示）
            ],
            stream=True,        # 启用流式响应
            temperature=0.5,    # 稍高的温度，使报告语言更自然
            # stream_options={"include_usage": True} # 取消注释以在末尾接收 token 使用信息
        )

        # 在打印流式报告之前显示标题
        industry_name_display = industry_config.get("name", "分析")
        print(f"\n{'=' * 20} 最终 {industry_name_display} 报告 {'=' * 20}\n")
        full_response = ""
        # 迭代从流接收到的块
        for chunk in stream:
            # 检查块是否包含使用信息（如果启用，通常在末尾发送）
            if not getattr(chunk, 'choices', None) and hasattr(chunk, 'usage') and chunk.usage:
                 print("\n" + "=" * 20 + " Token 使用量 " + "=" * 20 + "\n")
                 print(chunk.usage) # 打印 token 使用详情
                 continue # 继续处理下一个块

            # 检查块是否包含实际内容
            if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                content_piece = chunk.choices[0].delta.content
                print(content_piece, end='', flush=True) # 立即将内容块打印到控制台
                full_response += content_piece           # 将块附加到完整响应字符串

        # 流结束后打印页脚
        print(f"\n{'=' * 20} 报告结束 {'=' * 20}\n")
        logging.info("Deepseek 流式合成完成。")
        return full_response # 返回完整的连接后的报告文本

    except Exception as e:
        # 处理流式 LLM 调用期间的任何错误
        logging.error(f"调用 Deepseek LLM 流时出错: {e}", exc_info=True) # 记录完整的追溯信息
        print("\n最终报告合成期间出错。")
        # 返回用户友好的错误消息
        return f"抱歉，在生成最终{industry_name_display}报告时遇到错误。"

# --- 数据分析函数（使用行业配置）---
def simple_data_analyzer(text_data, industry_config):
    """
    对收集的文本片段执行非常基本的数据分析。
    它查找数值、百分比，并统计与所选行业相关的预定义关键词的出现次数。

    Args:
        text_data (list): 字符串列表，通常是来自网络搜索结果的摘要或片段。
        industry_config (dict): 所选行业的配置字典，用于检索相关关键词。

    Returns:
        str: 分析结果的摘要字符串（计数、数字/百分比的基本统计信息、热门关键词）。
             如果没有找到重要内容，则返回指示缺少数据的消息。
    """
    industry_name = industry_config.get("name", "数据") # 获取用于显示的行业名称
    print(f"\n{'=' * 20} 基本 {industry_name} 扫描 {'=' * 20}\n") # 打印标题

    # 初始化列表/计数器以存储提取的数据
    numbers = []
    percentages = []
    keywords = Counter() # 使用 Counter 进行高效的关键词频率计数

    # 定义正则表达式模式以查找数字和百分比
    # 此模式捕获整数和小数，可能带有逗号作为千位分隔符。
    # 它还捕获可能跟在数字后面的常见单位，如 %、亿、万、千、百。
    # 注意：这是一个简化的模式，可能会捕获非预期的数字。
    number_pattern = re.compile(r'(\d{1,3}(?:,\d{3})*(?:\.\d+)?|\d+(?:\.\d+)?)\s*(?:%|亿|万|千|百)?')
    # 此模式专门捕获后跟百分号的数字，允许 +/- 符号。
    percentage_pattern = re.compile(r'([+-]?\d+(?:\.\d+)?)\s*%')

    # 从行业配置中获取相关关键词列表
    relevant_keywords = industry_config.get("analyzer_keywords", [])
    if not relevant_keywords:
        logging.warning("在行业配置中未找到 analyzer_keywords。将跳过关键词分析。")

    # 将所有文本片段合并为一个字符串，以便于正则表达式搜索
    full_text = " ".join(text_data)
    logging.info(f"正在对 {len(text_data)} 个文本片段（{industry_name} 上下文）执行基本分析。")

    # --- 使用 Regex 提取数据 ---
    # 查找所有匹配通用数字模式的出现
    num_values = number_pattern.findall(full_text)
    # 将找到的字符串转换为浮点数，首先删除逗号。检查它是否是有效的数字。
    numbers.extend([float(val.replace(',', '')) for val in num_values if val and val.replace(',', '').replace('.', '', 1).isdigit()])
    logging.info(f"找到 {len(numbers)} 个潜在的数值。")

    # 查找所有匹配百分比模式的出现
    perc_values = percentage_pattern.findall(full_text)
    # 将找到的百分比字符串（不含 '%'）转换为浮点数
    percentages.extend([float(p) for p in perc_values])
    logging.info(f"找到 {len(percentages)} 个百分比值。")

    # --- 统计关键词 ---
    if relevant_keywords:
        lower_full_text = full_text.lower() # 将文本转换为小写以进行不区分大小写的匹配
        for keyword in relevant_keywords:
            # 简单的子字符串计数。对于像中文这样的语言，这通常足够了。
            # 对于英语中需要词边界的计数，使用 re.findall(r'\b' + keyword + r'\b', ...)
            count = lower_full_text.count(keyword.lower())
            if count > 0:
                keywords[keyword] += count # 在 Counter 对象中增加计数
        logging.info(f"关键词计数 (Top 5): {keywords.most_common(5)}")
    else:
         logging.info("由于未提供关键词，跳过关键词分析。")

    # --- 生成摘要字符串 ---
    analysis_summary = "简要数据扫描摘要:\n" # 开始摘要字符串
    found_data = False # 标记是否找到任何有意义的数据

    # 总结数值发现
    if numbers:
        found_data = True
        try: # 如果可能，计算基本统计信息
            analysis_summary += f"- 扫描到 {len(numbers)} 个数值点. 平均值: {sum(numbers)/len(numbers):.2f}, 最小值: {min(numbers):.2f}, 最大值: {max(numbers):.2f}\n"
        except ZeroDivisionError: # 处理有数字但除以零的情况（如果 len>0 则不应发生）
             analysis_summary += f"- 扫描到 {len(numbers)} 个数值点, 但无法计算统计数据。\n"
    else:
        analysis_summary += "- 未明确扫描到可用于统计分析的数值点。\n"

    # 总结百分比发现
    if percentages:
        found_data = True
        try: # 如果可能，计算基本统计信息
            analysis_summary += f"- 扫描到 {len(percentages)} 个百分比值. 平均值: {sum(percentages)/len(percentages):.2f}%, 最小值: {min(percentages):.2f}%, 最大值: {max(percentages):.2f}%\n"
        except ZeroDivisionError:
             analysis_summary += f"- 扫描到 {len(percentages)} 个百分比值, 但无法计算统计数据。\n"
    else:
        analysis_summary += "- 未扫描到明确的百分比值。\n"

    # 总结关键词发现
    if keywords:
        found_data = True
        # 列出前 5 个最常见的关键词及其计数
        analysis_summary += f"- 主要相关关键词频率: " + ", ".join([f"{k}({v})" for k, v in keywords.most_common(5)]) + "\n"
    elif relevant_keywords: # 仅在预期有关键词时才提及缺少关键词
        analysis_summary += "- 未扫描到相关的关键词。\n"

    # 将摘要打印到控制台
    print(analysis_summary)
    print(f"{'=' * 20} 基本扫描结束 {'=' * 20}\n") # 打印页脚

    # 返回摘要字符串，如果未找到数据则返回特定消息
    return analysis_summary if found_data else "未在收集的信息中发现足够的可量化数据进行扫描分析。"


# --- 深度研究工作流（使用行业配置）---
def deep_research_workflow(initial_query, industry_config, max_iterations=3):
    """
    协调整个行业分析过程。
    它遵循一个循环：规划（可选）、搜索、整合和反思，
    最终进行数据分析（可选）和最终报告合成。

    Args:
        initial_query (str): 用户关于行业的起始问题。
        industry_config (dict): 所选行业的配置字典。
        max_iterations (int): 要执行的最大搜索-反思周期数（默认值：3）。

    Returns:
        str | None: 生成的最终报告文本（字符串形式），如果过程严重失败
                    （例如，未收集到信息、配置错误），则返回 None。
                    在某些失败情况下可能返回错误消息字符串。
    """
    industry_name = industry_config.get("name", "Selected Industry") # 获取行业显示名称
    logging.info(f"开始为查询 '{initial_query}' 进行 {industry_name} 分析")

    # 初始化内存结构
    # 'memory' 存储每个有用搜索结果的字典 {'subquery', 'url', 'name', 'summary', 'snippet'}
    memory = []
    # 'processed_urls' 跟踪已添加到内存的 URL 以避免重复
    processed_urls = set()
    # 'current_subqueries' 保存当前迭代中要搜索的问题列表
    current_subqueries = []
    # 'all_subqueries_history' 跟踪所有曾搜索过的子查询以防止重复搜索
    all_subqueries_history = set()

    # 从行业配置中检索必要的提示模板
    plan_template = industry_config.get("plan_prompt_template")
    reflection_template = industry_config.get("reflection_prompt_template")
    synthesis_template = industry_config.get("synthesis_prompt_template")

    # 验证配置中是否存在所有必需的模板
    if not all([plan_template, reflection_template, synthesis_template]):
        logging.error(f"行业 '{industry_name}' 的配置缺少一个或多个必需的提示模板。")
        print(f"错误：所选行业 '{industry_name}' 的配置不完整，缺少必要的提示模板。")
        return None # 关键配置错误，无法继续

    # --- 主要研究循环 ---
    for iteration in range(max_iterations):
        logging.info(f"--- 开始 {industry_name} 分析迭代 {iteration + 1} ---")

        # === 1. 规划阶段（生成子查询）===
        if iteration == 0:
            # 第一次迭代：根据用户的主要查询生成初始子查询
            plan_prompt = plan_template.format(
                industry_name=industry_name,
                initial_query=initial_query
            )
            logging.info(f"正在生成初始 {industry_name} 子查询...")
            # 调用 LLM，请求 JSON 输出以获取结构化的子查询
            llm_response = qwen_llm(
                plan_prompt,
                industry_config=industry_config, # 传递配置以获取系统提示
                response_format={"type": "json_object"} # 请求 JSON
            )

            # 处理潜在的 LLM 错误或无效的 JSON 响应
            if not llm_response:
                logging.error("从 LLM 获取初始规划响应失败。")
                current_subqueries = [initial_query] # 后备方案：使用原始查询
                logging.warning(f"回退到使用初始查询: {initial_query}")
            else:
                try:
                    plan_result = json.loads(llm_response) # 解析 JSON 响应
                    current_subqueries = plan_result.get('subqueries', []) # 提取子查询列表
                    # 验证提取的子查询
                    if not current_subqueries or not isinstance(current_subqueries, list):
                         logging.warning(f"LLM 返回了 JSON，但 'subqueries' 键丢失、无效或为空。响应: {llm_response}")
                         current_subqueries = [initial_query] # 后备方案
                         logging.warning(f"回退到使用初始查询: {initial_query}")
                    else:
                        # 成功生成子查询
                        current_subqueries = [q for q in current_subqueries if isinstance(q, str) and q.strip()] # 确保它们是非空字符串
                        logging.info(f"生成的初始子查询: {current_subqueries}")
                except json.JSONDecodeError as e:
                    logging.error(f"解码初始规划的 JSON 时失败: {e}。响应: {llm_response[:500]}...")
                    current_subqueries = [initial_query] # 后备方案
                    logging.warning(f"回退到使用初始查询: {initial_query}")
                except Exception as e:
                    logging.error(f"解析初始规划时发生意外错误: {e}", exc_info=True)
                    current_subqueries = [initial_query] # 后备方案
                    logging.warning(f"回退到使用初始查询: {initial_query}")

        elif not current_subqueries:
             # 后续迭代：如果之前的反思没有生成新的查询，则停止。
             logging.info("上一步反思未生成新的子查询。结束迭代周期。")
             break # 退出循环

        # 过滤掉先前迭代中已经搜索过的子查询
        subqueries_to_search = [q for q in current_subqueries if q and q not in all_subqueries_history]
        logging.info(f"本次迭代选择的子查询: {subqueries_to_search}")

        # 如果没有 *新的* 查询可搜索（且不是第一次迭代），则转到反思/合成
        if not subqueries_to_search and iteration > 0 :
             logging.info("所有生成的子查询都已被搜索或列表为空。转到反思阶段。")
             # 暂时不中断，让反思在当前内存上最后运行一次

        # === 2. 执行网络搜索 ===
        new_results_count = 0 # 跟踪本次迭代添加的结果数
        for subquery in subqueries_to_search:
            if not subquery: continue # 以防万一，跳过空查询
            logging.info(f"正在为 {industry_name} 信息搜索网络: '{subquery}'...")
            all_subqueries_history.add(subquery) # 将此查询标记为已搜索
            search_results = websearch(subquery) # 调用网络搜索函数
            time.sleep(1.5) # 短暂暂停以避免给搜索 API 带来过大压力

            # === 3. 整合和去重结果 ===
            for result in search_results:
                url = result.get('url')
                # 检查 URL 是否有效且之前未被处理过
                if url and url not in processed_urls:
                    processed_urls.add(url) # 将 URL 添加到已处理 URL 集合中
                    # 优先使用 'summary' 而不是 'snippet' 作为内容
                    summary = result.get('summary', '') or result.get('snippet', '')
                    if summary: # 仅添加具有某些文本内容的结果
                        memory.append({
                            "subquery": subquery,         # 找到此结果的查询
                            "url": url,                   # 来源 URL
                            "name": result.get('name', 'N/A'), # 页面标题
                            "summary": summary,           # 内容摘要/片段
                            "snippet": result.get('snippet', '') # 如果可用，也存储片段
                        })
                        new_results_count += 1
                    else:
                        logging.debug(f"跳过没有摘要/片段的结果: {url}")

        logging.info(f"添加了 {new_results_count} 个新的独立结果。总内存大小: {len(memory)}")

        # === 4. 准备反思上下文 ===
        # 创建一个字符串，总结当前收集到的信息，供 LLM 使用
        memory_context_for_llm = ""
        if memory:
             context_items = []
             token_estimate = 0 # 粗略估计以避免上下文过长
             max_tokens_estimate = 10000 # 限制发送给 LLM 的上下文大小（根据需要调整）
             # 反向迭代内存（最新的在前）
             for item in reversed(memory):
                 # 简洁地格式化每个内存项
                 item_text = f"  - 查询 '{item['subquery']}': {item['summary'][:250]}... (来源: {item['url']})\n" # 截断摘要
                 token_estimate += len(item_text) / 2 # 非常粗略的 token 估计（如果需要，调整系数）
                 if token_estimate > max_tokens_estimate:
                     logging.warning("用于反思的内存上下文因估计的 token 限制而被截断。")
                     break # 如果可能超过限制，则停止添加项目
                 context_items.append(item_text)
             # 连接项目，确保最近的项目出现在最终字符串的末尾
             memory_context_for_llm = "".join(reversed(context_items))
        else:
            # 如果尚未收集到任何信息的消息
            memory_context_for_llm = "当前没有收集到任何相关信息。"


        # === 5. 反思阶段（评估数据，生成新查询）===
        # 使用模板构建反思提示
        reflection_prompt = reflection_template.format(
            industry_name=industry_name,
            initial_query=initial_query,
            memory_context_for_llm=memory_context_for_llm
        )
        logging.info(f"正在反思收集到的 {industry_name} 数据...")
        # 调用 LLM，请求 JSON 输出以获取结构化的反思结果
        llm_response = qwen_llm(
            reflection_prompt,
            industry_config=industry_config, # 传递配置以获取系统提示
            response_format={"type": "json_object"} # 请求 JSON
            )

        # 解析反思响应前重置变量
        can_answer = False
        current_subqueries = [] # 这将保存反思建议的 *新* 子查询

        # 处理潜在的 LLM 错误或无效的 JSON 响应
        if not llm_response:
            logging.error("从 LLM 获取反思响应失败。")
            # 决定如何继续：也许中断，也许在没有新查询的情况下继续？
            # 目前，让它继续到下一次迭代或在达到最大迭代次数时进行合成。
        else:
            try:
                reflection_result = json.loads(llm_response) # 解析 JSON

                # 提取反思结果，带有默认值和类型检查
                can_answer = reflection_result.get('can_answer', False)
                irrelevant_urls_list = reflection_result.get('irrelevant_urls', [])
                new_subqueries_list = reflection_result.get('new_subqueries', [])

                # --- 验证反思结果 ---
                if not isinstance(can_answer, bool):
                    logging.warning(f"LLM 为 'can_answer' 返回了无效类型。默认为 False。值: {can_answer}")
                    can_answer = False

                # 确保 irrelevant_urls 是字符串列表，转换为集合以便高效查找
                if not isinstance(irrelevant_urls_list, list):
                     logging.warning(f"LLM 为 'irrelevant_urls' 返回了无效类型。默认为空列表。值: {irrelevant_urls_list}")
                     irrelevant_urls = set()
                else:
                     irrelevant_urls = set(u for u in irrelevant_urls_list if isinstance(u, str)) # 仅过滤字符串

                # 确保 new_subqueries 是非空字符串的列表
                if not isinstance(new_subqueries_list, list):
                    logging.warning(f"LLM 为 'new_subqueries' 返回了无效类型。默认为空列表。值: {new_subqueries_list}")
                    current_subqueries = []
                else:
                     # 过滤列表以仅保留非空字符串
                     current_subqueries = [q for q in new_subqueries_list if isinstance(q, str) and q.strip()]

                # --- 处理反思结果 ---
                logging.info(f"反思 - 能否回答 {industry_name} 查询: {can_answer}")

                # 根据识别出的不相关 URL 修剪内存
                if irrelevant_urls:
                    logging.info(f"反思 - 发现 {len(irrelevant_urls)} 个可能不相关的项目需要修剪。")
                    original_memory_size = len(memory)
                    # 创建一个新列表，仅包含 URL 不在 irrelevant_urls 中的项目
                    memory = [item for item in memory if item['url'] not in irrelevant_urls]
                    logging.info(f"内存从 {original_memory_size} 项修剪到 {len(memory)} 项。")

                # 检查 LLM 是否确定信息足够
                if can_answer:
                    logging.info(f"反思完成: {industry_name} 信息被认为足够。")
                    break # 退出主循环，继续进行合成

                # 检查是否生成了新的子查询
                if not current_subqueries and iteration < max_iterations - 1:
                    # 如果信息不足但没有新想法，记录警告。
                    # 循环将继续，如果未达到 max_iterations，可能会重试反思。
                    logging.warning("反思：信息不足，但未建议新的有效子查询。继续可能导致没有进展的循环。")
                elif current_subqueries:
                     logging.info(f"反思 - 需要新的 {industry_name} 子查询: {current_subqueries}")
                     # 这些 `current_subqueries` 将在 *下一个* 迭代的搜索阶段使用。

            except json.JSONDecodeError as e:
                logging.error(f"解码反思的 JSON 时失败: {e}。响应: {llm_response[:500]}...")
                # 让循环继续，也许下一次迭代效果更好。
            except Exception as e:
                logging.error(f"处理反思 JSON 时发生意外错误: {e}", exc_info=True)
                # 让循环继续。

        # 安全中断：如果达到最大迭代次数
        if iteration == max_iterations - 1:
            logging.warning("达到最大迭代次数。")
            if not can_answer:
                 # 如果完成迭代但 LLM 仍然认为信息不足
                 logging.warning(f"继续进行合成，尽管 {industry_name} 信息可能不完整。")
            # 循环在此自然终止


    # --- 迭代后处理 ---

    # 检查是否收集到了任何信息
    if not memory:
        logging.error(f"在 {max_iterations} 次迭代后未收集到 {industry_name} 信息。无法生成分析报告。")
        print(f"抱歉，经过搜索未能收集到足够的 {industry_name} 信息来生成分析报告。")
        return None # 指示失败，返回 None

    # === 6. 可选：数据分析 ===
    # 根据收集到的数据量决定是否运行基本数据分析器
    needs_analysis = len(memory) >= 4 # 如果我们至少有几个片段，则运行分析（阈值可调）
    analysis_summary = "" # 初始化分析摘要字符串

    if needs_analysis:
        logging.info("发现足够的数据，尝试进行基本数据扫描。")
        # 准备用于分析器函数的文本数据（摘要）
        texts_for_analysis = [item['summary'] for item in memory]
        # 调用分析器函数，传递文本和行业配置（用于关键词）
        analysis_summary = simple_data_analyzer(texts_for_analysis, industry_config)
    else:
        logging.info("数据不足以进行有意义的扫描，跳过数据分析。")

    # === 7. 合成最终答案 ===
    logging.info(f"正在准备最终 {industry_name} 报告的上下文...")
    # 将收集到的信息（内存）格式化为单个字符串，用于合成提示
    # 为每个项目包含 URL、源子查询、标题和摘要。
    final_memory_context = "\n\n".join([
        f"来源 URL: {item['url']}\n相关子问题: {item['subquery']}\n标题/名称: {item['name']}\n内容摘要: {item['summary']}"
        for item in memory
    ])

    # 准备要包含在提示中的分析部分，如果进行了分析并找到了数据
    analysis_section = ""
    if analysis_summary and "未在收集的信息中发现足够的可量化数据进行扫描分析" not in analysis_summary :
        # 在提示上下文中包含分析摘要
        analysis_section = f"\n\n补充数据扫描摘要:\n{analysis_summary}\n"

    # 使用模板构建最终的合成提示
    synthesis_prompt = synthesis_template.format(
        industry_name=industry_name,
        initial_query=initial_query,
        final_memory_context=final_memory_context, # 提供收集到的数据
        analysis_section=analysis_section         # 提供分析摘要（如果有）
    )

    # 调用流式 LLM（Deepseek）生成最终报告
    final_answer = deepseek_stream(synthesis_prompt, industry_config) # 传递配置以获取系统提示
    return final_answer # 返回生成的报告文本


# --- 示例用法 ---
if __name__ == "__main__":
    # 检查 API 密钥看起来是否可能有效（简单检查 'sk-'）
    # 这不是有效性的保证，只是一个基本检查。
    if "sk-" not in SEARCH_API_KEY or "sk-" not in LLM_API_KEY:
         logging.warning("API 密钥似乎丢失或无效。请设置 BOCHAAI_API_KEY 和 DASHSCOPE_API_KEY 环境变量或更新脚本中的默认值。")

    # --- 选择行业配置 ---
    # `SELECTED_INDUSTRY` 变量在脚本顶部附近定义。
    try:
        # 检索所选行业的配置字典
        current_industry_config = INDUSTRY_CONFIGS[SELECTED_INDUSTRY]
        logging.info(f"正在为行业运行分析: {current_industry_config['name']}")
    except KeyError:
        # 处理所选行业键在配置中不存在的情况
        logging.error(f"在 INDUSTRY_CONFIGS 中找不到行业 '{SELECTED_INDUSTRY}'。请检查配置和 SELECTED_INDUSTRY 变量。")
        exit() # 如果缺少配置，则停止脚本执行

    # --- 定义用户查询 ---
    # 根据所选行业设置初始用户查询。
    # 您可以自定义这些查询或动态提供一个。
    if SELECTED_INDUSTRY == "finance":
        # user_query = "分析一下今天中国A股市场的整体行情和主要特点"
        user_query = "2025年4月新能源汽车板块的市场表现和相关新闻有哪些？" # 更具体的查询
        # user_query = "分析一下最近美联储利率政策对科技股的影响"
    elif SELECTED_INDUSTRY == "tech":
        # user_query = "分析一下当前全球半导体行业的最新趋势和主要挑战是什么？"
        user_query = "2025年4月，大型语言模型 (LLM) 领域有哪些重要的技术突破或产品发布？" # 更具体的查询
        # user_query = "中国云计算市场的竞争格局如何？主要玩家有哪些？"
    else:
        # 如果所选行业在上面没有特定示例，则使用默认查询
        user_query = f"请分析一下 {current_industry_config['name']} 的最新动态和趋势"


    # --- 运行工作流 ---
    # 使用查询、配置和迭代限制调用主工作流函数
    final_report_text = deep_research_workflow(
        user_query,
        industry_config=current_industry_config,
        max_iterations=2 # 使用较少的迭代次数进行更快的测试/调试（例如 2）；增加以进行更彻底的分析（例如 3 或 4）
    )

    # --- 保存输出 ---
    # 检查工作流是否生成了有效的报告文本
    if final_report_text and "未能生成报告" not in final_report_text and "遇到错误" not in final_report_text:
        # 为报告创建一个文件名
        # 清理用户查询，使其可以安全地用作文件名
        safe_query_part = re.sub(r'[^\w\s-]', '', user_query[:30]).strip() # 保留字母数字、空格、连字符；限制长度
        safe_query_part = re.sub(r'[-\s]+', '_', safe_query_part) # 用下划线替换空格/连字符
        # 组合行业前缀和清理后的查询作为基本文件名
        base_filename = f"{current_industry_config.get('filename_prefix', '分析报告_')}{safe_query_part}"

        # 1. 将原始文本报告保存到 .txt 文件
        text_filename = f"{base_filename}.txt"
        try:
            with open(text_filename, "w", encoding="utf-8") as f:
                # 在文件开头写入一些元数据
                f.write(f"分析主题：{current_industry_config['name']}\n")
                f.write(f"分析问题：{user_query}\n\n")
                f.write("="*10 + " 分析报告 " + "="*10 + "\n\n")
                # 写入 LLM 生成的主要报告内容
                f.write(final_report_text)
            logging.info(f"原始文本报告已保存到 {text_filename}")
        except Exception as e:
            # 处理文件写入期间的潜在错误
            logging.error(f"保存文本报告到文件 {text_filename} 时出错: {e}")

    elif not final_report_text:
         # 处理工作流返回 None 的情况（严重失败）
         logging.error("工作流未生成报告（返回 None）。跳过文件保存。")
    else:
         # 处理工作流返回错误消息字符串的情况
         logging.warning(f"工作流返回了错误消息或可能不完整的报告: '{final_report_text[:100]}...'")
         # 可选地，将错误消息本身保存到文件以供调试
         try:
             error_filename = f"{current_industry_config.get('filename_prefix', '错误报告_')}_error.txt"
             with open(error_filename, "w", encoding="utf-8") as f:
                f.write(f"分析主题：{current_industry_config['name']}\n")
                f.write(f"分析问题：{user_query}\n\n")
                f.write("错误或不完整信息：\n")
                f.write(final_report_text) # 写入从工作流收到的错误消息
             logging.info(f"错误消息已保存到 {error_filename}")
         except Exception as e:
             logging.error(f"保存错误消息到文件 {error_filename} 时出错: {e}")

# --- 脚本结束 ---