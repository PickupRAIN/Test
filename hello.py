"""
Think标签过滤器模块 - 支持LlamaIndex流式响应
"""

from typing import Tuple, Any, Union, Optional
import json
import re

class OptimizedDeltaThinkFilter:
    """
    优化版Delta Think标签过滤器
    专门处理LlamaIndex等框架的流式响应
    """
    
    def __init__(self, content_key: str = "text"):
        """
        初始化过滤器
        
        Args:
            content_key: 从LlamaIndex响应中提取内容的键名
        """
        self.buffer = ""          # 用于缓冲可能跨chunk的标签
        self.in_think = False     # 当前是否在think标签内
        self.output_text = ""     # 累积的输出文本
        self.content_key = content_key  # 内容键名
        self._reset_state()
    
    def _reset_state(self):
        """重置内部状态"""
        self.buffer = ""
        self.in_think = False
        self.output_text = ""
    
    def _extract_llamaindex_content(self, chunk: Any) -> str:
        """
        从LlamaIndex流式响应中提取文本内容
        
        Args:
            chunk: LlamaIndex的astream_complete返回的chunk
            
        Returns:
            提取的文本内容字符串
        """
        if chunk is None:
            return ""
        
        # 如果是字符串，直接返回
        if isinstance(chunk, str):
            return chunk
        
        # LlamaIndex常见的Response/CompletionResponse对象
        try:
            # 尝试访问.delta属性
            if hasattr(chunk, 'delta'):
                delta = chunk.delta
                if isinstance(delta, str):
                    return delta
                elif hasattr(delta, 'text'):
                    return delta.text
                elif hasattr(delta, 'content'):
                    return delta.content
            
            # 尝试访问.text属性
            if hasattr(chunk, 'text'):
                text = chunk.text
                if isinstance(text, str):
                    return text
            
            # 尝试访问.content属性
            if hasattr(chunk, 'content'):
                content = chunk.content
                if isinstance(content, str):
                    return content
            
            # 尝试访问.response属性
            if hasattr(chunk, 'response'):
                response = chunk.response
                if hasattr(response, 'text'):
                    return response.text
                elif hasattr(response, 'content'):
                    return response.content
            
            # 尝试直接访问对象的字符串表示
            if hasattr(chunk, '__str__'):
                str_repr = str(chunk)
                # 检查是否包含常见字段
                if 'text=' in str_repr or 'delta=' in str_repr or 'content=' in str_repr:
                    # 尝试提取引号内的内容
                    text_match = re.search(r'text=[\'"](.*?)[\'"]', str_repr)
                    if text_match:
                        return text_match.group(1)
                    
                    delta_match = re.search(r'delta=[\'"](.*?)[\'"]', str_repr)
                    if delta_match:
                        return delta_match.group(1)
                    
                    content_match = re.search(r'content=[\'"](.*?)[\'"]', str_repr)
                    if content_match:
                        return content_match.group(1)
            
        except Exception as e:
            # 调试信息
            print(f"Warning: Error extracting from LlamaIndex chunk: {e}")
            print(f"Chunk type: {type(chunk)}")
            print(f"Chunk attributes: {dir(chunk) if hasattr(chunk, '__dir__') else 'N/A'}")
        
        # 如果是字典
        if isinstance(chunk, dict):
            # 尝试常见键名
            for key in ['text', 'delta', 'content', 'response', 'message']:
                if key in chunk:
                    value = chunk[key]
                    if isinstance(value, str):
                        return value
                    elif isinstance(value, dict):
                        # 递归查找
                        for subkey in ['text', 'content']:
                            if subkey in value:
                                subvalue = value[subkey]
                                if isinstance(subvalue, str):
                                    return subvalue
        
        # 最后尝试转换为字符串
        try:
            return str(chunk)
        except:
            return ""
    
    def process_delta(self, chunk: Any) -> Tuple[str, str, bool]:
        """
        处理LlamaIndex流式chunk，过滤think标签
        
        Args:
            chunk: LlamaIndex的astream_complete返回的chunk
            
        Returns:
            Tuple[str, str, bool]: 
                - delta_output: 本次过滤后的增量输出
                - full_text: 当前完整的输出文本
                - has_output: 本次是否有输出
        """
        # 提取文本内容
        delta_text = self._extract_llamaindex_content(chunk)
        
        if not delta_text:
            return "", self.output_text, False
        
        # 添加到buffer
        self.buffer += delta_text
        output_delta = ""
        
        # 使用状态机处理
        i = 0
        while i < len(self.buffer):
            if not self.in_think:
                # 寻找开始标签
                start_idx = self.buffer.lower().find('<think>', i)
                if start_idx == -1:
                    # FIXED: 检查是否有孤立的结束标签
                    end_idx = self.buffer.lower().find('</think>', i)
                    if end_idx != -1:
                        # FIXED: 如果发现孤立结束标签，说明之前的内容都是think内容，应该丢弃
                        # 直接跳过结束标签之前的所有内容
                        # print(f"DEBUG: 发现孤立结束标签，丢弃 {self.buffer[i:end_idx]}")
                        i = end_idx + len('</think>')  # 只跳过结束标签
                        continue
                    else:
                        # 没有think标签，剩余都是有效文本
                        valid_text = self.buffer[i:]
                        output_delta += valid_text
                        self.output_text += valid_text
                        i = len(self.buffer)
                else:
                    # 输出think标签前的部分
                    valid_before = self.buffer[i:start_idx]
                    output_delta += valid_before
                    self.output_text += valid_before
                    
                    # 进入think模式
                    self.in_think = True
                    i = start_idx + len('<think>')
                    
                    # 处理立即闭合的标签
                    if self.buffer.lower().find('</think>', i) == i:
                        i += len('</think>')
                        self.in_think = False
            else:
                # 在think标签内，查找结束标签
                end_idx = self.buffer.lower().find('</think>', i)
                if end_idx == -1:
                    # 结束标签还没到，跳过所有内容
                    i = len(self.buffer)
                else:
                    # 找到结束标签
                    self.in_think = False
                    i = end_idx + len('</think>')
        
        # 清理已处理的buffer
        self.buffer = self.buffer[i:] if i < len(self.buffer) else ""
        
        has_output = len(output_delta) > 0
        return output_delta, self.output_text, has_output


    def process_delta_robust(self, chunk: Any) -> Tuple[str, str, bool]:
        """
        更健壮的处理方法，使用正则表达式
        
        Args:
            chunk: LlamaIndex流式chunk
            
        Returns:
            Tuple[str, str, bool]: 处理结果
        """
        # 提取文本
        delta_text = self._extract_llamaindex_content(chunk)
        
        if not delta_text:
            return "", self.output_text, False
        
        # 添加到buffer
        self.buffer += delta_text
        
        output_delta = ""
        processed_text = ""
        
        # 状态机处理
        while True:
            if not self.in_think:
                # 寻找开始标签
                start_match = re.search(r'<think>', self.buffer, re.IGNORECASE)
                if not start_match:
                    # FIXED: 检查是否有孤立的结束标签
                    end_match = re.search(r'</think>', self.buffer, re.IGNORECASE)
                    if end_match:
                        # FIXED: 如果发现孤立结束标签，说明之前的内容都是think内容，应该丢弃
                        # 只移除结束标签，之前的内容已在上个chunk中被跳过
                        end_pos = end_match.start()
                        # print(f"DEBUG: 发现孤立结束标签，位置 {end_pos}")
                        self.buffer = self.buffer[end_pos + len('</think>'):]
                        continue
                    else:
                        # 既没有开始也没有结束标签，全部输出
                        processed_text = self.buffer
                        self.buffer = ""
                        break
                
                # 输出开始标签之前的内容
                start_pos = start_match.start()
                output_before = self.buffer[:start_pos]
                processed_text += output_before
                
                # 移动buffer
                self.buffer = self.buffer[start_pos + len('<think>'):]
                self.in_think = True
            else:
                # 在think标签内，寻找结束标签
                end_match = re.search(r'</think>', self.buffer, re.IGNORECASE)
                if not end_match:
                    # 结束标签不在这个buffer中
                    # 清空buffer（think内容），等待下一个chunk
                    self.buffer = ""
                    break
                
                # 找到结束标签
                end_pos = end_match.start()
                # 跳过think内容
                self.buffer = self.buffer[end_pos + len('</think>'):]
                self.in_think = False
        
        # 更新输出
        output_delta = processed_text
        self.output_text += processed_text
        
        has_output = len(output_delta) > 0
        return output_delta, self.output_text, has_output
    
    def process_with_metadata(self, chunk: Any) -> dict:
        """
        处理chunk，返回包含元数据的结果
        
        Args:
            chunk: LlamaIndex流式chunk
            
        Returns:
            dict: 包含处理结果的字典
        """
        # 先提取原始内容
        original_content = self._extract_llamaindex_content(chunk)
        
        # 处理内容
        delta_output, full_text, has_output = self.process_delta_robust(original_content)
        
        return {
            "delta": delta_output,
            "full_text": full_text,
            "has_output": has_output,
            "filtered": len(original_content) > 0 and len(delta_output) == 0,
            "in_think": self.in_think,
            "buffer_size": len(self.buffer),
            "original_content": original_content,
            "original_type": type(chunk).__name__,
            "chunk_raw": str(chunk)[:100]  # 截取前100字符用于调试
        }
    
    def reset(self):
        """重置过滤器状态"""
        self._reset_state()
        return self
    
    def get_state(self) -> dict:
        """获取当前状态"""
        return {
            "in_think": self.in_think,
            "output_text": self.output_text,
            "buffer": self.buffer,
            "buffer_size": len(self.buffer)
        }
    # 以下用于增量测试
    
    def process_text_batch(self, text_chunks: list) -> Tuple[str, str, bool]:
        """
        批量处理文本块，适用于非流式场景
        
        Args:
            text_chunks: 文本块列表
            
        Returns:
            Tuple[str, str, bool]: 最终输出、完整文本、是否有输出
        """
        self.reset()
        final_output = ""
        
        for chunk in text_chunks:
            delta, full_text, has_output = self.process_delta_robust(chunk)
            if has_output:
                final_output += delta
        
        return final_output, self.output_text, len(final_output) > 0
    
    def extract_think_content(self, text: str) -> str:
        """
        从文本中提取think标签内的内容
        
        Args:
            text: 输入文本
            
        Returns:
            str: think标签内的内容
        """
        think_pattern = re.compile(r'<think>(.*?)</think>', re.DOTALL | re.IGNORECASE)
        matches = think_pattern.findall(text)
        return "\n".join(matches)
    
    def remove_all_think_tags(self, text: str) -> str:
        """
        从文本中移除所有think标签及其内容
        
        Args:
            text: 输入文本
            
        Returns:
            str: 移除think标签后的文本
        """
        think_pattern = re.compile(r'<think>.*?</think>', re.DOTALL | re.IGNORECASE)
        return think_pattern.sub('', text)
    
    def count_think_tags(self, text: str) -> int:
        """
        计算文本中think标签的数量
        
        Args:
            text: 输入文本
            
        Returns:
            int: think标签的数量
        """
        open_tags = len(re.findall(r'<think>', text, re.IGNORECASE))
        close_tags = len(re.findall(r'</think>', text, re.IGNORECASE))
        return min(open_tags, close_tags)
    
    def validate_think_tags(self, text: str) -> dict:
        """
        验证文本中think标签的配对情况
        
        Args:
            text: 输入文本
            
        Returns:
            dict: 验证结果
        """
        open_tags = len(re.findall(r'<think>', text, re.IGNORECASE))
        close_tags = len(re.findall(r'</think>', text, re.IGNORECASE))
        
        return {
            "is_valid": open_tags == close_tags,
            "open_count": open_tags,
            "close_count": close_tags,
            "mismatch": abs(open_tags - close_tags)
        }
    
    def split_by_think_tags(self, text: str) -> list:
        """
        根据think标签分割文本
        
        Args:
            text: 输入文本
            
        Returns:
            list: 分割后的文本片段列表，交替包含非think内容和think内容
        """
        parts = []
        think_pattern = re.compile(r'(<think>.*?</think>)', re.DOTALL | re.IGNORECASE)
        
        segments = think_pattern.split(text)
        
        # 第一个片段是非think内容
        if segments[0]:
            parts.append({"type": "content", "text": segments[0]})
        
        # 后续片段交替为think内容和非think内容
        for i in range(1, len(segments)):
            if i % 2 == 1:  # think内容
                parts.append({"type": "think", "text": segments[i]})
            else:  # 非think内容
                if segments[i]:
                    parts.append({"type": "content", "text": segments[i]})
        
        return parts
    
    def get_statistics(self) -> dict:
        """
        获取过滤器的统计信息
        
        Returns:
            dict: 统计信息
        """
        return {
            "state": self.get_state(),
            "content_length": len(self.output_text),
            "buffer_length": len(self.buffer),
            "is_in_think": self.in_think
        }
    
    def create_custom_filter(self, start_tag: str = "<think>", end_tag: str = "</think>"):
        """
        创建一个使用自定义标签的过滤器实例
        
        Args:
            start_tag: 开始标签
            end_tag: 结束标签
            
        Returns:
            CustomTagFilter: 自定义标签过滤器实例
        """
        return CustomTagFilter(start_tag, end_tag, self.content_key)


class CustomTagFilter:
    """
    自定义标签过滤器，基于OptimizedDeltaThinkFilter的实现
    """
    
    def __init__(self, start_tag: str, end_tag: str, content_key: str = "text"):
        """
        初始化自定义标签过滤器
        
        Args:
            start_tag: 开始标签
            end_tag: 结束标签
            content_key: 内容键名
        """
        self.start_tag = start_tag
        self.end_tag = end_tag
        self.content_key = content_key
        self.buffer = ""
        self.in_tag = False
        self.output_text = ""
    
    def process_chunk(self, chunk: Any) -> Tuple[str, str, bool]:
        """
        处理流式chunk，过滤自定义标签
        
        Args:
            chunk: 流式响应chunk
            
        Returns:
            Tuple[str, str, bool]: 过滤结果
        """
        # 复用OptimizedDeltaThinkFilter的内容提取方法
        filter_instance = OptimizedDeltaThinkFilter(self.content_key)
        delta_text = filter_instance._extract_llamaindex_content(chunk)
        
        if not delta_text:
            return "", self.output_text, False
        
        self.buffer += delta_text
        output_delta = ""
        
        i = 0
        while i < len(self.buffer):
            if not self.in_tag:
                start_idx = self.buffer.lower().find(self.start_tag.lower(), i)
                if start_idx == -1:
                    valid_text = self.buffer[i:]
                    output_delta += valid_text
                    self.output_text += valid_text
                    i = len(self.buffer)
                else:
                    valid_before = self.buffer[i:start_idx]
                    output_delta += valid_before
                    self.output_text += valid_before
                    
                    self.in_tag = True
                    i = start_idx + len(self.start_tag)
                    
                    if self.buffer.lower().find(self.end_tag.lower(), i) == i:
                        i += len(self.end_tag)
                        self.in_tag = False
            else:
                end_idx = self.buffer.lower().find(self.end_tag.lower(), i)
                if end_idx == -1:
                    i = len(self.buffer)
                else:
                    self.in_tag = False
                    i = end_idx + len(self.end_tag)
        
        self.buffer = self.buffer[i:] if i < len(self.buffer) else ""
        
        has_output = len(output_delta) > 0
        return output_delta, self.output_text, has_output
    
    def reset(self):
        """重置过滤器状态"""
        self.buffer = ""
        self.in_tag = False
        self.output_text = ""
        return self

    # 新增的十个函数
    
    def extract_nested_think_content(self, text: str) -> list:
        """
        提取嵌套的think标签内容
        
        Args:
            text: 输入文本
            
        Returns:
            list: 嵌套think内容的层级列表
        """
        result = []
        stack = []
        current_level = 0
        
        # 使用正则表达式找到所有think标签
        pattern = re.compile(r'(</?think>)', re.IGNORECASE)
        segments = pattern.split(text)
        
        current_content = ""
        in_think = False
        
        for segment in segments:
            if segment.lower() == '<think>':
                if in_think:
                    # 嵌套开始
                    stack.append((current_level, current_content))
                    current_level += 1
                else:
                    in_think = True
                current_content = ""
            elif segment.lower() == '</think>':
                if in_think:
                    if stack:
                        # 嵌套结束
                        level, parent_content = stack.pop()
                        result.append({
                            "level": current_level,
                            "content": current_content
                        })
                        current_content = parent_content
                        current_level = level
                    else:
                        # 最外层结束
                        result.append({
                            "level": current_level,
                            "content": current_content
                        })
                        in_think = False
                        current_content = ""
                current_content = ""
            else:
                if in_think:
                    current_content += segment
        
        return result
    
    def replace_think_content(self, text: str, replacement: str = "[思考内容已过滤]") -> str:
        """
        替换think标签内容为指定文本
        
        Args:
            text: 输入文本
            replacement: 替换文本
            
        Returns:
            str: 替换后的文本
        """
        think_pattern = re.compile(r'<think>.*?</think>', re.DOTALL | re.IGNORECASE)
        return think_pattern.sub(replacement, text)
    
    def get_think_context(self, text: str, context_size: int = 50) -> list:
        """
        获取think标签周围的上下文
        
        Args:
            text: 输入文本
            context_size: 上下文大小（字符数）
            
        Returns:
            list: 包含think内容及其上下文的字典列表
        """
        result = []
        think_pattern = re.compile(r'<think>(.*?)</think>', re.DOTALL | re.IGNORECASE)
        
        for match in think_pattern.finditer(text):
            start, end = match.span()
            think_content = match.group(1)
            
            # 获取上下文
            context_start = max(0, start - context_size)
            context_end = min(len(text), end + context_size)
            
            result.append({
                "think_content": think_content,
                "before_context": text[context_start:start],
                "after_context": text[end:context_end],
                "full_context": text[context_start:context_end],
                "position": {
                    "start": start,
                    "end": end
                }
            })
        
        return result
    
    def analyze_think_patterns(self, text: str) -> dict:
        """
        分析think标签的使用模式
        
        Args:
            text: 输入文本
            
        Returns:
            dict: 分析结果
        """
        # 计算think标签数量
        open_tags = len(re.findall(r'<think>', text, re.IGNORECASE))
        close_tags = len(re.findall(r'</think>', text, re.IGNORECASE))
        
        # 提取所有think内容
        think_pattern = re.compile(r'<think>(.*?)</think>', re.DOTALL | re.IGNORECASE)
        think_contents = think_pattern.findall(text)
        
        # 计算统计信息
        total_chars = sum(len(content) for content in think_contents)
        avg_length = total_chars / len(think_contents) if think_contents else 0
        
        # 分析内容特征
        has_code = any('```' in content for content in think_contents)
        has_questions = any('?' in content for content in think_contents)
        has_lists = any(('* ' in content or '- ' in content or '1.' in content) for content in think_contents)
        
        return {
            "tag_count": {
                "open": open_tags,
                "close": close_tags,
                "valid_pairs": min(open_tags, close_tags)
            },
            "content_stats": {
                "total_think_blocks": len(think_contents),
                "total_characters": total_chars,
                "average_length": avg_length,
                "max_length": max((len(content) for content in think_contents), default=0),
                "min_length": min((len(content) for content in think_contents), default=0)
            },
            "content_features": {
                "contains_code": has_code,
                "contains_questions": has_questions,
                "contains_lists": has_lists
            }
        }
    
    def merge_adjacent_think_tags(self, text: str) -> str:
        """
        合并相邻的think标签
        
        Args:
            text: 输入文本
            
        Returns:
            str: 合并后的文本
        """
        # 将连续的think标签合并为一个
        pattern = re.compile(r'</think>\s*<think>', re.IGNORECASE | re.DOTALL)
        return pattern.sub('', text)
    
    def extract_think_summary(self, text: str, max_sentences: int = 3) -> str:
        """
        提取think内容的摘要
        
        Args:
            text: 输入文本
            max_sentences: 最大句子数
            
        Returns:
            str: think内容摘要
        """
        think_pattern = re.compile(r'<think>(.*?)</think>', re.DOTALL | re.IGNORECASE)
        think_contents = think_pattern.findall(text)
        
        if not think_contents:
            return ""
        
        # 合并所有think内容
        all_think_content = " ".join(think_contents)
        
        # 简单的句子分割
        sentences = re.split(r'[.!?]+', all_think_content)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        # 返回前N个句子
        summary_sentences = sentences[:max_sentences]
        return ". ".join(summary_sentences) + ("." if summary_sentences else "")
    
    def create_think_toc(self, text: str) -> str:
        """
        为think内容创建目录
        
        Args:
            text: 输入文本
            
        Returns:
            str: think内容目录
        """
        think_pattern = re.compile(r'<think>(.*?)</think>', re.DOTALL | re.IGNORECASE)
        think_contents = think_pattern.findall(text)
        
        if not think_contents:
            return "没有找到think标签内容"
        
        toc_lines = ["Think内容目录:"]
        
        for i, content in enumerate(think_contents, 1):
            # 提取第一行作为标题
            first_line = content.strip().split('\n')[0][:50]
            if len(first_line) < len(content.strip().split('\n')[0]):
                first_line += "..."
            
            toc_lines.append(f"{i}. {first_line}")
        
        return "\n".join(toc_lines)
    
    def filter_think_by_keywords(self, text: str, keywords: list, include: bool = True) -> str:
        """
        根据关键词过滤think内容
        
        Args:
            text: 输入文本
            keywords: 关键词列表
            include: True表示保留包含关键词的think，False表示排除包含关键词的think
            
        Returns:
            str: 过滤后的文本
        """
        think_pattern = re.compile(r'(<think>.*?</think>)', re.DOTALL | re.IGNORECASE)
        
        def replace_func(match):
            think_content = match.group(1)
            
            # 检查是否包含关键词
            has_keyword = any(keyword.lower() in think_content.lower() for keyword in keywords)
            
            # 根据include参数决定是否保留
            if include:
                return think_content if has_keyword else ""
            else:
                return "" if has_keyword else think_content
        
        return think_pattern.sub(replace_func, text)
    
    def convert_think_to_comments(self, text: str, comment_style: str = "python") -> str:
        """
        将think标签转换为注释
        
        Args:
            text: 输入文本
            comment_style: 注释风格 ("python", "javascript", "html", "custom")
            
        Returns:
            str: 转换后的文本
        """
        think_pattern = re.compile(r'<think>(.*?)</think>', re.DOTALL | re.IGNORECASE)
        
        # 根据注释风格确定注释符号
        comment_symbols = {
            "python": "# ",
            "javascript": "// ",
            "html": "<!-- ",
            "custom": "[COMMENT] "
        }
        
        comment_prefix = comment_symbols.get(comment_style, "# ")
        comment_suffix = "" if comment_style != "html" else " -->"
        
        def replace_func(match):
            content = match.group(1).strip()
            lines = content.split('\n')
            commented_lines = [f"{comment_prefix}{line}{comment_suffix}" for line in lines]
            return "\n".join(commented_lines)
        
        return think_pattern.sub(replace_func, text)
    
    def extract_think_with_metadata(self, text: str) -> list:
        """
        提取think内容及其元数据
        
        Args:
            text: 输入文本
            
        Returns:
            list: 包含think内容和元数据的字典列表
        """
        result = []
        think_pattern = re.compile(r'<think>(.*?)</think>', re.DOTALL | re.IGNORECASE)
        
        for i, match in enumerate(think_pattern.finditer(text)):
            start, end = match.span()
            content = match.group(1)
            
            # 分析内容特征
            sentences = len(re.split(r'[.!?]+', content))
            words = len(re.findall(r'\b\w+\b', content))
            chars = len(content)
            
            # 检测内容类型
            has_code = '```' in content
            has_questions = '?' in content
            has_lists = any(marker in content for marker in ['* ', '- ', '1.', '2.'])
            
            result.append({
                "id": i + 1,
                "content": content,
                "position": {
                    "start": start,
                    "end": end,
                    "length": end - start
                },
                "statistics": {
                    "sentences": sentences,
                    "words": words,
                    "characters": chars
                },
                "features": {
                    "has_code": has_code,
                    "has_questions": has_questions,
                    "has_lists": has_lists
                }
            })
        
        return result