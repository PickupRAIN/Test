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
