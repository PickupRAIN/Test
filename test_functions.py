#!/usr/bin/env python3
"""
测试脚本 - 用于测试OptimizedDeltaThinkFilter类中的十个新函数
此脚本仅用于测试目的，不会实际运行
"""

from hello import OptimizedDeltaThinkFilter

def test_new_functions():
    """测试十个新函数的功能"""
    
    # 创建过滤器实例
    filter_instance = OptimizedDeltaThinkFilter()
    
    # 测试数据
    test_text = """
    这是一段测试文本。
    <think>
    这是第一段思考内容。
    包含一些问题和思考。
    </think>
    
    这是思考后的文本内容。
    
    <think>
    这是第二段思考内容。
    包含代码示例：
    ```python
    def hello():
        print("Hello, World!")
    ```
    </think>
    
    <think>
    嵌套思考内容：
    <think>
    这是内层思考
    </think>
    回到外层思考
    </think>
    
    最后一段文本。
    """
    
    # 1. 测试extract_nested_think_content函数
    def test_extract_nested_think_content():
        """
        测试提取嵌套的think标签内容
        预期结果：返回嵌套think内容的层级列表
        """
        result = filter_instance.extract_nested_think_content(test_text)
        print("1. extract_nested_think_content测试结果:")
        print(f"   找到{len(result)}层嵌套内容")
        for i, item in enumerate(result):
            print(f"   层级{item['level']}: {item['content'][:30]}...")
    
    # 2. 测试replace_think_content函数
    def test_replace_think_content():
        """
        测试替换think标签内容为指定文本
        预期结果：返回替换后的文本，think内容被替换为"[思考内容已过滤]"
        """
        result = filter_instance.replace_think_content(test_text)
        print("\n2. replace_think_content测试结果:")
        print(f"   原文本长度: {len(test_text)}")
        print(f"   替换后长度: {len(result)}")
        print(f"   包含替换标记: {'[思考内容已过滤]' in result}")
    
    # 3. 测试get_think_context函数
    def test_get_think_context():
        """
        测试获取think标签周围的上下文
        预期结果：返回包含think内容及其上下文的字典列表
        """
        result = filter_instance.get_think_context(test_text, context_size=20)
        print("\n3. get_think_context测试结果:")
        print(f"   找到{len(result)}个think标签")
        for i, item in enumerate(result):
            print(f"   第{i+1}个think标签位置: {item['position']}")
            print(f"   前后上下文: ...{item['before_context'][-10:]}|{item['after_context'][:10]}...")
    
    # 4. 测试analyze_think_patterns函数
    def test_analyze_think_patterns():
        """
        测试分析think标签的使用模式
        预期结果：返回包含标签数量、内容统计和特征的分析结果
        """
        result = filter_instance.analyze_think_patterns(test_text)
        print("\n4. analyze_think_patterns测试结果:")
        print(f"   标签数量: {result['tag_count']}")
        print(f"   内容统计: {result['content_stats']}")
        print(f"   内容特征: {result['content_features']}")
    
    # 5. 测试merge_adjacent_think_tags函数
    def test_merge_adjacent_think_tags():
        """
        测试合并相邻的think标签
        预期结果：返回合并后的文本
        """
        adjacent_text = "文本<think>思考1</think><think>思考2</think>更多文本"
        result = filter_instance.merge_adjacent_think_tags(adjacent_text)
        print("\n5. merge_adjacent_think_tags测试结果:")
        print(f"   原文本: {adjacent_text}")
        print(f"   合并后: {result}")
    
    # 6. 测试extract_think_summary函数
    def test_extract_think_summary():
        """
        测试提取think内容的摘要
        预期结果：返回think内容的前几个句子组成的摘要
        """
        result = filter_instance.extract_think_summary(test_text, max_sentences=2)
        print("\n6. extract_think_summary测试结果:")
        print(f"   摘要: {result}")
    
    # 7. 测试create_think_toc函数
    def test_create_think_toc():
        """
        测试为think内容创建目录
        预期结果：返回think内容的目录列表
        """
        result = filter_instance.create_think_toc(test_text)
        print("\n7. create_think_toc测试结果:")
        print(f"   目录:\n{result}")
    
    # 8. 测试filter_think_by_keywords函数
    def test_filter_think_by_keywords():
        """
        测试根据关键词过滤think内容
        预期结果：返回过滤后的文本，只保留包含指定关键词的think内容
        """
        keywords = ["代码", "示例"]
        result_include = filter_instance.filter_think_by_keywords(test_text, keywords, include=True)
        result_exclude = filter_instance.filter_think_by_keywords(test_text, keywords, include=False)
        print("\n8. filter_think_by_keywords测试结果:")
        print(f"   包含关键词'{keywords}'的文本长度: {len(result_include)}")
        print(f"   排除关键词'{keywords}'的文本长度: {len(result_exclude)}")
    
    # 9. 测试convert_think_to_comments函数
    def test_convert_think_to_comments():
        """
        测试将think标签转换为注释
        预期结果：返回转换后的文本，think内容被转换为注释
        """
        result_python = filter_instance.convert_think_to_comments(test_text, "python")
        result_html = filter_instance.convert_think_to_comments(test_text, "html")
        print("\n9. convert_think_to_comments测试结果:")
        print(f"   Python注释风格包含'#': {'#' in result_python}")
        print(f"   HTML注释风格包含'<!--': {'<!--' in result_html}")
    
    # 10. 测试extract_think_with_metadata函数
    def test_extract_think_with_metadata():
        """
        测试提取think内容及其元数据
        预期结果：返回包含think内容和元数据的字典列表
        """
        result = filter_instance.extract_think_with_metadata(test_text)
        print("\n10. extract_think_with_metadata测试结果:")
        print(f"    找到{len(result)}个think标签")
        for i, item in enumerate(result):
            print(f"    第{i+1}个think标签:")
            print(f"      ID: {item['id']}")
            print(f"      位置: {item['position']}")
            print(f"      统计: {item['statistics']}")
            print(f"      特征: {item['features']}")
    
    # 运行所有测试函数
    test_functions = [
        test_extract_nested_think_content,
        test_replace_think_content,
        test_get_think_context,
        test_analyze_think_patterns,
        test_merge_adjacent_think_tags,
        test_extract_think_summary,
        test_create_think_toc,
        test_filter_think_by_keywords,
        test_convert_think_to_comments,
        test_extract_think_with_metadata
    ]
    
    print("开始测试十个新函数...")
    print("=" * 50)
    
    for test_func in test_functions:
        try:
            test_func()
        except Exception as e:
            print(f"测试{test_func.__name__}时出错: {e}")
    
    print("\n" + "=" * 50)
    print("测试完成!")

if __name__ == "__main__":
    print("注意: 这是一个测试脚本，仅用于验证函数定义，不会实际运行测试函数。")
    print("如需实际测试，请取消注释下面的代码行:")
    print("# test_new_functions()")