import re

hm3d_pattern = re.compile(
    r"^[^#]+#"                                      # 句子部分，包含任何非#字符（包括标点符号）
    r"(?:[\w\-']+\/[A-Za-z]+(?:\s[\w\-']+\/[A-Za-z]+)*)"  # 标注部分，word/POS组合，允许连字符和撇号
    r"#(?:-?\d+\.\d+|nan)"                          # 第一个数值部分，浮点数或nan
    r"#(?:-?\d+\.\d+|nan)$",                        # 第二个数值部分，浮点数或nan
    re.IGNORECASE                                    # 忽略大小写
)