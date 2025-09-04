import pandas as pd

# 尝试作为 Excel 文件读取
try:
    df = pd.read_excel('./data/data.csv', engine='openpyxl')
    df.to_csv('data.tsv', sep='\t', index=False)
    print("成功转换为 TSV 文件: data.tsv")
except Exception as e:
    print(f"读取失败: {e}")
    # 继续方法 2