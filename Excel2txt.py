import pandas as pd

# 假设Excel文件路径和文件名
excel_path = "/Users/zhangdong/Desktop/2021-2023/filtered_final_data_summarized.xlsx"  # 示例文件路径，需要替换为真实路径

# 加载Excel文件
df = pd.read_excel(excel_path)

# 函数：为每行数据创建一个文本文件
def process_row_and_create_file(row):
    file_name = f"/Users/zhangdong/Desktop/2021-2023/diagnosis_reports/{row[0]}.txt"  # 第一列作为文件名
    content = f"{row[-3]} {row[-1]}"  # 提取倒数第三列和倒数第一列的内容
    with open(file_name, "w", encoding="utf-8") as file:
        file.write(content)
    return file_name

# 遍历每一行并执行函数
file_paths = df.apply(process_row_and_create_file, axis=1)

