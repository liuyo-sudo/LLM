# 导入 pandas 库，用于生成和保存 CSV 数据
import pandas as pd
# 导入 random 模块，用于随机选择问答
import random

# 定义生成健康问答数据集的函数
def generate_health_qa_data(output_file="health.csv", num_samples=1000):
    # 定义问题列表，模拟医疗领域的问题
    questions = [
        "心脏病有什么症状？", "甲状腺结节如何治疗？", "痔疮手术需要多久恢复？",
        "肠道息肉需要切除吗？", "脂肪瘤会恶化吗？", "肝癌如何早期发现？"
    ]
    # 定义答案列表，对应医疗问题的答案
    answers = [
        "心脏病的症状包括胸痛、气短、心悸等，建议及时就医检查。",
        "甲状腺结节治疗包括药物治疗、手术切除等，需根据结节性质决定。",
        "痔疮手术通常需要7-14天恢复，具体取决于手术方式和患者体质。",
        "肠道息肉是否切除取决于大小和性质，建议定期内镜检查。",
        "脂肪瘤通常为良性，但若增长迅速或伴疼痛，需就医评估。",
        "肝癌早期可通过超声、甲胎蛋白检测等发现，定期体检很重要。"
    ]
    # 初始化数据列表，用于存储问答对
    data = []
    # 循环生成 num_samples 条数据
    for _ in range(num_samples):
        # 随机选择一个问题
        q = random.choice(questions)
        # 随机选择一个答案
        a = random.choice(answers)
        # 将问答对添加到数据列表
        data.append({"question": q, "answer": a})
    # 将数据转换为 pandas DataFrame
    df = pd.DataFrame(data)
    # 将 DataFrame 保存为 CSV 文件，编码为 UTF-8
    df.to_csv(output_file, index=False, encoding="utf-8")
    # 打印保存成功的提示信息
    print(f"数据集已生成并保存到 {output_file}")

# 示例代码：生成数据集
if __name__ == "__main__":
    # 调用生成数据集函数
    generate_health_qa_data()