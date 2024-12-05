import json

def read_jsonl_and_write_sentences(jsonl_file_path):
    # 从.jsonl文件中读取数据
    with open(jsonl_file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    # 获取输出文件的基础名称
    output_file_base_name = jsonl_file_path.rsplit('.', 1)[0]

    # 打开一个同名的.txt文件准备写入
    with open(f'{output_file_base_name}.txt', 'w', encoding='utf-8') as output_file:
        for line in lines:
            # 解析每一行的JSON对象
            data = json.loads(line)
            # 检查'sentence'键是否存在
            if 'sentence' in data:
                # 将'sentence'值写入到.txt文件中
                output_file.write(data['sentence'] + '\n')


for d in ["action", "color", "detail", "shape", "texture"]:
    read_jsonl_and_write_sentences(f'/home/Inseval/dimensions/multiple/{d}.jsonl')

