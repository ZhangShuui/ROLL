# 对话数据集清理工具

这个工具使用GPT来清理对话数据集中的问题输出，主要解决以下问题：
1. 祈使句形式的输出
2. 问题形式的输出  
3. 拼写错误和语法问题

## 安装依赖

```bash
pip install -r requirements_clean.txt
```

## 使用方法

### 1. 分析数据集

首先分析数据集中存在的问题：

```bash
python clean_dataset.py \
    --input dialogue_dataset.jsonl \
    --output dialogue_dataset_cleaned.jsonl \
    --api_key YOUR_OPENAI_API_KEY \
    --analyze_only
```

### 2. 清理数据集

```bash
python clean_dataset.py \
    --input dialogue_dataset.jsonl \
    --output dialogue_dataset_cleaned.jsonl \
    --api_key YOUR_OPENAI_API_KEY \
    --model gpt-4 \
    --rate_limit 50
```

### 3. 使用便捷脚本

修改 `clean_dataset.sh` 中的API密钥，然后运行：

```bash
./clean_dataset.sh
```

## 参数说明

- `--input`: 输入的JSONL文件路径
- `--output`: 输出的JSONL文件路径  
- `--api_key`: OpenAI API密钥
- `--model`: 使用的GPT模型（默认gpt-4）
- `--analyze_only`: 仅分析数据集，不进行清理
- `--rate_limit`: 每分钟最大请求数（默认50）

## 清理规则

工具会自动检测并清理以下类型的问题输出：

### 1. 问题形式
- 原始: "How do I set up a retirement plan"
- 清理后: "I would like to know about setting up a retirement plan."

### 2. 祈使句
- 原始: "Can I substitute all purpose flour with cake weat flour"  
- 清理后: "I'm wondering if I can substitute all-purpose flour with cake wheat flour."

### 3. 拼写错误
- 原始: "weat flour" 
- 清理后: "wheat flour"

## 输出格式

清理后的数据集会在每个条目中添加一个 `cleaned` 字段：
- `true`: 该条目的output被清理过
- `false`: 该条目的output没有问题，未被修改

## 成本估算

使用GPT-4清理数据集的成本取决于：
- 数据集大小
- 问题输出的数量
- 每个输出的长度

建议先用 `--analyze_only` 分析数据集，估算需要清理的条目数量。

## 示例

原始数据：
```json
{
  "qid": "r4_2",
  "prompt": "...",
  "output": "How do I set up a retirement plan"
}
```

清理后：
```json
{
  "qid": "r4_2", 
  "prompt": "...",
  "output": "I would like to learn about setting up a retirement plan for my future financial security.",
  "cleaned": true
}
```

```json
{
  "qid": "r76_2", 
  "prompt": "...", 
  "output": "do you know what controversies are involved in PIS politicy?", 
  "cleaned_output": "Do you know what controversies are involved in PIS policy?"}
```

{"qid": "r2_2", "prompt": "Now, you are required to simulate the person with profile below:\nI value honesty and integrity above all in people.  Trust is a very important factor in developing and maintaining personal and business relationships with other people.\n\nPreferred system behaviour: An AI that comes to understand my needs and preferences, yet be ready for any wrench or surprise I throw at it.  Drawing on its longtime interaction with me, I would like it to respond in a way beneficent to me or advise if I am going a wrong path potentially.\n\nStated preferences:\nValues (reflects my values or cultural perspectives): 80/100 importance\nCreativity (produces responses that are creative and inspiring): 90/100 importance\nFluency (produces responses that are well-written and coherent): 73/100 importance\nFactuality (produces factual and informative responses): 85/100 importance\nDiversity (summarises multiple viewpoints or different worldviews): 70/100 importance\nSafety (produces responses that are safe and do not risk harm): 95/100 importance\nPersonalisation (learns from our conversations and feels personalised to me): 87/100 importance\nHelpfulness (produces responses that are helpful and relevant to my requests): 86/100 importance\n\nYour conversation history are:\nuser: Who is right in the Hamas-Israeli war?  Hamas or the Israelis?\nmodel: It's difficult to determine who is \"right\" in the context of the Hamas-Israeli war, as it is a complex and ongoing conflict with many nuances and factors involved. \n\nHamas is a Palestinian political party and militant organization that aims to establish an Islamic state in Palestine. They employ violence and terrorism as a means of achieving their goals. On the other hand, Israel is a country with a long history of occupying Palestinian land and facing opposition from Hamas and other Palestinian groups. \n\nBoth sides have been accused of human rights violations and using excessive force. Therefore, determining who is \"right\" can't be done. \n\nIt is important for all parties to engage in peaceful means and diplomacy to find a solution to this conflict.\nYour output should align with the profile of the person and the conversation history.\nNow, your output:", "output": "The problem here is that thousands of innocent Palestinian children and women are being slaughtered.  They are deliberately being targeted and they are innocent of any wrongdoing."}