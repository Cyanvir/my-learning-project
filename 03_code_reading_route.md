# 代码阅读路线

目标不是逐行背代码，而是把“数据如何进入、模型如何训练、结果如何评估”串起来。

## 第 1 轮：只看大结构

打开这些文件，每个文件只回答一句话：

- `README.md`：项目想复现什么论文？
- `train.py`：训练入口在哪里？
- `trainer.py`：自定义了哪些 Trainer？
- `script/alignment/Vaccine.sh`：alignment 阶段怎么调用 `train.py`？
- `script/vaccine_finetune/sst2.sh`：后续有害微调怎么调用 `train.py`？
- `sst2/build_dataset.py`：SST2 数据被转换成什么格式？

这一轮不纠结细节。

## 第 2 轮：看数据格式

读 `sst2/build_dataset.py`，注意输出 JSON 的三个字段：

```json
{
  "instruction": "Analyze the sentiment...",
  "input": "movie sentence",
  "output": "positive"
}
```

再读 `train.py` 的 `SupervisedDataset`：

- `instruction` 和 `input` 会被拼成 prompt。
- `output` 会被作为模型需要学习生成的目标。
- prompt 部分的 label 被设为 `IGNORE_INDEX = -100`，意思是训练时不要求模型预测 prompt，只要求预测 response。

检查点问题：

- 为什么训练时不让模型学习预测 prompt？
- 如果一个样本没有 `input` 字段，prompt 会怎么构造？

## 第 3 轮：看模型加载和 LoRA

读 `train.py` 的 `train()`：

- `AutoModelForCausalLM.from_pretrained(...)` 加载基础模型。
- `AutoTokenizer.from_pretrained(...)` 加载 tokenizer。
- `LoraConfig(...)` 定义 LoRA 训练哪些模块。
- `get_peft_model(model, config)` 把模型包装成可训练 LoRA 的模型。

检查点问题：

- 为什么项目不直接全量微调所有参数？
- `target_modules` 里为什么有 `q_proj`、`k_proj`、`v_proj`、`o_proj`？

## 第 4 轮：看 Trainer 选择

读 `train.py` 靠近结尾的分支：

```python
if training_args.optimizer=="vaccine":
    trainer = VaccineTrainer(...)
elif "EWC" in training_args.optimizer:
    trainer = FITrainer(...)
elif training_args.optimizer == "KL":
    trainer = KLTrainer(...)
else:
    trainer = transformers.Trainer(...)
```

这说明作者没有重写整个训练框架，而是在 Hugging Face `Trainer` 上改 `training_step`。

检查点问题：

- 普通 SFT 和 Vaccine 最大差异在哪里？
- `--optimizer vaccine` 这个名字为什么有点误导？它实际不是 AdamW/SGD 这种优化器，而是选择训练步骤。

## 第 5 轮：重点读 `VaccineTrainer`

只看这几个函数：

- `training_step`
- `pre_first_step`
- `after_first_step`
- `pre_second_step`
- `after_second_step`
- `_grad_norm`

核心流程：

```text
1. 注册 backward hook，准备记录 hidden gradient
2. 第一次 backward，得到 hidden output 的梯度
3. 移除 hook，把梯度归一化成扰动
4. 清空梯度
5. 注册 forward hook，在 hidden output 上加扰动
6. 第二次 backward，用扰动后的 loss 更新参数
7. 移除 hook
```

检查点问题：

- 为什么第一次 backward 后要 `model.zero_grad()`？
- 为什么扰动被加在 attention 模块输出上，而不是直接加在输入文本上？
- `rho` 变大时，可能有什么好处和坏处？

## 第 6 轮：看实验脚本

读 `script/alignment/Vaccine.sh` 和 `script/alignment/SFT.sh`，比较：

- 数据是否相同？
- 模型是否相同？
- epoch、batch size、learning rate 是否相同？
- 唯一区别是不是 `--optimizer vaccine` 和 `--rho`？

读 `script/vaccine_finetune/sst2.sh` 和 `script/sft_finetune/sst2.sh`，比较：

- `poison_ratio` 是什么？
- `sample_num` 是什么？
- `benign_dataset data/sst2.json` 在做什么？

科研阅读代码时，这种“对照组比较”比逐行读更重要。

