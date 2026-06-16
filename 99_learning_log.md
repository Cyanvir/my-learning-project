# 学习记录

每次学习后写一点，不求漂亮，只求真实。科研入门最怕“看了很多但没有留下可复盘的东西”。

## 第 1 次：项目初识

日期：26.6.16

我今天读了：

- `README.md`

运行了：

```powershell
python learning\toy_vaccine_manual.py
```
运行结果：
```powershell
Tiny Vaccine demo
=================
Clean accuracy:
  normal training :  99.0%
  vaccine-style   :  98.0%
Accuracy under hidden perturbation:
  normal training :  94.0%
  vaccine-style   :  96.0%

What to notice:
- Vaccine-style training is not magic; it trains on a harder perturbed version of the hidden state.
- The real project applies this idea inside attention modules of LLMs with PyTorch hooks.
- In the real code, rho controls the perturbation strength.
```
我能解释的 3 个概念：

1. ```
 Vaccine 关注的是模型内部 hidden representation 的稳定性。
```
2. ```
clean accuracy 高不代表模型鲁棒，扰动后性能下降才暴露问题。
```
3. ```
Vaccine-style training 会故意制造更难的训练条件，让模型适应扰动
```

我还不懂的 3 个概念：

1. 
2. 
3. 

我发现的 1 个代码细节：

- 

我下一轮想问 Codex 的问题：

- 

## 第 2 次：数据和 prompt

日期：

我今天读了：

- 

我能解释的 3 个概念：

1. 
2. 
3. 

我还不懂的 3 个概念：

1. 
2. 
3. 

我发现的 1 个代码细节：

- 

下一步：

- 

## 第 3 次：VaccineTrainer

日期：

我今天读了：

- 

我能用自己的话解释 Vaccine 的训练流程：

1. 
2. 
3. 
4. 
5. 

我还不懂的地方：

- 

下一步：

- 

