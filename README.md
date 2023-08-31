# JuejueziGen

## Description

利用MindSpore框架，对预训练好的语言模型进行RLHF下游微调，结合自行收集的中文互联网平台小红书的“绝绝子”语调，引导语言模型的生成结果具有“绝绝子”语调的风格

举例：
> 投递日常💙 拿来吧你 今天去进行MindSpore语言模型RLHF微调了 不是吧 咩咩咩 这家的MindSpore语言模型RLHF微调太牛了❕❕❕这个蜜雪冰城也太🉑了⁉⁉⁉绝了 u1s1 呀呀呀 好想谈一场双向奔赴的恋爱🥝

## TODO List

- 8.30

- [x] 确定选题和方法
- [x] 调研RLHF训练代码库
- [x] 制作小红书绝绝子数据集
- [ ] 调研chatGLM中文大模型实现

- 8.31

- [ ] 整理绝绝子数据集格式，训练reward model
- [ ] RLHF训练生成模型主训练代码：mindspore实现
- [ ] huggingface开源项目trl（强化学习）代码：mindspore实现
- [ ] 实现RLHF和ChatGLM集成