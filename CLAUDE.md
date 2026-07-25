# 项目开发约定

## 执行环境

- **本机**：只做文件修改和 git 操作。**不在本机编译、不在本机执行脚本。**
- **开发机**：所有编译、训练、脚本执行都在开发机上完成。
- 工作目录在本机与开发机上映射为**同一路径**（`/home/scratch.gemsg_sw/gomo-claude-opus-5`），文件修改自动同步。
  **不要**用 `scp` / `rsync` 在两机之间显式拷贝文件。
- 开发机上推荐用 docker 运行：镜像 `nvcr.io/nvidia/pytorch:26.06-py3`，把工作目录映射进容器（容器内保持同一路径）。

## 文件读写边界

- 只允许读写**工作目录内**的文件，以及 `/tmp`、`$HOME/.claude/`。
- `$HOME` **不属于**工作目录，除 `$HOME/.claude/` 外，不要在 `$HOME` 下写任何文件。
- 需要越出上述范围读写时，**先和用户确认**。

## 训练与容灾

开发机有使用时长限制，随时可能需要切换机器。因此：

- 训练脚本必须**定期保存 checkpoint 与训练状态**（step/epoch、model、optimizer、lr scheduler、RNG state、dataloader 位置等），保证换机后能原地续训。
- 默认实现 **resume**：启动时自动检测最新 checkpoint 并从中恢复，而不是从头开始。
- 不要写"必须一口气跑完"的训练流程；任何长任务都要能中断后继续。

## Git

- 由 Claude 负责 `git commit`；**push 由用户手动执行**，不要自动 push。
- commit message及文档、代码注释 中**不得出现开发机的任何信息**：主机名、IP、用户名、绝对路径前缀、集群/调度器配置等。
- 提交邮箱：`gems@moonshot.ai`（在仓库内配置 `git config user.email`，不要改全局配置）。

## 其他

- 所有工作**详细记录**在项目根目录的 `WORKLOG.md`；**训练进度不要记入**该文档（loss、step 数、指标曲线等不写）。
