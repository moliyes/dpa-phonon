# 内置模型

包内默认模型现在是 `src/phonon_kit/data/models/dpa4_0704.pt` checkpoint。
配置中使用 `model: builtin:dpa4` 即可引用；若当前机器已经运行过
`ph model freeze`，则优先使用本机缓存中的 `dpa4_0704.pt2`，否则回退到该
`.pt` checkpoint。

冻结模型依赖生成它的 DeepMD、PyTorch、CUDA 和机器环境，因此不再把 `.pt2`
作为可跨机器分发的内置资产。每位用户应在自己的 DeepMD 环境中执行：

```bash
ph model freeze
```

该文件不受 phonon-kit 的 MIT 许可证覆盖。公开仓库或向第三方分发前，请先确认
原模型许可证和再分发权限。checkpoint 校验值随包数据提供。
