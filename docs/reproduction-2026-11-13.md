# gfx936 复现环境与兼容修复

本次以 v1.0 的 a03f43fd74be3c4ebf618dc47aa630795b364233 为基线，在 f12r1n09 的八张 gfx936 上复现。以下结果包含 VAE 接口适配，不代表原始提交可无修改运行。

## 已验证环境

- Python 3.11.15；厂商 PyTorch 2.9.0+das.opt1.dtk2604。
- DTK 26.04、HIP 6.3.26093、GCC 11.4、xmake 3.1.2。原环境 xmake 2.9.8 无法满足此次构建，建议使用已验证版本；--builddir 在 3.1.2 有效。
- Triton 3.5.1、Diffusers 0.40.0、Transformers 5.14.1、Accelerate 1.14.0、PyAV 18.1。
- FlashAttention 厂商 wheel 元数据为 2.8.3+das.opt1.dtk2604.torch290，模块版本显示 2.6.1。不能仅以模块版本认定不同安装包等价。

使用与 DTK、PyTorch 配套的厂商 wheel。setup_env.sh 自动定位 flash_attn_2_cuda 扩展，并为厂商 SDPA 设置 FA_SO_PATH；用户已设置的 FA_SO_PATH 保留。build_infinicore.sh 将扩展所在目录传给 xmake 的 --flash-attn，也允许显式设置 FLASH_ATTN_ROOT 为相应 site-packages 目录。链接时保留扩展依赖，仅在存在独立 libflash_attention.so 时额外链接该库。

## 模型包与输入

本次模型来自共享目录 /work2/ai_data/models/MiniMax/MiniMax-H3。实际运行目录的 diffusers 下需能访问 text_encoder、transformer、video_vae、audio_vae；本次用相对链接指向同级 FL2VA 组件。已检查权重索引分片存在，未对全部权重哈希。

VAE 包装器未暴露管线需要的外层配置。修复读取对应组件 config.json 中已有的 latents_mean、latents_std，将底层视频模型的 vae_ratio、clip_length 以及音频模型的 sample_rate 转接给管线；适配音频 decode 返回格式并转换输入至权重 dtype。没有修改权重或归一化数值。该适配针对本次发布包，尚未验证其他 VAE 实现。

首帧从报告 DOCX 提取，尚未获得原始 Golde33443.jpg，因此未证明输入像素一致。官方参照视频未建立对应关系。本次 SSIM 只与同输入、同 seed 的本地 legacy 输出比较。

缓存可通过 NINETOOTHED_CACHE_DIR 设置；未设置时仍使用 ~/.ninetoothed。

## 验证结果和边界

完整构建及官方 Hygon SwiGLU 测试通过。契约检查 17 项通过，1 个模块因缺少 ninetoothed.targets 跳过。正式路径无 --legacy 或视觉回退；两路使用 50 步、124 帧、576×384、24 FPS、seed 42、提示词 A dog falls asleep。

| 指标 | 报告 | 本次正式路径 | 本次 legacy |
| --- | ---: | ---: | ---: |
| Transformer（49 次） | 150.177 秒 | 150.889 秒 | 119.190 秒 |
| Pipeline | 161.897 秒 | 165.982 秒 | 135.881 秒 |
| 端到端 | 233.814 秒 | 238.707 秒 | 217.848 秒 |
| 相对本地 legacy 的 SSIM | 82.767% | 60.178% | 参照 |

以上为单次观测，不能作为严格性能回归统计。legacy 的 Python FlashAttention 接口曾报告 _wrapped_flash_attn_backward 缺失并回退，最终经厂商 SDPA 路径完成，不能视为原报告环境的精确副本。

两段媒体规格均通过：124 帧、576×384、24 FPS，含双声道音轨。正式路径第 43–46 帧（从零计数）出现花屏，legacy 对应帧未见同类异常。正式进程保存视频后发生 Segmentation fault，legacy 正常退出。本补丁解决构建和接口阻断，**未修复花屏、退出崩溃，也未复现报告的 SSIM**。

脚本遇到段错误返回失败是正确行为，不应隐藏退出码。如文件已保存，可独立执行 scripts/verify_fl2va_output.py 检查媒体；规格通过不等于进程正常退出或模型数值正确。
