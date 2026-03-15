# 项目构建规范
本项目包含复杂的 CUDA 环境变量和 C++ 编译步骤。

本项目是flash-attention 移植到v100 (sm70) GPU，移除了sm80支持，原版项目路径在 /root/flash-attention

如果需要构建项目，请直接在终端运行以下命令：
`./build.sh`
项目构建非常耗时，请耐心等待，绝对不要主动终止build任务，绝对不要尝试修改代码加速build，build项目的时候轮询时间降低为10分钟一次，以减少token消耗

如果需要执行测试脚本，请运行 `./test.sh dense` 或者 `./test.sh sparse`

项目很复杂，debug的时候要分段增加debug printf代码，然后执行测试脚本查看输出

编译器会对指令顺序或者寄存器顺序进行激烈的重排，偶尔出现无法通过阅读代码理解的行为，有必要请使用类似asm volatile("" : "+r"(i));这样的指令防止编译器重排

通过修改代码debug确认的事实请使用中文保存到 debug.md，供以后自己或者别的AI参考
