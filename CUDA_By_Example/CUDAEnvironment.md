在Visual Studio里面添加external header files
1. 右键Project -> Properties打开该Project的Property Pages
2. 在Property Pages左侧选择**VC++ Directories**
3. 点击General -> Include Directories并在这里添加所需header file的directory路径，例如：`C:\ProgramData\NVIDIA Corporation\CUDA Samples\v10.1\common\inc`

注意很奇怪的一点是我们要在**VC++ Directories**里面添加这个路径而不是在 **CUDA C/C++** 里面。
如果直接在**CUDA C/C++**里面添加路径是可以使用的，不过编译器会报Warning。
