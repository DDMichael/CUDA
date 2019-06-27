## Kernel Call
**Kernel:** a function that executes on the device is typically called a kernel.

### Hello World in GPU
1. **Qualifier**: __global__ is a qualifier to alert the compiler that this function needs to compiled for running on the device rather than host.
2. **Triple angle brackets<<<blocksPerGrid, threadsPerBlock>>>**: angle brackets pass the argument to the runtime system. The first is blocks per grid and the second is thread per block.
```C++
#include <iostream>
__global__ void **helloWorld( void )** {

}

int main( void ){
	helloWorld<<<1, 1>>>();
	printf("Hello, World\n");
	return 0;
}
```
The above code showed us how to invoke kernel in our C++ code. We use __global__ qualifier defined a function that will be processed on the GPU named **helloWorld**. We then call this kernel function inside our main function with only 1 block and 1 thread.

