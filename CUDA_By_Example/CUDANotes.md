## Kernel Call
**Kernel:** a function that executes on the device is typically called a kernel.

### Hello World in GPU
1. **Qualifier**: __global__ is a qualifier to alert the compiler that this function needs to compiled for running on the device rather than host.
2. **Triple angle brackets<<<blocksPerGrid, threadsPerBlock>>>**: angle brackets pass the argument to the runtime system. The first is blocks per grid and the second is thread per block.

#include <iostream>
__global__ void **helloWorld( void )** {

}

int main( void ){
	helloWorld<<<1, 1>>>();
	printf("Hello, World\n");
	return 0;
}

```
