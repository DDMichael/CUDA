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

## Passing parameters
Here we use simple add function to illustrate how to pass parameters into kernel.
**cudaMalloc( arg1, arg2):** 
1. arg1 is **a pointer to the pointer** you want to hold the address of the newly allocated memory. This isidentical behavior to *malloc()*, void pointer return type.
2. arg2 is the size of allocation you want to make.

**Note on use cudaMalloc():** Do not dereference the pointer returned by *cudaMalloc()* from the code executes on the host. Host code may pass the pointer around, perform arithmetic on it, or even cast it to a different type. But you **cannot** use it to read or write from memory

```C++
#include <iostream>

__global__ void simpleAdd(int a, int b, int *c){
	*c = a + b;
}

int main(){
	int c;
	int *dev_c;
	
	cudaMalloc( (void**)&dev_c, sizeof(int) );// &dev_c(GPU) ---> dev_c(CPU) ---> c(CPU)
	simpleAdd<<<1, 1>>>(2, 8, dev_c);

	cudaMemcpy( &c, dev_c, sizeof(int), cudaMemcpyDeviceToHost );
	
	cudaFree( dev_c );
	
	printf("2 + 8 = %d\n", c);
	return 0;
}
```

Host can **allocate** and **free** memory on the device, but host **cannot modify** that memory. Host can only access memory on device by copy the device pointer back to some memory on host by **cudaMemcpy()** method.

## Variable blockIdx.x
Use variable **blockIDx.x** will let the GPU know which block is currently running for a piece of code. There is **No need to define the blockIdx** because this is one of the built-in variables that the CUDA runtime defines.

### Why blockIdx.x instead of simple blockIdx
CUDA C allows programmer define **a group of blocks in two dimensions** For problems with 2D domains, such as matrix math or image processing, it is often convenient to use 2D indexing to avoid annoying translations from liner to rectangular indices.

```C++
__global__ void add(int *a, int *b, int *c){
	int tid = blockIdx.x;
	if (tid < N)
		c[tid] = a[tid] + b[tid];
}

#define N 10

int main(int argc, char **argv){
	int a[N], b[N], c[N];
	int *dev_a, *dev_b, *dev_c;
	...
	add<<<N, 1>>>(dev_a, dev_b, dev_c);
	...	
}
```
