#include <stdio.h>

#define N 10

__global__ void vectorAdd(int *a, int *b, int *c){
	int tid = blockIdx.x;
	c[tid] = a[tid]+b[tid];
}

int main(){
	int a[N], b[N], c[N];
	int *dev_a, *dev_b, *dev_c;
	
	for(int i = 0; i < N; i++){
		*(a+i) = i;
		*(a+i) = i*i;
	}

	cudaMalloc( (void**)&dev_a, N*sizeof(int) );
	cudaMalloc( (void**)&dev_b, N*sizeof(int) );
	cudaMalloc( (void**)&dev_c, N*sizeof(int) );

	cudaMemcpy( dev_a, a, N*sizeof(int), cudaMemcpyHostToDevice );
	cudaMemcpy( dev_b, b, N*sizeof(int), cudaMemcpyHostToDevice );

	vectorAdd<<<N, 1>>>(a, b, c);

	cudaMemcpy( c, dev_c, N*sizeof(int), cudaMemcpyDeviceToHost );
	cudaFree(dev_a);
	cudaFree(dev_b);
	cudaFree(dev_c);

	for(int i = 0; i < N; i++){
		printf("%d + %d = %d\n", a[i], b[i], c[i]);
	}

	return 0;
}
