#include <stdio.h>

__global__ void add( int a, int b, int *c ){
	*c = a + b;
}

int main(void){
	int c;
	int *dev_c; // pointer to device
	cudaMalloc( (void**)&dev_c, sizeof(int) );
	add<<<1, 1>>>( 2, 8, dev_c);
	cudaMemcpy( &c, dev_c, sizeof(int), cudaMemcpyDeviceToHost);
	printf( "2 + 7 = %d\n", c );
	cudaFree( dev_c );
	return 0;
}
