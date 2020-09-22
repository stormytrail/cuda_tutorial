#include <iostream>

using namespace std;

#define N  20

__global__ void add( int *a, int *b, int *c ) {
	int block_index = blockIdx.x + blockIdx.y * gridDim.x;
	int gap = gridDim.x * gridDim.y * blockDim.x * blockDim.y;
	int thread_index = block_index*(blockDim.x*blockDim.y) + threadIdx.x + threadIdx.y*blockDim.x;

	while (thread_index < N){
		c[thread_index] = a[thread_index] + b[thread_index];
		thread_index += gap;
	}
}

int main( void ) {
	int a[N], b[N], c[N];
	int *dev_a, *dev_b, *dev_c;

	cudaMalloc( (void**)&dev_a, N * sizeof(int) );
	cudaMalloc( (void**)&dev_b, N * sizeof(int) );
	cudaMalloc( (void**)&dev_c, N * sizeof(int) );

	for (int i=0; i<N; i++) {
		a[i] = -i;
		b[i] = i * i;
	}

	cudaMemcpy( dev_a, a, N * sizeof(int),cudaMemcpyHostToDevice );
	cudaMemcpy( dev_b, b, N * sizeof(int),cudaMemcpyHostToDevice );

	dim3 grid(2,2);
	dim3 blocks(2,2);

	add<<<grid,blocks>>>( dev_a, dev_b, dev_c );

	cudaMemcpy( c, dev_c, N * sizeof(int),cudaMemcpyDeviceToHost );

	for (int i = 0;i < N;i++){
		cout << a[i] << " " << b[i] << " " << c[i] << endl;
	}

	cudaFree( dev_a );
	cudaFree( dev_b );
	cudaFree( dev_c );

	return 0;
}
