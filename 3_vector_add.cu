#include <iostream>

using namespace std;

#define N   10

__global__ void add( int *a, int *b, int *c ) {
	int tid = blockIdx.x;
	if (tid < N){
		c[tid] = a[tid] + b[tid];
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

	add<<<N,1>>>( dev_a, dev_b, dev_c );
	//参数1:设备执行核函数使用的并行线程块数量(Block)
	//blockIdx.x
	//一个并行线程块集合



	cudaMemcpy( c, dev_c, N * sizeof(int),cudaMemcpyDeviceToHost );

	for (int i = 0;i < N;i++){
		cout << a[i] << " " << b[i] << " " << c[i] << endl;
	}

	cudaFree( dev_a );
	cudaFree( dev_b );
	cudaFree( dev_c );

	return 0;
}
