#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

using namespace std;

#define N 100

#define GRIDDIMx 2
#define GRIDDIMy 2

#define BLOCKDIMx 2
#define BLOCKDIMy 2

#define NUM_BLOCK GRIDDIMx*GRIDDIMy

#define GAP GRIDDIMx*GRIDDIMy*BLOCKDIMx*BLOCKDIMy
#define BLOCK_SIZE BLOCKDIMx*BLOCKDIMy


__global__ void dot_product(float* a,float*b,float*partial_c){
	__shared__ float cache[BLOCK_SIZE];

	int block_index = blockIdx.x + blockIdx.y * gridDim.x;
	int thread_id_in_grid = threadIdx.x + threadIdx.y*blockDim.x + block_index * BLOCK_SIZE;
	int thread_id_in_block = threadIdx.x + threadIdx.y * blockDim.x;

	int pos = thread_id_in_grid;

	float thread_buffer = 0;
	while (pos < N){
		thread_buffer += a[pos] * b[pos];
		pos += GAP;
	}
	cache[thread_id_in_block] = thread_buffer;

//	printf("%d %d %d %lf\n",thread_id_in_grid,block_index,thread_id_in_block,thread_buffer);

	__syncthreads();

	int range_thread_allowed = NUM_BLOCK / 2;
	//之前给自己找麻烦，把BLOCK_SIZE设置的不是2的幂，想了半天想不到怎么实现处理mid以及mid对应的另一个位置
	//因为没有对应的thread去赋初值
	while (range_thread_allowed != 0){
		if (thread_id_in_block < range_thread_allowed){
			cache[thread_id_in_block] += cache[thread_id_in_block + range_thread_allowed];
		}
		__syncthreads();
		range_thread_allowed /=2;
	}

	partial_c[block_index] = cache[0];

	return;
}


int main(){
	float *a,*b,c,*partial_c;
	float *dev_a,*dev_b,*dev_partial_c;

	//malloc memory
	//on host
	a = (float*)malloc(sizeof(float) * N);
	b = (float*)malloc(sizeof(float) * N);
	partial_c = (float*)malloc(sizeof(float) * NUM_BLOCK);
	//on device
	cudaMalloc((void**)&dev_a,sizeof(float) * N);
	cudaMalloc((void**)&dev_b,sizeof(float) * N);
	cudaMalloc((void**)&dev_partial_c,sizeof(float) * NUM_BLOCK);

	//init data
	for (int i = 0;i < N;i++){
		a[i] = i;
		b[i] = i * 2;
	}

	//host 2 device
	cudaMemcpy(dev_a,a,N * sizeof(float),cudaMemcpyHostToDevice);
	cudaMemcpy(dev_b,b,N * sizeof(float),cudaMemcpyHostToDevice);

	//computation
	dim3 grid_dim(GRIDDIMy,GRIDDIMx);
	dim3 block_dim(BLOCKDIMy,BLOCKDIMx);

	dot_product<<<grid_dim,block_dim>>>(dev_a,dev_b,dev_partial_c);

	//device 2 host
	cudaMemcpy(partial_c,dev_partial_c,sizeof(float)*NUM_BLOCK,cudaMemcpyDeviceToHost);

	cout << NUM_BLOCK << endl;
	cout << GAP << endl;

	c = 0;
	for (int i = 0;i < NUM_BLOCK;i++){
		c += partial_c[i];
//		cout << partial_c[i] << " ";
	}
	//cout << endl;

	cout << c << endl;

	float d = 0;
	for (int i = 0;i < N;i++){
//		cout << a[i] * b[i] << " ";
		d += a[i] * b[i];
	}
//	cout << endl;

	cout << d << endl;
	return 0;
}



