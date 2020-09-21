//#include <iostream>

//using namespace std;

#include <stdio.h>


__global__ void add( int a, int b, int *c ) {
	*c = a + b;
}

int main(){
	int c;			//c in host
	int *dev_c;		//c in device
	
	//malloc memory in device
	//params : 1.address;2.size of memory
	//return void*
	cudaMalloc((void**)&dev_c,sizeof(int));

	add<<<1,1>>>( 2, 7, dev_c );

	//context switch
	cudaMemcpy(&c,dev_c,sizeof(int),cudaMemcpyDeviceToHost);

	cudaFree(dev_c);

	return 0;
}

