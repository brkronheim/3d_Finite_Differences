/*
Compile and run inside a Google Colab notebook with a GPU using the commands:

!nvidia-smi
!nvcc  -o diffusion -x cu -lnvToolsExt drive/MyDrive/path/to/file/diffusionCUDARevised.cu
!./diffusion

The first line is not strictly necesary, but it lets us check what GPU we have,
probably a Tesla T4. 

The second line runs the compiller. It is recomended you put this file on your
Google Drive and mount your drive to the Colab session.

The third line runs the code. Every 100 steps it will print out the current step
number.
*/

#include <string>
#include <math.h> 
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <vector>
#include <nvToolsExt.h>
#include <stdio.h>
#include <ctime>


#define alpha 0.1 // Numerical fit paramter, making it too big will cause the PDE to not converge
#define dimX 100 // Number of x bins, not recomened to play with this
#define dimY 100 // Number of y bins
#define dimZ 1000 // Number of z bins
#define dimT 10000 // Number of t bins, may need to increase
#define DSIZE (dimX*dimY*dimZ) // 10 million
#define SAVESTEP 100  // Save every 100 steps, results in 100 saves, size is 4 GB
#define blocks 80 // Should be number of streaming multiprocessors x2 
#define threads 1024 // Should probably be 1024

#define cudaCheckErrors(msg)                                    \
    do {                                                        \
        cudaError_t __err = cudaGetLastError();                 \
        if (__err != cudaSuccess) {                             \
            fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n",  \
                    msg, cudaGetErrorString(__err),             \
                    __FILE__, __LINE__);                        \
            fprintf(stderr, "*** FAILED - ABORTING\n");         \
            exit(1);                                            \
        }                                                       \
    } while (0)


__global__ void finiteDiff(const float *inputVal, float *outputVal, int ds, int tStamp){
    
    for (int index = threadIdx.x+blockDim.x*blockIdx.x; index < ds; index+=gridDim.x*blockDim.x){   
        
        // Determine array location
        int x = index%dimX;
        int y = ((index-x)/dimX)%dimY;
        int z = (index - x - y*dimX)/dimY/dimX;
        
        // Assuming not a boundary of the array
        if(x>0 && y>0 && z>0 && x<dimX-1 && y<dimY-1 && z <dimZ-1){
            // The actual finite differences update
            outputVal[index] = inputVal[index];
            outputVal[index] += alpha*(inputVal[index-1]         + inputVal[index+1]         - 2*inputVal[index]); // Step the diffeq along the x dimension
            outputVal[index] += alpha*(inputVal[index-dimX]      + inputVal[index+dimX]      - 2*inputVal[index]); // Step the diffeq along the y dimension
            outputVal[index] += alpha*(inputVal[index-dimX*dimY] + inputVal[index+dimX*dimY] - 2*inputVal[index]); // Step the diffeq along the z dimension            
        } 
        else { // Fixed concentration outside
            outputVal[index] = inputVal[index];
        }
    }
}


//Take one step of the algorithm
void stepAlgo(float *d_inputVal, float *d_outputVal, int tStamp){

    // Run the main algorithm
    finiteDiff<<<blocks, threads>>>(d_inputVal, d_outputVal, DSIZE, tStamp);
    cudaCheckErrors("main kernel launch failure");
    cudaDeviceSynchronize();

}

void run(float *inputVal, float *saveData){
    int counter = 1;
    
    // Declare device pointers
    float *d_inputVal;
    float *d_outputVal;
    float *d_saveData; 

    // Allocate memory on the gpu
    cudaMalloc(&d_inputVal, DSIZE*sizeof(float));
    cudaMalloc(&d_outputVal, DSIZE*sizeof(float));
    cudaMalloc(&d_saveData, DSIZE*(dimT/SAVESTEP)*sizeof(float));
    cudaCheckErrors("cudaMalloc failure"); // error checking

    // Copy data to the GPU
    cudaMemcpy(d_inputVal, inputVal, DSIZE*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_outputVal, inputVal, DSIZE*sizeof(float), cudaMemcpyHostToDevice);
    cudaCheckErrors("cudaMemcpy H2D failure");
    
    
    std::time_t msStart = std::time(nullptr);
    while(counter<=dimT) {
        // If it is time to save the data, use it in the finite differences instead so nothing needs copied
        if((counter-1)%1000==0){
            std::cout << counter - 1 << std::endl;
            // Run 1 step of the algorithm
            stepAlgo(d_inputVal, d_saveData+DSIZE*((counter-1)/SAVESTEP), counter-1);
            counter++;
    
            // Run another step but with the input and output arrays flipped so
            // the memory doesn't need copied        
            stepAlgo(d_saveData+DSIZE*((counter-2)/SAVESTEP), d_inputVal, counter-1);
            counter++;
        } else {
            // Run 1 step of the algorithm
            stepAlgo(d_inputVal, d_outputVal, counter-1);
            counter++;
            
            // Run another step but with the input and output arrays flipped so
            // the memory doesn't need copied
            stepAlgo(d_outputVal, d_inputVal, counter-1);
            counter++;
        }

    }
    std::time_t msEnd = std::time(nullptr);
    
    // Give timing information
    std::cout << double(msEnd-msStart)*double(1000)/double(counter) << " ms per step\n";
   
    // Copy data off the GPU
    cudaMemcpy(inputVal, d_inputVal, DSIZE*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(saveData, d_saveData, DSIZE*(dimT/SAVESTEP)*sizeof(float), cudaMemcpyDeviceToHost);
    
    cudaCheckErrors("kernel execution failure or cudaMemcpy H2D failure");
    
    // Free the memory on the GPU
    cudaFree(d_inputVal);
    cudaFree(d_outputVal);
    cudaFree(d_saveData);

    

}

int main(int argc, char *argv[]){
    
    // Allocate arrays for data storage
    float *inputArray = new float[DSIZE];
    float *saveData = new float[DSIZE*(dimT/SAVESTEP)];

    float inside = 0;  // Concentration inside
    float outside = 1; // Concentration outside

    // Initialize the first array
    for(int x=0; x<dimX; x++){
        for(int y=0; y<dimY; y++){
            for(int z=0; z<dimZ; z++){
                int index = x + dimX*(y+z*dimY); // Location in flat array
                if(x>0 && y>0 && z>0 && x<dimX-1 && y<dimY-1 && z <dimZ-1){
                    // inside
                    inputArray[index] = inside;
                } else {
                    //outside
                    inputArray[index] = outside;
                }
            }
        }
    }


    // Run the algorithm
    run(inputArray, saveData);
    
    // Store the data in a binary file
    // This can be opened in python with:
    // np.fromfile("data.dat", dtype=np.float32)
    // data = np.reshape(data,(100,1000,100,100))
    // Shape is (t, z, y, x)
    FILE *data = fopen("data.dat", "wb");
    fwrite(saveData, sizeof(float), DSIZE*(dimT/SAVESTEP), data);
    fclose(data);

    // Free memory
    free(inputArray);
    free(saveData);
    
    return(0);
}    

