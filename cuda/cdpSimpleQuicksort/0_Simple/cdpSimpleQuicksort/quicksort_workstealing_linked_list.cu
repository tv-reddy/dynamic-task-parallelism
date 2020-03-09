le/*
* Copyright 1993-2014 NVIDIA Corporation.  All rights reserved.
*
* Please refer to the NVIDIA end user license agreement (EULA) associated
* with this source code for terms and conditions that govern your use of
* this software. Any use, reproduction, disclosure, or distribution of
* this software and related documentation outside the terms of the EULA
* is strictly prohibited.
*
*/
#include <iostream>
#include <cstdio>
#include <helper_cuda.h>
#include <helper_string.h>
#include <thrust/device_vector.h>

#define MAX_DEPTH       16
#define MAX_TASKS       6
#define INSERTION_SORT  32

struct Task {
    int* array;
    unsigned int left;
    unsigned int right;
    unsigned int depth;
};

struct Head {
    unsigned short index;
    unsigned short ctr;
}

struct Deque {
    Head head;
    unsigned int tail;
    Task tasks[MAX_TASKS];    
}

typedef struct Task* Task_t;
typedef struct Deque* Deque_t;
typedef struct Head* Head_t;


// __global__
// Deque deques[MAX_THREAD_BLOCKS];
// Each block should create it's own queue


// to push the task on to the work queue
__device__ 
void push(Deque_t queue, Task newTask) 
{
    if (queue->tail < MAX_TASKS)
    {
        queue->tasks[queue->tail] = newTask;
        queue->tail++;
    }
    else
    {
        printf("Work queue full!\n");
    }
}

// pop the task from the work queque
__device__
Task pop(Deque_t queue)
{
    Head oldHead, newHead;
    unsigned int oldTail;
    Task task;

    if(queue->tail == 0)
        return NULL;

    queue->tail--;
    task = tasks[tail];

    oldHead = queue->head;
    if(queue->tail > oldHead.index)
        return task;
    
    oldTail = queue->tail;
    queue->tail = 0;
    newHead.index = 0;
    newHead.ctr = oldHead.ctr + 1;

    if(oldTail == oldHead.index)
        if(atomicCAS(&(queue->head),oldHead,newHead))
            return task;

    head = newHead;
    return NULL;
}

// to steal tasks from the work queue
__device__ 
Task steal(Deque_t queue) 
{
    Head oldHead, newHead;
    Task task;

    oldHead = queue->head;
    if(tail <= oldHead.index)
        return NULL;
    
    task = queue->tasks[oldHead.index];

    newHead = oldHead;
    newHead.index++;
    if( atomicCAS(&(queue->head),oldHead,newHead))
        return task;
    
    // fix this
    return NULL;
}

////////////////////////////////////////////////////////////////////////////////
// Selection sort used when depth gets too big or the number of elements drops
// below a threshold.
////////////////////////////////////////////////////////////////////////////////
__device__ void selection_sort(unsigned int *data, int left, int right)
{
    for (int i = left ; i <= right ; ++i)
    {
        unsigned min_val = data[i];
        int min_idx = i;

        // Find the smallest value in the range [left, right].
        for (int j = i+1 ; j <= right ; ++j)
        {
            unsigned val_j = data[j];

            if (val_j < min_val)
            {
                min_idx = j;
                min_val = val_j;
            }
        }

        // Swap the values.
        if (i != min_idx)
        {
            data[min_idx] = data[i];
            data[i] = min_val;
        }
    }
}

////////////////////////////////////////////////////////////////////////////////
// Very basic quicksort algorithm, recursively launching the next level.
////////////////////////////////////////////////////////////////////////////////
__global__ void cdp_simple_quicksort(unsigned int *data,
                                    int left, int right,
                                    int depth, thrust::device_vector<task> d_queue,
                                    unsigned int num_items)
{
    
    if (blockIdx.x == 0)
    {
        // create sperate queues for each block in global memory
        __global__
        Deque_t queue;
        
        // TODO: Initialize the pointers
        queue->tail = 0;
        queue->head = 0;
        // If we're too deep or there are few elements left, we use an insertion sort...
        if (depth >= MAX_DEPTH || right-left <= INSERTION_SORT)
        {
            selection_sort(data, left, right);
            return;
        }

        unsigned int *lptr = data+left;
        unsigned int *rptr = data+right;
        unsigned int  pivot = data[(left+right)/2];

        // Do the partitioning.
        while (lptr <= rptr)
        {
            // Find the next left- and right-hand values to swap
            unsigned int lval = *lptr;
            unsigned int rval = *rptr;

            // Move the left pointer as long as the pointed element is smaller than the pivot.
            while (lval < pivot)
            {
                lptr++;
                lval = *lptr;
            }

            // Move the right pointer as long as the pointed element is larger than the pivot.
            while (rval > pivot)
            {
                rptr--;
                rval = *rptr;
            }

            // If the swap points are valid, do the swap!
            if (lptr <= rptr)
            {
                *lptr++ = rval;
                *rptr-- = lval;
            }
        }

        // Now the recursive part
        int nright = rptr - data;
        int nleft  = lptr - data;

        // Launch a new block to sort the left part.
        if (left < (rptr-data))
        {
            // cudaStream_t s;
            // cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking);
            // cdp_simple_quicksort<<< 1, 1, 0, s >>>(data, left, nright, depth+1);
            // cudaStreamDestroy(s);

            task_t task1;
            task1 = (task_t)malloc(sizeof(struct task));
            task1->array = (int*)malloc((nright - left) * sizeof(int))
            task1->left_ptr = (int*)malloc(sizeof(int));
            task1->right_ptr = (int*)malloc(sizeof(int));
            // TODO: point to the subarray 
            task1->array = data;
            // TODO: left and right limit
            task1->left = left;
            task1->left = nright;
            // TODO: depth
            task1->depth = depth + 1;

            // TODO: push this task to the queue
            push(queue, task1);
        }

        // Launch a new block to sort the right part.
        if ((lptr-data) < right)
        {
            // cudaStream_t s1;
            // cudaStreamCreateWithFlags(&s1, cudaStreamNonBlocking);
            // cdp_simple_quicksort<<< 1, 1, 0, s1 >>>(data, nleft, right, depth+1);
            // cudaStreamDestroy(s1);

            task_t task2;
            task2 = (task_t)malloc(sizeof(struct task));
            task2->array = (int*)malloc((nright - left) * sizeof(int))
            task2->left_ptr = (int*)malloc(sizeof(int));
            task2->right_ptr = (int*)malloc(sizeof(int));
            // TODO: point to the subarray 
            task2->array = data;
            // TODO: left and right limit
            task2->left = nleft;
            task2->left = right;
            // TODO: depth
            task2->depth = depth + 1;
            // TODO: push this task to the queue
            push(queue, task2);
        }

        // the parent block pops the first task
        // TODO: pop the task
        pop(queue);
        // TODO: launch the task
        cudaStream_t s;
        cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking);
        cdp_simple_quicksort<<< 2, 1, 0, s >>>(data, left, nright, depth+1);
        cudaStreamDestroy(s);
    }

    // second task is stolen by the consumer
    if(blockIdx.x == 1)
    {
        // TODO: steal the task
        steal(queue);
        // TODO: launch the task
        cudaStream_t s1;
        cudaStreamCreateWithFlags(&s1, cudaStreamNonBlocking);
        cdp_simple_quicksort<<< 2, 1, 0, s1 >>>(data, nleft, right, depth+1);
        cudaStreamDestroy(s1);
    }

    // TODO: freeup the memory
    free();
  
}

////////////////////////////////////////////////////////////////////////////////
// Call the quicksort kernel from the host.
////////////////////////////////////////////////////////////////////////////////
void run_qsort(unsigned int *data, unsigned int nitems, thrust::device_vector<task> d_queue)
{
    // Prepare CDP for the max depth 'MAX_DEPTH'.
    checkCudaErrors(cudaDeviceSetLimit(cudaLimitDevRuntimeSyncDepth, MAX_DEPTH));

    // Launch on device
    int left = 0;
    int right = nitems-1;
    std::cout << "Launching kernel on the GPU" << std::endl;
    cdp_simple_quicksort<<< 2, 1 >>>(data, left, right, 0, d_queue, nitems);
    checkCudaErrors(cudaDeviceSynchronize());
}

////////////////////////////////////////////////////////////////////////////////
// Initialize data on the host.
////////////////////////////////////////////////////////////////////////////////
void initialize_data(unsigned int *dst, unsigned int nitems)
{
    // Fixed seed for illustration
    srand(2047);

    // Fill dst with random values
    for (unsigned i = 0 ; i < nitems ; i++)
        dst[i] = rand() % nitems ;
}

////////////////////////////////////////////////////////////////////////////////
// Verify the results.
////////////////////////////////////////////////////////////////////////////////
void check_results(int n, unsigned int *results_d)
{
    unsigned int *results_h = new unsigned[n];
    checkCudaErrors(cudaMemcpy(results_h, results_d, n*sizeof(unsigned), cudaMemcpyDeviceToHost));

    for (int i = 1 ; i < n ; ++i)
        if (results_h[i-1] > results_h[i])
        {
            std::cout << "Invalid item[" << i-1 << "]: " << results_h[i-1] << " greater than " << results_h[i] << std::endl;
            exit(EXIT_FAILURE);
        }

    std::cout << "OK" << std::endl;
    delete[] results_h;
}

////////////////////////////////////////////////////////////////////////////////
// Main entry point.
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv)
{
    int num_items = 128;
    bool verbose = false;

    if (checkCmdLineFlag(argc, (const char **)argv, "help") ||
        checkCmdLineFlag(argc, (const char **)argv, "h"))
    {
        std::cerr << "Usage: " << argv[0] << " num_items=<num_items>\twhere num_items is the number of items to sort" << std::endl;
        exit(EXIT_SUCCESS);
    }

    if (checkCmdLineFlag(argc, (const char **)argv, "v"))
    {
        verbose = true;
    }

    if (checkCmdLineFlag(argc, (const char **)argv, "num_items"))
    {
        num_items = getCmdLineArgumentInt(argc, (const char **)argv, "num_items");

        if (num_items < 1)
        {
            std::cerr << "ERROR: num_items has to be greater than 1" << std::endl;
            exit(EXIT_FAILURE);
        }
    }

    // Get device properties
    int device_count = 0, device = -1;

    if(checkCmdLineFlag(argc, (const char **)argv, "device"))
    {
        device = getCmdLineArgumentInt(argc, (const char **)argv, "device");

        cudaDeviceProp properties;
        checkCudaErrors(cudaGetDeviceProperties(&properties, device));

        if (properties.major > 3 || (properties.major == 3 && properties.minor >= 5))
        {
            std::cout << "Running on GPU " << device << " (" << properties.name << ")" << std::endl;
        }
        else
        {
            std::cout << "ERROR: cdpsimpleQuicksort requires GPU devices with compute SM 3.5 or higher."<< std::endl;
            std::cout << "Current GPU device has compute SM" << properties.major <<"."<< properties.minor <<". Exiting..." << std::endl;
            exit(EXIT_FAILURE);
        }

    }
    else
    {
        checkCudaErrors(cudaGetDeviceCount(&device_count));

        for (int i = 0 ; i < device_count ; ++i)
        {
            cudaDeviceProp properties;
            checkCudaErrors(cudaGetDeviceProperties(&properties, i));

            if (properties.major > 3 || (properties.major == 3 && properties.minor >= 5))
            {
                device = i;
                std::cout << "Running on GPU " << i << " (" << properties.name << ")" << std::endl;
                break;
            }

            std::cout << "GPU " << i << " (" << properties.name << ") does not support CUDA Dynamic Parallelism" << std::endl;
         }
     }

    if (device == -1)
    {
        std::cerr << "cdpSimpleQuicksort requires GPU devices with compute SM 3.5 or higher.  Exiting..." << std::endl;
        exit(EXIT_SUCCESS);
    }

    cudaSetDevice(device);

    // Create input data
    unsigned int *h_data = 0;
    unsigned int *d_data = 0;

    // Allocate CPU memory and initialize data.
    std::cout << "Initializing data:" << std::endl;
    h_data =(unsigned int *)malloc(num_items*sizeof(unsigned int));
    initialize_data(h_data, num_items);

    if (verbose)
    {
        for (int i=0 ; i<num_items ; i++)
            std::cout << "Data [" << i << "]: " << h_data[i] << std::endl;
    }

    // Allocate GPU memory.
    checkCudaErrors(cudaMalloc((void **)&d_data, num_items * sizeof(unsigned int)));
    checkCudaErrors(cudaMemcpy(d_data, h_data, num_items * sizeof(unsigned int), cudaMemcpyHostToDevice));

    // Create thrust vectors to be used as task queues
    //thrust::host_vector<task> host_task_queue_1(0);
    thrust::host_vector<task> host_task_queue[MAX_DEPTH];
    //thrust::device_vector<task> device_task_queue_1 = host_task_queue_1;
    thrust::device_vector<task> device_task_queue = host_task_queue;

    // Execute
    std::cout << "Running quicksort on " << num_items << " elements" << std::endl;
    run_qsort(d_data, num_items, device_task_queue);

    // Copy result from GPU back to CPU
    unsigned int *results_h = new unsigned[num_items];
    checkCudaErrors(cudaMemcpy(results_h, d_data, num_items*sizeof(unsigned), cudaMemcpyDeviceToHost));

    // Check result
    std::cout << "Validating results: ";
    check_results(num_items, d_data);

    // Print result
    std::cout<<"[";
    for(int i = 0; i < num_items; i++) {
        std::cout<<results_h[i];
        if(i < num_items -1) {
            std::cout<<", ";
        }
    }
    std::cout<<"]"<<std::endl;

    free(h_data);
    checkCudaErrors(cudaFree(d_data));

    // cudaDeviceReset causes the driver to clean up all state. While
    // not mandatory in normal operation, it is good practice.  It is also
    // needed to ensure correct operation when the application is being
    // profiled. Calling cudaDeviceReset causes all profile data to be
    // flushed before the application exits
    cudaDeviceReset();
    exit(EXIT_SUCCESS);
}