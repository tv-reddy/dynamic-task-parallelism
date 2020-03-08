/*
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

#define MAX_DEPTH       16
#define INSERTION_SORT  32

// Pointers to the data, node and queue struct types
typedef struct data* data_t;
typedef struct node* node_t;
typedef struct queue* queue_t;

// Struct for every data item
// *array - the full array of values to be sorted
// *lptr - left pointer for the starting point
// *rptr - right pointer for the end point
struct data {
    unsigned int *array;
    unsigned int *lptr
    unsigned int *rptr;
}

// Struct for a node in the queue
// data_item - holds the information about a piece of work
// next - pointer to the next node in the linked-list
struct node {
    data_t data_item;
    node_t next;
}

// Struct a queue of nodes implemented as a singly linked-list
// count - number of nodes in the queue
// head - pointer to the head of the queue
// tail - pointer to the tail of the queue
struct queue {
    int count;
    node_t head;
    node_t tail;
}

// Creates new queue
queue_t create_queue(void) {
    queue_t new_queue;

    new_queue = (queue_t)malloc(sizeof(struct queue));
    new_queue->head = (node_t)malloc(sizeof(node_t));
    new_queue->tail = (node_t)malloc(sizeof(node_t));
    new_queue->count = 0;

    return new_queue;
}

// Destroy existing queue
// TODO: Cater for case when queue is non-empty
int destroy_queue(queue_t queue) {
    if(!queue)
        return -1;
    if(queue->count > 0)
        return -1;

    free(queue);
    return 0;
}

// Add new node into queue
int enqueue(queue_t queue, data_t data) {
    node_t new_task;
    new_task = (node_t)malloc(sizeof(struct node));
    new_task->data_item = (data_t)malloc(sizeof(struct data));
    new_task->data_item = data;
    new_task->next = NULL;


    if(queue->count == 0) {
        // If queue is empty, make the new task the head and tail
        queue->tail = new_task;
        queue->head = new_task;
    } else {
        // If queue is non-empty, set the next task as the new head
        new_task->next = queue->head;
        queue->head = new_task;
    }

    queue->count++;

    return 0;
}


node_t pop_from_tail(queue_t queue) {
    node_t task;
    new_task = (node_t)malloc(sizeof(struct node));
    if(queue->count == 0)
        return NULL;

    new_task = queue->tail;
    //TODO: Continue from here. Might need to use doubly linked-list to set the previous node as the new tail
    //queue->tail =
}

node_t steal_from_head(queue_t queue) {
    node_t task;
    new_task = (node_t)malloc(sizeof(struct node));
    // If queue is empty, there's no task to steal
    // If queue is element, don't steal to avoid contention with the queue's owner
    if(queue->count == 0 || queue->count==1)
        return NULL;

    // TODO: Use atomicCAS here to resolve contention between multiple threads trying to steal
    new_task = queue->head;
    queue->head = queue->head->next;
    return new_task;
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
__global__ void cdp_simple_quicksort(unsigned int *data, int left, int right, int depth)
{
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
        cudaStream_t s;
        cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking);
        cdp_simple_quicksort<<< 1, 1, 0, s >>>(data, left, nright, depth+1);
        cudaStreamDestroy(s);
    }

    // Launch a new block to sort the right part.
    if ((lptr-data) < right)
    {
        cudaStream_t s1;
        cudaStreamCreateWithFlags(&s1, cudaStreamNonBlocking);
        cdp_simple_quicksort<<< 1, 1, 0, s1 >>>(data, nleft, right, depth+1);
        cudaStreamDestroy(s1);
    }
}

////////////////////////////////////////////////////////////////////////////////
// Call the quicksort kernel from the host.
////////////////////////////////////////////////////////////////////////////////
void run_qsort(unsigned int *data, unsigned int nitems)
{
    // Prepare CDP for the max depth 'MAX_DEPTH'.
    checkCudaErrors(cudaDeviceSetLimit(cudaLimitDevRuntimeSyncDepth, MAX_DEPTH));

    // Launch on device
    int left = 0;
    int right = nitems-1;
    std::cout << "Launching kernel on the GPU" << std::endl;
    cdp_simple_quicksort<<< 1, 1 >>>(data, left, right, 0);
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


    // Execute
    std::cout << "Running quicksort on " << num_items << " elements" << std::endl;
    run_qsort(d_data, num_items);

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