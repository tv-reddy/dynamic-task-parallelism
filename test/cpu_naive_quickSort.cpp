/* C++ implementation of QuickSort */
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <stdlib.h>
#define ARRAY_SIZE 1024
#define FILE_NAME "data.txt"

using namespace std;  

// A utility function to swap two elements  
void swap(int* a, int* b)  
{  
    int t = *a;  
    *a = *b;  
    *b = t;  
}  
  
/* This function takes last element as pivot, places  
the pivot element at its correct position in sorted  
array, and places all smaller (smaller than pivot)  
to left of pivot and all greater elements to right  
of pivot */
int partition (int arr[], int low, int high)  
{  
    int pivot = arr[high]; // pivot  
    int i = (low - 1); // Index of smaller element  
  
    for (int j = low; j <= high - 1; j++)  
    {  
        // If current element is smaller than the pivot  
        if (arr[j] < pivot)  
        {  
            i++; // increment index of smaller element  
            swap(&arr[i], &arr[j]);  
        }  
    }  
    swap(&arr[i + 1], &arr[high]);  
    return (i + 1);  
}  
  
/* The main function that implements QuickSort  
arr[] --> Array to be sorted,  
low --> Starting index,  
high --> Ending index */
void quickSort(int arr[], int low, int high)  
{  
    if (low < high)  
    {  
        /* pi is partitioning index, arr[p] is now  
        at right place */
        int pi = partition(arr, low, high);  
  
        // Separately sort elements before  
        // partition and after partition  
        quickSort(arr, low, pi - 1);  
        quickSort(arr, pi + 1, high);  
    }  
}  
  
/* Function to print an array */
void printArray(int arr[], int size)  
{  
    int i;  
    for (i = 0; i < size; i++)  
        cout << arr[i] << " ";  
    cout << endl;  
} 

void fileRead(int *arr)
{
    ifstream myfile ("data.txt");

    int idx = 0;
    string line;
    if (myfile.is_open())
    {
        while(!myfile.eof())
        {
            getline(myfile,line,'\n');
            //cout << line << endl;
            arr[idx] = stoi(line);
            idx++;
        }
        myfile.close();
    }
    else cout << "Unable to open file"; 

    //cout << "idx" << idx << endl;
    //printArray(array, ARRAY_SIZE);
}
  
// Driver Code 
int main()  
{  
    int arr[ARRAY_SIZE];
    fileRead(arr);
    quickSort(arr, 0, ARRAY_SIZE - 1);  
    cout << "Sorted array: \n";  
    printArray(arr, ARRAY_SIZE);  
    return 0;  
}