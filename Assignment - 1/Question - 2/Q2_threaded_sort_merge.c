
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <sys/time.h>
#include <time.h>
#include <limits.h>
#include <string.h>
#include <unistd.h>

#define ARRAY_SIZE 10000

typedef struct {
    int* array;
    int left;
    int right;
} thread_args;

typedef struct {
    int* src;
    int* dst;
    int left_start;
    int left_end;
    int right_start;
    int right_end;
} merge_args;

void swap(int* a, int* b) {
    int temp = *a;
    *a = *b;
    *b = temp;
}

int partition(int arr[], int low, int high) {
    int pivot = arr[high];
    int i = low - 1;
    for (int j = low; j <= high - 1; j++) {
        if (arr[j] < pivot) {
            i++;
            swap(&arr[i], &arr[j]);
        }
    }
    swap(&arr[i + 1], &arr[high]);
    return i + 1;
}

void quicksort(int arr[], int low, int high) {
    if (low < high) {
        int pi = partition(arr, low, high);
        quicksort(arr, low, pi - 1);
        quicksort(arr, pi + 1, high);
    }
}

void* thread_quicksort(void* args) {
    thread_args* t_args = (thread_args*)args;
    quicksort(t_args->array, t_args->left, t_args->right);
    return NULL;
}

void parallel_merge(int array[], int left, int mid, int right, int* temp) {
    int i = left, j = mid + 1, k = left;
    while (i <= mid && j <= right) {
        if (array[i] <= array[j]) {
            temp[k++] = array[i++];
        } else {
            temp[k++] = array[j++];
        }
    }
    while (i <= mid) temp[k++] = array[i++];
    while (j <= right) temp[k++] = array[j++];
    memcpy(array + left, temp + left, (right - left + 1) * sizeof(int));
}

void* thread_merge(void* args) {
    merge_args* m_args = (merge_args*)args;
    int* temp = malloc(ARRAY_SIZE * sizeof(int));
    if (!temp) {
        perror("Failed to allocate memory");
        exit(EXIT_FAILURE);
    }
    int i = m_args->left_start, j = m_args->right_start, k = m_args->left_start;
    while (i <= m_args->left_end && j <= m_args->right_end) {
        if (m_args->src[i] <= m_args->src[j]) {
            temp[k++] = m_args->src[i++];
        } else {
            temp[k++] = m_args->src[j++];
        }
    }
    while (i <= m_args->left_end) temp[k++] = m_args->src[i++];
    while (j <= m_args->right_end) temp[k++] = m_args->src[j++];
    memcpy(m_args->dst + m_args->left_start, temp + m_args->left_start, 
           (m_args->right_end - m_args->left_start + 1) * sizeof(int));
    free(temp);
    return NULL;
}

void merge_sorted_arrays(int array[], int num_chunks, thread_args t_args[]) {
    int* temp = malloc(ARRAY_SIZE * sizeof(int));
    if (!temp) {
        perror("Failed to allocate memory");
        exit(EXIT_FAILURE);
    }

    int chunk_size = ARRAY_SIZE / num_chunks;
    for (int width = chunk_size; width < ARRAY_SIZE; width *= 2) {
        pthread_t threads[num_chunks];
        merge_args m_args[num_chunks];
        int thread_count = 0;

        for (int i = 0; i < ARRAY_SIZE; i += 2 * width) {
            int left = i;
            int mid = i + width - 1;
            int right = (i + 2 * width - 1 < ARRAY_SIZE) ? i + 2 * width - 1 : ARRAY_SIZE - 1;
            if (mid >= ARRAY_SIZE) break;

            m_args[thread_count].src = array;
            m_args[thread_count].dst = temp;
            m_args[thread_count].left_start = left;
            m_args[thread_count].left_end = mid;
            m_args[thread_count].right_start = mid + 1;
            m_args[thread_count].right_end = right;

            pthread_create(&threads[thread_count], NULL, thread_merge, &m_args[thread_count]);
            thread_count++;
        }

        for (int j = 0; j < thread_count; j++) {
            pthread_join(threads[j], NULL);
        }

        memcpy(array, temp, ARRAY_SIZE * sizeof(int));
    }

    free(temp);
}

int main(int argc, char* argv[]) {
    if (argc != 2) {
        printf("Usage: %s <max_threads>\n", argv[0]);
        return 1;
    }

    int max_threads = atoi(argv[1]);
    int available_cores = sysconf(_SC_NPROCESSORS_ONLN);
    int num_threads = (max_threads < available_cores) ? max_threads : available_cores;
    printf("Using %d threads (available cores: %d)\n", num_threads, available_cores);

    int array[ARRAY_SIZE];
    srand(time(NULL));

    for (int i = 0; i < ARRAY_SIZE; i++) {
        array[i] = rand() % 10000 + 1;
    }

    printf("Unsorted sample: ");
    for (int i = 0; i < 5; i++) printf("%d ", array[i]);
    printf("... ");
    for (int i = ARRAY_SIZE - 5; i < ARRAY_SIZE; i++) printf("%d ", array[i]);
    printf("\n");

    pthread_t threads[num_threads];
    thread_args t_args[num_threads];
    int chunk_size = ARRAY_SIZE / num_threads;

    for (int i = 0; i < num_threads; i++) {
        t_args[i].array = array;
        t_args[i].left = i * chunk_size;
        t_args[i].right = (i == num_threads - 1) ? ARRAY_SIZE - 1 : (i + 1) * chunk_size - 1;
    }

    struct timeval start, end;
    gettimeofday(&start, NULL);

    for (int i = 0; i < num_threads; i++) {
        pthread_create(&threads[i], NULL, thread_quicksort, &t_args[i]);
    }

    for (int i = 0; i < num_threads; i++) {
        pthread_join(threads[i], NULL);
    }

    merge_sorted_arrays(array, num_threads, t_args);

    gettimeofday(&end, NULL);
    double elapsed = (end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec) / 1e6;
    printf("Threads: %d | Time: %.4f seconds\n", num_threads, elapsed);

    int sorted = 1;
    for (int i = 0; i < ARRAY_SIZE - 1; i++) {
        if (array[i] > array[i + 1]) {
            sorted = 0;
            break;
        }
    }

    printf(sorted ? "Sorted successfully.\n" : "Sorting failed.\n");

    printf("Sorted sample: ");
    for (int i = 0; i < 5; i++) printf("%d ", array[i]);
    printf("... ");
    for (int i = ARRAY_SIZE - 5; i < ARRAY_SIZE; i++) printf("%d ", array[i]);
    printf("\n");

    return 0;
}
