
#include <iostream>
#include <pthread.h>
#include <unistd.h>
#include <vector>
#include <cstdlib>
#include <ctime>

enum PhilosopherState { THINKING, HUNGRY, EATING };

static const char* stateNames[] = {"THINKING", "HUNGRY", "EATING"};

static int g_numPhilosophers = 0;
static std::vector<pthread_mutex_t> g_forks;
static std::vector<PhilosopherState> g_states;
static pthread_mutex_t g_stateMutex = PTHREAD_MUTEX_INITIALIZER;

inline int leftFork(int philosopherId) {
    return philosopherId;
}

inline int rightFork(int philosopherId) {
    return (philosopherId + 1) % g_numPhilosophers;
}

void randomDelay() {
    usleep(100000 + (rand() % 400000));
}

void printStates() {
    pthread_mutex_lock(&g_stateMutex);
    std::cout << "[Current States] ";
    for (int i = 0; i < g_numPhilosophers; ++i) {
        std::cout << "P" << i << ":" << stateNames[g_states[i]] << " ";
    }
    std::cout << std::endl;
    pthread_mutex_unlock(&g_stateMutex);
}

void pickUpForks(int id) {
    pthread_mutex_lock(&g_stateMutex);
    g_states[id] = HUNGRY;
    pthread_mutex_unlock(&g_stateMutex);
    if (id % 2 == 0) {
        pthread_mutex_lock(&g_forks[leftFork(id)]);
        pthread_mutex_lock(&g_forks[rightFork(id)]);
    } else {
        pthread_mutex_lock(&g_forks[rightFork(id)]);
        pthread_mutex_lock(&g_forks[leftFork(id)]);
    }
    pthread_mutex_lock(&g_stateMutex);
    g_states[id] = EATING;
    pthread_mutex_unlock(&g_stateMutex);
}

void putDownForks(int id) {
    if (id % 2 == 0) {
        pthread_mutex_unlock(&g_forks[leftFork(id)]);
        pthread_mutex_unlock(&g_forks[rightFork(id)]);
    } else {
        pthread_mutex_unlock(&g_forks[rightFork(id)]);
        pthread_mutex_unlock(&g_forks[leftFork(id)]);
    }
    pthread_mutex_lock(&g_stateMutex);
    g_states[id] = THINKING;
    pthread_mutex_unlock(&g_stateMutex);
}

void* philosopher(void* arg) {
    int id = *(int*)arg;
    while (true) {
        randomDelay();
        pickUpForks(id);
        randomDelay();
        putDownForks(id);
    }
    return nullptr;
}

void* monitor(void* arg) {
    int iterationCount = 0;
    while (iterationCount < 12) {
        printStates();
        usleep(500000);
        iterationCount++;
    }
    return nullptr;
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <number_of_philosophers>\n";
        return 1;
    }
    g_numPhilosophers = std::atoi(argv[1]);
    srand(static_cast<unsigned int>(time(NULL)));
    g_forks.resize(g_numPhilosophers);
    g_states.resize(g_numPhilosophers, THINKING);
    for (int i = 0; i < g_numPhilosophers; ++i) {
        pthread_mutex_init(&g_forks[i], nullptr);
    }
    std::vector<pthread_t> threads(g_numPhilosophers);
    std::vector<int> ids(g_numPhilosophers);
    for (int i = 0; i < g_numPhilosophers; ++i) {
        ids[i] = i;
        pthread_create(&threads[i], nullptr, philosopher, &ids[i]);
    }
    pthread_t monitorThread;
    pthread_create(&monitorThread, nullptr, monitor, nullptr);
    pthread_join(monitorThread, nullptr);
    for (int i = 0; i < g_numPhilosophers; ++i) {
        pthread_cancel(threads[i]);
        pthread_join(threads[i], NULL);
    }
    for (int i = 0; i < g_numPhilosophers; ++i) {
        pthread_mutex_destroy(&g_forks[i]);
    }
    pthread_mutex_destroy(&g_stateMutex);
    return 0;
}
