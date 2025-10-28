
# README: Dining Philosophers Problem (C++ Implementation using Pthreads)

## Overview
This project implements the **Dining Philosophers Problem** using **C++ and POSIX threads (pthreads)**. The program simulates philosophers who alternate between **thinking** and **eating**, with shared forks placed between them. The solution prevents **deadlocks** and **race conditions** by using **mutexes for synchronization**.

Each philosopher requires **two forks** to eat, and they must acquire them **one at a time**. The program prints the state of each philosopher and the availability of each fork over **12 iterations** to demonstrate the execution process.

---

## Compilation and Execution
### 1. Compiling the Code
To compile the program, use **g++** with the pthread library:
```bash
g++ -o dining_philosophers dining_philosophers.cpp -lpthread
```

### 2. Running the Program
Execute the compiled binary with the number of philosophers as an argument:
```bash
./dining_philosophers <number_of_philosophers>
```
For example, to run the program with **5 philosophers**:
```bash
./dining_philosophers 5
```

---

## Program Explanation
### 1. Problem Setup
- The program simulates **n philosophers** sitting around a table.
- There are **n forks**, one between each pair of adjacent philosophers.
- Each philosopher alternates between **thinking** and **eating**.
- A philosopher must acquire **both the left and right forks** before eating.

### 2. Synchronization Mechanism
- Each fork is represented by a **pthread_mutex_t**.
- Philosophers must acquire the mutex for both their left and right forks to simulate taking the forks.
- Fork states are managed to ensure no two philosophers can eat simultaneously with the same fork.

### 3. Monitoring and Output
- The monitor thread prints the current state of each philosopher and the status of the forks every 500 milliseconds.
- This helps in visualizing the process flow and understanding how deadlocks are avoided and resources are managed.

---

## Expected Output
When the program runs, it prints updates on the states of philosophers and forks every 500ms. Below is an example of what the output might look like:

```
[Philosophers] P0:EATING P1:THINKING P2:HUNGRY P3:THINKING P4:EATING 
[Forks]        F0:IN USE F1:AVAILABLE F2:IN USE F3:AVAILABLE F4:IN USE 

[Philosophers] P0:THINKING P1:EATING P2:THINKING P3:HUNGRY P4:THINKING 
[Forks]        F0:AVAILABLE F1:IN USE F2:AVAILABLE F3:IN USE F4:AVAILABLE 

[Philosophers] P0:HUNGRY P1:THINKING P2:EATING P3:THINKING P4:HUNGRY 
[Forks]        F0:IN USE F1:AVAILABLE F2:IN USE F3:AVAILABLE F4:IN USE 
```

The output shows:
- **Philosopher states**: THINKING, HUNGRY, or EATING.
- **Fork states**: AVAILABLE or IN USE.

Each iteration updates the state, demonstrating how philosophers pick up and release forks to avoid deadlock.

---

## Requirements
- **C++ Compiler**: g++ with support for C++11 or later.
- **POSIX Threads**: The pthreads library must be available on your system, as it is used for threading and synchronization.

## Source
https://github.com/DZwe/Dining-Philosophers-Problem/blob/master/Dining.c
https://github.com/atellaluca/dining-philosophers/blob/master/dining-philosophers.cpp
https://github.com/brycesulin/DiningPhilosopher/blob/master/Dining-Philosophers.c

## Notes
- Ensure that the system supports POSIX threads for this program to compile and run correctly.
- Adjust the number of iterations or the delay in the monitor thread as needed for different demonstrations or testing scenarios.

