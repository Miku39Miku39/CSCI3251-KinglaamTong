// g++ matmul.cpp -o matmul -std=c++17 -O3 -Wall && ./matmul

//#include <sys/time.h>
#include <time.h>
#include <chrono>
#include <iostream>
#include <cstring>
#include <cassert>
#include <cstdio>


// double get_time() {
//   struct timeval tv;
//   gettimeofday(&tv, nullptr);
//   return 1e6 * tv.tv_sec + tv.tv_usec;
// }

double get_time_new() {
  return std::chrono::duration<double, std::micro>(
             std::chrono::high_resolution_clock::now().time_since_epoch())
      .count();
}

constexpr int n = 512;
int A[n][n];
int B[n][n];
int BT[n][n];
int AT[n][n];
int C[n][n];
int C_groundtruth[n][n];

void init() {
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      A[i][j] = rand(); 
      B[i][j] = rand(); 
    } 
  }
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      for (int k = 0; k < n; k++) {
        C_groundtruth[i][j] += A[i][k] * B[k][j];
      }
    }
  }
}

void test() {
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      assert(C[i][j] == C_groundtruth[i][j]);
    }
  }
}

void matmul() {
  memset(C, 0, sizeof(C));
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      for (int k = 0; k < n; k++) {
        C[i][j] += A[i][k] * B[k][j];    
      }   
    }
  }
}

void matmul_ikj() {
  memset(C, 0, sizeof(C));
  for (int i = 0; i < n; i++) {
    for (int k = 0; k < n; k++) {
      for (int j = 0; j < n; j++) {
        C[i][j] += A[i][k] * B[k][j];    
      }   
    }
  }
}

void matmul_AT() {
  memset(C, 0, sizeof(C));
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      AT[i][j] = A[j][i];
    }
  }
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      for (int k = 0; k < n; k++) {
        C[i][j] += AT[k][i] * B[k][j];    
      }   
    }
  }
}

void matmul_BT() {
  memset(C, 0, sizeof(C));
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      BT[i][j] = B[j][i];
    }
  }
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      for (int k = 0; k < n; k++) {
        C[i][j] += A[i][k] * BT[j][k];    
      }   
    }
  }
}

void matmul_tiling() {
  // Cache-blocked (tiled) matrix multiplication. Loop order (ii, jj, kk, i, k, j)
  // maximizes spatial locality for B and C, and reuses A[i][k] across the inner j loop.
  memset(C, 0, sizeof(C));

  // A conservative tile size that works well on Raspberry Pi 4B (Cortex-A72) for int32.
  // 32 divides 512 exactly but the bounds logic below handles non-multiples too.
  const int block = 32;

  for (int ii = 0; ii < n; ii += block) {
    int i_end = ii + block;
    if (i_end > n) i_end = n;
    for (int jj = 0; jj < n; jj += block) {
      int j_end = jj + block;
      if (j_end > n) j_end = n;
      for (int kk = 0; kk < n; kk += block) {
        int k_end = kk + block;
        if (k_end > n) k_end = n;
        for (int i = ii; i < i_end; i++) {
          for (int k = kk; k < k_end; k++) {
            int aik = A[i][k];
            for (int j = jj; j < j_end; j++) {
              C[i][j] += aik * B[k][j];
            }
          }
        }
      }
    }
  }
}

void matmul_unrolled() {
  // Loop unrolling (factor 4) of the inner k-loop to reduce loop overhead
  // and expose more ILP to the compiler on Raspberry Pi 4B.
  memset(C, 0, sizeof(C));
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      int sum = 0;
      int k = 0;
      for (; k + 3 < n; k += 4) {
        sum += A[i][k] * B[k][j];
        sum += A[i][k + 1] * B[k + 1][j];
        sum += A[i][k + 2] * B[k + 2][j];
        sum += A[i][k + 3] * B[k + 3][j];
      }
      for (; k < n; k++) {
        sum += A[i][k] * B[k][j];
      }
      C[i][j] = sum;
    }
  }
}

// int main() {
//   init();
//   float avg_time = 0.0f;
//   for (int K = 0; K < 32; K++) {
//     auto t = get_time();
//     // matmul_ikj();
//     // matmul(); 
//     // matmul_AT();
//     // matmul_BT();
//     // matmul_tiling();
//     // matmul_unrolled();
//     printf("%f\n", get_time() - t);
//     avg_time += get_time() - t;
//     test();
//   }
//   printf("Avg Time for Calculation: %f us\n", avg_time / 32);
//   return 0;
// }

int main() {
  init();
  float avg_time = 0.0f;
  for (int K = 0; K < 32; K++) {
    auto t = get_time_new();
    // matmul_ikj();
    matmul(); 
    // matmul_AT();
    // matmul_BT();
    // matmul_tiling();
    // matmul_unrolled();
    printf("%f\n", get_time_new() - t);
    avg_time += get_time_new() - t;
    test();
  }
  printf("Avg Time for Calculation: %f us\n", avg_time / 32);
  return 0;
}