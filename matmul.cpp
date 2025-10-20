// g++ matmul.cpp -o matmul -std=c++17 -O3 -Wall && ./matmul

#include <sys/time.h>
#include <iostream>
#include <cstring>
#include <cassert>


double get_time() {
  struct timeval tv;
  gettimeofday(&tv, nullptr);
  return 1e6 * tv.tv_sec + tv.tv_usec;
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
      BT[j][i] = B[i][j];
    }
  }
  for (int i = 0; i < n; i++) {
    const int *a_row = A[i];
    for (int j = 0; j < n; j++) {
      const int *b_row = BT[j];
      int sum = 0;
      for (int k = 0; k < n; k++) {
        sum += a_row[k] * b_row[k];    
      }
      C[i][j] = sum;
    }
  }
}

void matmul_tiling() {
  memset(C, 0, sizeof(C));

  constexpr int block_i = 64;
  constexpr int block_j = 64;
  constexpr int block_k = 32;
  constexpr int MR = 4;
  constexpr int NR = 4;

  alignas(64) static int B_tile[block_j * block_k];

  for (int ii = 0; ii < n; ii += block_i) {
    int i_end = ii + block_i;
    if (i_end > n) i_end = n;
    for (int jj = 0; jj < n; jj += block_j) {
      int j_end = jj + block_j;
      if (j_end > n) j_end = n;
      for (int kk = 0; kk < n; kk += block_k) {
        int k_end = kk + block_k;
        if (k_end > n) k_end = n;
        int k_len = k_end - kk;
        int j_len = j_end - jj;

        // Pack B block (transposed) so columns become contiguous.
        for (int j = 0; j < j_len; ++j) {
          int *dst = &B_tile[j * block_k];
          for (int k = 0; k < k_len; ++k) {
            dst[k] = B[kk + k][jj + j];
          }
        }

        int i = ii;
        for (; i + MR <= i_end; i += MR) {
          int *c0 = C[i];
          int *c1 = C[i + 1];
          int *c2 = C[i + 2];
          int *c3 = C[i + 3];
          const int *a0_panel = &A[i][kk];
          const int *a1_panel = &A[i + 1][kk];
          const int *a2_panel = &A[i + 2][kk];
          const int *a3_panel = &A[i + 3][kk];

          int j = 0;
          for (; j + NR <= j_len; j += NR) {
            int col = jj + j;

            int c00 = c0[col];
            int c01 = c0[col + 1];
            int c02 = c0[col + 2];
            int c03 = c0[col + 3];
            int c10 = c1[col];
            int c11 = c1[col + 1];
            int c12 = c1[col + 2];
            int c13 = c1[col + 3];
            int c20 = c2[col];
            int c21 = c2[col + 1];
            int c22 = c2[col + 2];
            int c23 = c2[col + 3];
            int c30 = c3[col];
            int c31 = c3[col + 1];
            int c32 = c3[col + 2];
            int c33 = c3[col + 3];

            for (int k = 0; k < k_len; ++k) {
              int b0 = B_tile[(j + 0) * block_k + k];
              int b1 = B_tile[(j + 1) * block_k + k];
              int b2 = B_tile[(j + 2) * block_k + k];
              int b3 = B_tile[(j + 3) * block_k + k];
              int a0k = a0_panel[k];
              int a1k = a1_panel[k];
              int a2k = a2_panel[k];
              int a3k = a3_panel[k];

              c00 += a0k * b0;
              c01 += a0k * b1;
              c02 += a0k * b2;
              c03 += a0k * b3;

              c10 += a1k * b0;
              c11 += a1k * b1;
              c12 += a1k * b2;
              c13 += a1k * b3;

              c20 += a2k * b0;
              c21 += a2k * b1;
              c22 += a2k * b2;
              c23 += a2k * b3;

              c30 += a3k * b0;
              c31 += a3k * b1;
              c32 += a3k * b2;
              c33 += a3k * b3;
            }

            c0[col] = c00;
            c0[col + 1] = c01;
            c0[col + 2] = c02;
            c0[col + 3] = c03;
            c1[col] = c10;
            c1[col + 1] = c11;
            c1[col + 2] = c12;
            c1[col + 3] = c13;
            c2[col] = c20;
            c2[col + 1] = c21;
            c2[col + 2] = c22;
            c2[col + 3] = c23;
            c3[col] = c30;
            c3[col + 1] = c31;
            c3[col + 2] = c32;
            c3[col + 3] = c33;
          }

          for (; j < j_len; ++j) {
            int col = jj + j;
            int sum0 = c0[col];
            int sum1 = c1[col];
            int sum2 = c2[col];
            int sum3 = c3[col];
            for (int k = 0; k < k_len; ++k) {
              int b = B_tile[j * block_k + k];
              int a0k = a0_panel[k];
              int a1k = a1_panel[k];
              int a2k = a2_panel[k];
              int a3k = a3_panel[k];
              sum0 += a0k * b;
              sum1 += a1k * b;
              sum2 += a2k * b;
              sum3 += a3k * b;
            }
            c0[col] = sum0;
            c1[col] = sum1;
            c2[col] = sum2;
            c3[col] = sum3;
          }
        }

        for (; i < i_end; ++i) {
          int *c_row = C[i];
          const int *a_panel = &A[i][kk];
          for (int j = 0; j < j_len; ++j) {
            int col = jj + j;
            int sum = c_row[col];
            const int *b_col = &B_tile[j * block_k];
            for (int k = 0; k < k_len; ++k) {
              sum += a_panel[k] * b_col[k];
            }
            c_row[col] = sum;
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


void run_and_report(const char *label, void (*kernel)()) {
  const int repeats = 32;
  double total = 0.0;
  double best = 1e100;

  for (int r = 0; r < repeats; r++) {
    printf("%s iter %d\n", label, r);
    fflush(stdout);
    double t0 = get_time();
    kernel();
    double elapsed = get_time() - t0;
    total += elapsed;
    if (elapsed < best) best = elapsed;
    test();
  }

  printf("%-16s avg=%9.0f us best=%9.0f us\n", label, total / repeats, best);
}

int main() {
  init();

  run_and_report("matmul_ikj", matmul_ikj);
  run_and_report("matmul_AT", matmul_AT);
  run_and_report("matmul_BT", matmul_BT);
  run_and_report("matmul_tiling", matmul_tiling);
  run_and_report("matmul_unrolled", matmul_unrolled);

  return 0;
}