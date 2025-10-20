template<int H, int Ww, int K, int Ci, int Co, int P, int Q>
void build_lowered_2(const float (&X)[H][Ww][Ci],
                   const float (&W)[K][K][Ci][Co],
                   float (&Xprime)[K*K*Ci][P*Q],
                   float (&Wprime)[Co][K*K*Ci]) {
  int col = 0;
  for (int i = 0; i < P; i++) {
    for (int j = 0; j < Q; j++) {
      int count = 0;
      for (int c = 0; c < Ci; c++) {
        for (int h = 0; h < K; h++) {
          for (int w = 0; w < K; w++) {
            int height = i + h;
            int width = j + w;
            Xprime[count++][col] = X[height][width][c];
          }
        }
      }
      ++col;
    }
  }

  for (int co = 0; co < Co; co++) {
    int count = 0;
    for (int c = 0; c < Ci; c++) {
      for (int h = 0; h < K; h++) {
        for (int w = 0; w < K; w++) {
          Wprime[co][count++] = W[h][w][c][co];
        }
      }
    }
  }
}


template<int H, int Ww, int K, int Ci, int Co, int P, int Q>
void build_lowered_1(const float (&X)[H][Ww][Ci],
                    const float (&W)[K][K][Ci][Co],
                    float (&Xprime)[P*Q][K*K*Ci],
                    float (&Wprime)[K*K*Ci][Co]) {
  int row = 0;
  for (int i = 0; i < P; i++) {
    for (int j = 0; j < Q; j++) {
      int count = 0;
      for (int c = 0; c < Ci; c++) {
        for (int h = 0; h < K; h++) {
          for (int w = 0; w < K; w++) {
            int height = i + h;
            int width = j + w;
            Xprime[row][count++] = X[height][width][c];
          }
        }
      }
      row++;
    }
  }

  for (int co = 0; co < Co; co++) {
    int count = 0;
    for (int c = 0; c < Ci; c++) {
      for (int h = 0; h < K; h++) {
        for (int w = 0; w < K; w++) {
          Wprime[count++][co] = W[h][w][c][co];
        }
      }
    }
  }
}

template<int H, int Ww, int K, int C, int P, int Q, int R, int S>
void direct_conv2d(const float (&X)[H][Ww][C],
                   const float (&W)[R][S][C][K],
                   float (&Y)[P][Q][K]) {
  for (int i = 0; i < P; i++) {
    for (int j = 0; j < Q; j++) {
      for (int k = 0; k < K; k++) {
        float sum = 0.0f;
        for (int c = 0; c < C; c++) {
          for (int r = 0; r < R; r++) {
            for (int s = 0; s < S; s++) {
              int height = i + r;
              int width = j + s;
              sum += X[height][width][c] * W[r][s][c][k];
            }
          }
        }
        Y[i][j][k] = sum;
      }
    }
  }
}    
