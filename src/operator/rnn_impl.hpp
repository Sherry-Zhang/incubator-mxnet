/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * Copyright (c) 2015 by Contributors
 * \file    rnn_impl.hpp
 * \brief
 * \author  Shu Zhang(shu.zhang@intel.com)
*/
#ifndef MXNET_OPERATOR_RNN_IMPL_HPP_
#define MXNET_OPERATOR_RNN_IMPL_HPP_

#include <dmlc/logging.h>
#include <dmlc/parameter.h>
#include <mxnet/operator.h>
#include <algorithm>
#include <map>
#include <vector>
#include <string>
#include <utility>
#include "./math.h"
#include "./math_functions-inl.h"
#include "./operator_common.h"
#include "./mshadow_op.h"
#include "./linalg.h"

template<typename DType>
inline DType sigmoid(DType x) {
    return 1.0f / (1.0f + exp(-x));
}

template<typename DType>
void LstmForwardTrainingSingleLayer(DType* ws,
                                    DType* rs,
                                    const int D,
                                    const int T,
                                    const int N,
                                    const int I,
                                    const int H,
                                    const Tensor<cpu, 2, DType> &x,
                                    const Tensor<cpu, 2, DType> &hx,
                                    const Tensor<cpu, 2, DType> &cx,
                                    DType* w_ptr) {
  using namespace mshadow;
  const Tensor<cpu, 2, DType> wx(w_ptr, Shape2(H * 4, I));
  const Tensor<cpu, 2, DType> wh(w_ptr + I * H * 4, Shape2(H * 4, H));
  const Tensor<cpu, 2, DType> bx(wh.dptr_ + H * H * 4, Shape2(4, H));
  const Tensor<cpu, 2, DType> bh(bx.dptr_ + H * 4, Shape2(4, H));
  const Tensor<cpu, 2, DType> yx_flat(ws, Shape2(T * N, 4 * H));
  const Tensor<cpu, 2, DType> yh_flat(ws + T * N * H * 4, Shape2(N, 4 * H));
  const Tensor<cpu, 4, DType> yx(yx_flat.dptr_, Shape4(T, N, 4, H));
  const Tensor<cpu, 3, DType> yh(yh_flat.dptr_, Shape3(N, 4, H));
  Tensor<cpu, 3, DType> h(rs, Shape3(T, N, H));
  Tensor<cpu, 3, DType> c(rs + T * N * H, Shape3(T, N, H));
  Tensor<cpu, 4, DType> ifgo(rs + T * N * H * 2, Shape4(T, N, H, 4));

  const DType alpha = 1.0;
  const DType beta = 0.0;
  linalg_gemm(x, wx, yx_flat, alpha, beta, false, true);

  for (int i = 0; i < T; ++i) {
    linalg_gemm((i == 0) ? hx : h[i-1], wh, yh_flat, alpha, beta, false, true);
    #pragma omp parallel for collapse(2)
    for (int j = 0; j < N; ++j) {
      for (int k = 0; k < H; ++k) {
        DType it = sigmoid<DType>(yx[i][j][0][k] + yh[j][0][k] + bx[0][k] + bh[0][k]);
        DType ft = sigmoid<DType>(yx[i][j][1][k] + yh[j][1][k] + bx[1][k] + bh[1][k]);
        DType gt =           tanh(yx[i][j][2][k] + yh[j][2][k] + bx[2][k] + bh[2][k]);
        DType ot = sigmoid<DType>(yx[i][j][3][k] + yh[j][3][k] + bx[3][k] + bh[3][k]);
        DType ct = ((i == 0) ? cx[j][k] : c[i-1][j][k]) * ft + it * gt;
        h[i][j][k] = ot * tanh(ct);
        c[i][j][k] = ct;
        // reserve
        ifgo[i][j][k][0] = it;
        ifgo[i][j][k][1] = ft;
        ifgo[i][j][k][2] = gt;
        ifgo[i][j][k][3] = ot;
      }
    }
  }
}

template <typename DType>
void LstmForwardTraining(DType* ws,
                         DType* rs,
                         bool state_outputs,
                         const int L,
                         const int D,
                         const int T,
                         const int N,
                         const int I,
                         const int H,
                         DType* x_ptr,
                         DType* hx_ptr,
                         DType* cx_ptr,
                         DType* w_ptr,
                         DType* y_ptr,
                         DType* hy_ptr,
                         DType* cy_ptr) {
  Tensor<cpu, 2, DType> x(x_ptr, Shape2(T * N, I));
  Tensor<cpu, 3, DType> hx(hx_ptr, Shape3(L, N, H));
  Tensor<cpu, 3, DType> cx(cx_ptr, Shape3(L, N, H));
  LstmForwardTrainingSingleLayer<DType>(ws, rs, D, T, N, I, H, x, hx[0], cx[0], w_ptr);
  if (state_outputs) {
    memcpy(hy_ptr, rs + (T - 1) * N * H, N * H * sizeof(DType));
    memcpy(cy_ptr, rs + (T + T - 1) * N * H, N * H * sizeof(DType));
  }
  memcpy(y_ptr, rs, T * N * H * sizeof(DType));
}

template<typename DType>
void LstmForwardInferenceSingleLayer(DType* ws,
                                     bool state_outputs,
                                     const int D,
                                     const int T,
                                     const int N,
                                     const int I,
                                     const int H,
                                     const Tensor<cpu, 2, DType> &x,
                                     const Tensor<cpu, 2, DType> &hx,
                                     const Tensor<cpu, 2, DType> &cx,
                                     DType* w_ptr,
                                     DType* y_ptr,
                                     DType* hy_ptr,
                                     DType* cy_ptr) {
  using namespace mshadow;
  const Tensor<cpu, 2, DType> wx(w_ptr, Shape2(H * 4, I));
  const Tensor<cpu, 2, DType> wh(w_ptr + I * H * 4, Shape2(H * 4, H));
  const Tensor<cpu, 2, DType> bx(wh.dptr_ + H * H * 4, Shape2(4, H));
  const Tensor<cpu, 2, DType> bh(bx.dptr_ + H * 4, Shape2(4, H));
  Tensor<cpu, 2, DType> yx_flat(ws, Shape2(T * N, H * 4));
  Tensor<cpu, 2, DType> yh_flat(ws + T * N * H * 4, Shape2(N, H * 4));
  const Tensor<cpu, 4, DType> yx(yx_flat.dptr_, Shape4(T, N, 4, H));
  const Tensor<cpu, 3, DType> yh(yh_flat.dptr_, Shape3(N, 4, H));
  Tensor<cpu, 2, DType> c(yh_flat.dptr_ + N * H * 4, Shape2(N, H));
  Tensor<cpu, 3, DType> h(y_ptr, Shape3(T, N, H));
  const DType alpha = 1.0;
  const DType beta = 0.0;
  linalg_gemm(x, wx, yx_flat, alpha, beta, false, true);

  for (int i = 0; i < T; ++i) {
    linalg_gemm((i == 0) ? hx : h[i-1], wh, yh_flat, alpha, beta, false, true);
    #pragma omp parallel for collapse(2)
    for (int j = 0; j < N; ++j) {
      for (int k = 0; k < H; ++k) {
        DType it = sigmoid<DType>(yx[i][j][0][k] + yh[j][0][k] + bx[0][k] + bh[0][k]);
        DType ft = sigmoid<DType>(yx[i][j][1][k] + yh[j][1][k] + bx[1][k] + bh[1][k]);
        DType gt =           tanh(yx[i][j][2][k] + yh[j][2][k] + bx[2][k] + bh[2][k]);
        DType ot = sigmoid<DType>(yx[i][j][3][k] + yh[j][3][k] + bx[3][k] + bh[3][k]);
        DType ct = ((i == 0) ? cx[j][k] : c[j][k]) * ft + it * gt;
        h[i][j][k] = ot * tanh(ct);
        c[j][k] = ct;
      }
    }
  }
  if (state_outputs) {
    memcpy(hy_ptr, y_ptr + (T - 1) * N * H, N * H * sizeof(DType));
    memcpy(cy_ptr, c.dptr_, N * H * sizeof(DType));
  }
}

template <typename DType>
void LstmForwardInference(DType* ws,
                          bool state_outputs,
                          const int L,
                          const int D,
                          const int T,
                          const int N,
                          const int I,
                          const int H,
                          DType* x_ptr,
                          DType* hx_ptr,
                          DType* cx_ptr,
                          DType* w_ptr,
                          DType* y_ptr,
                          DType* hy_ptr,
                          DType* cy_ptr) {
  Tensor<cpu, 2, DType> x(x_ptr, Shape2(T * N, I));
  Tensor<cpu, 3, DType> hx(hx_ptr, Shape3(L, N, H));
  Tensor<cpu, 3, DType> cx(cx_ptr, Shape3(L, N, H));
  LstmForwardInferenceSingleLayer<DType>(ws, state_outputs, D, T, N, I, H,
                                         x, hx[0], cx[0], w_ptr, y_ptr, hy_ptr, cy_ptr);
}

template <typename DType>
void LstmBackwardSingleLayer(DType* ws,
                             DType* rs,
                             const int D,
                             const int T,
                             const int N,
                             const int I,
                             const int H,
                             const Tensor<cpu, 2, DType> &x,
                             const Tensor<cpu, 2, DType> &hx,
                             const Tensor<cpu, 2, DType> &cx,
                             const Tensor<cpu, 3, DType> &y,
                             const Tensor<cpu, 3, DType> &dy,
                             const Tensor<cpu, 2, DType> &dx,
                             const Tensor<cpu, 2, DType> &dhx,
                             const Tensor<cpu, 2, DType> &dcx,
                             DType* dhy_ptr,
                             DType* dcy_ptr,
                             DType* w_ptr,
                             DType* dw_ptr) {
  using namespace mshadow;
  const Tensor<cpu, 2, DType> wx(w_ptr, Shape2(H * 4, I));
  const Tensor<cpu, 2, DType> wh(w_ptr + I * H * 4, Shape2(H * 4, H));
  Tensor<cpu, 2, DType> dwx(dw_ptr, Shape2(H * 4, I));
  Tensor<cpu, 2, DType> dwh(dw_ptr + I * H * 4, Shape2(H * 4, H));
  Tensor<cpu, 1, DType> dbx(dwh.dptr_ + H * H * 4, Shape1(H * 4));
  Tensor<cpu, 1, DType> dbh(dbx.dptr_ + H * 4, Shape1(H * 4));
  const Tensor<cpu, 3, DType> h(rs, Shape3(T, N, H));
  const Tensor<cpu, 3, DType> c(rs + T * N * H, Shape3(T, N, H));
  const Tensor<cpu, 4, DType> ifgo(rs + T * N * H * 2, Shape4(T, N, H, 4));

  memset(dwh.dptr_, 0, H * H * 4 * sizeof(DType));
  memset(dbx.dptr_, 0, H * 4 * sizeof(DType));
  memset(dbh.dptr_, 0, H * 4 * sizeof(DType));
  Tensor<cpu, 4, DType> difgo(ws, Shape4(T, N, 4, H));
  Tensor<cpu, 2, DType> dh(ws + T * N * H * 4, Shape2(N, H));
  Tensor<cpu, 2, DType> dc(dh.dptr_ + N * H, Shape2(N, H));
  if (dhy_ptr != NULL) {
    memcpy(dh.dptr_, dhy_ptr, N * H * sizeof(DType));
  }
  if (dcy_ptr != NULL) {
    memcpy(dc.dptr_, dcy_ptr, N * H * sizeof(DType));
  }
  const DType alpha = 1.0;
  const DType beta0 = 0.0;
  const DType beta1 = 1.0;
  for (int i = T - 1; i >= 0; --i) {
    const Tensor<cpu, 2, DType>& dhnext = i ? dh : dhx;
    const Tensor<cpu, 2, DType>& dcnext = i ? dc : dcx;
    const Tensor<cpu, 2, DType>& hnext = i ? h[i-1] : hx;
    const Tensor<cpu, 2, DType>& cnext = i ? c[i-1] : cx;
    #pragma omp parallel for collapse(2)
    for (int j = 0; j < N; ++j) {
      for (int k = 0; k < H; ++k) {
         DType tc = tanh(c[i][j][k]);
         DType it = ifgo[i][j][k][0];
         DType ft = ifgo[i][j][k][1];
         DType gt = ifgo[i][j][k][2];
         DType ot = ifgo[i][j][k][3];

         dh[j][k] += dy[i][j][k];
         dc[j][k] += dh[j][k] * ot * (1 - tc * tc);

         difgo[i][j][0][k] = dc[j][k] * gt * it * (1 - it);
         difgo[i][j][1][k] = dc[j][k] * cnext[j][k] * ft * (1 - ft);
         difgo[i][j][2][k] = dc[j][k] * it * (1 - gt * gt);
         difgo[i][j][3][k] = dh[j][k] * tc * ot * (1 - ot);
         dcnext[j][k] = dc[j][k] * ft;
      }
    }
    Tensor<cpu, 2, DType> dyh(difgo[i].dptr_, Shape2(N, H * 4));
    linalg_gemm(dyh, wh, dhnext, alpha, beta0, false, false);
    linalg_gemm(dyh, hnext, dwh, alpha, beta1, true, false);
  }
  Tensor<cpu, 2, DType> dyx(difgo.dptr_, Shape2(T * N, H * 4));
  linalg_gemm(dyx, wx, dx, alpha, beta0, false, false);
  linalg_gemm(dyx, x, dwx, alpha, beta0, true, false);
  for (int i = 0; i < T * N; ++i) {
    for (int j = 0; j < H * 4; ++j) {
      dbx[j] += dyx[i][j];
      dbh[j] = dbx[j];
    }
  }
}

template <typename DType>
void LstmBackward(DType* ws,
                  DType* rs,
                  const int L,
                  const int D,
                  const int T,
                  const int N,
                  const int I,
                  const int H,
                  DType* x_ptr,
                  DType* hx_ptr,
                  DType* cx_ptr,
                  DType* w_ptr,
                  DType* y_ptr,
                  DType* dy_ptr,
                  DType* dhy_ptr,
                  DType* dcy_ptr,
                  DType* dx_ptr,
                  DType* dhx_ptr,
                  DType* dcx_ptr,
                  DType* dw_ptr) {
  Tensor<cpu, 2, DType> x(x_ptr, Shape2(T * N, I));
  Tensor<cpu, 3, DType> hx(hx_ptr, Shape3(L, N, H));
  Tensor<cpu, 3, DType> cx(cx_ptr, Shape3(L, N, H));
  Tensor<cpu, 2, DType> dx(dx_ptr, Shape2(T * N, I));
  Tensor<cpu, 3, DType> dhx(dhx_ptr, Shape3(L, N, H));
  Tensor<cpu, 3, DType> dcx(dcx_ptr, Shape3(L, N, H));
  Tensor<cpu, 3, DType> y(y_ptr, Shape3(T, N, H));
  Tensor<cpu, 3, DType> dy(dy_ptr, Shape3(T, N, H));

  // current layer dcx and dhx
  Tensor<cpu, 2, DType> dcx_cl(dcx[0].dptr_, Shape2(N, H));
  Tensor<cpu, 2, DType> dhx_cl(dhx[0].dptr_, Shape2(N, H));
  LstmBackwardSingleLayer<DType>(ws, rs, D, T, N, I, H, x, hx[0], cx[0], y, dy, dx,
                                 dhx_cl, dcx_cl, dhy_ptr, dcy_ptr, w_ptr, dw_ptr);
}

template<typename DType>
void GruForwardInferenceSingleLayer(DType* ws,
                                    bool state_outputs,
                                    const int D,
                                    const int T,
                                    const int N,
                                    const int I,
                                    const int H,
                                    const Tensor<cpu, 2, DType> &x,
                                    const Tensor<cpu, 2, DType> &hx,
                                    const Tensor<cpu, 2, DType> &wx,
									const Tensor<cpu, 2, DType> &wh,
									const Tensor<cpu, 2, DType> &bx,
									const Tensor<cpu, 2, DType> &bh,
                                    DType* y_ptr,
                                    DType* hy_ptr) {

  #pragma omp parallel for collapse(2)
  for (int i = 0; i < N; i++)
        for (int j = 0; j < H; j++) {
            y_ptr[i * H + j] = hx[i][j];
        }

  DType* ht = y_ptr;
  DType* ht_1 = y_ptr;  
  DType* gemmC1  = ws;              // [D, T, N, 3 * H]                
  DType* gemmC2  = gemmC1 + D * T * N * 3 * H;  // N * 3 * H  
  DType* rt = gemmC2 + N * 3 * H;
  DType* zt = rt + N * H;
  DType* nt = zt + N * H;  
  DType* gemmC1_t = gemmC1;

  
  Tensor<cpu, 2, DType> dgemmC1(ws, Shape2(D * T * N, 3 * H));
  Tensor<cpu, 2, DType> dgemmC2(gemmC2, Shape2(D * N, 3 * H));


  //x * wx.T : [T * N, I] * [I, 3 * H]  
  DType alpha = 1.0;
  DType beta = 0.0;  
  linalg_gemm(x, wx, dgemmC1, alpha, beta, false, true);

  for (int t = 0; t < T; t++) {
      //  perform the first direction, X * wx and H * wh for each step
      //  ht-1 * wh, ht-1:[N, H] wh:[3 * H, H]      
	  Tensor<cpu, 2, DType> dht_1(ht_1, Shape2(N, D * H)); 

      linalg_gemm(dht_1, wh, dgemmC2, alpha, beta, false, true);

      gemmC1_t = gemmC1 + t * N * 3 * H;
        
      #pragma omp parallel for collapse(2)
      for (int i = 0; i < N; ++i) {
          for (int j = 0; j < H; ++j) {                
              int rtb = i * 3 * H;
              int ztb = i * 3 * H + H;
              int ntb = i * 3 * H + 2 * H;
			  
              rt[i * H + j] = 1/(1 + exp(-gemmC1_t[rtb + j] - gemmC2[rtb + j]
                  - bx[0][j] - bh[0][j]));
              zt[i * H + j] = 1/(1 + exp(-gemmC1_t[ztb + j] - gemmC2[ztb + j]
                  - bx[1][j] - bh[1][j]));             
			  nt[i * H + j] = tanh(gemmC1_t[ntb + j] + bx[2][j] +
                  rt[i * H + j] * (gemmC2[ntb + j] + bh[2][j]));
              ht[i * D * H + j] = (1-zt[i * H + j]) * nt[i * H + j] +
                    zt[i * H + j] * ht_1[i * D * H + j];
          }
      }        

      ht_1 = ht;
      ht = ht + D * H * N;
  }    
  //  copy last state to hy, from(N, H * D) to (D, N, H)
  if (state_outputs) {
        
      DType* y_start = y_ptr + (T - 1) * N * H;
      #pragma omp parallel for collapse(2)
      for (int i = 0; i < N; i++)
          for (int j = 0; j < H; j++) {
              hy_ptr[i * H + j] = y_start[i * H + j];
          }
  
  }

}

template <typename DType>
void GruForwardInference(DType* ws,
                          bool state_outputs,
                          const int L,
                          const int D,
                          const int T,
                          const int N,
                          const int I,
                          const int H,
                          DType* x_ptr,
                          DType* hx_ptr,
                          DType* w_ptr,
                          DType* y_ptr,
                          DType* hy_ptr) {

  const Tensor<cpu, 2, DType> wx(w_ptr, Shape2(H * 3, I));
  const Tensor<cpu, 2, DType> wh(w_ptr + I * H * 3, Shape2(H * 3, H));
  const Tensor<cpu, 2, DType> bx(wh.dptr_ + H * H * 3, Shape2(3, H));
  const Tensor<cpu, 2, DType> bh(bx.dptr_ + H * 3, Shape2(3, H));  
   

  Tensor<cpu, 2, DType> x(x_ptr, Shape2(T * N, I));
  Tensor<cpu, 3, DType> hx(hx_ptr, Shape3(L, N, H));
  Tensor<cpu, 3, DType> hy(hy_ptr, Shape3(L, N, H));

  Tensor<cpu, 2, DType> x_l = x;
  Tensor<cpu, 2, DType> hx_l = hx[0];
  
  DType* y_tmp = ws;
  DType* hy_l = hy_ptr;  
  DType* y_l = y_ptr;
  DType* ws2 = y_tmp + D * T * N * H;

  const Tensor<cpu, 2, DType> wx_l = wx;
  const Tensor<cpu, 2, DType> wh_l = wh;
  const Tensor<cpu, 2, DType> bx_l = bx;
  const Tensor<cpu, 2, DType> bh_l = bh;


  GruForwardInferenceSingleLayer<DType>(ws2, state_outputs, D, T, N, I, H,
                                         x_l, hx_l, wx_l, wh_l, bx_l, bh_l, y_l, hy_l);
  
}

#endif  // MXNET_OPERATOR_RNN_IMPL_HPP_
