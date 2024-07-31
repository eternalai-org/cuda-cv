#pragma once

#if __cplusplus
extern "C" {
#endif

// element wise operations
void matAddLongLong(
    long long *A, long long *B, // input mat 
    long long *C, // output mat
    int m, int n // spatial size
);

void matSubLongLong(
    long long *A, long long *B, // input mat 
    long long *C, // output mat
    int m, int n // spatial size
);

void matMulLongLong(
    long long *A, long long *B, // input mat 
    long long *C, // output mat
    int m, int n // spatial size
);

void matDivLongLong(
    long long *A, long long *B, // input mat 
    long long *C, // output mat
    int m, int n // spatial size
);

void matSqrtLongLong(
    long long *A, // input mat
    long long *B, // output mat
    int m, int n // spatial size
);

// conv2d operations
void conv2dFixedLongLong(
    long long* inp, long long* kernel, long long* bias, long long* out, // data io
    int kernel_size, int in_channel, int out_channel, // kernel properties
    int h, int w, // spatial size of inp,
    int padding, int stride_h, int stride_w // padding: same(0) or valid(1)
);

void estimateConvOutputSize(
    int kernel_size, int in_channel, int out_channel, // kernel properties
    int h, int w, // spatial size of inp,
    int padding, int stride_h, int stride_w, // padding: same(0) or valid(1)
    int& out_h, int& out_w // spatial size of out
);

// pooling operations
void avgPoolingFixedLongLong(
    long long* inp, long long* out,
    int h, int w, int in_channel,
    int pool_size, int stride_h, int stride_w,
    int padding
);

void maxPoolingFixedLongLong(
    long long* inp, long long* out,
    int h, int w, int in_channel,
    int pool_size, int stride_h, int stride_w,
    int padding
);

void estimatePoolingOutputSize(
    int h, int w, int in_channel,
    int pool_size, int padding, 
    int stride_h, int stride_w,
    int& out_h, int& out_w
); // for avg pooling and max pooling only

void globalAvgPoolingFixedLongLong(
    long long* inp, long long* out,
    int h, int w, int in_channel
);


// reduction
long long sumReduction(long long* inp, int n);
long long avgReduction(long long* inp, int n);
long long maxReduction(long long* inp, int n);
long long minReduction(long long* inp, int n);
long long meanReduction(long long* inp, int n);
long long stdReduction(long long* inp, int n);
void maxMinScale(long long* inp, long long* out, int n);
void zScore(long long* inp, long long* out, long long eps, int n);

// merging
bool estimateConcatenate(
    long long* shapes, 
    long long axis, 
    long long ndims, 
    long long n, 
    long long* out
);

void concatenate(
    long long* inp, 
    long long* out, 
    long long* shapes, 
    long long axis, 
    long long ndims, 
    long long n
);

// activations
void softmaxFixedLongLong(long long *A, long long* B, int m);
void sigmoidFixedLongLong(long long *A, long long* B, int m);
void tanhFixedLongLong(long long *A, long long *B, int m);
void reluFixedLongLong(long long *A, long long *B, int m);
void relu3DFixedLongLong(long long *A, long long *B, int h, int w, int c);
void sigmoid3DFixedLongLong(long long *A, long long *B, int h, int w, int c);
void tanh3DFixedLongLong(long long *A, long long *B, int h, int w, int c);
void softmax2DFixedLongLong(long long* A, long long* B, int h, int w, int c);

// normalization
void layerNormalizeFixedLongLong(
    long long *X, // input of shape (m, n, c) 
    long long *Y, // output of shape (m, n, c)
    long long *ma, // moving average (c)
    long long *mv, // mong variance  (c)
    long long *gamma, // scale (c)
    long long *beta, // offset (c)
    long long epsilon, // epsilon 
    int h, int w, int c  // m, n, c 
);

void batchNormalizeFixedLongLong(
    long long *X, // input of shape (m, n, c) 
    long long *Y, // output of shape (m, n, c)
    long long *ma, // moving average (c)
    long long *mv, // mong variance  (c)
    long long *gamma, // scale (c)
    long long *beta, // offset (c)
    long long epsilon, // epsilon 
    int h, int w, int c  // m, n, c 
);

void maxmulFloat(float *A, float *B, float *C, int m, int n, int k);
void maxmulFixedLongLong(long long *A, long long *B, long long *C, int m, int n, int k) ;
void maxmulLong(long *A, long *B, long *C, long m, long n, long k);
void maxmulInt(int *A, int *B, int *C, int m, int n, int k);
void maxmulDouble(double *A, double *B, double *C, int m, int n, int k);

#if __cplusplus
}
#endif