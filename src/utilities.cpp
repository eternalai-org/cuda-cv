#include <stdio.h>
#include <utilities.cuh>

void printmat3d(long long* mat, int h, int w, int c)
{
    for (int i = 0; i < h; ++i)
    {
        for (int j = 0; j < w; ++j)
        {
            printf("[");
            for (int k = 0; k < c; ++k)
            {
                printf("%lf ", 1.0f * mat[i * w * c + j * c + k] / (1LL << 32));
            }
            printf("], ");
        }
        printf("\n");
    }
}

