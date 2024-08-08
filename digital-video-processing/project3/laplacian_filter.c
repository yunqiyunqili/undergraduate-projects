#define _CRT_SECURE_NO_DEPRECATE
#include <stdio.h>
#include <stdlib.h>

// Function to apply convolution mask
int Mask(unsigned char* img, int i, int j, int height, int width, int* mask, int order)
{
    int sum = 0;
    int offset_x = i - 1;
    int offset_y = j - 1;
    
    for (int m = 0; m < order; m++)
    {
        for (int n = 0; n < order; n++)
        {
            if ((offset_x + m < 0) || (offset_y + n < 0) || (offset_x + m + 1 >= height) || (offset_y + n + 1 >= width))
                sum += 0; // Boundary condition
            else
                sum += img[(offset_x + m) * width + (offset_y + n)] * mask[m * order + n];
        }
    }
    return abs(sum);
}

int main()
{
    // Define Laplacian mask
    int Laplacian[9] = {1, 1, 1, 1, -8, 1, 1, 1, 1};
    int moon_width = 464;
    int moon_height = 538;
    int moon_size = moon_height * moon_width * 3;

    FILE *fin = fopen("/Users/yunqili/Documents/本科课程作业/大三上/数字视频处理/digital-video-processing/project1/lena_noise.yuv", "rb");
    if (fin == NULL)
    {
        perror("Failed to open file");
        return -1;
    }

    unsigned char *moon = (unsigned char *)malloc(moon_size);
    if (moon == NULL)
    {
        fprintf(stderr, "Memory allocation failed\n");
        fclose(fin);
        return -1;
    }

    fread(moon, 1, moon_size, fin);
    fclose(fin);

    unsigned char *y = moon;
    unsigned char *u = moon + moon_width * moon_height;
    unsigned char *v = u + moon_width * moon_height;

    for (int i = 0; i < moon_height; i++)
    {
        for (int j = 0; j < moon_width; j++)
        {
            y[i * moon_width + j] = Mask(y, i, j, moon_height, moon_width, Laplacian, 3);
        }
    }

    // Set U and V channels to 128
    for (int i = 0; i < moon_height; i++)
    {
        for (int j = 0; j < moon_width; j++)
        {
            u[i * moon_width + j] = 128;
            v[i * moon_width + j] = 128;
        }
    }

    FILE *fout = fopen("/Users/yunqili/Documents/本科课程作业/大三上/数字视频处理/digital-video-processing/project3/lena_noise_Laplacian.yuv", "wb");
    if (fout == NULL)
    {
        perror("Failed to open output file");
        free(moon);
        return -1;
    }

    fwrite(moon, 1, moon_size, fout);
    fclose(fout);
    free(moon);

    return 0;
}
