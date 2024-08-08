// 3*3的邻域平均模板对灰度图lena_noise.yuv进⾏平滑滤波

#define _CRT_SECURE_NO_DEPRECATE
#include <stdio.h>
#include <stdlib.h>

#define WIDTH 512
#define HEIGHT 512
#define TEMPLATE_SIZE 3

// 模板卷积函数
int Mask(unsigned char* image, int x, int y, int height, int width, int* kernel, int order) {
    int sum = 0;
    int x_start = x - order / 2;
    int y_start = y - order / 2;

    for (int m = 0; m < order; m++) {
        for (int n = 0; n < order; n++) {
            int xi = x_start + m;
            int yi = y_start + n;

            if (xi < 0 || yi < 0 || xi >= height || yi >= width) {
                sum += 0; // 边界处理
            } else {
                sum += image[xi * width + yi] * kernel[m * order + n];
            }
        }
    }
    return sum >= 0 ? sum : abs(sum);
}

// 平滑滤波主函数
void smooth_image(const char* input_file, const char* output_file, int* kernel, int kernel_size, int divisor) {
    FILE *fin = fopen(input_file, "rb");
    if (!fin) {
        perror("Failed to open input file");
        exit(-2);
    }

    unsigned char *lena = (unsigned char *)malloc(WIDTH * HEIGHT);
    if (!lena) {
        fprintf(stderr, "Failed to allocate memory for image\n");
        fclose(fin);
        exit(-3);
    }

    unsigned char *buff = (unsigned char *)malloc(WIDTH * HEIGHT);
    if (!buff) {
        fprintf(stderr, "Failed to allocate memory for buffer\n");
        fclose(fin);
        free(lena);
        exit(-4);
    }

    fread(buff, 1, WIDTH * HEIGHT, fin);
    fclose(fin);

    for (int i = 0; i < HEIGHT; i++) {
        for (int j = 0; j < WIDTH; j++) {
            lena[i * WIDTH + j] = Mask(buff, i, j, HEIGHT, WIDTH, kernel, kernel_size) / divisor;
        }
    }

    FILE *fout = fopen(output_file, "wb");
    if (!fout) {
        perror("Failed to open output file");
        free(buff);
        free(lena);
        exit(-3);
    }

    fwrite(lena, 1, WIDTH * HEIGHT, fout);
    fclose(fout);

    free(buff);
    free(lena);
    printf("Smoothing completed and saved to %s\n", output_file);
}

int main() {
    // 平滑模板
    int Smooth[9] = {1,1,1,1,1,1,1,1,1};
    int Gaussian[9] = {1,2,1,2,4,2,1,2,1};
    int Laplace[9] = {1,1,1,1,-8,1,1,1,1};
    int Sobel[9] = {-1,-2,-1,0,0,0,1,2,1};

    // 对图像应用不同的模板
    smooth_image("C:\\lena_noise.yuv", "C:\\test\\output\\lena_smooth_1.yuv", Smooth, TEMPLATE_SIZE, 9);
    smooth_image("C:\\test\\output\\lena_smooth_2.yuv", "C:\\test\\output\\lena_smooth_2.yuv", Gaussian, TEMPLATE_SIZE, 16);

    return 0;
}
