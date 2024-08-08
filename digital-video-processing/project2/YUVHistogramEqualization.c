/*
 * 文件名: YUVHistogramEqualization.c
 * 功能: 对 YUV 图像进行直方图均衡化处理，并将其从 4:4:4 格式转换为 4:2:2 格式。
 * 说明:
 *   1. 读取 YUV 图像数据。
 *   2. 对灰度分量进行直方图均衡化。
 *   3. 将 U 和 V 分量从 4:4:4 格式转换为 4:2:2 格式。
 *   4. 将处理后的数据保存到新文件中。
 */
#define _CRT_SECURE_NO_DEPRECATE
#include <stdio.h>
#include <stdlib.h>

int main() {
    int width = 500;          // 图片的宽度
    int height = 500;         // 图片的高度
    int len = width * height; // 图像数据的大小

    // 打开输入文件
    FILE *fin = fopen("/Users/yunqili/Documents/本科课程作业/大三上/数字视频处理/digital-video-processing/project1/lena_noise.yuv", "rb");
    if (fin == NULL) {
        perror("Failed to open input file");
        return -1;
    }

    // 为转换后的数据分配内存
    unsigned char *buff1 = (unsigned char *)malloc(len * 3);
    if (buff1 == NULL) {
        fprintf(stderr, "Memory allocation failed for buff1\n");
        fclose(fin);
        return -2;
    }

    // 读取图像数据
    fread(buff1, 1, len * 3, fin);
    fclose(fin);

    // 设置缓存区指针
    unsigned char *buff2 = buff1;

    // 统计每个灰度级像素点的个数
    unsigned int sum[256] = {0};
    unsigned int sum_all[256] = {0};

    for (int i = 0; i < len * 3; i += 3) {
        sum[*buff2]++;
        buff2++;
    }

    // 计算累加灰度值
    for (int i = 0; i < 256; i++) {
        for (int j = 0; j <= i; j++) {
            sum_all[i] += sum[j];
        }
    }

    // 重置指针
    buff2 = buff1;

    // 归一化处理
    for (int i = 0; i < len * 3; i += 3) {
        *buff2 = (255 * sum_all[*buff2] + (width * height / 2)) / (width * height);
        buff2++;
    }

    // 处理 U 和 V 分量，转化为 4:2:2 格式
    unsigned char *u = buff1 + width * height;
    unsigned char *v = u + width * height / 2;

    for (int i = 0; i < height; i++) {
        fread(buff2, 1, width * 3, fin);
        for (int j = 0; j < width * 3; j += 6) {
            *u = 128; // U 分量
            *v = 128; // V 分量
            u++;
            v++;
        }
    }

    // 打开输出文件
    FILE *fout = fopen("/Users/yunqili/Documents/本科课程作业/大三上/数字视频处理/digital-video-processing/project2/lena_noise_equalized", "wb");
    if (fout == NULL) {
        perror("Failed to open output file");
        free(buff1);
        return -3;
    }

    // 写入处理后的数据到输出文件
    fwrite(buff1, 1, len * 3, fout);
    fclose(fout);
    free(buff1);

    return 0;
}
