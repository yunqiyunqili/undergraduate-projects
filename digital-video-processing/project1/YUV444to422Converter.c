#define _CRT_SECURE_NO_DEPRECATE
#include <stdio.h>
#include <stdlib.h>



int main()
{
    int width;                          // 图片的宽度
    int height;                         // 图片的高度

    FILE *fin;
    fin = fopen("/Users/yunqili/Documents/本科课程作业/大三上/数字视频处理/lena_noise.yuv", "rb");
    if (NULL == fin)
    {
        perror("open file is failed\n");
        return -2;
    }

    printf("请输入图片的宽和高，以逗号分隔：");
    scanf("%d,%d", &width, &height);  // 输入图片宽高 

    int len = width * height * 2;    // 文件大小 (4:2:2)
    unsigned char *data = (unsigned char *)malloc(len);  // 转换后数据存放的位置
    if (NULL == data)
    {
        fprintf(stderr, "malloc data failed\n");
        fclose(fin);
        return -3;
    }

    /* 配置 y u v 数据位置 */
    unsigned char *y = data;                    // y数据存放的位置
    unsigned char *u = data + width * height;   // u数据存放的位置
    unsigned char *v = u + width * height / 2;  // v数据存放的位置 

    unsigned char *buff = (unsigned char *)malloc(width * 3);  // 设置缓存区
    if (NULL == buff)
    {
        fprintf(stderr, "malloc buff failed\n");
        fclose(fin);
        return -4;
    }

    /* 读取并转换数据 */
    int i, j;
    for (i = 0; i < height; i++)
    {
        fread(buff, 1, width * 3, fin);  // 读取YUV 4:4:4数据
        for (j = 0; j < width; j += 2)
        {
            *y++ = buff[3 * j];      // Y
            *y++ = buff[3 * (j + 1)]; // Y
        }
    }

    fseek(fin, 0, SEEK_SET); // 重置文件指针
    for (i = 0; i < height; i++)
    {
        fread(buff, 1, width * 3, fin);  // 读取YUV 4:4:4数据
        for (j = 0; j < width; j += 2)
        {
            *u++ = buff[3 * j + 1];  // U
            *v++ = buff[3 * j + 2];  // V
        }
    }

    /* 创建目标文件并把数据写入到文件中 */
    FILE *fout;
    fout = fopen("/Users/yunqili/Documents/本科课程作业/大三上/数字视频处理/lena_noise_422.yuv", "wb");
    
    if (NULL == fout)
    {
        perror("open newfile is failed\n");
        return -3;
    }

    fwrite(data, 1, len, fout);
    fclose(fout);
    fclose(fin);
    free(data);
    free(buff);

    printf("转换完成，输出文件为C:\\test\\output\\seed_new.yuv\n");

    return 0;
}
