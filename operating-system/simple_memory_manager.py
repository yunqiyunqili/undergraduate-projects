#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define SIZE_MIN 2
#define MEMSIZE_MAX 1024
#define FALSE 0
#define TRUE !FALSE

typedef int BOOL; // 为 int 型起一个新的别名 BOOL

typedef struct _MEM_LINK { // 声明链表结点
    char cName;              // 作业名
    int iStartAddr;          // 分区起始地址
    int iMemSize;            // 分区大小
    BOOL iState;             // 分区状态，1 表示已分配, 0 表示未分配
    struct _MEM_LINK* next;  // 指向下一个节点的指针
} MEM_LINK, *PMEM_LINK;

PMEM_LINK g_pslnkHead; // 全局变量，链表头指针

// 初始化内存使用情况
void init() {
    g_pslnkHead = (PMEM_LINK)malloc(sizeof(MEM_LINK));
    memset(g_pslnkHead, 0, sizeof(MEM_LINK));
    g_pslnkHead->iMemSize = MEMSIZE_MAX;
}

// 初始化目录选择
int menu() {
    int i;
    printf("\n\n1. 分配内存\n");
    printf("2. 回收内存\n");
    printf("3. 显示内存使用情况\n");
    printf("4. 退出\n");
    printf("\n请输入选择: ");
    scanf("%d", &i);
    getchar();
    return i;
}

// 分配内存函数，c 为作业名，usize 是要分配的大小
int my_malloc(char c, int usize) {
    PMEM_LINK psNewMem = NULL, plnkTmp = NULL;
    BOOL bRepeatName = FALSE;
    int iTmp = g_pslnkHead->iMemSize - usize;

    if (iTmp <= SIZE_MIN) { // 如果没有足够的空间分配
        return FALSE;
    }

    plnkTmp = g_pslnkHead;
    while (plnkTmp != NULL) {
        if (plnkTmp->cName == c) {
            bRepeatName = TRUE;
            break;
        }
        plnkTmp = plnkTmp->next;
    }

    if (bRepeatName) { // 如果作业名重复
        return FALSE;
    }

    // 创建新的节点
    psNewMem = (PMEM_LINK)malloc(sizeof(MEM_LINK));
    // 结构体设零
    memset(psNewMem, 0, sizeof(MEM_LINK));
    // 设置节点内容
    psNewMem->cName = c;
    psNewMem->iMemSize = usize;
    psNewMem->iStartAddr = MEMSIZE_MAX - g_pslnkHead->iMemSize;
    psNewMem->iState = TRUE;

    plnkTmp = g_pslnkHead;
    // 查找链表最尾节点
    while (plnkTmp->next != NULL) {
        plnkTmp = plnkTmp->next;
    }

    // 把新创建的节点加入到链表中
    plnkTmp->next = psNewMem;
    // 在整体内存中去掉已分配的部分
    g_pslnkHead->iMemSize -= usize;

    return TRUE;
}

// 回收内存函数，c 是撤销的进程的作业名
int my_free(char c) {
    PMEM_LINK plnkBK = g_pslnkHead, // 保留上次搜索的节点
              plnkTmp = g_pslnkHead->next;
    BOOL bFind = FALSE;
    int iFreeSize = 0;

    // 搜索链表
    while (plnkTmp != NULL) {
        if (plnkTmp->cName == c) {
            // 如果找到节点，退出循环
            bFind = TRUE;
            break;
        }
        plnkBK = plnkTmp;
        plnkTmp = plnkTmp->next;
    }

    if (bFind) {
        // 把找到的节点从链表中摘除并释放
        g_pslnkHead->iMemSize += plnkTmp->iMemSize;
        plnkBK->next = plnkTmp->next;
        // 保留要释放内存的大小
        iFreeSize = plnkTmp->iMemSize;
        // 释放
        free(plnkTmp);
        // 把未释放内存的开始地址提前，防止内存碎片
        plnkTmp = plnkBK->next;
        while (plnkTmp != NULL) {
            plnkTmp->iStartAddr -= iFreeSize;
            plnkTmp = plnkTmp->next;
        }
    }

    return bFind;
}

// 打印分区号、作业名、起始地址、分区大小、状态
void disp() {
    PMEM_LINK pTmp;
    int i = 0;
    pTmp = g_pslnkHead;
    printf("\n分区号 作业名 起始地址 分区大小 状态");
    while (pTmp) {
        printf("\n%4d     %c   %4d       %4d   %4d", i, pTmp->cName, pTmp->iStartAddr, pTmp->iMemSize, pTmp->iState);
        pTmp = pTmp->next;
        i++;
    }
}

// 创建 main 函数
int main() {
    int i;
    char c;
    init();
    i = menu();
    while (i != 4) {
        if (i == 1) { // 分配内存
            printf("\n作业名(一个字符): ");
            scanf("%c", &c);
            getchar(); // 清除缓冲区中的换行符
            printf("作业占内存大小：");
            scanf("%d", &i);
            getchar(); // 清除缓冲区中的换行符
            if (my_malloc(c, i)) {
                printf("\n分配成功！！！");
            } else {
                printf("\n分配失败！！！");
            }
        } else if (i == 2) { // 回收内存
            printf("\n输入要回收分区的作业名(一个字符): ");
            scanf("%c", &c);
            getchar(); // 清除缓冲区中的换行符
            if (my_free(c)) {
                printf("\n回收成功！！！");
            } else {
                printf("\n回收失败！！！");
            }
        } else if (i == 3) {
            disp(); // 显示内存使用情况
        }
        i = menu();
    }
    return 0;
}

