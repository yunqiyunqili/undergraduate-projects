clear;
close all;


I = imread('rice.png');
n=graythresh(I);
L=im2bw(I,n); % 原图二值化
[height,width] = size(L);
labelmap=zeros(height,width); %初始化一个与输入图像同样尺寸的标记矩阵labelmap

subplot(1,2,1);
imshow(I); % 显示原图

%图像预处理：开运算去噪
SE = strel('square', 5); 
L=imerode(L,SE);
L=imdilate(L,SE);

subplot(1,2,2);
imshow(L); % 显示二值化图像
L = uint8(L);%把L由logic类型转化为uint8类型



for i = 1:height
    for j = 1:width
        if L(i,j) == 1
            L(i,j) = 255;%把白色像素点像素值赋值为255
        end
    end
end

MAXSIZE = 999999;
queue = zeros(MAXSIZE,2);%用数组模拟队列,存储种子点坐标
front = 1;%指明队头的位置
rear = 1;%指明队尾的下一个位置；front=rear表示队空
labelIndex = 0;%米粒的标号

for i = 1:height
    for j = 1:width
        if L(i,j) == 255 %白色像素点入队列
            if front == rear %队列空，找到新米粒，米粒标号加一
                labelIndex = labelIndex+1;
            end
            L(i,j) = labelIndex; %给白色像素赋值为米粒的标号
            labelmap(i,j) = labelIndex;
            queue(rear,1) = i;
            queue(rear,2) = j;
            rear = rear+1;%队尾后移
            while front ~= rear
                %队头出队
                temp_i = queue(front,1);
                temp_j = queue(front,2);
                front = front + 1;
                %把队头位置像素点8连通邻域中未作标记的白色像素点入队,并加上米粒标号
                
                
                if(((temp_i - 1)>0)&&((temp_j - 1)>0))
                    %左上角的像素点
                    if L(temp_i - 1,temp_j - 1) == 255
                        L(temp_i - 1,temp_j - 1) = labelIndex;
                        labelmap(temp_i - 1,temp_j - 1) = labelIndex;
                        queue(rear,1) = temp_i - 1;
                        queue(rear,2) = temp_j - 1;
                        rear = rear + 1;
                    end
                end
                if(((temp_i - 1)>0)&&((temp_j + 1)<=width))
                    %右上方的像素点
                    if L(temp_i - 1,temp_j + 1) == 255
                        L(temp_i - 1,temp_j + 1) = labelIndex;
                        labelmap(temp_i - 1,temp_j + 1) = labelIndex;
                        queue(rear,1) = temp_i - 1;
                        queue(rear,2) = temp_j + 1;
                        rear = rear + 1;
                    end
                end
                if(((temp_i + 1)<=height)&&((temp_j - 1)>0))
                    %左下方的像素点
                    if L(temp_i + 1,temp_j - 1) == 255
                        L(temp_i + 1,temp_j - 1) = labelIndex;
                        labelmap(temp_i + 1,temp_j - 1) = labelIndex;
                        queue(rear,1) = temp_i + 1;
                        queue(rear,2) = temp_j - 1;
                        rear = rear + 1;
                    end
                end
                if(((temp_i + 1)<=height)&&((temp_j + 1)<=width))
                    %右下方的像素点
                    if L(temp_i + 1,temp_j + 1) == 255
                        L(temp_i + 1,temp_j + 1) = labelIndex;
                        labelmap(temp_i + 1,temp_j + 1) = labelIndex;
                        queue(rear,1) = temp_i + 1;
                        queue(rear,2) = temp_j + 1;
                        rear = rear + 1;
                    end
                end

                
                if(((temp_i - 1)>0)&&((temp_j )>0))
                    %正上方的像素点
                    if L(temp_i - 1,temp_j) == 255
                        L(temp_i - 1,temp_j) = labelIndex;
                        labelmap(temp_i - 1,temp_j) = labelIndex;
                        queue(rear,1) = temp_i - 1;
                        queue(rear,2) = temp_j;
                        rear = rear + 1;
                    end
                end
                if(((temp_i )>0)&&((temp_j - 1)>0))
                    %正左方的像素点
                    if L(temp_i,temp_j - 1) == 255
                        L(temp_i,temp_j - 1) = labelIndex;
                        labelmap(temp_i,temp_j - 1) = labelIndex;
                        queue(rear,1) = temp_i;
                        queue(rear,2) = temp_j - 1;
                        rear = rear + 1;
                    end
                end
                if(((temp_i )>0)&&((temp_j + 1)<=width))
                    %正右方的像素点
                    if L(temp_i,temp_j + 1) == 255
                        L(temp_i,temp_j + 1) = labelIndex;
                        labelmap(temp_i,temp_j + 1) = labelIndex;
                        queue(rear,1) = temp_i;
                        queue(rear,2) = temp_j + 1;
                        rear = rear + 1;
                    end
                end
                if(((temp_i + 1)<=height)&&((temp_j )>0))
                    %正下方的像素点
                    if L(temp_i + 1,temp_j) == 255
                        L(temp_i + 1,temp_j) = labelIndex;
                        labelmap(temp_i + 1,temp_j) = labelIndex;
                        queue(rear,1) = temp_i + 1;
                        queue(rear,2) = temp_j;
                        rear = rear + 1;
                    end
                end
            end
        end
    end
end
figure,imshow(L);
RiceNumber = labelIndex;%记录米粒的总个数
labelmap
fprintf('8连通米粒的总个数:%d',RiceNumber)

