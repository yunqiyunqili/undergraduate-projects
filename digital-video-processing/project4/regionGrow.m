clear;
close all;


I = imread('rice.png');
n=graythresh(I);
L=im2bw(I,n); % ԭͼ��ֵ��
[height,width] = size(L);
labelmap=zeros(height,width); %��ʼ��һ��������ͼ��ͬ���ߴ�ı�Ǿ���labelmap

subplot(1,2,1);
imshow(I); % ��ʾԭͼ

%ͼ��Ԥ����������ȥ��
SE = strel('square', 5); 
L=imerode(L,SE);
L=imdilate(L,SE);

subplot(1,2,2);
imshow(L); % ��ʾ��ֵ��ͼ��
L = uint8(L);%��L��logic����ת��Ϊuint8����



for i = 1:height
    for j = 1:width
        if L(i,j) == 1
            L(i,j) = 255;%�Ѱ�ɫ���ص�����ֵ��ֵΪ255
        end
    end
end

MAXSIZE = 999999;
queue = zeros(MAXSIZE,2);%������ģ�����,�洢���ӵ�����
front = 1;%ָ����ͷ��λ��
rear = 1;%ָ����β����һ��λ�ã�front=rear��ʾ�ӿ�
labelIndex = 0;%�����ı��

for i = 1:height
    for j = 1:width
        if L(i,j) == 255 %��ɫ���ص������
            if front == rear %���пգ��ҵ���������������ż�һ
                labelIndex = labelIndex+1;
            end
            L(i,j) = labelIndex; %����ɫ���ظ�ֵΪ�����ı��
            labelmap(i,j) = labelIndex;
            queue(rear,1) = i;
            queue(rear,2) = j;
            rear = rear+1;%��β����
            while front ~= rear
                %��ͷ����
                temp_i = queue(front,1);
                temp_j = queue(front,2);
                front = front + 1;
                %�Ѷ�ͷλ�����ص�8��ͨ������δ����ǵİ�ɫ���ص����,�������������
                
                
                if(((temp_i - 1)>0)&&((temp_j - 1)>0))
                    %���Ͻǵ����ص�
                    if L(temp_i - 1,temp_j - 1) == 255
                        L(temp_i - 1,temp_j - 1) = labelIndex;
                        labelmap(temp_i - 1,temp_j - 1) = labelIndex;
                        queue(rear,1) = temp_i - 1;
                        queue(rear,2) = temp_j - 1;
                        rear = rear + 1;
                    end
                end
                if(((temp_i - 1)>0)&&((temp_j + 1)<=width))
                    %���Ϸ������ص�
                    if L(temp_i - 1,temp_j + 1) == 255
                        L(temp_i - 1,temp_j + 1) = labelIndex;
                        labelmap(temp_i - 1,temp_j + 1) = labelIndex;
                        queue(rear,1) = temp_i - 1;
                        queue(rear,2) = temp_j + 1;
                        rear = rear + 1;
                    end
                end
                if(((temp_i + 1)<=height)&&((temp_j - 1)>0))
                    %���·������ص�
                    if L(temp_i + 1,temp_j - 1) == 255
                        L(temp_i + 1,temp_j - 1) = labelIndex;
                        labelmap(temp_i + 1,temp_j - 1) = labelIndex;
                        queue(rear,1) = temp_i + 1;
                        queue(rear,2) = temp_j - 1;
                        rear = rear + 1;
                    end
                end
                if(((temp_i + 1)<=height)&&((temp_j + 1)<=width))
                    %���·������ص�
                    if L(temp_i + 1,temp_j + 1) == 255
                        L(temp_i + 1,temp_j + 1) = labelIndex;
                        labelmap(temp_i + 1,temp_j + 1) = labelIndex;
                        queue(rear,1) = temp_i + 1;
                        queue(rear,2) = temp_j + 1;
                        rear = rear + 1;
                    end
                end

                
                if(((temp_i - 1)>0)&&((temp_j )>0))
                    %���Ϸ������ص�
                    if L(temp_i - 1,temp_j) == 255
                        L(temp_i - 1,temp_j) = labelIndex;
                        labelmap(temp_i - 1,temp_j) = labelIndex;
                        queue(rear,1) = temp_i - 1;
                        queue(rear,2) = temp_j;
                        rear = rear + 1;
                    end
                end
                if(((temp_i )>0)&&((temp_j - 1)>0))
                    %���󷽵����ص�
                    if L(temp_i,temp_j - 1) == 255
                        L(temp_i,temp_j - 1) = labelIndex;
                        labelmap(temp_i,temp_j - 1) = labelIndex;
                        queue(rear,1) = temp_i;
                        queue(rear,2) = temp_j - 1;
                        rear = rear + 1;
                    end
                end
                if(((temp_i )>0)&&((temp_j + 1)<=width))
                    %���ҷ������ص�
                    if L(temp_i,temp_j + 1) == 255
                        L(temp_i,temp_j + 1) = labelIndex;
                        labelmap(temp_i,temp_j + 1) = labelIndex;
                        queue(rear,1) = temp_i;
                        queue(rear,2) = temp_j + 1;
                        rear = rear + 1;
                    end
                end
                if(((temp_i + 1)<=height)&&((temp_j )>0))
                    %���·������ص�
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
RiceNumber = labelIndex;%��¼�������ܸ���
labelmap
fprintf('8��ͨ�������ܸ���:%d',RiceNumber)

