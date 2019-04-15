%% ͼ��ת
L = 255;
imgOrig = imread('mammogram.png');
figure();
subplot(1, 2, 1);
imshow(imgOrig);
title('Origin');
imgNegative = L - imgOrig;
subplot(1, 2, 2);
imshow(imgNegative);
title('Negative');

%% �Աȶ�����
imgContrastStretch=imadjust(imgOrig, [0.1 0.4], [0.3 0.5 ])
figure();
subplot(1, 2, 1);
imshow(imgOrig);
title('Origin');
subplot(1, 2, 2);
imshow(imgContrastStretch);
title('ContrastStretch');

%% ��̬��Χѹ��
img_orig1 = rgb2gray(imread('circle.png'));
c = 150 / log(1 + 255);
img_size = size(img_orig1);
for i = 1:img_size(1)
    for j = 1:img_size(2)
        s(i, j) = c * log(double(1 + img_orig1(i, j)));
    end
end
figure();
subplot(1, 2, 1);
imshow(img_orig1);
title('Origin');
subplot(1, 2, 2);
imshow(uint8(s));
title('Compression');

%% �Ҷ���Ƭ
img_orig2 = rgb2gray(imread('GLS.png'));
figure();
subplot(1, 2, 1);
imshow(img_orig2);
title('Origin');
img_size = size(img_orig2);
for i = 1:img_size(1)
    for j = 1:img_size(2)
        if img_orig2(i, j) > 100 || img_orig2(i, j) > 180
            img_orig2(i, j) = 150;
        else
            img_orig2(i, j) = 25;
        end
    end
end
subplot(1, 2, 2);
imshow(img_orig2);
title('Gray level slicing');

%% ͼ�����
img_orig3 = rgb2gray(imread('son1.png'));
img_add = uint16(img_orig3) + 100;
whitepaper = rgb2gray(imread('son2.png'));
img_sub = uint8(img_add - uint16(whitepaper));
figure();
subplot(1, 3, 1);
imshow(img_orig3);
title('Origin');
subplot(1, 3, 2);
imshow(whitepaper);
title('whitepaper');
subplot(1, 3, 3);
imshow(img_sub);
title('Subtraction');

%% ͼ��ƽ��
img_orig4 = rgb2gray(imread('cat.jpg'));
img_noise1 = imnoise(img_orig4, 'gaussian', 0, 0.01);
img_noise2 = imnoise(img_orig4, 'gaussian', 0, 0.01);
img_noise3 = imnoise(img_orig4, 'gaussian', 0, 0.01);
img_noise4 = imnoise(img_orig4, 'gaussian', 0, 0.01);
img_noise5 = imnoise(img_orig4, 'gaussian', 0, 0.01);
img_average = imlincomb(0.2,img_noise1, 0.2,img_noise2, 0.2,img_noise3, 0.2,img_noise4, 0.2,img_noise5);
figure();
subplot(1, 3, 1);
imshow(img_orig4);
title('Origin');
subplot(1, 3, 2);
imshow(img_noise1);
title('img_noise1');
subplot(1, 3, 3);
imshow(img_average);
title('img_average');

%% ƽ������
img_orig5 = imread('a.png');
H1 = fspecial('average', 3);
img_smooth1 = imfilter(img_orig5, H1);
H2 = fspecial('average', 7);
img_smooth2 = imfilter(img_orig5, H2);
H3 = fspecial('average', 11);
img_smooth3 = imfilter(img_orig5, H3 );
figure();
subplot(2, 2, 1);
imshow(img_orig5);
title('Origin');
subplot(2, 2, 2);
imshow(img_smooth1);
title('kernel=3');
subplot(2, 2, 3);
imshow(img_smooth2);
title('kernel=7');
subplot(2, 2, 4);
imshow(img_smooth3);
title('kernel=11');

%% ��ֵ�˲�
img_orig6 = rgb2gray(imread('lena.png'));
img_noise6=imnoise(img_orig6, 'salt & pepper', 0.02);
img_recover = medfilt2(img_noise6);
figure();
subplot(1, 3, 1);
imshow(img_orig6);
title('Origin');
subplot(1, 3, 2);
imshow(img_noise6);
title('img_salt & pepper');
subplot(1, 3, 3);
imshow(img_recover);
title('img_recover');

%% ��
img_orig7 = imread('t.png');
H5 = fspecial('sobel');
edge = imfilter(img_orig7, H5);
sharpened = img_orig7 + edge;
figure();
subplot(1, 3, 1);
imshow(img_orig7);
title('Origin');
subplot(1, 3, 2);
imshow(edge);
title('Edge');
subplot(1, 3, 3);
imshow(sharpened);
title('img_sharpened');

%% ƫ��
img_orig7 = rgb2gray(imread('lena.png'));
H6 = fspecial('laplacian');
H7 = fspecial('sobel');
laplacian = imfilter(img_orig7, H6);
sobel = imfilter(img_orig7, H7);
figure();
subplot(1, 3, 1);
imshow(img_orig7);
title('Origin');
subplot(1, 3, 2);
imshow(laplacian);
title('laplacian');
subplot(1, 3, 3);
imshow(sobel);
title('sobel');

%% ��ͨ�˲� ref https://blog.csdn.net/ytang_/article/details/75451934
img_origin=rgb2gray(imread('lena.png'));
d0=50;  %��ֹƵ��
img_noise=imnoise(img_origin,'salt'); % �ӽ�������
img_f=fftshift(fft2(double(img_noise)));  %����Ҷ�任�õ�Ƶ��
[m n]=size(img_f);
m_mid=fix(m/2);  
n_mid=fix(n/2);  
img_lpf=zeros(m,n);
for i=1:m
    for j=1:n
        d=sqrt((i-m_mid)^2+(j-n_mid)^2);   %�����ͨ�˲��������
        if d<=d0
            h(i,j)=1;
        else
            h(i,j)=0;
        end
        img_lpf(i,j)=h(i,j)*img_f(i,j);  
    end
end

img_lpf=ifftshift(img_lpf);    %������Ҷ�任
img_lpf=uint8(real(ifft2(img_lpf)));  %ȡʵ������

subplot(1,3,1);imshow(img_origin);title('ԭͼ');
subplot(1,3,2);imshow(img_noise);title('����ͼ');
subplot(1,3,3);imshow(img_lpf);title('�����ͨ�˲�');

%% ��ͨ�˲�
img_origin8 = rgb2gray(imread('lena.png'));
g= fftshift(fft2(double(img_origin8)));
[N1,N2]=size(g);
n=2;
d0=30; 
%d0����ֹƵ��
n1=fix(N1/2);
n2=fix(N2/2);
%n1��n2ָ���ĵ�����꣬fix()��������0ȡ��
for i=1:N1
  for j=1:N2
      d=sqrt((i-n1)^2+(j-n2)^2);  
    if d>=d0
        h=1;  
    else
        h=0;  
    end  
    result(i,j)=h*g(i,j); 
  end
end
final=ifft2(ifftshift(result));
final=uint8(real(final));
figure();
subplot(2,2,1); imshow(img_origin8); title('ԭͼ');
subplot(2,2,2); imshow(abs(g),[]); title('ԭͼƵ��');
subplot(2,2,3); imshow(final); title('��ͨ�˲����ͼ��');
subplot(2,2,4); imshow(abs(result), []); title('��ͨ�˲����Ƶ��');

%% ��ͨ�˲�
img_origin9=rgb2gray(imread('lena.png')); 
g= fftshift(fft2(double(img_origin9))); 
[N1,N2]=size(g);  
n=2;  
d0=0;  
d1=200;  
n1=fix(N1/2);  
n2=fix(N2/2);  
for i=1:N1  
   for j=1:N2  
    d=sqrt((i-n1)^2+(j-n2)^2);  
    if d>=d0 || d<=d1  
        h=1;  
    else
        h=0;  
    end  
    result(i,j)=h*g(i,j);  
   end  
end 
final=ifft2(ifftshift(result));
final=uint8(real(final));
figure();
subplot(2,2,1); imshow(img_origin9); title('ԭͼ');
subplot(2,2,2); imshow(abs(fftI),[]); title('ԭͼƵ��');
subplot(2,2,3); imshow(final); title('��ͨ�˲����ͼ��');
subplot(2,2,4); imshow(abs(result), []); title('��ͨ�˲����Ƶ��');

%% ̬ͬ�˲�
img_origin10 = rgb2gray(imread('homo.jpg')); 
[M,N]=size(img_origin10);
rL=0.5;
rH=4.7;
c=2;
d0=10;
log_img=log(double(img_origin10)+1);
FI=fft2(log_img);
n1=floor(M/2);
n2=floor(N/2);
for i=1:M
    for j=1:N
        D(i,j)=((i-n1).^2+(j-n2).^2);
        H(i,j)=(rH-rL).*(exp(c*(-D(i,j)./(d0^2))))+rL;%��˹̬ͬ�˲�
    end
end
G = H.*FI;
final=ifft2(G);
final=real(exp(final));
figure();
subplot(2,2,1); imshow(img_origin10); title('ԭͼ');
subplot(2,2,2); imshow(abs(FI),[]); title('ԭͼƵ��');
subplot(2,2,3); imshow(final, []); title('̬ͬ�˲����ͼ��');
subplot(2,2,4); imshow(abs(G), []); title('̬ͬ�˲����Ƶ��');

%% α��ɫ
img_origin11 = rgb2gray(imread('lena.png')); 
FalseRGB = label2rgb(gray2ind(img_origin11, 255),jet(255));
figure();
subplot(1,2,1); imshow(img_origin11); title('ԭͼ');
subplot(1,2,2); imshow(FalseRGB); title('α��ɫ');

%% ȫ��ɫ������ֵ�˲�Ϊ��
img_orig6 = imread('lena.png');
for i = 1:3
img_noise6(:, :, i) = imnoise(img_orig6(:, :, i), 'salt & pepper', 0.02);
img_recover(:, :, i) = medfilt2(img_noise6(:, :, i));
end
figure();
subplot(1, 3, 1);
imshow(img_orig6);
title('Origin');
subplot(1, 3, 2);
imshow(img_noise6);
title('img_salt & pepper');
subplot(1, 3, 3);
imshow(img_recover);
title('img_recover');