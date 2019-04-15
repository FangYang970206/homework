%% 图像反转
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

%% 对比度拉伸
imgContrastStretch=imadjust(imgOrig, [0.1 0.4], [0.3 0.5 ])
figure();
subplot(1, 2, 1);
imshow(imgOrig);
title('Origin');
subplot(1, 2, 2);
imshow(imgContrastStretch);
title('ContrastStretch');

%% 动态范围压缩
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

%% 灰度切片
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

%% 图像相减
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

%% 图像平均
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

%% 平滑操作
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

%% 中值滤波
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

%% 锐化
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

%% 偏导
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

%% 低通滤波 ref https://blog.csdn.net/ytang_/article/details/75451934
img_origin=rgb2gray(imread('lena.png'));
d0=50;  %截止频率
img_noise=imnoise(img_origin,'salt'); % 加椒盐噪声
img_f=fftshift(fft2(double(img_noise)));  %傅里叶变换得到频谱
[m n]=size(img_f);
m_mid=fix(m/2);  
n_mid=fix(n/2);  
img_lpf=zeros(m,n);
for i=1:m
    for j=1:n
        d=sqrt((i-m_mid)^2+(j-n_mid)^2);   %理想低通滤波，求距离
        if d<=d0
            h(i,j)=1;
        else
            h(i,j)=0;
        end
        img_lpf(i,j)=h(i,j)*img_f(i,j);  
    end
end

img_lpf=ifftshift(img_lpf);    %反傅里叶变换
img_lpf=uint8(real(ifft2(img_lpf)));  %取实数部分

subplot(1,3,1);imshow(img_origin);title('原图');
subplot(1,3,2);imshow(img_noise);title('噪声图');
subplot(1,3,3);imshow(img_lpf);title('理想低通滤波');

%% 高通滤波
img_origin8 = rgb2gray(imread('lena.png'));
g= fftshift(fft2(double(img_origin8)));
[N1,N2]=size(g);
n=2;
d0=30; 
%d0是终止频率
n1=fix(N1/2);
n2=fix(N2/2);
%n1，n2指中心点的坐标，fix()函数是往0取整
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
subplot(2,2,1); imshow(img_origin8); title('原图');
subplot(2,2,2); imshow(abs(g),[]); title('原图频谱');
subplot(2,2,3); imshow(final); title('高通滤波后的图像');
subplot(2,2,4); imshow(abs(result), []); title('高通滤波后的频谱');

%% 带通滤波
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
subplot(2,2,1); imshow(img_origin9); title('原图');
subplot(2,2,2); imshow(abs(fftI),[]); title('原图频谱');
subplot(2,2,3); imshow(final); title('带通滤波后的图像');
subplot(2,2,4); imshow(abs(result), []); title('带通滤波后的频谱');

%% 同态滤波
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
        H(i,j)=(rH-rL).*(exp(c*(-D(i,j)./(d0^2))))+rL;%高斯同态滤波
    end
end
G = H.*FI;
final=ifft2(G);
final=real(exp(final));
figure();
subplot(2,2,1); imshow(img_origin10); title('原图');
subplot(2,2,2); imshow(abs(FI),[]); title('原图频谱');
subplot(2,2,3); imshow(final, []); title('同态滤波后的图像');
subplot(2,2,4); imshow(abs(G), []); title('同态滤波后的频谱');

%% 伪彩色
img_origin11 = rgb2gray(imread('lena.png')); 
FalseRGB = label2rgb(gray2ind(img_origin11, 255),jet(255));
figure();
subplot(1,2,1); imshow(img_origin11); title('原图');
subplot(1,2,2); imshow(FalseRGB); title('伪彩色');

%% 全彩色处理，中值滤波为例
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