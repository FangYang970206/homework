% 图像滤波
%% 中值滤波
img_orig = imread('cameraman.tif');
img_noise=imnoise(img_orig, 'salt & pepper', 0.03);
img_noise=imnoise(img_noise, 'gaussian', 0, 0.02);
img_recover1 = medfilt2(img_noise);
figure();
subplot(1, 3, 1);
imshow(img_orig);
title('原图');
subplot(1, 3, 2);
imshow(img_noise);
title('噪声图');
subplot(1, 3, 3);
imshow(img_recover1);
title('中值滤波恢复图');
%% 均值滤波
H1 = fspecial('average', 3);
img_recover2 = imfilter(img_noise, H1);
figure();
subplot(1, 3, 1);
imshow(img_orig);
title('原图');
subplot(1, 3, 2);
imshow(img_noise);
title('噪声图');
subplot(1, 3, 3);
imshow(img_recover2);
title('均值滤波恢复图');
%% 中值加均值滤波
img_recover3 = medfilt2(img_noise);
img_recover3 = imfilter(img_recover3, H1);
figure();
subplot(1, 3, 1);
imshow(img_orig);
title('原图');
subplot(1, 3, 2);
imshow(img_noise);
title('噪声图');
subplot(1, 3, 3);
imshow(img_recover3);
title('中值加均值滤波恢复图');

%% DCT
%读取源图像
I=imread('cameraman.tif');
%对图像进行离散余弦变换
J=dct2(I);
v = flip(sort(abs(J(:))));
%1/4
c1 = v(length(v)/4);
[col row] = size(find(abs(J)< c1));
A=col*row;%置为0的变换系数的个数
%1/8
c2 = v(length(v)/8);
[col row]= size(find(abs(J)< c2));
B=col*row;%置为0的变换系数的个数
%1/16
c3 = v(length(v)/16);
[col row]= size(find(abs(J)< c3));
C=col*row;%置为0的变换系数的个数
%将小于1/4的最大值变换系数置为0后做离散余弦反变换
J(abs(J) < c1 ) = 0;I1=idct2(J);
%将小于1/8的最大值变换系数置为0后做离散余弦反变换
J(abs(J) < c2) = 0;I2=idct2(J);
%将小于1/16的最大值变换系数置为0后做离散余弦反变换
J(abs(J) < c3) = 0;I3=idct2(J);
%显示原图及反变换结果
figure(2);
subplot(2,2,1);
imshow(I);
title('原图');
subplot(2,2,2);
imshow(I1,[0,255]);
title('小于1/4最大值');
subplot(2,2,3);
imshow(I2,[0,255]);
title('小于1/8最大值');
subplot(2,2,4);
imshow(I3,[0,255]);
title('小于1/16最大值');
%计算反重构时，DCT的变换系数的置0个数小于5时的峰值信噪比及置为0的变换系数的个数
I = double(I);
I1 = double(I1);
[Row,Col] = size(I);
[Row,Col] = size(I1);
MSE1 = sum(sum((I-I1).^2))/(Row * Col);
PSNR1 = 10 * log10(255^2/MSE1);
fprintf('图像的峰值信噪比：PSNR1=%f\n',PSNR1);
%计算反重构时，DCT的变换系数的置0个数小于10时的峰值信噪比及置为0的变换系数的个数
I = double(I);
I2 = double(I2);
[Row,Col] = size(I);
[Row,Col] = size(I2);
MSE2 = sum(sum((I-I2).^2))/(Row * Col);
PSNR2 = 10 * log10(255^2/MSE2);
fprintf('图像的峰值信噪比：PSNR2=%f\n',PSNR2);
%计算反重构时，DCT的变换系数的置0个数小于20时的峰值信噪比及置为0的变换系数的个数
I = double(I);
I3 = double(I3);
[Row,Col] = size(I);
[Row,Col] = size(I3);
MSE3 = sum(sum((I-I3).^2))/(Row * Col);
PSNR3 = 10 * log10(255^2/MSE3);
fprintf('图像的峰值信噪比：PSNR3=%f\n',PSNR3);