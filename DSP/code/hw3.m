% ͼ���˲�
%% ��ֵ�˲�
img_orig = imread('cameraman.tif');
img_noise=imnoise(img_orig, 'salt & pepper', 0.03);
img_noise=imnoise(img_noise, 'gaussian', 0, 0.02);
img_recover1 = medfilt2(img_noise);
figure();
subplot(1, 3, 1);
imshow(img_orig);
title('ԭͼ');
subplot(1, 3, 2);
imshow(img_noise);
title('����ͼ');
subplot(1, 3, 3);
imshow(img_recover1);
title('��ֵ�˲��ָ�ͼ');
%% ��ֵ�˲�
H1 = fspecial('average', 3);
img_recover2 = imfilter(img_noise, H1);
figure();
subplot(1, 3, 1);
imshow(img_orig);
title('ԭͼ');
subplot(1, 3, 2);
imshow(img_noise);
title('����ͼ');
subplot(1, 3, 3);
imshow(img_recover2);
title('��ֵ�˲��ָ�ͼ');
%% ��ֵ�Ӿ�ֵ�˲�
img_recover3 = medfilt2(img_noise);
img_recover3 = imfilter(img_recover3, H1);
figure();
subplot(1, 3, 1);
imshow(img_orig);
title('ԭͼ');
subplot(1, 3, 2);
imshow(img_noise);
title('����ͼ');
subplot(1, 3, 3);
imshow(img_recover3);
title('��ֵ�Ӿ�ֵ�˲��ָ�ͼ');

%% DCT
%��ȡԴͼ��
I=imread('cameraman.tif');
%��ͼ�������ɢ���ұ任
J=dct2(I);
v = flip(sort(abs(J(:))));
%1/4
c1 = v(length(v)/4);
[col row] = size(find(abs(J)< c1));
A=col*row;%��Ϊ0�ı任ϵ���ĸ���
%1/8
c2 = v(length(v)/8);
[col row]= size(find(abs(J)< c2));
B=col*row;%��Ϊ0�ı任ϵ���ĸ���
%1/16
c3 = v(length(v)/16);
[col row]= size(find(abs(J)< c3));
C=col*row;%��Ϊ0�ı任ϵ���ĸ���
%��С��1/4�����ֵ�任ϵ����Ϊ0������ɢ���ҷ��任
J(abs(J) < c1 ) = 0;I1=idct2(J);
%��С��1/8�����ֵ�任ϵ����Ϊ0������ɢ���ҷ��任
J(abs(J) < c2) = 0;I2=idct2(J);
%��С��1/16�����ֵ�任ϵ����Ϊ0������ɢ���ҷ��任
J(abs(J) < c3) = 0;I3=idct2(J);
%��ʾԭͼ�����任���
figure(2);
subplot(2,2,1);
imshow(I);
title('ԭͼ');
subplot(2,2,2);
imshow(I1,[0,255]);
title('С��1/4���ֵ');
subplot(2,2,3);
imshow(I2,[0,255]);
title('С��1/8���ֵ');
subplot(2,2,4);
imshow(I3,[0,255]);
title('С��1/16���ֵ');
%���㷴�ع�ʱ��DCT�ı任ϵ������0����С��5ʱ�ķ�ֵ����ȼ���Ϊ0�ı任ϵ���ĸ���
I = double(I);
I1 = double(I1);
[Row,Col] = size(I);
[Row,Col] = size(I1);
MSE1 = sum(sum((I-I1).^2))/(Row * Col);
PSNR1 = 10 * log10(255^2/MSE1);
fprintf('ͼ��ķ�ֵ����ȣ�PSNR1=%f\n',PSNR1);
%���㷴�ع�ʱ��DCT�ı任ϵ������0����С��10ʱ�ķ�ֵ����ȼ���Ϊ0�ı任ϵ���ĸ���
I = double(I);
I2 = double(I2);
[Row,Col] = size(I);
[Row,Col] = size(I2);
MSE2 = sum(sum((I-I2).^2))/(Row * Col);
PSNR2 = 10 * log10(255^2/MSE2);
fprintf('ͼ��ķ�ֵ����ȣ�PSNR2=%f\n',PSNR2);
%���㷴�ع�ʱ��DCT�ı任ϵ������0����С��20ʱ�ķ�ֵ����ȼ���Ϊ0�ı任ϵ���ĸ���
I = double(I);
I3 = double(I3);
[Row,Col] = size(I);
[Row,Col] = size(I3);
MSE3 = sum(sum((I-I3).^2))/(Row * Col);
PSNR3 = 10 * log10(255^2/MSE3);
fprintf('ͼ��ķ�ֵ����ȣ�PSNR3=%f\n',PSNR3);