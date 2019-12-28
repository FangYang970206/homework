close all; 

%读取图像
image1 = im2single(imread('../data/einstein.bmp'));
image2 = im2single(imread('../data/marilyn.bmp'));

%设置截断频率
cutoff_frequency = 12; 

%设立高斯低通滤波器
filter = fspecial('Gaussian', cutoff_frequency*4+1, cutoff_frequency);

%进行高斯低通滤波，得到低通分量
low_frequencies = my_imfilter(my_imfilter(image1, filter), filter');

%减去低频得到高频分量
high_frequencies = image2 - my_imfilter(my_imfilter(image2, filter), filter');

%结合低频和高频分量
hybrid_image = low_frequencies + high_frequencies;

%可视化并保存结果
figure(1); imshow(low_frequencies)
figure(2); imshow(high_frequencies + 0.5);
vis = vis_hybrid_image(hybrid_image);
figure(3); imshow(vis);
imwrite(low_frequencies, 'low_frequencies.jpg', 'quality', 95);
imwrite(high_frequencies + 0.5, 'high_frequencies.jpg', 'quality', 95);
imwrite(hybrid_image, 'hybrid_image.jpg', 'quality', 95);
imwrite(vis, 'hybrid_image_scales.jpg', 'quality', 95);