function output = vis_hybrid_image(hybrid_image)
%通过逐步向下采样图像并将所有图像连接在一起，可视化混合图像。

scales = 5; %5个下采样版本
scale_factor = 0.5; %每次下采样缩小2倍
padding = 5; %像素填充5

original_height = size(hybrid_image,1);
num_colors = size(hybrid_image,3); %计算图像通道数
output = hybrid_image;
cur_image = hybrid_image;

for i = 2:scales
    %填充部分
    output = cat(2, output, ones(original_height, padding, num_colors));
    
    %下采样图像
    cur_image = imresize(cur_image, scale_factor, 'bilinear');
    %合并图像
    tmp = cat(1,ones(original_height - size(cur_image,1), size(cur_image,2), num_colors), cur_image);
    output = cat(2, output, tmp);    
end


