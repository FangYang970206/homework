function output = my_imfilter(image, filter)

intput_image = image;

% 获取输入图像的行和列大小，并过滤以允许多尺寸图像
[intput_row, intput_col] = size(intput_image(:,:,1));
[filter_row, filter_col] = size(filter);

% 用零填充图像(数量=过滤器的最小需求=行和列的一半)
pad_input_image = padarray(intput_image, [(filter_row - 1)/2, (filter_col - 1)/2]);

output = [];

for layer = 1:size(intput_image, 3) % 当输入是灰色图像时，确保正常
    % 使输入图像的所有filter_row*filter_col大小块都是列
    columns = im2col(pad_input_image(:,:,layer), [filter_row, filter_col]);
    
    % 转置滤波器，使其卷积(但不相关)
    filter2 = transpose(filter(:));
    
    % 过滤图像
    filterd_columns = filter2 * columns;
    
    % 从列恢复到图像形式s
    output(:,:,layer) = col2im(filterd_columns, [1, 1], [intput_row, intput_col]);
end




