function output = my_imfilter(image, filter)
% This function is intended to behave like the built in function imfilter()
% See 'help imfilter' or 'help conv2'. While terms like "filtering" and
% "convolution" might be used interchangeably, and they are indeed nearly
% the same thing, there is a difference:
% from 'help filter2'
%    2-D correlation is related to 2-D convolution by a 180 degree rotation
%    of the filter matrix.

% Your function should work for color images. Simply filter each color
% channel independently.

% Your function should work for filters of any width and height
% combination, as long as the width and height are odd (e.g. 1, 7, 9). This
% restriction makes it unambigious which pixel in the filter is the center
% pixel.

% Boundary handling can be tricky. The filter can't be centered on pixels
% at the image boundary without parts of the filter being out of bounds. If
% you look at 'help conv2' and 'help imfilter' you see that they have
% several options to deal with boundaries. You should simply recreate the
% default behavior of imfilter -- pad the input image with zeros, and
% return a filtered image which matches the input resolution. A better
% approach is to mirror the image content over the boundaries for padding.

% % Uncomment if you want to simply call imfilter so you can see the desired
% % behavior. When you write your actual solution, you can't use imfilter,
% % filter2, conv2, etc. Simply loop over all the pixels and do the actual
% % computation. It might be slow.
% output = imfilter(image, filter);


%%%%%%%%%%%%%%%%
% Your code here
%%%%%%%%%%%%%%%%
intput_image = image;

% Get the row & column size of input image and filter in order to admit
% multi size picture
[intput_row, intput_col] = size(intput_image(:,:,1));
[filter_row, filter_col] = size(filter);

% Pad image with zeros (amount = minimum need of filter = half of row and
% column
pad_input_image = padarray(intput_image, [(filter_row - 1)/2, (filter_col - 1)/2]);

output = [];

for layer = 1:size(intput_image, 3) % ensure to be OK when input is gray image
    % make all filter_row*filter_col size patch of input image be columns
    columns = im2col(pad_input_image(:,:,layer), [filter_row, filter_col]);
    
    % transpose the filter in order to make it convolution (but not correlation)
    filter2 = transpose(filter(:));
    
    % filter the image
    filterd_columns = filter2 * columns;
    
    % recover from columns to image form
    output(:,:,layer) = col2im(filterd_columns, [1, 1], [intput_row, intput_col]);
end




