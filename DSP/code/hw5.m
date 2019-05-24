function [nn_1, nn_2, nn_3, test_x] = hw5()
load mnist_uint8; %����mnist���ݼ�

train_x = double(train_x) / 255; %��һ��
test_x  = double(test_x)  / 255; %��һ��
train_y = double(train_y);
test_y  = double(test_y);

%% �����������(dbn) + ������(nn)
rand('state',0)
%ѵ��dpn
dbn.sizes = [100 100]; %784->100->100->784
opts.numepochs =   1; %dpn epochs
opts.batchsize = 100;  % dpn batchsize
opts.momentum  =   0; % sgd no momentum
opts.alpha     =   1;  %dpn����
dbn = dbnsetup(dbn, train_x, opts); % ��ʼ��dpn
dbn = dbntrain(dbn, train_x, opts); %ѵ��dpn

%��������������������������
nn_1 = dbnunfoldtonn(dbn, 10);
nn_1.learningRate = 1;   %nn ѧϰ��
nn_1.activation_function = 'sigm'; %nn�����
nn_1.output  = 'softmax'; %���softmax
nn_1.weightPenaltyL2 = 1e-4; %Ȩ�ض������ͷ�
opts.numepochs =  3; %nn epochs
opts.batchsize = 100; %nn batchsize
% opts.plot = 1;  %ʹ�ܻ�ͼ
%ѵ��nn
nn_1 = nntrain(nn_1, train_x, train_y, opts);
%����nn
[er1, bad1] = nntest(nn_1, test_x, test_y);%����
fprintf('dpn+nn test error: %f\n', er1);
%% ������(nn)
rand('state',0)
nn_2 = nnsetup([784 100 10]); %784->100->10
nn_2.learningRate = 1;   %nn ѧϰ��
nn_2.activation_function = 'sigm';%nn�����
nn_2.output  = 'softmax'; %���softmax
nn_2.weightPenaltyL2 = 1e-4; %Ȩ�ض������ͷ�
opts.numepochs =  3; %nn epochs
opts.batchsize = 100; %nn batchsize
% opts.plot = 1; %ʹ�ܻ�ͼ
%ѵ��nn
[nn_2, L] = nntrain(nn_2, train_x, train_y, opts);
%����nn
[er2, bad2] = nntest(nn_2, test_x, test_y);
fprintf('nn2 test error: %f\n', er2);

%% ��ջʽ�Ա�����(sae)+������(nn)
rand('state',0)
sae = saesetup([784 100]); %784->100->100->784
sae.ae{1}.activation_function  = 'sigm'; %�����
sae.ae{1}.learningRate  = 1;  %ѧϰ��
sae.ae{1}.inputZeroMaskedFraction   = 0.5;
opts.numepochs =   3;%epochs
opts.batchsize = 100;%batchsize
%ѵ��sae
sae = saetrain(sae, train_x, opts);
%����nn�ṹ
nn_3 = nnsetup([784 100 10]); %784->100->10
nn_3.activation_function    = 'sigm';%�����
nn_3.learningRate  = 1;%ѧϰ��
nn_3.output  = 'softmax'; %���softmax
nn_3.weightPenaltyL2 = 1e-4;%Ȩ�ض������ͷ�
nn_3.W{1} = sae.ae{1}.W{1};%��ʼ��nnȨ��
opts.numepochs =   3;%nn epochs
opts.batchsize = 100;%nn batchsize
% opts.plot = 1; %ʹ�ܻ�ͼ
%ѵ��nn
nn_3 = nntrain(nn_3, train_x, train_y, opts);
%����nn
[er3, bad3] = nntest(nn_3, test_x, test_y);
fprintf('sae+nn test error: %f\n', er3);
end