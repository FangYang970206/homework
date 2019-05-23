%% 二分类
clc;
close all;
clear all;
% 每个类别数量为10
N = 10;
%随机初始化两个类的数据
data1 = rand(2, 10);
data2 = rand(2, 10);
%设置两个类的label
data1_label = zeros(1,10);
data2_label = ones(1,10);
%将两个类的数据分开
data1(1,:) = data1(1,:) - 0.5;
data1(2,:) = data1(2,:) + 0.5;
data2(1,:) = data2(1,:) + 0.5;
data2(2,:) = data2(2,:) - 0.5;
%组合data和label
data = [data1, data2];
label = [data1_label, data2_label];
% 定义感知器神经元并对其初始化 
net=newp([0 1;0 1],1);
net.initFcn='initlay';
net.layers{1}.initFcn='initwb';
net.inputWeights{1,1}.initFcn='rands';
net.layerWeights{1,1}.initFcn='rands';
net.biases{1}.initFcn='rands';
net=init(net);
% 训练感知器神经元
net=train(net,data,label);
cell2mat(net.iw);
cell2mat(net.b);
%绘制结果
figure();
scatter(data1(1,:), data1(2,:));
hold on;
scatter(data2(1,:), data2(2,:));
plotpc(net.iw{1,1},net.b{1});

%% 回归
clc;
clear all;
% 产生训练样本与测试样本
x1= 0:0.5:4*pi;
x2= 0:0.12:4*pi;
% P1 = (x1.^2-2*x1).*exp(-x1.^2-x2.^2-x1.*x2); 
P1 = 0.12*exp(-0.23*x1) + 0.54*exp(-0.17*x1).*sin(1.23*x1);% 训练样本
T1 = P1; % 训练目标
P2 = 0.12*exp(-0.23*x2) + 0.54*exp(-0.17*x2).*sin(1.23*x2);% 测试样本
T2 = P2; % 测试目标
% 归一化
[PN1,minp,maxp,TN1,mint,maxt] = premnmx(P1,T1);
PN2 = tramnmx(P2,minp,maxp);
TN2 = tramnmx(T2,mint,maxt);
% 设置网络参数
HideNum=1;   % 隐层层数
NodeNum = 5; % 隐层节点数 
TypeNum = 1; % 输出维数
TF1 = 'tansig';TF2 = 'purelin'; % 判别函数(缺省值)
net = newff(minmax(PN1),[NodeNum TypeNum],{TF1 TF2});
net.trainFcn = 'trainlm'; 
net.trainParam.show = 20; % 训练显示间隔
net.trainParam.lr = 0.3; % 学习步长 - traingd,traingdm
net.trainParam.mc = 0.95; % 动量项系数 - traingdm,traingdx
net.trainParam.mem_reduc = 1; % 分块计算Hessian矩阵net.trainParam.epochs = 1000; % 最大训练次数
net.trainParam.goal = 1e-4; % 最小均方误差
net.trainParam.min_grad = 1e-20; % 最小梯度
net.trainParam.time = inf; % 最大训练时间
net = train(net,PN1,TN1); % 训练
YN1 = sim(net,PN1); % 训练样本实际输出
YN2 = sim(net,PN2); % 测试样本实际输出
MSE1 = mean((TN1-YN1).^2) % 训练均方误差
MSE2 = mean((TN2-YN2).^2) % 测试均方误差
Y2 = postmnmx(YN2,mint,maxt); % 反归一化
% 结果作图
plot(1:length(T2),T2,'r',1:length(Y2),Y2,'b');
fprintf('测试均方误差为：%f\n', MSE2);
legend('测试集','训练集'); 
title('期望输出与实际输出对比');

