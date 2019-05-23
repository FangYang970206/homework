%% 线性方程组的解
A = [6,3,4; -2,5,7; 8,-4,-3];
B = [3; -4; -7];
x = A\B;
%x1,x2,x3分别为0.6000,7.0000,-5.4000

%% 等比数列求和
%-----------for循环-----------%
m = sqrt(3);
sum = 0;
for i =1:106
    sum = sum + 1/(2^i);
end
sum1 = m*sum

%----构建等比数列求和公式----%
a1 = sqrt(3)/2;
q = 1/2;
n = 106;
sum2 = a1*(1 - q^n)/(1 - q);

%% 同一坐标系绘制曲线
x = linspace(0, pi, 200);
y1 = sin(x);
y2 = sin(x).*sin(10*x);
y3 = -cos(x);
figure();
plot(x, y1, 'r');
hold on;
plot(x, y2, 'g');
hold on;
plot(x, y3, 'b');
xlabel('x');
ylabel('y');
legend('y1', 'y2', 'y3');