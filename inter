#BACKWARD
def u_cal(u, n):
    temp = u
    for i in range(n):
        temp = temp * (u + i)
    return temp
 
def fact(n):
    f = 1
    for i in range(2, n + 1):
        f *= i
    return f

n = 5
x = [1891, 1901, 1911, 1921, 1931]

y = [[0.0 for _ in range(n)] for __ in range(n)]

y[0][0] = 46
y[1][0] = 66
y[2][0] = 81
y[3][0] = 93
y[4][0] = 101
 
for i in range(1, n):
    for j in range(n - 1, i - 1, -1):
        y[j][i] = y[j][i - 1] - y[j - 1][i - 1]
 
 
for i in range(n):
    for j in range(i + 1):
        print(y[i][j], end="\t")
    print()
 
value = 1925
 
sum = y[n - 1][0]
u = (value - x[n - 1]) / (x[1] - x[0])
for i in range(1, n):
    sum = sum + (u_cal(u, i) * y[n - 1][i]) / fact(i)
 
print("\n Value at", value,  "is", sum)

#FORWARD
def u_cal(u, n):

    temp = u;
    for i in range(1, n):
        temp = temp * (u - i);
    return temp;

def fact(n):
    f = 1;
    for i in range(2, n + 1):
        f *= i;
    return f;

n = 4;
x = [ 45, 50, 55, 60 ];

y = [[0 for i in range(n)]
        for j in range(n)];
y[0][0] = 0.7071;
y[1][0] = 0.7660;
y[2][0] = 0.8192;
y[3][0] = 0.8660;

for i in range(1, n):
    for j in range(n - i):
        y[j][i] = y[j + 1][i - 1] - y[j][i - 1];

for i in range(n):
    print(x[i], end = "\t");
    for j in range(n - i):
        print(y[i][j], end = "\t");
    print("");

value = 52;

sum = y[0][0];
u = (value - x[0]) / (x[1] - x[0]);
for i in range(1,n):
    sum = sum + (u_cal(u, i) * y[0][i]) / fact(i);

print("\nValue at", value,
    "is", round(sum, 6));

# CUBIC SPLINE

from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
import math as m
import operator as op
from functools import reduce
import scipy.stats as stats
from statistics import mean

t = [0, 8, 16, 24, 32, 40]
o = [14.621, 11.843, 9.870, 8.418, 7.305, 6.413]

cs = CubicSpline(t, o, bc_type = "natural")

new_t = [4, 36]
pred_o = []

for i in new_t:
  pred_o.append(cs(i))

pred_o

plt.figure(figsize = (10,8))
plt.plot(new_t, pred_o, 'ro')
plt.plot(t, o, 'green', marker = "o")
plt.title('Cubic Spline Interpolation')
plt.xlabel('x')
plt.ylabel('y')
plt.show()


#CENTRAL

import math

def Stirling(x, fx, x1, n):

    y1 = 0; N1 = 1; d = 1;
    N2 = 1; d2 = 1;
    temp1 = 1; temp2 = 1;
    k = 1; l = 1;
    delta = [[0 for i in range(n)]
                for j in range(n)];

    h = x[1] - x[0];
    s = math.floor(n / 2);
    a = x[s];
    u = (x1 - a) / h;


    for i in range(n - 1):
        delta[i][0] = fx[i + 1] - fx[i];
    for i in range(1, n - 1):
        for j in range(n - i - 1):
            delta[j][i] = (delta[j + 1][i - 1] -
                        delta[j][i - 1]);

    y1 = fx[s];

    for i in range(1, n):
        if (i % 2 != 0):
            if (k != 2):
                temp1 *= (pow(u, k) - pow((k - 1), 2));
            else:
                temp1 *= (pow(u, 2) - pow((k - 1), 2));
            k += 1;
            d *= i;
            s = math.floor((n - i) / 2);
            y1 += (temp1 / (2 * d)) * (delta[s][i - 1] +
                                    delta[s - 1][i - 1]);
        else:
            temp2 *= (pow(u, 2) - pow((l - 1), 2));
            l += 1;
            d *= i;
            s = math.floor((n - i) / 2);
            y1 += (temp2 / (d)) * (delta[s][i - 1]);

    print(round(y1, 3));

n = 5;
x = [0, 0.5, 1.0, 1.5, 2.0 ];
fx = [ 0, 0.191, 0.341, 0.433, 0.477];

x1 = 1.22;
Stirling(x, fx, x1, n);


#NEWTON DIVIDED DIFFERNECE FORMULA
def proterm(i, value, x):
    pro = 1;
    for j in range(i):
        pro = pro * (value - x[j]);
    return pro;


def dividedDiffTable(x, y, n):

    for i in range(1, n):
        for j in range(n - i):
            y[j][i] = ((y[j][i - 1] - y[j + 1][i - 1]) /
                                    (x[j] - x[i + j]));
    return y;


def applyFormula(value, x, y, n):

    sum = y[0][0];

    for i in range(1, n):
        sum = sum + (proterm(i, value, x) * y[0][i]);
    
    return sum;


def printDiffTable(y, n):

    for i in range(n):
        for j in range(n - i):
            print(round(y[i][j], 4), "\t",
                            end = " ");

        print("");


n = 4;
y = [[0 for i in range(10)]
        for j in range(10)];
x = [ 5, 6, 9, 11 ];


y[0][0] = 12;
y[1][0] = 13;
y[2][0] = 14;
y[3][0] = 16;

y=dividedDiffTable(x, y, n);

printDiffTable(y, n);

value = 7;

print("\nValue at", value, "is",
        round(applyFormula(value, x, y, n), 2))

#LAGRANGE INTERPOLATION

class Data:
    def __init__(self, x, y):
        self.x = x
        self.y = y


def interpolate(f: list, xi: int, n: int) -> float:

    result = 0.0
    for i in range(n):

        term = f[i].y
        for j in range(n):
            if j != i:
                term = term * (xi - f[j].x) / (f[i].x - f[j].x)

        result += term

    return result

if __name__ == "__main__":

    f = [Data(0, 2), Data(1, 3), Data(2, 12), Data(5, 147)]


    print("Value of f(3) is :", interpolate(f, 3, 4))


