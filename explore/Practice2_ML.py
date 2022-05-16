# %% 
import numpy as np
from matplotlib import pyplot as plt
import sklearn

# %%
# Matplotlib example
plt.plot([1,2,3], [3,2,1])
#plt.scatter([1,2,3], [3,2,1])
plt.plot([1,2,3,6], [3,2,1,6])
plt.show()


# %%
def draw_plot(X,Y, x_sample, y_sample):
    for i in range(len(X)):
        plt.plot(X[i],Y[i])
    plt.scatter(x_sample, y_sample)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.axhline(0, color='black')
    plt.axvline(0, color='black')
    plt.show()

# %%
# Drawing a function 
foo = lambda x: -(2/7*x**3-9/2*x**2+15*x-10.)
x_line = np.linspace(0, 10,100)

# Quiz: Draw the function foo using x_line
y_line = foo(x_line)
plt.plot(x_line, y_line)

draw_plot([x_line], [y_line], x_line, y_line)
draw_plot([x_line, y_line], [y_line, x_line], x_line, y_line)

# Quiz: Sample 5 points of foo in the domain [0, 10] and visualize with draw_plot
x_sample = np.linspace(0, 10, 5)
y_sample = foo(x_sample)
draw_plot([x_line], [y_line], x_sample, y_sample)

# Quiz: Sample 5 points of foo in the domain [0, 10] with Gaussian noise where mu=0, sigma=0.1 and visualize.
# noise = np.random.normal(0,1)
# x_sample = np.linspace(0, 10, 5) + noise
# y_sample = foo(x_sample) + noise
# draw_plot([x_line], [y_line], x_sample, y_sample)

num_points = 5
x_sample = np.linspace(0, 10, num_points)
np.random.seed(seed=0)
y_sample = foo(x_sample) + np.random.normal(0, 5, num_points)
draw_plot([x_line], [y_line], x_sample, y_sample)




# %%
#### Linear Regression #### 
"""간단히 생각해서 선을 가지고 데이터를 놓고 그것을 가장 잘 설명할 수 있는 선을 찾는 분석을 하는 방법"""
from sklearn.linear_model import LinearRegression

# Define model 
lr = LinearRegression()
# train model - dataset은 multi dimensional array 상태여야 가능 
lr.fit(x_sample[: ,None], y_sample)
# evalutate model - R squared 
r2 = lr.score(x_sample[: ,None], y_sample)
# predict model 
y_hat = lr.predict(x_sample[[0] ,None])
print(x_sample[0])
print(y_hat) # [7.16793716] -> noise 땜에 그런지 결과가 완전 꽝

"""mean squared erro 평균 제곱 오차 
-> 오차의 제곱에 평균을 취한 것 , 적을 수록 원본과의 오차가 적은 것이니 정확성이 높다 판단 가능 """

# Quiz: Calculate Mean Squared Error using x_sample and y_sample and lr.predict()
y_hat = lr.predict(x_sample[:, None])
print("MSE:%f" % ((y_sample - y_hat)**2).mean())



# %%
#### Polynomial Regression ####
"""다항 회귀: 분석하고자 하는 데이터가 선형적인 관계가 아닌 곡선의 형태로 되어 있는 경우 linear model을 활용하면 오차가 크게 남 
그래서 이러한 경우 데이터 분포가 2차원 곡선 형태로 되어 있으면 2차원 곡선으로, 3차원이면 3차원으로 접근을 해야지 오차가 적음 """
from sklearn.preprocessing import PolynomialFeatures

# Defining a polynomial feature transfromer
poly = PolynomialFeatures(degree=2)
# Transform the original features to polynomial features
x_sample_poly = poly.fit_transform(x_sample[:,None])
# Train a linear regression model using the polynomial features
lr = LinearRegression().fit(x_sample_poly, y_sample)



# %%

# Train Test Split - 내가 torch로 진행했던 방식보다는 훨씬 코드가 간편하긴 함 (근데 나는 np.array 데이터의 경우는 내 방식으로 진행)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

# Classifiers 
from sklearn.linear_model import LogisticRegression
logistic = LogisticRegression(random_state=1234)
logistic.fit(X_train[:, :2], y_train)
# %%
