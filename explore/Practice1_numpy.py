# %% 
import numpy as np

def print_obj(obj,name):
    print("%s:\n%s\n" % (name, obj))

def check_each(a,b):
    return (a ==b).astype('bool')

def check_mean(a,b):
    return np.mean(a == b).astype('bool')
# %%
# scalars, vectors, matrices
a = np.array(1)  # scalars
b = np.array([1.,2.,3.]) # vectors
c = np.array([[1.,2.,3.], [4.,5.,6.]]) # matrices

print_obj(a.ndim, "a.dim")
print_obj(b.ndim, "b.dim")
print_obj(c.ndim, "c.dim")

print(c.shape)
print(c.shape[0]) # 원하는 dim 선택가능 

# %%
# Tensors (N-dimensional arrays)
d = np.array([[[1., 2., 3.], 
               [4., 5., 6.]], 
              [[7., 8., 9.], 
              [10., 11., 12.]]])

e = np.array([[[[1., 2., 3.], [1., 2., 3.]], 
               [[4., 5., 6.], [4., 5., 6.]]],
              [[[7., 8., 9.], [7., 8., 9.]], 
               [[10., 11., 12.], [10., 11., 12.]]]])

print(d.ndim)  # 3
print(d.shape) # (2,2,3)
print(e.ndim)  # 4
print(e.shape) # (2,2,2,3)
# %%
# Quiz: What is the shape of [[[1], [2], [3]], [[4], [5], [6]]]?
quiz = np.array([[[1], [2], [3]], 
                 [[4], [5], [6]]])
quiz.shape # (2, 3, 1)

# %%
# Defining Numpy arrays
a = np.ones(10)
a = np.zeros((2,5))
a = np.full((2,5),5)  # array([[5, 5, 5, 5, 5],
                      #        [5, 5, 5, 5, 5]])
a = np.random.random((2,3,4)) # 0~1사이 안에 숫자를 random 으로 반환 
a = np.arange(10) # int 가 기본 defualt
a = np.arange(10).astype(float) # int -> 가 아닌 다른 datatype을 원할 경우 

# reshape 
a = np.arange(10).reshape((5,2))
a

# Quiz: Create a 4-by-3-by-2 tensor filled with 0.0 to 23.0
# 안에 값들을 random으로 해야되나 .?

for i in range(0,23):
    quiz2 = np.full((4,3,2),i)
print(quiz2.shape) # (4, 3, 2)


# %%
################# Indexing & Slicing #################
a = np.arange(10) # array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
print(a[0])
print(a[1])
print(a[-1]) # 9 , 뒤에서부터 진행할 때는 -1시작 
print(a[-3]) # 7

print(a[0:2])
print(a[0:])
print(a[:10])
print(a[:])
print(a[2:5])

# Quiz: What is a[-4:]? [6, 7, 8, 9]
# Quiz: What is a[:-8]?  0, 1

print(a[0:10:2])
print(a[::-1]) # 역순으로 -> [9 8 7 6 5 4 3 2 1 0]
print(a[8:5:-1]) # 뒤에서 한칸식 역순으로 -> [8 7 6]

# Quiz: Create [9, 6, 3] using a.
print(a[9:2:-3])  # a[-1:1:-3] 정답은 여러개라구 함


# %%
# Indexing a matrix 
a = np.arange(9).reshape((3,3))  
"""array([[0, 1, 2],
          [3, 4, 5],
          [6, 7, 8]])"""
print(a[0][0])
print(a[0][2])
print(a[2][0])

# Quiz: How to access the last row?
print(a[2])  # == 
print(a[2,:]) # 이것도 동일하다고 함, 근데 나는 주로 위에 사용 
# Quiz: How to access the second column?
print(a[1])
# 틀림!! Quiz: How to create [8, 5] using a? -> 나는 flatten 한 다음에 이제 concat해서 구할려고 했는데 dim0은 np.concat불가능
print(a[2:0:-1,2])


# %%
# Indexing and slicing a 3D tensor
a = np.arange(4*3*2).reshape((4, 3, 2))
a[2,1,0] # 사실 정확한 값을 보고 싶으면 a를 펼쳐서 확인해야됨 

# Quiz: What would be a[0]?   #[ 0,  1],
                              #[ 2,  3],
                              #[ 4,  5]]
# Quiz: What would be a[0, 1]?   
# Quiz: Create [[0, 2, 4], [6, 8, 10]] 
# 1) quiz3 = a[0:2, : ,:1] 이렇게 우선은 만들면 앞의 값들은 도출 가능, quiz.shape = (2,3,1) 
# 2) 만들어야 될 사이즈는 (2,3)
# 3) 그러니 quiz3.reshape(2,3) 하면 정답 
# 정답은 그냥 print(a[:2,:,0]) / 완전 간단..

# %%
# Conditional indexing
a = np.arange(3*2).reshape((3,2))

idx = a % 2 == 0  
idx # 원하는 조건을 줘서 true, false로 이제 값을 알려주는 것 
print(a[idx]) # 오직 true값만 반환을 해줌. array([0, 2, 4])

# Quiz: How would you create [3, 4, 5] using a?
"""idx_quiz = a > 2
a[idx_quiz]"""

# %%
# Taking specific elements from a vector
a = np.arange(10)
idx = [0, 2, 3]
a[idx]  # array([0, 2, 3]) -> 해당 값으로만 반환을 해줌 
 
# %%
# Taking specific elements from a tensor
a = np.arange(24).reshape((6,4))
a[:, [0,2,3]]
idx = ((0,0,1,5),(1,2,0,3)) # (0,1), (0,2), (1,0), (5,3) 에 해당하는 값을 튜플 형태로 가져옴
print(a[idx])

idx = np.array([[0,0,1,5],[1,2,0,3]]) # array 형태로 각각에 해당하는 행을 반환해줌 
print(a[idx])
print(a[idx].shape) # (2, 4, 4)




# %%
# Math Operations 
a = np.arange(6).reshape((3, 2))
b = np.ones((3, 2))
# shape이 동일하니 계산 가능 + - * / 

# Unary operations 
print(a.sum())
print(a.sum(axis=0)) # [6 9]
print(a.sum(axis=1)) # [1 5 9]

print(a.mean()) # 동일하게 axis에 대해서 계산 가능 ! 
print(a.max())  # normalization 을 진행할 때 활용 가능 
print(a.min())

# Quiz: Given a = np.arange(24).reshape((2,3,4)), what is the mean of the sum w.r.t to the last dimension?
a = np.arange(24).reshape((2,3,4))
a.sum(axis=2).mean()
# 정답 np.arange(24).reshape((2,3,4)).sum(axis=2).mean()

# %%
a = np.arange(3).astype('float')
b = np.ones(3)
np.dot(a,b) # 각원소 곱셈 

# Matrix dot product, matrix multiplication
a = np.arange(6).reshape((3, 2))
b = np.ones((2, 3))
print(np.dot(a,b))
print(a@b) # matrix multiplication operation @  == * 이라 생각하면됨 

# Tensor dot product, tensor multiplication
# tensor의 경우 제일 앞에 batch 부분을 제외하고 뒤에 부분에 대한 곱을 진행하는 것을 확인 가능 
a = np.arange(24).reshape((4, 3, 2))
b = np.ones((4, 2, 3))

print((a@b).shape) # (4, 3, 3), 
print(np.matmul(a,b).shape) # 동일한 tensor multiplication

# Quiz: what would happen if a.shape==(4,3,2) and b.shape==(2,3)?
a = np.arange(24).reshape((4, 3, 2))
b = np.ones((2,3))
print((a@b).shape) # (4, 3, 3)



# %%
############## Shape Manipulation #################
# Reshape
a = np.arange(24).reshape((2, 3, 4))
a.shape
b = a.reshape((6, 4))
b.shape
c = a.reshape((3,2,-1)) # == c = a.reshape((3,2,4))
c.shape # (3, 2, 4)

# Quiz: What would d=a.reshape((6, 4, -1)) look like?
#  X:6 4 4 -> 정답 (6,4,1) / -1 -> 1로 변경되는 부분 확인 가능 

# %%
# Adding an extra dimension
a = np.arange(3)
a[:, None].shape # (3, 1) , reshape 동일한 방법 
a.reshape(3,1).shape # (3, 1)  

# Quiz: How to make a = np.ones((3,4)) into shape (3, 1, 1, 4) using reshape and None?
a = np.ones((3,4)) 
a.reshape(3,1,1,4) # 이 방법이 훨 간단한듯 나는 
print(a[:, None, None].shape)


# %%
# Stack, concatenation
a = np.ones((3,2))
b = np.zeros((3,2))

print(np.vstack([a,b])) # 수직으로 concat 
print(np.hstack([a,b])) # 수평으로 concat 
print(np.hstack([a,b,a])) # 연속으로 더 추가해서 concat 가능 

print(np.concatenate([a,b], axis=0)) # 축 0에 대해 vstack이랑 결과는 동일 
print(np.concatenate([a,b], axis=1))


# Quiz: Would concatenating two tensors whose shapes are (4, 3, 2) and (5, 4, 2) on axis=2 work? NO
"""a = np.ones((4,3,2))
b = np.zeros((5,4,2))
quiz = np.concatenate([a,b], axis=2)""" 
# the axis that you want to concatenate along -> don't have to match, but!!다른 dimension들의 사이즈는 동일해야됨 

 
# Matrix transpose
a = np.arange(6).reshape((3, 2))
print(a)
a.T # .T function 바로 transpose 해주는 역할 

# %%
# Tensor transpose
a = np.arange(24).reshape((4, 3, 2))

b = np.transpose(a, [0, 2, 1])
print(b.shape) # (4, 2, 3) -> dimension을 변ㄱㅇ해주는 것 이건 많이 사용하니 꼭 기억해주나 

c = np.transpose(a, [1, 0, 2])
print(c.shape) # (3, 4, 2)




# %%
# Broadcasting - numpy가 연산 주에 다른 모양의 배열을 처리하는 방법 
# Vector and scalar
a = np.arange(3)
#b = 2. 
b = np.array([2.]) # shape이 어떻게 다른지에 따라서 이제 결과도 달라짐 
print(a+b)
print(a-b)
print(a*b)
print(a/b)


# Matrix and vector
a = np.arange(6).reshape((3,2))
b = np.arange(2).reshape(2) + 1
print(b.shape)
print(a+b)

# Tensor and matrix
a = np.arange(12).reshape((2,3,2))
b = np.arange(6).reshape((3,2))


#Quiz: How can we use None to do a+b?
print(a + b[None, :, :])
print(a + b.reshape(1,3,2))
#print(b.reshape(2,3,2)) -> cannot reshape array of size 6 into shape (2,3,2)

# print(a+b[:,:, None]) -> 이렇게 하면 앞에 batch size 차원이 안맞아서 broadcasting이 되지 않음 

# %%
#### Final Quiz #####
def sigmoid(x):
    return 1./(1. + np.exp(-x))

# Define a function that, given M of shape (m,n) and W of shape (4n, n), executes the following:
# - Take the first half rows of M
# - Take the second half rows of M
# - Take the odd-numbered rows of M
# - Take the even-numbered rows of M
# - Append them horizontally in the listed order so that you obtain a matrix X of shape (?, 4n)
# - Linearly transform X with W so that you obtain a matrix Y of shape (?, ?)
# - Put Y through the sigmoid function
# - Obtain the sum of the row-wise mean


"""근데 이 함수에서 만약 내가 m = np.arange(6).reshape((3,2)) 만들고 진행했을 경우 first half rows, second half rows로 나눌수가 있는지? 
이게 절반으로 딱 나누지를 못하니깐 가능한거지를 잘 모르겠음 .. 
"""

def foo(M, W):
    sh = M.shape
    a = M[:int(sh[0]/2)]
    b = M[int(sh[0]/2):]
    c = M[1:sh[0]:2]
    d = M[0:sh[0]:2]

    e = sigmoid(np.dot(np.concatenate([a, b, c, d], axis=1),W))
    f = e.mean(axis=0)

    return np.sum(f)

# %%
# 위에 함수 실행 X .. 
"""M = np.arange(6).reshape((3,2))
W = np.arange(16).reshape((8,2))

foo(M,W)"""