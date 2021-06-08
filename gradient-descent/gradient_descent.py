import numpy as np

'''
先使用known_w, known_b參數來設計一個線性方程式
再假設我們不知道此方程式的參數, 使用gradient descent來找出線性方程式的未知參數
'''

known_w = 23.987
known_b = 779.43

unknown_w = -4 # initial w
unknown_b = -120 # initial b

lr = 0.1 * 10000
epoch = 1000
w_grad_current = 0.0
b_grad_current = 0.0

def calculate_mse(x_data, y_data, w, b):
	err_sum = 0.0
	mse = 0.0
	for n in range(len(x_data)):
		err_sum = err_sum + (y_data[n] - w * x_data[n] - b)**2

	mse = err_sum/len(x_data)

	if mse < 0.000000000001:
		mse = 0

	return mse

def f(x, w, b):
	return w * x + b

x_data = np.arange(start = 0, stop = 5, step = 1)
y_data = f(x_data, known_w, known_b)

# for i in range(epoch):

# 	w_grad = 0.0
# 	b_grad = 0.0

# 	'''
# 	此迴圈用來計算Loss function分別對unknown_w,unknown_b的偏微分
# 	L(w, b) = sigma[(yi - wxi - b)^2]
# 	'''
# 	for ix in range(len(x_data)):
# 		w_grad = w_grad -2.0 * (y_data[ix] - unknown_b - unknown_w * x_data[ix]) * x_data[ix]
# 		b_grad = b_grad -2.0 * (y_data[ix] - unknown_b - unknown_w * x_data[ix]) * 1.0

# 	# Update parameters.
# 	unknown_w = unknown_w - lr * w_grad
# 	unknown_b = unknown_b - lr * b_grad

# 	if (i%100 == 0):
# 		print("epoch[{}], mse[{}]:".format(i, calculate_mse(x_data, y_data, unknown_w, unknown_b)))

w_gradient_sum = 0.0
b_gradient_sum = 0.0
eps = 0.0000000001

for i in range(epoch):

	w_grad_current = 0.0
	b_grad_current = 0.0

	for ix in range(len(x_data)):
		w_grad_current = w_grad_current -2.0 * (y_data[ix] - unknown_b - unknown_w * x_data[ix]) * x_data[ix]
		b_grad_current = b_grad_current -2.0 * (y_data[ix] - unknown_b - unknown_w * x_data[ix]) * 1.0

	w_gradient_sum = w_gradient_sum + w_grad_current**2
	b_gradient_sum = b_gradient_sum + b_grad_current**2

	# Update parameters.
	unknown_w = unknown_w - lr/np.sqrt(w_gradient_sum + eps) * w_grad_current
	unknown_b = unknown_b - lr/np.sqrt(b_gradient_sum + eps) * b_grad_current

	if (i%100 == 0):
		print("epoch[{}], mse[{}]:".format(i, calculate_mse(x_data, y_data, unknown_w, unknown_b)))

print(f'w:{unknown_w}')
print(f'b:{unknown_b}')