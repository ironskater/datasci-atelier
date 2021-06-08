from matplotlib import pyplot as plt
import subprocess
import pandas as pd
import numpy as np
import os.path
import math
import csv

plot_loss_vs_iter_flag = True
print_each_prediction_flag = True
print_all_sampling_data = True

def download_data():
	file_name = "data.tar"

	if not os.path.exists(file_name):
		subprocess.run(["wget", "-O", "data.tar", "https://drive.google.com/uc?id=1wNKAxQ29G15kgpBy_asjTcZRRgmsCZRm"])
	else:
		print(file_name, " is already existed")

	if not os.path.exists("test.csv") or not os.path.exists("train.csv"):
		subprocess.run(["unzip", "-O", "big5", file_name])
	else:
		print("test.csv and train.csv are already existed")

def assign_plot_xy(array, learning_rate):
	x_iter = array[:, 0].reshape(1, -1).flatten()
	y_loss = array[:, 1].reshape(1, -1).flatten()
	plt.plot(x_iter, y_loss, label = "learning rate: " + str(learning_rate))

def adagrad_algorithm(dim, training_set, expected_pm25_10th, iter_times, learning_rate, out):
	w = np.zeros([dim, 1])
	adagrad = np.zeros([dim, 1])
	eps = 0.0000000001
	plot = np.empty([1, 2], dtype = float)

	for t in range(iter_times):
		loss = np.sqrt(np.sum(np.power(np.dot(training_set, w) - expected_pm25_10th, 2))/471/12)#rmse

		if(t == 0):
			plot[0, 0] = t
			plot[0, 1] = loss
		elif(t % 10 == 0):
			# print(str(t) + ":" + str(loss))
			plot = np.append(plot, [[t, loss]], axis = 0)

		gradient = 2 * np.dot(training_set.transpose(), np.dot(training_set, w) - expected_pm25_10th) #dim*1
		adagrad += gradient ** 2
		w = w - learning_rate * gradient / np.sqrt(adagrad + eps)

	if(out != ""):
		np.save(out, w)

	return plot

download_data()

data = pd.read_csv('./train.csv', encoding = 'big5')

# ===== Preprocessing =====
data = data.iloc[:, 3:] # 忽略前3項：日期，測站，測項, 只保留每小時資料
data[data == 'NR'] = 0 # 將資料為`NR`的部份轉成0
raw_data = data.to_numpy() # 將DataFrame轉成NumPy

# ===== Extract Features =====
month_data = {}
for month in range(12):
	sample = np.empty([18, 480]) # 18(features) x 480(每個月20天，共24*20 = 480小時)
	for day in range(20):
		sample[:, day * 24 : (day + 1) * 24] = raw_data[18 * (20 * month + day) : 18 * (20 * month + day + 1), :]
	month_data[month] = sample

# ===== Extract Features2 =====
# 將month_data轉成(472*12) x (9*18)矩陣
all_sampling_data = np.empty([471 * 12, 9 * 18], dtype = float)
all_sampling_pm25_10th = np.empty([471 * 12, 1], dtype = float)
for month in range(12):
	for day in range(20):
		for hour in range(24):

			if day == 19 and hour > 14:
				continue

			all_sampling_data[month * 471 + day * 24 + hour, :] = month_data[month][:, day * 24 + hour : day * 24 + hour + 9].reshape(1, -1)
			all_sampling_pm25_10th[month * 471 + day * 24 + hour, 0] = month_data[month][9, day * 24 + hour + 9]

if(print_all_sampling_data == True):
	print("all_sampling_data: \n", all_sampling_data, "\nshape: ", all_sampling_data.shape, "\n=== End ===\n")

# ===== Normalize =====
mean_x = np.mean(all_sampling_data, axis = 0) # 9 * 18
std_x = np.std(all_sampling_data, axis = 0) # 9 * 18
for ix in range(len(all_sampling_data)): # 471 * 12
	for jx in range(len(all_sampling_data[0])): # 9 * 18
		if std_x[jx] != 0:
			all_sampling_data[ix][jx] = (all_sampling_data[ix][jx] - mean_x[jx]) / std_x[jx]

# ===== Split Training Data Into "train_set" and "validation_set" =====
x_train_set = all_sampling_data[: math.floor(len(all_sampling_data) * 0.8), :]
y_train_set = all_sampling_pm25_10th[: math.floor(len(all_sampling_pm25_10th) * 0.8), :]
x_validation = all_sampling_data[math.floor(len(all_sampling_data) * 0.8): , :]
y_validation = all_sampling_pm25_10th[math.floor(len(all_sampling_pm25_10th) * 0.8): , :]
print(x_train_set)
print(y_train_set)
print(x_validation)
print(y_validation)
print(len(x_train_set))
print(len(y_train_set))
print(len(x_validation))
print(len(y_validation))

# ===== training =====
x_train_set = np.concatenate((np.ones([len(x_train_set), 1]), x_train_set), axis = 1).astype(float) # 加上常數項
iter_time = 1000
learning_rate1 = 10
learning_rate2 = 10
learning_rate3 = 1
learning_rate4 = 0.1
dim_of_param = 9 * 18 + 1
file_of_param = "weight9h.npy"

plot1 = adagrad_algorithm(dim_of_param, x_train_set, y_train_set, iter_time, learning_rate1, file_of_param)
# plot2 = adagrad_algorithm(x_train_set, y_train_set, iter_time, learning_rate2, '')
plot3 = adagrad_algorithm(dim_of_param, x_train_set, y_train_set, iter_time, learning_rate3, '')
# plot4 = adagrad_algorithm(x_train_set, y_train_set, iter_time, learning_rate4, '')

if(plot_loss_vs_iter_flag == True):
	assign_plot_xy(plot1, learning_rate1)
	# assign_plot_xy(plot2, learning_rate2)
	assign_plot_xy(plot3, learning_rate3)
	# assign_plot_xy(plot4, learning_rate4)
	plt.xlabel('iter times')
	plt.ylabel('loss')
	plt.legend()
	plt.show()

# ===== Testing =====
print("===== Testing =====")
print("x_validation before concatenate: \n", x_validation, "\n shape: ", x_validation.shape)
x_validation = np.concatenate((np.ones([len(x_validation), 1]), x_validation), axis = 1).astype(float)
print("x_validation after concatenate: \n", x_validation, "\n shape: ", x_validation.shape)

# ==== Prediction =====
w = np.load(file_of_param)

result_index = np.arange(0, len(y_validation))
predict_result = np.dot(x_validation, w)
actual_result = y_validation
err = np.sqrt(np.sum(np.power(predict_result.flatten() - actual_result.flatten(), 2))/len(y_validation))

print("predict result: \n", predict_result.flatten(), "\n")
print("actual result: \n", actual_result.flatten(), "\n")

plt.text(500, 90, "Err: " + str(err))
plt.plot(result_index, actual_result.flatten(), label = "actual result")
plt.plot(result_index, predict_result.flatten(), label = "predict result")
plt.legend()
plt.title("9h")
plt.xlabel("index")
plt.ylabel("10th pm2.5 value")
plt.show()

# ===== Save Prediction to CSV File =====
with open('submit_9h.csv', mode = 'w', newline = '') as submit_file:
	csv_writer = csv.writer(submit_file)
	header = ['id', 'value']
	csv_writer.writerow(header)

	for ix in range(len(x_validation)):
		row = ['id_' + str(ix), predict_result[ix][0]]
		csv_writer.writerow(row)