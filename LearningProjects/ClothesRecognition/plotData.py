import csv
import matplotlib.pyplot as plt

data = []

# with open("Plotting.csv", "r") as csvfile:
#     read = csv.reader(csvfile)
#     for i, line in enumerate(read):
#         print(line)
#         if not i%2:
#             newData = [100 * (5 * float(v) / 6) for v in line]
#             data.append(newData)
#         else:
#             data.append([int(i) for i in line])

with open("Plotting2.csv", "r") as csvfile:
    read = csv.reader(csvfile)
    for line in read:
        data.append([float(v) for v in line])

for line in data:
    print(line)

for x in range(3):
    plt.plot(data[2 * x + 1], data[2 * x], label=f"Model with {x+1} hidden layers")


plt.xlabel("Number of Epochs")
plt.ylabel("Accuracy of the model on test data (%)")
plt.legend()
plt.show()
