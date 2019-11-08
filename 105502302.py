import random
import numpy as np
import matplotlib.pyplot as plt
import sys
import math

#參數
learn_rate = 0.2
epoch = 100
accuracy = 0.9
number_of_hidden_layer = 0
number_of_hidden_layer_neuron = []
#input data
data = []
sol = []
tmp_sol = []
sol_class = []

#store final solution
best_ac = 0.0
best_w = []
final_ac = 0.0


def data_input(path):
     #read file
     file_c = False
     while(file_c == False):
          try:
               f = open(path)
               file_c = True          
          except:
               print("can't find file")
               return False
     
     for line in f:
          #repace '\n' to ''
          line = line.replace('\n', '')
          #split and string to int
          sp = line.split(" ")
          data_x = [-1]
          data_y = 0
          count = 0
          for item in sp:               
               if(count == (len(sp)-1)):
                    data_y = int(item)
               else:
                    data_x.append(float(item))
               count= count + 1

          
          data.append(data_x)
          tmp_sol.append(data_y)

          #紀錄class label
          if len(sol_class) == 0:
               sol_class.append(data_y)
          else:
               ch = True
               for item in sol_class:
                    if(item == data_y):
                         ch = False
                         break
               if(ch == True):
                    sol_class.append(data_y)

     
     for ans in tmp_sol:
          tmp = []
          for i in range(len(sol_class)):
               if(sol_class[i] != ans):
                    tmp.append(0)
               else:
                    tmp.append(1)
          sol.append(tmp)
    
     
     f.close()
   
  
def paint(data, sol, status):
     
     x = []
     y = []
     tmp_sol = []
     for i in range(len(data)):
          tmp_x = []
          tmp_y = []
          ch = True
          index = 0
          for j in range(len(tmp_sol)):    
               if(tmp_sol[j] == sol[i]):
                    ch = False
                    index = j 
                    break
          if(ch == True):
               tmp_sol.append(sol[i])
               
               x.append([data[i][1]])
               y.append([data[i][2]])
          else:
               x[index].append(data[i][1])
               y[index].append(data[i][2])
     max_x = max(x[0])
     min_x = min(x[0])
     max_y = max(y[0])
     min_y = min(y[0])
     for item in x:
         
          if(max_x < max(item)):
               max_x = max(item)
          if(min_x > min(item)):
               min_x = min(item)

     for item in y:
          if(max_y < max(item)):
               max_y = max(item)

          if(min_y > min(item)):
               min_y = min(item)
               
          
     color = ['b^', 'g^', 'r^', 'c^', 'm^' , 'y^', 'k^']
     for i in range(len(x)):
          plt.plot(x[i], y[i] ,color[i])
     
     #plt.xlim(min_x, max_x)
     #plt.ylim(min_y, max_y)
          #存檔的path
     path_spilt = path.split('\\')
     path_spilt = path.split('/')
     name = (path_spilt[len(path_spilt)-1].split('.'))[0]
     try:
          if(status == 0):
               plt.savefig('dataset/image/' + name + '_train_data.jpg')
          else:
               plt.savefig('dataset/image/' + name + '_all_data.jpg')
     except:
          print("--------*******************************************  -----")
          print("--------   can't find path, path is dataset/image/   -----")
          print("--------*******************************************  -----")
     plt.show()


def data_split():
     data_train = []
     sol_train = [] 
     data_test = []
     sol_test = []
     #use random() 
     for j in range(len(data)):
          if(random.random() >= 0.33):
               data_train.append(data[j])
               sol_train.append(sol[j])
          else:
               data_test.append(data[j])
               sol_test.append(sol[j])
               
          #防止data_test沒有資料                     
          if(len(data_test) == 0):
               ran = random.randint(0, len(data_train)-1)
               data_test.append(data_train[ran])
               sol_test.append(sol[j])
               del data_train[ran]
               del sol_train[ran]

     return data_train, sol_train, data_test, sol_test


     
def train():
     global best_ac
     global best_w 
     global accuracy
     global final_ac
     
     #初始weight
     w = []
     output = []
     
     for i in range(number_of_hidden_layer+1):
          dimension = 0
          d_tmp_w = []
          if(i == 0):
               dimension = len(data[0])-1
               for k in range(number_of_hidden_layer_neuron[i]):
                    tmp_w = [-1]
                    for j in range(dimension):
                         tmp_w.append(random.uniform(-1, 1))
                    d_tmp_w.append(tmp_w)
          elif(i == number_of_hidden_layer):
               
               for k in range(len(sol_class)):
                    tmp_w = [-1]
                    for j in range(number_of_hidden_layer_neuron[i-1]):
                         tmp_w.append(random.uniform(-1, 1))
                    d_tmp_w.append(tmp_w)
          else:
               for k in range(number_of_hidden_layer_neuron[i]):
                    print(i, number_of_hidden_layer_neuron[i])
                    tmp_w = [-1]
                    for j in range(number_of_hidden_layer_neuron[i-1]):
                         tmp_w.append(random.uniform(-1, 1))
                    d_tmp_w.append(tmp_w)
          w.append(d_tmp_w)
          
     data_train = []
     data_test = []
     sol_train = []
     sol_test = []
     predict = 0

     pre_ac = 0
     pre_w = []
     
     data_train, sol_train, data_test, sol_test = data_split()
    
     
     print(w)
     #start train
     for i in range(epoch):
          #forwarding
          
          for item_x, item_y in zip(data_train, sol_train):
               
               y = []
               change_weight = []
               for j in range(number_of_hidden_layer+1):
                    tmp_y = []
                    if(j == 0):
                         for k in range(number_of_hidden_layer_neuron[j]):
                              tmp = 0
                              
                              for z in range(len(item_x)):     
                                   tmp = tmp + item_x[z] * w[j][k][z]
                              tmp_y.append(1/(1+math.exp(-tmp)))
                         y.append(tmp_y)
                    elif(j == number_of_hidden_layer):
                         for k in range(len(sol_class)):
                              tmp = 0
                              for z in range(number_of_hidden_layer_neuron[j-1]+1):
                                   if(z == 0):
                                        tmp = tmp + -1 * w[j][k][z]
                                   else:
                                        tmp = tmp + y[j-1][z-1] * w[j][k][z]
                                        
                              tmp_y.append(1/(1+math.exp(-tmp)))
                         y.append(tmp_y)
                    else:
                         for k in range(number_of_hidden_layer_neuron[j]):
                              tmp = 0
                              for z in range(number_of_hidden_layer_neuron[j-1]+1):
                                   if(z == 0):
                                        tmp = tmp + -1 * w[j][k][z]
                                   else:
                                        tmp = tmp + y[j-1][z-1] * w[j][k][z]
                              tmp_y.append(1/(1+math.exp(-tmp)))
                         y.append(tmp_y)
               #backward
               count = -2
               for j in (reversed (range(number_of_hidden_layer+1))):
                    tmp_change_weight = []
                    count = count + 1 
                    if(j == number_of_hidden_layer):
                         for k in range(len(sol_class)):
                              tmp = 0
                              tmp = (item_y[k] - y[j][k]) * y[j][k] * (1 - y[j][k])
                              tmp_change_weight.append(tmp)
                         change_weight.append(tmp_change_weight)
                         
                    elif(j == number_of_hidden_layer - 1):
                         for k in range(number_of_hidden_layer_neuron[j]):
                              tmp = y[j][k] * (1 - y[j][k])
                              sigma = 0
                              for z in range(len(sol_class)):
                                   
                                   sigma = sigma + w[j+1][z][k+1] * change_weight[count][z] 
                              tmp = tmp * sigma
                              tmp_change_weight.append(tmp)
                         change_weight.append(tmp_change_weight)
                    else:
                         for k in range(number_of_hidden_layer_neuron[j]):
                              tmp = y[j][k] * (1 - y[j][k])
                              sigma = 0
                              for z in range(number_of_hidden_layer_neuron[j+1]):
                                   sigma = sigma + w[j+1][z][k+1] * change_weight[count][z]
                              tmp = tmp * sigma
                              tmp_change_weight.append(tmp)
                         change_weight.append(tmp_change_weight)
               
               
               #change wight
               change_weight = list(reversed(change_weight))
               for j in range(number_of_hidden_layer+1):
                    if(j == 0):     
                         for k in range(number_of_hidden_layer_neuron[j]):
                              for z in range(len(item_x)): 
                                   w[j][k][z] = w[j][k][z] + (learn_rate)/(1+i/10) * change_weight[j][k] * item_x[z]
                    elif(j == number_of_hidden_layer):
                         for k in range(len(sol_class)):
                              for z in range(number_of_hidden_layer_neuron[j-1]+1):
                                   if(z == 0): 
                                        w[j][k][z] = w[j][k][z] + (learn_rate)/(1+i/10) * change_weight[j][k] * -1
                                   else:
                                        w[j][k][z] = w[j][k][z] + (learn_rate)/(1+i/10) * change_weight[j][k] * y[j-1][z-1]
                    else:
                         for k in range(number_of_hidden_layer_neuron[j]):
                              for z in range(number_of_hidden_layer_neuron[j-1]+1):
                                   if(z == 0): 
                                        w[j][k][z] = w[j][k][z] + (learn_rate)/(1+i/10) * change_weight[j][k] * -1
                                   else:
                                        w[j][k][z] = w[j][k][z] + (learn_rate)/(1+i/10)* change_weight[j][k] * y[j-1][z-1]
                    
          #evaluate
          count = 0
          rmse_train = 0
          for item_x, item_y in zip(data_train, sol_train):
               y = []
               predict = []
               for j in range(number_of_hidden_layer+1):
                    tmp_y = []
                    if(j == 0):
                         for k in range(number_of_hidden_layer_neuron[j]):
                              tmp = 0
                              for z in range(len(item_x)):     
                                   tmp = tmp + item_x[z] * w[j][k][z]
                              tmp_y.append(1/(1+math.exp(-tmp)))
                         y.append(tmp_y)
                    elif(j == number_of_hidden_layer):
                         for k in range(len(sol_class)):
                              tmp = 0
                              for z in range(number_of_hidden_layer_neuron[j-1]+1):
                                   if(z == 0):
                                        tmp = tmp + -1 * w[j][k][z]
                                   else:
                                        tmp = tmp + y[j-1][z-1] * w[j][k][z]
                                        
                              tmp_y.append(1/(1+math.exp(-tmp)))
                         y.append(tmp_y)
                         predict = tmp_y
                    else:
                         for k in range(number_of_hidden_layer_neuron[j]):
                              tmp = 0
                              for z in range(number_of_hidden_layer_neuron[j-1]+1):
                                   if(z == 0):
                                        tmp = tmp + -1 * w[j][k][z]
                                   else:
                                        tmp = tmp + y[j-1][z-1] * w[j][k][z]
                              tmp_y.append(1/(1+math.exp(-tmp)))
                         y.append(tmp_y)
               sol_index = 0
               predict_index = 0
               predict_content = 0
               sol_index = 0
               sol_content = 0
               for k in range(len(item_y)):
                    if(sol_content < item_y[k]):
                         sol_index = k
                         sol_content = item_y[k]
               for k in range(len(predict)):
                    if(predict_content < predict[k]):
                         predict_index = k
                         predict_content = predict[k]

               for k in range(len(predict)):
                    rmse_train = rmse_train + (predict[k] - item_y[k]) * (predict[k] - item_y[k])
                    
               if(predict_index == sol_index):
                    count = count + 1
               

          print("train -> accuracy:" , count/len(data_train), end = "")
          print(", RMSE", math.sqrt(rmse_train / len(data_train)))
          
          count = 0
          rmse_test = 0
          for item_x, item_y in zip(data_test, sol_test):
               y = []
               predict = []
               for j in range(number_of_hidden_layer+1):
                    tmp_y = []
                    if(j == 0):
                         for k in range(number_of_hidden_layer_neuron[j]):
                              tmp = 0
                              for z in range(len(item_x)):     
                                   tmp = tmp + item_x[z] * w[j][k][z]
                              tmp_y.append(1/(1+math.exp(-tmp)))
                         y.append(tmp_y)
                    elif(j == number_of_hidden_layer):
                         for k in range(len(sol_class)):
                              tmp = 0
                              for z in range(number_of_hidden_layer_neuron[j-1]+1):
                                   if(z == 0):
                                        tmp = tmp + -1 * w[j][k][z]
                                   else:
                                        tmp = tmp + y[j-1][z-1] * w[j][k][z]
                                        
                              tmp_y.append(1/(1+math.exp(-tmp)))
                         y.append(tmp_y)
                         predict = tmp_y
                    else:
                         for k in range(number_of_hidden_layer_neuron[j]):
                              tmp = 0
                              for z in range(number_of_hidden_layer_neuron[j-1]+1):
                                   if(z == 0):
                                        tmp = tmp + -1 * w[j][k][z]
                                   else:
                                        tmp = tmp + y[j-1][z-1] * w[j][k][z]
                              tmp_y.append(1/(1+math.exp(-tmp)))
                         y.append(tmp_y)
               sol_index = 0
               predict_index = 0
               predict_content = 0
               sol_index = 0
               sol_content = 0
               for k in range(len(item_y)):
                    if(sol_content < item_y[k]):
                         sol_index = k
                         sol_content = item_y[k]
               for k in range(len(predict)):
                    if(predict_content < predict[k]):
                         predict_index = k
                         predict_content = predict[k]
               for k in range(len(predict)):
                    rmse_test = rmse_test + (predict[k] - item_y[k]) * (predict[k] - item_y[k])
               if(predict_index == sol_index):
                    count = count + 1
          print("test accuracy" , count/len(data_test), end = " ")
          print(", RMSE", math.sqrt(rmse_test / len(data_test)))
        
               
                    
               
                         
                              
               
          
if __name__ == '__main__':
     while(True):
          data = []
          sol = []
          sol_class = []
          best_ac = 0.0
          best_w = []
          final_ac = 0.0
          path = str(input("檔案路徑: "))
          c1 = False
          c2 = False
          c3 = False
          c4 = False
          c5 = False
          while(True):
               if(c1 == False):
                    try:
                         learn_rate = float(input("type:float learning_rate: "))
                         break
                    except:
                         print("learning_rate error")
          while(True):
               if(c2 == False):
                    try:
                         epoch = int(input("type:int epoch: "))
                         break
                    except:
                         print("epoch error")
          while(True):
               if(c3 == False):
                    try:
                         accuracy = float(input("type:float accuracy: "))
                         break
                    except:
                         print("accuracy error")

          while(True):
               if(c4 == False):
                    try:
                         number_of_hidden_layer = int(input("type:int how many hidden layer are: "))
                         break
                    except:
                         print("hidden layer error")
          while(True):
               if(c5 == False):
                    try:
                         for i in range(number_of_hidden_layer):
                              print("type:int how many neuron are in ", end = "")
                              print(i+1, end = "")
                              print(" hidden layer", end = ": ")
                              t = int(input())
                              number_of_hidden_layer_neuron.append(t)      
                         break 
                    except:
                         print("neuron error")
          if(data_input(path) != False):
               train()
               print(len(data[0]))
               if(len(data[0]) <= 3):
                    
                    paint(data, sol, 1)
               
          print("")
          
          status = int(input("input->1 continue, input->2 stop: "))
          if(status == 2):
               sys.exit()
