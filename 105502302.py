import random
import numpy as np
import matplotlib.pyplot as plt
import sys
import math

#參數
learn_rate = 0.2
epoch = 100
accuracy = 0.9
number_of_hidden_layer = 1
number_of_hidden_layer_neuron = [2]
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
   
  
def paint(data, sol, status ):
     x = []
     y = []
     x2 = []
     y2 = []
     for i  in range(len(data)):
          for j in range(len(data[i])):
               
               if(sol[i] == 1):
                    x.append(data[i][1])
                    y.append(data[i][2])
               else:
                    x2.append(data[i][1])
                    y2.append(data[i][2])
     max_x = 0
     min_x = 0
     max_y = 0
     min_y = 0
     # 座標最大 最小
     if(len(x) != 0 and len(x2)!= 0):
          max_x = max([max(x), max(x2)])
          min_x = min([min(x), min(x2)])
     elif(len(x) != 0):
          max_x = max(x)
          min_x = min(x)
     elif(len(x2)!= 0):
          max_x = max(x2)
          min_x = min(x2)


     if(len(y) != 0 and len(y2)!= 0):
          max_y = max([max(y), max(y2)])
          min_y = min([min(y), min(y2)])
     elif(len(y) != 0):
          max_y = max(y)
          min_y = min(y)
     elif(len(y2)!= 0):
          max_y = max(y2)
          min_y = min(y2)
    
     plt.plot(x, y ,'r^')
     plt.plot(x2, y2 ,'gs')
     if(best_w[2] != 0):
          plt.plot([max_x, min_x], [(best_w[0] - best_w[1] * max_x)/best_w[2], (best_w[0] - best_w[1] * min_x)/best_w[2]] ,'y--')
     else:
          plt.plot([max_x, min_x], [max_y, min_y] ,'y--')
               

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
     y = []
     for i in range(number_of_hidden_layer+1):
          dimension = 0
          d_tmp_w = []
          if(i == 0):
               dimension = len(sol_class)
               for k in range(number_of_hidden_layer_neuron[i]):
                    tmp_w = [-1]
                    for j in range(dimension):
                         tmp_w.append(random.uniform(-1, 1))
                    d_tmp_w.append(tmp_w)
          else:
               dimension = number_of_hidden_layer_neuron[i-1]
               num = 0
               if(i == len(number_of_hidden_layer_neuron)):
                    num = len(sol_class)
               else:
                    num = number_of_hidden_layer_neuron[i]
               for k in range(num):
                    tmp_w = [-1]
                    for j in range(dimension):
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
     
     data_train = [[-1, 1, 1]]
     w = [[[-1.2, 1, 1], [0.3, 1, 1]], [[0.5, 0.4, 0.8], [0.5, 0.4, 0.8]]]
     
     #start train
     for i in range(epoch):
          #forwarding
          for item_x, item_y in zip(data_train, sol_train):
               for j in range(number_of_hidden_layer+1):
                    tmp_y = []
                    if(j == 0):
                         for k in range(number_of_hidden_layer_neuron[j]):
                              tmp = 0
                              
                              for z in range(len(sol_class)+1):
                                   #print(len(sol_class), len(w[j][k]))
                                   
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
               print(y)
               return
          #evalute
          count = 0
          for j in range(len(data_train)):
               sum = 0
               for k in range(len(data_train[j])):
                    sum = sum + data_train[j][k] * w[k]
               
               
               if(sum >= 0):
                    predict = sol_class[0]
               if(sum < 0):
                    predict = sol_class[1]
               if(predict == sol[j]):
                    count = count + 1
          print("epoch: " + str(i+1) + "\ntrain accuracy:", count/len(data_train))


                    
          count = 0
          for j in range(len(data_test)):
               sum = 0
               for k in range(len(data_test[j])):
                    sum = sum + data_test[j][k] * w[k]
               
               if(sum >= 0):
                    predict = sol_class[0]
               if(sum < 0):
                    predict = sol_class[1]
               if(predict == sol_test[j]):
                    count = count + 1
          ac = count/len(data_test)
          
          if(pre_ac > ac):
               w = pre_w
               ac = pre_ac
          else:
               pre_ac = ac
               pre_w = w
               
          #record best accuracy
          if(best_ac < ac):
               best_ac = ac
               best_w = w
          
          
  
          print("test accuracy:", str(ac))
          
          count = 0
          for j in range(len(data)):
               sum = 0
               for k in range(len(data[j])):
                    sum = sum + data[j][k] * w[k]
               
               
               if(sum >= 0):
                    predict = sol_class[0]
               if(sum < 0):
                    predict = sol_class[1]
               if(predict == sol[j]):
                    count = count + 1
               
          final_ac = count/len(data)
          print("total_accuracy", final_ac)
          if(best_ac >= accuracy):
               paint(data_train, sol_train, 0)
               return  

     if(len(sol_class) <= 2):     
          paint(data_train, sol_train, 0)
         
     '''
     if(len(sol_class) > 2):
          for i in range(len(sol_class)):
               w.append(random.random())
     else:
          w = [-1,1, 0]
        '''  
     
          
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

                
          if(data_input(path) != False):
               train()
               '''
               print("test data  accuracy: " + str(best_ac),)
               print("wieght ", end = "")
               print(best_w)

               print("all data accuracy: ", final_ac)
               if(len(sol_class) <= 2):     
                    paint(data, sol, 1)
               '''
          print("")
          
          status = int(input("input->1 continue, input->2 stop: "))
          if(status == 2):
               sys.exit()
