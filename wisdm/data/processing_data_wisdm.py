import numpy as np
from pandas import read_csv
from numpy import dstack
import h5py
import csv 
from keras.utils.np_utils import to_categorical
from sklearn import preprocessing
	
# WISDM
path =r'F:\PHD\GAN\code.12.2020\PGAN1\RL_acc\WISDM_data.csv'

segement_time_size = 180
sensors = 3

mvts = ["Downstairs","Jogging","Sitting","Standing","Upstairs","Walking"]
id = [i for i in range(1,37)]
def biggest_multiple(multiple_of, input_number):
    return input_number - input_number % multiple_of

x_d_1= []
y_d_1= []
p_d_1= []
x_d_2= []
y_d_2= []
p_d_2= []
x_d_3= []
y_d_3= []
p_d_3= []
x_d_4= []
y_d_4= []
p_d_4= []
x_d_5= []
y_d_5= []
p_d_5= []
x_d_6= []
y_d_6= []
p_d_6= []
x_d_7= []
y_d_7= []
p_d_7= []
x_d_8= []
y_d_8= []
p_d_8= []
x_d_9= []
y_d_9= []
p_d_9= []
x_d_10= []
y_d_10= []
p_d_10= []
x_d_11= []
y_d_11= []
p_d_11= []
x_d_12= []
y_d_12= []
p_d_12= []
x_d_13= []
y_d_13= []
p_d_13= []
x_d_14= []
y_d_14= []
p_d_14= []
x_d_15= []
y_d_15= []
p_d_15= []
x_d_16= []
y_d_16= []
p_d_16= []
x_d_17= []
y_d_17= []
p_d_17= []
x_d_18= []
y_d_18= []
p_d_18= []
x_d_19= []
y_d_19= []
p_d_19= []
x_d_20= []
y_d_20= []
p_d_20= []
x_d_21= []
y_d_21= []
p_d_21= []
x_d_22= []
y_d_22= []
p_d_22= []
x_d_23= []
y_d_23= []
p_d_23= []
x_d_24= []
y_d_24= []
p_d_24= []
x_d_25= []
y_d_25= []
p_d_25= []
x_d_26= []
y_d_26= []
p_d_26= []
x_d_27= []
y_d_27= []
p_d_27= []
x_d_28= []
y_d_28= []
p_d_28= []
x_d_29= []
y_d_29= []
p_d_29= []
x_d_30= []
y_d_30= []
p_d_30= []
x_d_31= []
y_d_31= []
p_d_31= []
x_d_32= []
y_d_32= []
p_d_32= []
x_d_33= []
y_d_33= []
p_d_33= []
x_d_34= []
y_d_34= []
p_d_34= []
x_d_35= []
y_d_35= []
p_d_35= []
x_d_36= []
y_d_36= []
p_d_36= []
x_j_1= []
y_j_1= []
p_j_1= []
x_j_2= []
y_j_2= []
p_j_2= []
x_j_3= []
y_j_3= []
p_j_3= []
x_j_4= []
y_j_4= []
p_j_4= []
x_j_5= []
y_j_5= []
p_j_5= []
x_j_6= []
y_j_6= []
p_j_6= []
x_j_7= []
y_j_7= []
p_j_7= []
x_j_8= []
y_j_8= []
p_j_8= []
x_j_9= []
y_j_9= []
p_j_9= []
x_j_10= []
y_j_10= []
p_j_10= []
x_j_11= []
y_j_11= []
p_j_11= []
x_j_12= []
y_j_12= []
p_j_12= []
x_j_13= []
y_j_13= []
p_j_13= []
x_j_14= []
y_j_14= []
p_j_14= []
x_j_15= []
y_j_15= []
p_j_15= []
x_j_16= []
y_j_16= []
p_j_16= []
x_j_17= []
y_j_17= []
p_j_17= []
x_j_18= []
y_j_18= []
p_j_18= []
x_j_19= []
y_j_19= []
p_j_19= []
x_j_20= []
y_j_20= []
p_j_20= []
x_j_21= []
y_j_21= []
p_j_21= []
x_j_22= []
y_j_22= []
p_j_22= []
x_j_23= []
y_j_23= []
p_j_23= []
x_j_24= []
y_j_24= []
p_j_24= []
x_j_25= []
y_j_25= []
p_j_25= []
x_j_26= []
y_j_26= []
p_j_26= []
x_j_27= []
y_j_27= []
p_j_27= []
x_j_28= []
y_j_28= []
p_j_28= []
x_j_29= []
y_j_29= []
p_j_29= []
x_j_30= []
y_j_30= []
p_j_30= []
x_j_31= []
y_j_31= []
p_j_31= []
x_j_32= []
y_j_32= []
p_j_32= []
x_j_33= []
y_j_33= []
p_j_33= []
x_j_34= []
y_j_34= []
p_j_34= []
x_j_35= []
y_j_35= []
p_j_35= []
x_j_36= []
y_j_36= []
p_j_36= []
x_s_1= []
y_s_1= []
p_s_1= []
x_s_2= []
y_s_2= []
p_s_2= []
x_s_3= []
y_s_3= []
p_s_3= []
x_s_4= []
y_s_4= []
p_s_4= []
x_s_5= []
y_s_5= []
p_s_5= []
x_s_6= []
y_s_6= []
p_s_6= []
x_s_7= []
y_s_7= []
p_s_7= []
x_s_8= []
y_s_8= []
p_s_8= []
x_s_9= []
y_s_9= []
p_s_9= []
x_s_10= []
y_s_10= []
p_s_10= []
x_s_11= []
y_s_11= []
p_s_11= []
x_s_12= []
y_s_12= []
p_s_12= []
x_s_13= []
y_s_13= []
p_s_13= []
x_s_14= []
y_s_14= []
p_s_14= []
x_s_15= []
y_s_15= []
p_s_15= []
x_s_16= []
y_s_16= []
p_s_16= []
x_s_17= []
y_s_17= []
p_s_17= []
x_s_18= []
y_s_18= []
p_s_18= []
x_s_19= []
y_s_19= []
p_s_19= []
x_s_20= []
y_s_20= []
p_s_20= []
x_s_21= []
y_s_21= []
p_s_21= []
x_s_22= []
y_s_22= []
p_s_22= []
x_s_23= []
y_s_23= []
p_s_23= []
x_s_24= []
y_s_24= []
p_s_24= []
x_s_25= []
y_s_25= []
p_s_25= []
x_s_26= []
y_s_26= []
p_s_26= []
x_s_27= []
y_s_27= []
p_s_27= []
x_s_28= []
y_s_28= []
p_s_28= []
x_s_29= []
y_s_29= []
p_s_29= []
x_s_30= []
y_s_30= []
p_s_30= []
x_s_31= []
y_s_31= []
p_s_31= []
x_s_32= []
y_s_32= []
p_s_32= []
x_s_33= []
y_s_33= []
p_s_33= []
x_s_34= []
y_s_34= []
p_s_34= []
x_s_35= []
y_s_35= []
p_s_35= []
x_s_36= []
y_s_36= []
p_s_36= []
x_st_1= []
y_st_1= []
p_st_1= []
x_st_2= []
y_st_2= []
p_st_2= []
x_st_3= []
y_st_3= []
p_st_3= []
x_st_4= []
y_st_4= []
p_st_4= []
x_st_5= []
y_st_5= []
p_st_5= []
x_st_6= []
y_st_6= []
p_st_6= []
x_st_7= []
y_st_7= []
p_st_7= []
x_st_8= []
y_st_8= []
p_st_8= []
x_st_9= []
y_st_9= []
p_st_9= []
x_st_10= []
y_st_10= []
p_st_10= []
x_st_11= []
y_st_11= []
p_st_11= []
x_st_12= []
y_st_12= []
p_st_12= []
x_st_13= []
y_st_13= []
p_st_13= []
x_st_14= []
y_st_14= []
p_st_14= []
x_st_15= []
y_st_15= []
p_st_15= []
x_st_16= []
y_st_16= []
p_st_16= []
x_st_17= []
y_st_17= []
p_st_17= []
x_st_18= []
y_st_18= []
p_st_18= []
x_st_19= []
y_st_19= []
p_st_19= []
x_st_20= []
y_st_20= []
p_st_20= []
x_st_21= []
y_st_21= []
p_st_21= []
x_st_22= []
y_st_22= []
p_st_22= []
x_st_23= []
y_st_23= []
p_st_23= []
x_st_24= []
y_st_24= []
p_st_24= []
x_st_25= []
y_st_25= []
p_st_25= []
x_st_26= []
y_st_26= []
p_st_26= []
x_st_27= []
y_st_27= []
p_st_27= []
x_st_28= []
y_st_28= []
p_st_28= []
x_st_29= []
y_st_29= []
p_st_29= []
x_st_30= []
y_st_30= []
p_st_30= []
x_st_31= []
y_st_31= []
p_st_31= []
x_st_32= []
y_st_32= []
p_st_32= []
x_st_33= []
y_st_33= []
p_st_33= []
x_st_34= []
y_st_34= []
p_st_34= []
x_st_35= []
y_st_35= []
p_st_35= []
x_st_36= []
y_st_36= []
p_st_36= []
x_u_1= []
y_u_1= []
p_u_1= []
x_u_2= []
y_u_2= []
p_u_2= []
x_u_3= []
y_u_3= []
p_u_3= []
x_u_4= []
y_u_4= []
p_u_4= []
x_u_5= []
y_u_5= []
p_u_5= []
x_u_6= []
y_u_6= []
p_u_6= []
x_u_7= []
y_u_7= []
p_u_7= []
x_u_8= []
y_u_8= []
p_u_8= []
x_u_9= []
y_u_9= []
p_u_9= []
x_u_10= []
y_u_10= []
p_u_10= []
x_u_11= []
y_u_11= []
p_u_11= []
x_u_12= []
y_u_12= []
p_u_12= []
x_u_13= []
y_u_13= []
p_u_13= []
x_u_14= []
y_u_14= []
p_u_14= []
x_u_15= []
y_u_15= []
p_u_15= []
x_u_16= []
y_u_16= []
p_u_16= []
x_u_17= []
y_u_17= []
p_u_17= []
x_u_18= []
y_u_18= []
p_u_18= []
x_u_19= []
y_u_19= []
p_u_19= []
x_u_20= []
y_u_20= []
p_u_20= []
x_u_21= []
y_u_21= []
p_u_21= []
x_u_22= []
y_u_22= []
p_u_22= []
x_u_23= []
y_u_23= []
p_u_23= []
x_u_24= []
y_u_24= []
p_u_24= []
x_u_25= []
y_u_25= []
p_u_25= []
x_u_26= []
y_u_26= []
p_u_26= []
x_u_27= []
y_u_27= []
p_u_27= []
x_u_28= []
y_u_28= []
p_u_28= []
x_u_29= []
y_u_29= []
p_u_29= []
x_u_30= []
y_u_30= []
p_u_30= []
x_u_31= []
y_u_31= []
p_u_31= []
x_u_32= []
y_u_32= []
p_u_32= []
x_u_33= []
y_u_33= []
p_u_33= []
x_u_34= []
y_u_34= []
p_u_34= []
x_u_35= []
y_u_35= []
p_u_35= []
x_u_36= []
y_u_36= []
p_u_36= []
x_w_1= []
y_w_1= []
p_w_1= []
x_w_2= []
y_w_2= []
p_w_2= []
x_w_3= []
y_w_3= []
p_w_3= []
x_w_4= []
y_w_4= []
p_w_4= []
x_w_5= []
y_w_5= []
p_w_5= []
x_w_6= []
y_w_6= []
p_w_6= []
x_w_7= []
y_w_7= []
p_w_7= []
x_w_8= []
y_w_8= []
p_w_8= []
x_w_9= []
y_w_9= []
p_w_9= []
x_w_10= []
y_w_10= []
p_w_10= []
x_w_11= []
y_w_11= []
p_w_11= []
x_w_12= []
y_w_12= []
p_w_12= []
x_w_13= []
y_w_13= []
p_w_13= []
x_w_14= []
y_w_14= []
p_w_14= []
x_w_15= []
y_w_15= []
p_w_15= []
x_w_16= []
y_w_16= []
p_w_16= []
x_w_17= []
y_w_17= []
p_w_17= []
x_w_18= []
y_w_18= []
p_w_18= []
x_w_19= []
y_w_19= []
p_w_19= []
x_w_20= []
y_w_20= []
p_w_20= []
x_w_21= []
y_w_21= []
p_w_21= []
x_w_22= []
y_w_22= []
p_w_22= []
x_w_23= []
y_w_23= []
p_w_23= []
x_w_24= []
y_w_24= []
p_w_24= []
x_w_25= []
y_w_25= []
p_w_25= []
x_w_26= []
y_w_26= []
p_w_26= []
x_w_27= []
y_w_27= []
p_w_27= []
x_w_28= []
y_w_28= []
p_w_28= []
x_w_29= []
y_w_29= []
p_w_29= []
x_w_30= []
y_w_30= []
p_w_30= []
x_w_31= []
y_w_31= []
p_w_31= []
x_w_32= []
y_w_32= []
p_w_32= []
x_w_33= []
y_w_33= []
p_w_33= []
x_w_34= []
y_w_34= []
p_w_34= []
x_w_35= []
y_w_35= []
p_w_35= []
x_w_36= []
y_w_36= []
p_w_36= []


csvfile= open(path,'r')
readf =  csv.reader(csvfile)
for row in readf:
        if row:
            if row[2]!=0:
                if row[1]=='Downstairs':
                        if row[0] == '1':
                                x1=row[3]
                                x_d_1.append(x1)
                                x2=row[4]
                                x_d_1.append(x2)
                                x3= row[5]
                                x_d_1.append(x3)
                                y_d_1.append(mvts.index(row[1]))
                                p_d_1.append(id.index(int(row[0])))
                        if row[0] == '2':
                                x1=row[3]
                                x_d_2.append(x1)
                                x2=row[4]
                                x_d_2.append(x2)
                                x3= row[5]
                                x_d_2.append(x3)
                                y_d_2.append(mvts.index(row[1]))
                                p_d_2.append(id.index(int(row[0])))
                        if row[0] == '3':
                                x1=row[3]
                                x_d_3.append(x1)
                                x2=row[4]
                                x_d_3.append(x2)
                                x3= row[5]
                                x_d_3.append(x3)
                                y_d_3.append(mvts.index(row[1]))
                                p_d_3.append(id.index(int(row[0])))
                        if row[0] == '4':
                                x1=row[3]
                                x_d_4.append(x1)
                                x2=row[4]
                                x_d_4.append(x2)
                                x3= row[5]
                                x_d_4.append(x3)
                                y_d_4.append(mvts.index(row[1]))
                                p_d_4.append(id.index(int(row[0])))
                        if row[0] == '5':
                                x1=row[3]
                                x_d_5.append(x1)
                                x2=row[4]
                                x_d_5.append(x2)
                                x3= row[5]
                                x_d_5.append(x3)
                                y_d_5.append(mvts.index(row[1]))
                                p_d_5.append(id.index(int(row[0])))
                        if row[0] == '6':
                                x1=row[3]
                                x_d_6.append(x1)
                                x2=row[4]
                                x_d_6.append(x2)
                                x3= row[5]
                                x_d_6.append(x3)
                                y_d_6.append(mvts.index(row[1]))
                                p_d_6.append(id.index(int(row[0])))
                        if row[0] == '7':
                                x1=row[3]
                                x_d_7.append(x1)
                                x2=row[4]
                                x_d_7.append(x2)
                                x3= row[5]
                                x_d_7.append(x3)
                                y_d_7.append(mvts.index(row[1]))
                                p_d_7.append(id.index(int(row[0])))
                        if row[0] == '8':
                                x1=row[3]
                                x_d_8.append(x1)
                                x2=row[4]
                                x_d_8.append(x2)
                                x3= row[5]
                                x_d_8.append(x3)
                                y_d_8.append(mvts.index(row[1]))
                                p_d_8.append(id.index(int(row[0])))
                        if row[0] == '9':
                                x1=row[3]
                                x_d_9.append(x1)
                                x2=row[4]
                                x_d_9.append(x2)
                                x3= row[5]
                                x_d_9.append(x3)
                                y_d_9.append(mvts.index(row[1]))
                                p_d_9.append(id.index(int(row[0])))
                        if row[0] == '10':
                                x1=row[3]
                                x_d_10.append(x1)
                                x2=row[4]
                                x_d_10.append(x2)
                                x3= row[5]
                                x_d_10.append(x3)
                                y_d_10.append(mvts.index(row[1]))
                                p_d_10.append(id.index(int(row[0])))
                        if row[0] == '11':
                                x1=row[3]
                                x_d_11.append(x1)
                                x2=row[4]
                                x_d_11.append(x2)
                                x3= row[5]
                                x_d_11.append(x3)
                                y_d_11.append(mvts.index(row[1]))
                                p_d_11.append(id.index(int(row[0])))
                        if row[0] == '12':
                                x1=row[3]
                                x_d_12.append(x1)
                                x2=row[4]
                                x_d_12.append(x2)
                                x3= row[5]
                                x_d_12.append(x3)
                                y_d_12.append(mvts.index(row[1]))
                                p_d_12.append(id.index(int(row[0])))
                        if row[0] == '13':
                                x1=row[3]
                                x_d_13.append(x1)
                                x2=row[4]
                                x_d_13.append(x2)
                                x3= row[5]
                                x_d_13.append(x3)
                                y_d_13.append(mvts.index(row[1]))
                                p_d_13.append(id.index(int(row[0])))
                        if row[0] == '14':
                                x1=row[3]
                                x_d_14.append(x1)
                                x2=row[4]
                                x_d_14.append(x2)
                                x3= row[5]
                                x_d_14.append(x3)
                                y_d_14.append(mvts.index(row[1]))
                                p_d_14.append(id.index(int(row[0])))
                        if row[0] == '15':
                                x1=row[3]
                                x_d_15.append(x1)
                                x2=row[4]
                                x_d_15.append(x2)
                                x3= row[5]
                                x_d_15.append(x3)
                                y_d_15.append(mvts.index(row[1]))
                                p_d_15.append(id.index(int(row[0])))
                        if row[0] == '16':
                                x1=row[3]
                                x_d_16.append(x1)
                                x2=row[4]
                                x_d_16.append(x2)
                                x3= row[5]
                                x_d_16.append(x3)
                                y_d_16.append(mvts.index(row[1]))
                                p_d_16.append(id.index(int(row[0])))
                        if row[0] == '17':
                                x1=row[3]
                                x_d_17.append(x1)
                                x2=row[4]
                                x_d_17.append(x2)
                                x3= row[5]
                                x_d_17.append(x3)
                                y_d_17.append(mvts.index(row[1]))
                                p_d_17.append(id.index(int(row[0])))
                        if row[0] == '18':
                                x1=row[3]
                                x_d_18.append(x1)
                                x2=row[4]
                                x_d_18.append(x2)
                                x3= row[5]
                                x_d_18.append(x3)
                                y_d_18.append(mvts.index(row[1]))
                                p_d_18.append(id.index(int(row[0])))
                        if row[0] == '19':
                                x1=row[3]
                                x_d_19.append(x1)
                                x2=row[4]
                                x_d_19.append(x2)
                                x3= row[5]
                                x_d_19.append(x3)
                                y_d_19.append(mvts.index(row[1]))
                                p_d_19.append(id.index(int(row[0])))
                        if row[0] == '20':
                                x1=row[3]
                                x_d_20.append(x1)
                                x2=row[4]
                                x_d_20.append(x2)
                                x3= row[5]
                                x_d_20.append(x3)
                                y_d_20.append(mvts.index(row[1]))
                                p_d_20.append(id.index(int(row[0])))
                        if row[0] == '21':
                                x1=row[3]
                                x_d_21.append(x1)
                                x2=row[4]
                                x_d_21.append(x2)
                                x3= row[5]
                                x_d_21.append(x3)
                                y_d_21.append(mvts.index(row[1]))
                                p_d_21.append(id.index(int(row[0])))
                        if row[0] == '22':
                                x1=row[3]
                                x_d_22.append(x1)
                                x2=row[4]
                                x_d_22.append(x2)
                                x3= row[5]
                                x_d_22.append(x3)
                                y_d_22.append(mvts.index(row[1]))
                                p_d_22.append(id.index(int(row[0])))
                        if row[0] == '23':
                                x1=row[3]
                                x_d_23.append(x1)
                                x2=row[4]
                                x_d_23.append(x2)
                                x3= row[5]
                                x_d_23.append(x3)
                                y_d_23.append(mvts.index(row[1]))
                                p_d_23.append(id.index(int(row[0])))
                        if row[0] == '24':
                                x1=row[3]
                                x_d_24.append(x1)
                                x2=row[4]
                                x_d_24.append(x2)
                                x3= row[5]
                                x_d_24.append(x3)
                                y_d_24.append(mvts.index(row[1]))
                                p_d_24.append(id.index(int(row[0])))
                        if row[0] == '25':
                                x1=row[3]
                                x_d_25.append(x1)
                                x2=row[4]
                                x_d_25.append(x2)
                                x3= row[5]
                                x_d_25.append(x3)
                                y_d_25.append(mvts.index(row[1]))
                                p_d_25.append(id.index(int(row[0])))
                        if row[0] == '26':
                                x1=row[3]
                                x_d_26.append(x1)
                                x2=row[4]
                                x_d_26.append(x2)
                                x3= row[5]
                                x_d_26.append(x3)
                                y_d_26.append(mvts.index(row[1]))
                                p_d_26.append(id.index(int(row[0])))
                        if row[0] == '27':
                                x1=row[3]
                                x_d_27.append(x1)
                                x2=row[4]
                                x_d_27.append(x2)
                                x3= row[5]
                                x_d_27.append(x3)
                                y_d_27.append(mvts.index(row[1]))
                                p_d_27.append(id.index(int(row[0])))
                        if row[0] == '28':
                                x1=row[3]
                                x_d_28.append(x1)
                                x2=row[4]
                                x_d_28.append(x2)
                                x3= row[5]
                                x_d_28.append(x3)
                                y_d_28.append(mvts.index(row[1]))
                                p_d_28.append(id.index(int(row[0])))
                        if row[0] == '29':
                                x1=row[3]
                                x_d_29.append(x1)
                                x2=row[4]
                                x_d_29.append(x2)
                                x3= row[5]
                                x_d_29.append(x3)
                                y_d_29.append(mvts.index(row[1]))
                                p_d_29.append(id.index(int(row[0])))
                        if row[0] == '30':
                                x1=row[3]
                                x_d_30.append(x1)
                                x2=row[4]
                                x_d_30.append(x2)
                                x3= row[5]
                                x_d_30.append(x3)
                                y_d_30.append(mvts.index(row[1]))
                                p_d_30.append(id.index(int(row[0])))
                        if row[0] == '31':
                                x1=row[3]
                                x_d_31.append(x1)
                                x2=row[4]
                                x_d_31.append(x2)
                                x3= row[5]
                                x_d_31.append(x3)
                                y_d_31.append(mvts.index(row[1]))
                                p_d_31.append(id.index(int(row[0])))
                        if row[0] == '32':
                                x1=row[3]
                                x_d_32.append(x1)
                                x2=row[4]
                                x_d_32.append(x2)
                                x3= row[5]
                                x_d_32.append(x3)
                                y_d_32.append(mvts.index(row[1]))
                                p_d_32.append(id.index(int(row[0])))
                        if row[0] == '33':
                                x1=row[3]
                                x_d_33.append(x1)
                                x2=row[4]
                                x_d_33.append(x2)
                                x3= row[5]
                                x_d_33.append(x3)
                                y_d_33.append(mvts.index(row[1]))
                                p_d_33.append(id.index(int(row[0])))
                        if row[0] == '34':
                                x1=row[3]
                                x_d_34.append(x1)
                                x2=row[4]
                                x_d_34.append(x2)
                                x3= row[5]
                                x_d_34.append(x3)
                                y_d_34.append(mvts.index(row[1]))
                                p_d_34.append(id.index(int(row[0])))
                        if row[0] == '35':
                                x1=row[3]
                                x_d_35.append(x1)
                                x2=row[4]
                                x_d_35.append(x2)
                                x3= row[5]
                                x_d_35.append(x3)
                                y_d_35.append(mvts.index(row[1]))
                                p_d_35.append(id.index(int(row[0])))
                        if row[0] == '36':
                                x1=row[3]
                                x_d_36.append(x1)
                                x2=row[4]
                                x_d_36.append(x2)
                                x3= row[5]
                                x_d_36.append(x3)
                                y_d_36.append(mvts.index(row[1]))
                                p_d_36.append(id.index(int(row[0])))
                if row[1]== 'Jogging':
                        if row[0] == '1':
                                x1=row[3]
                                x_j_1.append(x1)
                                x2=row[4]
                                x_j_1.append(x2)
                                x3= row[5]
                                x_j_1.append(x3)
                                y_j_1.append(mvts.index(row[1]))
                                p_j_1.append(id.index(int(row[0])))
                        if row[0] == '2':
                                x1=row[3]
                                x_j_2.append(x1)
                                x2=row[4]
                                x_j_2.append(x2)
                                x3= row[5]
                                x_j_2.append(x3)
                                y_j_2.append(mvts.index(row[1]))
                                p_j_2.append(id.index(int(row[0])))
                        if row[0] == '3':
                                x1=row[3]
                                x_j_3.append(x1)
                                x2=row[4]
                                x_j_3.append(x2)
                                x3= row[5]
                                x_j_3.append(x3)
                                y_j_3.append(mvts.index(row[1]))
                                p_j_3.append(id.index(int(row[0])))
                        if row[0] == '4':
                                x1=row[3]
                                x_j_4.append(x1)
                                x2=row[4]
                                x_j_4.append(x2)
                                x3= row[5]
                                x_j_4.append(x3)
                                y_j_4.append(mvts.index(row[1]))
                                p_j_4.append(id.index(int(row[0])))
                        if row[0] == '5':
                                x1=row[3]
                                x_j_5.append(x1)
                                x2=row[4]
                                x_j_5.append(x2)
                                x3= row[5]
                                x_j_5.append(x3)
                                y_j_5.append(mvts.index(row[1]))
                                p_j_5.append(id.index(int(row[0])))
                        if row[0] == '6':
                                x1=row[3]
                                x_j_6.append(x1)
                                x2=row[4]
                                x_j_6.append(x2)
                                x3= row[5]
                                x_j_6.append(x3)
                                y_j_6.append(mvts.index(row[1]))
                                p_j_6.append(id.index(int(row[0])))
                        if row[0] == '7':
                                x1=row[3]
                                x_j_7.append(x1)
                                x2=row[4]
                                x_j_7.append(x2)
                                x3= row[5]
                                x_j_7.append(x3)
                                y_j_7.append(mvts.index(row[1]))
                                p_j_7.append(id.index(int(row[0])))
                        if row[0] == '8':
                                x1=row[3]
                                x_j_8.append(x1)
                                x2=row[4]
                                x_j_8.append(x2)
                                x3= row[5]
                                x_j_8.append(x3)
                                y_j_8.append(mvts.index(row[1]))
                                p_j_8.append(id.index(int(row[0])))
                        if row[0] == '9':
                                x1=row[3]
                                x_j_9.append(x1)
                                x2=row[4]
                                x_j_9.append(x2)
                                x3= row[5]
                                x_j_9.append(x3)
                                y_j_9.append(mvts.index(row[1]))
                                p_j_9.append(id.index(int(row[0])))
                        if row[0] == '10':
                                x1=row[3]
                                x_j_10.append(x1)
                                x2=row[4]
                                x_j_10.append(x2)
                                x3= row[5]
                                x_j_10.append(x3)
                                y_j_10.append(mvts.index(row[1]))
                                p_j_10.append(id.index(int(row[0])))
                        if row[0] == '11':
                                x1=row[3]
                                x_j_11.append(x1)
                                x2=row[4]
                                x_j_11.append(x2)
                                x3= row[5]
                                x_j_11.append(x3)
                                y_j_11.append(mvts.index(row[1]))
                                p_j_11.append(id.index(int(row[0])))
                        if row[0] == '12':
                                x1=row[3]
                                x_j_12.append(x1)
                                x2=row[4]
                                x_j_12.append(x2)
                                x3= row[5]
                                x_j_12.append(x3)
                                y_j_12.append(mvts.index(row[1]))
                                p_j_12.append(id.index(int(row[0])))
                        if row[0] == '13':
                                x1=row[3]
                                x_j_13.append(x1)
                                x2=row[4]
                                x_j_13.append(x2)
                                x3= row[5]
                                x_j_13.append(x3)
                                y_j_13.append(mvts.index(row[1]))
                                p_j_13.append(id.index(int(row[0])))
                        if row[0] == '14':
                                x1=row[3]
                                x_j_14.append(x1)
                                x2=row[4]
                                x_j_14.append(x2)
                                x3= row[5]
                                x_j_14.append(x3)
                                y_j_14.append(mvts.index(row[1]))
                                p_j_14.append(id.index(int(row[0])))
                        if row[0] == '15':
                                x1=row[3]
                                x_j_15.append(x1)
                                x2=row[4]
                                x_j_15.append(x2)
                                x3= row[5]
                                x_j_15.append(x3)
                                y_j_15.append(mvts.index(row[1]))
                                p_j_15.append(id.index(int(row[0])))
                        if row[0] == '16':
                                x1=row[3]
                                x_j_16.append(x1)
                                x2=row[4]
                                x_j_16.append(x2)
                                x3= row[5]
                                x_j_16.append(x3)
                                y_j_16.append(mvts.index(row[1]))
                                p_j_16.append(id.index(int(row[0])))
                        if row[0] == '17':
                                x1=row[3]
                                x_j_17.append(x1)
                                x2=row[4]
                                x_j_17.append(x2)
                                x3= row[5]
                                x_j_17.append(x3)
                                y_j_17.append(mvts.index(row[1]))
                                p_j_17.append(id.index(int(row[0])))
                        if row[0] == '18':
                                x1=row[3]
                                x_j_18.append(x1)
                                x2=row[4]
                                x_j_18.append(x2)
                                x3= row[5]
                                x_j_18.append(x3)
                                y_j_18.append(mvts.index(row[1]))
                                p_j_18.append(id.index(int(row[0])))
                        if row[0] == '19':
                                x1=row[3]
                                x_j_19.append(x1)
                                x2=row[4]
                                x_j_19.append(x2)
                                x3= row[5]
                                x_j_19.append(x3)
                                y_j_19.append(mvts.index(row[1]))
                                p_j_19.append(id.index(int(row[0])))
                        if row[0] == '20':
                                x1=row[3]
                                x_j_20.append(x1)
                                x2=row[4]
                                x_j_20.append(x2)
                                x3= row[5]
                                x_j_20.append(x3)
                                y_j_20.append(mvts.index(row[1]))
                                p_j_20.append(id.index(int(row[0])))
                        if row[0] == '21':
                                x1=row[3]
                                x_j_21.append(x1)
                                x2=row[4]
                                x_j_21.append(x2)
                                x3= row[5]
                                x_j_21.append(x3)
                                y_j_21.append(mvts.index(row[1]))
                                p_j_21.append(id.index(int(row[0])))
                        if row[0] == '22':
                                x1=row[3]
                                x_j_22.append(x1)
                                x2=row[4]
                                x_j_22.append(x2)
                                x3= row[5]
                                x_j_22.append(x3)
                                y_j_22.append(mvts.index(row[1]))
                                p_j_22.append(id.index(int(row[0])))
                        if row[0] == '23':
                                x1=row[3]
                                x_j_23.append(x1)
                                x2=row[4]
                                x_j_23.append(x2)
                                x3= row[5]
                                x_j_23.append(x3)
                                y_j_23.append(mvts.index(row[1]))
                                p_j_23.append(id.index(int(row[0])))
                        if row[0] == '24':
                                x1=row[3]
                                x_j_24.append(x1)
                                x2=row[4]
                                x_j_24.append(x2)
                                x3= row[5]
                                x_j_24.append(x3)
                                y_j_24.append(mvts.index(row[1]))
                                p_j_24.append(id.index(int(row[0])))
                        if row[0] == '25':
                                x1=row[3]
                                x_j_25.append(x1)
                                x2=row[4]
                                x_j_25.append(x2)
                                x3= row[5]
                                x_j_25.append(x3)
                                y_j_25.append(mvts.index(row[1]))
                                p_j_25.append(id.index(int(row[0])))
                        if row[0] == '26':
                                x1=row[3]
                                x_j_26.append(x1)
                                x2=row[4]
                                x_j_26.append(x2)
                                x3= row[5]
                                x_j_26.append(x3)
                                y_j_26.append(mvts.index(row[1]))
                                p_j_26.append(id.index(int(row[0])))
                        if row[0] == '27':
                                x1=row[3]
                                x_j_27.append(x1)
                                x2=row[4]
                                x_j_27.append(x2)
                                x3= row[5]
                                x_j_27.append(x3)
                                y_j_27.append(mvts.index(row[1]))
                                p_j_27.append(id.index(int(row[0])))
                        if row[0] == '28':
                                x1=row[3]
                                x_j_28.append(x1)
                                x2=row[4]
                                x_j_28.append(x2)
                                x3= row[5]
                                x_j_28.append(x3)
                                y_j_28.append(mvts.index(row[1]))
                                p_j_28.append(id.index(int(row[0])))
                        if row[0] == '29':
                                x1=row[3]
                                x_j_29.append(x1)
                                x2=row[4]
                                x_j_29.append(x2)
                                x3= row[5]
                                x_j_29.append(x3)
                                y_j_29.append(mvts.index(row[1]))
                                p_j_29.append(id.index(int(row[0])))
                        if row[0] == '30':
                                x1=row[3]
                                x_j_30.append(x1)
                                x2=row[4]
                                x_j_30.append(x2)
                                x3= row[5]
                                x_j_30.append(x3)
                                y_j_30.append(mvts.index(row[1]))
                                p_j_30.append(id.index(int(row[0])))
                        if row[0] == '31':
                                x1=row[3]
                                x_j_31.append(x1)
                                x2=row[4]
                                x_j_31.append(x2)
                                x3= row[5]
                                x_j_31.append(x3)
                                y_j_31.append(mvts.index(row[1]))
                                p_j_31.append(id.index(int(row[0])))
                        if row[0] == '32':
                                x1=row[3]
                                x_j_32.append(x1)
                                x2=row[4]
                                x_j_32.append(x2)
                                x3= row[5]
                                x_j_32.append(x3)
                                y_j_32.append(mvts.index(row[1]))
                                p_j_32.append(id.index(int(row[0])))
                        if row[0] == '33':
                                x1=row[3]
                                x_j_33.append(x1)
                                x2=row[4]
                                x_j_33.append(x2)
                                x3= row[5]
                                x_j_33.append(x3)
                                y_j_33.append(mvts.index(row[1]))
                                p_j_33.append(id.index(int(row[0])))
                        if row[0] == '34':
                                x1=row[3]
                                x_j_34.append(x1)
                                x2=row[4]
                                x_j_34.append(x2)
                                x3= row[5]
                                x_j_34.append(x3)
                                y_j_34.append(mvts.index(row[1]))
                                p_j_34.append(id.index(int(row[0])))
                        if row[0] == '35':
                                x1=row[3]
                                x_j_35.append(x1)
                                x2=row[4]
                                x_j_35.append(x2)
                                x3= row[5]
                                x_j_35.append(x3)
                                y_j_35.append(mvts.index(row[1]))
                                p_j_35.append(id.index(int(row[0])))
                        if row[0] == '36':
                                x1=row[3]
                                x_j_36.append(x1)
                                x2=row[4]
                                x_j_36.append(x2)
                                x3= row[5]
                                x_j_36.append(x3)
                                y_j_36.append(mvts.index(row[1]))
                                p_j_36.append(id.index(int(row[0])))
                if row[1]== 'Sitting':
                        if row[0] == '1':
                                x1=row[3]
                                x_s_1.append(x1)
                                x2=row[4]
                                x_s_1.append(x2)
                                x3= row[5]
                                x_s_1.append(x3)
                                y_s_1.append(mvts.index(row[1]))
                                p_s_1.append(id.index(int(row[0])))
                        if row[0] == '2':
                                x1=row[3]
                                x_s_2.append(x1)
                                x2=row[4]
                                x_s_2.append(x2)
                                x3= row[5]
                                x_s_2.append(x3)
                                y_s_2.append(mvts.index(row[1]))
                                p_s_2.append(id.index(int(row[0])))
                        if row[0] == '3':
                                x1=row[3]
                                x_s_3.append(x1)
                                x2=row[4]
                                x_s_3.append(x2)
                                x3= row[5]
                                x_s_3.append(x3)
                                y_s_3.append(mvts.index(row[1]))
                                p_s_3.append(id.index(int(row[0])))
                        if row[0] == '4':
                                x1=row[3]
                                x_s_4.append(x1)
                                x2=row[4]
                                x_s_4.append(x2)
                                x3= row[5]
                                x_s_4.append(x3)
                                y_s_4.append(mvts.index(row[1]))
                                p_s_4.append(id.index(int(row[0])))
                        if row[0] == '5':
                                x1=row[3]
                                x_s_5.append(x1)
                                x2=row[4]
                                x_s_5.append(x2)
                                x3= row[5]
                                x_s_5.append(x3)
                                y_s_5.append(mvts.index(row[1]))
                                p_s_5.append(id.index(int(row[0])))
                        if row[0] == '6':
                                x1=row[3]
                                x_s_6.append(x1)
                                x2=row[4]
                                x_s_6.append(x2)
                                x3= row[5]
                                x_s_6.append(x3)
                                y_s_6.append(mvts.index(row[1]))
                                p_s_6.append(id.index(int(row[0])))
                        if row[0] == '7':
                                x1=row[3]
                                x_s_7.append(x1)
                                x2=row[4]
                                x_s_7.append(x2)
                                x3= row[5]
                                x_s_7.append(x3)
                                y_s_7.append(mvts.index(row[1]))
                                p_s_7.append(id.index(int(row[0])))
                        if row[0] == '8':
                                x1=row[3]
                                x_s_8.append(x1)
                                x2=row[4]
                                x_s_8.append(x2)
                                x3= row[5]
                                x_s_8.append(x3)
                                y_s_8.append(mvts.index(row[1]))
                                p_s_8.append(id.index(int(row[0])))
                        if row[0] == '9':
                                x1=row[3]
                                x_s_9.append(x1)
                                x2=row[4]
                                x_s_9.append(x2)
                                x3= row[5]
                                x_s_9.append(x3)
                                y_s_9.append(mvts.index(row[1]))
                                p_s_9.append(id.index(int(row[0])))
                        if row[0] == '10':
                                x1=row[3]
                                x_s_10.append(x1)
                                x2=row[4]
                                x_s_10.append(x2)
                                x3= row[5]
                                x_s_10.append(x3)
                                y_s_10.append(mvts.index(row[1]))
                                p_s_10.append(id.index(int(row[0])))
                        if row[0] == '11':
                                x1=row[3]
                                x_s_11.append(x1)
                                x2=row[4]
                                x_s_11.append(x2)
                                x3= row[5]
                                x_s_11.append(x3)
                                y_s_11.append(mvts.index(row[1]))
                                p_s_11.append(id.index(int(row[0])))
                        if row[0] == '12':
                                x1=row[3]
                                x_s_12.append(x1)
                                x2=row[4]
                                x_s_12.append(x2)
                                x3= row[5]
                                x_s_12.append(x3)
                                y_s_12.append(mvts.index(row[1]))
                                p_s_12.append(id.index(int(row[0])))
                        if row[0] == '13':
                                x1=row[3]
                                x_s_13.append(x1)
                                x2=row[4]
                                x_s_13.append(x2)
                                x3= row[5]
                                x_s_13.append(x3)
                                y_s_13.append(mvts.index(row[1]))
                                p_s_13.append(id.index(int(row[0])))
                        if row[0] == '14':
                                x1=row[3]
                                x_s_14.append(x1)
                                x2=row[4]
                                x_s_14.append(x2)
                                x3= row[5]
                                x_s_14.append(x3)
                                y_s_14.append(mvts.index(row[1]))
                                p_s_14.append(id.index(int(row[0])))
                        if row[0] == '15':
                                x1=row[3]
                                x_s_15.append(x1)
                                x2=row[4]
                                x_s_15.append(x2)
                                x3= row[5]
                                x_s_15.append(x3)
                                y_s_15.append(mvts.index(row[1]))
                                p_s_15.append(id.index(int(row[0])))
                        if row[0] == '16':
                                x1=row[3]
                                x_s_16.append(x1)
                                x2=row[4]
                                x_s_16.append(x2)
                                x3= row[5]
                                x_s_16.append(x3)
                                y_s_16.append(mvts.index(row[1]))
                                p_s_16.append(id.index(int(row[0])))
                        if row[0] == '17':
                                x1=row[3]
                                x_s_17.append(x1)
                                x2=row[4]
                                x_s_17.append(x2)
                                x3= row[5]
                                x_s_17.append(x3)
                                y_s_17.append(mvts.index(row[1]))
                                p_s_17.append(id.index(int(row[0])))
                        if row[0] == '18':
                                x1=row[3]
                                x_s_18.append(x1)
                                x2=row[4]
                                x_s_18.append(x2)
                                x3= row[5]
                                x_s_18.append(x3)
                                y_s_18.append(mvts.index(row[1]))
                                p_s_18.append(id.index(int(row[0])))
                        if row[0] == '19':
                                x1=row[3]
                                x_s_19.append(x1)
                                x2=row[4]
                                x_s_19.append(x2)
                                x3= row[5]
                                x_s_19.append(x3)
                                y_s_19.append(mvts.index(row[1]))
                                p_s_19.append(id.index(int(row[0])))
                        if row[0] == '20':
                                x1=row[3]
                                x_s_20.append(x1)
                                x2=row[4]
                                x_s_20.append(x2)
                                x3= row[5]
                                x_s_20.append(x3)
                                y_s_20.append(mvts.index(row[1]))
                                p_s_20.append(id.index(int(row[0])))
                        if row[0] == '21':
                                x1=row[3]
                                x_s_21.append(x1)
                                x2=row[4]
                                x_s_21.append(x2)
                                x3= row[5]
                                x_s_21.append(x3)
                                y_s_21.append(mvts.index(row[1]))
                                p_s_21.append(id.index(int(row[0])))
                        if row[0] == '22':
                                x1=row[3]
                                x_s_22.append(x1)
                                x2=row[4]
                                x_s_22.append(x2)
                                x3= row[5]
                                x_s_22.append(x3)
                                y_s_22.append(mvts.index(row[1]))
                                p_s_22.append(id.index(int(row[0])))
                        if row[0] == '23':
                                x1=row[3]
                                x_s_23.append(x1)
                                x2=row[4]
                                x_s_23.append(x2)
                                x3= row[5]
                                x_s_23.append(x3)
                                y_s_23.append(mvts.index(row[1]))
                                p_s_23.append(id.index(int(row[0])))
                        if row[0] == '24':
                                x1=row[3]
                                x_s_24.append(x1)
                                x2=row[4]
                                x_s_24.append(x2)
                                x3= row[5]
                                x_s_24.append(x3)
                                y_s_24.append(mvts.index(row[1]))
                                p_s_24.append(id.index(int(row[0])))
                        if row[0] == '25':
                                x1=row[3]
                                x_s_25.append(x1)
                                x2=row[4]
                                x_s_25.append(x2)
                                x3= row[5]
                                x_s_25.append(x3)
                                y_s_25.append(mvts.index(row[1]))
                                p_s_25.append(id.index(int(row[0])))
                        if row[0] == '26':
                                x1=row[3]
                                x_s_26.append(x1)
                                x2=row[4]
                                x_s_26.append(x2)
                                x3= row[5]
                                x_s_26.append(x3)
                                y_s_26.append(mvts.index(row[1]))
                                p_s_26.append(id.index(int(row[0])))
                        if row[0] == '27':
                                x1=row[3]
                                x_s_27.append(x1)
                                x2=row[4]
                                x_s_27.append(x2)
                                x3= row[5]
                                x_s_27.append(x3)
                                y_s_27.append(mvts.index(row[1]))
                                p_s_27.append(id.index(int(row[0])))
                        if row[0] == '28':
                                x1=row[3]
                                x_s_28.append(x1)
                                x2=row[4]
                                x_s_28.append(x2)
                                x3= row[5]
                                x_s_28.append(x3)
                                y_s_28.append(mvts.index(row[1]))
                                p_s_28.append(id.index(int(row[0])))
                        if row[0] == '29':
                                x1=row[3]
                                x_s_29.append(x1)
                                x2=row[4]
                                x_s_29.append(x2)
                                x3= row[5]
                                x_s_29.append(x3)
                                y_s_29.append(mvts.index(row[1]))
                                p_s_29.append(id.index(int(row[0])))
                        if row[0] == '30':
                                x1=row[3]
                                x_s_30.append(x1)
                                x2=row[4]
                                x_s_30.append(x2)
                                x3= row[5]
                                x_s_30.append(x3)
                                y_s_30.append(mvts.index(row[1]))
                                p_s_30.append(id.index(int(row[0])))
                        if row[0] == '31':
                                x1=row[3]
                                x_s_31.append(x1)
                                x2=row[4]
                                x_s_31.append(x2)
                                x3= row[5]
                                x_s_31.append(x3)
                                y_s_31.append(mvts.index(row[1]))
                                p_s_31.append(id.index(int(row[0])))
                        if row[0] == '32':
                                x1=row[3]
                                x_s_32.append(x1)
                                x2=row[4]
                                x_s_32.append(x2)
                                x3= row[5]
                                x_s_32.append(x3)
                                y_s_32.append(mvts.index(row[1]))
                                p_s_32.append(id.index(int(row[0])))
                        if row[0] == '33':
                                x1=row[3]
                                x_s_33.append(x1)
                                x2=row[4]
                                x_s_33.append(x2)
                                x3= row[5]
                                x_s_33.append(x3)
                                y_s_33.append(mvts.index(row[1]))
                                p_s_33.append(id.index(int(row[0])))
                        if row[0] == '34':
                                x1=row[3]
                                x_s_34.append(x1)
                                x2=row[4]
                                x_s_34.append(x2)
                                x3= row[5]
                                x_s_34.append(x3)
                                y_s_34.append(mvts.index(row[1]))
                                p_s_34.append(id.index(int(row[0])))
                        if row[0] == '35':
                                x1=row[3]
                                x_s_35.append(x1)
                                x2=row[4]
                                x_s_35.append(x2)
                                x3= row[5]
                                x_s_35.append(x3)
                                y_s_35.append(mvts.index(row[1]))
                                p_s_35.append(id.index(int(row[0])))
                        if row[0] == '36':
                                x1=row[3]
                                x_s_36.append(x1)
                                x2=row[4]
                                x_s_36.append(x2)
                                x3= row[5]
                                x_s_36.append(x3)
                                y_s_36.append(mvts.index(row[1]))
                                p_s_36.append(id.index(int(row[0])))                
                if row[1]=='Standing':
                        if row[0] == '1':
                                x1=row[3]
                                x_st_1.append(x1)
                                x2=row[4]
                                x_st_1.append(x2)
                                x3= row[5]
                                x_st_1.append(x3)
                                y_st_1.append(mvts.index(row[1]))
                                p_st_1.append(id.index(int(row[0])))
                        if row[0] == '2':
                                x1=row[3]
                                x_st_2.append(x1)
                                x2=row[4]
                                x_st_2.append(x2)
                                x3= row[5]
                                x_st_2.append(x3)
                                y_st_2.append(mvts.index(row[1]))
                                p_st_2.append(id.index(int(row[0])))
                        if row[0] == '3':
                                x1=row[3]
                                x_st_3.append(x1)
                                x2=row[4]
                                x_st_3.append(x2)
                                x3= row[5]
                                x_st_3.append(x3)
                                y_st_3.append(mvts.index(row[1]))
                                p_st_3.append(id.index(int(row[0])))
                        if row[0] == '4':
                                x1=row[3]
                                x_st_4.append(x1)
                                x2=row[4]
                                x_st_4.append(x2)
                                x3= row[5]
                                x_st_4.append(x3)
                                y_st_4.append(mvts.index(row[1]))
                                p_st_4.append(id.index(int(row[0])))
                        if row[0] == '5':
                                x1=row[3]
                                x_st_5.append(x1)
                                x2=row[4]
                                x_st_5.append(x2)
                                x3= row[5]
                                x_st_5.append(x3)
                                y_st_5.append(mvts.index(row[1]))
                                p_st_5.append(id.index(int(row[0])))
                        if row[0] == '6':
                                x1=row[3]
                                x_st_6.append(x1)
                                x2=row[4]
                                x_st_6.append(x2)
                                x3= row[5]
                                x_st_6.append(x3)
                                y_st_6.append(mvts.index(row[1]))
                                p_st_6.append(id.index(int(row[0])))
                        if row[0] == '7':
                                x1=row[3]
                                x_st_7.append(x1)
                                x2=row[4]
                                x_st_7.append(x2)
                                x3= row[5]
                                x_st_7.append(x3)
                                y_st_7.append(mvts.index(row[1]))
                                p_st_7.append(id.index(int(row[0])))
                        if row[0] == '8':
                                x1=row[3]
                                x_st_8.append(x1)
                                x2=row[4]
                                x_st_8.append(x2)
                                x3= row[5]
                                x_st_8.append(x3)
                                y_st_8.append(mvts.index(row[1]))
                                p_st_8.append(id.index(int(row[0])))
                        if row[0] == '9':
                                x1=row[3]
                                x_st_9.append(x1)
                                x2=row[4]
                                x_st_9.append(x2)
                                x3= row[5]
                                x_st_9.append(x3)
                                y_st_9.append(mvts.index(row[1]))
                                p_st_9.append(id.index(int(row[0])))
                        if row[0] == '10':
                                x1=row[3]
                                x_st_10.append(x1)
                                x2=row[4]
                                x_st_10.append(x2)
                                x3= row[5]
                                x_st_10.append(x3)
                                y_st_10.append(mvts.index(row[1]))
                                p_st_10.append(id.index(int(row[0])))
                        if row[0] == '11':
                                x1=row[3]
                                x_st_11.append(x1)
                                x2=row[4]
                                x_st_11.append(x2)
                                x3= row[5]
                                x_st_11.append(x3)
                                y_st_11.append(mvts.index(row[1]))
                                p_st_11.append(id.index(int(row[0])))
                        if row[0] == '12':
                                x1=row[3]
                                x_st_12.append(x1)
                                x2=row[4]
                                x_st_12.append(x2)
                                x3= row[5]
                                x_st_12.append(x3)
                                y_st_12.append(mvts.index(row[1]))
                                p_st_12.append(id.index(int(row[0])))
                        if row[0] == '13':
                                x1=row[3]
                                x_st_13.append(x1)
                                x2=row[4]
                                x_st_13.append(x2)
                                x3= row[5]
                                x_st_13.append(x3)
                                y_st_13.append(mvts.index(row[1]))
                                p_st_13.append(id.index(int(row[0])))
                        if row[0] == '14':
                                x1=row[3]
                                x_st_14.append(x1)
                                x2=row[4]
                                x_st_14.append(x2)
                                x3= row[5]
                                x_st_14.append(x3)
                                y_st_14.append(mvts.index(row[1]))
                                p_st_14.append(id.index(int(row[0])))
                        if row[0] == '15':
                                x1=row[3]
                                x_st_15.append(x1)
                                x2=row[4]
                                x_st_15.append(x2)
                                x3= row[5]
                                x_st_15.append(x3)
                                y_st_15.append(mvts.index(row[1]))
                                p_st_15.append(id.index(int(row[0])))
                        if row[0] == '16':
                                x1=row[3]
                                x_st_16.append(x1)
                                x2=row[4]
                                x_st_16.append(x2)
                                x3= row[5]
                                x_st_16.append(x3)
                                y_st_16.append(mvts.index(row[1]))
                                p_st_16.append(id.index(int(row[0])))
                        if row[0] == '17':
                                x1=row[3]
                                x_st_17.append(x1)
                                x2=row[4]
                                x_st_17.append(x2)
                                x3= row[5]
                                x_st_17.append(x3)
                                y_st_17.append(mvts.index(row[1]))
                                p_st_17.append(id.index(int(row[0])))
                        if row[0] == '18':
                                x1=row[3]
                                x_st_18.append(x1)
                                x2=row[4]
                                x_st_18.append(x2)
                                x3= row[5]
                                x_st_18.append(x3)
                                y_st_18.append(mvts.index(row[1]))
                                p_st_18.append(id.index(int(row[0])))
                        if row[0] == '19':
                                x1=row[3]
                                x_st_19.append(x1)
                                x2=row[4]
                                x_st_19.append(x2)
                                x3= row[5]
                                x_st_19.append(x3)
                                y_st_19.append(mvts.index(row[1]))
                                p_st_19.append(id.index(int(row[0])))
                        if row[0] == '20':
                                x1=row[3]
                                x_st_20.append(x1)
                                x2=row[4]
                                x_st_20.append(x2)
                                x3= row[5]
                                x_st_20.append(x3)
                                y_st_20.append(mvts.index(row[1]))
                                p_st_20.append(id.index(int(row[0])))
                        if row[0] == '21':
                                x1=row[3]
                                x_st_21.append(x1)
                                x2=row[4]
                                x_st_21.append(x2)
                                x3= row[5]
                                x_st_21.append(x3)
                                y_st_21.append(mvts.index(row[1]))
                                p_st_21.append(id.index(int(row[0])))
                        if row[0] == '22':
                                x1=row[3]
                                x_st_22.append(x1)
                                x2=row[4]
                                x_st_22.append(x2)
                                x3= row[5]
                                x_st_22.append(x3)
                                y_st_22.append(mvts.index(row[1]))
                                p_st_22.append(id.index(int(row[0])))
                        if row[0] == '23':
                                x1=row[3]
                                x_st_23.append(x1)
                                x2=row[4]
                                x_st_23.append(x2)
                                x3= row[5]
                                x_st_23.append(x3)
                                y_st_23.append(mvts.index(row[1]))
                                p_st_23.append(id.index(int(row[0])))
                        if row[0] == '24':
                                x1=row[3]
                                x_st_24.append(x1)
                                x2=row[4]
                                x_st_24.append(x2)
                                x3= row[5]
                                x_st_24.append(x3)
                                y_st_24.append(mvts.index(row[1]))
                                p_st_24.append(id.index(int(row[0])))
                        if row[0] == '25':
                                x1=row[3]
                                x_st_25.append(x1)
                                x2=row[4]
                                x_st_25.append(x2)
                                x3= row[5]
                                x_st_25.append(x3)
                                y_st_25.append(mvts.index(row[1]))
                                p_st_25.append(id.index(int(row[0])))
                        if row[0] == '26':
                                x1=row[3]
                                x_st_26.append(x1)
                                x2=row[4]
                                x_st_26.append(x2)
                                x3= row[5]
                                x_st_26.append(x3)
                                y_st_26.append(mvts.index(row[1]))
                                p_st_26.append(id.index(int(row[0])))
                        if row[0] == '27':
                                x1=row[3]
                                x_st_27.append(x1)
                                x2=row[4]
                                x_st_27.append(x2)
                                x3= row[5]
                                x_st_27.append(x3)
                                y_st_27.append(mvts.index(row[1]))
                                p_st_27.append(id.index(int(row[0])))
                        if row[0] == '28':
                                x1=row[3]
                                x_st_28.append(x1)
                                x2=row[4]
                                x_st_28.append(x2)
                                x3= row[5]
                                x_st_28.append(x3)
                                y_st_28.append(mvts.index(row[1]))
                                p_st_28.append(id.index(int(row[0])))
                        if row[0] == '29':
                                x1=row[3]
                                x_st_29.append(x1)
                                x2=row[4]
                                x_st_29.append(x2)
                                x3= row[5]
                                x_st_29.append(x3)
                                y_st_29.append(mvts.index(row[1]))
                                p_st_29.append(id.index(int(row[0])))
                        if row[0] == '30':
                                x1=row[3]
                                x_st_30.append(x1)
                                x2=row[4]
                                x_st_30.append(x2)
                                x3= row[5]
                                x_st_30.append(x3)
                                y_st_30.append(mvts.index(row[1]))
                                p_st_30.append(id.index(int(row[0])))
                        if row[0] == '31':
                                x1=row[3]
                                x_st_31.append(x1)
                                x2=row[4]
                                x_st_31.append(x2)
                                x3= row[5]
                                x_st_31.append(x3)
                                y_st_31.append(mvts.index(row[1]))
                                p_st_31.append(id.index(int(row[0])))
                        if row[0] == '32':
                                x1=row[3]
                                x_st_32.append(x1)
                                x2=row[4]
                                x_st_32.append(x2)
                                x3= row[5]
                                x_st_32.append(x3)
                                y_st_32.append(mvts.index(row[1]))
                                p_st_32.append(id.index(int(row[0])))
                        if row[0] == '33':
                                x1=row[3]
                                x_st_33.append(x1)
                                x2=row[4]
                                x_st_33.append(x2)
                                x3= row[5]
                                x_st_33.append(x3)
                                y_st_33.append(mvts.index(row[1]))
                                p_st_33.append(id.index(int(row[0])))
                        if row[0] == '34':
                                x1=row[3]
                                x_st_34.append(x1)
                                x2=row[4]
                                x_st_34.append(x2)
                                x3= row[5]
                                x_st_34.append(x3)
                                y_st_34.append(mvts.index(row[1]))
                                p_st_34.append(id.index(int(row[0])))
                        if row[0] == '35':
                                x1=row[3]
                                x_st_35.append(x1)
                                x2=row[4]
                                x_st_35.append(x2)
                                x3= row[5]
                                x_st_35.append(x3)
                                y_st_35.append(mvts.index(row[1]))
                                p_st_35.append(id.index(int(row[0])))
                        if row[0] == '36':
                                x1=row[3]
                                x_st_36.append(x1)
                                x2=row[4]
                                x_st_36.append(x2)
                                x3= row[5]
                                x_st_36.append(x3)
                                y_st_36.append(mvts.index(row[1]))
                                p_st_36.append(id.index(int(row[0])))
                if row[1]=='Upstairs':
                        if row[0] == '1':
                                x1=row[3]
                                x_u_1.append(x1)
                                x2=row[4]
                                x_u_1.append(x2)
                                x3= row[5]
                                x_u_1.append(x3)
                                y_u_1.append(mvts.index(row[1]))
                                p_u_1.append(id.index(int(row[0])))
                        if row[0] == '2':
                                x1=row[3]
                                x_u_2.append(x1)
                                x2=row[4]
                                x_u_2.append(x2)
                                x3= row[5]
                                x_u_2.append(x3)
                                y_u_2.append(mvts.index(row[1]))
                                p_u_2.append(id.index(int(row[0])))
                        if row[0] == '3':
                                x1=row[3]
                                x_u_3.append(x1)
                                x2=row[4]
                                x_u_3.append(x2)
                                x3= row[5]
                                x_u_3.append(x3)
                                y_u_3.append(mvts.index(row[1]))
                                p_u_3.append(id.index(int(row[0])))
                        if row[0] == '4':
                                x1=row[3]
                                x_u_4.append(x1)
                                x2=row[4]
                                x_u_4.append(x2)
                                x3= row[5]
                                x_u_4.append(x3)
                                y_u_4.append(mvts.index(row[1]))
                                p_u_4.append(id.index(int(row[0])))
                        if row[0] == '5':
                                x1=row[3]
                                x_u_5.append(x1)
                                x2=row[4]
                                x_u_5.append(x2)
                                x3= row[5]
                                x_u_5.append(x3)
                                y_u_5.append(mvts.index(row[1]))
                                p_u_5.append(id.index(int(row[0])))
                        if row[0] == '6':
                                x1=row[3]
                                x_u_6.append(x1)
                                x2=row[4]
                                x_u_6.append(x2)
                                x3= row[5]
                                x_u_6.append(x3)
                                y_u_6.append(mvts.index(row[1]))
                                p_u_6.append(id.index(int(row[0])))
                        if row[0] == '7':
                                x1=row[3]
                                x_u_7.append(x1)
                                x2=row[4]
                                x_u_7.append(x2)
                                x3= row[5]
                                x_u_7.append(x3)
                                y_u_7.append(mvts.index(row[1]))
                                p_u_7.append(id.index(int(row[0])))
                        if row[0] == '8':
                                x1=row[3]
                                x_u_8.append(x1)
                                x2=row[4]
                                x_u_8.append(x2)
                                x3= row[5]
                                x_u_8.append(x3)
                                y_u_8.append(mvts.index(row[1]))
                                p_u_8.append(id.index(int(row[0])))
                        if row[0] == '9':
                                x1=row[3]
                                x_u_9.append(x1)
                                x2=row[4]
                                x_u_9.append(x2)
                                x3= row[5]
                                x_u_9.append(x3)
                                y_u_9.append(mvts.index(row[1]))
                                p_u_9.append(id.index(int(row[0])))
                        if row[0] == '10':
                                x1=row[3]
                                x_u_10.append(x1)
                                x2=row[4]
                                x_u_10.append(x2)
                                x3= row[5]
                                x_u_10.append(x3)
                                y_u_10.append(mvts.index(row[1]))
                                p_u_10.append(id.index(int(row[0])))
                        if row[0] == '11':
                                x1=row[3]
                                x_u_11.append(x1)
                                x2=row[4]
                                x_u_11.append(x2)
                                x3= row[5]
                                x_u_11.append(x3)
                                y_u_11.append(mvts.index(row[1]))
                                p_u_11.append(id.index(int(row[0])))
                        if row[0] == '12':
                                x1=row[3]
                                x_u_12.append(x1)
                                x2=row[4]
                                x_u_12.append(x2)
                                x3= row[5]
                                x_u_12.append(x3)
                                y_u_12.append(mvts.index(row[1]))
                                p_u_12.append(id.index(int(row[0])))
                        if row[0] == '13':
                                x1=row[3]
                                x_u_13.append(x1)
                                x2=row[4]
                                x_u_13.append(x2)
                                x3= row[5]
                                x_u_13.append(x3)
                                y_u_13.append(mvts.index(row[1]))
                                p_u_13.append(id.index(int(row[0])))
                        if row[0] == '14':
                                x1=row[3]
                                x_u_14.append(x1)
                                x2=row[4]
                                x_u_14.append(x2)
                                x3= row[5]
                                x_u_14.append(x3)
                                y_u_14.append(mvts.index(row[1]))
                                p_u_14.append(id.index(int(row[0])))
                        if row[0] == '15':
                                x1=row[3]
                                x_u_15.append(x1)
                                x2=row[4]
                                x_u_15.append(x2)
                                x3= row[5]
                                x_u_15.append(x3)
                                y_u_15.append(mvts.index(row[1]))
                                p_u_15.append(id.index(int(row[0])))
                        if row[0] == '16':
                                x1=row[3]
                                x_u_16.append(x1)
                                x2=row[4]
                                x_u_16.append(x2)
                                x3= row[5]
                                x_u_16.append(x3)
                                y_u_16.append(mvts.index(row[1]))
                                p_u_16.append(id.index(int(row[0])))
                        if row[0] == '17':
                                x1=row[3]
                                x_u_17.append(x1)
                                x2=row[4]
                                x_u_17.append(x2)
                                x3= row[5]
                                x_u_17.append(x3)
                                y_u_17.append(mvts.index(row[1]))
                                p_u_17.append(id.index(int(row[0])))
                        if row[0] == '18':
                                x1=row[3]
                                x_u_18.append(x1)
                                x2=row[4]
                                x_u_18.append(x2)
                                x3= row[5]
                                x_u_18.append(x3)
                                y_u_18.append(mvts.index(row[1]))
                                p_u_18.append(id.index(int(row[0])))
                        if row[0] == '19':
                                x1=row[3]
                                x_u_19.append(x1)
                                x2=row[4]
                                x_u_19.append(x2)
                                x3= row[5]
                                x_u_19.append(x3)
                                y_u_19.append(mvts.index(row[1]))
                                p_u_19.append(id.index(int(row[0])))
                        if row[0] == '20':
                                x1=row[3]
                                x_u_20.append(x1)
                                x2=row[4]
                                x_u_20.append(x2)
                                x3= row[5]
                                x_u_20.append(x3)
                                y_u_20.append(mvts.index(row[1]))
                                p_u_20.append(id.index(int(row[0])))
                        if row[0] == '21':
                                x1=row[3]
                                x_u_21.append(x1)
                                x2=row[4]
                                x_u_21.append(x2)
                                x3= row[5]
                                x_u_21.append(x3)
                                y_u_21.append(mvts.index(row[1]))
                                p_u_21.append(id.index(int(row[0])))
                        if row[0] == '22':
                                x1=row[3]
                                x_u_22.append(x1)
                                x2=row[4]
                                x_u_22.append(x2)
                                x3= row[5]
                                x_u_22.append(x3)
                                y_u_22.append(mvts.index(row[1]))
                                p_u_22.append(id.index(int(row[0])))
                        if row[0] == '23':
                                x1=row[3]
                                x_u_23.append(x1)
                                x2=row[4]
                                x_u_23.append(x2)
                                x3= row[5]
                                x_u_23.append(x3)
                                y_u_23.append(mvts.index(row[1]))
                                p_u_23.append(id.index(int(row[0])))
                        if row[0] == '24':
                                x1=row[3]
                                x_u_24.append(x1)
                                x2=row[4]
                                x_u_24.append(x2)
                                x3= row[5]
                                x_u_24.append(x3)
                                y_u_24.append(mvts.index(row[1]))
                                p_u_24.append(id.index(int(row[0])))
                        if row[0] == '25':
                                x1=row[3]
                                x_u_25.append(x1)
                                x2=row[4]
                                x_u_25.append(x2)
                                x3= row[5]
                                x_u_25.append(x3)
                                y_u_25.append(mvts.index(row[1]))
                                p_u_25.append(id.index(int(row[0])))
                        if row[0] == '26':
                                x1=row[3]
                                x_u_26.append(x1)
                                x2=row[4]
                                x_u_26.append(x2)
                                x3= row[5]
                                x_u_26.append(x3)
                                y_u_26.append(mvts.index(row[1]))
                                p_u_26.append(id.index(int(row[0])))
                        if row[0] == '27':
                                x1=row[3]
                                x_u_27.append(x1)
                                x2=row[4]
                                x_u_27.append(x2)
                                x3= row[5]
                                x_u_27.append(x3)
                                y_u_27.append(mvts.index(row[1]))
                                p_u_27.append(id.index(int(row[0])))
                        if row[0] == '28':
                                x1=row[3]
                                x_u_28.append(x1)
                                x2=row[4]
                                x_u_28.append(x2)
                                x3= row[5]
                                x_u_28.append(x3)
                                y_u_28.append(mvts.index(row[1]))
                                p_u_28.append(id.index(int(row[0])))
                        if row[0] == '29':
                                x1=row[3]
                                x_u_29.append(x1)
                                x2=row[4]
                                x_u_29.append(x2)
                                x3= row[5]
                                x_u_29.append(x3)
                                y_u_29.append(mvts.index(row[1]))
                                p_u_29.append(id.index(int(row[0])))
                        if row[0] == '30':
                                x1=row[3]
                                x_u_30.append(x1)
                                x2=row[4]
                                x_u_30.append(x2)
                                x3= row[5]
                                x_u_30.append(x3)
                                y_u_30.append(mvts.index(row[1]))
                                p_u_30.append(id.index(int(row[0])))
                        if row[0] == '31':
                                x1=row[3]
                                x_u_31.append(x1)
                                x2=row[4]
                                x_u_31.append(x2)
                                x3= row[5]
                                x_u_31.append(x3)
                                y_u_31.append(mvts.index(row[1]))
                                p_u_31.append(id.index(int(row[0])))
                        if row[0] == '32':
                                x1=row[3]
                                x_u_32.append(x1)
                                x2=row[4]
                                x_u_32.append(x2)
                                x3= row[5]
                                x_u_32.append(x3)
                                y_u_32.append(mvts.index(row[1]))
                                p_u_32.append(id.index(int(row[0])))
                        if row[0] == '33':
                                x1=row[3]
                                x_u_33.append(x1)
                                x2=row[4]
                                x_u_33.append(x2)
                                x3= row[5]
                                x_u_33.append(x3)
                                y_u_33.append(mvts.index(row[1]))
                                p_u_33.append(id.index(int(row[0])))
                        if row[0] == '34':
                                x1=row[3]
                                x_u_34.append(x1)
                                x2=row[4]
                                x_u_34.append(x2)
                                x3= row[5]
                                x_u_34.append(x3)
                                y_u_34.append(mvts.index(row[1]))
                                p_u_34.append(id.index(int(row[0])))
                        if row[0] == '35':
                                x1=row[3]
                                x_u_35.append(x1)
                                x2=row[4]
                                x_u_35.append(x2)
                                x3= row[5]
                                x_u_35.append(x3)
                                y_u_35.append(mvts.index(row[1]))
                                p_u_35.append(id.index(int(row[0])))
                        if row[0] == '36':
                                x1=row[3]
                                x_u_36.append(x1)
                                x2=row[4]
                                x_u_36.append(x2)
                                x3= row[5]
                                x_u_36.append(x3)
                                y_u_36.append(mvts.index(row[1]))
                                p_u_36.append(id.index(int(row[0])))
                if row[1]=='Walking':
                        if row[0] == '1':
                                x1=row[3]
                                x_w_1.append(x1)
                                x2=row[4]
                                x_w_1.append(x2)
                                x3= row[5]
                                x_w_1.append(x3)
                                y_w_1.append(mvts.index(row[1]))
                                p_w_1.append(id.index(int(row[0])))
                        if row[0] == '2':
                                x1=row[3]
                                x_w_2.append(x1)
                                x2=row[4]
                                x_w_2.append(x2)
                                x3= row[5]
                                x_w_2.append(x3)
                                y_w_2.append(mvts.index(row[1]))
                                p_w_2.append(id.index(int(row[0])))
                        if row[0] == '3':
                                x1=row[3]
                                x_w_3.append(x1)
                                x2=row[4]
                                x_w_3.append(x2)
                                x3= row[5]
                                x_w_3.append(x3)
                                y_w_3.append(mvts.index(row[1]))
                                p_w_3.append(id.index(int(row[0])))
                        if row[0] == '4':
                                x1=row[3]
                                x_w_4.append(x1)
                                x2=row[4]
                                x_w_4.append(x2)
                                x3= row[5]
                                x_w_4.append(x3)
                                y_w_4.append(mvts.index(row[1]))
                                p_w_4.append(id.index(int(row[0])))
                        if row[0] == '5':
                                x1=row[3]
                                x_w_5.append(x1)
                                x2=row[4]
                                x_w_5.append(x2)
                                x3= row[5]
                                x_w_5.append(x3)
                                y_w_5.append(mvts.index(row[1]))
                                p_w_5.append(id.index(int(row[0])))
                        if row[0] == '6':
                                x1=row[3]
                                x_w_6.append(x1)
                                x2=row[4]
                                x_w_6.append(x2)
                                x3= row[5]
                                x_w_6.append(x3)
                                y_w_6.append(mvts.index(row[1]))
                                p_w_6.append(id.index(int(row[0])))
                        if row[0] == '7':
                                x1=row[3]
                                x_w_7.append(x1)
                                x2=row[4]
                                x_w_7.append(x2)
                                x3= row[5]
                                x_w_7.append(x3)
                                y_w_7.append(mvts.index(row[1]))
                                p_w_7.append(id.index(int(row[0])))
                        if row[0] == '8':
                                x1=row[3]
                                x_w_8.append(x1)
                                x2=row[4]
                                x_w_8.append(x2)
                                x3= row[5]
                                x_w_8.append(x3)
                                y_w_8.append(mvts.index(row[1]))
                                p_w_8.append(id.index(int(row[0])))
                        if row[0] == '9':
                                x1=row[3]
                                x_w_9.append(x1)
                                x2=row[4]
                                x_w_9.append(x2)
                                x3= row[5]
                                x_w_9.append(x3)
                                y_w_9.append(mvts.index(row[1]))
                                p_w_9.append(id.index(int(row[0])))
                        if row[0] == '10':
                                x1=row[3]
                                x_w_10.append(x1)
                                x2=row[4]
                                x_w_10.append(x2)
                                x3= row[5]
                                x_w_10.append(x3)
                                y_w_10.append(mvts.index(row[1]))
                                p_w_10.append(id.index(int(row[0])))
                        if row[0] == '11':
                                x1=row[3]
                                x_w_11.append(x1)
                                x2=row[4]
                                x_w_11.append(x2)
                                x3= row[5]
                                x_w_11.append(x3)
                                y_w_11.append(mvts.index(row[1]))
                                p_w_11.append(id.index(int(row[0])))
                        if row[0] == '12':
                                x1=row[3]
                                x_w_12.append(x1)
                                x2=row[4]
                                x_w_12.append(x2)
                                x3= row[5]
                                x_w_12.append(x3)
                                y_w_12.append(mvts.index(row[1]))
                                p_w_12.append(id.index(int(row[0])))
                        if row[0] == '13':
                                x1=row[3]
                                x_w_13.append(x1)
                                x2=row[4]
                                x_w_13.append(x2)
                                x3= row[5]
                                x_w_13.append(x3)
                                y_w_13.append(mvts.index(row[1]))
                                p_w_13.append(id.index(int(row[0])))
                        if row[0] == '14':
                                x1=row[3]
                                x_w_14.append(x1)
                                x2=row[4]
                                x_w_14.append(x2)
                                x3= row[5]
                                x_w_14.append(x3)
                                y_w_14.append(mvts.index(row[1]))
                                p_w_14.append(id.index(int(row[0])))
                        if row[0] == '15':
                                x1=row[3]
                                x_w_15.append(x1)
                                x2=row[4]
                                x_w_15.append(x2)
                                x3= row[5]
                                x_w_15.append(x3)
                                y_w_15.append(mvts.index(row[1]))
                                p_w_15.append(id.index(int(row[0])))
                        if row[0] == '16':
                                x1=row[3]
                                x_w_16.append(x1)
                                x2=row[4]
                                x_w_16.append(x2)
                                x3= row[5]
                                x_w_16.append(x3)
                                y_w_16.append(mvts.index(row[1]))
                                p_w_16.append(id.index(int(row[0])))
                        if row[0] == '17':
                                x1=row[3]
                                x_w_17.append(x1)
                                x2=row[4]
                                x_w_17.append(x2)
                                x3= row[5]
                                x_w_17.append(x3)
                                y_w_17.append(mvts.index(row[1]))
                                p_w_17.append(id.index(int(row[0])))
                        if row[0] == '18':
                                x1=row[3]
                                x_w_18.append(x1)
                                x2=row[4]
                                x_w_18.append(x2)
                                x3= row[5]
                                x_w_18.append(x3)
                                y_w_18.append(mvts.index(row[1]))
                                p_w_18.append(id.index(int(row[0])))
                        if row[0] == '19':
                                x1=row[3]
                                x_w_19.append(x1)
                                x2=row[4]
                                x_w_19.append(x2)
                                x3= row[5]
                                x_w_19.append(x3)
                                y_w_19.append(mvts.index(row[1]))
                                p_w_19.append(id.index(int(row[0])))
                        if row[0] == '20':
                                x1=row[3]
                                x_w_20.append(x1)
                                x2=row[4]
                                x_w_20.append(x2)
                                x3= row[5]
                                x_w_20.append(x3)
                                y_w_20.append(mvts.index(row[1]))
                                p_w_20.append(id.index(int(row[0])))
                        if row[0] == '21':
                                x1=row[3]
                                x_w_21.append(x1)
                                x2=row[4]
                                x_w_21.append(x2)
                                x3= row[5]
                                x_w_21.append(x3)
                                y_w_21.append(mvts.index(row[1]))
                                p_w_21.append(id.index(int(row[0])))
                        if row[0] == '22':
                                x1=row[3]
                                x_w_22.append(x1)
                                x2=row[4]
                                x_w_22.append(x2)
                                x3= row[5]
                                x_w_22.append(x3)
                                y_w_22.append(mvts.index(row[1]))
                                p_w_22.append(id.index(int(row[0])))
                        if row[0] == '23':
                                x1=row[3]
                                x_w_23.append(x1)
                                x2=row[4]
                                x_w_23.append(x2)
                                x3= row[5]
                                x_w_23.append(x3)
                                y_w_23.append(mvts.index(row[1]))
                                p_w_23.append(id.index(int(row[0])))
                        if row[0] == '24':
                                x1=row[3]
                                x_w_24.append(x1)
                                x2=row[4]
                                x_w_24.append(x2)
                                x3= row[5]
                                x_w_24.append(x3)
                                y_w_24.append(mvts.index(row[1]))
                                p_w_24.append(id.index(int(row[0])))
                        if row[0] == '25':
                                x1=row[3]
                                x_w_25.append(x1)
                                x2=row[4]
                                x_w_25.append(x2)
                                x3= row[5]
                                x_w_25.append(x3)
                                y_w_25.append(mvts.index(row[1]))
                                p_w_25.append(id.index(int(row[0])))
                        if row[0] == '26':
                                x1=row[3]
                                x_w_26.append(x1)
                                x2=row[4]
                                x_w_26.append(x2)
                                x3= row[5]
                                x_w_26.append(x3)
                                y_w_26.append(mvts.index(row[1]))
                                p_w_26.append(id.index(int(row[0])))
                        if row[0] == '27':
                                x1=row[3]
                                x_w_27.append(x1)
                                x2=row[4]
                                x_w_27.append(x2)
                                x3= row[5]
                                x_w_27.append(x3)
                                y_w_27.append(mvts.index(row[1]))
                                p_w_27.append(id.index(int(row[0])))
                        if row[0] == '28':
                                x1=row[3]
                                x_w_28.append(x1)
                                x2=row[4]
                                x_w_28.append(x2)
                                x3= row[5]
                                x_w_28.append(x3)
                                y_w_28.append(mvts.index(row[1]))
                                p_w_28.append(id.index(int(row[0])))
                        if row[0] == '29':
                                x1=row[3]
                                x_w_29.append(x1)
                                x2=row[4]
                                x_w_29.append(x2)
                                x3= row[5]
                                x_w_29.append(x3)
                                y_w_29.append(mvts.index(row[1]))
                                p_w_29.append(id.index(int(row[0])))
                        if row[0] == '30':
                                x1=row[3]
                                x_w_30.append(x1)
                                x2=row[4]
                                x_w_30.append(x2)
                                x3= row[5]
                                x_w_30.append(x3)
                                y_w_30.append(mvts.index(row[1]))
                                p_w_30.append(id.index(int(row[0])))
                        if row[0] == '31':
                                x1=row[3]
                                x_w_31.append(x1)
                                x2=row[4]
                                x_w_31.append(x2)
                                x3= row[5]
                                x_w_31.append(x3)
                                y_w_31.append(mvts.index(row[1]))
                                p_w_31.append(id.index(int(row[0])))
                        if row[0] == '32':
                                x1=row[3]
                                x_w_32.append(x1)
                                x2=row[4]
                                x_w_32.append(x2)
                                x3= row[5]
                                x_w_32.append(x3)
                                y_w_32.append(mvts.index(row[1]))
                                p_w_32.append(id.index(int(row[0])))
                        if row[0] == '33':
                                x1=row[3]
                                x_w_33.append(x1)
                                x2=row[4]
                                x_w_33.append(x2)
                                x3= row[5]
                                x_w_33.append(x3)
                                y_w_33.append(mvts.index(row[1]))
                                p_w_33.append(id.index(int(row[0])))
                        if row[0] == '34':
                                x1=row[3]
                                x_w_34.append(x1)
                                x2=row[4]
                                x_w_34.append(x2)
                                x3= row[5]
                                x_w_34.append(x3)
                                y_w_34.append(mvts.index(row[1]))
                                p_w_34.append(id.index(int(row[0])))
                        if row[0] == '35':
                                x1=row[3]
                                x_w_35.append(x1)
                                x2=row[4]
                                x_w_35.append(x2)
                                x3= row[5]
                                x_w_35.append(x3)
                                y_w_35.append(mvts.index(row[1]))
                                p_w_35.append(id.index(int(row[0])))
                        if row[0] == '36':
                                x1=row[3]
                                x_w_36.append(x1)
                                x2=row[4]
                                x_w_36.append(x2)
                                x3= row[5]
                                x_w_36.append(x3)
                                y_w_36.append(mvts.index(row[1]))
                                p_w_36.append(id.index(int(row[0])))
    

x_d_1= np.array(x_d_1)
y_d_1= np.array(y_d_1)
p_d_1= np.array(p_d_1)
x_d_2= np.array(x_d_2)
y_d_2= np.array(y_d_2)
p_d_2= np.array(p_d_2)
x_d_3= np.array(x_d_3)
y_d_3= np.array(y_d_3)
p_d_3= np.array(p_d_3)
x_d_4= np.array(x_d_4)
y_d_4= np.array(y_d_4)
p_d_4= np.array(p_d_4)
x_d_5= np.array(x_d_5)
y_d_5= np.array(y_d_5)
p_d_5= np.array(p_d_5)
x_d_6= np.array(x_d_6)
y_d_6= np.array(y_d_6)
p_d_6= np.array(p_d_6)
x_d_7= np.array(x_d_7)
y_d_7= np.array(y_d_7)
p_d_7= np.array(p_d_7)
x_d_8= np.array(x_d_8)
y_d_8= np.array(y_d_8)
p_d_8= np.array(p_d_8)
x_d_9= np.array(x_d_9)
y_d_9= np.array(y_d_9)
p_d_9= np.array(p_d_9)
x_d_10= np.array(x_d_10)
y_d_10= np.array(y_d_10)
p_d_10= np.array(p_d_10)
x_d_11= np.array(x_d_11)
y_d_11= np.array(y_d_11)
p_d_11= np.array(p_d_11)
x_d_12= np.array(x_d_12)
y_d_12= np.array(y_d_12)
p_d_12= np.array(p_d_12)
x_d_13= np.array(x_d_13)
y_d_13= np.array(y_d_13)
p_d_13= np.array(p_d_13)
x_d_14= np.array(x_d_14)
y_d_14= np.array(y_d_14)
p_d_14= np.array(p_d_14)
x_d_15= np.array(x_d_15)
y_d_15= np.array(y_d_15)
p_d_15= np.array(p_d_15)
x_d_16= np.array(x_d_16)
y_d_16= np.array(y_d_16)
p_d_16= np.array(p_d_16)
x_d_17= np.array(x_d_17)
y_d_17= np.array(y_d_17)
p_d_17= np.array(p_d_17)
x_d_18= np.array(x_d_18)
y_d_18= np.array(y_d_18)
p_d_18= np.array(p_d_18)
x_d_19= np.array(x_d_19)
y_d_19= np.array(y_d_19)
p_d_19= np.array(p_d_19)
x_d_20= np.array(x_d_20)
y_d_20= np.array(y_d_20)
p_d_20= np.array(p_d_20)
x_d_21= np.array(x_d_21)
y_d_21= np.array(y_d_21)
p_d_21= np.array(p_d_21)
x_d_22= np.array(x_d_22)
y_d_22= np.array(y_d_22)
p_d_22= np.array(p_d_22)
x_d_23= np.array(x_d_23)
y_d_23= np.array(y_d_23)
p_d_23= np.array(p_d_23)
x_d_24= np.array(x_d_24)
y_d_24= np.array(y_d_24)
p_d_24= np.array(p_d_24)
x_d_25= np.array(x_d_25)
y_d_25= np.array(y_d_25)
p_d_25= np.array(p_d_25)
x_d_26= np.array(x_d_26)
y_d_26= np.array(y_d_26)
p_d_26= np.array(p_d_26)
x_d_27= np.array(x_d_27)
y_d_27= np.array(y_d_27)
p_d_27= np.array(p_d_27)
x_d_28= np.array(x_d_28)
y_d_28= np.array(y_d_28)
p_d_28= np.array(p_d_28)
x_d_29= np.array(x_d_29)
y_d_29= np.array(y_d_29)
p_d_29= np.array(p_d_29)
x_d_30= np.array(x_d_30)
y_d_30= np.array(y_d_30)
p_d_30= np.array(p_d_30)
x_d_31= np.array(x_d_31)
y_d_31= np.array(y_d_31)
p_d_31= np.array(p_d_31)
x_d_32= np.array(x_d_32)
y_d_32= np.array(y_d_32)
p_d_32= np.array(p_d_32)
x_d_33= np.array(x_d_33)
y_d_33= np.array(y_d_33)
p_d_33= np.array(p_d_33)
x_d_34= np.array(x_d_34)
y_d_34= np.array(y_d_34)
p_d_34= np.array(p_d_34)
x_d_35= np.array(x_d_35)
y_d_35= np.array(y_d_35)
p_d_35= np.array(p_d_35)
x_d_36= np.array(x_d_36)
y_d_36= np.array(y_d_36)
p_d_36= np.array(p_d_36)
x_j_1= np.array(x_j_1)
y_j_1= np.array(y_j_1)
p_j_1= np.array(p_j_1)
x_j_2= np.array(x_j_2)
y_j_2= np.array(y_j_2)
p_j_2= np.array(p_j_2)
x_j_3= np.array(x_j_3)
y_j_3= np.array(y_j_3)
p_j_3= np.array(p_j_3)
x_j_4= np.array(x_j_4)
y_j_4= np.array(y_j_4)
p_j_4= np.array(p_j_4)
x_j_5= np.array(x_j_5)
y_j_5= np.array(y_j_5)
p_j_5= np.array(p_j_5)
x_j_6= np.array(x_j_6)
y_j_6= np.array(y_j_6)
p_j_6= np.array(p_j_6)
x_j_7= np.array(x_j_7)
y_j_7= np.array(y_j_7)
p_j_7= np.array(p_j_7)
x_j_8= np.array(x_j_8)
y_j_8= np.array(y_j_8)
p_j_8= np.array(p_j_8)
x_j_9= np.array(x_j_9)
y_j_9= np.array(y_j_9)
p_j_9= np.array(p_j_9)
x_j_10= np.array(x_j_10)
y_j_10= np.array(y_j_10)
p_j_10= np.array(p_j_10)
x_j_11= np.array(x_j_11)
y_j_11= np.array(y_j_11)
p_j_11= np.array(p_j_11)
x_j_12= np.array(x_j_12)
y_j_12= np.array(y_j_12)
p_j_12= np.array(p_j_12)
x_j_13= np.array(x_j_13)
y_j_13= np.array(y_j_13)
p_j_13= np.array(p_j_13)
x_j_14= np.array(x_j_14)
y_j_14= np.array(y_j_14)
p_j_14= np.array(p_j_14)
x_j_15= np.array(x_j_15)
y_j_15= np.array(y_j_15)
p_j_15= np.array(p_j_15)
x_j_16= np.array(x_j_16)
y_j_16= np.array(y_j_16)
p_j_16= np.array(p_j_16)
x_j_17= np.array(x_j_17)
y_j_17= np.array(y_j_17)
p_j_17= np.array(p_j_17)
x_j_18= np.array(x_j_18)
y_j_18= np.array(y_j_18)
p_j_18= np.array(p_j_18)
x_j_19= np.array(x_j_19)
y_j_19= np.array(y_j_19)
p_j_19= np.array(p_j_19)
x_j_20= np.array(x_j_20)
y_j_20= np.array(y_j_20)
p_j_20= np.array(p_j_20)
x_j_21= np.array(x_j_21)
y_j_21= np.array(y_j_21)
p_j_21= np.array(p_j_21)
x_j_22= np.array(x_j_22)
y_j_22= np.array(y_j_22)
p_j_22= np.array(p_j_22)
x_j_23= np.array(x_j_23)
y_j_23= np.array(y_j_23)
p_j_23= np.array(p_j_23)
x_j_24= np.array(x_j_24)
y_j_24= np.array(y_j_24)
p_j_24= np.array(p_j_24)
x_j_25= np.array(x_j_25)
y_j_25= np.array(y_j_25)
p_j_25= np.array(p_j_25)
x_j_26= np.array(x_j_26)
y_j_26= np.array(y_j_26)
p_j_26= np.array(p_j_26)
x_j_27= np.array(x_j_27)
y_j_27= np.array(y_j_27)
p_j_27= np.array(p_j_27)
x_j_28= np.array(x_j_28)
y_j_28= np.array(y_j_28)
p_j_28= np.array(p_j_28)
x_j_29= np.array(x_j_29)
y_j_29= np.array(y_j_29)
p_j_29= np.array(p_j_29)
x_j_30= np.array(x_j_30)
y_j_30= np.array(y_j_30)
p_j_30= np.array(p_j_30)
x_j_31= np.array(x_j_31)
y_j_31= np.array(y_j_31)
p_j_31= np.array(p_j_31)
x_j_32= np.array(x_j_32)
y_j_32= np.array(y_j_32)
p_j_32= np.array(p_j_32)
x_j_33= np.array(x_j_33)
y_j_33= np.array(y_j_33)
p_j_33= np.array(p_j_33)
x_j_34= np.array(x_j_34)
y_j_34= np.array(y_j_34)
p_j_34= np.array(p_j_34)
x_j_35= np.array(x_j_35)
y_j_35= np.array(y_j_35)
p_j_35= np.array(p_j_35)
x_j_36= np.array(x_j_36)
y_j_36= np.array(y_j_36)
p_j_36= np.array(p_j_36)
x_s_1= np.array(x_s_1)
y_s_1= np.array(y_s_1)
p_s_1= np.array(p_s_1)
x_s_2= np.array(x_s_2)
y_s_2= np.array(y_s_2)
p_s_2= np.array(p_s_2)
x_s_3= np.array(x_s_3)
y_s_3= np.array(y_s_3)
p_s_3= np.array(p_s_3)
x_s_4= np.array(x_s_4)
y_s_4= np.array(y_s_4)
p_s_4= np.array(p_s_4)
x_s_5= np.array(x_s_5)
y_s_5= np.array(y_s_5)
p_s_5= np.array(p_s_5)
x_s_6= np.array(x_s_6)
y_s_6= np.array(y_s_6)
p_s_6= np.array(p_s_6)
x_s_7= np.array(x_s_7)
y_s_7= np.array(y_s_7)
p_s_7= np.array(p_s_7)
x_s_8= np.array(x_s_8)
y_s_8= np.array(y_s_8)
p_s_8= np.array(p_s_8)
x_s_9= np.array(x_s_9)
y_s_9= np.array(y_s_9)
p_s_9= np.array(p_s_9)
x_s_10= np.array(x_s_10)
y_s_10= np.array(y_s_10)
p_s_10= np.array(p_s_10)
x_s_11= np.array(x_s_11)
y_s_11= np.array(y_s_11)
p_s_11= np.array(p_s_11)
x_s_12= np.array(x_s_12)
y_s_12= np.array(y_s_12)
p_s_12= np.array(p_s_12)
x_s_13= np.array(x_s_13)
y_s_13= np.array(y_s_13)
p_s_13= np.array(p_s_13)
x_s_14= np.array(x_s_14)
y_s_14= np.array(y_s_14)
p_s_14= np.array(p_s_14)
x_s_15= np.array(x_s_15)
y_s_15= np.array(y_s_15)
p_s_15= np.array(p_s_15)
x_s_16= np.array(x_s_16)
y_s_16= np.array(y_s_16)
p_s_16= np.array(p_s_16)
x_s_17= np.array(x_s_17)
y_s_17= np.array(y_s_17)
p_s_17= np.array(p_s_17)
x_s_18= np.array(x_s_18)
y_s_18= np.array(y_s_18)
p_s_18= np.array(p_s_18)
x_s_19= np.array(x_s_19)
y_s_19= np.array(y_s_19)
p_s_19= np.array(p_s_19)
x_s_20= np.array(x_s_20)
y_s_20= np.array(y_s_20)
p_s_20= np.array(p_s_20)
x_s_21= np.array(x_s_21)
y_s_21= np.array(y_s_21)
p_s_21= np.array(p_s_21)
x_s_22= np.array(x_s_22)
y_s_22= np.array(y_s_22)
p_s_22= np.array(p_s_22)
x_s_23= np.array(x_s_23)
y_s_23= np.array(y_s_23)
p_s_23= np.array(p_s_23)
x_s_24= np.array(x_s_24)
y_s_24= np.array(y_s_24)
p_s_24= np.array(p_s_24)
x_s_25= np.array(x_s_25)
y_s_25= np.array(y_s_25)
p_s_25= np.array(p_s_25)
x_s_26= np.array(x_s_26)
y_s_26= np.array(y_s_26)
p_s_26= np.array(p_s_26)
x_s_27= np.array(x_s_27)
y_s_27= np.array(y_s_27)
p_s_27= np.array(p_s_27)
x_s_28= np.array(x_s_28)
y_s_28= np.array(y_s_28)
p_s_28= np.array(p_s_28)
x_s_29= np.array(x_s_29)
y_s_29= np.array(y_s_29)
p_s_29= np.array(p_s_29)
x_s_30= np.array(x_s_30)
y_s_30= np.array(y_s_30)
p_s_30= np.array(p_s_30)
x_s_31= np.array(x_s_31)
y_s_31= np.array(y_s_31)
p_s_31= np.array(p_s_31)
x_s_32= np.array(x_s_32)
y_s_32= np.array(y_s_32)
p_s_32= np.array(p_s_32)
x_s_33= np.array(x_s_33)
y_s_33= np.array(y_s_33)
p_s_33= np.array(p_s_33)
x_s_34= np.array(x_s_34)
y_s_34= np.array(y_s_34)
p_s_34= np.array(p_s_34)
x_s_35= np.array(x_s_35)
y_s_35= np.array(y_s_35)
p_s_35= np.array(p_s_35)
x_s_36= np.array(x_s_36)
y_s_36= np.array(y_s_36)
p_s_36= np.array(p_s_36)
x_st_1= np.array(x_st_1)
y_st_1= np.array(y_st_1)
p_st_1= np.array(p_st_1)
x_st_2= np.array(x_st_2)
y_st_2= np.array(y_st_2)
p_st_2= np.array(p_st_2)
x_st_3= np.array(x_st_3)
y_st_3= np.array(y_st_3)
p_st_3= np.array(p_st_3)
x_st_4= np.array(x_st_4)
y_st_4= np.array(y_st_4)
p_st_4= np.array(p_st_4)
x_st_5= np.array(x_st_5)
y_st_5= np.array(y_st_5)
p_st_5= np.array(p_st_5)
x_st_6= np.array(x_st_6)
y_st_6= np.array(y_st_6)
p_st_6= np.array(p_st_6)
x_st_7= np.array(x_st_7)
y_st_7= np.array(y_st_7)
p_st_7= np.array(p_st_7)
x_st_8= np.array(x_st_8)
y_st_8= np.array(y_st_8)
p_st_8= np.array(p_st_8)
x_st_9= np.array(x_st_9)
y_st_9= np.array(y_st_9)
p_st_9= np.array(p_st_9)
x_st_10= np.array(x_st_10)
y_st_10= np.array(y_st_10)
p_st_10= np.array(p_st_10)
x_st_11= np.array(x_st_11)
y_st_11= np.array(y_st_11)
p_st_11= np.array(p_st_11)
x_st_12= np.array(x_st_12)
y_st_12= np.array(y_st_12)
p_st_12= np.array(p_st_12)
x_st_13= np.array(x_st_13)
y_st_13= np.array(y_st_13)
p_st_13= np.array(p_st_13)
x_st_14= np.array(x_st_14)
y_st_14= np.array(y_st_14)
p_st_14= np.array(p_st_14)
x_st_15= np.array(x_st_15)
y_st_15= np.array(y_st_15)
p_st_15= np.array(p_st_15)
x_st_16= np.array(x_st_16)
y_st_16= np.array(y_st_16)
p_st_16= np.array(p_st_16)
x_st_17= np.array(x_st_17)
y_st_17= np.array(y_st_17)
p_st_17= np.array(p_st_17)
x_st_18= np.array(x_st_18)
y_st_18= np.array(y_st_18)
p_st_18= np.array(p_st_18)
x_st_19= np.array(x_st_19)
y_st_19= np.array(y_st_19)
p_st_19= np.array(p_st_19)
x_st_20= np.array(x_st_20)
y_st_20= np.array(y_st_20)
p_st_20= np.array(p_st_20)
x_st_21= np.array(x_st_21)
y_st_21= np.array(y_st_21)
p_st_21= np.array(p_st_21)
x_st_22= np.array(x_st_22)
y_st_22= np.array(y_st_22)
p_st_22= np.array(p_st_22)
x_st_23= np.array(x_st_23)
y_st_23= np.array(y_st_23)
p_st_23= np.array(p_st_23)
x_st_24= np.array(x_st_24)
y_st_24= np.array(y_st_24)
p_st_24= np.array(p_st_24)
x_st_25= np.array(x_st_25)
y_st_25= np.array(y_st_25)
p_st_25= np.array(p_st_25)
x_st_26= np.array(x_st_26)
y_st_26= np.array(y_st_26)
p_st_26= np.array(p_st_26)
x_st_27= np.array(x_st_27)
y_st_27= np.array(y_st_27)
p_st_27= np.array(p_st_27)
x_st_28= np.array(x_st_28)
y_st_28= np.array(y_st_28)
p_st_28= np.array(p_st_28)
x_st_29= np.array(x_st_29)
y_st_29= np.array(y_st_29)
p_st_29= np.array(p_st_29)
x_st_30= np.array(x_st_30)
y_st_30= np.array(y_st_30)
p_st_30= np.array(p_st_30)
x_st_31= np.array(x_st_31)
y_st_31= np.array(y_st_31)
p_st_31= np.array(p_st_31)
x_st_32= np.array(x_st_32)
y_st_32= np.array(y_st_32)
p_st_32= np.array(p_st_32)
x_st_33= np.array(x_st_33)
y_st_33= np.array(y_st_33)
p_st_33= np.array(p_st_33)
x_st_34= np.array(x_st_34)
y_st_34= np.array(y_st_34)
p_st_34= np.array(p_st_34)
x_st_35= np.array(x_st_35)
y_st_35= np.array(y_st_35)
p_st_35= np.array(p_st_35)
x_st_36= np.array(x_st_36)
y_st_36= np.array(y_st_36)
p_st_36= np.array(p_st_36)
x_u_1= np.array(x_u_1)
y_u_1= np.array(y_u_1)
p_u_1= np.array(p_u_1)
x_u_2= np.array(x_u_2)
y_u_2= np.array(y_u_2)
p_u_2= np.array(p_u_2)
x_u_3= np.array(x_u_3)
y_u_3= np.array(y_u_3)
p_u_3= np.array(p_u_3)
x_u_4= np.array(x_u_4)
y_u_4= np.array(y_u_4)
p_u_4= np.array(p_u_4)
x_u_5= np.array(x_u_5)
y_u_5= np.array(y_u_5)
p_u_5= np.array(p_u_5)
x_u_6= np.array(x_u_6)
y_u_6= np.array(y_u_6)
p_u_6= np.array(p_u_6)
x_u_7= np.array(x_u_7)
y_u_7= np.array(y_u_7)
p_u_7= np.array(p_u_7)
x_u_8= np.array(x_u_8)
y_u_8= np.array(y_u_8)
p_u_8= np.array(p_u_8)
x_u_9= np.array(x_u_9)
y_u_9= np.array(y_u_9)
p_u_9= np.array(p_u_9)
x_u_10= np.array(x_u_10)
y_u_10= np.array(y_u_10)
p_u_10= np.array(p_u_10)
x_u_11= np.array(x_u_11)
y_u_11= np.array(y_u_11)
p_u_11= np.array(p_u_11)
x_u_12= np.array(x_u_12)
y_u_12= np.array(y_u_12)
p_u_12= np.array(p_u_12)
x_u_13= np.array(x_u_13)
y_u_13= np.array(y_u_13)
p_u_13= np.array(p_u_13)
x_u_14= np.array(x_u_14)
y_u_14= np.array(y_u_14)
p_u_14= np.array(p_u_14)
x_u_15= np.array(x_u_15)
y_u_15= np.array(y_u_15)
p_u_15= np.array(p_u_15)
x_u_16= np.array(x_u_16)
y_u_16= np.array(y_u_16)
p_u_16= np.array(p_u_16)
x_u_17= np.array(x_u_17)
y_u_17= np.array(y_u_17)
p_u_17= np.array(p_u_17)
x_u_18= np.array(x_u_18)
y_u_18= np.array(y_u_18)
p_u_18= np.array(p_u_18)
x_u_19= np.array(x_u_19)
y_u_19= np.array(y_u_19)
p_u_19= np.array(p_u_19)
x_u_20= np.array(x_u_20)
y_u_20= np.array(y_u_20)
p_u_20= np.array(p_u_20)
x_u_21= np.array(x_u_21)
y_u_21= np.array(y_u_21)
p_u_21= np.array(p_u_21)
x_u_22= np.array(x_u_22)
y_u_22= np.array(y_u_22)
p_u_22= np.array(p_u_22)
x_u_23= np.array(x_u_23)
y_u_23= np.array(y_u_23)
p_u_23= np.array(p_u_23)
x_u_24= np.array(x_u_24)
y_u_24= np.array(y_u_24)
p_u_24= np.array(p_u_24)
x_u_25= np.array(x_u_25)
y_u_25= np.array(y_u_25)
p_u_25= np.array(p_u_25)
x_u_26= np.array(x_u_26)
y_u_26= np.array(y_u_26)
p_u_26= np.array(p_u_26)
x_u_27= np.array(x_u_27)
y_u_27= np.array(y_u_27)
p_u_27= np.array(p_u_27)
x_u_28= np.array(x_u_28)
y_u_28= np.array(y_u_28)
p_u_28= np.array(p_u_28)
x_u_29= np.array(x_u_29)
y_u_29= np.array(y_u_29)
p_u_29= np.array(p_u_29)
x_u_30= np.array(x_u_30)
y_u_30= np.array(y_u_30)
p_u_30= np.array(p_u_30)
x_u_31= np.array(x_u_31)
y_u_31= np.array(y_u_31)
p_u_31= np.array(p_u_31)
x_u_32= np.array(x_u_32)
y_u_32= np.array(y_u_32)
p_u_32= np.array(p_u_32)
x_u_33= np.array(x_u_33)
y_u_33= np.array(y_u_33)
p_u_33= np.array(p_u_33)
x_u_34= np.array(x_u_34)
y_u_34= np.array(y_u_34)
p_u_34= np.array(p_u_34)
x_u_35= np.array(x_u_35)
y_u_35= np.array(y_u_35)
p_u_35= np.array(p_u_35)
x_u_36= np.array(x_u_36)
y_u_36= np.array(y_u_36)
p_u_36= np.array(p_u_36)
x_w_1= np.array(x_w_1)
y_w_1= np.array(y_w_1)
p_w_1= np.array(p_w_1)
x_w_2= np.array(x_w_2)
y_w_2= np.array(y_w_2)
p_w_2= np.array(p_w_2)
x_w_3= np.array(x_w_3)
y_w_3= np.array(y_w_3)
p_w_3= np.array(p_w_3)
x_w_4= np.array(x_w_4)
y_w_4= np.array(y_w_4)
p_w_4= np.array(p_w_4)
x_w_5= np.array(x_w_5)
y_w_5= np.array(y_w_5)
p_w_5= np.array(p_w_5)
x_w_6= np.array(x_w_6)
y_w_6= np.array(y_w_6)
p_w_6= np.array(p_w_6)
x_w_7= np.array(x_w_7)
y_w_7= np.array(y_w_7)
p_w_7= np.array(p_w_7)
x_w_8= np.array(x_w_8)
y_w_8= np.array(y_w_8)
p_w_8= np.array(p_w_8)
x_w_9= np.array(x_w_9)
y_w_9= np.array(y_w_9)
p_w_9= np.array(p_w_9)
x_w_10= np.array(x_w_10)
y_w_10= np.array(y_w_10)
p_w_10= np.array(p_w_10)
x_w_11= np.array(x_w_11)
y_w_11= np.array(y_w_11)
p_w_11= np.array(p_w_11)
x_w_12= np.array(x_w_12)
y_w_12= np.array(y_w_12)
p_w_12= np.array(p_w_12)
x_w_13= np.array(x_w_13)
y_w_13= np.array(y_w_13)
p_w_13= np.array(p_w_13)
x_w_14= np.array(x_w_14)
y_w_14= np.array(y_w_14)
p_w_14= np.array(p_w_14)
x_w_15= np.array(x_w_15)
y_w_15= np.array(y_w_15)
p_w_15= np.array(p_w_15)
x_w_16= np.array(x_w_16)
y_w_16= np.array(y_w_16)
p_w_16= np.array(p_w_16)
x_w_17= np.array(x_w_17)
y_w_17= np.array(y_w_17)
p_w_17= np.array(p_w_17)
x_w_18= np.array(x_w_18)
y_w_18= np.array(y_w_18)
p_w_18= np.array(p_w_18)
x_w_19= np.array(x_w_19)
y_w_19= np.array(y_w_19)
p_w_19= np.array(p_w_19)
x_w_20= np.array(x_w_20)
y_w_20= np.array(y_w_20)
p_w_20= np.array(p_w_20)
x_w_21= np.array(x_w_21)
y_w_21= np.array(y_w_21)
p_w_21= np.array(p_w_21)
x_w_22= np.array(x_w_22)
y_w_22= np.array(y_w_22)
p_w_22= np.array(p_w_22)
x_w_23= np.array(x_w_23)
y_w_23= np.array(y_w_23)
p_w_23= np.array(p_w_23)
x_w_24= np.array(x_w_24)
y_w_24= np.array(y_w_24)
p_w_24= np.array(p_w_24)
x_w_25= np.array(x_w_25)
y_w_25= np.array(y_w_25)
p_w_25= np.array(p_w_25)
x_w_26= np.array(x_w_26)
y_w_26= np.array(y_w_26)
p_w_26= np.array(p_w_26)
x_w_27= np.array(x_w_27)
y_w_27= np.array(y_w_27)
p_w_27= np.array(p_w_27)
x_w_28= np.array(x_w_28)
y_w_28= np.array(y_w_28)
p_w_28= np.array(p_w_28)
x_w_29= np.array(x_w_29)
y_w_29= np.array(y_w_29)
p_w_29= np.array(p_w_29)
x_w_30= np.array(x_w_30)
y_w_30= np.array(y_w_30)
p_w_30= np.array(p_w_30)
x_w_31= np.array(x_w_31)
y_w_31= np.array(y_w_31)
p_w_31= np.array(p_w_31)
x_w_32= np.array(x_w_32)
y_w_32= np.array(y_w_32)
p_w_32= np.array(p_w_32)
x_w_33= np.array(x_w_33)
y_w_33= np.array(y_w_33)
p_w_33= np.array(p_w_33)
x_w_34= np.array(x_w_34)
y_w_34= np.array(y_w_34)
p_w_34= np.array(p_w_34)
x_w_35= np.array(x_w_35)
y_w_35= np.array(y_w_35)
p_w_35= np.array(p_w_35)
x_w_36= np.array(x_w_36)
y_w_36= np.array(y_w_36)
p_w_36= np.array(p_w_36)
    
start = 0

shape_y = y_d_1.shape[0]
shape_p = p_d_1.shape[0]
shp=(x_d_1.shape)[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_d_1 =   np.delete(x_d_1, [k for k in range(nece_shp,shp)], None)
x_d_1 = x_d_1.reshape(-1,sensors)
shape_x = x_d_1.shape[0]
x_d_1 = np.reshape(x_d_1, (-1,segement_time_size, np.shape(x_d_1)[1]))
y_d_1= np.delete( y_d_1, [k for k in range(x_d_1.shape[0],shape_y)], None)
p_d_1 = np.delete(p_d_1,[k for k in range(x_d_1.shape[0],shape_p)], None)
if start==0:
    all_seqs = x_d_1
    y = y_d_1
    p = p_d_1
    start=1
else:
    all_seqs = np.concatenate((all_seqs,x_d_1), axis=0)
    y = np.concatenate((y,y_d_1),axis=0)
    p = np.concatenate((p,p_d_1), axis=0)
shape_y = y_d_2.shape[0]
shape_p = p_d_2.shape[0]
shp=(x_d_2.shape)[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_d_2 =   np.delete(x_d_2, [k for k in range(nece_shp,shp)], None)
x_d_2 = x_d_2.reshape(-1,sensors)
shape_x = x_d_2.shape[0]
x_d_2 = np.reshape(x_d_2, (-1,segement_time_size, np.shape(x_d_2)[1]))
y_d_2= np.delete( y_d_2, [k for k in range(x_d_2.shape[0],shape_y)], None)
p_d_2 = np.delete(p_d_2,[k for k in range(x_d_2.shape[0],shape_p)], None)
if start==0:
    all_seqs = x_d_2
    y = y_d_2
    p = p_d_2
    start=1
else:
    all_seqs = np.concatenate((all_seqs,x_d_2), axis=0)
    y = np.concatenate((y,y_d_2),axis=0)
    p = np.concatenate((p,p_d_2), axis=0)
shape_y = y_d_3.shape[0]
shape_p = p_d_3.shape[0]
shp=(x_d_3.shape)[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_d_3 =   np.delete(x_d_3, [k for k in range(nece_shp,shp)], None)
x_d_3 = x_d_3.reshape(-1,sensors)
shape_x = x_d_3.shape[0]
x_d_3 = np.reshape(x_d_3, (-1,segement_time_size, np.shape(x_d_3)[1]))
y_d_3= np.delete( y_d_3, [k for k in range(x_d_3.shape[0],shape_y)], None)
p_d_3 = np.delete(p_d_3,[k for k in range(x_d_3.shape[0],shape_p)], None)
if start==0:
    all_seqs = x_d_3
    y = y_d_3
    p = p_d_3
    start=1
else:
    all_seqs = np.concatenate((all_seqs,x_d_3), axis=0)
    y = np.concatenate((y,y_d_3),axis=0)
    p = np.concatenate((p,p_d_3), axis=0)
shape_y = y_d_4.shape[0]
shape_p = p_d_4.shape[0]
shp=(x_d_4.shape)[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_d_4 =   np.delete(x_d_4, [k for k in range(nece_shp,shp)], None)
x_d_4 = x_d_4.reshape(-1,sensors)
shape_x = x_d_4.shape[0]
x_d_4 = np.reshape(x_d_4, (-1,segement_time_size, np.shape(x_d_4)[1]))
y_d_4= np.delete( y_d_4, [k for k in range(x_d_4.shape[0],shape_y)], None)
p_d_4 = np.delete(p_d_4,[k for k in range(x_d_4.shape[0],shape_p)], None)
if start==0:
    all_seqs = x_d_4
    y = y_d_4
    p = p_d_4
    start=1
else:
    all_seqs = np.concatenate((all_seqs,x_d_4), axis=0)
    y = np.concatenate((y,y_d_4),axis=0)
    p = np.concatenate((p,p_d_4), axis=0)
shape_y = y_d_5.shape[0]
shape_p = p_d_5.shape[0]
shp=(x_d_5.shape)[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_d_5 =   np.delete(x_d_5, [k for k in range(nece_shp,shp)], None)
x_d_5 = x_d_5.reshape(-1,sensors)
shape_x = x_d_5.shape[0]
x_d_5 = np.reshape(x_d_5, (-1,segement_time_size, np.shape(x_d_5)[1]))
y_d_5= np.delete( y_d_5, [k for k in range(x_d_5.shape[0],shape_y)], None)
p_d_5 = np.delete(p_d_5,[k for k in range(x_d_5.shape[0],shape_p)], None)
if start==0:
    all_seqs = x_d_5
    y = y_d_5
    p = p_d_5
    start=1
else:
    all_seqs = np.concatenate((all_seqs,x_d_5), axis=0)
    y = np.concatenate((y,y_d_5),axis=0)
    p = np.concatenate((p,p_d_5), axis=0)
shape_y = y_d_6.shape[0]
shape_p = p_d_6.shape[0]
shp=(x_d_6.shape)[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_d_6 =   np.delete(x_d_6, [k for k in range(nece_shp,shp)], None)
x_d_6 = x_d_6.reshape(-1,sensors)
shape_x = x_d_6.shape[0]
x_d_6 = np.reshape(x_d_6, (-1,segement_time_size, np.shape(x_d_6)[1]))
y_d_6= np.delete( y_d_6, [k for k in range(x_d_6.shape[0],shape_y)], None)
p_d_6 = np.delete(p_d_6,[k for k in range(x_d_6.shape[0],shape_p)], None)
if start==0:
    all_seqs = x_d_6
    y = y_d_6
    p = p_d_6
    start=1
else:
    all_seqs = np.concatenate((all_seqs,x_d_6), axis=0)
    y = np.concatenate((y,y_d_6),axis=0)
    p = np.concatenate((p,p_d_6), axis=0)
shape_y = y_d_7.shape[0]
shape_p = p_d_7.shape[0]
shp=(x_d_7.shape)[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_d_7 =   np.delete(x_d_7, [k for k in range(nece_shp,shp)], None)
x_d_7 = x_d_7.reshape(-1,sensors)
shape_x = x_d_7.shape[0]
x_d_7 = np.reshape(x_d_7, (-1,segement_time_size, np.shape(x_d_7)[1]))
y_d_7= np.delete( y_d_7, [k for k in range(x_d_7.shape[0],shape_y)], None)
p_d_7 = np.delete(p_d_7,[k for k in range(x_d_7.shape[0],shape_p)], None)
if start==0:
    all_seqs = x_d_7
    y = y_d_7
    p = p_d_7
    start=1
else:
    all_seqs = np.concatenate((all_seqs,x_d_7), axis=0)
    y = np.concatenate((y,y_d_7),axis=0)
    p = np.concatenate((p,p_d_7), axis=0)
shape_y = y_d_8.shape[0]
shape_p = p_d_8.shape[0]
shp=(x_d_8.shape)[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_d_8 =   np.delete(x_d_8, [k for k in range(nece_shp,shp)], None)
x_d_8 = x_d_8.reshape(-1,sensors)
shape_x = x_d_8.shape[0]
x_d_8 = np.reshape(x_d_8, (-1,segement_time_size, np.shape(x_d_8)[1]))
y_d_8= np.delete( y_d_8, [k for k in range(x_d_8.shape[0],shape_y)], None)
p_d_8 = np.delete(p_d_8,[k for k in range(x_d_8.shape[0],shape_p)], None)
if start==0:
    all_seqs = x_d_8
    y = y_d_8
    p = p_d_8
    start=1
else:
    all_seqs = np.concatenate((all_seqs,x_d_8), axis=0)
    y = np.concatenate((y,y_d_8),axis=0)
    p = np.concatenate((p,p_d_8), axis=0)
shape_y = y_d_9.shape[0]
shape_p = p_d_9.shape[0]
shp=(x_d_9.shape)[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_d_9 =   np.delete(x_d_9, [k for k in range(nece_shp,shp)], None)
x_d_9 = x_d_9.reshape(-1,sensors)
shape_x = x_d_9.shape[0]
x_d_9 = np.reshape(x_d_9, (-1,segement_time_size, np.shape(x_d_9)[1]))
y_d_9= np.delete( y_d_9, [k for k in range(x_d_9.shape[0],shape_y)], None)
p_d_9 = np.delete(p_d_9,[k for k in range(x_d_9.shape[0],shape_p)], None)
if start==0:
    all_seqs = x_d_9
    y = y_d_9
    p = p_d_9
    start=1
else:
    all_seqs = np.concatenate((all_seqs,x_d_9), axis=0)
    y = np.concatenate((y,y_d_9),axis=0)
    p = np.concatenate((p,p_d_9), axis=0)
shape_y = y_d_10.shape[0]
shape_p = p_d_10.shape[0]
shp=(x_d_10.shape)[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_d_10 =   np.delete(x_d_10, [k for k in range(nece_shp,shp)], None)
x_d_10 = x_d_10.reshape(-1,sensors)
shape_x = x_d_10.shape[0]
x_d_10 = np.reshape(x_d_10, (-1,segement_time_size, np.shape(x_d_10)[1]))
y_d_10= np.delete( y_d_10, [k for k in range(x_d_10.shape[0],shape_y)], None)
p_d_10 = np.delete(p_d_10,[k for k in range(x_d_10.shape[0],shape_p)], None)
if start==0:
    all_seqs = x_d_10
    y = y_d_10
    p = p_d_10
    start=1
else:
    all_seqs = np.concatenate((all_seqs,x_d_10), axis=0)
    y = np.concatenate((y,y_d_10),axis=0)
    p = np.concatenate((p,p_d_10), axis=0)
shape_y = y_d_11.shape[0]
shape_p = p_d_11.shape[0]
shp=(x_d_11.shape)[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_d_11 =   np.delete(x_d_11, [k for k in range(nece_shp,shp)], None)
x_d_11 = x_d_11.reshape(-1,sensors)
shape_x = x_d_11.shape[0]
x_d_11 = np.reshape(x_d_11, (-1,segement_time_size, np.shape(x_d_11)[1]))
y_d_11= np.delete( y_d_11, [k for k in range(x_d_11.shape[0],shape_y)], None)
p_d_11 = np.delete(p_d_11,[k for k in range(x_d_11.shape[0],shape_p)], None)
if start==0:
    all_seqs = x_d_11
    y = y_d_11
    p = p_d_11
    start=1
else:
    all_seqs = np.concatenate((all_seqs,x_d_11), axis=0)
    y = np.concatenate((y,y_d_11),axis=0)
    p = np.concatenate((p,p_d_11), axis=0)
shape_y = y_d_12.shape[0]
shape_p = p_d_12.shape[0]
shp=(x_d_12.shape)[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_d_12 =   np.delete(x_d_12, [k for k in range(nece_shp,shp)], None)
x_d_12 = x_d_12.reshape(-1,sensors)
shape_x = x_d_12.shape[0]
x_d_12 = np.reshape(x_d_12, (-1,segement_time_size, np.shape(x_d_12)[1]))
y_d_12= np.delete( y_d_12, [k for k in range(x_d_12.shape[0],shape_y)], None)
p_d_12 = np.delete(p_d_12,[k for k in range(x_d_12.shape[0],shape_p)], None)
if start==0:
    all_seqs = x_d_12
    y = y_d_12
    p = p_d_12
    start=1
else:
    all_seqs = np.concatenate((all_seqs,x_d_12), axis=0)
    y = np.concatenate((y,y_d_12),axis=0)
    p = np.concatenate((p,p_d_12), axis=0)
shape_y = y_d_13.shape[0]
shape_p = p_d_13.shape[0]
shp=(x_d_13.shape)[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_d_13 =   np.delete(x_d_13, [k for k in range(nece_shp,shp)], None)
x_d_13 = x_d_13.reshape(-1,sensors)
shape_x = x_d_13.shape[0]
x_d_13 = np.reshape(x_d_13, (-1,segement_time_size, np.shape(x_d_13)[1]))
y_d_13= np.delete( y_d_13, [k for k in range(x_d_13.shape[0],shape_y)], None)
p_d_13 = np.delete(p_d_13,[k for k in range(x_d_13.shape[0],shape_p)], None)
if start==0:
    all_seqs = x_d_13
    y = y_d_13
    p = p_d_13
    start=1
else:
    all_seqs = np.concatenate((all_seqs,x_d_13), axis=0)
    y = np.concatenate((y,y_d_13),axis=0)
    p = np.concatenate((p,p_d_13), axis=0)
shape_y = y_d_14.shape[0]
shape_p = p_d_14.shape[0]
shp=(x_d_14.shape)[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_d_14 =   np.delete(x_d_14, [k for k in range(nece_shp,shp)], None)
x_d_14 = x_d_14.reshape(-1,sensors)
shape_x = x_d_14.shape[0]
x_d_14 = np.reshape(x_d_14, (-1,segement_time_size, np.shape(x_d_14)[1]))
y_d_14= np.delete( y_d_14, [k for k in range(x_d_14.shape[0],shape_y)], None)
p_d_14 = np.delete(p_d_14,[k for k in range(x_d_14.shape[0],shape_p)], None)
if start==0:
    all_seqs = x_d_14
    y = y_d_14
    p = p_d_14
    start=1
else:
    all_seqs = np.concatenate((all_seqs,x_d_14), axis=0)
    y = np.concatenate((y,y_d_14),axis=0)
    p = np.concatenate((p,p_d_14), axis=0)
shape_y = y_d_15.shape[0]
shape_p = p_d_15.shape[0]
shp=(x_d_15.shape)[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_d_15 =   np.delete(x_d_15, [k for k in range(nece_shp,shp)], None)
x_d_15 = x_d_15.reshape(-1,sensors)
shape_x = x_d_15.shape[0]
x_d_15 = np.reshape(x_d_15, (-1,segement_time_size, np.shape(x_d_15)[1]))
y_d_15= np.delete( y_d_15, [k for k in range(x_d_15.shape[0],shape_y)], None)
p_d_15 = np.delete(p_d_15,[k for k in range(x_d_15.shape[0],shape_p)], None)
if start==0:
    all_seqs = x_d_15
    y = y_d_15
    p = p_d_15
    start=1
else:
    all_seqs = np.concatenate((all_seqs,x_d_15), axis=0)
    y = np.concatenate((y,y_d_15),axis=0)
    p = np.concatenate((p,p_d_15), axis=0)
shape_y = y_d_16.shape[0]
shape_p = p_d_16.shape[0]
shp=(x_d_16.shape)[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_d_16 =   np.delete(x_d_16, [k for k in range(nece_shp,shp)], None)
x_d_16 = x_d_16.reshape(-1,sensors)
shape_x = x_d_16.shape[0]
x_d_16 = np.reshape(x_d_16, (-1,segement_time_size, np.shape(x_d_16)[1]))
y_d_16= np.delete( y_d_16, [k for k in range(x_d_16.shape[0],shape_y)], None)
p_d_16 = np.delete(p_d_16,[k for k in range(x_d_16.shape[0],shape_p)], None)
if start==0:
    all_seqs = x_d_16
    y = y_d_16
    p = p_d_16
    start=1
else:
    all_seqs = np.concatenate((all_seqs,x_d_16), axis=0)
    y = np.concatenate((y,y_d_16),axis=0)
    p = np.concatenate((p,p_d_16), axis=0)
shape_y = y_d_17.shape[0]
shape_p = p_d_17.shape[0]
shp=(x_d_17.shape)[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_d_17 =   np.delete(x_d_17, [k for k in range(nece_shp,shp)], None)
x_d_17 = x_d_17.reshape(-1,sensors)
shape_x = x_d_17.shape[0]
x_d_17 = np.reshape(x_d_17, (-1,segement_time_size, np.shape(x_d_17)[1]))
y_d_17= np.delete( y_d_17, [k for k in range(x_d_17.shape[0],shape_y)], None)
p_d_17 = np.delete(p_d_17,[k for k in range(x_d_17.shape[0],shape_p)], None)
if start==0:
    all_seqs = x_d_17
    y = y_d_17
    p = p_d_17
    start=1
else:
    all_seqs = np.concatenate((all_seqs,x_d_17), axis=0)
    y = np.concatenate((y,y_d_17),axis=0)
    p = np.concatenate((p,p_d_17), axis=0)
shape_y = y_d_18.shape[0]
shape_p = p_d_18.shape[0]
shp=(x_d_18.shape)[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_d_18 =   np.delete(x_d_18, [k for k in range(nece_shp,shp)], None)
x_d_18 = x_d_18.reshape(-1,sensors)
shape_x = x_d_18.shape[0]
x_d_18 = np.reshape(x_d_18, (-1,segement_time_size, np.shape(x_d_18)[1]))
y_d_18= np.delete( y_d_18, [k for k in range(x_d_18.shape[0],shape_y)], None)
p_d_18 = np.delete(p_d_18,[k for k in range(x_d_18.shape[0],shape_p)], None)
if start==0:
    all_seqs = x_d_18
    y = y_d_18
    p = p_d_18
    start=1
else:
    all_seqs = np.concatenate((all_seqs,x_d_18), axis=0)
    y = np.concatenate((y,y_d_18),axis=0)
    p = np.concatenate((p,p_d_18), axis=0)
shape_y = y_d_19.shape[0]
shape_p = p_d_19.shape[0]
shp=(x_d_19.shape)[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_d_19 =   np.delete(x_d_19, [k for k in range(nece_shp,shp)], None)
x_d_19 = x_d_19.reshape(-1,sensors)
shape_x = x_d_19.shape[0]
x_d_19 = np.reshape(x_d_19, (-1,segement_time_size, np.shape(x_d_19)[1]))
y_d_19= np.delete( y_d_19, [k for k in range(x_d_19.shape[0],shape_y)], None)
p_d_19 = np.delete(p_d_19,[k for k in range(x_d_19.shape[0],shape_p)], None)
if start==0:
    all_seqs = x_d_19
    y = y_d_19
    p = p_d_19
    start=1
else:
    all_seqs = np.concatenate((all_seqs,x_d_19), axis=0)
    y = np.concatenate((y,y_d_19),axis=0)
    p = np.concatenate((p,p_d_19), axis=0)
shape_y = y_d_20.shape[0]
shape_p = p_d_20.shape[0]
shp=(x_d_20.shape)[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_d_20 =   np.delete(x_d_20, [k for k in range(nece_shp,shp)], None)
x_d_20 = x_d_20.reshape(-1,sensors)
shape_x = x_d_20.shape[0]
x_d_20 = np.reshape(x_d_20, (-1,segement_time_size, np.shape(x_d_20)[1]))
y_d_20= np.delete( y_d_20, [k for k in range(x_d_20.shape[0],shape_y)], None)
p_d_20 = np.delete(p_d_20,[k for k in range(x_d_20.shape[0],shape_p)], None)
if start==0:
    all_seqs = x_d_20
    y = y_d_20
    p = p_d_20
    start=1
else:
    all_seqs = np.concatenate((all_seqs,x_d_20), axis=0)
    y = np.concatenate((y,y_d_20),axis=0)
    p = np.concatenate((p,p_d_20), axis=0)
shape_y = y_d_21.shape[0]
shape_p = p_d_21.shape[0]
shp=(x_d_21.shape)[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_d_21 =   np.delete(x_d_21, [k for k in range(nece_shp,shp)], None)
x_d_21 = x_d_21.reshape(-1,sensors)
shape_x = x_d_21.shape[0]
x_d_21 = np.reshape(x_d_21, (-1,segement_time_size, np.shape(x_d_21)[1]))
y_d_21= np.delete( y_d_21, [k for k in range(x_d_21.shape[0],shape_y)], None)
p_d_21 = np.delete(p_d_21,[k for k in range(x_d_21.shape[0],shape_p)], None)
if start==0:
    all_seqs = x_d_21
    y = y_d_21
    p = p_d_21
    start=1
else:
    all_seqs = np.concatenate((all_seqs,x_d_21), axis=0)
    y = np.concatenate((y,y_d_21),axis=0)
    p = np.concatenate((p,p_d_21), axis=0)
shape_y = y_d_22.shape[0]
shape_p = p_d_22.shape[0]
shp=(x_d_22.shape)[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_d_22 =   np.delete(x_d_22, [k for k in range(nece_shp,shp)], None)
x_d_22 = x_d_22.reshape(-1,sensors)
shape_x = x_d_22.shape[0]
x_d_22 = np.reshape(x_d_22, (-1,segement_time_size, np.shape(x_d_22)[1]))
y_d_22= np.delete( y_d_22, [k for k in range(x_d_22.shape[0],shape_y)], None)
p_d_22 = np.delete(p_d_22,[k for k in range(x_d_22.shape[0],shape_p)], None)
if start==0:
    all_seqs = x_d_22
    y = y_d_22
    p = p_d_22
    start=1
else:
    all_seqs = np.concatenate((all_seqs,x_d_22), axis=0)
    y = np.concatenate((y,y_d_22),axis=0)
    p = np.concatenate((p,p_d_22), axis=0)
shape_y = y_d_23.shape[0]
shape_p = p_d_23.shape[0]
shp=(x_d_23.shape)[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_d_23 =   np.delete(x_d_23, [k for k in range(nece_shp,shp)], None)
x_d_23 = x_d_23.reshape(-1,sensors)
shape_x = x_d_23.shape[0]
x_d_23 = np.reshape(x_d_23, (-1,segement_time_size, np.shape(x_d_23)[1]))
y_d_23= np.delete( y_d_23, [k for k in range(x_d_23.shape[0],shape_y)], None)
p_d_23 = np.delete(p_d_23,[k for k in range(x_d_23.shape[0],shape_p)], None)
if start==0:
    all_seqs = x_d_23
    y = y_d_23
    p = p_d_23
    start=1
else:
    all_seqs = np.concatenate((all_seqs,x_d_23), axis=0)
    y = np.concatenate((y,y_d_23),axis=0)
    p = np.concatenate((p,p_d_23), axis=0)
shape_y = y_d_24.shape[0]
shape_p = p_d_24.shape[0]
shp=(x_d_24.shape)[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_d_24 =   np.delete(x_d_24, [k for k in range(nece_shp,shp)], None)
x_d_24 = x_d_24.reshape(-1,sensors)
shape_x = x_d_24.shape[0]
x_d_24 = np.reshape(x_d_24, (-1,segement_time_size, np.shape(x_d_24)[1]))
y_d_24= np.delete( y_d_24, [k for k in range(x_d_24.shape[0],shape_y)], None)
p_d_24 = np.delete(p_d_24,[k for k in range(x_d_24.shape[0],shape_p)], None)
if start==0:
    all_seqs = x_d_24
    y = y_d_24
    p = p_d_24
    start=1
else:
    all_seqs = np.concatenate((all_seqs,x_d_24), axis=0)
    y = np.concatenate((y,y_d_24),axis=0)
    p = np.concatenate((p,p_d_24), axis=0)
shape_y = y_d_25.shape[0]
shape_p = p_d_25.shape[0]
shp=(x_d_25.shape)[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_d_25 =   np.delete(x_d_25, [k for k in range(nece_shp,shp)], None)
x_d_25 = x_d_25.reshape(-1,sensors)
shape_x = x_d_25.shape[0]
x_d_25 = np.reshape(x_d_25, (-1,segement_time_size, np.shape(x_d_25)[1]))
y_d_25= np.delete( y_d_25, [k for k in range(x_d_25.shape[0],shape_y)], None)
p_d_25 = np.delete(p_d_25,[k for k in range(x_d_25.shape[0],shape_p)], None)
if start==0:
    all_seqs = x_d_25
    y = y_d_25
    p = p_d_25
    start=1
else:
    all_seqs = np.concatenate((all_seqs,x_d_25), axis=0)
    y = np.concatenate((y,y_d_25),axis=0)
    p = np.concatenate((p,p_d_25), axis=0)
shape_y = y_d_26.shape[0]
shape_p = p_d_26.shape[0]
shp=(x_d_26.shape)[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_d_26 =   np.delete(x_d_26, [k for k in range(nece_shp,shp)], None)
x_d_26 = x_d_26.reshape(-1,sensors)
shape_x = x_d_26.shape[0]
x_d_26 = np.reshape(x_d_26, (-1,segement_time_size, np.shape(x_d_26)[1]))
y_d_26= np.delete( y_d_26, [k for k in range(x_d_26.shape[0],shape_y)], None)
p_d_26 = np.delete(p_d_26,[k for k in range(x_d_26.shape[0],shape_p)], None)
if start==0:
    all_seqs = x_d_26
    y = y_d_26
    p = p_d_26
    start=1
else:
    all_seqs = np.concatenate((all_seqs,x_d_26), axis=0)
    y = np.concatenate((y,y_d_26),axis=0)
    p = np.concatenate((p,p_d_26), axis=0)
shape_y = y_d_27.shape[0]
shape_p = p_d_27.shape[0]
shp=(x_d_27.shape)[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_d_27 =   np.delete(x_d_27, [k for k in range(nece_shp,shp)], None)
x_d_27 = x_d_27.reshape(-1,sensors)
shape_x = x_d_27.shape[0]
x_d_27 = np.reshape(x_d_27, (-1,segement_time_size, np.shape(x_d_27)[1]))
y_d_27= np.delete( y_d_27, [k for k in range(x_d_27.shape[0],shape_y)], None)
p_d_27 = np.delete(p_d_27,[k for k in range(x_d_27.shape[0],shape_p)], None)
if start==0:
    all_seqs = x_d_27
    y = y_d_27
    p = p_d_27
    start=1
else:
    all_seqs = np.concatenate((all_seqs,x_d_27), axis=0)
    y = np.concatenate((y,y_d_27),axis=0)
    p = np.concatenate((p,p_d_27), axis=0)
shape_y = y_d_28.shape[0]
shape_p = p_d_28.shape[0]
shp=(x_d_28.shape)[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_d_28 =   np.delete(x_d_28, [k for k in range(nece_shp,shp)], None)
x_d_28 = x_d_28.reshape(-1,sensors)
shape_x = x_d_28.shape[0]
x_d_28 = np.reshape(x_d_28, (-1,segement_time_size, np.shape(x_d_28)[1]))
y_d_28= np.delete( y_d_28, [k for k in range(x_d_28.shape[0],shape_y)], None)
p_d_28 = np.delete(p_d_28,[k for k in range(x_d_28.shape[0],shape_p)], None)
if start==0:
    all_seqs = x_d_28
    y = y_d_28
    p = p_d_28
    start=1
else:
    all_seqs = np.concatenate((all_seqs,x_d_28), axis=0)
    y = np.concatenate((y,y_d_28),axis=0)
    p = np.concatenate((p,p_d_28), axis=0)
shape_y = y_d_29.shape[0]
shape_p = p_d_29.shape[0]
shp=(x_d_29.shape)[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_d_29 =   np.delete(x_d_29, [k for k in range(nece_shp,shp)], None)
x_d_29 = x_d_29.reshape(-1,sensors)
shape_x = x_d_29.shape[0]
x_d_29 = np.reshape(x_d_29, (-1,segement_time_size, np.shape(x_d_29)[1]))
y_d_29= np.delete( y_d_29, [k for k in range(x_d_29.shape[0],shape_y)], None)
p_d_29 = np.delete(p_d_29,[k for k in range(x_d_29.shape[0],shape_p)], None)
if start==0:
    all_seqs = x_d_29
    y = y_d_29
    p = p_d_29
    start=1
else:
    all_seqs = np.concatenate((all_seqs,x_d_29), axis=0)
    y = np.concatenate((y,y_d_29),axis=0)
    p = np.concatenate((p,p_d_29), axis=0)
shape_y = y_d_30.shape[0]
shape_p = p_d_30.shape[0]
shp=(x_d_30.shape)[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_d_30 =   np.delete(x_d_30, [k for k in range(nece_shp,shp)], None)
x_d_30 = x_d_30.reshape(-1,sensors)
shape_x = x_d_30.shape[0]
x_d_30 = np.reshape(x_d_30, (-1,segement_time_size, np.shape(x_d_30)[1]))
y_d_30= np.delete( y_d_30, [k for k in range(x_d_30.shape[0],shape_y)], None)
p_d_30 = np.delete(p_d_30,[k for k in range(x_d_30.shape[0],shape_p)], None)
if start==0:
    all_seqs = x_d_30
    y = y_d_30
    p = p_d_30
    start=1
else:
    all_seqs = np.concatenate((all_seqs,x_d_30), axis=0)
    y = np.concatenate((y,y_d_30),axis=0)
    p = np.concatenate((p,p_d_30), axis=0)
shape_y = y_d_31.shape[0]
shape_p = p_d_31.shape[0]
shp=(x_d_31.shape)[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_d_31 =   np.delete(x_d_31, [k for k in range(nece_shp,shp)], None)
x_d_31 = x_d_31.reshape(-1,sensors)
shape_x = x_d_31.shape[0]
x_d_31 = np.reshape(x_d_31, (-1,segement_time_size, np.shape(x_d_31)[1]))
y_d_31= np.delete( y_d_31, [k for k in range(x_d_31.shape[0],shape_y)], None)
p_d_31 = np.delete(p_d_31,[k for k in range(x_d_31.shape[0],shape_p)], None)
if start==0:
    all_seqs = x_d_31
    y = y_d_31
    p = p_d_31
    start=1
else:
    all_seqs = np.concatenate((all_seqs,x_d_31), axis=0)
    y = np.concatenate((y,y_d_31),axis=0)
    p = np.concatenate((p,p_d_31), axis=0)
shape_y = y_d_32.shape[0]
shape_p = p_d_32.shape[0]
shp=(x_d_32.shape)[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_d_32 =   np.delete(x_d_32, [k for k in range(nece_shp,shp)], None)
x_d_32 = x_d_32.reshape(-1,sensors)
shape_x = x_d_32.shape[0]
x_d_32 = np.reshape(x_d_32, (-1,segement_time_size, np.shape(x_d_32)[1]))
y_d_32= np.delete( y_d_32, [k for k in range(x_d_32.shape[0],shape_y)], None)
p_d_32 = np.delete(p_d_32,[k for k in range(x_d_32.shape[0],shape_p)], None)
if start==0:
    all_seqs = x_d_32
    y = y_d_32
    p = p_d_32
    start=1
else:
    all_seqs = np.concatenate((all_seqs,x_d_32), axis=0)
    y = np.concatenate((y,y_d_32),axis=0)
    p = np.concatenate((p,p_d_32), axis=0)
shape_y = y_d_33.shape[0]
shape_p = p_d_33.shape[0]
shp=(x_d_33.shape)[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_d_33 =   np.delete(x_d_33, [k for k in range(nece_shp,shp)], None)
x_d_33 = x_d_33.reshape(-1,sensors)
shape_x = x_d_33.shape[0]
x_d_33 = np.reshape(x_d_33, (-1,segement_time_size, np.shape(x_d_33)[1]))
y_d_33= np.delete( y_d_33, [k for k in range(x_d_33.shape[0],shape_y)], None)
p_d_33 = np.delete(p_d_33,[k for k in range(x_d_33.shape[0],shape_p)], None)
if start==0:
    all_seqs = x_d_33
    y = y_d_33
    p = p_d_33
    start=1
else:
    all_seqs = np.concatenate((all_seqs,x_d_33), axis=0)
    y = np.concatenate((y,y_d_33),axis=0)
    p = np.concatenate((p,p_d_33), axis=0)
shape_y = y_d_34.shape[0]
shape_p = p_d_34.shape[0]
shp=(x_d_34.shape)[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_d_34 =   np.delete(x_d_34, [k for k in range(nece_shp,shp)], None)
x_d_34 = x_d_34.reshape(-1,sensors)
shape_x = x_d_34.shape[0]
x_d_34 = np.reshape(x_d_34, (-1,segement_time_size, np.shape(x_d_34)[1]))
y_d_34= np.delete( y_d_34, [k for k in range(x_d_34.shape[0],shape_y)], None)
p_d_34 = np.delete(p_d_34,[k for k in range(x_d_34.shape[0],shape_p)], None)
if start==0:
    all_seqs = x_d_34
    y = y_d_34
    p = p_d_34
    start=1
else:
    all_seqs = np.concatenate((all_seqs,x_d_34), axis=0)
    y = np.concatenate((y,y_d_34),axis=0)
    p = np.concatenate((p,p_d_34), axis=0)
shape_y = y_d_35.shape[0]
shape_p = p_d_35.shape[0]
shp=(x_d_35.shape)[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_d_35 =   np.delete(x_d_35, [k for k in range(nece_shp,shp)], None)
x_d_35 = x_d_35.reshape(-1,sensors)
shape_x = x_d_35.shape[0]
x_d_35 = np.reshape(x_d_35, (-1,segement_time_size, np.shape(x_d_35)[1]))
y_d_35= np.delete( y_d_35, [k for k in range(x_d_35.shape[0],shape_y)], None)
p_d_35 = np.delete(p_d_35,[k for k in range(x_d_35.shape[0],shape_p)], None)
if start==0:
    all_seqs = x_d_35
    y = y_d_35
    p = p_d_35
    start=1
else:
    all_seqs = np.concatenate((all_seqs,x_d_35), axis=0)
    y = np.concatenate((y,y_d_35),axis=0)
    p = np.concatenate((p,p_d_35), axis=0)
shape_y = y_d_36.shape[0]
shape_p = p_d_36.shape[0]
shp=(x_d_36.shape)[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_d_36 =   np.delete(x_d_36, [k for k in range(nece_shp,shp)], None)
x_d_36 = x_d_36.reshape(-1,sensors)
shape_x = x_d_36.shape[0]
x_d_36 = np.reshape(x_d_36, (-1,segement_time_size, np.shape(x_d_36)[1]))
y_d_36= np.delete( y_d_36, [k for k in range(x_d_36.shape[0],shape_y)], None)
p_d_36 = np.delete(p_d_36,[k for k in range(x_d_36.shape[0],shape_p)], None)
if start==0:
    all_seqs = x_d_36
    y = y_d_36
    p = p_d_36
    start=1
else:
    all_seqs = np.concatenate((all_seqs,x_d_36), axis=0)
    y = np.concatenate((y,y_d_36),axis=0)
    p = np.concatenate((p,p_d_36), axis=0)

shape_y = y_st_1.shape[0]
shape_p = p_st_1.shape[0]
shp=(x_st_1.shape)[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_st_1 =   np.delete(x_st_1, [k for k in range(nece_shp,shp)], None)
x_st_1 = x_st_1.reshape(-1,sensors)
shape_x = x_st_1.shape[0]
x_st_1 = np.reshape(x_st_1, (-1,segement_time_size, np.shape(x_st_1)[1]))
y_st_1= np.delete( y_st_1, [k for k in range(x_st_1.shape[0],shape_y)], None)
p_st_1 = np.delete(p_st_1,[k for k in range(x_st_1.shape[0],shape_p)], None)
if start==0:
    all_seqs = x_st_1
    y = y_st_1
    p = p_st_1
    start=1
else:
    all_seqs = np.concatenate((all_seqs,x_st_1), axis=0)
    y = np.concatenate((y,y_st_1),axis=0)
    p = np.concatenate((p,p_st_1), axis=0)
shape_y = y_st_2.shape[0]
shape_p = p_st_2.shape[0]
shp=(x_st_2.shape)[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_st_2 =   np.delete(x_st_2, [k for k in range(nece_shp,shp)], None)
x_st_2 = x_st_2.reshape(-1,sensors)
shape_x = x_st_2.shape[0]
x_st_2 = np.reshape(x_st_2, (-1,segement_time_size, np.shape(x_st_2)[1]))
y_st_2= np.delete( y_st_2, [k for k in range(x_st_2.shape[0],shape_y)], None)
p_st_2 = np.delete(p_st_2,[k for k in range(x_st_2.shape[0],shape_p)], None)
if start==0:
    all_seqs = x_st_2
    y = y_st_2
    p = p_st_2
    start=1
else:
    all_seqs = np.concatenate((all_seqs,x_st_2), axis=0)
    y = np.concatenate((y,y_st_2),axis=0)
    p = np.concatenate((p,p_st_2), axis=0)
shape_y = y_st_3.shape[0]
shape_p = p_st_3.shape[0]
shp=(x_st_3.shape)[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_st_3 =   np.delete(x_st_3, [k for k in range(nece_shp,shp)], None)
x_st_3 = x_st_3.reshape(-1,sensors)
shape_x = x_st_3.shape[0]
x_st_3 = np.reshape(x_st_3, (-1,segement_time_size, np.shape(x_st_3)[1]))
y_st_3= np.delete( y_st_3, [k for k in range(x_st_3.shape[0],shape_y)], None)
p_st_3 = np.delete(p_st_3,[k for k in range(x_st_3.shape[0],shape_p)], None)
if start==0:
    all_seqs = x_st_3
    y = y_st_3
    p = p_st_3
    start=1
else:
    all_seqs = np.concatenate((all_seqs,x_st_3), axis=0)
    y = np.concatenate((y,y_st_3),axis=0)
    p = np.concatenate((p,p_st_3), axis=0)
shape_y = y_st_4.shape[0]
shape_p = p_st_4.shape[0]
shp=(x_st_4.shape)[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_st_4 =   np.delete(x_st_4, [k for k in range(nece_shp,shp)], None)
x_st_4 = x_st_4.reshape(-1,sensors)
shape_x = x_st_4.shape[0]
x_st_4 = np.reshape(x_st_4, (-1,segement_time_size, np.shape(x_st_4)[1]))
y_st_4= np.delete( y_st_4, [k for k in range(x_st_4.shape[0],shape_y)], None)
p_st_4 = np.delete(p_st_4,[k for k in range(x_st_4.shape[0],shape_p)], None)
if start==0:
    all_seqs = x_st_4
    y = y_st_4
    p = p_st_4
    start=1
else:
    all_seqs = np.concatenate((all_seqs,x_st_4), axis=0)
    y = np.concatenate((y,y_st_4),axis=0)
    p = np.concatenate((p,p_st_4), axis=0)
shape_y = y_st_5.shape[0]
shape_p = p_st_5.shape[0]
shp=(x_st_5.shape)[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_st_5 =   np.delete(x_st_5, [k for k in range(nece_shp,shp)], None)
x_st_5 = x_st_5.reshape(-1,sensors)
shape_x = x_st_5.shape[0]
x_st_5 = np.reshape(x_st_5, (-1,segement_time_size, np.shape(x_st_5)[1]))
y_st_5= np.delete( y_st_5, [k for k in range(x_st_5.shape[0],shape_y)], None)
p_st_5 = np.delete(p_st_5,[k for k in range(x_st_5.shape[0],shape_p)], None)
if start==0:
    all_seqs = x_st_5
    y = y_st_5
    p = p_st_5
    start=1
else:
    all_seqs = np.concatenate((all_seqs,x_st_5), axis=0)
    y = np.concatenate((y,y_st_5),axis=0)
    p = np.concatenate((p,p_st_5), axis=0)
shape_y = y_st_6.shape[0]
shape_p = p_st_6.shape[0]
shp=(x_st_6.shape)[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_st_6 =   np.delete(x_st_6, [k for k in range(nece_shp,shp)], None)
x_st_6 = x_st_6.reshape(-1,sensors)
shape_x = x_st_6.shape[0]
x_st_6 = np.reshape(x_st_6, (-1,segement_time_size, np.shape(x_st_6)[1]))
y_st_6= np.delete( y_st_6, [k for k in range(x_st_6.shape[0],shape_y)], None)
p_st_6 = np.delete(p_st_6,[k for k in range(x_st_6.shape[0],shape_p)], None)
if start==0:
    all_seqs = x_st_6
    y = y_st_6
    p = p_st_6
    start=1
else:
    all_seqs = np.concatenate((all_seqs,x_st_6), axis=0)
    y = np.concatenate((y,y_st_6),axis=0)
    p = np.concatenate((p,p_st_6), axis=0)
shape_y = y_st_7.shape[0]
shape_p = p_st_7.shape[0]
shp=(x_st_7.shape)[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_st_7 =   np.delete(x_st_7, [k for k in range(nece_shp,shp)], None)
x_st_7 = x_st_7.reshape(-1,sensors)
shape_x = x_st_7.shape[0]
x_st_7 = np.reshape(x_st_7, (-1,segement_time_size, np.shape(x_st_7)[1]))
y_st_7= np.delete( y_st_7, [k for k in range(x_st_7.shape[0],shape_y)], None)
p_st_7 = np.delete(p_st_7,[k for k in range(x_st_7.shape[0],shape_p)], None)
if start==0:
    all_seqs = x_st_7
    y = y_st_7
    p = p_st_7
    start=1
else:
    all_seqs = np.concatenate((all_seqs,x_st_7), axis=0)
    y = np.concatenate((y,y_st_7),axis=0)
    p = np.concatenate((p,p_st_7), axis=0)
shape_y = y_st_8.shape[0]
shape_p = p_st_8.shape[0]
shp=(x_st_8.shape)[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_st_8 =   np.delete(x_st_8, [k for k in range(nece_shp,shp)], None)
x_st_8 = x_st_8.reshape(-1,sensors)
shape_x = x_st_8.shape[0]
x_st_8 = np.reshape(x_st_8, (-1,segement_time_size, np.shape(x_st_8)[1]))
y_st_8= np.delete( y_st_8, [k for k in range(x_st_8.shape[0],shape_y)], None)
p_st_8 = np.delete(p_st_8,[k for k in range(x_st_8.shape[0],shape_p)], None)
if start==0:
    all_seqs = x_st_8
    y = y_st_8
    p = p_st_8
    start=1
else:
    all_seqs = np.concatenate((all_seqs,x_st_8), axis=0)
    y = np.concatenate((y,y_st_8),axis=0)
    p = np.concatenate((p,p_st_8), axis=0)
shape_y = y_st_9.shape[0]
shape_p = p_st_9.shape[0]
shp=(x_st_9.shape)[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_st_9 =   np.delete(x_st_9, [k for k in range(nece_shp,shp)], None)
x_st_9 = x_st_9.reshape(-1,sensors)
shape_x = x_st_9.shape[0]
x_st_9 = np.reshape(x_st_9, (-1,segement_time_size, np.shape(x_st_9)[1]))
y_st_9= np.delete( y_st_9, [k for k in range(x_st_9.shape[0],shape_y)], None)
p_st_9 = np.delete(p_st_9,[k for k in range(x_st_9.shape[0],shape_p)], None)
if start==0:
    all_seqs = x_st_9
    y = y_st_9
    p = p_st_9
    start=1
else:
    all_seqs = np.concatenate((all_seqs,x_st_9), axis=0)
    y = np.concatenate((y,y_st_9),axis=0)
    p = np.concatenate((p,p_st_9), axis=0)
shape_y = y_st_10.shape[0]
shape_p = p_st_10.shape[0]
shp=(x_st_10.shape)[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_st_10 =   np.delete(x_st_10, [k for k in range(nece_shp,shp)], None)
x_st_10 = x_st_10.reshape(-1,sensors)
shape_x = x_st_10.shape[0]
x_st_10 = np.reshape(x_st_10, (-1,segement_time_size, np.shape(x_st_10)[1]))
y_st_10= np.delete( y_st_10, [k for k in range(x_st_10.shape[0],shape_y)], None)
p_st_10 = np.delete(p_st_10,[k for k in range(x_st_10.shape[0],shape_p)], None)
if start==0:
    all_seqs = x_st_10
    y = y_st_10
    p = p_st_10
    start=1
else:
    all_seqs = np.concatenate((all_seqs,x_st_10), axis=0)
    y = np.concatenate((y,y_st_10),axis=0)
    p = np.concatenate((p,p_st_10), axis=0)
shape_y = y_st_11.shape[0]
shape_p = p_st_11.shape[0]
shp=(x_st_11.shape)[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_st_11 =   np.delete(x_st_11, [k for k in range(nece_shp,shp)], None)
x_st_11 = x_st_11.reshape(-1,sensors)
shape_x = x_st_11.shape[0]
x_st_11 = np.reshape(x_st_11, (-1,segement_time_size, np.shape(x_st_11)[1]))
y_st_11= np.delete( y_st_11, [k for k in range(x_st_11.shape[0],shape_y)], None)
p_st_11 = np.delete(p_st_11,[k for k in range(x_st_11.shape[0],shape_p)], None)
if start==0:
    all_seqs = x_st_11
    y = y_st_11
    p = p_st_11
    start=1
else:
    all_seqs = np.concatenate((all_seqs,x_st_11), axis=0)
    y = np.concatenate((y,y_st_11),axis=0)
    p = np.concatenate((p,p_st_11), axis=0)
shape_y = y_st_12.shape[0]
shape_p = p_st_12.shape[0]
shp=(x_st_12.shape)[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_st_12 =   np.delete(x_st_12, [k for k in range(nece_shp,shp)], None)
x_st_12 = x_st_12.reshape(-1,sensors)
shape_x = x_st_12.shape[0]
x_st_12 = np.reshape(x_st_12, (-1,segement_time_size, np.shape(x_st_12)[1]))
y_st_12= np.delete( y_st_12, [k for k in range(x_st_12.shape[0],shape_y)], None)
p_st_12 = np.delete(p_st_12,[k for k in range(x_st_12.shape[0],shape_p)], None)
if start==0:
    all_seqs = x_st_12
    y = y_st_12
    p = p_st_12
    start=1
else:
    all_seqs = np.concatenate((all_seqs,x_st_12), axis=0)
    y = np.concatenate((y,y_st_12),axis=0)
    p = np.concatenate((p,p_st_12), axis=0)
shape_y = y_st_13.shape[0]
shape_p = p_st_13.shape[0]
shp=(x_st_13.shape)[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_st_13 =   np.delete(x_st_13, [k for k in range(nece_shp,shp)], None)
x_st_13 = x_st_13.reshape(-1,sensors)
shape_x = x_st_13.shape[0]
x_st_13 = np.reshape(x_st_13, (-1,segement_time_size, np.shape(x_st_13)[1]))
y_st_13= np.delete( y_st_13, [k for k in range(x_st_13.shape[0],shape_y)], None)
p_st_13 = np.delete(p_st_13,[k for k in range(x_st_13.shape[0],shape_p)], None)
if start==0:
    all_seqs = x_st_13
    y = y_st_13
    p = p_st_13
    start=1
else:
    all_seqs = np.concatenate((all_seqs,x_st_13), axis=0)
    y = np.concatenate((y,y_st_13),axis=0)
    p = np.concatenate((p,p_st_13), axis=0)
shape_y = y_st_14.shape[0]
shape_p = p_st_14.shape[0]
shp=(x_st_14.shape)[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_st_14 =   np.delete(x_st_14, [k for k in range(nece_shp,shp)], None)
x_st_14 = x_st_14.reshape(-1,sensors)
shape_x = x_st_14.shape[0]
x_st_14 = np.reshape(x_st_14, (-1,segement_time_size, np.shape(x_st_14)[1]))
y_st_14= np.delete( y_st_14, [k for k in range(x_st_14.shape[0],shape_y)], None)
p_st_14 = np.delete(p_st_14,[k for k in range(x_st_14.shape[0],shape_p)], None)
if start==0:
    all_seqs = x_st_14
    y = y_st_14
    p = p_st_14
    start=1
else:
    all_seqs = np.concatenate((all_seqs,x_st_14), axis=0)
    y = np.concatenate((y,y_st_14),axis=0)
    p = np.concatenate((p,p_st_14), axis=0)
shape_y = y_st_15.shape[0]
shape_p = p_st_15.shape[0]
shp=(x_st_15.shape)[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_st_15 =   np.delete(x_st_15, [k for k in range(nece_shp,shp)], None)
x_st_15 = x_st_15.reshape(-1,sensors)
shape_x = x_st_15.shape[0]
x_st_15 = np.reshape(x_st_15, (-1,segement_time_size, np.shape(x_st_15)[1]))
y_st_15= np.delete( y_st_15, [k for k in range(x_st_15.shape[0],shape_y)], None)
p_st_15 = np.delete(p_st_15,[k for k in range(x_st_15.shape[0],shape_p)], None)
if start==0:
    all_seqs = x_st_15
    y = y_st_15
    p = p_st_15
    start=1
else:
    all_seqs = np.concatenate((all_seqs,x_st_15), axis=0)
    y = np.concatenate((y,y_st_15),axis=0)
    p = np.concatenate((p,p_st_15), axis=0)
shape_y = y_st_16.shape[0]
shape_p = p_st_16.shape[0]
shp=(x_st_16.shape)[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_st_16 =   np.delete(x_st_16, [k for k in range(nece_shp,shp)], None)
x_st_16 = x_st_16.reshape(-1,sensors)
shape_x = x_st_16.shape[0]
x_st_16 = np.reshape(x_st_16, (-1,segement_time_size, np.shape(x_st_16)[1]))
y_st_16= np.delete( y_st_16, [k for k in range(x_st_16.shape[0],shape_y)], None)
p_st_16 = np.delete(p_st_16,[k for k in range(x_st_16.shape[0],shape_p)], None)
if start==0:
    all_seqs = x_st_16
    y = y_st_16
    p = p_st_16
    start=1
else:
    all_seqs = np.concatenate((all_seqs,x_st_16), axis=0)
    y = np.concatenate((y,y_st_16),axis=0)
    p = np.concatenate((p,p_st_16), axis=0)
shape_y = y_st_17.shape[0]
shape_p = p_st_17.shape[0]
shp=(x_st_17.shape)[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_st_17 =   np.delete(x_st_17, [k for k in range(nece_shp,shp)], None)
x_st_17 = x_st_17.reshape(-1,sensors)
shape_x = x_st_17.shape[0]
x_st_17 = np.reshape(x_st_17, (-1,segement_time_size, np.shape(x_st_17)[1]))
y_st_17= np.delete( y_st_17, [k for k in range(x_st_17.shape[0],shape_y)], None)
p_st_17 = np.delete(p_st_17,[k for k in range(x_st_17.shape[0],shape_p)], None)
if start==0:
    all_seqs = x_st_17
    y = y_st_17
    p = p_st_17
    start=1
else:
    all_seqs = np.concatenate((all_seqs,x_st_17), axis=0)
    y = np.concatenate((y,y_st_17),axis=0)
    p = np.concatenate((p,p_st_17), axis=0)
shape_y = y_st_18.shape[0]
shape_p = p_st_18.shape[0]
shp=(x_st_18.shape)[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_st_18 =   np.delete(x_st_18, [k for k in range(nece_shp,shp)], None)
x_st_18 = x_st_18.reshape(-1,sensors)
shape_x = x_st_18.shape[0]
x_st_18 = np.reshape(x_st_18, (-1,segement_time_size, np.shape(x_st_18)[1]))
y_st_18= np.delete( y_st_18, [k for k in range(x_st_18.shape[0],shape_y)], None)
p_st_18 = np.delete(p_st_18,[k for k in range(x_st_18.shape[0],shape_p)], None)
if start==0:
    all_seqs = x_st_18
    y = y_st_18
    p = p_st_18
    start=1
else:
    all_seqs = np.concatenate((all_seqs,x_st_18), axis=0)
    y = np.concatenate((y,y_st_18),axis=0)
    p = np.concatenate((p,p_st_18), axis=0)
shape_y = y_st_19.shape[0]
shape_p = p_st_19.shape[0]
shp=(x_st_19.shape)[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_st_19 =   np.delete(x_st_19, [k for k in range(nece_shp,shp)], None)
x_st_19 = x_st_19.reshape(-1,sensors)
shape_x = x_st_19.shape[0]
x_st_19 = np.reshape(x_st_19, (-1,segement_time_size, np.shape(x_st_19)[1]))
y_st_19= np.delete( y_st_19, [k for k in range(x_st_19.shape[0],shape_y)], None)
p_st_19 = np.delete(p_st_19,[k for k in range(x_st_19.shape[0],shape_p)], None)
if start==0:
    all_seqs = x_st_19
    y = y_st_19
    p = p_st_19
    start=1
else:
    all_seqs = np.concatenate((all_seqs,x_st_19), axis=0)
    y = np.concatenate((y,y_st_19),axis=0)
    p = np.concatenate((p,p_st_19), axis=0)
shape_y = y_st_20.shape[0]
shape_p = p_st_20.shape[0]
shp=(x_st_20.shape)[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_st_20 =   np.delete(x_st_20, [k for k in range(nece_shp,shp)], None)
x_st_20 = x_st_20.reshape(-1,sensors)
shape_x = x_st_20.shape[0]
x_st_20 = np.reshape(x_st_20, (-1,segement_time_size, np.shape(x_st_20)[1]))
y_st_20= np.delete( y_st_20, [k for k in range(x_st_20.shape[0],shape_y)], None)
p_st_20 = np.delete(p_st_20,[k for k in range(x_st_20.shape[0],shape_p)], None)
if start==0:
    all_seqs = x_st_20
    y = y_st_20
    p = p_st_20
    start=1
else:
    all_seqs = np.concatenate((all_seqs,x_st_20), axis=0)
    y = np.concatenate((y,y_st_20),axis=0)
    p = np.concatenate((p,p_st_20), axis=0)
shape_y = y_st_21.shape[0]
shape_p = p_st_21.shape[0]
shp=(x_st_21.shape)[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_st_21 =   np.delete(x_st_21, [k for k in range(nece_shp,shp)], None)
x_st_21 = x_st_21.reshape(-1,sensors)
shape_x = x_st_21.shape[0]
x_st_21 = np.reshape(x_st_21, (-1,segement_time_size, np.shape(x_st_21)[1]))
y_st_21= np.delete( y_st_21, [k for k in range(x_st_21.shape[0],shape_y)], None)
p_st_21 = np.delete(p_st_21,[k for k in range(x_st_21.shape[0],shape_p)], None)
if start==0:
    all_seqs = x_st_21
    y = y_st_21
    p = p_st_21
    start=1
else:
    all_seqs = np.concatenate((all_seqs,x_st_21), axis=0)
    y = np.concatenate((y,y_st_21),axis=0)
    p = np.concatenate((p,p_st_21), axis=0)
shape_y = y_st_22.shape[0]
shape_p = p_st_22.shape[0]
shp=(x_st_22.shape)[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_st_22 =   np.delete(x_st_22, [k for k in range(nece_shp,shp)], None)
x_st_22 = x_st_22.reshape(-1,sensors)
shape_x = x_st_22.shape[0]
x_st_22 = np.reshape(x_st_22, (-1,segement_time_size, np.shape(x_st_22)[1]))
y_st_22= np.delete( y_st_22, [k for k in range(x_st_22.shape[0],shape_y)], None)
p_st_22 = np.delete(p_st_22,[k for k in range(x_st_22.shape[0],shape_p)], None)
if start==0:
    all_seqs = x_st_22
    y = y_st_22
    p = p_st_22
    start=1
else:
    all_seqs = np.concatenate((all_seqs,x_st_22), axis=0)
    y = np.concatenate((y,y_st_22),axis=0)
    p = np.concatenate((p,p_st_22), axis=0)
shape_y = y_st_23.shape[0]
shape_p = p_st_23.shape[0]
shp=(x_st_23.shape)[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_st_23 =   np.delete(x_st_23, [k for k in range(nece_shp,shp)], None)
x_st_23 = x_st_23.reshape(-1,sensors)
shape_x = x_st_23.shape[0]
x_st_23 = np.reshape(x_st_23, (-1,segement_time_size, np.shape(x_st_23)[1]))
y_st_23= np.delete( y_st_23, [k for k in range(x_st_23.shape[0],shape_y)], None)
p_st_23 = np.delete(p_st_23,[k for k in range(x_st_23.shape[0],shape_p)], None)
if start==0:
    all_seqs = x_st_23
    y = y_st_23
    p = p_st_23
    start=1
else:
    all_seqs = np.concatenate((all_seqs,x_st_23), axis=0)
    y = np.concatenate((y,y_st_23),axis=0)
    p = np.concatenate((p,p_st_23), axis=0)
shape_y = y_st_24.shape[0]
shape_p = p_st_24.shape[0]
shp=(x_st_24.shape)[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_st_24 =   np.delete(x_st_24, [k for k in range(nece_shp,shp)], None)
x_st_24 = x_st_24.reshape(-1,sensors)
shape_x = x_st_24.shape[0]
x_st_24 = np.reshape(x_st_24, (-1,segement_time_size, np.shape(x_st_24)[1]))
y_st_24= np.delete( y_st_24, [k for k in range(x_st_24.shape[0],shape_y)], None)
p_st_24 = np.delete(p_st_24,[k for k in range(x_st_24.shape[0],shape_p)], None)
if start==0:
    all_seqs = x_st_24
    y = y_st_24
    p = p_st_24
    start=1
else:
    all_seqs = np.concatenate((all_seqs,x_st_24), axis=0)
    y = np.concatenate((y,y_st_24),axis=0)
    p = np.concatenate((p,p_st_24), axis=0)
shape_y = y_st_25.shape[0]
shape_p = p_st_25.shape[0]
shp=(x_st_25.shape)[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_st_25 =   np.delete(x_st_25, [k for k in range(nece_shp,shp)], None)
x_st_25 = x_st_25.reshape(-1,sensors)
shape_x = x_st_25.shape[0]
x_st_25 = np.reshape(x_st_25, (-1,segement_time_size, np.shape(x_st_25)[1]))
y_st_25= np.delete( y_st_25, [k for k in range(x_st_25.shape[0],shape_y)], None)
p_st_25 = np.delete(p_st_25,[k for k in range(x_st_25.shape[0],shape_p)], None)
if start==0:
    all_seqs = x_st_25
    y = y_st_25
    p = p_st_25
    start=1
else:
    all_seqs = np.concatenate((all_seqs,x_st_25), axis=0)
    y = np.concatenate((y,y_st_25),axis=0)
    p = np.concatenate((p,p_st_25), axis=0)
shape_y = y_st_26.shape[0]
shape_p = p_st_26.shape[0]
shp=(x_st_26.shape)[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_st_26 =   np.delete(x_st_26, [k for k in range(nece_shp,shp)], None)
x_st_26 = x_st_26.reshape(-1,sensors)
shape_x = x_st_26.shape[0]
x_st_26 = np.reshape(x_st_26, (-1,segement_time_size, np.shape(x_st_26)[1]))
y_st_26= np.delete( y_st_26, [k for k in range(x_st_26.shape[0],shape_y)], None)
p_st_26 = np.delete(p_st_26,[k for k in range(x_st_26.shape[0],shape_p)], None)
if start==0:
    all_seqs = x_st_26
    y = y_st_26
    p = p_st_26
    start=1
else:
    all_seqs = np.concatenate((all_seqs,x_st_26), axis=0)
    y = np.concatenate((y,y_st_26),axis=0)
    p = np.concatenate((p,p_st_26), axis=0)
shape_y = y_st_27.shape[0]
shape_p = p_st_27.shape[0]
shp=(x_st_27.shape)[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_st_27 =   np.delete(x_st_27, [k for k in range(nece_shp,shp)], None)
x_st_27 = x_st_27.reshape(-1,sensors)
shape_x = x_st_27.shape[0]
x_st_27 = np.reshape(x_st_27, (-1,segement_time_size, np.shape(x_st_27)[1]))
y_st_27= np.delete( y_st_27, [k for k in range(x_st_27.shape[0],shape_y)], None)
p_st_27 = np.delete(p_st_27,[k for k in range(x_st_27.shape[0],shape_p)], None)
if start==0:
    all_seqs = x_st_27
    y = y_st_27
    p = p_st_27
    start=1
else:
    all_seqs = np.concatenate((all_seqs,x_st_27), axis=0)
    y = np.concatenate((y,y_st_27),axis=0)
    p = np.concatenate((p,p_st_27), axis=0)
shape_y = y_st_28.shape[0]
shape_p = p_st_28.shape[0]
shp=(x_st_28.shape)[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_st_28 =   np.delete(x_st_28, [k for k in range(nece_shp,shp)], None)
x_st_28 = x_st_28.reshape(-1,sensors)
shape_x = x_st_28.shape[0]
x_st_28 = np.reshape(x_st_28, (-1,segement_time_size, np.shape(x_st_28)[1]))
y_st_28= np.delete( y_st_28, [k for k in range(x_st_28.shape[0],shape_y)], None)
p_st_28 = np.delete(p_st_28,[k for k in range(x_st_28.shape[0],shape_p)], None)
if start==0:
    all_seqs = x_st_28
    y = y_st_28
    p = p_st_28
    start=1
else:
    all_seqs = np.concatenate((all_seqs,x_st_28), axis=0)
    y = np.concatenate((y,y_st_28),axis=0)
    p = np.concatenate((p,p_st_28), axis=0)
shape_y = y_st_29.shape[0]
shape_p = p_st_29.shape[0]
shp=(x_st_29.shape)[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_st_29 =   np.delete(x_st_29, [k for k in range(nece_shp,shp)], None)
x_st_29 = x_st_29.reshape(-1,sensors)
shape_x = x_st_29.shape[0]
x_st_29 = np.reshape(x_st_29, (-1,segement_time_size, np.shape(x_st_29)[1]))
y_st_29= np.delete( y_st_29, [k for k in range(x_st_29.shape[0],shape_y)], None)
p_st_29 = np.delete(p_st_29,[k for k in range(x_st_29.shape[0],shape_p)], None)
if start==0:
    all_seqs = x_st_29
    y = y_st_29
    p = p_st_29
    start=1
else:
    all_seqs = np.concatenate((all_seqs,x_st_29), axis=0)
    y = np.concatenate((y,y_st_29),axis=0)
    p = np.concatenate((p,p_st_29), axis=0)
shape_y = y_st_30.shape[0]
shape_p = p_st_30.shape[0]
shp=(x_st_30.shape)[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_st_30 =   np.delete(x_st_30, [k for k in range(nece_shp,shp)], None)
x_st_30 = x_st_30.reshape(-1,sensors)
shape_x = x_st_30.shape[0]
x_st_30 = np.reshape(x_st_30, (-1,segement_time_size, np.shape(x_st_30)[1]))
y_st_30= np.delete( y_st_30, [k for k in range(x_st_30.shape[0],shape_y)], None)
p_st_30 = np.delete(p_st_30,[k for k in range(x_st_30.shape[0],shape_p)], None)
if start==0:
    all_seqs = x_st_30
    y = y_st_30
    p = p_st_30
    start=1
else:
    all_seqs = np.concatenate((all_seqs,x_st_30), axis=0)
    y = np.concatenate((y,y_st_30),axis=0)
    p = np.concatenate((p,p_st_30), axis=0)
shape_y = y_st_31.shape[0]
shape_p = p_st_31.shape[0]
shp=(x_st_31.shape)[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_st_31 =   np.delete(x_st_31, [k for k in range(nece_shp,shp)], None)
x_st_31 = x_st_31.reshape(-1,sensors)
shape_x = x_st_31.shape[0]
x_st_31 = np.reshape(x_st_31, (-1,segement_time_size, np.shape(x_st_31)[1]))
y_st_31= np.delete( y_st_31, [k for k in range(x_st_31.shape[0],shape_y)], None)
p_st_31 = np.delete(p_st_31,[k for k in range(x_st_31.shape[0],shape_p)], None)
if start==0:
    all_seqs = x_st_31
    y = y_st_31
    p = p_st_31
    start=1
else:
    all_seqs = np.concatenate((all_seqs,x_st_31), axis=0)
    y = np.concatenate((y,y_st_31),axis=0)
    p = np.concatenate((p,p_st_31), axis=0)
shape_y = y_st_32.shape[0]
shape_p = p_st_32.shape[0]
shp=(x_st_32.shape)[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_st_32 =   np.delete(x_st_32, [k for k in range(nece_shp,shp)], None)
x_st_32 = x_st_32.reshape(-1,sensors)
shape_x = x_st_32.shape[0]
x_st_32 = np.reshape(x_st_32, (-1,segement_time_size, np.shape(x_st_32)[1]))
y_st_32= np.delete( y_st_32, [k for k in range(x_st_32.shape[0],shape_y)], None)
p_st_32 = np.delete(p_st_32,[k for k in range(x_st_32.shape[0],shape_p)], None)
if start==0:
    all_seqs = x_st_32
    y = y_st_32
    p = p_st_32
    start=1
else:
    all_seqs = np.concatenate((all_seqs,x_st_32), axis=0)
    y = np.concatenate((y,y_st_32),axis=0)
    p = np.concatenate((p,p_st_32), axis=0)
shape_y = y_st_33.shape[0]
shape_p = p_st_33.shape[0]
shp=(x_st_33.shape)[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_st_33 =   np.delete(x_st_33, [k for k in range(nece_shp,shp)], None)
x_st_33 = x_st_33.reshape(-1,sensors)
shape_x = x_st_33.shape[0]
x_st_33 = np.reshape(x_st_33, (-1,segement_time_size, np.shape(x_st_33)[1]))
y_st_33= np.delete( y_st_33, [k for k in range(x_st_33.shape[0],shape_y)], None)
p_st_33 = np.delete(p_st_33,[k for k in range(x_st_33.shape[0],shape_p)], None)
if start==0:
    all_seqs = x_st_33
    y = y_st_33
    p = p_st_33
    start=1
else:
    all_seqs = np.concatenate((all_seqs,x_st_33), axis=0)
    y = np.concatenate((y,y_st_33),axis=0)
    p = np.concatenate((p,p_st_33), axis=0)
shape_y = y_st_34.shape[0]
shape_p = p_st_34.shape[0]
shp=(x_st_34.shape)[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_st_34 =   np.delete(x_st_34, [k for k in range(nece_shp,shp)], None)
x_st_34 = x_st_34.reshape(-1,sensors)
shape_x = x_st_34.shape[0]
x_st_34 = np.reshape(x_st_34, (-1,segement_time_size, np.shape(x_st_34)[1]))
y_st_34= np.delete( y_st_34, [k for k in range(x_st_34.shape[0],shape_y)], None)
p_st_34 = np.delete(p_st_34,[k for k in range(x_st_34.shape[0],shape_p)], None)
if start==0:
    all_seqs = x_st_34
    y = y_st_34
    p = p_st_34
    start=1
else:
    all_seqs = np.concatenate((all_seqs,x_st_34), axis=0)
    y = np.concatenate((y,y_st_34),axis=0)
    p = np.concatenate((p,p_st_34), axis=0)
shape_y = y_st_35.shape[0]
shape_p = p_st_35.shape[0]
shp=(x_st_35.shape)[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_st_35 =   np.delete(x_st_35, [k for k in range(nece_shp,shp)], None)
x_st_35 = x_st_35.reshape(-1,sensors)
shape_x = x_st_35.shape[0]
x_st_35 = np.reshape(x_st_35, (-1,segement_time_size, np.shape(x_st_35)[1]))
y_st_35= np.delete( y_st_35, [k for k in range(x_st_35.shape[0],shape_y)], None)
p_st_35 = np.delete(p_st_35,[k for k in range(x_st_35.shape[0],shape_p)], None)
if start==0:
    all_seqs = x_st_35
    y = y_st_35
    p = p_st_35
    start=1
else:
    all_seqs = np.concatenate((all_seqs,x_st_35), axis=0)
    y = np.concatenate((y,y_st_35),axis=0)
    p = np.concatenate((p,p_st_35), axis=0)
shape_y = y_st_36.shape[0]
shape_p = p_st_36.shape[0]
shp=(x_st_36.shape)[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_st_36 =   np.delete(x_st_36, [k for k in range(nece_shp,shp)], None)
x_st_36 = x_st_36.reshape(-1,sensors)
shape_x = x_st_36.shape[0]
x_st_36 = np.reshape(x_st_36, (-1,segement_time_size, np.shape(x_st_36)[1]))
y_st_36= np.delete( y_st_36, [k for k in range(x_st_36.shape[0],shape_y)], None)
p_st_36 = np.delete(p_st_36,[k for k in range(x_st_36.shape[0],shape_p)], None)
if start==0:
    all_seqs = x_st_36
    y = y_st_36
    p = p_st_36
    start=1
else:
    all_seqs = np.concatenate((all_seqs,x_st_36), axis=0)
    y = np.concatenate((y,y_st_36),axis=0)
    p = np.concatenate((p,p_st_36), axis=0)

shape_y = y_u_1.shape[0]
shape_p = p_u_1.shape[0]
shp=(x_u_1.shape)[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_u_1 =   np.delete(x_u_1, [k for k in range(nece_shp,shp)], None)
x_u_1 = x_u_1.reshape(-1,sensors)
shape_x = x_u_1.shape[0]
x_u_1 = np.reshape(x_u_1, (-1,segement_time_size, np.shape(x_u_1)[1]))
y_u_1= np.delete( y_u_1, [k for k in range(x_u_1.shape[0],shape_y)], None)
p_u_1 = np.delete(p_u_1,[k for k in range(x_u_1.shape[0],shape_p)], None)
if start==0:
    all_seqs = x_u_1
    y = y_u_1
    p = p_u_1
    start=1
else:
    all_seqs = np.concatenate((all_seqs,x_u_1), axis=0)
    y = np.concatenate((y,y_u_1),axis=0)
    p = np.concatenate((p,p_u_1), axis=0)
shape_y = y_u_2.shape[0]
shape_p = p_u_2.shape[0]
shp=(x_u_2.shape)[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_u_2 =   np.delete(x_u_2, [k for k in range(nece_shp,shp)], None)
x_u_2 = x_u_2.reshape(-1,sensors)
shape_x = x_u_2.shape[0]
x_u_2 = np.reshape(x_u_2, (-1,segement_time_size, np.shape(x_u_2)[1]))
y_u_2= np.delete( y_u_2, [k for k in range(x_u_2.shape[0],shape_y)], None)
p_u_2 = np.delete(p_u_2,[k for k in range(x_u_2.shape[0],shape_p)], None)
if start==0:
    all_seqs = x_u_2
    y = y_u_2
    p = p_u_2
    start=1
else:
    all_seqs = np.concatenate((all_seqs,x_u_2), axis=0)
    y = np.concatenate((y,y_u_2),axis=0)
    p = np.concatenate((p,p_u_2), axis=0)
shape_y = y_u_3.shape[0]
shape_p = p_u_3.shape[0]
shp=(x_u_3.shape)[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_u_3 =   np.delete(x_u_3, [k for k in range(nece_shp,shp)], None)
x_u_3 = x_u_3.reshape(-1,sensors)
shape_x = x_u_3.shape[0]
x_u_3 = np.reshape(x_u_3, (-1,segement_time_size, np.shape(x_u_3)[1]))
y_u_3= np.delete( y_u_3, [k for k in range(x_u_3.shape[0],shape_y)], None)
p_u_3 = np.delete(p_u_3,[k for k in range(x_u_3.shape[0],shape_p)], None)
if start==0:
    all_seqs = x_u_3
    y = y_u_3
    p = p_u_3
    start=1
else:
    all_seqs = np.concatenate((all_seqs,x_u_3), axis=0)
    y = np.concatenate((y,y_u_3),axis=0)
    p = np.concatenate((p,p_u_3), axis=0)
shape_y = y_u_4.shape[0]
shape_p = p_u_4.shape[0]
shp=(x_u_4.shape)[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_u_4 =   np.delete(x_u_4, [k for k in range(nece_shp,shp)], None)
x_u_4 = x_u_4.reshape(-1,sensors)
shape_x = x_u_4.shape[0]
x_u_4 = np.reshape(x_u_4, (-1,segement_time_size, np.shape(x_u_4)[1]))
y_u_4= np.delete( y_u_4, [k for k in range(x_u_4.shape[0],shape_y)], None)
p_u_4 = np.delete(p_u_4,[k for k in range(x_u_4.shape[0],shape_p)], None)
if start==0:
    all_seqs = x_u_4
    y = y_u_4
    p = p_u_4
    start=1
else:
    all_seqs = np.concatenate((all_seqs,x_u_4), axis=0)
    y = np.concatenate((y,y_u_4),axis=0)
    p = np.concatenate((p,p_u_4), axis=0)
shape_y = y_u_5.shape[0]
shape_p = p_u_5.shape[0]
shp=(x_u_5.shape)[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_u_5 =   np.delete(x_u_5, [k for k in range(nece_shp,shp)], None)
x_u_5 = x_u_5.reshape(-1,sensors)
shape_x = x_u_5.shape[0]
x_u_5 = np.reshape(x_u_5, (-1,segement_time_size, np.shape(x_u_5)[1]))
y_u_5= np.delete( y_u_5, [k for k in range(x_u_5.shape[0],shape_y)], None)
p_u_5 = np.delete(p_u_5,[k for k in range(x_u_5.shape[0],shape_p)], None)
if start==0:
    all_seqs = x_u_5
    y = y_u_5
    p = p_u_5
    start=1
else:
    all_seqs = np.concatenate((all_seqs,x_u_5), axis=0)
    y = np.concatenate((y,y_u_5),axis=0)
    p = np.concatenate((p,p_u_5), axis=0)
shape_y = y_u_6.shape[0]
shape_p = p_u_6.shape[0]
shp=(x_u_6.shape)[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_u_6 =   np.delete(x_u_6, [k for k in range(nece_shp,shp)], None)
x_u_6 = x_u_6.reshape(-1,sensors)
shape_x = x_u_6.shape[0]
x_u_6 = np.reshape(x_u_6, (-1,segement_time_size, np.shape(x_u_6)[1]))
y_u_6= np.delete( y_u_6, [k for k in range(x_u_6.shape[0],shape_y)], None)
p_u_6 = np.delete(p_u_6,[k for k in range(x_u_6.shape[0],shape_p)], None)
if start==0:
    all_seqs = x_u_6
    y = y_u_6
    p = p_u_6
    start=1
else:
    all_seqs = np.concatenate((all_seqs,x_u_6), axis=0)
    y = np.concatenate((y,y_u_6),axis=0)
    p = np.concatenate((p,p_u_6), axis=0)
shape_y = y_u_7.shape[0]
shape_p = p_u_7.shape[0]
shp=(x_u_7.shape)[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_u_7 =   np.delete(x_u_7, [k for k in range(nece_shp,shp)], None)
x_u_7 = x_u_7.reshape(-1,sensors)
shape_x = x_u_7.shape[0]
x_u_7 = np.reshape(x_u_7, (-1,segement_time_size, np.shape(x_u_7)[1]))
y_u_7= np.delete( y_u_7, [k for k in range(x_u_7.shape[0],shape_y)], None)
p_u_7 = np.delete(p_u_7,[k for k in range(x_u_7.shape[0],shape_p)], None)
if start==0:
    all_seqs = x_u_7
    y = y_u_7
    p = p_u_7
    start=1
else:
    all_seqs = np.concatenate((all_seqs,x_u_7), axis=0)
    y = np.concatenate((y,y_u_7),axis=0)
    p = np.concatenate((p,p_u_7), axis=0)
shape_y = y_u_8.shape[0]
shape_p = p_u_8.shape[0]
shp=(x_u_8.shape)[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_u_8 =   np.delete(x_u_8, [k for k in range(nece_shp,shp)], None)
x_u_8 = x_u_8.reshape(-1,sensors)
shape_x = x_u_8.shape[0]
x_u_8 = np.reshape(x_u_8, (-1,segement_time_size, np.shape(x_u_8)[1]))
y_u_8= np.delete( y_u_8, [k for k in range(x_u_8.shape[0],shape_y)], None)
p_u_8 = np.delete(p_u_8,[k for k in range(x_u_8.shape[0],shape_p)], None)
if start==0:
    all_seqs = x_u_8
    y = y_u_8
    p = p_u_8
    start=1
else:
    all_seqs = np.concatenate((all_seqs,x_u_8), axis=0)
    y = np.concatenate((y,y_u_8),axis=0)
    p = np.concatenate((p,p_u_8), axis=0)
shape_y = y_u_9.shape[0]
shape_p = p_u_9.shape[0]
shp=(x_u_9.shape)[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_u_9 =   np.delete(x_u_9, [k for k in range(nece_shp,shp)], None)
x_u_9 = x_u_9.reshape(-1,sensors)
shape_x = x_u_9.shape[0]
x_u_9 = np.reshape(x_u_9, (-1,segement_time_size, np.shape(x_u_9)[1]))
y_u_9= np.delete( y_u_9, [k for k in range(x_u_9.shape[0],shape_y)], None)
p_u_9 = np.delete(p_u_9,[k for k in range(x_u_9.shape[0],shape_p)], None)
if start==0:
    all_seqs = x_u_9
    y = y_u_9
    p = p_u_9
    start=1
else:
    all_seqs = np.concatenate((all_seqs,x_u_9), axis=0)
    y = np.concatenate((y,y_u_9),axis=0)
    p = np.concatenate((p,p_u_9), axis=0)
shape_y = y_u_10.shape[0]
shape_p = p_u_10.shape[0]
shp=(x_u_10.shape)[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_u_10 =   np.delete(x_u_10, [k for k in range(nece_shp,shp)], None)
x_u_10 = x_u_10.reshape(-1,sensors)
shape_x = x_u_10.shape[0]
x_u_10 = np.reshape(x_u_10, (-1,segement_time_size, np.shape(x_u_10)[1]))
y_u_10= np.delete( y_u_10, [k for k in range(x_u_10.shape[0],shape_y)], None)
p_u_10 = np.delete(p_u_10,[k for k in range(x_u_10.shape[0],shape_p)], None)
if start==0:
    all_seqs = x_u_10
    y = y_u_10
    p = p_u_10
    start=1
else:
    all_seqs = np.concatenate((all_seqs,x_u_10), axis=0)
    y = np.concatenate((y,y_u_10),axis=0)
    p = np.concatenate((p,p_u_10), axis=0)
shape_y = y_u_11.shape[0]
shape_p = p_u_11.shape[0]
shp=(x_u_11.shape)[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_u_11 =   np.delete(x_u_11, [k for k in range(nece_shp,shp)], None)
x_u_11 = x_u_11.reshape(-1,sensors)
shape_x = x_u_11.shape[0]
x_u_11 = np.reshape(x_u_11, (-1,segement_time_size, np.shape(x_u_11)[1]))
y_u_11= np.delete( y_u_11, [k for k in range(x_u_11.shape[0],shape_y)], None)
p_u_11 = np.delete(p_u_11,[k for k in range(x_u_11.shape[0],shape_p)], None)
if start==0:
    all_seqs = x_u_11
    y = y_u_11
    p = p_u_11
    start=1
else:
    all_seqs = np.concatenate((all_seqs,x_u_11), axis=0)
    y = np.concatenate((y,y_u_11),axis=0)
    p = np.concatenate((p,p_u_11), axis=0)
shape_y = y_u_12.shape[0]
shape_p = p_u_12.shape[0]
shp=(x_u_12.shape)[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_u_12 =   np.delete(x_u_12, [k for k in range(nece_shp,shp)], None)
x_u_12 = x_u_12.reshape(-1,sensors)
shape_x = x_u_12.shape[0]
x_u_12 = np.reshape(x_u_12, (-1,segement_time_size, np.shape(x_u_12)[1]))
y_u_12= np.delete( y_u_12, [k for k in range(x_u_12.shape[0],shape_y)], None)
p_u_12 = np.delete(p_u_12,[k for k in range(x_u_12.shape[0],shape_p)], None)
if start==0:
    all_seqs = x_u_12
    y = y_u_12
    p = p_u_12
    start=1
else:
    all_seqs = np.concatenate((all_seqs,x_u_12), axis=0)
    y = np.concatenate((y,y_u_12),axis=0)
    p = np.concatenate((p,p_u_12), axis=0)
shape_y = y_u_13.shape[0]
shape_p = p_u_13.shape[0]
shp=(x_u_13.shape)[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_u_13 =   np.delete(x_u_13, [k for k in range(nece_shp,shp)], None)
x_u_13 = x_u_13.reshape(-1,sensors)
shape_x = x_u_13.shape[0]
x_u_13 = np.reshape(x_u_13, (-1,segement_time_size, np.shape(x_u_13)[1]))
y_u_13= np.delete( y_u_13, [k for k in range(x_u_13.shape[0],shape_y)], None)
p_u_13 = np.delete(p_u_13,[k for k in range(x_u_13.shape[0],shape_p)], None)
if start==0:
    all_seqs = x_u_13
    y = y_u_13
    p = p_u_13
    start=1
else:
    all_seqs = np.concatenate((all_seqs,x_u_13), axis=0)
    y = np.concatenate((y,y_u_13),axis=0)
    p = np.concatenate((p,p_u_13), axis=0)
shape_y = y_u_14.shape[0]
shape_p = p_u_14.shape[0]
shp=(x_u_14.shape)[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_u_14 =   np.delete(x_u_14, [k for k in range(nece_shp,shp)], None)
x_u_14 = x_u_14.reshape(-1,sensors)
shape_x = x_u_14.shape[0]
x_u_14 = np.reshape(x_u_14, (-1,segement_time_size, np.shape(x_u_14)[1]))
y_u_14= np.delete( y_u_14, [k for k in range(x_u_14.shape[0],shape_y)], None)
p_u_14 = np.delete(p_u_14,[k for k in range(x_u_14.shape[0],shape_p)], None)
if start==0:
    all_seqs = x_u_14
    y = y_u_14
    p = p_u_14
    start=1
else:
    all_seqs = np.concatenate((all_seqs,x_u_14), axis=0)
    y = np.concatenate((y,y_u_14),axis=0)
    p = np.concatenate((p,p_u_14), axis=0)
shape_y = y_u_15.shape[0]
shape_p = p_u_15.shape[0]
shp=(x_u_15.shape)[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_u_15 =   np.delete(x_u_15, [k for k in range(nece_shp,shp)], None)
x_u_15 = x_u_15.reshape(-1,sensors)
shape_x = x_u_15.shape[0]
x_u_15 = np.reshape(x_u_15, (-1,segement_time_size, np.shape(x_u_15)[1]))
y_u_15= np.delete( y_u_15, [k for k in range(x_u_15.shape[0],shape_y)], None)
p_u_15 = np.delete(p_u_15,[k for k in range(x_u_15.shape[0],shape_p)], None)
if start==0:
    all_seqs = x_u_15
    y = y_u_15
    p = p_u_15
    start=1
else:
    all_seqs = np.concatenate((all_seqs,x_u_15), axis=0)
    y = np.concatenate((y,y_u_15),axis=0)
    p = np.concatenate((p,p_u_15), axis=0)
shape_y = y_u_16.shape[0]
shape_p = p_u_16.shape[0]
shp=(x_u_16.shape)[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_u_16 =   np.delete(x_u_16, [k for k in range(nece_shp,shp)], None)
x_u_16 = x_u_16.reshape(-1,sensors)
shape_x = x_u_16.shape[0]
x_u_16 = np.reshape(x_u_16, (-1,segement_time_size, np.shape(x_u_16)[1]))
y_u_16= np.delete( y_u_16, [k for k in range(x_u_16.shape[0],shape_y)], None)
p_u_16 = np.delete(p_u_16,[k for k in range(x_u_16.shape[0],shape_p)], None)
if start==0:
    all_seqs = x_u_16
    y = y_u_16
    p = p_u_16
    start=1
else:
    all_seqs = np.concatenate((all_seqs,x_u_16), axis=0)
    y = np.concatenate((y,y_u_16),axis=0)
    p = np.concatenate((p,p_u_16), axis=0)
shape_y = y_u_17.shape[0]
shape_p = p_u_17.shape[0]
shp=(x_u_17.shape)[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_u_17 =   np.delete(x_u_17, [k for k in range(nece_shp,shp)], None)
x_u_17 = x_u_17.reshape(-1,sensors)
shape_x = x_u_17.shape[0]
x_u_17 = np.reshape(x_u_17, (-1,segement_time_size, np.shape(x_u_17)[1]))
y_u_17= np.delete( y_u_17, [k for k in range(x_u_17.shape[0],shape_y)], None)
p_u_17 = np.delete(p_u_17,[k for k in range(x_u_17.shape[0],shape_p)], None)
if start==0:
    all_seqs = x_u_17
    y = y_u_17
    p = p_u_17
    start=1
else:
    all_seqs = np.concatenate((all_seqs,x_u_17), axis=0)
    y = np.concatenate((y,y_u_17),axis=0)
    p = np.concatenate((p,p_u_17), axis=0)
shape_y = y_u_18.shape[0]
shape_p = p_u_18.shape[0]
shp=(x_u_18.shape)[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_u_18 =   np.delete(x_u_18, [k for k in range(nece_shp,shp)], None)
x_u_18 = x_u_18.reshape(-1,sensors)
shape_x = x_u_18.shape[0]
x_u_18 = np.reshape(x_u_18, (-1,segement_time_size, np.shape(x_u_18)[1]))
y_u_18= np.delete( y_u_18, [k for k in range(x_u_18.shape[0],shape_y)], None)
p_u_18 = np.delete(p_u_18,[k for k in range(x_u_18.shape[0],shape_p)], None)
if start==0:
    all_seqs = x_u_18
    y = y_u_18
    p = p_u_18
    start=1
else:
    all_seqs = np.concatenate((all_seqs,x_u_18), axis=0)
    y = np.concatenate((y,y_u_18),axis=0)
    p = np.concatenate((p,p_u_18), axis=0)
shape_y = y_u_19.shape[0]
shape_p = p_u_19.shape[0]
shp=(x_u_19.shape)[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_u_19 =   np.delete(x_u_19, [k for k in range(nece_shp,shp)], None)
x_u_19 = x_u_19.reshape(-1,sensors)
shape_x = x_u_19.shape[0]
x_u_19 = np.reshape(x_u_19, (-1,segement_time_size, np.shape(x_u_19)[1]))
y_u_19= np.delete( y_u_19, [k for k in range(x_u_19.shape[0],shape_y)], None)
p_u_19 = np.delete(p_u_19,[k for k in range(x_u_19.shape[0],shape_p)], None)
if start==0:
    all_seqs = x_u_19
    y = y_u_19
    p = p_u_19
    start=1
else:
    all_seqs = np.concatenate((all_seqs,x_u_19), axis=0)
    y = np.concatenate((y,y_u_19),axis=0)
    p = np.concatenate((p,p_u_19), axis=0)
shape_y = y_u_20.shape[0]
shape_p = p_u_20.shape[0]
shp=(x_u_20.shape)[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_u_20 =   np.delete(x_u_20, [k for k in range(nece_shp,shp)], None)
x_u_20 = x_u_20.reshape(-1,sensors)
shape_x = x_u_20.shape[0]
x_u_20 = np.reshape(x_u_20, (-1,segement_time_size, np.shape(x_u_20)[1]))
y_u_20= np.delete( y_u_20, [k for k in range(x_u_20.shape[0],shape_y)], None)
p_u_20 = np.delete(p_u_20,[k for k in range(x_u_20.shape[0],shape_p)], None)
if start==0:
    all_seqs = x_u_20
    y = y_u_20
    p = p_u_20
    start=1
else:
    all_seqs = np.concatenate((all_seqs,x_u_20), axis=0)
    y = np.concatenate((y,y_u_20),axis=0)
    p = np.concatenate((p,p_u_20), axis=0)
shape_y = y_u_21.shape[0]
shape_p = p_u_21.shape[0]
shp=(x_u_21.shape)[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_u_21 =   np.delete(x_u_21, [k for k in range(nece_shp,shp)], None)
x_u_21 = x_u_21.reshape(-1,sensors)
shape_x = x_u_21.shape[0]
x_u_21 = np.reshape(x_u_21, (-1,segement_time_size, np.shape(x_u_21)[1]))
y_u_21= np.delete( y_u_21, [k for k in range(x_u_21.shape[0],shape_y)], None)
p_u_21 = np.delete(p_u_21,[k for k in range(x_u_21.shape[0],shape_p)], None)
if start==0:
    all_seqs = x_u_21
    y = y_u_21
    p = p_u_21
    start=1
else:
    all_seqs = np.concatenate((all_seqs,x_u_21), axis=0)
    y = np.concatenate((y,y_u_21),axis=0)
    p = np.concatenate((p,p_u_21), axis=0)
shape_y = y_u_22.shape[0]
shape_p = p_u_22.shape[0]
shp=(x_u_22.shape)[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_u_22 =   np.delete(x_u_22, [k for k in range(nece_shp,shp)], None)
x_u_22 = x_u_22.reshape(-1,sensors)
shape_x = x_u_22.shape[0]
x_u_22 = np.reshape(x_u_22, (-1,segement_time_size, np.shape(x_u_22)[1]))
y_u_22= np.delete( y_u_22, [k for k in range(x_u_22.shape[0],shape_y)], None)
p_u_22 = np.delete(p_u_22,[k for k in range(x_u_22.shape[0],shape_p)], None)
if start==0:
    all_seqs = x_u_22
    y = y_u_22
    p = p_u_22
    start=1
else:
    all_seqs = np.concatenate((all_seqs,x_u_22), axis=0)
    y = np.concatenate((y,y_u_22),axis=0)
    p = np.concatenate((p,p_u_22), axis=0)
shape_y = y_u_23.shape[0]
shape_p = p_u_23.shape[0]
shp=(x_u_23.shape)[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_u_23 =   np.delete(x_u_23, [k for k in range(nece_shp,shp)], None)
x_u_23 = x_u_23.reshape(-1,sensors)
shape_x = x_u_23.shape[0]
x_u_23 = np.reshape(x_u_23, (-1,segement_time_size, np.shape(x_u_23)[1]))
y_u_23= np.delete( y_u_23, [k for k in range(x_u_23.shape[0],shape_y)], None)
p_u_23 = np.delete(p_u_23,[k for k in range(x_u_23.shape[0],shape_p)], None)
if start==0:
    all_seqs = x_u_23
    y = y_u_23
    p = p_u_23
    start=1
else:
    all_seqs = np.concatenate((all_seqs,x_u_23), axis=0)
    y = np.concatenate((y,y_u_23),axis=0)
    p = np.concatenate((p,p_u_23), axis=0)
shape_y = y_u_24.shape[0]
shape_p = p_u_24.shape[0]
shp=(x_u_24.shape)[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_u_24 =   np.delete(x_u_24, [k for k in range(nece_shp,shp)], None)
x_u_24 = x_u_24.reshape(-1,sensors)
shape_x = x_u_24.shape[0]
x_u_24 = np.reshape(x_u_24, (-1,segement_time_size, np.shape(x_u_24)[1]))
y_u_24= np.delete( y_u_24, [k for k in range(x_u_24.shape[0],shape_y)], None)
p_u_24 = np.delete(p_u_24,[k for k in range(x_u_24.shape[0],shape_p)], None)
if start==0:
    all_seqs = x_u_24
    y = y_u_24
    p = p_u_24
    start=1
else:
    all_seqs = np.concatenate((all_seqs,x_u_24), axis=0)
    y = np.concatenate((y,y_u_24),axis=0)
    p = np.concatenate((p,p_u_24), axis=0)
shape_y = y_u_25.shape[0]
shape_p = p_u_25.shape[0]
shp=(x_u_25.shape)[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_u_25 =   np.delete(x_u_25, [k for k in range(nece_shp,shp)], None)
x_u_25 = x_u_25.reshape(-1,sensors)
shape_x = x_u_25.shape[0]
x_u_25 = np.reshape(x_u_25, (-1,segement_time_size, np.shape(x_u_25)[1]))
y_u_25= np.delete( y_u_25, [k for k in range(x_u_25.shape[0],shape_y)], None)
p_u_25 = np.delete(p_u_25,[k for k in range(x_u_25.shape[0],shape_p)], None)
if start==0:
    all_seqs = x_u_25
    y = y_u_25
    p = p_u_25
    start=1
else:
    all_seqs = np.concatenate((all_seqs,x_u_25), axis=0)
    y = np.concatenate((y,y_u_25),axis=0)
    p = np.concatenate((p,p_u_25), axis=0)
shape_y = y_u_26.shape[0]
shape_p = p_u_26.shape[0]
shp=(x_u_26.shape)[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_u_26 =   np.delete(x_u_26, [k for k in range(nece_shp,shp)], None)
x_u_26 = x_u_26.reshape(-1,sensors)
shape_x = x_u_26.shape[0]
x_u_26 = np.reshape(x_u_26, (-1,segement_time_size, np.shape(x_u_26)[1]))
y_u_26= np.delete( y_u_26, [k for k in range(x_u_26.shape[0],shape_y)], None)
p_u_26 = np.delete(p_u_26,[k for k in range(x_u_26.shape[0],shape_p)], None)
if start==0:
    all_seqs = x_u_26
    y = y_u_26
    p = p_u_26
    start=1
else:
    all_seqs = np.concatenate((all_seqs,x_u_26), axis=0)
    y = np.concatenate((y,y_u_26),axis=0)
    p = np.concatenate((p,p_u_26), axis=0)
shape_y = y_u_27.shape[0]
shape_p = p_u_27.shape[0]
shp=(x_u_27.shape)[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_u_27 =   np.delete(x_u_27, [k for k in range(nece_shp,shp)], None)
x_u_27 = x_u_27.reshape(-1,sensors)
shape_x = x_u_27.shape[0]
x_u_27 = np.reshape(x_u_27, (-1,segement_time_size, np.shape(x_u_27)[1]))
y_u_27= np.delete( y_u_27, [k for k in range(x_u_27.shape[0],shape_y)], None)
p_u_27 = np.delete(p_u_27,[k for k in range(x_u_27.shape[0],shape_p)], None)
if start==0:
    all_seqs = x_u_27
    y = y_u_27
    p = p_u_27
    start=1
else:
    all_seqs = np.concatenate((all_seqs,x_u_27), axis=0)
    y = np.concatenate((y,y_u_27),axis=0)
    p = np.concatenate((p,p_u_27), axis=0)
shape_y = y_u_28.shape[0]
shape_p = p_u_28.shape[0]
shp=(x_u_28.shape)[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_u_28 =   np.delete(x_u_28, [k for k in range(nece_shp,shp)], None)
x_u_28 = x_u_28.reshape(-1,sensors)
shape_x = x_u_28.shape[0]
x_u_28 = np.reshape(x_u_28, (-1,segement_time_size, np.shape(x_u_28)[1]))
y_u_28= np.delete( y_u_28, [k for k in range(x_u_28.shape[0],shape_y)], None)
p_u_28 = np.delete(p_u_28,[k for k in range(x_u_28.shape[0],shape_p)], None)
if start==0:
    all_seqs = x_u_28
    y = y_u_28
    p = p_u_28
    start=1
else:
    all_seqs = np.concatenate((all_seqs,x_u_28), axis=0)
    y = np.concatenate((y,y_u_28),axis=0)
    p = np.concatenate((p,p_u_28), axis=0)
shape_y = y_u_29.shape[0]
shape_p = p_u_29.shape[0]
shp=(x_u_29.shape)[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_u_29 =   np.delete(x_u_29, [k for k in range(nece_shp,shp)], None)
x_u_29 = x_u_29.reshape(-1,sensors)
shape_x = x_u_29.shape[0]
x_u_29 = np.reshape(x_u_29, (-1,segement_time_size, np.shape(x_u_29)[1]))
y_u_29= np.delete( y_u_29, [k for k in range(x_u_29.shape[0],shape_y)], None)
p_u_29 = np.delete(p_u_29,[k for k in range(x_u_29.shape[0],shape_p)], None)
if start==0:
    all_seqs = x_u_29
    y = y_u_29
    p = p_u_29
    start=1
else:
    all_seqs = np.concatenate((all_seqs,x_u_29), axis=0)
    y = np.concatenate((y,y_u_29),axis=0)
    p = np.concatenate((p,p_u_29), axis=0)
shape_y = y_u_30.shape[0]
shape_p = p_u_30.shape[0]
shp=(x_u_30.shape)[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_u_30 =   np.delete(x_u_30, [k for k in range(nece_shp,shp)], None)
x_u_30 = x_u_30.reshape(-1,sensors)
shape_x = x_u_30.shape[0]
x_u_30 = np.reshape(x_u_30, (-1,segement_time_size, np.shape(x_u_30)[1]))
y_u_30= np.delete( y_u_30, [k for k in range(x_u_30.shape[0],shape_y)], None)
p_u_30 = np.delete(p_u_30,[k for k in range(x_u_30.shape[0],shape_p)], None)
if start==0:
    all_seqs = x_u_30
    y = y_u_30
    p = p_u_30
    start=1
else:
    all_seqs = np.concatenate((all_seqs,x_u_30), axis=0)
    y = np.concatenate((y,y_u_30),axis=0)
    p = np.concatenate((p,p_u_30), axis=0)
shape_y = y_u_31.shape[0]
shape_p = p_u_31.shape[0]
shp=(x_u_31.shape)[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_u_31 =   np.delete(x_u_31, [k for k in range(nece_shp,shp)], None)
x_u_31 = x_u_31.reshape(-1,sensors)
shape_x = x_u_31.shape[0]
x_u_31 = np.reshape(x_u_31, (-1,segement_time_size, np.shape(x_u_31)[1]))
y_u_31= np.delete( y_u_31, [k for k in range(x_u_31.shape[0],shape_y)], None)
p_u_31 = np.delete(p_u_31,[k for k in range(x_u_31.shape[0],shape_p)], None)
if start==0:
    all_seqs = x_u_31
    y = y_u_31
    p = p_u_31
    start=1
else:
    all_seqs = np.concatenate((all_seqs,x_u_31), axis=0)
    y = np.concatenate((y,y_u_31),axis=0)
    p = np.concatenate((p,p_u_31), axis=0)
shape_y = y_u_32.shape[0]
shape_p = p_u_32.shape[0]
shp=(x_u_32.shape)[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_u_32 =   np.delete(x_u_32, [k for k in range(nece_shp,shp)], None)
x_u_32 = x_u_32.reshape(-1,sensors)
shape_x = x_u_32.shape[0]
x_u_32 = np.reshape(x_u_32, (-1,segement_time_size, np.shape(x_u_32)[1]))
y_u_32= np.delete( y_u_32, [k for k in range(x_u_32.shape[0],shape_y)], None)
p_u_32 = np.delete(p_u_32,[k for k in range(x_u_32.shape[0],shape_p)], None)
if start==0:
    all_seqs = x_u_32
    y = y_u_32
    p = p_u_32
    start=1
else:
    all_seqs = np.concatenate((all_seqs,x_u_32), axis=0)
    y = np.concatenate((y,y_u_32),axis=0)
    p = np.concatenate((p,p_u_32), axis=0)
shape_y = y_u_33.shape[0]
shape_p = p_u_33.shape[0]
shp=(x_u_33.shape)[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_u_33 =   np.delete(x_u_33, [k for k in range(nece_shp,shp)], None)
x_u_33 = x_u_33.reshape(-1,sensors)
shape_x = x_u_33.shape[0]
x_u_33 = np.reshape(x_u_33, (-1,segement_time_size, np.shape(x_u_33)[1]))
y_u_33= np.delete( y_u_33, [k for k in range(x_u_33.shape[0],shape_y)], None)
p_u_33 = np.delete(p_u_33,[k for k in range(x_u_33.shape[0],shape_p)], None)
if start==0:
    all_seqs = x_u_33
    y = y_u_33
    p = p_u_33
    start=1
else:
    all_seqs = np.concatenate((all_seqs,x_u_33), axis=0)
    y = np.concatenate((y,y_u_33),axis=0)
    p = np.concatenate((p,p_u_33), axis=0)
shape_y = y_u_34.shape[0]
shape_p = p_u_34.shape[0]
shp=(x_u_34.shape)[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_u_34 =   np.delete(x_u_34, [k for k in range(nece_shp,shp)], None)
x_u_34 = x_u_34.reshape(-1,sensors)
shape_x = x_u_34.shape[0]
x_u_34 = np.reshape(x_u_34, (-1,segement_time_size, np.shape(x_u_34)[1]))
y_u_34= np.delete( y_u_34, [k for k in range(x_u_34.shape[0],shape_y)], None)
p_u_34 = np.delete(p_u_34,[k for k in range(x_u_34.shape[0],shape_p)], None)
if start==0:
    all_seqs = x_u_34
    y = y_u_34
    p = p_u_34
    start=1
else:
    all_seqs = np.concatenate((all_seqs,x_u_34), axis=0)
    y = np.concatenate((y,y_u_34),axis=0)
    p = np.concatenate((p,p_u_34), axis=0)
shape_y = y_u_35.shape[0]
shape_p = p_u_35.shape[0]
shp=(x_u_35.shape)[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_u_35 =   np.delete(x_u_35, [k for k in range(nece_shp,shp)], None)
x_u_35 = x_u_35.reshape(-1,sensors)
shape_x = x_u_35.shape[0]
x_u_35 = np.reshape(x_u_35, (-1,segement_time_size, np.shape(x_u_35)[1]))
y_u_35= np.delete( y_u_35, [k for k in range(x_u_35.shape[0],shape_y)], None)
p_u_35 = np.delete(p_u_35,[k for k in range(x_u_35.shape[0],shape_p)], None)
if start==0:
    all_seqs = x_u_35
    y = y_u_35
    p = p_u_35
    start=1
else:
    all_seqs = np.concatenate((all_seqs,x_u_35), axis=0)
    y = np.concatenate((y,y_u_35),axis=0)
    p = np.concatenate((p,p_u_35), axis=0)
shape_y = y_u_36.shape[0]
shape_p = p_u_36.shape[0]
shp=(x_u_36.shape)[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_u_36 =   np.delete(x_u_36, [k for k in range(nece_shp,shp)], None)
x_u_36 = x_u_36.reshape(-1,sensors)
shape_x = x_u_36.shape[0]
x_u_36 = np.reshape(x_u_36, (-1,segement_time_size, np.shape(x_u_36)[1]))
y_u_36= np.delete( y_u_36, [k for k in range(x_u_36.shape[0],shape_y)], None)
p_u_36 = np.delete(p_u_36,[k for k in range(x_u_36.shape[0],shape_p)], None)
if start==0:
    all_seqs = x_u_36
    y = y_u_36
    p = p_u_36
    start=1
else:
    all_seqs = np.concatenate((all_seqs,x_u_36), axis=0)
    y = np.concatenate((y,y_u_36),axis=0)
    p = np.concatenate((p,p_u_36), axis=0)

shape_y = y_w_1.shape[0]
shape_p = p_w_1.shape[0]
shp=(x_w_1.shape)[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_w_1 =   np.delete(x_w_1, [k for k in range(nece_shp,shp)], None)
x_w_1 = x_w_1.reshape(-1,sensors)
shape_x = x_w_1.shape[0]
x_w_1 = np.reshape(x_w_1, (-1,segement_time_size, np.shape(x_w_1)[1]))
y_w_1= np.delete( y_w_1, [k for k in range(x_w_1.shape[0],shape_y)], None)
p_w_1 = np.delete(p_w_1,[k for k in range(x_w_1.shape[0],shape_p)], None)
if start==0:
    all_seqs = x_w_1
    y = y_w_1
    p = p_w_1
    start=1
else:
    all_seqs = np.concatenate((all_seqs,x_w_1), axis=0)
    y = np.concatenate((y,y_w_1),axis=0)
    p = np.concatenate((p,p_w_1), axis=0)
shape_y = y_w_2.shape[0]
shape_p = p_w_2.shape[0]
shp=(x_w_2.shape)[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_w_2 =   np.delete(x_w_2, [k for k in range(nece_shp,shp)], None)
x_w_2 = x_w_2.reshape(-1,sensors)
shape_x = x_w_2.shape[0]
x_w_2 = np.reshape(x_w_2, (-1,segement_time_size, np.shape(x_w_2)[1]))
y_w_2= np.delete( y_w_2, [k for k in range(x_w_2.shape[0],shape_y)], None)
p_w_2 = np.delete(p_w_2,[k for k in range(x_w_2.shape[0],shape_p)], None)
if start==0:
    all_seqs = x_w_2
    y = y_w_2
    p = p_w_2
    start=1
else:
    all_seqs = np.concatenate((all_seqs,x_w_2), axis=0)
    y = np.concatenate((y,y_w_2),axis=0)
    p = np.concatenate((p,p_w_2), axis=0)
shape_y = y_w_3.shape[0]
shape_p = p_w_3.shape[0]
shp=(x_w_3.shape)[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_w_3 =   np.delete(x_w_3, [k for k in range(nece_shp,shp)], None)
x_w_3 = x_w_3.reshape(-1,sensors)
shape_x = x_w_3.shape[0]
x_w_3 = np.reshape(x_w_3, (-1,segement_time_size, np.shape(x_w_3)[1]))
y_w_3= np.delete( y_w_3, [k for k in range(x_w_3.shape[0],shape_y)], None)
p_w_3 = np.delete(p_w_3,[k for k in range(x_w_3.shape[0],shape_p)], None)
if start==0:
    all_seqs = x_w_3
    y = y_w_3
    p = p_w_3
    start=1
else:
    all_seqs = np.concatenate((all_seqs,x_w_3), axis=0)
    y = np.concatenate((y,y_w_3),axis=0)
    p = np.concatenate((p,p_w_3), axis=0)
shape_y = y_w_4.shape[0]
shape_p = p_w_4.shape[0]
shp=(x_w_4.shape)[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_w_4 =   np.delete(x_w_4, [k for k in range(nece_shp,shp)], None)
x_w_4 = x_w_4.reshape(-1,sensors)
shape_x = x_w_4.shape[0]
x_w_4 = np.reshape(x_w_4, (-1,segement_time_size, np.shape(x_w_4)[1]))
y_w_4= np.delete( y_w_4, [k for k in range(x_w_4.shape[0],shape_y)], None)
p_w_4 = np.delete(p_w_4,[k for k in range(x_w_4.shape[0],shape_p)], None)
if start==0:
    all_seqs = x_w_4
    y = y_w_4
    p = p_w_4
    start=1
else:
    all_seqs = np.concatenate((all_seqs,x_w_4), axis=0)
    y = np.concatenate((y,y_w_4),axis=0)
    p = np.concatenate((p,p_w_4), axis=0)
shape_y = y_w_5.shape[0]
shape_p = p_w_5.shape[0]
shp=(x_w_5.shape)[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_w_5 =   np.delete(x_w_5, [k for k in range(nece_shp,shp)], None)
x_w_5 = x_w_5.reshape(-1,sensors)
shape_x = x_w_5.shape[0]
x_w_5 = np.reshape(x_w_5, (-1,segement_time_size, np.shape(x_w_5)[1]))
y_w_5= np.delete( y_w_5, [k for k in range(x_w_5.shape[0],shape_y)], None)
p_w_5 = np.delete(p_w_5,[k for k in range(x_w_5.shape[0],shape_p)], None)
if start==0:
    all_seqs = x_w_5
    y = y_w_5
    p = p_w_5
    start=1
else:
    all_seqs = np.concatenate((all_seqs,x_w_5), axis=0)
    y = np.concatenate((y,y_w_5),axis=0)
    p = np.concatenate((p,p_w_5), axis=0)
shape_y = y_w_6.shape[0]
shape_p = p_w_6.shape[0]
shp=(x_w_6.shape)[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_w_6 =   np.delete(x_w_6, [k for k in range(nece_shp,shp)], None)
x_w_6 = x_w_6.reshape(-1,sensors)
shape_x = x_w_6.shape[0]
x_w_6 = np.reshape(x_w_6, (-1,segement_time_size, np.shape(x_w_6)[1]))
y_w_6= np.delete( y_w_6, [k for k in range(x_w_6.shape[0],shape_y)], None)
p_w_6 = np.delete(p_w_6,[k for k in range(x_w_6.shape[0],shape_p)], None)
if start==0:
    all_seqs = x_w_6
    y = y_w_6
    p = p_w_6
    start=1
else:
    all_seqs = np.concatenate((all_seqs,x_w_6), axis=0)
    y = np.concatenate((y,y_w_6),axis=0)
    p = np.concatenate((p,p_w_6), axis=0)
shape_y = y_w_7.shape[0]
shape_p = p_w_7.shape[0]
shp=(x_w_7.shape)[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_w_7 =   np.delete(x_w_7, [k for k in range(nece_shp,shp)], None)
x_w_7 = x_w_7.reshape(-1,sensors)
shape_x = x_w_7.shape[0]
x_w_7 = np.reshape(x_w_7, (-1,segement_time_size, np.shape(x_w_7)[1]))
y_w_7= np.delete( y_w_7, [k for k in range(x_w_7.shape[0],shape_y)], None)
p_w_7 = np.delete(p_w_7,[k for k in range(x_w_7.shape[0],shape_p)], None)
if start==0:
    all_seqs = x_w_7
    y = y_w_7
    p = p_w_7
    start=1
else:
    all_seqs = np.concatenate((all_seqs,x_w_7), axis=0)
    y = np.concatenate((y,y_w_7),axis=0)
    p = np.concatenate((p,p_w_7), axis=0)
shape_y = y_w_8.shape[0]
shape_p = p_w_8.shape[0]
shp=(x_w_8.shape)[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_w_8 =   np.delete(x_w_8, [k for k in range(nece_shp,shp)], None)
x_w_8 = x_w_8.reshape(-1,sensors)
shape_x = x_w_8.shape[0]
x_w_8 = np.reshape(x_w_8, (-1,segement_time_size, np.shape(x_w_8)[1]))
y_w_8= np.delete( y_w_8, [k for k in range(x_w_8.shape[0],shape_y)], None)
p_w_8 = np.delete(p_w_8,[k for k in range(x_w_8.shape[0],shape_p)], None)
if start==0:
    all_seqs = x_w_8
    y = y_w_8
    p = p_w_8
    start=1
else:
    all_seqs = np.concatenate((all_seqs,x_w_8), axis=0)
    y = np.concatenate((y,y_w_8),axis=0)
    p = np.concatenate((p,p_w_8), axis=0)
shape_y = y_w_9.shape[0]
shape_p = p_w_9.shape[0]
shp=(x_w_9.shape)[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_w_9 =   np.delete(x_w_9, [k for k in range(nece_shp,shp)], None)
x_w_9 = x_w_9.reshape(-1,sensors)
shape_x = x_w_9.shape[0]
x_w_9 = np.reshape(x_w_9, (-1,segement_time_size, np.shape(x_w_9)[1]))
y_w_9= np.delete( y_w_9, [k for k in range(x_w_9.shape[0],shape_y)], None)
p_w_9 = np.delete(p_w_9,[k for k in range(x_w_9.shape[0],shape_p)], None)
if start==0:
    all_seqs = x_w_9
    y = y_w_9
    p = p_w_9
    start=1
else:
    all_seqs = np.concatenate((all_seqs,x_w_9), axis=0)
    y = np.concatenate((y,y_w_9),axis=0)
    p = np.concatenate((p,p_w_9), axis=0)
shape_y = y_w_10.shape[0]
shape_p = p_w_10.shape[0]
shp=(x_w_10.shape)[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_w_10 =   np.delete(x_w_10, [k for k in range(nece_shp,shp)], None)
x_w_10 = x_w_10.reshape(-1,sensors)
shape_x = x_w_10.shape[0]
x_w_10 = np.reshape(x_w_10, (-1,segement_time_size, np.shape(x_w_10)[1]))
y_w_10= np.delete( y_w_10, [k for k in range(x_w_10.shape[0],shape_y)], None)
p_w_10 = np.delete(p_w_10,[k for k in range(x_w_10.shape[0],shape_p)], None)
if start==0:
    all_seqs = x_w_10
    y = y_w_10
    p = p_w_10
    start=1
else:
    all_seqs = np.concatenate((all_seqs,x_w_10), axis=0)
    y = np.concatenate((y,y_w_10),axis=0)
    p = np.concatenate((p,p_w_10), axis=0)
shape_y = y_w_11.shape[0]
shape_p = p_w_11.shape[0]
shp=(x_w_11.shape)[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_w_11 =   np.delete(x_w_11, [k for k in range(nece_shp,shp)], None)
x_w_11 = x_w_11.reshape(-1,sensors)
shape_x = x_w_11.shape[0]
x_w_11 = np.reshape(x_w_11, (-1,segement_time_size, np.shape(x_w_11)[1]))
y_w_11= np.delete( y_w_11, [k for k in range(x_w_11.shape[0],shape_y)], None)
p_w_11 = np.delete(p_w_11,[k for k in range(x_w_11.shape[0],shape_p)], None)
if start==0:
    all_seqs = x_w_11
    y = y_w_11
    p = p_w_11
    start=1
else:
    all_seqs = np.concatenate((all_seqs,x_w_11), axis=0)
    y = np.concatenate((y,y_w_11),axis=0)
    p = np.concatenate((p,p_w_11), axis=0)
shape_y = y_w_12.shape[0]
shape_p = p_w_12.shape[0]
shp=(x_w_12.shape)[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_w_12 =   np.delete(x_w_12, [k for k in range(nece_shp,shp)], None)
x_w_12 = x_w_12.reshape(-1,sensors)
shape_x = x_w_12.shape[0]
x_w_12 = np.reshape(x_w_12, (-1,segement_time_size, np.shape(x_w_12)[1]))
y_w_12= np.delete( y_w_12, [k for k in range(x_w_12.shape[0],shape_y)], None)
p_w_12 = np.delete(p_w_12,[k for k in range(x_w_12.shape[0],shape_p)], None)
if start==0:
    all_seqs = x_w_12
    y = y_w_12
    p = p_w_12
    start=1
else:
    all_seqs = np.concatenate((all_seqs,x_w_12), axis=0)
    y = np.concatenate((y,y_w_12),axis=0)
    p = np.concatenate((p,p_w_12), axis=0)
shape_y = y_w_13.shape[0]
shape_p = p_w_13.shape[0]
shp=(x_w_13.shape)[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_w_13 =   np.delete(x_w_13, [k for k in range(nece_shp,shp)], None)
x_w_13 = x_w_13.reshape(-1,sensors)
shape_x = x_w_13.shape[0]
x_w_13 = np.reshape(x_w_13, (-1,segement_time_size, np.shape(x_w_13)[1]))
y_w_13= np.delete( y_w_13, [k for k in range(x_w_13.shape[0],shape_y)], None)
p_w_13 = np.delete(p_w_13,[k for k in range(x_w_13.shape[0],shape_p)], None)
if start==0:
    all_seqs = x_w_13
    y = y_w_13
    p = p_w_13
    start=1
else:
    all_seqs = np.concatenate((all_seqs,x_w_13), axis=0)
    y = np.concatenate((y,y_w_13),axis=0)
    p = np.concatenate((p,p_w_13), axis=0)
shape_y = y_w_14.shape[0]
shape_p = p_w_14.shape[0]
shp=(x_w_14.shape)[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_w_14 =   np.delete(x_w_14, [k for k in range(nece_shp,shp)], None)
x_w_14 = x_w_14.reshape(-1,sensors)
shape_x = x_w_14.shape[0]
x_w_14 = np.reshape(x_w_14, (-1,segement_time_size, np.shape(x_w_14)[1]))
y_w_14= np.delete( y_w_14, [k for k in range(x_w_14.shape[0],shape_y)], None)
p_w_14 = np.delete(p_w_14,[k for k in range(x_w_14.shape[0],shape_p)], None)
if start==0:
    all_seqs = x_w_14
    y = y_w_14
    p = p_w_14
    start=1
else:
    all_seqs = np.concatenate((all_seqs,x_w_14), axis=0)
    y = np.concatenate((y,y_w_14),axis=0)
    p = np.concatenate((p,p_w_14), axis=0)
shape_y = y_w_15.shape[0]
shape_p = p_w_15.shape[0]
shp=(x_w_15.shape)[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_w_15 =   np.delete(x_w_15, [k for k in range(nece_shp,shp)], None)
x_w_15 = x_w_15.reshape(-1,sensors)
shape_x = x_w_15.shape[0]
x_w_15 = np.reshape(x_w_15, (-1,segement_time_size, np.shape(x_w_15)[1]))
y_w_15= np.delete( y_w_15, [k for k in range(x_w_15.shape[0],shape_y)], None)
p_w_15 = np.delete(p_w_15,[k for k in range(x_w_15.shape[0],shape_p)], None)
if start==0:
    all_seqs = x_w_15
    y = y_w_15
    p = p_w_15
    start=1
else:
    all_seqs = np.concatenate((all_seqs,x_w_15), axis=0)
    y = np.concatenate((y,y_w_15),axis=0)
    p = np.concatenate((p,p_w_15), axis=0)
shape_y = y_w_16.shape[0]
shape_p = p_w_16.shape[0]
shp=(x_w_16.shape)[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_w_16 =   np.delete(x_w_16, [k for k in range(nece_shp,shp)], None)
x_w_16 = x_w_16.reshape(-1,sensors)
shape_x = x_w_16.shape[0]
x_w_16 = np.reshape(x_w_16, (-1,segement_time_size, np.shape(x_w_16)[1]))
y_w_16= np.delete( y_w_16, [k for k in range(x_w_16.shape[0],shape_y)], None)
p_w_16 = np.delete(p_w_16,[k for k in range(x_w_16.shape[0],shape_p)], None)
if start==0:
    all_seqs = x_w_16
    y = y_w_16
    p = p_w_16
    start=1
else:
    all_seqs = np.concatenate((all_seqs,x_w_16), axis=0)
    y = np.concatenate((y,y_w_16),axis=0)
    p = np.concatenate((p,p_w_16), axis=0)
shape_y = y_w_17.shape[0]
shape_p = p_w_17.shape[0]
shp=(x_w_17.shape)[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_w_17 =   np.delete(x_w_17, [k for k in range(nece_shp,shp)], None)
x_w_17 = x_w_17.reshape(-1,sensors)
shape_x = x_w_17.shape[0]
x_w_17 = np.reshape(x_w_17, (-1,segement_time_size, np.shape(x_w_17)[1]))
y_w_17= np.delete( y_w_17, [k for k in range(x_w_17.shape[0],shape_y)], None)
p_w_17 = np.delete(p_w_17,[k for k in range(x_w_17.shape[0],shape_p)], None)
if start==0:
    all_seqs = x_w_17
    y = y_w_17
    p = p_w_17
    start=1
else:
    all_seqs = np.concatenate((all_seqs,x_w_17), axis=0)
    y = np.concatenate((y,y_w_17),axis=0)
    p = np.concatenate((p,p_w_17), axis=0)
shape_y = y_w_18.shape[0]
shape_p = p_w_18.shape[0]
shp=(x_w_18.shape)[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_w_18 =   np.delete(x_w_18, [k for k in range(nece_shp,shp)], None)
x_w_18 = x_w_18.reshape(-1,sensors)
shape_x = x_w_18.shape[0]
x_w_18 = np.reshape(x_w_18, (-1,segement_time_size, np.shape(x_w_18)[1]))
y_w_18= np.delete( y_w_18, [k for k in range(x_w_18.shape[0],shape_y)], None)
p_w_18 = np.delete(p_w_18,[k for k in range(x_w_18.shape[0],shape_p)], None)
if start==0:
    all_seqs = x_w_18
    y = y_w_18
    p = p_w_18
    start=1
else:
    all_seqs = np.concatenate((all_seqs,x_w_18), axis=0)
    y = np.concatenate((y,y_w_18),axis=0)
    p = np.concatenate((p,p_w_18), axis=0)
shape_y = y_w_19.shape[0]
shape_p = p_w_19.shape[0]
shp=(x_w_19.shape)[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_w_19 =   np.delete(x_w_19, [k for k in range(nece_shp,shp)], None)
x_w_19 = x_w_19.reshape(-1,sensors)
shape_x = x_w_19.shape[0]
x_w_19 = np.reshape(x_w_19, (-1,segement_time_size, np.shape(x_w_19)[1]))
y_w_19= np.delete( y_w_19, [k for k in range(x_w_19.shape[0],shape_y)], None)
p_w_19 = np.delete(p_w_19,[k for k in range(x_w_19.shape[0],shape_p)], None)
if start==0:
    all_seqs = x_w_19
    y = y_w_19
    p = p_w_19
    start=1
else:
    all_seqs = np.concatenate((all_seqs,x_w_19), axis=0)
    y = np.concatenate((y,y_w_19),axis=0)
    p = np.concatenate((p,p_w_19), axis=0)
shape_y = y_w_20.shape[0]
shape_p = p_w_20.shape[0]
shp=(x_w_20.shape)[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_w_20 =   np.delete(x_w_20, [k for k in range(nece_shp,shp)], None)
x_w_20 = x_w_20.reshape(-1,sensors)
shape_x = x_w_20.shape[0]
x_w_20 = np.reshape(x_w_20, (-1,segement_time_size, np.shape(x_w_20)[1]))
y_w_20= np.delete( y_w_20, [k for k in range(x_w_20.shape[0],shape_y)], None)
p_w_20 = np.delete(p_w_20,[k for k in range(x_w_20.shape[0],shape_p)], None)
if start==0:
    all_seqs = x_w_20
    y = y_w_20
    p = p_w_20
    start=1
else:
    all_seqs = np.concatenate((all_seqs,x_w_20), axis=0)
    y = np.concatenate((y,y_w_20),axis=0)
    p = np.concatenate((p,p_w_20), axis=0)
shape_y = y_w_21.shape[0]
shape_p = p_w_21.shape[0]
shp=(x_w_21.shape)[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_w_21 =   np.delete(x_w_21, [k for k in range(nece_shp,shp)], None)
x_w_21 = x_w_21.reshape(-1,sensors)
shape_x = x_w_21.shape[0]
x_w_21 = np.reshape(x_w_21, (-1,segement_time_size, np.shape(x_w_21)[1]))
y_w_21= np.delete( y_w_21, [k for k in range(x_w_21.shape[0],shape_y)], None)
p_w_21 = np.delete(p_w_21,[k for k in range(x_w_21.shape[0],shape_p)], None)
if start==0:
    all_seqs = x_w_21
    y = y_w_21
    p = p_w_21
    start=1
else:
    all_seqs = np.concatenate((all_seqs,x_w_21), axis=0)
    y = np.concatenate((y,y_w_21),axis=0)
    p = np.concatenate((p,p_w_21), axis=0)
shape_y = y_w_22.shape[0]
shape_p = p_w_22.shape[0]
shp=(x_w_22.shape)[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_w_22 =   np.delete(x_w_22, [k for k in range(nece_shp,shp)], None)
x_w_22 = x_w_22.reshape(-1,sensors)
shape_x = x_w_22.shape[0]
x_w_22 = np.reshape(x_w_22, (-1,segement_time_size, np.shape(x_w_22)[1]))
y_w_22= np.delete( y_w_22, [k for k in range(x_w_22.shape[0],shape_y)], None)
p_w_22 = np.delete(p_w_22,[k for k in range(x_w_22.shape[0],shape_p)], None)
if start==0:
    all_seqs = x_w_22
    y = y_w_22
    p = p_w_22
    start=1
else:
    all_seqs = np.concatenate((all_seqs,x_w_22), axis=0)
    y = np.concatenate((y,y_w_22),axis=0)
    p = np.concatenate((p,p_w_22), axis=0)
shape_y = y_w_23.shape[0]
shape_p = p_w_23.shape[0]
shp=(x_w_23.shape)[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_w_23 =   np.delete(x_w_23, [k for k in range(nece_shp,shp)], None)
x_w_23 = x_w_23.reshape(-1,sensors)
shape_x = x_w_23.shape[0]
x_w_23 = np.reshape(x_w_23, (-1,segement_time_size, np.shape(x_w_23)[1]))
y_w_23= np.delete( y_w_23, [k for k in range(x_w_23.shape[0],shape_y)], None)
p_w_23 = np.delete(p_w_23,[k for k in range(x_w_23.shape[0],shape_p)], None)
if start==0:
    all_seqs = x_w_23
    y = y_w_23
    p = p_w_23
    start=1
else:
    all_seqs = np.concatenate((all_seqs,x_w_23), axis=0)
    y = np.concatenate((y,y_w_23),axis=0)
    p = np.concatenate((p,p_w_23), axis=0)
shape_y = y_w_24.shape[0]
shape_p = p_w_24.shape[0]
shp=(x_w_24.shape)[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_w_24 =   np.delete(x_w_24, [k for k in range(nece_shp,shp)], None)
x_w_24 = x_w_24.reshape(-1,sensors)
shape_x = x_w_24.shape[0]
x_w_24 = np.reshape(x_w_24, (-1,segement_time_size, np.shape(x_w_24)[1]))
y_w_24= np.delete( y_w_24, [k for k in range(x_w_24.shape[0],shape_y)], None)
p_w_24 = np.delete(p_w_24,[k for k in range(x_w_24.shape[0],shape_p)], None)
if start==0:
    all_seqs = x_w_24
    y = y_w_24
    p = p_w_24
    start=1
else:
    all_seqs = np.concatenate((all_seqs,x_w_24), axis=0)
    y = np.concatenate((y,y_w_24),axis=0)
    p = np.concatenate((p,p_w_24), axis=0)
shape_y = y_w_25.shape[0]
shape_p = p_w_25.shape[0]
shp=(x_w_25.shape)[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_w_25 =   np.delete(x_w_25, [k for k in range(nece_shp,shp)], None)
x_w_25 = x_w_25.reshape(-1,sensors)
shape_x = x_w_25.shape[0]
x_w_25 = np.reshape(x_w_25, (-1,segement_time_size, np.shape(x_w_25)[1]))
y_w_25= np.delete( y_w_25, [k for k in range(x_w_25.shape[0],shape_y)], None)
p_w_25 = np.delete(p_w_25,[k for k in range(x_w_25.shape[0],shape_p)], None)
if start==0:
    all_seqs = x_w_25
    y = y_w_25
    p = p_w_25
    start=1
else:
    all_seqs = np.concatenate((all_seqs,x_w_25), axis=0)
    y = np.concatenate((y,y_w_25),axis=0)
    p = np.concatenate((p,p_w_25), axis=0)
shape_y = y_w_26.shape[0]
shape_p = p_w_26.shape[0]
shp=(x_w_26.shape)[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_w_26 =   np.delete(x_w_26, [k for k in range(nece_shp,shp)], None)
x_w_26 = x_w_26.reshape(-1,sensors)
shape_x = x_w_26.shape[0]
x_w_26 = np.reshape(x_w_26, (-1,segement_time_size, np.shape(x_w_26)[1]))
y_w_26= np.delete( y_w_26, [k for k in range(x_w_26.shape[0],shape_y)], None)
p_w_26 = np.delete(p_w_26,[k for k in range(x_w_26.shape[0],shape_p)], None)
if start==0:
    all_seqs = x_w_26
    y = y_w_26
    p = p_w_26
    start=1
else:
    all_seqs = np.concatenate((all_seqs,x_w_26), axis=0)
    y = np.concatenate((y,y_w_26),axis=0)
    p = np.concatenate((p,p_w_26), axis=0)
shape_y = y_w_27.shape[0]
shape_p = p_w_27.shape[0]
shp=(x_w_27.shape)[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_w_27 =   np.delete(x_w_27, [k for k in range(nece_shp,shp)], None)
x_w_27 = x_w_27.reshape(-1,sensors)
shape_x = x_w_27.shape[0]
x_w_27 = np.reshape(x_w_27, (-1,segement_time_size, np.shape(x_w_27)[1]))
y_w_27= np.delete( y_w_27, [k for k in range(x_w_27.shape[0],shape_y)], None)
p_w_27 = np.delete(p_w_27,[k for k in range(x_w_27.shape[0],shape_p)], None)
if start==0:
    all_seqs = x_w_27
    y = y_w_27
    p = p_w_27
    start=1
else:
    all_seqs = np.concatenate((all_seqs,x_w_27), axis=0)
    y = np.concatenate((y,y_w_27),axis=0)
    p = np.concatenate((p,p_w_27), axis=0)
shape_y = y_w_28.shape[0]
shape_p = p_w_28.shape[0]
shp=(x_w_28.shape)[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_w_28 =   np.delete(x_w_28, [k for k in range(nece_shp,shp)], None)
x_w_28 = x_w_28.reshape(-1,sensors)
shape_x = x_w_28.shape[0]
x_w_28 = np.reshape(x_w_28, (-1,segement_time_size, np.shape(x_w_28)[1]))
y_w_28= np.delete( y_w_28, [k for k in range(x_w_28.shape[0],shape_y)], None)
p_w_28 = np.delete(p_w_28,[k for k in range(x_w_28.shape[0],shape_p)], None)
if start==0:
    all_seqs = x_w_28
    y = y_w_28
    p = p_w_28
    start=1
else:
    all_seqs = np.concatenate((all_seqs,x_w_28), axis=0)
    y = np.concatenate((y,y_w_28),axis=0)
    p = np.concatenate((p,p_w_28), axis=0)
shape_y = y_w_29.shape[0]
shape_p = p_w_29.shape[0]
shp=(x_w_29.shape)[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_w_29 =   np.delete(x_w_29, [k for k in range(nece_shp,shp)], None)
x_w_29 = x_w_29.reshape(-1,sensors)
shape_x = x_w_29.shape[0]
x_w_29 = np.reshape(x_w_29, (-1,segement_time_size, np.shape(x_w_29)[1]))
y_w_29= np.delete( y_w_29, [k for k in range(x_w_29.shape[0],shape_y)], None)
p_w_29 = np.delete(p_w_29,[k for k in range(x_w_29.shape[0],shape_p)], None)
if start==0:
    all_seqs = x_w_29
    y = y_w_29
    p = p_w_29
    start=1
else:
    all_seqs = np.concatenate((all_seqs,x_w_29), axis=0)
    y = np.concatenate((y,y_w_29),axis=0)
    p = np.concatenate((p,p_w_29), axis=0)
shape_y = y_w_30.shape[0]
shape_p = p_w_30.shape[0]
shp=(x_w_30.shape)[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_w_30 =   np.delete(x_w_30, [k for k in range(nece_shp,shp)], None)
x_w_30 = x_w_30.reshape(-1,sensors)
shape_x = x_w_30.shape[0]
x_w_30 = np.reshape(x_w_30, (-1,segement_time_size, np.shape(x_w_30)[1]))
y_w_30= np.delete( y_w_30, [k for k in range(x_w_30.shape[0],shape_y)], None)
p_w_30 = np.delete(p_w_30,[k for k in range(x_w_30.shape[0],shape_p)], None)
if start==0:
    all_seqs = x_w_30
    y = y_w_30
    p = p_w_30
    start=1
else:
    all_seqs = np.concatenate((all_seqs,x_w_30), axis=0)
    y = np.concatenate((y,y_w_30),axis=0)
    p = np.concatenate((p,p_w_30), axis=0)
shape_y = y_w_31.shape[0]
shape_p = p_w_31.shape[0]
shp=(x_w_31.shape)[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_w_31 =   np.delete(x_w_31, [k for k in range(nece_shp,shp)], None)
x_w_31 = x_w_31.reshape(-1,sensors)
shape_x = x_w_31.shape[0]
x_w_31 = np.reshape(x_w_31, (-1,segement_time_size, np.shape(x_w_31)[1]))
y_w_31= np.delete( y_w_31, [k for k in range(x_w_31.shape[0],shape_y)], None)
p_w_31 = np.delete(p_w_31,[k for k in range(x_w_31.shape[0],shape_p)], None)
if start==0:
    all_seqs = x_w_31
    y = y_w_31
    p = p_w_31
    start=1
else:
    all_seqs = np.concatenate((all_seqs,x_w_31), axis=0)
    y = np.concatenate((y,y_w_31),axis=0)
    p = np.concatenate((p,p_w_31), axis=0)
shape_y = y_w_32.shape[0]
shape_p = p_w_32.shape[0]
shp=(x_w_32.shape)[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_w_32 =   np.delete(x_w_32, [k for k in range(nece_shp,shp)], None)
x_w_32 = x_w_32.reshape(-1,sensors)
shape_x = x_w_32.shape[0]
x_w_32 = np.reshape(x_w_32, (-1,segement_time_size, np.shape(x_w_32)[1]))
y_w_32= np.delete( y_w_32, [k for k in range(x_w_32.shape[0],shape_y)], None)
p_w_32 = np.delete(p_w_32,[k for k in range(x_w_32.shape[0],shape_p)], None)
if start==0:
    all_seqs = x_w_32
    y = y_w_32
    p = p_w_32
    start=1
else:
    all_seqs = np.concatenate((all_seqs,x_w_32), axis=0)
    y = np.concatenate((y,y_w_32),axis=0)
    p = np.concatenate((p,p_w_32), axis=0)
shape_y = y_w_33.shape[0]
shape_p = p_w_33.shape[0]
shp=(x_w_33.shape)[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_w_33 =   np.delete(x_w_33, [k for k in range(nece_shp,shp)], None)
x_w_33 = x_w_33.reshape(-1,sensors)
shape_x = x_w_33.shape[0]
x_w_33 = np.reshape(x_w_33, (-1,segement_time_size, np.shape(x_w_33)[1]))
y_w_33= np.delete( y_w_33, [k for k in range(x_w_33.shape[0],shape_y)], None)
p_w_33 = np.delete(p_w_33,[k for k in range(x_w_33.shape[0],shape_p)], None)
if start==0:
    all_seqs = x_w_33
    y = y_w_33
    p = p_w_33
    start=1
else:
    all_seqs = np.concatenate((all_seqs,x_w_33), axis=0)
    y = np.concatenate((y,y_w_33),axis=0)
    p = np.concatenate((p,p_w_33), axis=0)
shape_y = y_w_34.shape[0]
shape_p = p_w_34.shape[0]
shp=(x_w_34.shape)[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_w_34 =   np.delete(x_w_34, [k for k in range(nece_shp,shp)], None)
x_w_34 = x_w_34.reshape(-1,sensors)
shape_x = x_w_34.shape[0]
x_w_34 = np.reshape(x_w_34, (-1,segement_time_size, np.shape(x_w_34)[1]))
y_w_34= np.delete( y_w_34, [k for k in range(x_w_34.shape[0],shape_y)], None)
p_w_34 = np.delete(p_w_34,[k for k in range(x_w_34.shape[0],shape_p)], None)
if start==0:
    all_seqs = x_w_34
    y = y_w_34
    p = p_w_34
    start=1
else:
    all_seqs = np.concatenate((all_seqs,x_w_34), axis=0)
    y = np.concatenate((y,y_w_34),axis=0)
    p = np.concatenate((p,p_w_34), axis=0)
shape_y = y_w_35.shape[0]
shape_p = p_w_35.shape[0]
shp=(x_w_35.shape)[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_w_35 =   np.delete(x_w_35, [k for k in range(nece_shp,shp)], None)
x_w_35 = x_w_35.reshape(-1,sensors)
shape_x = x_w_35.shape[0]
x_w_35 = np.reshape(x_w_35, (-1,segement_time_size, np.shape(x_w_35)[1]))
y_w_35= np.delete( y_w_35, [k for k in range(x_w_35.shape[0],shape_y)], None)
p_w_35 = np.delete(p_w_35,[k for k in range(x_w_35.shape[0],shape_p)], None)
if start==0:
    all_seqs = x_w_35
    y = y_w_35
    p = p_w_35
    start=1
else:
    all_seqs = np.concatenate((all_seqs,x_w_35), axis=0)
    y = np.concatenate((y,y_w_35),axis=0)
    p = np.concatenate((p,p_w_35), axis=0)
shape_y = y_w_36.shape[0]
shape_p = p_w_36.shape[0]
shp=(x_w_36.shape)[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_w_36 =   np.delete(x_w_36, [k for k in range(nece_shp,shp)], None)
x_w_36 = x_w_36.reshape(-1,sensors)
shape_x = x_w_36.shape[0]
x_w_36 = np.reshape(x_w_36, (-1,segement_time_size, np.shape(x_w_36)[1]))
y_w_36= np.delete( y_w_36, [k for k in range(x_w_36.shape[0],shape_y)], None)
p_w_36 = np.delete(p_w_36,[k for k in range(x_w_36.shape[0],shape_p)], None)
if start==0:
    all_seqs = x_w_36
    y = y_w_36
    p = p_w_36
    start=1
else:
    all_seqs = np.concatenate((all_seqs,x_w_36), axis=0)
    y = np.concatenate((y,y_w_36),axis=0)
    p = np.concatenate((p,p_w_36), axis=0)

shape_y = y_j_1.shape[0]
shape_p = p_j_1.shape[0]
shp=(x_j_1.shape)[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_j_1 =   np.delete(x_j_1, [k for k in range(nece_shp,shp)], None)
x_j_1 = x_j_1.reshape(-1,sensors)
shape_x = x_j_1.shape[0]
x_j_1 = np.reshape(x_j_1, (-1,segement_time_size, np.shape(x_j_1)[1]))
y_j_1= np.delete( y_j_1, [k for k in range(x_j_1.shape[0],shape_y)], None)
p_j_1 = np.delete(p_j_1,[k for k in range(x_j_1.shape[0],shape_p)], None)
if start==0:
    all_seqs = x_j_1
    y = y_j_1
    p = p_j_1
    start=1
else:
    all_seqs = np.concatenate((all_seqs,x_j_1), axis=0)
    y = np.concatenate((y,y_j_1),axis=0)
    p = np.concatenate((p,p_j_1), axis=0)
shape_y = y_j_2.shape[0]
shape_p = p_j_2.shape[0]
shp=(x_j_2.shape)[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_j_2 =   np.delete(x_j_2, [k for k in range(nece_shp,shp)], None)
x_j_2 = x_j_2.reshape(-1,sensors)
shape_x = x_j_2.shape[0]
x_j_2 = np.reshape(x_j_2, (-1,segement_time_size, np.shape(x_j_2)[1]))
y_j_2= np.delete( y_j_2, [k for k in range(x_j_2.shape[0],shape_y)], None)
p_j_2 = np.delete(p_j_2,[k for k in range(x_j_2.shape[0],shape_p)], None)
if start==0:
    all_seqs = x_j_2
    y = y_j_2
    p = p_j_2
    start=1
else:
    all_seqs = np.concatenate((all_seqs,x_j_2), axis=0)
    y = np.concatenate((y,y_j_2),axis=0)
    p = np.concatenate((p,p_j_2), axis=0)
shape_y = y_j_3.shape[0]
shape_p = p_j_3.shape[0]
shp=(x_j_3.shape)[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_j_3 =   np.delete(x_j_3, [k for k in range(nece_shp,shp)], None)
x_j_3 = x_j_3.reshape(-1,sensors)
shape_x = x_j_3.shape[0]
x_j_3 = np.reshape(x_j_3, (-1,segement_time_size, np.shape(x_j_3)[1]))
y_j_3= np.delete( y_j_3, [k for k in range(x_j_3.shape[0],shape_y)], None)
p_j_3 = np.delete(p_j_3,[k for k in range(x_j_3.shape[0],shape_p)], None)
if start==0:
    all_seqs = x_j_3
    y = y_j_3
    p = p_j_3
    start=1
else:
    all_seqs = np.concatenate((all_seqs,x_j_3), axis=0)
    y = np.concatenate((y,y_j_3),axis=0)
    p = np.concatenate((p,p_j_3), axis=0)
shape_y = y_j_4.shape[0]
shape_p = p_j_4.shape[0]
shp=(x_j_4.shape)[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_j_4 =   np.delete(x_j_4, [k for k in range(nece_shp,shp)], None)
x_j_4 = x_j_4.reshape(-1,sensors)
shape_x = x_j_4.shape[0]
x_j_4 = np.reshape(x_j_4, (-1,segement_time_size, np.shape(x_j_4)[1]))
y_j_4= np.delete( y_j_4, [k for k in range(x_j_4.shape[0],shape_y)], None)
p_j_4 = np.delete(p_j_4,[k for k in range(x_j_4.shape[0],shape_p)], None)
if start==0:
    all_seqs = x_j_4
    y = y_j_4
    p = p_j_4
    start=1
else:
    all_seqs = np.concatenate((all_seqs,x_j_4), axis=0)
    y = np.concatenate((y,y_j_4),axis=0)
    p = np.concatenate((p,p_j_4), axis=0)
shape_y = y_j_5.shape[0]
shape_p = p_j_5.shape[0]
shp=(x_j_5.shape)[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_j_5 =   np.delete(x_j_5, [k for k in range(nece_shp,shp)], None)
x_j_5 = x_j_5.reshape(-1,sensors)
shape_x = x_j_5.shape[0]
x_j_5 = np.reshape(x_j_5, (-1,segement_time_size, np.shape(x_j_5)[1]))
y_j_5= np.delete( y_j_5, [k for k in range(x_j_5.shape[0],shape_y)], None)
p_j_5 = np.delete(p_j_5,[k for k in range(x_j_5.shape[0],shape_p)], None)
if start==0:
    all_seqs = x_j_5
    y = y_j_5
    p = p_j_5
    start=1
else:
    all_seqs = np.concatenate((all_seqs,x_j_5), axis=0)
    y = np.concatenate((y,y_j_5),axis=0)
    p = np.concatenate((p,p_j_5), axis=0)
shape_y = y_j_6.shape[0]
shape_p = p_j_6.shape[0]
shp=(x_j_6.shape)[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_j_6 =   np.delete(x_j_6, [k for k in range(nece_shp,shp)], None)
x_j_6 = x_j_6.reshape(-1,sensors)
shape_x = x_j_6.shape[0]
x_j_6 = np.reshape(x_j_6, (-1,segement_time_size, np.shape(x_j_6)[1]))
y_j_6= np.delete( y_j_6, [k for k in range(x_j_6.shape[0],shape_y)], None)
p_j_6 = np.delete(p_j_6,[k for k in range(x_j_6.shape[0],shape_p)], None)
if start==0:
    all_seqs = x_j_6
    y = y_j_6
    p = p_j_6
    start=1
else:
    all_seqs = np.concatenate((all_seqs,x_j_6), axis=0)
    y = np.concatenate((y,y_j_6),axis=0)
    p = np.concatenate((p,p_j_6), axis=0)
shape_y = y_j_7.shape[0]
shape_p = p_j_7.shape[0]
shp=(x_j_7.shape)[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_j_7 =   np.delete(x_j_7, [k for k in range(nece_shp,shp)], None)
x_j_7 = x_j_7.reshape(-1,sensors)
shape_x = x_j_7.shape[0]
x_j_7 = np.reshape(x_j_7, (-1,segement_time_size, np.shape(x_j_7)[1]))
y_j_7= np.delete( y_j_7, [k for k in range(x_j_7.shape[0],shape_y)], None)
p_j_7 = np.delete(p_j_7,[k for k in range(x_j_7.shape[0],shape_p)], None)
if start==0:
    all_seqs = x_j_7
    y = y_j_7
    p = p_j_7
    start=1
else:
    all_seqs = np.concatenate((all_seqs,x_j_7), axis=0)
    y = np.concatenate((y,y_j_7),axis=0)
    p = np.concatenate((p,p_j_7), axis=0)
shape_y = y_j_8.shape[0]
shape_p = p_j_8.shape[0]
shp=(x_j_8.shape)[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_j_8 =   np.delete(x_j_8, [k for k in range(nece_shp,shp)], None)
x_j_8 = x_j_8.reshape(-1,sensors)
shape_x = x_j_8.shape[0]
x_j_8 = np.reshape(x_j_8, (-1,segement_time_size, np.shape(x_j_8)[1]))
y_j_8= np.delete( y_j_8, [k for k in range(x_j_8.shape[0],shape_y)], None)
p_j_8 = np.delete(p_j_8,[k for k in range(x_j_8.shape[0],shape_p)], None)
if start==0:
    all_seqs = x_j_8
    y = y_j_8
    p = p_j_8
    start=1
else:
    all_seqs = np.concatenate((all_seqs,x_j_8), axis=0)
    y = np.concatenate((y,y_j_8),axis=0)
    p = np.concatenate((p,p_j_8), axis=0)
shape_y = y_j_9.shape[0]
shape_p = p_j_9.shape[0]
shp=(x_j_9.shape)[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_j_9 =   np.delete(x_j_9, [k for k in range(nece_shp,shp)], None)
x_j_9 = x_j_9.reshape(-1,sensors)
shape_x = x_j_9.shape[0]
x_j_9 = np.reshape(x_j_9, (-1,segement_time_size, np.shape(x_j_9)[1]))
y_j_9= np.delete( y_j_9, [k for k in range(x_j_9.shape[0],shape_y)], None)
p_j_9 = np.delete(p_j_9,[k for k in range(x_j_9.shape[0],shape_p)], None)
if start==0:
    all_seqs = x_j_9
    y = y_j_9
    p = p_j_9
    start=1
else:
    all_seqs = np.concatenate((all_seqs,x_j_9), axis=0)
    y = np.concatenate((y,y_j_9),axis=0)
    p = np.concatenate((p,p_j_9), axis=0)
shape_y = y_j_10.shape[0]
shape_p = p_j_10.shape[0]
shp=(x_j_10.shape)[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_j_10 =   np.delete(x_j_10, [k for k in range(nece_shp,shp)], None)
x_j_10 = x_j_10.reshape(-1,sensors)
shape_x = x_j_10.shape[0]
x_j_10 = np.reshape(x_j_10, (-1,segement_time_size, np.shape(x_j_10)[1]))
y_j_10= np.delete( y_j_10, [k for k in range(x_j_10.shape[0],shape_y)], None)
p_j_10 = np.delete(p_j_10,[k for k in range(x_j_10.shape[0],shape_p)], None)
if start==0:
    all_seqs = x_j_10
    y = y_j_10
    p = p_j_10
    start=1
else:
    all_seqs = np.concatenate((all_seqs,x_j_10), axis=0)
    y = np.concatenate((y,y_j_10),axis=0)
    p = np.concatenate((p,p_j_10), axis=0)
shape_y = y_j_11.shape[0]
shape_p = p_j_11.shape[0]
shp=(x_j_11.shape)[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_j_11 =   np.delete(x_j_11, [k for k in range(nece_shp,shp)], None)
x_j_11 = x_j_11.reshape(-1,sensors)
shape_x = x_j_11.shape[0]
x_j_11 = np.reshape(x_j_11, (-1,segement_time_size, np.shape(x_j_11)[1]))
y_j_11= np.delete( y_j_11, [k for k in range(x_j_11.shape[0],shape_y)], None)
p_j_11 = np.delete(p_j_11,[k for k in range(x_j_11.shape[0],shape_p)], None)
if start==0:
    all_seqs = x_j_11
    y = y_j_11
    p = p_j_11
    start=1
else:
    all_seqs = np.concatenate((all_seqs,x_j_11), axis=0)
    y = np.concatenate((y,y_j_11),axis=0)
    p = np.concatenate((p,p_j_11), axis=0)
shape_y = y_j_12.shape[0]
shape_p = p_j_12.shape[0]
shp=(x_j_12.shape)[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_j_12 =   np.delete(x_j_12, [k for k in range(nece_shp,shp)], None)
x_j_12 = x_j_12.reshape(-1,sensors)
shape_x = x_j_12.shape[0]
x_j_12 = np.reshape(x_j_12, (-1,segement_time_size, np.shape(x_j_12)[1]))
y_j_12= np.delete( y_j_12, [k for k in range(x_j_12.shape[0],shape_y)], None)
p_j_12 = np.delete(p_j_12,[k for k in range(x_j_12.shape[0],shape_p)], None)
if start==0:
    all_seqs = x_j_12
    y = y_j_12
    p = p_j_12
    start=1
else:
    all_seqs = np.concatenate((all_seqs,x_j_12), axis=0)
    y = np.concatenate((y,y_j_12),axis=0)
    p = np.concatenate((p,p_j_12), axis=0)
shape_y = y_j_13.shape[0]
shape_p = p_j_13.shape[0]
shp=(x_j_13.shape)[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_j_13 =   np.delete(x_j_13, [k for k in range(nece_shp,shp)], None)
x_j_13 = x_j_13.reshape(-1,sensors)
shape_x = x_j_13.shape[0]
x_j_13 = np.reshape(x_j_13, (-1,segement_time_size, np.shape(x_j_13)[1]))
y_j_13= np.delete( y_j_13, [k for k in range(x_j_13.shape[0],shape_y)], None)
p_j_13 = np.delete(p_j_13,[k for k in range(x_j_13.shape[0],shape_p)], None)
if start==0:
    all_seqs = x_j_13
    y = y_j_13
    p = p_j_13
    start=1
else:
    all_seqs = np.concatenate((all_seqs,x_j_13), axis=0)
    y = np.concatenate((y,y_j_13),axis=0)
    p = np.concatenate((p,p_j_13), axis=0)
shape_y = y_j_14.shape[0]
shape_p = p_j_14.shape[0]
shp=(x_j_14.shape)[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_j_14 =   np.delete(x_j_14, [k for k in range(nece_shp,shp)], None)
x_j_14 = x_j_14.reshape(-1,sensors)
shape_x = x_j_14.shape[0]
x_j_14 = np.reshape(x_j_14, (-1,segement_time_size, np.shape(x_j_14)[1]))
y_j_14= np.delete( y_j_14, [k for k in range(x_j_14.shape[0],shape_y)], None)
p_j_14 = np.delete(p_j_14,[k for k in range(x_j_14.shape[0],shape_p)], None)
if start==0:
    all_seqs = x_j_14
    y = y_j_14
    p = p_j_14
    start=1
else:
    all_seqs = np.concatenate((all_seqs,x_j_14), axis=0)
    y = np.concatenate((y,y_j_14),axis=0)
    p = np.concatenate((p,p_j_14), axis=0)
shape_y = y_j_15.shape[0]
shape_p = p_j_15.shape[0]
shp=(x_j_15.shape)[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_j_15 =   np.delete(x_j_15, [k for k in range(nece_shp,shp)], None)
x_j_15 = x_j_15.reshape(-1,sensors)
shape_x = x_j_15.shape[0]
x_j_15 = np.reshape(x_j_15, (-1,segement_time_size, np.shape(x_j_15)[1]))
y_j_15= np.delete( y_j_15, [k for k in range(x_j_15.shape[0],shape_y)], None)
p_j_15 = np.delete(p_j_15,[k for k in range(x_j_15.shape[0],shape_p)], None)
if start==0:
    all_seqs = x_j_15
    y = y_j_15
    p = p_j_15
    start=1
else:
    all_seqs = np.concatenate((all_seqs,x_j_15), axis=0)
    y = np.concatenate((y,y_j_15),axis=0)
    p = np.concatenate((p,p_j_15), axis=0)
shape_y = y_j_16.shape[0]
shape_p = p_j_16.shape[0]
shp=(x_j_16.shape)[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_j_16 =   np.delete(x_j_16, [k for k in range(nece_shp,shp)], None)
x_j_16 = x_j_16.reshape(-1,sensors)
shape_x = x_j_16.shape[0]
x_j_16 = np.reshape(x_j_16, (-1,segement_time_size, np.shape(x_j_16)[1]))
y_j_16= np.delete( y_j_16, [k for k in range(x_j_16.shape[0],shape_y)], None)
p_j_16 = np.delete(p_j_16,[k for k in range(x_j_16.shape[0],shape_p)], None)
if start==0:
    all_seqs = x_j_16
    y = y_j_16
    p = p_j_16
    start=1
else:
    all_seqs = np.concatenate((all_seqs,x_j_16), axis=0)
    y = np.concatenate((y,y_j_16),axis=0)
    p = np.concatenate((p,p_j_16), axis=0)
shape_y = y_j_17.shape[0]
shape_p = p_j_17.shape[0]
shp=(x_j_17.shape)[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_j_17 =   np.delete(x_j_17, [k for k in range(nece_shp,shp)], None)
x_j_17 = x_j_17.reshape(-1,sensors)
shape_x = x_j_17.shape[0]
x_j_17 = np.reshape(x_j_17, (-1,segement_time_size, np.shape(x_j_17)[1]))
y_j_17= np.delete( y_j_17, [k for k in range(x_j_17.shape[0],shape_y)], None)
p_j_17 = np.delete(p_j_17,[k for k in range(x_j_17.shape[0],shape_p)], None)
if start==0:
    all_seqs = x_j_17
    y = y_j_17
    p = p_j_17
    start=1
else:
    all_seqs = np.concatenate((all_seqs,x_j_17), axis=0)
    y = np.concatenate((y,y_j_17),axis=0)
    p = np.concatenate((p,p_j_17), axis=0)
shape_y = y_j_18.shape[0]
shape_p = p_j_18.shape[0]
shp=(x_j_18.shape)[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_j_18 =   np.delete(x_j_18, [k for k in range(nece_shp,shp)], None)
x_j_18 = x_j_18.reshape(-1,sensors)
shape_x = x_j_18.shape[0]
x_j_18 = np.reshape(x_j_18, (-1,segement_time_size, np.shape(x_j_18)[1]))
y_j_18= np.delete( y_j_18, [k for k in range(x_j_18.shape[0],shape_y)], None)
p_j_18 = np.delete(p_j_18,[k for k in range(x_j_18.shape[0],shape_p)], None)
if start==0:
    all_seqs = x_j_18
    y = y_j_18
    p = p_j_18
    start=1
else:
    all_seqs = np.concatenate((all_seqs,x_j_18), axis=0)
    y = np.concatenate((y,y_j_18),axis=0)
    p = np.concatenate((p,p_j_18), axis=0)
shape_y = y_j_19.shape[0]
shape_p = p_j_19.shape[0]
shp=(x_j_19.shape)[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_j_19 =   np.delete(x_j_19, [k for k in range(nece_shp,shp)], None)
x_j_19 = x_j_19.reshape(-1,sensors)
shape_x = x_j_19.shape[0]
x_j_19 = np.reshape(x_j_19, (-1,segement_time_size, np.shape(x_j_19)[1]))
y_j_19= np.delete( y_j_19, [k for k in range(x_j_19.shape[0],shape_y)], None)
p_j_19 = np.delete(p_j_19,[k for k in range(x_j_19.shape[0],shape_p)], None)
if start==0:
    all_seqs = x_j_19
    y = y_j_19
    p = p_j_19
    start=1
else:
    all_seqs = np.concatenate((all_seqs,x_j_19), axis=0)
    y = np.concatenate((y,y_j_19),axis=0)
    p = np.concatenate((p,p_j_19), axis=0)
shape_y = y_j_20.shape[0]
shape_p = p_j_20.shape[0]
shp=(x_j_20.shape)[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_j_20 =   np.delete(x_j_20, [k for k in range(nece_shp,shp)], None)
x_j_20 = x_j_20.reshape(-1,sensors)
shape_x = x_j_20.shape[0]
x_j_20 = np.reshape(x_j_20, (-1,segement_time_size, np.shape(x_j_20)[1]))
y_j_20= np.delete( y_j_20, [k for k in range(x_j_20.shape[0],shape_y)], None)
p_j_20 = np.delete(p_j_20,[k for k in range(x_j_20.shape[0],shape_p)], None)
if start==0:
    all_seqs = x_j_20
    y = y_j_20
    p = p_j_20
    start=1
else:
    all_seqs = np.concatenate((all_seqs,x_j_20), axis=0)
    y = np.concatenate((y,y_j_20),axis=0)
    p = np.concatenate((p,p_j_20), axis=0)
shape_y = y_j_21.shape[0]
shape_p = p_j_21.shape[0]
shp=(x_j_21.shape)[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_j_21 =   np.delete(x_j_21, [k for k in range(nece_shp,shp)], None)
x_j_21 = x_j_21.reshape(-1,sensors)
shape_x = x_j_21.shape[0]
x_j_21 = np.reshape(x_j_21, (-1,segement_time_size, np.shape(x_j_21)[1]))
y_j_21= np.delete( y_j_21, [k for k in range(x_j_21.shape[0],shape_y)], None)
p_j_21 = np.delete(p_j_21,[k for k in range(x_j_21.shape[0],shape_p)], None)
if start==0:
    all_seqs = x_j_21
    y = y_j_21
    p = p_j_21
    start=1
else:
    all_seqs = np.concatenate((all_seqs,x_j_21), axis=0)
    y = np.concatenate((y,y_j_21),axis=0)
    p = np.concatenate((p,p_j_21), axis=0)
shape_y = y_j_22.shape[0]
shape_p = p_j_22.shape[0]
shp=(x_j_22.shape)[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_j_22 =   np.delete(x_j_22, [k for k in range(nece_shp,shp)], None)
x_j_22 = x_j_22.reshape(-1,sensors)
shape_x = x_j_22.shape[0]
x_j_22 = np.reshape(x_j_22, (-1,segement_time_size, np.shape(x_j_22)[1]))
y_j_22= np.delete( y_j_22, [k for k in range(x_j_22.shape[0],shape_y)], None)
p_j_22 = np.delete(p_j_22,[k for k in range(x_j_22.shape[0],shape_p)], None)
if start==0:
    all_seqs = x_j_22
    y = y_j_22
    p = p_j_22
    start=1
else:
    all_seqs = np.concatenate((all_seqs,x_j_22), axis=0)
    y = np.concatenate((y,y_j_22),axis=0)
    p = np.concatenate((p,p_j_22), axis=0)
shape_y = y_j_23.shape[0]
shape_p = p_j_23.shape[0]
shp=(x_j_23.shape)[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_j_23 =   np.delete(x_j_23, [k for k in range(nece_shp,shp)], None)
x_j_23 = x_j_23.reshape(-1,sensors)
shape_x = x_j_23.shape[0]
x_j_23 = np.reshape(x_j_23, (-1,segement_time_size, np.shape(x_j_23)[1]))
y_j_23= np.delete( y_j_23, [k for k in range(x_j_23.shape[0],shape_y)], None)
p_j_23 = np.delete(p_j_23,[k for k in range(x_j_23.shape[0],shape_p)], None)
if start==0:
    all_seqs = x_j_23
    y = y_j_23
    p = p_j_23
    start=1
else:
    all_seqs = np.concatenate((all_seqs,x_j_23), axis=0)
    y = np.concatenate((y,y_j_23),axis=0)
    p = np.concatenate((p,p_j_23), axis=0)
shape_y = y_j_24.shape[0]
shape_p = p_j_24.shape[0]
shp=(x_j_24.shape)[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_j_24 =   np.delete(x_j_24, [k for k in range(nece_shp,shp)], None)
x_j_24 = x_j_24.reshape(-1,sensors)
shape_x = x_j_24.shape[0]
x_j_24 = np.reshape(x_j_24, (-1,segement_time_size, np.shape(x_j_24)[1]))
y_j_24= np.delete( y_j_24, [k for k in range(x_j_24.shape[0],shape_y)], None)
p_j_24 = np.delete(p_j_24,[k for k in range(x_j_24.shape[0],shape_p)], None)
if start==0:
    all_seqs = x_j_24
    y = y_j_24
    p = p_j_24
    start=1
else:
    all_seqs = np.concatenate((all_seqs,x_j_24), axis=0)
    y = np.concatenate((y,y_j_24),axis=0)
    p = np.concatenate((p,p_j_24), axis=0)
shape_y = y_j_25.shape[0]
shape_p = p_j_25.shape[0]
shp=(x_j_25.shape)[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_j_25 =   np.delete(x_j_25, [k for k in range(nece_shp,shp)], None)
x_j_25 = x_j_25.reshape(-1,sensors)
shape_x = x_j_25.shape[0]
x_j_25 = np.reshape(x_j_25, (-1,segement_time_size, np.shape(x_j_25)[1]))
y_j_25= np.delete( y_j_25, [k for k in range(x_j_25.shape[0],shape_y)], None)
p_j_25 = np.delete(p_j_25,[k for k in range(x_j_25.shape[0],shape_p)], None)
if start==0:
    all_seqs = x_j_25
    y = y_j_25
    p = p_j_25
    start=1
else:
    all_seqs = np.concatenate((all_seqs,x_j_25), axis=0)
    y = np.concatenate((y,y_j_25),axis=0)
    p = np.concatenate((p,p_j_25), axis=0)
shape_y = y_j_26.shape[0]
shape_p = p_j_26.shape[0]
shp=(x_j_26.shape)[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_j_26 =   np.delete(x_j_26, [k for k in range(nece_shp,shp)], None)
x_j_26 = x_j_26.reshape(-1,sensors)
shape_x = x_j_26.shape[0]
x_j_26 = np.reshape(x_j_26, (-1,segement_time_size, np.shape(x_j_26)[1]))
y_j_26= np.delete( y_j_26, [k for k in range(x_j_26.shape[0],shape_y)], None)
p_j_26 = np.delete(p_j_26,[k for k in range(x_j_26.shape[0],shape_p)], None)
if start==0:
    all_seqs = x_j_26
    y = y_j_26
    p = p_j_26
    start=1
else:
    all_seqs = np.concatenate((all_seqs,x_j_26), axis=0)
    y = np.concatenate((y,y_j_26),axis=0)
    p = np.concatenate((p,p_j_26), axis=0)
shape_y = y_j_27.shape[0]
shape_p = p_j_27.shape[0]
shp=(x_j_27.shape)[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_j_27 =   np.delete(x_j_27, [k for k in range(nece_shp,shp)], None)
x_j_27 = x_j_27.reshape(-1,sensors)
shape_x = x_j_27.shape[0]
x_j_27 = np.reshape(x_j_27, (-1,segement_time_size, np.shape(x_j_27)[1]))
y_j_27= np.delete( y_j_27, [k for k in range(x_j_27.shape[0],shape_y)], None)
p_j_27 = np.delete(p_j_27,[k for k in range(x_j_27.shape[0],shape_p)], None)
if start==0:
    all_seqs = x_j_27
    y = y_j_27
    p = p_j_27
    start=1
else:
    all_seqs = np.concatenate((all_seqs,x_j_27), axis=0)
    y = np.concatenate((y,y_j_27),axis=0)
    p = np.concatenate((p,p_j_27), axis=0)
shape_y = y_j_28.shape[0]
shape_p = p_j_28.shape[0]
shp=(x_j_28.shape)[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_j_28 =   np.delete(x_j_28, [k for k in range(nece_shp,shp)], None)
x_j_28 = x_j_28.reshape(-1,sensors)
shape_x = x_j_28.shape[0]
x_j_28 = np.reshape(x_j_28, (-1,segement_time_size, np.shape(x_j_28)[1]))
y_j_28= np.delete( y_j_28, [k for k in range(x_j_28.shape[0],shape_y)], None)
p_j_28 = np.delete(p_j_28,[k for k in range(x_j_28.shape[0],shape_p)], None)
if start==0:
    all_seqs = x_j_28
    y = y_j_28
    p = p_j_28
    start=1
else:
    all_seqs = np.concatenate((all_seqs,x_j_28), axis=0)
    y = np.concatenate((y,y_j_28),axis=0)
    p = np.concatenate((p,p_j_28), axis=0)
shape_y = y_j_29.shape[0]
shape_p = p_j_29.shape[0]
shp=(x_j_29.shape)[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_j_29 =   np.delete(x_j_29, [k for k in range(nece_shp,shp)], None)
x_j_29 = x_j_29.reshape(-1,sensors)
shape_x = x_j_29.shape[0]
x_j_29 = np.reshape(x_j_29, (-1,segement_time_size, np.shape(x_j_29)[1]))
y_j_29= np.delete( y_j_29, [k for k in range(x_j_29.shape[0],shape_y)], None)
p_j_29 = np.delete(p_j_29,[k for k in range(x_j_29.shape[0],shape_p)], None)
if start==0:
    all_seqs = x_j_29
    y = y_j_29
    p = p_j_29
    start=1
else:
    all_seqs = np.concatenate((all_seqs,x_j_29), axis=0)
    y = np.concatenate((y,y_j_29),axis=0)
    p = np.concatenate((p,p_j_29), axis=0)
shape_y = y_j_30.shape[0]
shape_p = p_j_30.shape[0]
shp=(x_j_30.shape)[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_j_30 =   np.delete(x_j_30, [k for k in range(nece_shp,shp)], None)
x_j_30 = x_j_30.reshape(-1,sensors)
shape_x = x_j_30.shape[0]
x_j_30 = np.reshape(x_j_30, (-1,segement_time_size, np.shape(x_j_30)[1]))
y_j_30= np.delete( y_j_30, [k for k in range(x_j_30.shape[0],shape_y)], None)
p_j_30 = np.delete(p_j_30,[k for k in range(x_j_30.shape[0],shape_p)], None)
if start==0:
    all_seqs = x_j_30
    y = y_j_30
    p = p_j_30
    start=1
else:
    all_seqs = np.concatenate((all_seqs,x_j_30), axis=0)
    y = np.concatenate((y,y_j_30),axis=0)
    p = np.concatenate((p,p_j_30), axis=0)
shape_y = y_j_31.shape[0]
shape_p = p_j_31.shape[0]
shp=(x_j_31.shape)[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_j_31 =   np.delete(x_j_31, [k for k in range(nece_shp,shp)], None)
x_j_31 = x_j_31.reshape(-1,sensors)
shape_x = x_j_31.shape[0]
x_j_31 = np.reshape(x_j_31, (-1,segement_time_size, np.shape(x_j_31)[1]))
y_j_31= np.delete( y_j_31, [k for k in range(x_j_31.shape[0],shape_y)], None)
p_j_31 = np.delete(p_j_31,[k for k in range(x_j_31.shape[0],shape_p)], None)
if start==0:
    all_seqs = x_j_31
    y = y_j_31
    p = p_j_31
    start=1
else:
    all_seqs = np.concatenate((all_seqs,x_j_31), axis=0)
    y = np.concatenate((y,y_j_31),axis=0)
    p = np.concatenate((p,p_j_31), axis=0)
shape_y = y_j_32.shape[0]
shape_p = p_j_32.shape[0]
shp=(x_j_32.shape)[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_j_32 =   np.delete(x_j_32, [k for k in range(nece_shp,shp)], None)
x_j_32 = x_j_32.reshape(-1,sensors)
shape_x = x_j_32.shape[0]
x_j_32 = np.reshape(x_j_32, (-1,segement_time_size, np.shape(x_j_32)[1]))
y_j_32= np.delete( y_j_32, [k for k in range(x_j_32.shape[0],shape_y)], None)
p_j_32 = np.delete(p_j_32,[k for k in range(x_j_32.shape[0],shape_p)], None)
if start==0:
    all_seqs = x_j_32
    y = y_j_32
    p = p_j_32
    start=1
else:
    all_seqs = np.concatenate((all_seqs,x_j_32), axis=0)
    y = np.concatenate((y,y_j_32),axis=0)
    p = np.concatenate((p,p_j_32), axis=0)
shape_y = y_j_33.shape[0]
shape_p = p_j_33.shape[0]
shp=(x_j_33.shape)[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_j_33 =   np.delete(x_j_33, [k for k in range(nece_shp,shp)], None)
x_j_33 = x_j_33.reshape(-1,sensors)
shape_x = x_j_33.shape[0]
x_j_33 = np.reshape(x_j_33, (-1,segement_time_size, np.shape(x_j_33)[1]))
y_j_33= np.delete( y_j_33, [k for k in range(x_j_33.shape[0],shape_y)], None)
p_j_33 = np.delete(p_j_33,[k for k in range(x_j_33.shape[0],shape_p)], None)
if start==0:
    all_seqs = x_j_33
    y = y_j_33
    p = p_j_33
    start=1
else:
    all_seqs = np.concatenate((all_seqs,x_j_33), axis=0)
    y = np.concatenate((y,y_j_33),axis=0)
    p = np.concatenate((p,p_j_33), axis=0)
shape_y = y_j_34.shape[0]
shape_p = p_j_34.shape[0]
shp=(x_j_34.shape)[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_j_34 =   np.delete(x_j_34, [k for k in range(nece_shp,shp)], None)
x_j_34 = x_j_34.reshape(-1,sensors)
shape_x = x_j_34.shape[0]
x_j_34 = np.reshape(x_j_34, (-1,segement_time_size, np.shape(x_j_34)[1]))
y_j_34= np.delete( y_j_34, [k for k in range(x_j_34.shape[0],shape_y)], None)
p_j_34 = np.delete(p_j_34,[k for k in range(x_j_34.shape[0],shape_p)], None)
if start==0:
    all_seqs = x_j_34
    y = y_j_34
    p = p_j_34
    start=1
else:
    all_seqs = np.concatenate((all_seqs,x_j_34), axis=0)
    y = np.concatenate((y,y_j_34),axis=0)
    p = np.concatenate((p,p_j_34), axis=0)
shape_y = y_j_35.shape[0]
shape_p = p_j_35.shape[0]
shp=(x_j_35.shape)[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_j_35 =   np.delete(x_j_35, [k for k in range(nece_shp,shp)], None)
x_j_35 = x_j_35.reshape(-1,sensors)
shape_x = x_j_35.shape[0]
x_j_35 = np.reshape(x_j_35, (-1,segement_time_size, np.shape(x_j_35)[1]))
y_j_35= np.delete( y_j_35, [k for k in range(x_j_35.shape[0],shape_y)], None)
p_j_35 = np.delete(p_j_35,[k for k in range(x_j_35.shape[0],shape_p)], None)
if start==0:
    all_seqs = x_j_35
    y = y_j_35
    p = p_j_35
    start=1
else:
    all_seqs = np.concatenate((all_seqs,x_j_35), axis=0)
    y = np.concatenate((y,y_j_35),axis=0)
    p = np.concatenate((p,p_j_35), axis=0)
shape_y = y_j_36.shape[0]
shape_p = p_j_36.shape[0]
shp=(x_j_36.shape)[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_j_36 =   np.delete(x_j_36, [k for k in range(nece_shp,shp)], None)
x_j_36 = x_j_36.reshape(-1,sensors)
shape_x = x_j_36.shape[0]
x_j_36 = np.reshape(x_j_36, (-1,segement_time_size, np.shape(x_j_36)[1]))
y_j_36= np.delete( y_j_36, [k for k in range(x_j_36.shape[0],shape_y)], None)
p_j_36 = np.delete(p_j_36,[k for k in range(x_j_36.shape[0],shape_p)], None)
if start==0:
    all_seqs = x_j_36
    y = y_j_36
    p = p_j_36
    start=1
else:
    all_seqs = np.concatenate((all_seqs,x_j_36), axis=0)
    y = np.concatenate((y,y_j_36),axis=0)
    p = np.concatenate((p,p_j_36), axis=0)

shape_y = y_s_1.shape[0]
shape_p = p_s_1.shape[0]
shp=(x_s_1.shape)[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_s_1 =   np.delete(x_s_1, [k for k in range(nece_shp,shp)], None)
x_s_1 = x_s_1.reshape(-1,sensors)
shape_x = x_s_1.shape[0]
x_s_1 = np.reshape(x_s_1, (-1,segement_time_size, np.shape(x_s_1)[1]))
y_s_1= np.delete( y_s_1, [k for k in range(x_s_1.shape[0],shape_y)], None)
p_s_1 = np.delete(p_s_1,[k for k in range(x_s_1.shape[0],shape_p)], None)
if start==0:
    all_seqs = x_s_1
    y = y_s_1
    p = p_s_1
    start=1
else:
    all_seqs = np.concatenate((all_seqs,x_s_1), axis=0)
    y = np.concatenate((y,y_s_1),axis=0)
    p = np.concatenate((p,p_s_1), axis=0)
shape_y = y_s_2.shape[0]
shape_p = p_s_2.shape[0]
shp=(x_s_2.shape)[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_s_2 =   np.delete(x_s_2, [k for k in range(nece_shp,shp)], None)
x_s_2 = x_s_2.reshape(-1,sensors)
shape_x = x_s_2.shape[0]
x_s_2 = np.reshape(x_s_2, (-1,segement_time_size, np.shape(x_s_2)[1]))
y_s_2= np.delete( y_s_2, [k for k in range(x_s_2.shape[0],shape_y)], None)
p_s_2 = np.delete(p_s_2,[k for k in range(x_s_2.shape[0],shape_p)], None)
if start==0:
    all_seqs = x_s_2
    y = y_s_2
    p = p_s_2
    start=1
else:
    all_seqs = np.concatenate((all_seqs,x_s_2), axis=0)
    y = np.concatenate((y,y_s_2),axis=0)
    p = np.concatenate((p,p_s_2), axis=0)
shape_y = y_s_3.shape[0]
shape_p = p_s_3.shape[0]
shp=(x_s_3.shape)[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_s_3 =   np.delete(x_s_3, [k for k in range(nece_shp,shp)], None)
x_s_3 = x_s_3.reshape(-1,sensors)
shape_x = x_s_3.shape[0]
x_s_3 = np.reshape(x_s_3, (-1,segement_time_size, np.shape(x_s_3)[1]))
y_s_3= np.delete( y_s_3, [k for k in range(x_s_3.shape[0],shape_y)], None)
p_s_3 = np.delete(p_s_3,[k for k in range(x_s_3.shape[0],shape_p)], None)
if start==0:
    all_seqs = x_s_3
    y = y_s_3
    p = p_s_3
    start=1
else:
    all_seqs = np.concatenate((all_seqs,x_s_3), axis=0)
    y = np.concatenate((y,y_s_3),axis=0)
    p = np.concatenate((p,p_s_3), axis=0)
shape_y = y_s_4.shape[0]
shape_p = p_s_4.shape[0]
shp=(x_s_4.shape)[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_s_4 =   np.delete(x_s_4, [k for k in range(nece_shp,shp)], None)
x_s_4 = x_s_4.reshape(-1,sensors)
shape_x = x_s_4.shape[0]
x_s_4 = np.reshape(x_s_4, (-1,segement_time_size, np.shape(x_s_4)[1]))
y_s_4= np.delete( y_s_4, [k for k in range(x_s_4.shape[0],shape_y)], None)
p_s_4 = np.delete(p_s_4,[k for k in range(x_s_4.shape[0],shape_p)], None)
if start==0:
    all_seqs = x_s_4
    y = y_s_4
    p = p_s_4
    start=1
else:
    all_seqs = np.concatenate((all_seqs,x_s_4), axis=0)
    y = np.concatenate((y,y_s_4),axis=0)
    p = np.concatenate((p,p_s_4), axis=0)
shape_y = y_s_5.shape[0]
shape_p = p_s_5.shape[0]
shp=(x_s_5.shape)[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_s_5 =   np.delete(x_s_5, [k for k in range(nece_shp,shp)], None)
x_s_5 = x_s_5.reshape(-1,sensors)
shape_x = x_s_5.shape[0]
x_s_5 = np.reshape(x_s_5, (-1,segement_time_size, np.shape(x_s_5)[1]))
y_s_5= np.delete( y_s_5, [k for k in range(x_s_5.shape[0],shape_y)], None)
p_s_5 = np.delete(p_s_5,[k for k in range(x_s_5.shape[0],shape_p)], None)
if start==0:
    all_seqs = x_s_5
    y = y_s_5
    p = p_s_5
    start=1
else:
    all_seqs = np.concatenate((all_seqs,x_s_5), axis=0)
    y = np.concatenate((y,y_s_5),axis=0)
    p = np.concatenate((p,p_s_5), axis=0)
shape_y = y_s_6.shape[0]
shape_p = p_s_6.shape[0]
shp=(x_s_6.shape)[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_s_6 =   np.delete(x_s_6, [k for k in range(nece_shp,shp)], None)
x_s_6 = x_s_6.reshape(-1,sensors)
shape_x = x_s_6.shape[0]
x_s_6 = np.reshape(x_s_6, (-1,segement_time_size, np.shape(x_s_6)[1]))
y_s_6= np.delete( y_s_6, [k for k in range(x_s_6.shape[0],shape_y)], None)
p_s_6 = np.delete(p_s_6,[k for k in range(x_s_6.shape[0],shape_p)], None)
if start==0:
    all_seqs = x_s_6
    y = y_s_6
    p = p_s_6
    start=1
else:
    all_seqs = np.concatenate((all_seqs,x_s_6), axis=0)
    y = np.concatenate((y,y_s_6),axis=0)
    p = np.concatenate((p,p_s_6), axis=0)
shape_y = y_s_7.shape[0]
shape_p = p_s_7.shape[0]
shp=(x_s_7.shape)[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_s_7 =   np.delete(x_s_7, [k for k in range(nece_shp,shp)], None)
x_s_7 = x_s_7.reshape(-1,sensors)
shape_x = x_s_7.shape[0]
x_s_7 = np.reshape(x_s_7, (-1,segement_time_size, np.shape(x_s_7)[1]))
y_s_7= np.delete( y_s_7, [k for k in range(x_s_7.shape[0],shape_y)], None)
p_s_7 = np.delete(p_s_7,[k for k in range(x_s_7.shape[0],shape_p)], None)
if start==0:
    all_seqs = x_s_7
    y = y_s_7
    p = p_s_7
    start=1
else:
    all_seqs = np.concatenate((all_seqs,x_s_7), axis=0)
    y = np.concatenate((y,y_s_7),axis=0)
    p = np.concatenate((p,p_s_7), axis=0)
shape_y = y_s_8.shape[0]
shape_p = p_s_8.shape[0]
shp=(x_s_8.shape)[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_s_8 =   np.delete(x_s_8, [k for k in range(nece_shp,shp)], None)
x_s_8 = x_s_8.reshape(-1,sensors)
shape_x = x_s_8.shape[0]
x_s_8 = np.reshape(x_s_8, (-1,segement_time_size, np.shape(x_s_8)[1]))
y_s_8= np.delete( y_s_8, [k for k in range(x_s_8.shape[0],shape_y)], None)
p_s_8 = np.delete(p_s_8,[k for k in range(x_s_8.shape[0],shape_p)], None)
if start==0:
    all_seqs = x_s_8
    y = y_s_8
    p = p_s_8
    start=1
else:
    all_seqs = np.concatenate((all_seqs,x_s_8), axis=0)
    y = np.concatenate((y,y_s_8),axis=0)
    p = np.concatenate((p,p_s_8), axis=0)
shape_y = y_s_9.shape[0]
shape_p = p_s_9.shape[0]
shp=(x_s_9.shape)[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_s_9 =   np.delete(x_s_9, [k for k in range(nece_shp,shp)], None)
x_s_9 = x_s_9.reshape(-1,sensors)
shape_x = x_s_9.shape[0]
x_s_9 = np.reshape(x_s_9, (-1,segement_time_size, np.shape(x_s_9)[1]))
y_s_9= np.delete( y_s_9, [k for k in range(x_s_9.shape[0],shape_y)], None)
p_s_9 = np.delete(p_s_9,[k for k in range(x_s_9.shape[0],shape_p)], None)
if start==0:
    all_seqs = x_s_9
    y = y_s_9
    p = p_s_9
    start=1
else:
    all_seqs = np.concatenate((all_seqs,x_s_9), axis=0)
    y = np.concatenate((y,y_s_9),axis=0)
    p = np.concatenate((p,p_s_9), axis=0)
shape_y = y_s_10.shape[0]
shape_p = p_s_10.shape[0]
shp=(x_s_10.shape)[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_s_10 =   np.delete(x_s_10, [k for k in range(nece_shp,shp)], None)
x_s_10 = x_s_10.reshape(-1,sensors)
shape_x = x_s_10.shape[0]
x_s_10 = np.reshape(x_s_10, (-1,segement_time_size, np.shape(x_s_10)[1]))
y_s_10= np.delete( y_s_10, [k for k in range(x_s_10.shape[0],shape_y)], None)
p_s_10 = np.delete(p_s_10,[k for k in range(x_s_10.shape[0],shape_p)], None)
if start==0:
    all_seqs = x_s_10
    y = y_s_10
    p = p_s_10
    start=1
else:
    all_seqs = np.concatenate((all_seqs,x_s_10), axis=0)
    y = np.concatenate((y,y_s_10),axis=0)
    p = np.concatenate((p,p_s_10), axis=0)
shape_y = y_s_11.shape[0]
shape_p = p_s_11.shape[0]
shp=(x_s_11.shape)[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_s_11 =   np.delete(x_s_11, [k for k in range(nece_shp,shp)], None)
x_s_11 = x_s_11.reshape(-1,sensors)
shape_x = x_s_11.shape[0]
x_s_11 = np.reshape(x_s_11, (-1,segement_time_size, np.shape(x_s_11)[1]))
y_s_11= np.delete( y_s_11, [k for k in range(x_s_11.shape[0],shape_y)], None)
p_s_11 = np.delete(p_s_11,[k for k in range(x_s_11.shape[0],shape_p)], None)
if start==0:
    all_seqs = x_s_11
    y = y_s_11
    p = p_s_11
    start=1
else:
    all_seqs = np.concatenate((all_seqs,x_s_11), axis=0)
    y = np.concatenate((y,y_s_11),axis=0)
    p = np.concatenate((p,p_s_11), axis=0)
shape_y = y_s_12.shape[0]
shape_p = p_s_12.shape[0]
shp=(x_s_12.shape)[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_s_12 =   np.delete(x_s_12, [k for k in range(nece_shp,shp)], None)
x_s_12 = x_s_12.reshape(-1,sensors)
shape_x = x_s_12.shape[0]
x_s_12 = np.reshape(x_s_12, (-1,segement_time_size, np.shape(x_s_12)[1]))
y_s_12= np.delete( y_s_12, [k for k in range(x_s_12.shape[0],shape_y)], None)
p_s_12 = np.delete(p_s_12,[k for k in range(x_s_12.shape[0],shape_p)], None)
if start==0:
    all_seqs = x_s_12
    y = y_s_12
    p = p_s_12
    start=1
else:
    all_seqs = np.concatenate((all_seqs,x_s_12), axis=0)
    y = np.concatenate((y,y_s_12),axis=0)
    p = np.concatenate((p,p_s_12), axis=0)
shape_y = y_s_13.shape[0]
shape_p = p_s_13.shape[0]
shp=(x_s_13.shape)[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_s_13 =   np.delete(x_s_13, [k for k in range(nece_shp,shp)], None)
x_s_13 = x_s_13.reshape(-1,sensors)
shape_x = x_s_13.shape[0]
x_s_13 = np.reshape(x_s_13, (-1,segement_time_size, np.shape(x_s_13)[1]))
y_s_13= np.delete( y_s_13, [k for k in range(x_s_13.shape[0],shape_y)], None)
p_s_13 = np.delete(p_s_13,[k for k in range(x_s_13.shape[0],shape_p)], None)
if start==0:
    all_seqs = x_s_13
    y = y_s_13
    p = p_s_13
    start=1
else:
    all_seqs = np.concatenate((all_seqs,x_s_13), axis=0)
    y = np.concatenate((y,y_s_13),axis=0)
    p = np.concatenate((p,p_s_13), axis=0)
shape_y = y_s_14.shape[0]
shape_p = p_s_14.shape[0]
shp=(x_s_14.shape)[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_s_14 =   np.delete(x_s_14, [k for k in range(nece_shp,shp)], None)
x_s_14 = x_s_14.reshape(-1,sensors)
shape_x = x_s_14.shape[0]
x_s_14 = np.reshape(x_s_14, (-1,segement_time_size, np.shape(x_s_14)[1]))
y_s_14= np.delete( y_s_14, [k for k in range(x_s_14.shape[0],shape_y)], None)
p_s_14 = np.delete(p_s_14,[k for k in range(x_s_14.shape[0],shape_p)], None)
if start==0:
    all_seqs = x_s_14
    y = y_s_14
    p = p_s_14
    start=1
else:
    all_seqs = np.concatenate((all_seqs,x_s_14), axis=0)
    y = np.concatenate((y,y_s_14),axis=0)
    p = np.concatenate((p,p_s_14), axis=0)
shape_y = y_s_15.shape[0]
shape_p = p_s_15.shape[0]
shp=(x_s_15.shape)[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_s_15 =   np.delete(x_s_15, [k for k in range(nece_shp,shp)], None)
x_s_15 = x_s_15.reshape(-1,sensors)
shape_x = x_s_15.shape[0]
x_s_15 = np.reshape(x_s_15, (-1,segement_time_size, np.shape(x_s_15)[1]))
y_s_15= np.delete( y_s_15, [k for k in range(x_s_15.shape[0],shape_y)], None)
p_s_15 = np.delete(p_s_15,[k for k in range(x_s_15.shape[0],shape_p)], None)
if start==0:
    all_seqs = x_s_15
    y = y_s_15
    p = p_s_15
    start=1
else:
    all_seqs = np.concatenate((all_seqs,x_s_15), axis=0)
    y = np.concatenate((y,y_s_15),axis=0)
    p = np.concatenate((p,p_s_15), axis=0)
shape_y = y_s_16.shape[0]
shape_p = p_s_16.shape[0]
shp=(x_s_16.shape)[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_s_16 =   np.delete(x_s_16, [k for k in range(nece_shp,shp)], None)
x_s_16 = x_s_16.reshape(-1,sensors)
shape_x = x_s_16.shape[0]
x_s_16 = np.reshape(x_s_16, (-1,segement_time_size, np.shape(x_s_16)[1]))
y_s_16= np.delete( y_s_16, [k for k in range(x_s_16.shape[0],shape_y)], None)
p_s_16 = np.delete(p_s_16,[k for k in range(x_s_16.shape[0],shape_p)], None)
if start==0:
    all_seqs = x_s_16
    y = y_s_16
    p = p_s_16
    start=1
else:
    all_seqs = np.concatenate((all_seqs,x_s_16), axis=0)
    y = np.concatenate((y,y_s_16),axis=0)
    p = np.concatenate((p,p_s_16), axis=0)
shape_y = y_s_17.shape[0]
shape_p = p_s_17.shape[0]
shp=(x_s_17.shape)[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_s_17 =   np.delete(x_s_17, [k for k in range(nece_shp,shp)], None)
x_s_17 = x_s_17.reshape(-1,sensors)
shape_x = x_s_17.shape[0]
x_s_17 = np.reshape(x_s_17, (-1,segement_time_size, np.shape(x_s_17)[1]))
y_s_17= np.delete( y_s_17, [k for k in range(x_s_17.shape[0],shape_y)], None)
p_s_17 = np.delete(p_s_17,[k for k in range(x_s_17.shape[0],shape_p)], None)
if start==0:
    all_seqs = x_s_17
    y = y_s_17
    p = p_s_17
    start=1
else:
    all_seqs = np.concatenate((all_seqs,x_s_17), axis=0)
    y = np.concatenate((y,y_s_17),axis=0)
    p = np.concatenate((p,p_s_17), axis=0)
shape_y = y_s_18.shape[0]
shape_p = p_s_18.shape[0]
shp=(x_s_18.shape)[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_s_18 =   np.delete(x_s_18, [k for k in range(nece_shp,shp)], None)
x_s_18 = x_s_18.reshape(-1,sensors)
shape_x = x_s_18.shape[0]
x_s_18 = np.reshape(x_s_18, (-1,segement_time_size, np.shape(x_s_18)[1]))
y_s_18= np.delete( y_s_18, [k for k in range(x_s_18.shape[0],shape_y)], None)
p_s_18 = np.delete(p_s_18,[k for k in range(x_s_18.shape[0],shape_p)], None)
if start==0:
    all_seqs = x_s_18
    y = y_s_18
    p = p_s_18
    start=1
else:
    all_seqs = np.concatenate((all_seqs,x_s_18), axis=0)
    y = np.concatenate((y,y_s_18),axis=0)
    p = np.concatenate((p,p_s_18), axis=0)
shape_y = y_s_19.shape[0]
shape_p = p_s_19.shape[0]
shp=(x_s_19.shape)[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_s_19 =   np.delete(x_s_19, [k for k in range(nece_shp,shp)], None)
x_s_19 = x_s_19.reshape(-1,sensors)
shape_x = x_s_19.shape[0]
x_s_19 = np.reshape(x_s_19, (-1,segement_time_size, np.shape(x_s_19)[1]))
y_s_19= np.delete( y_s_19, [k for k in range(x_s_19.shape[0],shape_y)], None)
p_s_19 = np.delete(p_s_19,[k for k in range(x_s_19.shape[0],shape_p)], None)
if start==0:
    all_seqs = x_s_19
    y = y_s_19
    p = p_s_19
    start=1
else:
    all_seqs = np.concatenate((all_seqs,x_s_19), axis=0)
    y = np.concatenate((y,y_s_19),axis=0)
    p = np.concatenate((p,p_s_19), axis=0)
shape_y = y_s_20.shape[0]
shape_p = p_s_20.shape[0]
shp=(x_s_20.shape)[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_s_20 =   np.delete(x_s_20, [k for k in range(nece_shp,shp)], None)
x_s_20 = x_s_20.reshape(-1,sensors)
shape_x = x_s_20.shape[0]
x_s_20 = np.reshape(x_s_20, (-1,segement_time_size, np.shape(x_s_20)[1]))
y_s_20= np.delete( y_s_20, [k for k in range(x_s_20.shape[0],shape_y)], None)
p_s_20 = np.delete(p_s_20,[k for k in range(x_s_20.shape[0],shape_p)], None)
if start==0:
    all_seqs = x_s_20
    y = y_s_20
    p = p_s_20
    start=1
else:
    all_seqs = np.concatenate((all_seqs,x_s_20), axis=0)
    y = np.concatenate((y,y_s_20),axis=0)
    p = np.concatenate((p,p_s_20), axis=0)
shape_y = y_s_21.shape[0]
shape_p = p_s_21.shape[0]
shp=(x_s_21.shape)[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_s_21 =   np.delete(x_s_21, [k for k in range(nece_shp,shp)], None)
x_s_21 = x_s_21.reshape(-1,sensors)
shape_x = x_s_21.shape[0]
x_s_21 = np.reshape(x_s_21, (-1,segement_time_size, np.shape(x_s_21)[1]))
y_s_21= np.delete( y_s_21, [k for k in range(x_s_21.shape[0],shape_y)], None)
p_s_21 = np.delete(p_s_21,[k for k in range(x_s_21.shape[0],shape_p)], None)
if start==0:
    all_seqs = x_s_21
    y = y_s_21
    p = p_s_21
    start=1
else:
    all_seqs = np.concatenate((all_seqs,x_s_21), axis=0)
    y = np.concatenate((y,y_s_21),axis=0)
    p = np.concatenate((p,p_s_21), axis=0)
shape_y = y_s_22.shape[0]
shape_p = p_s_22.shape[0]
shp=(x_s_22.shape)[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_s_22 =   np.delete(x_s_22, [k for k in range(nece_shp,shp)], None)
x_s_22 = x_s_22.reshape(-1,sensors)
shape_x = x_s_22.shape[0]
x_s_22 = np.reshape(x_s_22, (-1,segement_time_size, np.shape(x_s_22)[1]))
y_s_22= np.delete( y_s_22, [k for k in range(x_s_22.shape[0],shape_y)], None)
p_s_22 = np.delete(p_s_22,[k for k in range(x_s_22.shape[0],shape_p)], None)
if start==0:
    all_seqs = x_s_22
    y = y_s_22
    p = p_s_22
    start=1
else:
    all_seqs = np.concatenate((all_seqs,x_s_22), axis=0)
    y = np.concatenate((y,y_s_22),axis=0)
    p = np.concatenate((p,p_s_22), axis=0)
shape_y = y_s_23.shape[0]
shape_p = p_s_23.shape[0]
shp=(x_s_23.shape)[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_s_23 =   np.delete(x_s_23, [k for k in range(nece_shp,shp)], None)
x_s_23 = x_s_23.reshape(-1,sensors)
shape_x = x_s_23.shape[0]
x_s_23 = np.reshape(x_s_23, (-1,segement_time_size, np.shape(x_s_23)[1]))
y_s_23= np.delete( y_s_23, [k for k in range(x_s_23.shape[0],shape_y)], None)
p_s_23 = np.delete(p_s_23,[k for k in range(x_s_23.shape[0],shape_p)], None)
if start==0:
    all_seqs = x_s_23
    y = y_s_23
    p = p_s_23
    start=1
else:
    all_seqs = np.concatenate((all_seqs,x_s_23), axis=0)
    y = np.concatenate((y,y_s_23),axis=0)
    p = np.concatenate((p,p_s_23), axis=0)
shape_y = y_s_24.shape[0]
shape_p = p_s_24.shape[0]
shp=(x_s_24.shape)[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_s_24 =   np.delete(x_s_24, [k for k in range(nece_shp,shp)], None)
x_s_24 = x_s_24.reshape(-1,sensors)
shape_x = x_s_24.shape[0]
x_s_24 = np.reshape(x_s_24, (-1,segement_time_size, np.shape(x_s_24)[1]))
y_s_24= np.delete( y_s_24, [k for k in range(x_s_24.shape[0],shape_y)], None)
p_s_24 = np.delete(p_s_24,[k for k in range(x_s_24.shape[0],shape_p)], None)
if start==0:
    all_seqs = x_s_24
    y = y_s_24
    p = p_s_24
    start=1
else:
    all_seqs = np.concatenate((all_seqs,x_s_24), axis=0)
    y = np.concatenate((y,y_s_24),axis=0)
    p = np.concatenate((p,p_s_24), axis=0)
shape_y = y_s_25.shape[0]
shape_p = p_s_25.shape[0]
shp=(x_s_25.shape)[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_s_25 =   np.delete(x_s_25, [k for k in range(nece_shp,shp)], None)
x_s_25 = x_s_25.reshape(-1,sensors)
shape_x = x_s_25.shape[0]
x_s_25 = np.reshape(x_s_25, (-1,segement_time_size, np.shape(x_s_25)[1]))
y_s_25= np.delete( y_s_25, [k for k in range(x_s_25.shape[0],shape_y)], None)
p_s_25 = np.delete(p_s_25,[k for k in range(x_s_25.shape[0],shape_p)], None)
if start==0:
    all_seqs = x_s_25
    y = y_s_25
    p = p_s_25
    start=1
else:
    all_seqs = np.concatenate((all_seqs,x_s_25), axis=0)
    y = np.concatenate((y,y_s_25),axis=0)
    p = np.concatenate((p,p_s_25), axis=0)
shape_y = y_s_26.shape[0]
shape_p = p_s_26.shape[0]
shp=(x_s_26.shape)[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_s_26 =   np.delete(x_s_26, [k for k in range(nece_shp,shp)], None)
x_s_26 = x_s_26.reshape(-1,sensors)
shape_x = x_s_26.shape[0]
x_s_26 = np.reshape(x_s_26, (-1,segement_time_size, np.shape(x_s_26)[1]))
y_s_26= np.delete( y_s_26, [k for k in range(x_s_26.shape[0],shape_y)], None)
p_s_26 = np.delete(p_s_26,[k for k in range(x_s_26.shape[0],shape_p)], None)
if start==0:
    all_seqs = x_s_26
    y = y_s_26
    p = p_s_26
    start=1
else:
    all_seqs = np.concatenate((all_seqs,x_s_26), axis=0)
    y = np.concatenate((y,y_s_26),axis=0)
    p = np.concatenate((p,p_s_26), axis=0)
shape_y = y_s_27.shape[0]
shape_p = p_s_27.shape[0]
shp=(x_s_27.shape)[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_s_27 =   np.delete(x_s_27, [k for k in range(nece_shp,shp)], None)
x_s_27 = x_s_27.reshape(-1,sensors)
shape_x = x_s_27.shape[0]
x_s_27 = np.reshape(x_s_27, (-1,segement_time_size, np.shape(x_s_27)[1]))
y_s_27= np.delete( y_s_27, [k for k in range(x_s_27.shape[0],shape_y)], None)
p_s_27 = np.delete(p_s_27,[k for k in range(x_s_27.shape[0],shape_p)], None)
if start==0:
    all_seqs = x_s_27
    y = y_s_27
    p = p_s_27
    start=1
else:
    all_seqs = np.concatenate((all_seqs,x_s_27), axis=0)
    y = np.concatenate((y,y_s_27),axis=0)
    p = np.concatenate((p,p_s_27), axis=0)
shape_y = y_s_28.shape[0]
shape_p = p_s_28.shape[0]
shp=(x_s_28.shape)[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_s_28 =   np.delete(x_s_28, [k for k in range(nece_shp,shp)], None)
x_s_28 = x_s_28.reshape(-1,sensors)
shape_x = x_s_28.shape[0]
x_s_28 = np.reshape(x_s_28, (-1,segement_time_size, np.shape(x_s_28)[1]))
y_s_28= np.delete( y_s_28, [k for k in range(x_s_28.shape[0],shape_y)], None)
p_s_28 = np.delete(p_s_28,[k for k in range(x_s_28.shape[0],shape_p)], None)
if start==0:
    all_seqs = x_s_28
    y = y_s_28
    p = p_s_28
    start=1
else:
    all_seqs = np.concatenate((all_seqs,x_s_28), axis=0)
    y = np.concatenate((y,y_s_28),axis=0)
    p = np.concatenate((p,p_s_28), axis=0)
shape_y = y_s_29.shape[0]
shape_p = p_s_29.shape[0]
shp=(x_s_29.shape)[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_s_29 =   np.delete(x_s_29, [k for k in range(nece_shp,shp)], None)
x_s_29 = x_s_29.reshape(-1,sensors)
shape_x = x_s_29.shape[0]
x_s_29 = np.reshape(x_s_29, (-1,segement_time_size, np.shape(x_s_29)[1]))
y_s_29= np.delete( y_s_29, [k for k in range(x_s_29.shape[0],shape_y)], None)
p_s_29 = np.delete(p_s_29,[k for k in range(x_s_29.shape[0],shape_p)], None)
if start==0:
    all_seqs = x_s_29
    y = y_s_29
    p = p_s_29
    start=1
else:
    all_seqs = np.concatenate((all_seqs,x_s_29), axis=0)
    y = np.concatenate((y,y_s_29),axis=0)
    p = np.concatenate((p,p_s_29), axis=0)
shape_y = y_s_30.shape[0]
shape_p = p_s_30.shape[0]
shp=(x_s_30.shape)[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_s_30 =   np.delete(x_s_30, [k for k in range(nece_shp,shp)], None)
x_s_30 = x_s_30.reshape(-1,sensors)
shape_x = x_s_30.shape[0]
x_s_30 = np.reshape(x_s_30, (-1,segement_time_size, np.shape(x_s_30)[1]))
y_s_30= np.delete( y_s_30, [k for k in range(x_s_30.shape[0],shape_y)], None)
p_s_30 = np.delete(p_s_30,[k for k in range(x_s_30.shape[0],shape_p)], None)
if start==0:
    all_seqs = x_s_30
    y = y_s_30
    p = p_s_30
    start=1
else:
    all_seqs = np.concatenate((all_seqs,x_s_30), axis=0)
    y = np.concatenate((y,y_s_30),axis=0)
    p = np.concatenate((p,p_s_30), axis=0)
shape_y = y_s_31.shape[0]
shape_p = p_s_31.shape[0]
shp=(x_s_31.shape)[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_s_31 =   np.delete(x_s_31, [k for k in range(nece_shp,shp)], None)
x_s_31 = x_s_31.reshape(-1,sensors)
shape_x = x_s_31.shape[0]
x_s_31 = np.reshape(x_s_31, (-1,segement_time_size, np.shape(x_s_31)[1]))
y_s_31= np.delete( y_s_31, [k for k in range(x_s_31.shape[0],shape_y)], None)
p_s_31 = np.delete(p_s_31,[k for k in range(x_s_31.shape[0],shape_p)], None)
if start==0:
    all_seqs = x_s_31
    y = y_s_31
    p = p_s_31
    start=1
else:
    all_seqs = np.concatenate((all_seqs,x_s_31), axis=0)
    y = np.concatenate((y,y_s_31),axis=0)
    p = np.concatenate((p,p_s_31), axis=0)
shape_y = y_s_32.shape[0]
shape_p = p_s_32.shape[0]
shp=(x_s_32.shape)[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_s_32 =   np.delete(x_s_32, [k for k in range(nece_shp,shp)], None)
x_s_32 = x_s_32.reshape(-1,sensors)
shape_x = x_s_32.shape[0]
x_s_32 = np.reshape(x_s_32, (-1,segement_time_size, np.shape(x_s_32)[1]))
y_s_32= np.delete( y_s_32, [k for k in range(x_s_32.shape[0],shape_y)], None)
p_s_32 = np.delete(p_s_32,[k for k in range(x_s_32.shape[0],shape_p)], None)
if start==0:
    all_seqs = x_s_32
    y = y_s_32
    p = p_s_32
    start=1
else:
    all_seqs = np.concatenate((all_seqs,x_s_32), axis=0)
    y = np.concatenate((y,y_s_32),axis=0)
    p = np.concatenate((p,p_s_32), axis=0)
shape_y = y_s_33.shape[0]
shape_p = p_s_33.shape[0]
shp=(x_s_33.shape)[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_s_33 =   np.delete(x_s_33, [k for k in range(nece_shp,shp)], None)
x_s_33 = x_s_33.reshape(-1,sensors)
shape_x = x_s_33.shape[0]
x_s_33 = np.reshape(x_s_33, (-1,segement_time_size, np.shape(x_s_33)[1]))
y_s_33= np.delete( y_s_33, [k for k in range(x_s_33.shape[0],shape_y)], None)
p_s_33 = np.delete(p_s_33,[k for k in range(x_s_33.shape[0],shape_p)], None)
if start==0:
    all_seqs = x_s_33
    y = y_s_33
    p = p_s_33
    start=1
else:
    all_seqs = np.concatenate((all_seqs,x_s_33), axis=0)
    y = np.concatenate((y,y_s_33),axis=0)
    p = np.concatenate((p,p_s_33), axis=0)
shape_y = y_s_34.shape[0]
shape_p = p_s_34.shape[0]
shp=(x_s_34.shape)[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_s_34 =   np.delete(x_s_34, [k for k in range(nece_shp,shp)], None)
x_s_34 = x_s_34.reshape(-1,sensors)
shape_x = x_s_34.shape[0]
x_s_34 = np.reshape(x_s_34, (-1,segement_time_size, np.shape(x_s_34)[1]))
y_s_34= np.delete( y_s_34, [k for k in range(x_s_34.shape[0],shape_y)], None)
p_s_34 = np.delete(p_s_34,[k for k in range(x_s_34.shape[0],shape_p)], None)
if start==0:
    all_seqs = x_s_34
    y = y_s_34
    p = p_s_34
    start=1
else:
    all_seqs = np.concatenate((all_seqs,x_s_34), axis=0)
    y = np.concatenate((y,y_s_34),axis=0)
    p = np.concatenate((p,p_s_34), axis=0)
shape_y = y_s_35.shape[0]
shape_p = p_s_35.shape[0]
shp=(x_s_35.shape)[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_s_35 =   np.delete(x_s_35, [k for k in range(nece_shp,shp)], None)
x_s_35 = x_s_35.reshape(-1,sensors)
shape_x = x_s_35.shape[0]
x_s_35 = np.reshape(x_s_35, (-1,segement_time_size, np.shape(x_s_35)[1]))
y_s_35= np.delete( y_s_35, [k for k in range(x_s_35.shape[0],shape_y)], None)
p_s_35 = np.delete(p_s_35,[k for k in range(x_s_35.shape[0],shape_p)], None)
if start==0:
    all_seqs = x_s_35
    y = y_s_35
    p = p_s_35
    start=1
else:
    all_seqs = np.concatenate((all_seqs,x_s_35), axis=0)
    y = np.concatenate((y,y_s_35),axis=0)
    p = np.concatenate((p,p_s_35), axis=0)
shape_y = y_s_36.shape[0]
shape_p = p_s_36.shape[0]
shp=(x_s_36.shape)[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_s_36 =   np.delete(x_s_36, [k for k in range(nece_shp,shp)], None)
x_s_36 = x_s_36.reshape(-1,sensors)
shape_x = x_s_36.shape[0]
x_s_36 = np.reshape(x_s_36, (-1,segement_time_size, np.shape(x_s_36)[1]))
y_s_36= np.delete( y_s_36, [k for k in range(x_s_36.shape[0],shape_y)], None)
p_s_36 = np.delete(p_s_36,[k for k in range(x_s_36.shape[0],shape_p)], None)
if start==0:
    all_seqs = x_s_36
    y = y_s_36
    p = p_s_36
    start=1
else:
    all_seqs = np.concatenate((all_seqs,x_s_36), axis=0)
    y = np.concatenate((y,y_s_36),axis=0)
    p = np.concatenate((p,p_s_36), axis=0)


#
y = np.asarray(y)
y_onehot = to_categorical(np.asarray(y))
p = np.asarray(p)
p_onehot = to_categorical(np.asarray(p))
all_seqs = np.reshape(all_seqs, (-1, segement_time_size*sensors))
all_seqs= preprocessing.normalize(all_seqs)
all_seqs = np.reshape(all_seqs, (-1, segement_time_size,sensors))
print(p_onehot.shape)
print(y_onehot.shape)
print(all_seqs.shape)
print(y.shape)
print(p.shape)
#once all of the data has been read in and the files have been appended, save
h5f = h5py.File('RL_acc_wisdm_data.h5', 'w')
h5f.create_dataset('X', data=all_seqs)
h5f.create_dataset('y', data=y)
h5f.create_dataset('p', data=p)
h5f.create_dataset('y_onehot', data=y_onehot)
h5f.create_dataset('p_onehot', data=p_onehot)
h5f.close()
print("finished")



