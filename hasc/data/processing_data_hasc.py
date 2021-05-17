import numpy as np
from pandas import read_csv
from numpy import dstack
import h5py
import csv 
from keras.utils.np_utils import to_categorical
from sklearn import preprocessing

#hasc

path =  r'F:\PHD\GAN\code.12.2020\PGAN1\RL_acc\data_hasc.csv'
segement_time_size = 200
sensors = 3
mvts= [' jog', ' skip', ' walk', ' stUp', ' stDown', ' stay']
mvt = ['j', 's', 'w', 'stu', 'std', 'st' ]
id = [str(i).zfill(2) for i in range(0, 100)]
#id = [i for i in range(0, 100)]

def biggest_multiple(multiple_of, input_number):
    return input_number - input_number % multiple_of
y = []
p = []
x_j_0 = []
y_j_0 = []    
p_j_0 = []   
x_j_1 = []
y_j_1 = []
p_j_1 = []
x_j_2 = []
y_j_2 = []
p_j_2 = []
x_j_3 = []    
y_j_3 = []
p_j_3 = []
x_j_4 = []
y_j_4 = []
p_j_4 = []
x_j_5 = []
y_j_5 = []
p_j_5 = []
x_j_6 = []
y_j_6 = []
p_j_6 = []
x_j_7 = []
y_j_7 = []
p_j_7 = []
x_j_8 = []
y_j_8 = []
p_j_8 = []
x_j_9 = []
y_j_9 = []
p_j_9 = []
x_j_10 = []
y_j_10 = []
p_j_10 = []
x_j_11 = []
y_j_11 = []
p_j_11 = []
x_j_12 = []
y_j_12 = []
p_j_12 = []
x_j_13 = []
y_j_13 = []
p_j_13 = []
x_j_14 = []
y_j_14 = []
p_j_14 = []
x_j_15 = []
y_j_15 = []
p_j_15 = []
x_j_16 = []
y_j_16 = []
p_j_16 = []
x_j_17 = []
y_j_17 = []
p_j_17 = []
x_j_18 = []
y_j_18 = []
p_j_18 = []
x_j_19 = []
y_j_19 = []
p_j_19 = []
x_j_20 = []
y_j_20 = []
p_j_20 = []
x_j_21 = []
y_j_21 = []
p_j_21 = []
x_j_22 = []
y_j_22 = []
p_j_22 = []
x_j_23 = []
y_j_23 = []
p_j_23 = []
x_j_24 = []
y_j_24 = []
p_j_24 = []
x_j_25 = []
y_j_25 = []
p_j_25 = []
x_j_26 = []
y_j_26 = []
p_j_26 = []
x_j_27 = []
y_j_27 = []
p_j_27 = []
x_j_28 = []
y_j_28 = []
p_j_28 = []
x_j_29 = []
y_j_29 = []
p_j_29 = []
x_j_30 = []
y_j_30 = []
p_j_30 = []
x_j_31 = []
y_j_31 = []
p_j_31 = []
x_j_32 = []
y_j_32 = []
p_j_32 = []
x_j_33 = []
y_j_33 = []
p_j_33 = []
x_j_34 = []
y_j_34 = []
p_j_34 = []
x_j_35 = []
y_j_35 = []
p_j_35 = []
x_j_36 = []
y_j_36 = []
p_j_36 = []
x_j_37 = []
y_j_37 = []
p_j_37 = []
x_j_38 = []
y_j_38 = []
p_j_38 = []
x_j_39 = []
y_j_39 = []
p_j_39 = []
x_j_40 = []
y_j_40 = []
p_j_40 = []
x_j_41 = []
y_j_41 = []
p_j_41 = []
x_j_42 = []
y_j_42 = []
p_j_42 = []
x_j_43 = []
y_j_43 = []
p_j_43 = []
x_j_44 = []
y_j_44 = []
p_j_44 = []
x_j_45 = []
y_j_45 = []
p_j_45 = []
x_j_46 = []
y_j_46 = []
p_j_46 = []
x_j_47 = []
y_j_47 = []
p_j_47 = []
x_j_48 = []
y_j_48 = []
p_j_48 = []
x_j_49 = []
y_j_49 = []
p_j_49 = []
x_j_50 = []
y_j_50 = []
p_j_50 = []
x_j_51 = []
y_j_51 = []
p_j_51 = []
x_j_52 = []
y_j_52 = []
p_j_52 = []
x_j_53 = []
y_j_53 = []
p_j_53 = []
x_j_54 = []
y_j_54 = []
p_j_54 = []
x_j_55 = []
y_j_55 = []
p_j_55 = []
x_j_56 = []
y_j_56 = []
p_j_56 = []
x_j_57 = []
y_j_57 = []
p_j_57 = []
x_j_58 = []
y_j_58 = []
p_j_58 = []
x_j_59 = []
y_j_59 = []
p_j_59 = []
x_j_60 = []
y_j_60 = []
p_j_60 = []
x_j_61 = []
y_j_61 = []
p_j_61 = []
x_j_62 = []
y_j_62 = []
p_j_62 = []
x_j_63 = []
y_j_63 = []
p_j_63 = []
x_j_64 = []
y_j_64 = []
p_j_64 = []
x_j_65 = []
y_j_65 = []
p_j_65 = []
x_j_66 = []
y_j_66 = []
p_j_66 = []
x_j_67 = []
y_j_67 = []
p_j_67 = []
x_j_68 = []
y_j_68 = []
p_j_68 = []
x_j_69 = []
y_j_69 = []
p_j_69 = []
x_j_70 = []
y_j_70 = []
p_j_70 = []
x_j_71 = []
y_j_71 = []
p_j_71 = []
x_j_72 = []
y_j_72 = []
p_j_72 = []
x_j_73 = []
y_j_73 = []
p_j_73 = []
x_j_74 = []
y_j_74 = []
p_j_74 = []
x_j_75 = []
y_j_75 = []
p_j_75 = []
x_j_76 = []
y_j_76 = []
p_j_76 = []
x_j_77 = []
y_j_77 = []
p_j_77 = []
x_j_78 = []
y_j_78 = []
p_j_78 = []
x_j_79 = []
y_j_79 = []
p_j_79 = []
x_j_80 = []
y_j_80 = []
p_j_80 = []
x_j_81 = []
y_j_81 = []
p_j_81 = []
x_j_82 = []
y_j_82 = []
p_j_82 = []
x_j_83 = []
y_j_83 = []
p_j_83 = []
x_j_84 = []
y_j_84 = []
p_j_84 = []
x_j_85 = []
y_j_85 = []
p_j_85 = []
x_j_86 = []
y_j_86 = []
p_j_86 = []
x_j_87 = []
y_j_87 = []
p_j_87 = []
x_j_88 = []
y_j_88 = []
p_j_88 = []
x_j_89 = []
y_j_89 = []
p_j_89 = []
x_j_90 = []
y_j_90 = []
p_j_90 = []
x_j_91 = []
y_j_91 = []
p_j_91 = []
x_j_92 = []
y_j_92 = []
p_j_92 = []
x_j_93 = []
y_j_93 = []
p_j_93 = []
x_j_94 = []
y_j_94 = []
p_j_94 = []
x_j_95 = []
y_j_95 = []
p_j_95 = []
x_j_96 = []
y_j_96 = []
p_j_96 = []
x_j_97 = []
y_j_97 = []
p_j_97 = []
x_j_98 = []
y_j_98 = []
p_j_98 = []
x_j_99 = []
y_j_99 = []
p_j_99 = []
x_s_0 = []
x_s_1 = []
x_s_2 = []
x_s_3 = []
x_s_4 = []
x_s_5 = []
x_s_6 = []
x_s_7 = []
x_s_8 = []
x_s_9 = []
x_s_10 = []
x_s_11 = []
x_s_12 = []
x_s_13 = []
x_s_14 = []
x_s_15 = []
x_s_16 = []
x_s_17 = []
x_s_18 = []
x_s_19 = []
x_s_20 = []
x_s_21 = []
x_s_22 = []
x_s_23 = []
x_s_24 = []
x_s_25 = []
x_s_26 = []
x_s_27 = []
x_s_28 = []
x_s_29 = []
x_s_30 = []
x_s_31 = []
x_s_32 = []
x_s_33 = []
x_s_34 = []
x_s_35 = []
x_s_36 = []
x_s_37 = []
x_s_38 = []
x_s_39 = []
x_s_40 = []
x_s_41 = []
x_s_42 = []
x_s_43 = []
x_s_44 = []
x_s_45 = []
x_s_46 = []
x_s_47 = []
x_s_48 = []
x_s_49 = []
x_s_50 = []
x_s_51 = []
x_s_52 = []
x_s_53 = []
x_s_54 = []
x_s_55 = []
x_s_56 = []
x_s_57 = []
x_s_58 = []
x_s_59 = []
x_s_60 = []
x_s_61 = []
x_s_62 = []
x_s_63 = []
x_s_64 = []
x_s_65 = []
x_s_66 = []
x_s_67 = []
x_s_68 = []
x_s_69 = []
x_s_70 = []
x_s_71 = []
x_s_72 = []
x_s_73 = []
x_s_74 = []
x_s_75 = []
x_s_76 = []
x_s_77 = []
x_s_78 = []
x_s_79 = []
x_s_80 = []
x_s_81 = []
x_s_82 = []
x_s_83 = []
x_s_84 = []
x_s_85 = []
x_s_86 = []
x_s_87 = []
x_s_88 = []
x_s_89 = []
x_s_90 = []
x_s_91 = []
x_s_92 = []
x_s_93 = []
x_s_94 = []
x_s_95 = []
x_s_96 = []
x_s_97 = []
x_s_98 = []
x_s_99 = []
x_w_0 = []
x_w_1 = []
x_w_2 = []
x_w_3 = []
x_w_4 = []
x_w_5 = []
x_w_6 = []
x_w_7 = []
x_w_8 = []
x_w_9 = []
x_w_10 = []
x_w_11 = []
x_w_12 = []
x_w_13 = []
x_w_14 = []
x_w_15 = []
x_w_16 = []
x_w_17 = []
x_w_18 = []
x_w_19 = []
x_w_20 = []
x_w_21 = []
x_w_22 = []
x_w_23 = []
x_w_24 = []
x_w_25 = []
x_w_26 = []
x_w_27 = []
x_w_28 = []
x_w_29 = []
x_w_30 = []
x_w_31 = []
x_w_32 = []
x_w_33 = []
x_w_34 = []
x_w_35 = []
x_w_36 = []
x_w_37 = []
x_w_38 = []
x_w_39 = []
x_w_40 = []
x_w_41 = []
x_w_42 = []
x_w_43 = []
x_w_44 = []
x_w_45 = []
x_w_46 = []
x_w_47 = []
x_w_48 = []
x_w_49 = []
x_w_50 = []
x_w_51 = []
x_w_52 = []
x_w_53 = []
x_w_54 = []
x_w_55 = []
x_w_56 = []
x_w_57 = []
x_w_58 = []
x_w_59 = []
x_w_60 = []
x_w_61 = []
x_w_62 = []
x_w_63 = []
x_w_64 = []
x_w_65 = []
x_w_66 = []
x_w_67 = []
x_w_68 = []
x_w_69 = []
x_w_70 = []
x_w_71 = []
x_w_72 = []
x_w_73 = []
x_w_74 = []
x_w_75 = []
x_w_76 = []
x_w_77 = []
x_w_78 = []
x_w_79 = []
x_w_80 = []
x_w_81 = []
x_w_82 = []
x_w_83 = []
x_w_84 = []
x_w_85 = []
x_w_86 = []
x_w_87 = []
x_w_88 = []
x_w_89 = []
x_w_90 = []
x_w_91 = []
x_w_92 = []
x_w_93 = []
x_w_94 = []
x_w_95 = []
x_w_96 = []
x_w_97 = []
x_w_98 = []
x_w_99 = []
x_stu_0 = []
x_stu_1 = []
x_stu_2 = []
x_stu_3 = []
x_stu_4 = []
x_stu_5 = []
x_stu_6 = []
x_stu_7 = []
x_stu_8 = []
x_stu_9 = []
x_stu_10 = []
x_stu_11 = []
x_stu_12 = []
x_stu_13 = []
x_stu_14 = []
x_stu_15 = []
x_stu_16 = []
x_stu_17 = []
x_stu_18 = []
x_stu_19 = []
x_stu_20 = []
x_stu_21 = []
x_stu_22 = []
x_stu_23 = []
x_stu_24 = []
x_stu_25 = []
x_stu_26 = []
x_stu_27 = []
x_stu_28 = []
x_stu_29 = []
x_stu_30 = []
x_stu_31 = []
x_stu_32 = []
x_stu_33 = []
x_stu_34 = []
x_stu_35 = []
x_stu_36 = []
x_stu_37 = []
x_stu_38 = []
x_stu_39 = []
x_stu_40 = []
x_stu_41 = []
x_stu_42 = []
x_stu_43 = []
x_stu_44 = []
x_stu_45 = []
x_stu_46 = []
x_stu_47 = []
x_stu_48 = []
x_stu_49 = []
x_stu_50 = []
x_stu_51 = []
x_stu_52 = []
x_stu_53 = []
x_stu_54 = []
x_stu_55 = []
x_stu_56 = []
x_stu_57 = []
x_stu_58 = []
x_stu_59 = []
x_stu_60 = []
x_stu_61 = []
x_stu_62 = []
x_stu_63 = []
x_stu_64 = []
x_stu_65 = []
x_stu_66 = []
x_stu_67 = []
x_stu_68 = []
x_stu_69 = []
x_stu_70 = []
x_stu_71 = []
x_stu_72 = []
x_stu_73 = []
x_stu_74 = []
x_stu_75 = []
x_stu_76 = []
x_stu_77 = []
x_stu_78 = []
x_stu_79 = []
x_stu_80 = []
x_stu_81 = []
x_stu_82 = []
x_stu_83 = []
x_stu_84 = []
x_stu_85 = []
x_stu_86 = []
x_stu_87 = []
x_stu_88 = []
x_stu_89 = []
x_stu_90 = []
x_stu_91 = []
x_stu_92 = []
x_stu_93 = []
x_stu_94 = []
x_stu_95 = []
x_stu_96 = []
x_stu_97 = []
x_stu_98 = []
x_stu_99 = []
x_std_0 = []
x_std_1 = []
x_std_2 = []
x_std_3 = []
x_std_4 = []
x_std_5 = []
x_std_6 = []
x_std_7 = []
x_std_8 = []
x_std_9 = []
x_std_10 = []
x_std_11 = []
x_std_12 = []
x_std_13 = []
x_std_14 = []
x_std_15 = []
x_std_16 = []
x_std_17 = []
x_std_18 = []
x_std_19 = []
x_std_20 = []
x_std_21 = []
x_std_22 = []
x_std_23 = []
x_std_24 = []
x_std_25 = []
x_std_26 = []
x_std_27 = []
x_std_28 = []
x_std_29 = []
x_std_30 = []
x_std_31 = []
x_std_32 = []
x_std_33 = []
x_std_34 = []
x_std_35 = []
x_std_36 = []
x_std_37 = []
x_std_38 = []
x_std_39 = []
x_std_40 = []
x_std_41 = []
x_std_42 = []
x_std_43 = []
x_std_44 = []
x_std_45 = []
x_std_46 = []
x_std_47 = []
x_std_48 = []
x_std_49 = []
x_std_50 = []
x_std_51 = []
x_std_52 = []
x_std_53 = []
x_std_54 = []
x_std_55 = []
x_std_56 = []
x_std_57 = []
x_std_58 = []
x_std_59 = []
x_std_60 = []
x_std_61 = []
x_std_62 = []
x_std_63 = []
x_std_64 = []
x_std_65 = []
x_std_66 = []
x_std_67 = []
x_std_68 = []
x_std_69 = []
x_std_70 = []
x_std_71 = []
x_std_72 = []
x_std_73 = []
x_std_74 = []
x_std_75 = []
x_std_76 = []
x_std_77 = []
x_std_78 = []
x_std_79 = []
x_std_80 = []
x_std_81 = []
x_std_82 = []
x_std_83 = []
x_std_84 = []
x_std_85 = []
x_std_86 = []
x_std_87 = []
x_std_88 = []
x_std_89 = []
x_std_90 = []
x_std_91 = []
x_std_92 = []
x_std_93 = []
x_std_94 = []
x_std_95 = []
x_std_96 = []
x_std_97 = []
x_std_98 = []
x_std_99 = []
x_st_0 = []
x_st_1 = []
x_st_2 = []
x_st_3 = []
x_st_4 = []
x_st_5 = []
x_st_6 = []
x_st_7 = []
x_st_8 = []
x_st_9 = []
x_st_10 = []
x_st_11 = []
x_st_12 = []
x_st_13 = []
x_st_14 = []
x_st_15 = []
x_st_16 = []
x_st_17 = []
x_st_18 = []
x_st_19 = []
x_st_20 = []
x_st_21 = []
x_st_22 = []
x_st_23 = []
x_st_24 = []
x_st_25 = []
x_st_26 = []
x_st_27 = []
x_st_28 = []
x_st_29 = []
x_st_30 = []
x_st_31 = []
x_st_32 = []
x_st_33 = []
x_st_34 = []
x_st_35 = []
x_st_36 = []
x_st_37 = []
x_st_38 = []
x_st_39 = []
x_st_40 = []
x_st_41 = []
x_st_42 = []
x_st_43 = []
x_st_44 = []
x_st_45 = []
x_st_46 = []
x_st_47 = []
x_st_48 = []
x_st_49 = []
x_st_50 = []
x_st_51 = []
x_st_52 = []
x_st_53 = []
x_st_54 = []
x_st_55 = []
x_st_56 = []
x_st_57 = []
x_st_58 = []
x_st_59 = []
x_st_60 = []
x_st_61 = []
x_st_62 = []
x_st_63 = []
x_st_64 = []
x_st_65 = []
x_st_66 = []
x_st_67 = []
x_st_68 = []
x_st_69 = []
x_st_70 = []
x_st_71 = []
x_st_72 = []
x_st_73 = []
x_st_74 = []
x_st_75 = []
x_st_76 = []
x_st_77 = []
x_st_78 = []
x_st_79 = []
x_st_80 = []
x_st_81 = []
x_st_82 = []
x_st_83 = []
x_st_84 = []
x_st_85 = []
x_st_86 = []
x_st_87 = []
x_st_88 = []
x_st_89 = []
x_st_90 = []
x_st_91 = []
x_st_92 = []
x_st_93 = []
x_st_94 = []
x_st_95 = []
x_st_96 = []
x_st_97 = []
x_st_98 = []
x_st_99 = []
y_s_0 = []
y_s_1 = []
y_s_2 = []
y_s_3 = []
y_s_4 = []
y_s_5 = []
y_s_6 = []
y_s_7 = []
y_s_8 = []
y_s_9 = []
y_s_10 = []
y_s_11 = []
y_s_12 = []
y_s_13 = []
y_s_14 = []
y_s_15 = []
y_s_16 = []
y_s_17 = []
y_s_18 = []
y_s_19 = []
y_s_20 = []
y_s_21 = []
y_s_22 = []
y_s_23 = []
y_s_24 = []
y_s_25 = []
y_s_26 = []
y_s_27 = []
y_s_28 = []
y_s_29 = []
y_s_30 = []
y_s_31 = []
y_s_32 = []
y_s_33 = []
y_s_34 = []
y_s_35 = []
y_s_36 = []
y_s_37 = []
y_s_38 = []
y_s_39 = []
y_s_40 = []
y_s_41 = []
y_s_42 = []
y_s_43 = []
y_s_44 = []
y_s_45 = []
y_s_46 = []
y_s_47 = []
y_s_48 = []
y_s_49 = []
y_s_50 = []
y_s_51 = []
y_s_52 = []
y_s_53 = []
y_s_54 = []
y_s_55 = []
y_s_56 = []
y_s_57 = []
y_s_58 = []
y_s_59 = []
y_s_60 = []
y_s_61 = []
y_s_62 = []
y_s_63 = []
y_s_64 = []
y_s_65 = []
y_s_66 = []
y_s_67 = []
y_s_68 = []
y_s_69 = []
y_s_70 = []
y_s_71 = []
y_s_72 = []
y_s_73 = []
y_s_74 = []
y_s_75 = []
y_s_76 = []
y_s_77 = []
y_s_78 = []
y_s_79 = []
y_s_80 = []
y_s_81 = []
y_s_82 = []
y_s_83 = []
y_s_84 = []
y_s_85 = []
y_s_86 = []
y_s_87 = []
y_s_88 = []
y_s_89 = []
y_s_90 = []
y_s_91 = []
y_s_92 = []
y_s_93 = []
y_s_94 = []
y_s_95 = []
y_s_96 = []
y_s_97 = []
y_s_98 = []
y_s_99 = []
y_w_0 = []
y_w_1 = []
y_w_2 = []
y_w_3 = []
y_w_4 = []
y_w_5 = []
y_w_6 = []
y_w_7 = []
y_w_8 = []
y_w_9 = []
y_w_10 = []
y_w_11 = []
y_w_12 = []
y_w_13 = []
y_w_14 = []
y_w_15 = []
y_w_16 = []
y_w_17 = []
y_w_18 = []
y_w_19 = []
y_w_20 = []
y_w_21 = []
y_w_22 = []
y_w_23 = []
y_w_24 = []
y_w_25 = []
y_w_26 = []
y_w_27 = []
y_w_28 = []
y_w_29 = []
y_w_30 = []
y_w_31 = []
y_w_32 = []
y_w_33 = []
y_w_34 = []
y_w_35 = []
y_w_36 = []
y_w_37 = []
y_w_38 = []
y_w_39 = []
y_w_40 = []
y_w_41 = []
y_w_42 = []
y_w_43 = []
y_w_44 = []
y_w_45 = []
y_w_46 = []
y_w_47 = []
y_w_48 = []
y_w_49 = []
y_w_50 = []
y_w_51 = []
y_w_52 = []
y_w_53 = []
y_w_54 = []
y_w_55 = []
y_w_56 = []
y_w_57 = []
y_w_58 = []
y_w_59 = []
y_w_60 = []
y_w_61 = []
y_w_62 = []
y_w_63 = []
y_w_64 = []
y_w_65 = []
y_w_66 = []
y_w_67 = []
y_w_68 = []
y_w_69 = []
y_w_70 = []
y_w_71 = []
y_w_72 = []
y_w_73 = []
y_w_74 = []
y_w_75 = []
y_w_76 = []
y_w_77 = []
y_w_78 = []
y_w_79 = []
y_w_80 = []
y_w_81 = []
y_w_82 = []
y_w_83 = []
y_w_84 = []
y_w_85 = []
y_w_86 = []
y_w_87 = []
y_w_88 = []
y_w_89 = []
y_w_90 = []
y_w_91 = []
y_w_92 = []
y_w_93 = []
y_w_94 = []
y_w_95 = []
y_w_96 = []
y_w_97 = []
y_w_98 = []
y_w_99 = []
y_stu_0 = []
y_stu_1 = []
y_stu_2 = []
y_stu_3 = []
y_stu_4 = []
y_stu_5 = []
y_stu_6 = []
y_stu_7 = []
y_stu_8 = []
y_stu_9 = []
y_stu_10 = []
y_stu_11 = []
y_stu_12 = []
y_stu_13 = []
y_stu_14 = []
y_stu_15 = []
y_stu_16 = []
y_stu_17 = []
y_stu_18 = []
y_stu_19 = []
y_stu_20 = []
y_stu_21 = []
y_stu_22 = []
y_stu_23 = []
y_stu_24 = []
y_stu_25 = []
y_stu_26 = []
y_stu_27 = []
y_stu_28 = []
y_stu_29 = []
y_stu_30 = []
y_stu_31 = []
y_stu_32 = []
y_stu_33 = []
y_stu_34 = []
y_stu_35 = []
y_stu_36 = []
y_stu_37 = []
y_stu_38 = []
y_stu_39 = []
y_stu_40 = []
y_stu_41 = []
y_stu_42 = []
y_stu_43 = []
y_stu_44 = []
y_stu_45 = []
y_stu_46 = []
y_stu_47 = []
y_stu_48 = []
y_stu_49 = []
y_stu_50 = []
y_stu_51 = []
y_stu_52 = []
y_stu_53 = []
y_stu_54 = []
y_stu_55 = []
y_stu_56 = []
y_stu_57 = []
y_stu_58 = []
y_stu_59 = []
y_stu_60 = []
y_stu_61 = []
y_stu_62 = []
y_stu_63 = []
y_stu_64 = []
y_stu_65 = []
y_stu_66 = []
y_stu_67 = []
y_stu_68 = []
y_stu_69 = []
y_stu_70 = []
y_stu_71 = []
y_stu_72 = []
y_stu_73 = []
y_stu_74 = []
y_stu_75 = []
y_stu_76 = []
y_stu_77 = []
y_stu_78 = []
y_stu_79 = []
y_stu_80 = []
y_stu_81 = []
y_stu_82 = []
y_stu_83 = []
y_stu_84 = []
y_stu_85 = []
y_stu_86 = []
y_stu_87 = []
y_stu_88 = []
y_stu_89 = []
y_stu_90 = []
y_stu_91 = []
y_stu_92 = []
y_stu_93 = []
y_stu_94 = []
y_stu_95 = []
y_stu_96 = []
y_stu_97 = []
y_stu_98 = []
y_stu_99 = []
y_std_0 = []
y_std_1 = []
y_std_2 = []
y_std_3 = []
y_std_4 = []
y_std_5 = []
y_std_6 = []
y_std_7 = []
y_std_8 = []
y_std_9 = []
y_std_10 = []
y_std_11 = []
y_std_12 = []
y_std_13 = []
y_std_14 = []
y_std_15 = []
y_std_16 = []
y_std_17 = []
y_std_18 = []
y_std_19 = []
y_std_20 = []
y_std_21 = []
y_std_22 = []
y_std_23 = []
y_std_24 = []
y_std_25 = []
y_std_26 = []
y_std_27 = []
y_std_28 = []
y_std_29 = []
y_std_30 = []
y_std_31 = []
y_std_32 = []
y_std_33 = []
y_std_34 = []
y_std_35 = []
y_std_36 = []
y_std_37 = []
y_std_38 = []
y_std_39 = []
y_std_40 = []
y_std_41 = []
y_std_42 = []
y_std_43 = []
y_std_44 = []
y_std_45 = []
y_std_46 = []
y_std_47 = []
y_std_48 = []
y_std_49 = []
y_std_50 = []
y_std_51 = []
y_std_52 = []
y_std_53 = []
y_std_54 = []
y_std_55 = []
y_std_56 = []
y_std_57 = []
y_std_58 = []
y_std_59 = []
y_std_60 = []
y_std_61 = []
y_std_62 = []
y_std_63 = []
y_std_64 = []
y_std_65 = []
y_std_66 = []
y_std_67 = []
y_std_68 = []
y_std_69 = []
y_std_70 = []
y_std_71 = []
y_std_72 = []
y_std_73 = []
y_std_74 = []
y_std_75 = []
y_std_76 = []
y_std_77 = []
y_std_78 = []
y_std_79 = []
y_std_80 = []
y_std_81 = []
y_std_82 = []
y_std_83 = []
y_std_84 = []
y_std_85 = []
y_std_86 = []
y_std_87 = []
y_std_88 = []
y_std_89 = []
y_std_90 = []
y_std_91 = []
y_std_92 = []
y_std_93 = []
y_std_94 = []
y_std_95 = []
y_std_96 = []
y_std_97 = []
y_std_98 = []
y_std_99 = []
y_st_0 = []
y_st_1 = []
y_st_2 = []
y_st_3 = []
y_st_4 = []
y_st_5 = []
y_st_6 = []
y_st_7 = []
y_st_8 = []
y_st_9 = []
y_st_10 = []
y_st_11 = []
y_st_12 = []
y_st_13 = []
y_st_14 = []
y_st_15 = []
y_st_16 = []
y_st_17 = []
y_st_18 = []
y_st_19 = []
y_st_20 = []
y_st_21 = []
y_st_22 = []
y_st_23 = []
y_st_24 = []
y_st_25 = []
y_st_26 = []
y_st_27 = []
y_st_28 = []
y_st_29 = []
y_st_30 = []
y_st_31 = []
y_st_32 = []
y_st_33 = []
y_st_34 = []
y_st_35 = []
y_st_36 = []
y_st_37 = []
y_st_38 = []
y_st_39 = []
y_st_40 = []
y_st_41 = []
y_st_42 = []
y_st_43 = []
y_st_44 = []
y_st_45 = []
y_st_46 = []
y_st_47 = []
y_st_48 = []
y_st_49 = []
y_st_50 = []
y_st_51 = []
y_st_52 = []
y_st_53 = []
y_st_54 = []
y_st_55 = []
y_st_56 = []
y_st_57 = []
y_st_58 = []
y_st_59 = []
y_st_60 = []
y_st_61 = []
y_st_62 = []
y_st_63 = []
y_st_64 = []
y_st_65 = []
y_st_66 = []
y_st_67 = []
y_st_68 = []
y_st_69 = []
y_st_70 = []
y_st_71 = []
y_st_72 = []
y_st_73 = []
y_st_74 = []
y_st_75 = []
y_st_76 = []
y_st_77 = []
y_st_78 = []
y_st_79 = []
y_st_80 = []
y_st_81 = []
y_st_82 = []
y_st_83 = []
y_st_84 = []
y_st_85 = []
y_st_86 = []
y_st_87 = []
y_st_88 = []
y_st_89 = []
y_st_90 = []
y_st_91 = []
y_st_92 = []
y_st_93 = []
y_st_94 = []
y_st_95 = []
y_st_96 = []
y_st_97 = []
y_st_98 = []
y_st_99 = []
p_s_0 = []
p_s_1 = []
p_s_2 = []
p_s_3 = []
p_s_4 = []
p_s_5 = []
p_s_6 = []
p_s_7 = []
p_s_8 = []
p_s_9 = []
p_s_10 = []
p_s_11 = []
p_s_12 = []
p_s_13 = []
p_s_14 = []
p_s_15 = []
p_s_16 = []
p_s_17 = []
p_s_18 = []
p_s_19 = []
p_s_20 = []
p_s_21 = []
p_s_22 = []
p_s_23 = []
p_s_24 = []
p_s_25 = []
p_s_26 = []
p_s_27 = []
p_s_28 = []
p_s_29 = []
p_s_30 = []
p_s_31 = []
p_s_32 = []
p_s_33 = []
p_s_34 = []
p_s_35 = []
p_s_36 = []
p_s_37 = []
p_s_38 = []
p_s_39 = []
p_s_40 = []
p_s_41 = []
p_s_42 = []
p_s_43 = []
p_s_44 = []
p_s_45 = []
p_s_46 = []
p_s_47 = []
p_s_48 = []
p_s_49 = []
p_s_50 = []
p_s_51 = []
p_s_52 = []
p_s_53 = []
p_s_54 = []
p_s_55 = []
p_s_56 = []
p_s_57 = []
p_s_58 = []
p_s_59 = []
p_s_60 = []
p_s_61 = []
p_s_62 = []
p_s_63 = []
p_s_64 = []
p_s_65 = []
p_s_66 = []
p_s_67 = []
p_s_68 = []
p_s_69 = []
p_s_70 = []
p_s_71 = []
p_s_72 = []
p_s_73 = []
p_s_74 = []
p_s_75 = []
p_s_76 = []
p_s_77 = []
p_s_78 = []
p_s_79 = []
p_s_80 = []
p_s_81 = []
p_s_82 = []
p_s_83 = []
p_s_84 = []
p_s_85 = []
p_s_86 = []
p_s_87 = []
p_s_88 = []
p_s_89 = []
p_s_90 = []
p_s_91 = []
p_s_92 = []
p_s_93 = []
p_s_94 = []
p_s_95 = []
p_s_96 = []
p_s_97 = []
p_s_98 = []
p_s_99 = []
p_w_0 = []
p_w_1 = []
p_w_2 = []
p_w_3 = []
p_w_4 = []
p_w_5 = []
p_w_6 = []
p_w_7 = []
p_w_8 = []
p_w_9 = []
p_w_10 = []
p_w_11 = []
p_w_12 = []
p_w_13 = []
p_w_14 = []
p_w_15 = []
p_w_16 = []
p_w_17 = []
p_w_18 = []
p_w_19 = []
p_w_20 = []
p_w_21 = []
p_w_22 = []
p_w_23 = []
p_w_24 = []
p_w_25 = []
p_w_26 = []
p_w_27 = []
p_w_28 = []
p_w_29 = []
p_w_30 = []
p_w_31 = []
p_w_32 = []
p_w_33 = []
p_w_34 = []
p_w_35 = []
p_w_36 = []
p_w_37 = []
p_w_38 = []
p_w_39 = []
p_w_40 = []
p_w_41 = []
p_w_42 = []
p_w_43 = []
p_w_44 = []
p_w_45 = []
p_w_46 = []
p_w_47 = []
p_w_48 = []
p_w_49 = []
p_w_50 = []
p_w_51 = []
p_w_52 = []
p_w_53 = []
p_w_54 = []
p_w_55 = []
p_w_56 = []
p_w_57 = []
p_w_58 = []
p_w_59 = []
p_w_60 = []
p_w_61 = []
p_w_62 = []
p_w_63 = []
p_w_64 = []
p_w_65 = []
p_w_66 = []
p_w_67 = []
p_w_68 = []
p_w_69 = []
p_w_70 = []
p_w_71 = []
p_w_72 = []
p_w_73 = []
p_w_74 = []
p_w_75 = []
p_w_76 = []
p_w_77 = []
p_w_78 = []
p_w_79 = []
p_w_80 = []
p_w_81 = []
p_w_82 = []
p_w_83 = []
p_w_84 = []
p_w_85 = []
p_w_86 = []
p_w_87 = []
p_w_88 = []
p_w_89 = []
p_w_90 = []
p_w_91 = []
p_w_92 = []
p_w_93 = []
p_w_94 = []
p_w_95 = []
p_w_96 = []
p_w_97 = []
p_w_98 = []
p_w_99 = []
p_stu_0 = []
p_stu_1 = []
p_stu_2 = []
p_stu_3 = []
p_stu_4 = []
p_stu_5 = []
p_stu_6 = []
p_stu_7 = []
p_stu_8 = []
p_stu_9 = []
p_stu_10 = []
p_stu_11 = []
p_stu_12 = []
p_stu_13 = []
p_stu_14 = []
p_stu_15 = []
p_stu_16 = []
p_stu_17 = []
p_stu_18 = []
p_stu_19 = []
p_stu_20 = []
p_stu_21 = []
p_stu_22 = []
p_stu_23 = []
p_stu_24 = []
p_stu_25 = []
p_stu_26 = []
p_stu_27 = []
p_stu_28 = []
p_stu_29 = []
p_stu_30 = []
p_stu_31 = []
p_stu_32 = []
p_stu_33 = []
p_stu_34 = []
p_stu_35 = []
p_stu_36 = []
p_stu_37 = []
p_stu_38 = []
p_stu_39 = []
p_stu_40 = []
p_stu_41 = []
p_stu_42 = []
p_stu_43 = []
p_stu_44 = []
p_stu_45 = []
p_stu_46 = []
p_stu_47 = []
p_stu_48 = []
p_stu_49 = []
p_stu_50 = []
p_stu_51 = []
p_stu_52 = []
p_stu_53 = []
p_stu_54 = []
p_stu_55 = []
p_stu_56 = []
p_stu_57 = []
p_stu_58 = []
p_stu_59 = []
p_stu_60 = []
p_stu_61 = []
p_stu_62 = []
p_stu_63 = []
p_stu_64 = []
p_stu_65 = []
p_stu_66 = []
p_stu_67 = []
p_stu_68 = []
p_stu_69 = []
p_stu_70 = []
p_stu_71 = []
p_stu_72 = []
p_stu_73 = []
p_stu_74 = []
p_stu_75 = []
p_stu_76 = []
p_stu_77 = []
p_stu_78 = []
p_stu_79 = []
p_stu_80 = []
p_stu_81 = []
p_stu_82 = []
p_stu_83 = []
p_stu_84 = []
p_stu_85 = []
p_stu_86 = []
p_stu_87 = []
p_stu_88 = []
p_stu_89 = []
p_stu_90 = []
p_stu_91 = []
p_stu_92 = []
p_stu_93 = []
p_stu_94 = []
p_stu_95 = []
p_stu_96 = []
p_stu_97 = []
p_stu_98 = []
p_stu_99 = []
p_std_0 = []
p_std_1 = []
p_std_2 = []
p_std_3 = []
p_std_4 = []
p_std_5 = []
p_std_6 = []
p_std_7 = []
p_std_8 = []
p_std_9 = []
p_std_10 = []
p_std_11 = []
p_std_12 = []
p_std_13 = []
p_std_14 = []
p_std_15 = []
p_std_16 = []
p_std_17 = []
p_std_18 = []
p_std_19 = []
p_std_20 = []
p_std_21 = []
p_std_22 = []
p_std_23 = []
p_std_24 = []
p_std_25 = []
p_std_26 = []
p_std_27 = []
p_std_28 = []
p_std_29 = []
p_std_30 = []
p_std_31 = []
p_std_32 = []
p_std_33 = []
p_std_34 = []
p_std_35 = []
p_std_36 = []
p_std_37 = []
p_std_38 = []
p_std_39 = []
p_std_40 = []
p_std_41 = []
p_std_42 = []
p_std_43 = []
p_std_44 = []
p_std_45 = []
p_std_46 = []
p_std_47 = []
p_std_48 = []
p_std_49 = []
p_std_50 = []
p_std_51 = []
p_std_52 = []
p_std_53 = []
p_std_54 = []
p_std_55 = []
p_std_56 = []
p_std_57 = []
p_std_58 = []
p_std_59 = []
p_std_60 = []
p_std_61 = []
p_std_62 = []
p_std_63 = []
p_std_64 = []
p_std_65 = []
p_std_66 = []
p_std_67 = []
p_std_68 = []
p_std_69 = []
p_std_70 = []
p_std_71 = []
p_std_72 = []
p_std_73 = []
p_std_74 = []
p_std_75 = []
p_std_76 = []
p_std_77 = []
p_std_78 = []
p_std_79 = []
p_std_80 = []
p_std_81 = []
p_std_82 = []
p_std_83 = []
p_std_84 = []
p_std_85 = []
p_std_86 = []
p_std_87 = []
p_std_88 = []
p_std_89 = []
p_std_90 = []
p_std_91 = []
p_std_92 = []
p_std_93 = []
p_std_94 = []
p_std_95 = []
p_std_96 = []
p_std_97 = []
p_std_98 = []
p_std_99 = []
p_st_0 = []
p_st_1 = []
p_st_2 = []
p_st_3 = []
p_st_4 = []
p_st_5 = []
p_st_6 = []
p_st_7 = []
p_st_8 = []
p_st_9 = []
p_st_10 = []
p_st_11 = []
p_st_12 = []
p_st_13 = []
p_st_14 = []
p_st_15 = []
p_st_16 = []
p_st_17 = []
p_st_18 = []
p_st_19 = []
p_st_20 = []
p_st_21 = []
p_st_22 = []
p_st_23 = []
p_st_24 = []
p_st_25 = []
p_st_26 = []
p_st_27 = []
p_st_28 = []
p_st_29 = []
p_st_30 = []
p_st_31 = []
p_st_32 = []
p_st_33 = []
p_st_34 = []
p_st_35 = []
p_st_36 = []
p_st_37 = []
p_st_38 = []
p_st_39 = []
p_st_40 = []
p_st_41 = []
p_st_42 = []
p_st_43 = []
p_st_44 = []
p_st_45 = []
p_st_46 = []
p_st_47 = []
p_st_48 = []
p_st_49 = []
p_st_50 = []
p_st_51 = []
p_st_52 = []
p_st_53 = []
p_st_54 = []
p_st_55 = []
p_st_56 = []
p_st_57 = []
p_st_58 = []
p_st_59 = []
p_st_60 = []
p_st_61 = []
p_st_62 = []
p_st_63 = []
p_st_64 = []
p_st_65 = []
p_st_66 = []
p_st_67 = []
p_st_68 = []
p_st_69 = []
p_st_70 = []
p_st_71 = []
p_st_72 = []
p_st_73 = []
p_st_74 = []
p_st_75 = []
p_st_76 = []
p_st_77 = []
p_st_78 = []
p_st_79 = []
p_st_80 = []
p_st_81 = []
p_st_82 = []
p_st_83 = []
p_st_84 = []
p_st_85 = []
p_st_86 = []
p_st_87 = []
p_st_88 = []
p_st_89 = []
p_st_90 = []
p_st_91 = []
p_st_92 = []
p_st_93 = []
p_st_94 = []
p_st_95 = []
p_st_96 = []
p_st_97 = []
p_st_98 = []
p_st_99 = []

csvfile= open(path,'r')
readf =  csv.reader(csvfile)
for row in readf:
        if row:
                if row[3]== ' jog':
                        if row[4] == '00':
                                x1=row[0]
                                x_j_0.append(x1)
                                x2=row[1]
                                x_j_0.append(x2)
                                x3= row[2]
                                x_j_0.append(x3)
                                y_j_0.append(mvts.index(row[3]))
                                p_j_0.append(id.index(row[4]))
                        if row[4] == '01':
                                x1=row[0]
                                x_j_1.append(x1)
                                x2=row[1]
                                x_j_1.append(x2)
                                x3= row[2]
                                x_j_1.append(x3)
                                y_j_1.append(mvts.index(row[3]))
                                p_j_1.append(id.index(row[4]))
                        if row[4] == '02':
                                x1=row[0]
                                x_j_2.append(x1)
                                x2=row[1]
                                x_j_2.append(x2)
                                x3= row[2]
                                x_j_2.append(x3)
                                y_j_2.append(mvts.index(row[3]))
                                p_j_2.append(id.index(row[4]))
                        if row[4] == '03':
                                x1=row[0]
                                x_j_3.append(x1)
                                x2=row[1]
                                x_j_3.append(x2)
                                x3= row[2]
                                x_j_3.append(x3)
                                y_j_3.append(mvts.index(row[3]))
                                p_j_3.append(id.index(row[4]))
                        if row[4] == '04':
                                x1=row[0]
                                x_j_4.append(x1)
                                x2=row[1]
                                x_j_4.append(x2)
                                x3= row[2]
                                x_j_4.append(x3)
                                y_j_4.append(mvts.index(row[3]))
                                p_j_4.append(id.index(row[4]))
                        if row[4] == '05':
                                x1=row[0]
                                x_j_5.append(x1)
                                x2=row[1]
                                x_j_5.append(x2)
                                x3= row[2]
                                x_j_5.append(x3)
                                y_j_5.append(mvts.index(row[3]))
                                p_j_5.append(id.index(row[4]))
                        if row[4] == '06':
                                x1=row[0]
                                x_j_6.append(x1)
                                x2=row[1]
                                x_j_6.append(x2)
                                x3= row[2]
                                x_j_6.append(x3)
                                y_j_6.append(mvts.index(row[3]))
                                p_j_6.append(id.index(row[4]))
                        if row[4] == '07':
                                x1=row[0]
                                x_j_7.append(x1)
                                x2=row[1]
                                x_j_7.append(x2)
                                x3= row[2]
                                x_j_7.append(x3)
                                y_j_7.append(mvts.index(row[3]))
                                p_j_7.append(id.index(row[4]))
                        if row[4] == '08':
                                x1=row[0]
                                x_j_8.append(x1)
                                x2=row[1]
                                x_j_8.append(x2)
                                x3= row[2]
                                x_j_8.append(x3)
                                y_j_8.append(mvts.index(row[3]))
                                p_j_8.append(id.index(row[4]))
                        if row[4] == '09':
                                x1=row[0]
                                x_j_9.append(x1)
                                x2=row[1]
                                x_j_9.append(x2)
                                x3= row[2]
                                x_j_9.append(x3)
                                y_j_9.append(mvts.index(row[3]))
                                p_j_9.append(id.index(row[4]))
                        if row[4] == '10':
                                x1=row[0]
                                x_j_10.append(x1)
                                x2=row[1]
                                x_j_10.append(x2)
                                x3= row[2]
                                x_j_10.append(x3)
                                y_j_10.append(mvts.index(row[3]))
                                p_j_10.append(id.index(row[4]))
                        if row[4] == '11':
                                x1=row[0]
                                x_j_11.append(x1)
                                x2=row[1]
                                x_j_11.append(x2)
                                x3= row[2]
                                x_j_11.append(x3)
                                y_j_11.append(mvts.index(row[3]))
                                p_j_11.append(id.index(row[4]))
                        if row[4] == '12':
                                x1=row[0]
                                x_j_12.append(x1)
                                x2=row[1]
                                x_j_12.append(x2)
                                x3= row[2]
                                x_j_12.append(x3)
                                y_j_12.append(mvts.index(row[3]))
                                p_j_12.append(id.index(row[4]))
                        if row[4] == '13':
                                x1=row[0]
                                x_j_13.append(x1)
                                x2=row[1]
                                x_j_13.append(x2)
                                x3= row[2]
                                x_j_13.append(x3)
                                y_j_13.append(mvts.index(row[3]))
                                p_j_13.append(id.index(row[4]))
                        if row[4] == '14':
                                x1=row[0]
                                x_j_14.append(x1)
                                x2=row[1]
                                x_j_14.append(x2)
                                x3= row[2]
                                x_j_14.append(x3)
                                y_j_14.append(mvts.index(row[3]))
                                p_j_14.append(id.index(row[4]))
                        if row[4] == '15':
                                x1=row[0]
                                x_j_15.append(x1)
                                x2=row[1]
                                x_j_15.append(x2)
                                x3= row[2]
                                x_j_15.append(x3)
                                y_j_15.append(mvts.index(row[3]))
                                p_j_15.append(id.index(row[4]))
                        if row[4] == '16':
                                x1=row[0]
                                x_j_16.append(x1)
                                x2=row[1]
                                x_j_16.append(x2)
                                x3= row[2]
                                x_j_16.append(x3)
                                y_j_16.append(mvts.index(row[3]))
                                p_j_16.append(id.index(row[4]))
                        if row[4] == '17':
                                x1=row[0]
                                x_j_17.append(x1)
                                x2=row[1]
                                x_j_17.append(x2)
                                x3= row[2]
                                x_j_17.append(x3)
                                y_j_17.append(mvts.index(row[3]))
                                p_j_17.append(id.index(row[4]))
                        if row[4] == '18':
                                x1=row[0]
                                x_j_18.append(x1)
                                x2=row[1]
                                x_j_18.append(x2)
                                x3= row[2]
                                x_j_18.append(x3)
                                y_j_18.append(mvts.index(row[3]))
                                p_j_18.append(id.index(row[4]))
                        if row[4] == '19':
                                x1=row[0]
                                x_j_19.append(x1)
                                x2=row[1]
                                x_j_19.append(x2)
                                x3= row[2]
                                x_j_19.append(x3)
                                y_j_19.append(mvts.index(row[3]))
                                p_j_19.append(id.index(row[4]))
                        if row[4] == '20':
                                x1=row[0]
                                x_j_20.append(x1)
                                x2=row[1]
                                x_j_20.append(x2)
                                x3= row[2]
                                x_j_20.append(x3)
                                y_j_20.append(mvts.index(row[3]))
                                p_j_20.append(id.index(row[4]))
                        if row[4] == '21':
                                x1=row[0]
                                x_j_21.append(x1)
                                x2=row[1]
                                x_j_21.append(x2)
                                x3= row[2]
                                x_j_21.append(x3)
                                y_j_21.append(mvts.index(row[3]))
                                p_j_21.append(id.index(row[4]))
                        if row[4] == '22':
                                x1=row[0]
                                x_j_22.append(x1)
                                x2=row[1]
                                x_j_22.append(x2)
                                x3= row[2]
                                x_j_22.append(x3)
                                y_j_22.append(mvts.index(row[3]))
                                p_j_22.append(id.index(row[4]))
                        if row[4] == '23':
                                x1=row[0]
                                x_j_23.append(x1)
                                x2=row[1]
                                x_j_23.append(x2)
                                x3= row[2]
                                x_j_23.append(x3)
                                y_j_23.append(mvts.index(row[3]))
                                p_j_23.append(id.index(row[4]))
                        if row[4] == '24':
                                x1=row[0]
                                x_j_24.append(x1)
                                x2=row[1]
                                x_j_24.append(x2)
                                x3= row[2]
                                x_j_24.append(x3)
                                y_j_24.append(mvts.index(row[3]))
                                p_j_24.append(id.index(row[4]))
                        if row[4] == '25':
                                x1=row[0]
                                x_j_25.append(x1)
                                x2=row[1]
                                x_j_25.append(x2)
                                x3= row[2]
                                x_j_25.append(x3)
                                y_j_25.append(mvts.index(row[3]))
                                p_j_25.append(id.index(row[4]))
                        if row[4] == '26':
                                x1=row[0]
                                x_j_26.append(x1)
                                x2=row[1]
                                x_j_26.append(x2)
                                x3= row[2]
                                x_j_26.append(x3)
                                y_j_26.append(mvts.index(row[3]))
                                p_j_26.append(id.index(row[4]))
                        if row[4] == '27':
                                x1=row[0]
                                x_j_27.append(x1)
                                x2=row[1]
                                x_j_27.append(x2)
                                x3= row[2]
                                x_j_27.append(x3)
                                y_j_27.append(mvts.index(row[3]))
                                p_j_27.append(id.index(row[4]))
                        if row[4] == '28':
                                x1=row[0]
                                x_j_28.append(x1)
                                x2=row[1]
                                x_j_28.append(x2)
                                x3= row[2]
                                x_j_28.append(x3)
                                y_j_28.append(mvts.index(row[3]))
                                p_j_28.append(id.index(row[4]))
                        if row[4] == '29':
                                x1=row[0]
                                x_j_29.append(x1)
                                x2=row[1]
                                x_j_29.append(x2)
                                x3= row[2]
                                x_j_29.append(x3)
                                y_j_29.append(mvts.index(row[3]))
                                p_j_29.append(id.index(row[4]))
                        if row[4] == '30':
                                x1=row[0]
                                x_j_30.append(x1)
                                x2=row[1]
                                x_j_30.append(x2)
                                x3= row[2]
                                x_j_30.append(x3)
                                y_j_30.append(mvts.index(row[3]))
                                p_j_30.append(id.index(row[4]))
                        if row[4] == '31':
                                x1=row[0]
                                x_j_31.append(x1)
                                x2=row[1]
                                x_j_31.append(x2)
                                x3= row[2]
                                x_j_31.append(x3)
                                y_j_31.append(mvts.index(row[3]))
                                p_j_31.append(id.index(row[4]))
                        if row[4] == '32':
                                x1=row[0]
                                x_j_32.append(x1)
                                x2=row[1]
                                x_j_32.append(x2)
                                x3= row[2]
                                x_j_32.append(x3)
                                y_j_32.append(mvts.index(row[3]))
                                p_j_32.append(id.index(row[4]))
                        if row[4] == '33':
                                x1=row[0]
                                x_j_33.append(x1)
                                x2=row[1]
                                x_j_33.append(x2)
                                x3= row[2]
                                x_j_33.append(x3)
                                y_j_33.append(mvts.index(row[3]))
                                p_j_33.append(id.index(row[4]))
                        if row[4] == '34':
                                x1=row[0]
                                x_j_34.append(x1)
                                x2=row[1]
                                x_j_34.append(x2)
                                x3= row[2]
                                x_j_34.append(x3)
                                y_j_34.append(mvts.index(row[3]))
                                p_j_34.append(id.index(row[4]))
                        if row[4] == '35':
                                x1=row[0]
                                x_j_35.append(x1)
                                x2=row[1]
                                x_j_35.append(x2)
                                x3= row[2]
                                x_j_35.append(x3)
                                y_j_35.append(mvts.index(row[3]))
                                p_j_35.append(id.index(row[4]))
                        if row[4] == '36':
                                x1=row[0]
                                x_j_36.append(x1)
                                x2=row[1]
                                x_j_36.append(x2)
                                x3= row[2]
                                x_j_36.append(x3)
                                y_j_36.append(mvts.index(row[3]))
                                p_j_36.append(id.index(row[4]))
                        if row[4] == '37':
                                x1=row[0]
                                x_j_37.append(x1)
                                x2=row[1]
                                x_j_37.append(x2)
                                x3= row[2]
                                x_j_37.append(x3)
                                y_j_37.append(mvts.index(row[3]))
                                p_j_37.append(id.index(row[4]))
                        if row[4] == '38':
                                x1=row[0]
                                x_j_38.append(x1)
                                x2=row[1]
                                x_j_38.append(x2)
                                x3= row[2]
                                x_j_38.append(x3)
                                y_j_38.append(mvts.index(row[3]))
                                p_j_38.append(id.index(row[4]))
                        if row[4] == '39':
                                x1=row[0]
                                x_j_39.append(x1)
                                x2=row[1]
                                x_j_39.append(x2)
                                x3= row[2]
                                x_j_39.append(x3)
                                y_j_39.append(mvts.index(row[3]))
                                p_j_39.append(id.index(row[4]))
                        if row[4] == '40':
                                x1=row[0]
                                x_j_40.append(x1)
                                x2=row[1]
                                x_j_40.append(x2)
                                x3= row[2]
                                x_j_40.append(x3)
                                y_j_40.append(mvts.index(row[3]))
                                p_j_40.append(id.index(row[4]))
                        if row[4] == '41':
                                x1=row[0]
                                x_j_41.append(x1)
                                x2=row[1]
                                x_j_41.append(x2)
                                x3= row[2]
                                x_j_41.append(x3)
                                y_j_41.append(mvts.index(row[3]))
                                p_j_41.append(id.index(row[4]))
                        if row[4] == '42':
                                x1=row[0]
                                x_j_42.append(x1)
                                x2=row[1]
                                x_j_42.append(x2)
                                x3= row[2]
                                x_j_42.append(x3)
                                y_j_42.append(mvts.index(row[3]))
                                p_j_42.append(id.index(row[4]))
                        if row[4] == '43':
                                x1=row[0]
                                x_j_43.append(x1)
                                x2=row[1]
                                x_j_43.append(x2)
                                x3= row[2]
                                x_j_43.append(x3)
                                y_j_43.append(mvts.index(row[3]))
                                p_j_43.append(id.index(row[4]))
                        if row[4] == '44':
                                x1=row[0]
                                x_j_44.append(x1)
                                x2=row[1]
                                x_j_44.append(x2)
                                x3= row[2]
                                x_j_44.append(x3)
                                y_j_44.append(mvts.index(row[3]))
                                p_j_44.append(id.index(row[4]))
                        if row[4] == '45':
                                x1=row[0]
                                x_j_45.append(x1)
                                x2=row[1]
                                x_j_45.append(x2)
                                x3= row[2]
                                x_j_45.append(x3)
                                y_j_45.append(mvts.index(row[3]))
                                p_j_45.append(id.index(row[4]))
                        if row[4] == '46':
                                x1=row[0]
                                x_j_46.append(x1)
                                x2=row[1]
                                x_j_46.append(x2)
                                x3= row[2]
                                x_j_46.append(x3)
                                y_j_46.append(mvts.index(row[3]))
                                p_j_46.append(id.index(row[4]))
                        if row[4] == '47':
                                x1=row[0]
                                x_j_47.append(x1)
                                x2=row[1]
                                x_j_47.append(x2)
                                x3= row[2]
                                x_j_47.append(x3)
                                y_j_47.append(mvts.index(row[3]))
                                p_j_47.append(id.index(row[4]))
                        if row[4] == '48':
                                x1=row[0]
                                x_j_48.append(x1)
                                x2=row[1]
                                x_j_48.append(x2)
                                x3= row[2]
                                x_j_48.append(x3)
                                y_j_48.append(mvts.index(row[3]))
                                p_j_48.append(id.index(row[4]))
                        if row[4] == '49':
                                x1=row[0]
                                x_j_49.append(x1)
                                x2=row[1]
                                x_j_49.append(x2)
                                x3= row[2]
                                x_j_49.append(x3)
                                y_j_49.append(mvts.index(row[3]))
                                p_j_49.append(id.index(row[4]))
                        if row[4] == '50':
                                x1=row[0]
                                x_j_50.append(x1)
                                x2=row[1]
                                x_j_50.append(x2)
                                x3= row[2]
                                x_j_50.append(x3)
                                y_j_50.append(mvts.index(row[3]))
                                p_j_50.append(id.index(row[4]))
                        if row[4] == '51':
                                x1=row[0]
                                x_j_51.append(x1)
                                x2=row[1]
                                x_j_51.append(x2)
                                x3= row[2]
                                x_j_51.append(x3)
                                y_j_51.append(mvts.index(row[3]))
                                p_j_51.append(id.index(row[4]))
                        if row[4] == '52':
                                x1=row[0]
                                x_j_52.append(x1)
                                x2=row[1]
                                x_j_52.append(x2)
                                x3= row[2]
                                x_j_52.append(x3)
                                y_j_52.append(mvts.index(row[3]))
                                p_j_52.append(id.index(row[4]))
                        if row[4] == '53':
                                x1=row[0]
                                x_j_53.append(x1)
                                x2=row[1]
                                x_j_53.append(x2)
                                x3= row[2]
                                x_j_53.append(x3)
                                y_j_53.append(mvts.index(row[3]))
                                p_j_53.append(id.index(row[4]))
                        if row[4] == '54':
                                x1=row[0]
                                x_j_54.append(x1)
                                x2=row[1]
                                x_j_54.append(x2)
                                x3= row[2]
                                x_j_54.append(x3)
                                y_j_54.append(mvts.index(row[3]))
                                p_j_54.append(id.index(row[4]))
                        if row[4] == '55':
                                x1=row[0]
                                x_j_55.append(x1)
                                x2=row[1]
                                x_j_55.append(x2)
                                x3= row[2]
                                x_j_55.append(x3)
                                y_j_55.append(mvts.index(row[3]))
                                p_j_55.append(id.index(row[4]))
                        if row[4] == '56':
                                x1=row[0]
                                x_j_56.append(x1)
                                x2=row[1]
                                x_j_56.append(x2)
                                x3= row[2]
                                x_j_56.append(x3)
                                y_j_56.append(mvts.index(row[3]))
                                p_j_56.append(id.index(row[4]))
                        if row[4] == '57':
                                x1=row[0]
                                x_j_57.append(x1)
                                x2=row[1]
                                x_j_57.append(x2)
                                x3= row[2]
                                x_j_57.append(x3)
                                y_j_57.append(mvts.index(row[3]))
                                p_j_57.append(id.index(row[4]))
                        if row[4] == '58':
                                x1=row[0]
                                x_j_58.append(x1)
                                x2=row[1]
                                x_j_58.append(x2)
                                x3= row[2]
                                x_j_58.append(x3)
                                y_j_58.append(mvts.index(row[3]))
                                p_j_58.append(id.index(row[4]))
                        if row[4] == '59':
                                x1=row[0]
                                x_j_59.append(x1)
                                x2=row[1]
                                x_j_59.append(x2)
                                x3= row[2]
                                x_j_59.append(x3)
                                y_j_59.append(mvts.index(row[3]))
                                p_j_59.append(id.index(row[4]))
                        if row[4] == '60':
                                x1=row[0]
                                x_j_60.append(x1)
                                x2=row[1]
                                x_j_60.append(x2)
                                x3= row[2]
                                x_j_60.append(x3)
                                y_j_60.append(mvts.index(row[3]))
                                p_j_60.append(id.index(row[4]))
                        if row[4] == '61':
                                x1=row[0]
                                x_j_61.append(x1)
                                x2=row[1]
                                x_j_61.append(x2)
                                x3= row[2]
                                x_j_61.append(x3)
                                y_j_61.append(mvts.index(row[3]))
                                p_j_61.append(id.index(row[4]))
                        if row[4] == '62':
                                x1=row[0]
                                x_j_62.append(x1)
                                x2=row[1]
                                x_j_62.append(x2)
                                x3= row[2]
                                x_j_62.append(x3)
                                y_j_62.append(mvts.index(row[3]))
                                p_j_62.append(id.index(row[4]))
                        if row[4] == '63':
                                x1=row[0]
                                x_j_63.append(x1)
                                x2=row[1]
                                x_j_63.append(x2)
                                x3= row[2]
                                x_j_63.append(x3)
                                y_j_63.append(mvts.index(row[3]))
                                p_j_63.append(id.index(row[4]))
                        if row[4] == '64':
                                x1=row[0]
                                x_j_64.append(x1)
                                x2=row[1]
                                x_j_64.append(x2)
                                x3= row[2]
                                x_j_64.append(x3)
                                y_j_64.append(mvts.index(row[3]))
                                p_j_64.append(id.index(row[4]))
                        if row[4] == '65':
                                x1=row[0]
                                x_j_65.append(x1)
                                x2=row[1]
                                x_j_65.append(x2)
                                x3= row[2]
                                x_j_65.append(x3)
                                y_j_65.append(mvts.index(row[3]))
                                p_j_65.append(id.index(row[4]))
                        if row[4] == '66':
                                x1=row[0]
                                x_j_66.append(x1)
                                x2=row[1]
                                x_j_66.append(x2)
                                x3= row[2]
                                x_j_66.append(x3)
                                y_j_66.append(mvts.index(row[3]))
                                p_j_66.append(id.index(row[4]))
                        if row[4] == '67':
                                x1=row[0]
                                x_j_67.append(x1)
                                x2=row[1]
                                x_j_67.append(x2)
                                x3= row[2]
                                x_j_67.append(x3)
                                y_j_67.append(mvts.index(row[3]))
                                p_j_67.append(id.index(row[4]))
                        if row[4] == '68':
                                x1=row[0]
                                x_j_68.append(x1)
                                x2=row[1]
                                x_j_68.append(x2)
                                x3= row[2]
                                x_j_68.append(x3)
                                y_j_68.append(mvts.index(row[3]))
                                p_j_68.append(id.index(row[4]))
                        if row[4] == '69':
                                x1=row[0]
                                x_j_69.append(x1)
                                x2=row[1]
                                x_j_69.append(x2)
                                x3= row[2]
                                x_j_69.append(x3)
                                y_j_69.append(mvts.index(row[3]))
                                p_j_69.append(id.index(row[4]))
                        if row[4] == '70':
                                x1=row[0]
                                x_j_70.append(x1)
                                x2=row[1]
                                x_j_70.append(x2)
                                x3= row[2]
                                x_j_70.append(x3)
                                y_j_70.append(mvts.index(row[3]))
                                p_j_70.append(id.index(row[4]))
                        if row[4] == '71':
                                x1=row[0]
                                x_j_71.append(x1)
                                x2=row[1]
                                x_j_71.append(x2)
                                x3= row[2]
                                x_j_71.append(x3)
                                y_j_71.append(mvts.index(row[3]))
                                p_j_71.append(id.index(row[4]))
                        if row[4] == '72':
                                x1=row[0]
                                x_j_72.append(x1)
                                x2=row[1]
                                x_j_72.append(x2)
                                x3= row[2]
                                x_j_72.append(x3)
                                y_j_72.append(mvts.index(row[3]))
                                p_j_72.append(id.index(row[4]))
                        if row[4] == '73':
                                x1=row[0]
                                x_j_73.append(x1)
                                x2=row[1]
                                x_j_73.append(x2)
                                x3= row[2]
                                x_j_73.append(x3)
                                y_j_73.append(mvts.index(row[3]))
                                p_j_73.append(id.index(row[4]))
                        if row[4] == '74':
                                x1=row[0]
                                x_j_74.append(x1)
                                x2=row[1]
                                x_j_74.append(x2)
                                x3= row[2]
                                x_j_74.append(x3)
                                y_j_74.append(mvts.index(row[3]))
                                p_j_74.append(id.index(row[4]))
                        if row[4] == '75':
                                x1=row[0]
                                x_j_75.append(x1)
                                x2=row[1]
                                x_j_75.append(x2)
                                x3= row[2]
                                x_j_75.append(x3)
                                y_j_75.append(mvts.index(row[3]))
                                p_j_75.append(id.index(row[4]))
                        if row[4] == '76':
                                x1=row[0]
                                x_j_76.append(x1)
                                x2=row[1]
                                x_j_76.append(x2)
                                x3= row[2]
                                x_j_76.append(x3)
                                y_j_76.append(mvts.index(row[3]))
                                p_j_76.append(id.index(row[4]))
                        if row[4] == '77':
                                x1=row[0]
                                x_j_77.append(x1)
                                x2=row[1]
                                x_j_77.append(x2)
                                x3= row[2]
                                x_j_77.append(x3)
                                y_j_77.append(mvts.index(row[3]))
                                p_j_77.append(id.index(row[4]))
                        if row[4] == '78':
                                x1=row[0]
                                x_j_78.append(x1)
                                x2=row[1]
                                x_j_78.append(x2)
                                x3= row[2]
                                x_j_78.append(x3)
                                y_j_78.append(mvts.index(row[3]))
                                p_j_78.append(id.index(row[4]))
                        if row[4] == '79':
                                x1=row[0]
                                x_j_79.append(x1)
                                x2=row[1]
                                x_j_79.append(x2)
                                x3= row[2]
                                x_j_79.append(x3)
                                y_j_79.append(mvts.index(row[3]))
                                p_j_79.append(id.index(row[4]))
                        if row[4] == '80':
                                x1=row[0]
                                x_j_80.append(x1)
                                x2=row[1]
                                x_j_80.append(x2)
                                x3= row[2]
                                x_j_80.append(x3)
                                y_j_80.append(mvts.index(row[3]))
                                p_j_80.append(id.index(row[4]))
                        if row[4] == '81':
                                x1=row[0]
                                x_j_81.append(x1)
                                x2=row[1]
                                x_j_81.append(x2)
                                x3= row[2]
                                x_j_81.append(x3)
                                y_j_81.append(mvts.index(row[3]))
                                p_j_81.append(id.index(row[4]))
                        if row[4] == '82':
                                x1=row[0]
                                x_j_82.append(x1)
                                x2=row[1]
                                x_j_82.append(x2)
                                x3= row[2]
                                x_j_82.append(x3)
                                y_j_82.append(mvts.index(row[3]))
                                p_j_82.append(id.index(row[4]))
                        if row[4] == '83':
                                x1=row[0]
                                x_j_83.append(x1)
                                x2=row[1]
                                x_j_83.append(x2)
                                x3= row[2]
                                x_j_83.append(x3)
                                y_j_83.append(mvts.index(row[3]))
                                p_j_83.append(id.index(row[4]))
                        if row[4] == '84':
                                x1=row[0]
                                x_j_84.append(x1)
                                x2=row[1]
                                x_j_84.append(x2)
                                x3= row[2]
                                x_j_84.append(x3)
                                y_j_84.append(mvts.index(row[3]))
                                p_j_84.append(id.index(row[4]))
                        if row[4] == '85':
                                x1=row[0]
                                x_j_85.append(x1)
                                x2=row[1]
                                x_j_85.append(x2)
                                x3= row[2]
                                x_j_85.append(x3)
                                y_j_85.append(mvts.index(row[3]))
                                p_j_85.append(id.index(row[4]))
                        if row[4] == '86':
                                x1=row[0]
                                x_j_86.append(x1)
                                x2=row[1]
                                x_j_86.append(x2)
                                x3= row[2]
                                x_j_86.append(x3)
                                y_j_86.append(mvts.index(row[3]))
                                p_j_86.append(id.index(row[4]))
                        if row[4] == '87':
                                x1=row[0]
                                x_j_87.append(x1)
                                x2=row[1]
                                x_j_87.append(x2)
                                x3= row[2]
                                x_j_87.append(x3)
                                y_j_87.append(mvts.index(row[3]))
                                p_j_87.append(id.index(row[4]))
                        if row[4] == '88':
                                x1=row[0]
                                x_j_88.append(x1)
                                x2=row[1]
                                x_j_88.append(x2)
                                x3= row[2]
                                x_j_88.append(x3)
                                y_j_88.append(mvts.index(row[3]))
                                p_j_88.append(id.index(row[4]))
                        if row[4] == '89':
                                x1=row[0]
                                x_j_89.append(x1)
                                x2=row[1]
                                x_j_89.append(x2)
                                x3= row[2]
                                x_j_89.append(x3)
                                y_j_89.append(mvts.index(row[3]))
                                p_j_89.append(id.index(row[4]))
                        if row[4] == '90':
                                x1=row[0]
                                x_j_90.append(x1)
                                x2=row[1]
                                x_j_90.append(x2)
                                x3= row[2]
                                x_j_90.append(x3)
                                y_j_90.append(mvts.index(row[3]))
                                p_j_90.append(id.index(row[4]))
                        if row[4] == '91':
                                x1=row[0]
                                x_j_91.append(x1)
                                x2=row[1]
                                x_j_91.append(x2)
                                x3= row[2]
                                x_j_91.append(x3)
                                y_j_91.append(mvts.index(row[3]))
                                p_j_91.append(id.index(row[4]))
                        if row[4] == '92':
                                x1=row[0]
                                x_j_92.append(x1)
                                x2=row[1]
                                x_j_92.append(x2)
                                x3= row[2]
                                x_j_92.append(x3)
                                y_j_92.append(mvts.index(row[3]))
                                p_j_92.append(id.index(row[4]))
                        if row[4] == '93':
                                x1=row[0]
                                x_j_93.append(x1)
                                x2=row[1]
                                x_j_93.append(x2)
                                x3= row[2]
                                x_j_93.append(x3)
                                y_j_93.append(mvts.index(row[3]))
                                p_j_93.append(id.index(row[4]))
                        if row[4] == '94':
                                x1=row[0]
                                x_j_94.append(x1)
                                x2=row[1]
                                x_j_94.append(x2)
                                x3= row[2]
                                x_j_94.append(x3)
                                y_j_94.append(mvts.index(row[3]))
                                p_j_94.append(id.index(row[4]))
                        if row[4] == '95':
                                x1=row[0]
                                x_j_95.append(x1)
                                x2=row[1]
                                x_j_95.append(x2)
                                x3= row[2]
                                x_j_95.append(x3)
                                y_j_95.append(mvts.index(row[3]))
                                p_j_95.append(id.index(row[4]))
                        if row[4] == '96':
                                x1=row[0]
                                x_j_96.append(x1)
                                x2=row[1]
                                x_j_96.append(x2)
                                x3= row[2]
                                x_j_96.append(x3)
                                y_j_96.append(mvts.index(row[3]))
                                p_j_96.append(id.index(row[4]))
                        if row[4] == '97':
                                x1=row[0]
                                x_j_97.append(x1)
                                x2=row[1]
                                x_j_97.append(x2)
                                x3= row[2]
                                x_j_97.append(x3)
                                y_j_97.append(mvts.index(row[3]))
                                p_j_97.append(id.index(row[4]))
                        if row[4] == '98':
                                x1=row[0]
                                x_j_98.append(x1)
                                x2=row[1]
                                x_j_98.append(x2)
                                x3= row[2]
                                x_j_98.append(x3)
                                y_j_98.append(mvts.index(row[3]))
                                p_j_98.append(id.index(row[4]))
                        if row[4] == '99':
                                x1=row[0]
                                x_j_99.append(x1)
                                x2=row[1]
                                x_j_99.append(x2)
                                x3= row[2]
                                x_j_99.append(x3)
                                y_j_99.append(mvts.index(row[3]))
                                p_j_99.append(id.index(row[4]))
                if row[3]== ' skip':
                        if row[4] == '00':
                                x1=row[0]
                                x_s_0.append(x1)
                                x2=row[1]
                                x_s_0.append(x2)
                                x3= row[2]
                                x_s_0.append(x3)
                                y_s_0.append(mvts.index(row[3]))
                                p_s_0.append(id.index(row[4]))
                        if row[4] == '01':
                                x1=row[0]
                                x_s_1.append(x1)
                                x2=row[1]
                                x_s_1.append(x2)
                                x3= row[2]
                                x_s_1.append(x3)
                                y_s_1.append(mvts.index(row[3]))
                                p_s_1.append(id.index(row[4]))
                        if row[4] == '02':
                                x1=row[0]
                                x_s_2.append(x1)
                                x2=row[1]
                                x_s_2.append(x2)
                                x3= row[2]
                                x_s_2.append(x3)
                                y_s_2.append(mvts.index(row[3]))
                                p_s_2.append(id.index(row[4]))
                        if row[4] == '03':
                                x1=row[0]
                                x_s_3.append(x1)
                                x2=row[1]
                                x_s_3.append(x2)
                                x3= row[2]
                                x_s_3.append(x3)
                                y_s_3.append(mvts.index(row[3]))
                                p_s_3.append(id.index(row[4]))
                        if row[4] == '04':
                                x1=row[0]
                                x_s_4.append(x1)
                                x2=row[1]
                                x_s_4.append(x2)
                                x3= row[2]
                                x_s_4.append(x3)
                                y_s_4.append(mvts.index(row[3]))
                                p_s_4.append(id.index(row[4]))
                        if row[4] == '05':
                                x1=row[0]
                                x_s_5.append(x1)
                                x2=row[1]
                                x_s_5.append(x2)
                                x3= row[2]
                                x_s_5.append(x3)
                                y_s_5.append(mvts.index(row[3]))
                                p_s_5.append(id.index(row[4]))
                        if row[4] == '06':
                                x1=row[0]
                                x_s_6.append(x1)
                                x2=row[1]
                                x_s_6.append(x2)
                                x3= row[2]
                                x_s_6.append(x3)
                                y_s_6.append(mvts.index(row[3]))
                                p_s_6.append(id.index(row[4]))
                        if row[4] == '07':
                                x1=row[0]
                                x_s_7.append(x1)
                                x2=row[1]
                                x_s_7.append(x2)
                                x3= row[2]
                                x_s_7.append(x3)
                                y_s_7.append(mvts.index(row[3]))
                                p_s_7.append(id.index(row[4]))
                        if row[4] == '08':
                                x1=row[0]
                                x_s_8.append(x1)
                                x2=row[1]
                                x_s_8.append(x2)
                                x3= row[2]
                                x_s_8.append(x3)
                                y_s_8.append(mvts.index(row[3]))
                                p_s_8.append(id.index(row[4]))
                        if row[4] == '09':
                                x1=row[0]
                                x_s_9.append(x1)
                                x2=row[1]
                                x_s_9.append(x2)
                                x3= row[2]
                                x_s_9.append(x3)
                                y_s_9.append(mvts.index(row[3]))
                                p_s_9.append(id.index(row[4]))
                        if row[4] == '10':
                                x1=row[0]
                                x_s_10.append(x1)
                                x2=row[1]
                                x_s_10.append(x2)
                                x3= row[2]
                                x_s_10.append(x3)
                                y_s_10.append(mvts.index(row[3]))
                                p_s_10.append(id.index(row[4]))
                        if row[4] == '11':
                                x1=row[0]
                                x_s_11.append(x1)
                                x2=row[1]
                                x_s_11.append(x2)
                                x3= row[2]
                                x_s_11.append(x3)
                                y_s_11.append(mvts.index(row[3]))
                                p_s_11.append(id.index(row[4]))
                        if row[4] == '12':
                                x1=row[0]
                                x_s_12.append(x1)
                                x2=row[1]
                                x_s_12.append(x2)
                                x3= row[2]
                                x_s_12.append(x3)
                                y_s_12.append(mvts.index(row[3]))
                                p_s_12.append(id.index(row[4]))
                        if row[4] == '13':
                                x1=row[0]
                                x_s_13.append(x1)
                                x2=row[1]
                                x_s_13.append(x2)
                                x3= row[2]
                                x_s_13.append(x3)
                                y_s_13.append(mvts.index(row[3]))
                                p_s_13.append(id.index(row[4]))
                        if row[4] == '14':
                                x1=row[0]
                                x_s_14.append(x1)
                                x2=row[1]
                                x_s_14.append(x2)
                                x3= row[2]
                                x_s_14.append(x3)
                                y_s_14.append(mvts.index(row[3]))
                                p_s_14.append(id.index(row[4]))
                        if row[4] == '15':
                                x1=row[0]
                                x_s_15.append(x1)
                                x2=row[1]
                                x_s_15.append(x2)
                                x3= row[2]
                                x_s_15.append(x3)
                                y_s_15.append(mvts.index(row[3]))
                                p_s_15.append(id.index(row[4]))
                        if row[4] == '16':
                                x1=row[0]
                                x_s_16.append(x1)
                                x2=row[1]
                                x_s_16.append(x2)
                                x3= row[2]
                                x_s_16.append(x3)
                                y_s_16.append(mvts.index(row[3]))
                                p_s_16.append(id.index(row[4]))
                        if row[4] == '17':
                                x1=row[0]
                                x_s_17.append(x1)
                                x2=row[1]
                                x_s_17.append(x2)
                                x3= row[2]
                                x_s_17.append(x3)
                                y_s_17.append(mvts.index(row[3]))
                                p_s_17.append(id.index(row[4]))
                        if row[4] == '18':
                                x1=row[0]
                                x_s_18.append(x1)
                                x2=row[1]
                                x_s_18.append(x2)
                                x3= row[2]
                                x_s_18.append(x3)
                                y_s_18.append(mvts.index(row[3]))
                                p_s_18.append(id.index(row[4]))
                        if row[4] == '19':
                                x1=row[0]
                                x_s_19.append(x1)
                                x2=row[1]
                                x_s_19.append(x2)
                                x3= row[2]
                                x_s_19.append(x3)
                                y_s_19.append(mvts.index(row[3]))
                                p_s_19.append(id.index(row[4]))
                        if row[4] == '20':
                                x1=row[0]
                                x_s_20.append(x1)
                                x2=row[1]
                                x_s_20.append(x2)
                                x3= row[2]
                                x_s_20.append(x3)
                                y_s_20.append(mvts.index(row[3]))
                                p_s_20.append(id.index(row[4]))
                        if row[4] == '21':
                                x1=row[0]
                                x_s_21.append(x1)
                                x2=row[1]
                                x_s_21.append(x2)
                                x3= row[2]
                                x_s_21.append(x3)
                                y_s_21.append(mvts.index(row[3]))
                                p_s_21.append(id.index(row[4]))
                        if row[4] == '22':
                                x1=row[0]
                                x_s_22.append(x1)
                                x2=row[1]
                                x_s_22.append(x2)
                                x3= row[2]
                                x_s_22.append(x3)
                                y_s_22.append(mvts.index(row[3]))
                                p_s_22.append(id.index(row[4]))
                        if row[4] == '23':
                                x1=row[0]
                                x_s_23.append(x1)
                                x2=row[1]
                                x_s_23.append(x2)
                                x3= row[2]
                                x_s_23.append(x3)
                                y_s_23.append(mvts.index(row[3]))
                                p_s_23.append(id.index(row[4]))
                        if row[4] == '24':
                                x1=row[0]
                                x_s_24.append(x1)
                                x2=row[1]
                                x_s_24.append(x2)
                                x3= row[2]
                                x_s_24.append(x3)
                                y_s_24.append(mvts.index(row[3]))
                                p_s_24.append(id.index(row[4]))
                        if row[4] == '25':
                                x1=row[0]
                                x_s_25.append(x1)
                                x2=row[1]
                                x_s_25.append(x2)
                                x3= row[2]
                                x_s_25.append(x3)
                                y_s_25.append(mvts.index(row[3]))
                                p_s_25.append(id.index(row[4]))
                        if row[4] == '26':
                                x1=row[0]
                                x_s_26.append(x1)
                                x2=row[1]
                                x_s_26.append(x2)
                                x3= row[2]
                                x_s_26.append(x3)
                                y_s_26.append(mvts.index(row[3]))
                                p_s_26.append(id.index(row[4]))
                        if row[4] == '27':
                                x1=row[0]
                                x_s_27.append(x1)
                                x2=row[1]
                                x_s_27.append(x2)
                                x3= row[2]
                                x_s_27.append(x3)
                                y_s_27.append(mvts.index(row[3]))
                                p_s_27.append(id.index(row[4]))
                        if row[4] == '28':
                                x1=row[0]
                                x_s_28.append(x1)
                                x2=row[1]
                                x_s_28.append(x2)
                                x3= row[2]
                                x_s_28.append(x3)
                                y_s_28.append(mvts.index(row[3]))
                                p_s_28.append(id.index(row[4]))
                        if row[4] == '29':
                                x1=row[0]
                                x_s_29.append(x1)
                                x2=row[1]
                                x_s_29.append(x2)
                                x3= row[2]
                                x_s_29.append(x3)
                                y_s_29.append(mvts.index(row[3]))
                                p_s_29.append(id.index(row[4]))
                        if row[4] == '30':
                                x1=row[0]
                                x_s_30.append(x1)
                                x2=row[1]
                                x_s_30.append(x2)
                                x3= row[2]
                                x_s_30.append(x3)
                                y_s_30.append(mvts.index(row[3]))
                                p_s_30.append(id.index(row[4]))
                        if row[4] == '31':
                                x1=row[0]
                                x_s_31.append(x1)
                                x2=row[1]
                                x_s_31.append(x2)
                                x3= row[2]
                                x_s_31.append(x3)
                                y_s_31.append(mvts.index(row[3]))
                                p_s_31.append(id.index(row[4]))
                        if row[4] == '32':
                                x1=row[0]
                                x_s_32.append(x1)
                                x2=row[1]
                                x_s_32.append(x2)
                                x3= row[2]
                                x_s_32.append(x3)
                                y_s_32.append(mvts.index(row[3]))
                                p_s_32.append(id.index(row[4]))
                        if row[4] == '33':
                                x1=row[0]
                                x_s_33.append(x1)
                                x2=row[1]
                                x_s_33.append(x2)
                                x3= row[2]
                                x_s_33.append(x3)
                                y_s_33.append(mvts.index(row[3]))
                                p_s_33.append(id.index(row[4]))
                        if row[4] == '34':
                                x1=row[0]
                                x_s_34.append(x1)
                                x2=row[1]
                                x_s_34.append(x2)
                                x3= row[2]
                                x_s_34.append(x3)
                                y_s_34.append(mvts.index(row[3]))
                                p_s_34.append(id.index(row[4]))
                        if row[4] == '35':
                                x1=row[0]
                                x_s_35.append(x1)
                                x2=row[1]
                                x_s_35.append(x2)
                                x3= row[2]
                                x_s_35.append(x3)
                                y_s_35.append(mvts.index(row[3]))
                                p_s_35.append(id.index(row[4]))
                        if row[4] == '36':
                                x1=row[0]
                                x_s_36.append(x1)
                                x2=row[1]
                                x_s_36.append(x2)
                                x3= row[2]
                                x_s_36.append(x3)
                                y_s_36.append(mvts.index(row[3]))
                                p_s_36.append(id.index(row[4]))
                        if row[4] == '37':
                                x1=row[0]
                                x_s_37.append(x1)
                                x2=row[1]
                                x_s_37.append(x2)
                                x3= row[2]
                                x_s_37.append(x3)
                                y_s_37.append(mvts.index(row[3]))
                                p_s_37.append(id.index(row[4]))
                        if row[4] == '38':
                                x1=row[0]
                                x_s_38.append(x1)
                                x2=row[1]
                                x_s_38.append(x2)
                                x3= row[2]
                                x_s_38.append(x3)
                                y_s_38.append(mvts.index(row[3]))
                                p_s_38.append(id.index(row[4]))
                        if row[4] == '39':
                                x1=row[0]
                                x_s_39.append(x1)
                                x2=row[1]
                                x_s_39.append(x2)
                                x3= row[2]
                                x_s_39.append(x3)
                                y_s_39.append(mvts.index(row[3]))
                                p_s_39.append(id.index(row[4]))
                        if row[4] == '40':
                                x1=row[0]
                                x_s_40.append(x1)
                                x2=row[1]
                                x_s_40.append(x2)
                                x3= row[2]
                                x_s_40.append(x3)
                                y_s_40.append(mvts.index(row[3]))
                                p_s_40.append(id.index(row[4]))
                        if row[4] == '41':
                                x1=row[0]
                                x_s_41.append(x1)
                                x2=row[1]
                                x_s_41.append(x2)
                                x3= row[2]
                                x_s_41.append(x3)
                                y_s_41.append(mvts.index(row[3]))
                                p_s_41.append(id.index(row[4]))
                        if row[4] == '42':
                                x1=row[0]
                                x_s_42.append(x1)
                                x2=row[1]
                                x_s_42.append(x2)
                                x3= row[2]
                                x_s_42.append(x3)
                                y_s_42.append(mvts.index(row[3]))
                                p_s_42.append(id.index(row[4]))
                        if row[4] == '43':
                                x1=row[0]
                                x_s_43.append(x1)
                                x2=row[1]
                                x_s_43.append(x2)
                                x3= row[2]
                                x_s_43.append(x3)
                                y_s_43.append(mvts.index(row[3]))
                                p_s_43.append(id.index(row[4]))
                        if row[4] == '44':
                                x1=row[0]
                                x_s_44.append(x1)
                                x2=row[1]
                                x_s_44.append(x2)
                                x3= row[2]
                                x_s_44.append(x3)
                                y_s_44.append(mvts.index(row[3]))
                                p_s_44.append(id.index(row[4]))
                        if row[4] == '45':
                                x1=row[0]
                                x_s_45.append(x1)
                                x2=row[1]
                                x_s_45.append(x2)
                                x3= row[2]
                                x_s_45.append(x3)
                                y_s_45.append(mvts.index(row[3]))
                                p_s_45.append(id.index(row[4]))
                        if row[4] == '46':
                                x1=row[0]
                                x_s_46.append(x1)
                                x2=row[1]
                                x_s_46.append(x2)
                                x3= row[2]
                                x_s_46.append(x3)
                                y_s_46.append(mvts.index(row[3]))
                                p_s_46.append(id.index(row[4]))
                        if row[4] == '47':
                                x1=row[0]
                                x_s_47.append(x1)
                                x2=row[1]
                                x_s_47.append(x2)
                                x3= row[2]
                                x_s_47.append(x3)
                                y_s_47.append(mvts.index(row[3]))
                                p_s_47.append(id.index(row[4]))
                        if row[4] == '48':
                                x1=row[0]
                                x_s_48.append(x1)
                                x2=row[1]
                                x_s_48.append(x2)
                                x3= row[2]
                                x_s_48.append(x3)
                                y_s_48.append(mvts.index(row[3]))
                                p_s_48.append(id.index(row[4]))
                        if row[4] == '49':
                                x1=row[0]
                                x_s_49.append(x1)
                                x2=row[1]
                                x_s_49.append(x2)
                                x3= row[2]
                                x_s_49.append(x3)
                                y_s_49.append(mvts.index(row[3]))
                                p_s_49.append(id.index(row[4]))
                        if row[4] == '50':
                                x1=row[0]
                                x_s_50.append(x1)
                                x2=row[1]
                                x_s_50.append(x2)
                                x3= row[2]
                                x_s_50.append(x3)
                                y_s_50.append(mvts.index(row[3]))
                                p_s_50.append(id.index(row[4]))
                        if row[4] == '51':
                                x1=row[0]
                                x_s_51.append(x1)
                                x2=row[1]
                                x_s_51.append(x2)
                                x3= row[2]
                                x_s_51.append(x3)
                                y_s_51.append(mvts.index(row[3]))
                                p_s_51.append(id.index(row[4]))
                        if row[4] == '52':
                                x1=row[0]
                                x_s_52.append(x1)
                                x2=row[1]
                                x_s_52.append(x2)
                                x3= row[2]
                                x_s_52.append(x3)
                                y_s_52.append(mvts.index(row[3]))
                                p_s_52.append(id.index(row[4]))
                        if row[4] == '53':
                                x1=row[0]
                                x_s_53.append(x1)
                                x2=row[1]
                                x_s_53.append(x2)
                                x3= row[2]
                                x_s_53.append(x3)
                                y_s_53.append(mvts.index(row[3]))
                                p_s_53.append(id.index(row[4]))
                        if row[4] == '54':
                                x1=row[0]
                                x_s_54.append(x1)
                                x2=row[1]
                                x_s_54.append(x2)
                                x3= row[2]
                                x_s_54.append(x3)
                                y_s_54.append(mvts.index(row[3]))
                                p_s_54.append(id.index(row[4]))
                        if row[4] == '55':
                                x1=row[0]
                                x_s_55.append(x1)
                                x2=row[1]
                                x_s_55.append(x2)
                                x3= row[2]
                                x_s_55.append(x3)
                                y_s_55.append(mvts.index(row[3]))
                                p_s_55.append(id.index(row[4]))
                        if row[4] == '56':
                                x1=row[0]
                                x_s_56.append(x1)
                                x2=row[1]
                                x_s_56.append(x2)
                                x3= row[2]
                                x_s_56.append(x3)
                                y_s_56.append(mvts.index(row[3]))
                                p_s_56.append(id.index(row[4]))
                        if row[4] == '57':
                                x1=row[0]
                                x_s_57.append(x1)
                                x2=row[1]
                                x_s_57.append(x2)
                                x3= row[2]
                                x_s_57.append(x3)
                                y_s_57.append(mvts.index(row[3]))
                                p_s_57.append(id.index(row[4]))
                        if row[4] == '58':
                                x1=row[0]
                                x_s_58.append(x1)
                                x2=row[1]
                                x_s_58.append(x2)
                                x3= row[2]
                                x_s_58.append(x3)
                                y_s_58.append(mvts.index(row[3]))
                                p_s_58.append(id.index(row[4]))
                        if row[4] == '59':
                                x1=row[0]
                                x_s_59.append(x1)
                                x2=row[1]
                                x_s_59.append(x2)
                                x3= row[2]
                                x_s_59.append(x3)
                                y_s_59.append(mvts.index(row[3]))
                                p_s_59.append(id.index(row[4]))
                        if row[4] == '60':
                                x1=row[0]
                                x_s_60.append(x1)
                                x2=row[1]
                                x_s_60.append(x2)
                                x3= row[2]
                                x_s_60.append(x3)
                                y_s_60.append(mvts.index(row[3]))
                                p_s_60.append(id.index(row[4]))
                        if row[4] == '61':
                                x1=row[0]
                                x_s_61.append(x1)
                                x2=row[1]
                                x_s_61.append(x2)
                                x3= row[2]
                                x_s_61.append(x3)
                                y_s_61.append(mvts.index(row[3]))
                                p_s_61.append(id.index(row[4]))
                        if row[4] == '62':
                                x1=row[0]
                                x_s_62.append(x1)
                                x2=row[1]
                                x_s_62.append(x2)
                                x3= row[2]
                                x_s_62.append(x3)
                                y_s_62.append(mvts.index(row[3]))
                                p_s_62.append(id.index(row[4]))
                        if row[4] == '63':
                                x1=row[0]
                                x_s_63.append(x1)
                                x2=row[1]
                                x_s_63.append(x2)
                                x3= row[2]
                                x_s_63.append(x3)
                                y_s_63.append(mvts.index(row[3]))
                                p_s_63.append(id.index(row[4]))
                        if row[4] == '64':
                                x1=row[0]
                                x_s_64.append(x1)
                                x2=row[1]
                                x_s_64.append(x2)
                                x3= row[2]
                                x_s_64.append(x3)
                                y_s_64.append(mvts.index(row[3]))
                                p_s_64.append(id.index(row[4]))
                        if row[4] == '65':
                                x1=row[0]
                                x_s_65.append(x1)
                                x2=row[1]
                                x_s_65.append(x2)
                                x3= row[2]
                                x_s_65.append(x3)
                                y_s_65.append(mvts.index(row[3]))
                                p_s_65.append(id.index(row[4]))
                        if row[4] == '66':
                                x1=row[0]
                                x_s_66.append(x1)
                                x2=row[1]
                                x_s_66.append(x2)
                                x3= row[2]
                                x_s_66.append(x3)
                                y_s_66.append(mvts.index(row[3]))
                                p_s_66.append(id.index(row[4]))
                        if row[4] == '67':
                                x1=row[0]
                                x_s_67.append(x1)
                                x2=row[1]
                                x_s_67.append(x2)
                                x3= row[2]
                                x_s_67.append(x3)
                                y_s_67.append(mvts.index(row[3]))
                                p_s_67.append(id.index(row[4]))
                        if row[4] == '68':
                                x1=row[0]
                                x_s_68.append(x1)
                                x2=row[1]
                                x_s_68.append(x2)
                                x3= row[2]
                                x_s_68.append(x3)
                                y_s_68.append(mvts.index(row[3]))
                                p_s_68.append(id.index(row[4]))
                        if row[4] == '69':
                                x1=row[0]
                                x_s_69.append(x1)
                                x2=row[1]
                                x_s_69.append(x2)
                                x3= row[2]
                                x_s_69.append(x3)
                                y_s_69.append(mvts.index(row[3]))
                                p_s_69.append(id.index(row[4]))
                        if row[4] == '70':
                                x1=row[0]
                                x_s_70.append(x1)
                                x2=row[1]
                                x_s_70.append(x2)
                                x3= row[2]
                                x_s_70.append(x3)
                                y_s_70.append(mvts.index(row[3]))
                                p_s_70.append(id.index(row[4]))
                        if row[4] == '71':
                                x1=row[0]
                                x_s_71.append(x1)
                                x2=row[1]
                                x_s_71.append(x2)
                                x3= row[2]
                                x_s_71.append(x3)
                                y_s_71.append(mvts.index(row[3]))
                                p_s_71.append(id.index(row[4]))
                        if row[4] == '72':
                                x1=row[0]
                                x_s_72.append(x1)
                                x2=row[1]
                                x_s_72.append(x2)
                                x3= row[2]
                                x_s_72.append(x3)
                                y_s_72.append(mvts.index(row[3]))
                                p_s_72.append(id.index(row[4]))
                        if row[4] == '73':
                                x1=row[0]
                                x_s_73.append(x1)
                                x2=row[1]
                                x_s_73.append(x2)
                                x3= row[2]
                                x_s_73.append(x3)
                                y_s_73.append(mvts.index(row[3]))
                                p_s_73.append(id.index(row[4]))
                        if row[4] == '74':
                                x1=row[0]
                                x_s_74.append(x1)
                                x2=row[1]
                                x_s_74.append(x2)
                                x3= row[2]
                                x_s_74.append(x3)
                                y_s_74.append(mvts.index(row[3]))
                                p_s_74.append(id.index(row[4]))
                        if row[4] == '75':
                                x1=row[0]
                                x_s_75.append(x1)
                                x2=row[1]
                                x_s_75.append(x2)
                                x3= row[2]
                                x_s_75.append(x3)
                                y_s_75.append(mvts.index(row[3]))
                                p_s_75.append(id.index(row[4]))
                        if row[4] == '76':
                                x1=row[0]
                                x_s_76.append(x1)
                                x2=row[1]
                                x_s_76.append(x2)
                                x3= row[2]
                                x_s_76.append(x3)
                                y_s_76.append(mvts.index(row[3]))
                                p_s_76.append(id.index(row[4]))
                        if row[4] == '77':
                                x1=row[0]
                                x_s_77.append(x1)
                                x2=row[1]
                                x_s_77.append(x2)
                                x3= row[2]
                                x_s_77.append(x3)
                                y_s_77.append(mvts.index(row[3]))
                                p_s_77.append(id.index(row[4]))
                        if row[4] == '78':
                                x1=row[0]
                                x_s_78.append(x1)
                                x2=row[1]
                                x_s_78.append(x2)
                                x3= row[2]
                                x_s_78.append(x3)
                                y_s_78.append(mvts.index(row[3]))
                                p_s_78.append(id.index(row[4]))
                        if row[4] == '79':
                                x1=row[0]
                                x_s_79.append(x1)
                                x2=row[1]
                                x_s_79.append(x2)
                                x3= row[2]
                                x_s_79.append(x3)
                                y_s_79.append(mvts.index(row[3]))
                                p_s_79.append(id.index(row[4]))
                        if row[4] == '80':
                                x1=row[0]
                                x_s_80.append(x1)
                                x2=row[1]
                                x_s_80.append(x2)
                                x3= row[2]
                                x_s_80.append(x3)
                                y_s_80.append(mvts.index(row[3]))
                                p_s_80.append(id.index(row[4]))
                        if row[4] == '81':
                                x1=row[0]
                                x_s_81.append(x1)
                                x2=row[1]
                                x_s_81.append(x2)
                                x3= row[2]
                                x_s_81.append(x3)
                                y_s_81.append(mvts.index(row[3]))
                                p_s_81.append(id.index(row[4]))
                        if row[4] == '82':
                                x1=row[0]
                                x_s_82.append(x1)
                                x2=row[1]
                                x_s_82.append(x2)
                                x3= row[2]
                                x_s_82.append(x3)
                                y_s_82.append(mvts.index(row[3]))
                                p_s_82.append(id.index(row[4]))
                        if row[4] == '83':
                                x1=row[0]
                                x_s_83.append(x1)
                                x2=row[1]
                                x_s_83.append(x2)
                                x3= row[2]
                                x_s_83.append(x3)
                                y_s_83.append(mvts.index(row[3]))
                                p_s_83.append(id.index(row[4]))
                        if row[4] == '84':
                                x1=row[0]
                                x_s_84.append(x1)
                                x2=row[1]
                                x_s_84.append(x2)
                                x3= row[2]
                                x_s_84.append(x3)
                                y_s_84.append(mvts.index(row[3]))
                                p_s_84.append(id.index(row[4]))
                        if row[4] == '85':
                                x1=row[0]
                                x_s_85.append(x1)
                                x2=row[1]
                                x_s_85.append(x2)
                                x3= row[2]
                                x_s_85.append(x3)
                                y_s_85.append(mvts.index(row[3]))
                                p_s_85.append(id.index(row[4]))
                        if row[4] == '86':
                                x1=row[0]
                                x_s_86.append(x1)
                                x2=row[1]
                                x_s_86.append(x2)
                                x3= row[2]
                                x_s_86.append(x3)
                                y_s_86.append(mvts.index(row[3]))
                                p_s_86.append(id.index(row[4]))
                        if row[4] == '87':
                                x1=row[0]
                                x_s_87.append(x1)
                                x2=row[1]
                                x_s_87.append(x2)
                                x3= row[2]
                                x_s_87.append(x3)
                                y_s_87.append(mvts.index(row[3]))
                                p_s_87.append(id.index(row[4]))
                        if row[4] == '88':
                                x1=row[0]
                                x_s_88.append(x1)
                                x2=row[1]
                                x_s_88.append(x2)
                                x3= row[2]
                                x_s_88.append(x3)
                                y_s_88.append(mvts.index(row[3]))
                                p_s_88.append(id.index(row[4]))
                        if row[4] == '89':
                                x1=row[0]
                                x_s_89.append(x1)
                                x2=row[1]
                                x_s_89.append(x2)
                                x3= row[2]
                                x_s_89.append(x3)
                                y_s_89.append(mvts.index(row[3]))
                                p_s_89.append(id.index(row[4]))
                        if row[4] == '90':
                                x1=row[0]
                                x_s_90.append(x1)
                                x2=row[1]
                                x_s_90.append(x2)
                                x3= row[2]
                                x_s_90.append(x3)
                                y_s_90.append(mvts.index(row[3]))
                                p_s_90.append(id.index(row[4]))
                        if row[4] == '91':
                                x1=row[0]
                                x_s_91.append(x1)
                                x2=row[1]
                                x_s_91.append(x2)
                                x3= row[2]
                                x_s_91.append(x3)
                                y_s_91.append(mvts.index(row[3]))
                                p_s_91.append(id.index(row[4]))
                        if row[4] == '92':
                                x1=row[0]
                                x_s_92.append(x1)
                                x2=row[1]
                                x_s_92.append(x2)
                                x3= row[2]
                                x_s_92.append(x3)
                                y_s_92.append(mvts.index(row[3]))
                                p_s_92.append(id.index(row[4]))
                        if row[4] == '93':
                                x1=row[0]
                                x_s_93.append(x1)
                                x2=row[1]
                                x_s_93.append(x2)
                                x3= row[2]
                                x_s_93.append(x3)
                                y_s_93.append(mvts.index(row[3]))
                                p_s_93.append(id.index(row[4]))
                        if row[4] == '94':
                                x1=row[0]
                                x_s_94.append(x1)
                                x2=row[1]
                                x_s_94.append(x2)
                                x3= row[2]
                                x_s_94.append(x3)
                                y_s_94.append(mvts.index(row[3]))
                                p_s_94.append(id.index(row[4]))
                        if row[4] == '95':
                                x1=row[0]
                                x_s_95.append(x1)
                                x2=row[1]
                                x_s_95.append(x2)
                                x3= row[2]
                                x_s_95.append(x3)
                                y_s_95.append(mvts.index(row[3]))
                                p_s_95.append(id.index(row[4]))
                        if row[4] == '96':
                                x1=row[0]
                                x_s_96.append(x1)
                                x2=row[1]
                                x_s_96.append(x2)
                                x3= row[2]
                                x_s_96.append(x3)
                                y_s_96.append(mvts.index(row[3]))
                                p_s_96.append(id.index(row[4]))
                        if row[4] == '97':
                                x1=row[0]
                                x_s_97.append(x1)
                                x2=row[1]
                                x_s_97.append(x2)
                                x3= row[2]
                                x_s_97.append(x3)
                                y_s_97.append(mvts.index(row[3]))
                                p_s_97.append(id.index(row[4]))
                        if row[4] == '98':
                                x1=row[0]
                                x_s_98.append(x1)
                                x2=row[1]
                                x_s_98.append(x2)
                                x3= row[2]
                                x_s_98.append(x3)
                                y_s_98.append(mvts.index(row[3]))
                                p_s_98.append(id.index(row[4]))
                        if row[4] == '99':
                                x1=row[0]
                                x_s_99.append(x1)
                                x2=row[1]
                                x_s_99.append(x2)
                                x3= row[2]
                                x_s_99.append(x3)
                                y_s_99.append(mvts.index(row[3]))
                                p_s_99.append(id.index(row[4]))
                if row[3]==' walk':
                        if row[4] == '00':
                                x1=row[0]
                                x_w_0.append(x1)
                                x2=row[1]
                                x_w_0.append(x2)
                                x3= row[2]
                                x_w_0.append(x3)
                                y_w_0.append(mvts.index(row[3]))
                                p_w_0.append(id.index(row[4]))
                        if row[4] == '01':
                                x1=row[0]
                                x_w_1.append(x1)
                                x2=row[1]
                                x_w_1.append(x2)
                                x3= row[2]
                                x_w_1.append(x3)
                                y_w_1.append(mvts.index(row[3]))
                                p_w_1.append(id.index(row[4]))
                        if row[4] == '02':
                                x1=row[0]
                                x_w_2.append(x1)
                                x2=row[1]
                                x_w_2.append(x2)
                                x3= row[2]
                                x_w_2.append(x3)
                                y_w_2.append(mvts.index(row[3]))
                                p_w_2.append(id.index(row[4]))
                        if row[4] == '03':
                                x1=row[0]
                                x_w_3.append(x1)
                                x2=row[1]
                                x_w_3.append(x2)
                                x3= row[2]
                                x_w_3.append(x3)
                                y_w_3.append(mvts.index(row[3]))
                                p_w_3.append(id.index(row[4]))
                        if row[4] == '04':
                                x1=row[0]
                                x_w_4.append(x1)
                                x2=row[1]
                                x_w_4.append(x2)
                                x3= row[2]
                                x_w_4.append(x3)
                                y_w_4.append(mvts.index(row[3]))
                                p_w_4.append(id.index(row[4]))
                        if row[4] == '05':
                                x1=row[0]
                                x_w_5.append(x1)
                                x2=row[1]
                                x_w_5.append(x2)
                                x3= row[2]
                                x_w_5.append(x3)
                                y_w_5.append(mvts.index(row[3]))
                                p_w_5.append(id.index(row[4]))
                        if row[4] == '06':
                                x1=row[0]
                                x_w_6.append(x1)
                                x2=row[1]
                                x_w_6.append(x2)
                                x3= row[2]
                                x_w_6.append(x3)
                                y_w_6.append(mvts.index(row[3]))
                                p_w_6.append(id.index(row[4]))
                        if row[4] == '07':
                                x1=row[0]
                                x_w_7.append(x1)
                                x2=row[1]
                                x_w_7.append(x2)
                                x3= row[2]
                                x_w_7.append(x3)
                                y_w_7.append(mvts.index(row[3]))
                                p_w_7.append(id.index(row[4]))
                        if row[4] == '08':
                                x1=row[0]
                                x_w_8.append(x1)
                                x2=row[1]
                                x_w_8.append(x2)
                                x3= row[2]
                                x_w_8.append(x3)
                                y_w_8.append(mvts.index(row[3]))
                                p_w_8.append(id.index(row[4]))
                        if row[4] == '09':
                                x1=row[0]
                                x_w_9.append(x1)
                                x2=row[1]
                                x_w_9.append(x2)
                                x3= row[2]
                                x_w_9.append(x3)
                                y_w_9.append(mvts.index(row[3]))
                                p_w_9.append(id.index(row[4]))
                        if row[4] == '10':
                                x1=row[0]
                                x_w_10.append(x1)
                                x2=row[1]
                                x_w_10.append(x2)
                                x3= row[2]
                                x_w_10.append(x3)
                                y_w_10.append(mvts.index(row[3]))
                                p_w_10.append(id.index(row[4]))
                        if row[4] == '11':
                                x1=row[0]
                                x_w_11.append(x1)
                                x2=row[1]
                                x_w_11.append(x2)
                                x3= row[2]
                                x_w_11.append(x3)
                                y_w_11.append(mvts.index(row[3]))
                                p_w_11.append(id.index(row[4]))
                        if row[4] == '12':
                                x1=row[0]
                                x_w_12.append(x1)
                                x2=row[1]
                                x_w_12.append(x2)
                                x3= row[2]
                                x_w_12.append(x3)
                                y_w_12.append(mvts.index(row[3]))
                                p_w_12.append(id.index(row[4]))
                        if row[4] == '13':
                                x1=row[0]
                                x_w_13.append(x1)
                                x2=row[1]
                                x_w_13.append(x2)
                                x3= row[2]
                                x_w_13.append(x3)
                                y_w_13.append(mvts.index(row[3]))
                                p_w_13.append(id.index(row[4]))
                        if row[4] == '14':
                                x1=row[0]
                                x_w_14.append(x1)
                                x2=row[1]
                                x_w_14.append(x2)
                                x3= row[2]
                                x_w_14.append(x3)
                                y_w_14.append(mvts.index(row[3]))
                                p_w_14.append(id.index(row[4]))
                        if row[4] == '15':
                                x1=row[0]
                                x_w_15.append(x1)
                                x2=row[1]
                                x_w_15.append(x2)
                                x3= row[2]
                                x_w_15.append(x3)
                                y_w_15.append(mvts.index(row[3]))
                                p_w_15.append(id.index(row[4]))
                        if row[4] == '16':
                                x1=row[0]
                                x_w_16.append(x1)
                                x2=row[1]
                                x_w_16.append(x2)
                                x3= row[2]
                                x_w_16.append(x3)
                                y_w_16.append(mvts.index(row[3]))
                                p_w_16.append(id.index(row[4]))
                        if row[4] == '17':
                                x1=row[0]
                                x_w_17.append(x1)
                                x2=row[1]
                                x_w_17.append(x2)
                                x3= row[2]
                                x_w_17.append(x3)
                                y_w_17.append(mvts.index(row[3]))
                                p_w_17.append(id.index(row[4]))
                        if row[4] == '18':
                                x1=row[0]
                                x_w_18.append(x1)
                                x2=row[1]
                                x_w_18.append(x2)
                                x3= row[2]
                                x_w_18.append(x3)
                                y_w_18.append(mvts.index(row[3]))
                                p_w_18.append(id.index(row[4]))
                        if row[4] == '19':
                                x1=row[0]
                                x_w_19.append(x1)
                                x2=row[1]
                                x_w_19.append(x2)
                                x3= row[2]
                                x_w_19.append(x3)
                                y_w_19.append(mvts.index(row[3]))
                                p_w_19.append(id.index(row[4]))
                        if row[4] == '20':
                                x1=row[0]
                                x_w_20.append(x1)
                                x2=row[1]
                                x_w_20.append(x2)
                                x3= row[2]
                                x_w_20.append(x3)
                                y_w_20.append(mvts.index(row[3]))
                                p_w_20.append(id.index(row[4]))
                        if row[4] == '21':
                                x1=row[0]
                                x_w_21.append(x1)
                                x2=row[1]
                                x_w_21.append(x2)
                                x3= row[2]
                                x_w_21.append(x3)
                                y_w_21.append(mvts.index(row[3]))
                                p_w_21.append(id.index(row[4]))
                        if row[4] == '22':
                                x1=row[0]
                                x_w_22.append(x1)
                                x2=row[1]
                                x_w_22.append(x2)
                                x3= row[2]
                                x_w_22.append(x3)
                                y_w_22.append(mvts.index(row[3]))
                                p_w_22.append(id.index(row[4]))
                        if row[4] == '23':
                                x1=row[0]
                                x_w_23.append(x1)
                                x2=row[1]
                                x_w_23.append(x2)
                                x3= row[2]
                                x_w_23.append(x3)
                                y_w_23.append(mvts.index(row[3]))
                                p_w_23.append(id.index(row[4]))
                        if row[4] == '24':
                                x1=row[0]
                                x_w_24.append(x1)
                                x2=row[1]
                                x_w_24.append(x2)
                                x3= row[2]
                                x_w_24.append(x3)
                                y_w_24.append(mvts.index(row[3]))
                                p_w_24.append(id.index(row[4]))
                        if row[4] == '25':
                                x1=row[0]
                                x_w_25.append(x1)
                                x2=row[1]
                                x_w_25.append(x2)
                                x3= row[2]
                                x_w_25.append(x3)
                                y_w_25.append(mvts.index(row[3]))
                                p_w_25.append(id.index(row[4]))
                        if row[4] == '26':
                                x1=row[0]
                                x_w_26.append(x1)
                                x2=row[1]
                                x_w_26.append(x2)
                                x3= row[2]
                                x_w_26.append(x3)
                                y_w_26.append(mvts.index(row[3]))
                                p_w_26.append(id.index(row[4]))
                        if row[4] == '27':
                                x1=row[0]
                                x_w_27.append(x1)
                                x2=row[1]
                                x_w_27.append(x2)
                                x3= row[2]
                                x_w_27.append(x3)
                                y_w_27.append(mvts.index(row[3]))
                                p_w_27.append(id.index(row[4]))
                        if row[4] == '28':
                                x1=row[0]
                                x_w_28.append(x1)
                                x2=row[1]
                                x_w_28.append(x2)
                                x3= row[2]
                                x_w_28.append(x3)
                                y_w_28.append(mvts.index(row[3]))
                                p_w_28.append(id.index(row[4]))
                        if row[4] == '29':
                                x1=row[0]
                                x_w_29.append(x1)
                                x2=row[1]
                                x_w_29.append(x2)
                                x3= row[2]
                                x_w_29.append(x3)
                                y_w_29.append(mvts.index(row[3]))
                                p_w_29.append(id.index(row[4]))
                        if row[4] == '30':
                                x1=row[0]
                                x_w_30.append(x1)
                                x2=row[1]
                                x_w_30.append(x2)
                                x3= row[2]
                                x_w_30.append(x3)
                                y_w_30.append(mvts.index(row[3]))
                                p_w_30.append(id.index(row[4]))
                        if row[4] == '31':
                                x1=row[0]
                                x_w_31.append(x1)
                                x2=row[1]
                                x_w_31.append(x2)
                                x3= row[2]
                                x_w_31.append(x3)
                                y_w_31.append(mvts.index(row[3]))
                                p_w_31.append(id.index(row[4]))
                        if row[4] == '32':
                                x1=row[0]
                                x_w_32.append(x1)
                                x2=row[1]
                                x_w_32.append(x2)
                                x3= row[2]
                                x_w_32.append(x3)
                                y_w_32.append(mvts.index(row[3]))
                                p_w_32.append(id.index(row[4]))
                        if row[4] == '33':
                                x1=row[0]
                                x_w_33.append(x1)
                                x2=row[1]
                                x_w_33.append(x2)
                                x3= row[2]
                                x_w_33.append(x3)
                                y_w_33.append(mvts.index(row[3]))
                                p_w_33.append(id.index(row[4]))
                        if row[4] == '34':
                                x1=row[0]
                                x_w_34.append(x1)
                                x2=row[1]
                                x_w_34.append(x2)
                                x3= row[2]
                                x_w_34.append(x3)
                                y_w_34.append(mvts.index(row[3]))
                                p_w_34.append(id.index(row[4]))
                        if row[4] == '35':
                                x1=row[0]
                                x_w_35.append(x1)
                                x2=row[1]
                                x_w_35.append(x2)
                                x3= row[2]
                                x_w_35.append(x3)
                                y_w_35.append(mvts.index(row[3]))
                                p_w_35.append(id.index(row[4]))
                        if row[4] == '36':
                                x1=row[0]
                                x_w_36.append(x1)
                                x2=row[1]
                                x_w_36.append(x2)
                                x3= row[2]
                                x_w_36.append(x3)
                                y_w_36.append(mvts.index(row[3]))
                                p_w_36.append(id.index(row[4]))
                        if row[4] == '37':
                                x1=row[0]
                                x_w_37.append(x1)
                                x2=row[1]
                                x_w_37.append(x2)
                                x3= row[2]
                                x_w_37.append(x3)
                                y_w_37.append(mvts.index(row[3]))
                                p_w_37.append(id.index(row[4]))
                        if row[4] == '38':
                                x1=row[0]
                                x_w_38.append(x1)
                                x2=row[1]
                                x_w_38.append(x2)
                                x3= row[2]
                                x_w_38.append(x3)
                                y_w_38.append(mvts.index(row[3]))
                                p_w_38.append(id.index(row[4]))
                        if row[4] == '39':
                                x1=row[0]
                                x_w_39.append(x1)
                                x2=row[1]
                                x_w_39.append(x2)
                                x3= row[2]
                                x_w_39.append(x3)
                                y_w_39.append(mvts.index(row[3]))
                                p_w_39.append(id.index(row[4]))
                        if row[4] == '40':
                                x1=row[0]
                                x_w_40.append(x1)
                                x2=row[1]
                                x_w_40.append(x2)
                                x3= row[2]
                                x_w_40.append(x3)
                                y_w_40.append(mvts.index(row[3]))
                                p_w_40.append(id.index(row[4]))
                        if row[4] == '41':
                                x1=row[0]
                                x_w_41.append(x1)
                                x2=row[1]
                                x_w_41.append(x2)
                                x3= row[2]
                                x_w_41.append(x3)
                                y_w_41.append(mvts.index(row[3]))
                                p_w_41.append(id.index(row[4]))
                        if row[4] == '42':
                                x1=row[0]
                                x_w_42.append(x1)
                                x2=row[1]
                                x_w_42.append(x2)
                                x3= row[2]
                                x_w_42.append(x3)
                                y_w_42.append(mvts.index(row[3]))
                                p_w_42.append(id.index(row[4]))
                        if row[4] == '43':
                                x1=row[0]
                                x_w_43.append(x1)
                                x2=row[1]
                                x_w_43.append(x2)
                                x3= row[2]
                                x_w_43.append(x3)
                                y_w_43.append(mvts.index(row[3]))
                                p_w_43.append(id.index(row[4]))
                        if row[4] == '44':
                                x1=row[0]
                                x_w_44.append(x1)
                                x2=row[1]
                                x_w_44.append(x2)
                                x3= row[2]
                                x_w_44.append(x3)
                                y_w_44.append(mvts.index(row[3]))
                                p_w_44.append(id.index(row[4]))
                        if row[4] == '45':
                                x1=row[0]
                                x_w_45.append(x1)
                                x2=row[1]
                                x_w_45.append(x2)
                                x3= row[2]
                                x_w_45.append(x3)
                                y_w_45.append(mvts.index(row[3]))
                                p_w_45.append(id.index(row[4]))
                        if row[4] == '46':
                                x1=row[0]
                                x_w_46.append(x1)
                                x2=row[1]
                                x_w_46.append(x2)
                                x3= row[2]
                                x_w_46.append(x3)
                                y_w_46.append(mvts.index(row[3]))
                                p_w_46.append(id.index(row[4]))
                        if row[4] == '47':
                                x1=row[0]
                                x_w_47.append(x1)
                                x2=row[1]
                                x_w_47.append(x2)
                                x3= row[2]
                                x_w_47.append(x3)
                                y_w_47.append(mvts.index(row[3]))
                                p_w_47.append(id.index(row[4]))
                        if row[4] == '48':
                                x1=row[0]
                                x_w_48.append(x1)
                                x2=row[1]
                                x_w_48.append(x2)
                                x3= row[2]
                                x_w_48.append(x3)
                                y_w_48.append(mvts.index(row[3]))
                                p_w_48.append(id.index(row[4]))
                        if row[4] == '49':
                                x1=row[0]
                                x_w_49.append(x1)
                                x2=row[1]
                                x_w_49.append(x2)
                                x3= row[2]
                                x_w_49.append(x3)
                                y_w_49.append(mvts.index(row[3]))
                                p_w_49.append(id.index(row[4]))
                        if row[4] == '50':
                                x1=row[0]
                                x_w_50.append(x1)
                                x2=row[1]
                                x_w_50.append(x2)
                                x3= row[2]
                                x_w_50.append(x3)
                                y_w_50.append(mvts.index(row[3]))
                                p_w_50.append(id.index(row[4]))
                        if row[4] == '51':
                                x1=row[0]
                                x_w_51.append(x1)
                                x2=row[1]
                                x_w_51.append(x2)
                                x3= row[2]
                                x_w_51.append(x3)
                                y_w_51.append(mvts.index(row[3]))
                                p_w_51.append(id.index(row[4]))
                        if row[4] == '52':
                                x1=row[0]
                                x_w_52.append(x1)
                                x2=row[1]
                                x_w_52.append(x2)
                                x3= row[2]
                                x_w_52.append(x3)
                                y_w_52.append(mvts.index(row[3]))
                                p_w_52.append(id.index(row[4]))
                        if row[4] == '53':
                                x1=row[0]
                                x_w_53.append(x1)
                                x2=row[1]
                                x_w_53.append(x2)
                                x3= row[2]
                                x_w_53.append(x3)
                                y_w_53.append(mvts.index(row[3]))
                                p_w_53.append(id.index(row[4]))
                        if row[4] == '54':
                                x1=row[0]
                                x_w_54.append(x1)
                                x2=row[1]
                                x_w_54.append(x2)
                                x3= row[2]
                                x_w_54.append(x3)
                                y_w_54.append(mvts.index(row[3]))
                                p_w_54.append(id.index(row[4]))
                        if row[4] == '55':
                                x1=row[0]
                                x_w_55.append(x1)
                                x2=row[1]
                                x_w_55.append(x2)
                                x3= row[2]
                                x_w_55.append(x3)
                                y_w_55.append(mvts.index(row[3]))
                                p_w_55.append(id.index(row[4]))
                        if row[4] == '56':
                                x1=row[0]
                                x_w_56.append(x1)
                                x2=row[1]
                                x_w_56.append(x2)
                                x3= row[2]
                                x_w_56.append(x3)
                                y_w_56.append(mvts.index(row[3]))
                                p_w_56.append(id.index(row[4]))
                        if row[4] == '57':
                                x1=row[0]
                                x_w_57.append(x1)
                                x2=row[1]
                                x_w_57.append(x2)
                                x3= row[2]
                                x_w_57.append(x3)
                                y_w_57.append(mvts.index(row[3]))
                                p_w_57.append(id.index(row[4]))
                        if row[4] == '58':
                                x1=row[0]
                                x_w_58.append(x1)
                                x2=row[1]
                                x_w_58.append(x2)
                                x3= row[2]
                                x_w_58.append(x3)
                                y_w_58.append(mvts.index(row[3]))
                                p_w_58.append(id.index(row[4]))
                        if row[4] == '59':
                                x1=row[0]
                                x_w_59.append(x1)
                                x2=row[1]
                                x_w_59.append(x2)
                                x3= row[2]
                                x_w_59.append(x3)
                                y_w_59.append(mvts.index(row[3]))
                                p_w_59.append(id.index(row[4]))
                        if row[4] == '60':
                                x1=row[0]
                                x_w_60.append(x1)
                                x2=row[1]
                                x_w_60.append(x2)
                                x3= row[2]
                                x_w_60.append(x3)
                                y_w_60.append(mvts.index(row[3]))
                                p_w_60.append(id.index(row[4]))
                        if row[4] == '61':
                                x1=row[0]
                                x_w_61.append(x1)
                                x2=row[1]
                                x_w_61.append(x2)
                                x3= row[2]
                                x_w_61.append(x3)
                                y_w_61.append(mvts.index(row[3]))
                                p_w_61.append(id.index(row[4]))
                        if row[4] == '62':
                                x1=row[0]
                                x_w_62.append(x1)
                                x2=row[1]
                                x_w_62.append(x2)
                                x3= row[2]
                                x_w_62.append(x3)
                                y_w_62.append(mvts.index(row[3]))
                                p_w_62.append(id.index(row[4]))
                        if row[4] == '63':
                                x1=row[0]
                                x_w_63.append(x1)
                                x2=row[1]
                                x_w_63.append(x2)
                                x3= row[2]
                                x_w_63.append(x3)
                                y_w_63.append(mvts.index(row[3]))
                                p_w_63.append(id.index(row[4]))
                        if row[4] == '64':
                                x1=row[0]
                                x_w_64.append(x1)
                                x2=row[1]
                                x_w_64.append(x2)
                                x3= row[2]
                                x_w_64.append(x3)
                                y_w_64.append(mvts.index(row[3]))
                                p_w_64.append(id.index(row[4]))
                        if row[4] == '65':
                                x1=row[0]
                                x_w_65.append(x1)
                                x2=row[1]
                                x_w_65.append(x2)
                                x3= row[2]
                                x_w_65.append(x3)
                                y_w_65.append(mvts.index(row[3]))
                                p_w_65.append(id.index(row[4]))
                        if row[4] == '66':
                                x1=row[0]
                                x_w_66.append(x1)
                                x2=row[1]
                                x_w_66.append(x2)
                                x3= row[2]
                                x_w_66.append(x3)
                                y_w_66.append(mvts.index(row[3]))
                                p_w_66.append(id.index(row[4]))
                        if row[4] == '67':
                                x1=row[0]
                                x_w_67.append(x1)
                                x2=row[1]
                                x_w_67.append(x2)
                                x3= row[2]
                                x_w_67.append(x3)
                                y_w_67.append(mvts.index(row[3]))
                                p_w_67.append(id.index(row[4]))
                        if row[4] == '68':
                                x1=row[0]
                                x_w_68.append(x1)
                                x2=row[1]
                                x_w_68.append(x2)
                                x3= row[2]
                                x_w_68.append(x3)
                                y_w_68.append(mvts.index(row[3]))
                                p_w_68.append(id.index(row[4]))
                        if row[4] == '69':
                                x1=row[0]
                                x_w_69.append(x1)
                                x2=row[1]
                                x_w_69.append(x2)
                                x3= row[2]
                                x_w_69.append(x3)
                                y_w_69.append(mvts.index(row[3]))
                                p_w_69.append(id.index(row[4]))
                        if row[4] == '70':
                                x1=row[0]
                                x_w_70.append(x1)
                                x2=row[1]
                                x_w_70.append(x2)
                                x3= row[2]
                                x_w_70.append(x3)
                                y_w_70.append(mvts.index(row[3]))
                                p_w_70.append(id.index(row[4]))
                        if row[4] == '71':
                                x1=row[0]
                                x_w_71.append(x1)
                                x2=row[1]
                                x_w_71.append(x2)
                                x3= row[2]
                                x_w_71.append(x3)
                                y_w_71.append(mvts.index(row[3]))
                                p_w_71.append(id.index(row[4]))
                        if row[4] == '72':
                                x1=row[0]
                                x_w_72.append(x1)
                                x2=row[1]
                                x_w_72.append(x2)
                                x3= row[2]
                                x_w_72.append(x3)
                                y_w_72.append(mvts.index(row[3]))
                                p_w_72.append(id.index(row[4]))
                        if row[4] == '73':
                                x1=row[0]
                                x_w_73.append(x1)
                                x2=row[1]
                                x_w_73.append(x2)
                                x3= row[2]
                                x_w_73.append(x3)
                                y_w_73.append(mvts.index(row[3]))
                                p_w_73.append(id.index(row[4]))
                        if row[4] == '74':
                                x1=row[0]
                                x_w_74.append(x1)
                                x2=row[1]
                                x_w_74.append(x2)
                                x3= row[2]
                                x_w_74.append(x3)
                                y_w_74.append(mvts.index(row[3]))
                                p_w_74.append(id.index(row[4]))
                        if row[4] == '75':
                                x1=row[0]
                                x_w_75.append(x1)
                                x2=row[1]
                                x_w_75.append(x2)
                                x3= row[2]
                                x_w_75.append(x3)
                                y_w_75.append(mvts.index(row[3]))
                                p_w_75.append(id.index(row[4]))
                        if row[4] == '76':
                                x1=row[0]
                                x_w_76.append(x1)
                                x2=row[1]
                                x_w_76.append(x2)
                                x3= row[2]
                                x_w_76.append(x3)
                                y_w_76.append(mvts.index(row[3]))
                                p_w_76.append(id.index(row[4]))
                        if row[4] == '77':
                                x1=row[0]
                                x_w_77.append(x1)
                                x2=row[1]
                                x_w_77.append(x2)
                                x3= row[2]
                                x_w_77.append(x3)
                                y_w_77.append(mvts.index(row[3]))
                                p_w_77.append(id.index(row[4]))
                        if row[4] == '78':
                                x1=row[0]
                                x_w_78.append(x1)
                                x2=row[1]
                                x_w_78.append(x2)
                                x3= row[2]
                                x_w_78.append(x3)
                                y_w_78.append(mvts.index(row[3]))
                                p_w_78.append(id.index(row[4]))
                        if row[4] == '79':
                                x1=row[0]
                                x_w_79.append(x1)
                                x2=row[1]
                                x_w_79.append(x2)
                                x3= row[2]
                                x_w_79.append(x3)
                                y_w_79.append(mvts.index(row[3]))
                                p_w_79.append(id.index(row[4]))
                        if row[4] == '80':
                                x1=row[0]
                                x_w_80.append(x1)
                                x2=row[1]
                                x_w_80.append(x2)
                                x3= row[2]
                                x_w_80.append(x3)
                                y_w_80.append(mvts.index(row[3]))
                                p_w_80.append(id.index(row[4]))
                        if row[4] == '81':
                                x1=row[0]
                                x_w_81.append(x1)
                                x2=row[1]
                                x_w_81.append(x2)
                                x3= row[2]
                                x_w_81.append(x3)
                                y_w_81.append(mvts.index(row[3]))
                                p_w_81.append(id.index(row[4]))
                        if row[4] == '82':
                                x1=row[0]
                                x_w_82.append(x1)
                                x2=row[1]
                                x_w_82.append(x2)
                                x3= row[2]
                                x_w_82.append(x3)
                                y_w_82.append(mvts.index(row[3]))
                                p_w_82.append(id.index(row[4]))
                        if row[4] == '83':
                                x1=row[0]
                                x_w_83.append(x1)
                                x2=row[1]
                                x_w_83.append(x2)
                                x3= row[2]
                                x_w_83.append(x3)
                                y_w_83.append(mvts.index(row[3]))
                                p_w_83.append(id.index(row[4]))
                        if row[4] == '84':
                                x1=row[0]
                                x_w_84.append(x1)
                                x2=row[1]
                                x_w_84.append(x2)
                                x3= row[2]
                                x_w_84.append(x3)
                                y_w_84.append(mvts.index(row[3]))
                                p_w_84.append(id.index(row[4]))
                        if row[4] == '85':
                                x1=row[0]
                                x_w_85.append(x1)
                                x2=row[1]
                                x_w_85.append(x2)
                                x3= row[2]
                                x_w_85.append(x3)
                                y_w_85.append(mvts.index(row[3]))
                                p_w_85.append(id.index(row[4]))
                        if row[4] == '86':
                                x1=row[0]
                                x_w_86.append(x1)
                                x2=row[1]
                                x_w_86.append(x2)
                                x3= row[2]
                                x_w_86.append(x3)
                                y_w_86.append(mvts.index(row[3]))
                                p_w_86.append(id.index(row[4]))
                        if row[4] == '87':
                                x1=row[0]
                                x_w_87.append(x1)
                                x2=row[1]
                                x_w_87.append(x2)
                                x3= row[2]
                                x_w_87.append(x3)
                                y_w_87.append(mvts.index(row[3]))
                                p_w_87.append(id.index(row[4]))
                        if row[4] == '88':
                                x1=row[0]
                                x_w_88.append(x1)
                                x2=row[1]
                                x_w_88.append(x2)
                                x3= row[2]
                                x_w_88.append(x3)
                                y_w_88.append(mvts.index(row[3]))
                                p_w_88.append(id.index(row[4]))
                        if row[4] == '89':
                                x1=row[0]
                                x_w_89.append(x1)
                                x2=row[1]
                                x_w_89.append(x2)
                                x3= row[2]
                                x_w_89.append(x3)
                                y_w_89.append(mvts.index(row[3]))
                                p_w_89.append(id.index(row[4]))
                        if row[4] == '90':
                                x1=row[0]
                                x_w_90.append(x1)
                                x2=row[1]
                                x_w_90.append(x2)
                                x3= row[2]
                                x_w_90.append(x3)
                                y_w_90.append(mvts.index(row[3]))
                                p_w_90.append(id.index(row[4]))
                        if row[4] == '91':
                                x1=row[0]
                                x_w_91.append(x1)
                                x2=row[1]
                                x_w_91.append(x2)
                                x3= row[2]
                                x_w_91.append(x3)
                                y_w_91.append(mvts.index(row[3]))
                                p_w_91.append(id.index(row[4]))
                        if row[4] == '92':
                                x1=row[0]
                                x_w_92.append(x1)
                                x2=row[1]
                                x_w_92.append(x2)
                                x3= row[2]
                                x_w_92.append(x3)
                                y_w_92.append(mvts.index(row[3]))
                                p_w_92.append(id.index(row[4]))
                        if row[4] == '93':
                                x1=row[0]
                                x_w_93.append(x1)
                                x2=row[1]
                                x_w_93.append(x2)
                                x3= row[2]
                                x_w_93.append(x3)
                                y_w_93.append(mvts.index(row[3]))
                                p_w_93.append(id.index(row[4]))
                        if row[4] == '94':
                                x1=row[0]
                                x_w_94.append(x1)
                                x2=row[1]
                                x_w_94.append(x2)
                                x3= row[2]
                                x_w_94.append(x3)
                                y_w_94.append(mvts.index(row[3]))
                                p_w_94.append(id.index(row[4]))
                        if row[4] == '95':
                                x1=row[0]
                                x_w_95.append(x1)
                                x2=row[1]
                                x_w_95.append(x2)
                                x3= row[2]
                                x_w_95.append(x3)
                                y_w_95.append(mvts.index(row[3]))
                                p_w_95.append(id.index(row[4]))
                        if row[4] == '96':
                                x1=row[0]
                                x_w_96.append(x1)
                                x2=row[1]
                                x_w_96.append(x2)
                                x3= row[2]
                                x_w_96.append(x3)
                                y_w_96.append(mvts.index(row[3]))
                                p_w_96.append(id.index(row[4]))
                        if row[4] == '97':
                                x1=row[0]
                                x_w_97.append(x1)
                                x2=row[1]
                                x_w_97.append(x2)
                                x3= row[2]
                                x_w_97.append(x3)
                                y_w_97.append(mvts.index(row[3]))
                                p_w_97.append(id.index(row[4]))
                        if row[4] == '98':
                                x1=row[0]
                                x_w_98.append(x1)
                                x2=row[1]
                                x_w_98.append(x2)
                                x3= row[2]
                                x_w_98.append(x3)
                                y_w_98.append(mvts.index(row[3]))
                                p_w_98.append(id.index(row[4]))
                        if row[4] == '99':
                                x1=row[0]
                                x_w_99.append(x1)
                                x2=row[1]
                                x_w_99.append(x2)
                                x3= row[2]
                                x_w_99.append(x3)
                                y_w_99.append(mvts.index(row[3]))
                                p_w_99.append(id.index(row[4]))
                if row[3]==' stUp':
                        if row[4] == '00':
                                x1=row[0]
                                x_stu_0.append(x1)
                                x2=row[1]
                                x_stu_0.append(x2)
                                x3= row[2]
                                x_stu_0.append(x3)
                                y_stu_0.append(mvts.index(row[3]))
                                p_stu_0.append(id.index(row[4]))
                        if row[4] == '01':
                                x1=row[0]
                                x_stu_1.append(x1)
                                x2=row[1]
                                x_stu_1.append(x2)
                                x3= row[2]
                                x_stu_1.append(x3)
                                y_stu_1.append(mvts.index(row[3]))
                                p_stu_1.append(id.index(row[4]))
                        if row[4] == '02':
                                x1=row[0]
                                x_stu_2.append(x1)
                                x2=row[1]
                                x_stu_2.append(x2)
                                x3= row[2]
                                x_stu_2.append(x3)
                                y_stu_2.append(mvts.index(row[3]))
                                p_stu_2.append(id.index(row[4]))
                        if row[4] == '03':
                                x1=row[0]
                                x_stu_3.append(x1)
                                x2=row[1]
                                x_stu_3.append(x2)
                                x3= row[2]
                                x_stu_3.append(x3)
                                y_stu_3.append(mvts.index(row[3]))
                                p_stu_3.append(id.index(row[4]))
                        if row[4] == '04':
                                x1=row[0]
                                x_stu_4.append(x1)
                                x2=row[1]
                                x_stu_4.append(x2)
                                x3= row[2]
                                x_stu_4.append(x3)
                                y_stu_4.append(mvts.index(row[3]))
                                p_stu_4.append(id.index(row[4]))
                        if row[4] == '05':
                                x1=row[0]
                                x_stu_5.append(x1)
                                x2=row[1]
                                x_stu_5.append(x2)
                                x3= row[2]
                                x_stu_5.append(x3)
                                y_stu_5.append(mvts.index(row[3]))
                                p_stu_5.append(id.index(row[4]))
                        if row[4] == '06':
                                x1=row[0]
                                x_stu_6.append(x1)
                                x2=row[1]
                                x_stu_6.append(x2)
                                x3= row[2]
                                x_stu_6.append(x3)
                                y_stu_6.append(mvts.index(row[3]))
                                p_stu_6.append(id.index(row[4]))
                        if row[4] == '07':
                                x1=row[0]
                                x_stu_7.append(x1)
                                x2=row[1]
                                x_stu_7.append(x2)
                                x3= row[2]
                                x_stu_7.append(x3)
                                y_stu_7.append(mvts.index(row[3]))
                                p_stu_7.append(id.index(row[4]))
                        if row[4] == '08':
                                x1=row[0]
                                x_stu_8.append(x1)
                                x2=row[1]
                                x_stu_8.append(x2)
                                x3= row[2]
                                x_stu_8.append(x3)
                                y_stu_8.append(mvts.index(row[3]))
                                p_stu_8.append(id.index(row[4]))
                        if row[4] == '09':
                                x1=row[0]
                                x_stu_9.append(x1)
                                x2=row[1]
                                x_stu_9.append(x2)
                                x3= row[2]
                                x_stu_9.append(x3)
                                y_stu_9.append(mvts.index(row[3]))
                                p_stu_9.append(id.index(row[4]))
                        if row[4] == '10':
                                x1=row[0]
                                x_stu_10.append(x1)
                                x2=row[1]
                                x_stu_10.append(x2)
                                x3= row[2]
                                x_stu_10.append(x3)
                                y_stu_10.append(mvts.index(row[3]))
                                p_stu_10.append(id.index(row[4]))
                        if row[4] == '11':
                                x1=row[0]
                                x_stu_11.append(x1)
                                x2=row[1]
                                x_stu_11.append(x2)
                                x3= row[2]
                                x_stu_11.append(x3)
                                y_stu_11.append(mvts.index(row[3]))
                                p_stu_11.append(id.index(row[4]))
                        if row[4] == '12':
                                x1=row[0]
                                x_stu_12.append(x1)
                                x2=row[1]
                                x_stu_12.append(x2)
                                x3= row[2]
                                x_stu_12.append(x3)
                                y_stu_12.append(mvts.index(row[3]))
                                p_stu_12.append(id.index(row[4]))
                        if row[4] == '13':
                                x1=row[0]
                                x_stu_13.append(x1)
                                x2=row[1]
                                x_stu_13.append(x2)
                                x3= row[2]
                                x_stu_13.append(x3)
                                y_stu_13.append(mvts.index(row[3]))
                                p_stu_13.append(id.index(row[4]))
                        if row[4] == '14':
                                x1=row[0]
                                x_stu_14.append(x1)
                                x2=row[1]
                                x_stu_14.append(x2)
                                x3= row[2]
                                x_stu_14.append(x3)
                                y_stu_14.append(mvts.index(row[3]))
                                p_stu_14.append(id.index(row[4]))
                        if row[4] == '15':
                                x1=row[0]
                                x_stu_15.append(x1)
                                x2=row[1]
                                x_stu_15.append(x2)
                                x3= row[2]
                                x_stu_15.append(x3)
                                y_stu_15.append(mvts.index(row[3]))
                                p_stu_15.append(id.index(row[4]))
                        if row[4] == '16':
                                x1=row[0]
                                x_stu_16.append(x1)
                                x2=row[1]
                                x_stu_16.append(x2)
                                x3= row[2]
                                x_stu_16.append(x3)
                                y_stu_16.append(mvts.index(row[3]))
                                p_stu_16.append(id.index(row[4]))
                        if row[4] == '17':
                                x1=row[0]
                                x_stu_17.append(x1)
                                x2=row[1]
                                x_stu_17.append(x2)
                                x3= row[2]
                                x_stu_17.append(x3)
                                y_stu_17.append(mvts.index(row[3]))
                                p_stu_17.append(id.index(row[4]))
                        if row[4] == '18':
                                x1=row[0]
                                x_stu_18.append(x1)
                                x2=row[1]
                                x_stu_18.append(x2)
                                x3= row[2]
                                x_stu_18.append(x3)
                                y_stu_18.append(mvts.index(row[3]))
                                p_stu_18.append(id.index(row[4]))
                        if row[4] == '19':
                                x1=row[0]
                                x_stu_19.append(x1)
                                x2=row[1]
                                x_stu_19.append(x2)
                                x3= row[2]
                                x_stu_19.append(x3)
                                y_stu_19.append(mvts.index(row[3]))
                                p_stu_19.append(id.index(row[4]))
                        if row[4] == '20':
                                x1=row[0]
                                x_stu_20.append(x1)
                                x2=row[1]
                                x_stu_20.append(x2)
                                x3= row[2]
                                x_stu_20.append(x3)
                                y_stu_20.append(mvts.index(row[3]))
                                p_stu_20.append(id.index(row[4]))
                        if row[4] == '21':
                                x1=row[0]
                                x_stu_21.append(x1)
                                x2=row[1]
                                x_stu_21.append(x2)
                                x3= row[2]
                                x_stu_21.append(x3)
                                y_stu_21.append(mvts.index(row[3]))
                                p_stu_21.append(id.index(row[4]))
                        if row[4] == '22':
                                x1=row[0]
                                x_stu_22.append(x1)
                                x2=row[1]
                                x_stu_22.append(x2)
                                x3= row[2]
                                x_stu_22.append(x3)
                                y_stu_22.append(mvts.index(row[3]))
                                p_stu_22.append(id.index(row[4]))
                        if row[4] == '23':
                                x1=row[0]
                                x_stu_23.append(x1)
                                x2=row[1]
                                x_stu_23.append(x2)
                                x3= row[2]
                                x_stu_23.append(x3)
                                y_stu_23.append(mvts.index(row[3]))
                                p_stu_23.append(id.index(row[4]))
                        if row[4] == '24':
                                x1=row[0]
                                x_stu_24.append(x1)
                                x2=row[1]
                                x_stu_24.append(x2)
                                x3= row[2]
                                x_stu_24.append(x3)
                                y_stu_24.append(mvts.index(row[3]))
                                p_stu_24.append(id.index(row[4]))
                        if row[4] == '25':
                                x1=row[0]
                                x_stu_25.append(x1)
                                x2=row[1]
                                x_stu_25.append(x2)
                                x3= row[2]
                                x_stu_25.append(x3)
                                y_stu_25.append(mvts.index(row[3]))
                                p_stu_25.append(id.index(row[4]))
                        if row[4] == '26':
                                x1=row[0]
                                x_stu_26.append(x1)
                                x2=row[1]
                                x_stu_26.append(x2)
                                x3= row[2]
                                x_stu_26.append(x3)
                                y_stu_26.append(mvts.index(row[3]))
                                p_stu_26.append(id.index(row[4]))
                        if row[4] == '27':
                                x1=row[0]
                                x_stu_27.append(x1)
                                x2=row[1]
                                x_stu_27.append(x2)
                                x3= row[2]
                                x_stu_27.append(x3)
                                y_stu_27.append(mvts.index(row[3]))
                                p_stu_27.append(id.index(row[4]))
                        if row[4] == '28':
                                x1=row[0]
                                x_stu_28.append(x1)
                                x2=row[1]
                                x_stu_28.append(x2)
                                x3= row[2]
                                x_stu_28.append(x3)
                                y_stu_28.append(mvts.index(row[3]))
                                p_stu_28.append(id.index(row[4]))
                        if row[4] == '29':
                                x1=row[0]
                                x_stu_29.append(x1)
                                x2=row[1]
                                x_stu_29.append(x2)
                                x3= row[2]
                                x_stu_29.append(x3)
                                y_stu_29.append(mvts.index(row[3]))
                                p_stu_29.append(id.index(row[4]))
                        if row[4] == '30':
                                x1=row[0]
                                x_stu_30.append(x1)
                                x2=row[1]
                                x_stu_30.append(x2)
                                x3= row[2]
                                x_stu_30.append(x3)
                                y_stu_30.append(mvts.index(row[3]))
                                p_stu_30.append(id.index(row[4]))
                        if row[4] == '31':
                                x1=row[0]
                                x_stu_31.append(x1)
                                x2=row[1]
                                x_stu_31.append(x2)
                                x3= row[2]
                                x_stu_31.append(x3)
                                y_stu_31.append(mvts.index(row[3]))
                                p_stu_31.append(id.index(row[4]))
                        if row[4] == '32':
                                x1=row[0]
                                x_stu_32.append(x1)
                                x2=row[1]
                                x_stu_32.append(x2)
                                x3= row[2]
                                x_stu_32.append(x3)
                                y_stu_32.append(mvts.index(row[3]))
                                p_stu_32.append(id.index(row[4]))
                        if row[4] == '33':
                                x1=row[0]
                                x_stu_33.append(x1)
                                x2=row[1]
                                x_stu_33.append(x2)
                                x3= row[2]
                                x_stu_33.append(x3)
                                y_stu_33.append(mvts.index(row[3]))
                                p_stu_33.append(id.index(row[4]))
                        if row[4] == '34':
                                x1=row[0]
                                x_stu_34.append(x1)
                                x2=row[1]
                                x_stu_34.append(x2)
                                x3= row[2]
                                x_stu_34.append(x3)
                                y_stu_34.append(mvts.index(row[3]))
                                p_stu_34.append(id.index(row[4]))
                        if row[4] == '35':
                                x1=row[0]
                                x_stu_35.append(x1)
                                x2=row[1]
                                x_stu_35.append(x2)
                                x3= row[2]
                                x_stu_35.append(x3)
                                y_stu_35.append(mvts.index(row[3]))
                                p_stu_35.append(id.index(row[4]))
                        if row[4] == '36':
                                x1=row[0]
                                x_stu_36.append(x1)
                                x2=row[1]
                                x_stu_36.append(x2)
                                x3= row[2]
                                x_stu_36.append(x3)
                                y_stu_36.append(mvts.index(row[3]))
                                p_stu_36.append(id.index(row[4]))
                        if row[4] == '37':
                                x1=row[0]
                                x_stu_37.append(x1)
                                x2=row[1]
                                x_stu_37.append(x2)
                                x3= row[2]
                                x_stu_37.append(x3)
                                y_stu_37.append(mvts.index(row[3]))
                                p_stu_37.append(id.index(row[4]))
                        if row[4] == '38':
                                x1=row[0]
                                x_stu_38.append(x1)
                                x2=row[1]
                                x_stu_38.append(x2)
                                x3= row[2]
                                x_stu_38.append(x3)
                                y_stu_38.append(mvts.index(row[3]))
                                p_stu_38.append(id.index(row[4]))
                        if row[4] == '39':
                                x1=row[0]
                                x_stu_39.append(x1)
                                x2=row[1]
                                x_stu_39.append(x2)
                                x3= row[2]
                                x_stu_39.append(x3)
                                y_stu_39.append(mvts.index(row[3]))
                                p_stu_39.append(id.index(row[4]))
                        if row[4] == '40':
                                x1=row[0]
                                x_stu_40.append(x1)
                                x2=row[1]
                                x_stu_40.append(x2)
                                x3= row[2]
                                x_stu_40.append(x3)
                                y_stu_40.append(mvts.index(row[3]))
                                p_stu_40.append(id.index(row[4]))
                        if row[4] == '41':
                                x1=row[0]
                                x_stu_41.append(x1)
                                x2=row[1]
                                x_stu_41.append(x2)
                                x3= row[2]
                                x_stu_41.append(x3)
                                y_stu_41.append(mvts.index(row[3]))
                                p_stu_41.append(id.index(row[4]))
                        if row[4] == '42':
                                x1=row[0]
                                x_stu_42.append(x1)
                                x2=row[1]
                                x_stu_42.append(x2)
                                x3= row[2]
                                x_stu_42.append(x3)
                                y_stu_42.append(mvts.index(row[3]))
                                p_stu_42.append(id.index(row[4]))
                        if row[4] == '43':
                                x1=row[0]
                                x_stu_43.append(x1)
                                x2=row[1]
                                x_stu_43.append(x2)
                                x3= row[2]
                                x_stu_43.append(x3)
                                y_stu_43.append(mvts.index(row[3]))
                                p_stu_43.append(id.index(row[4]))
                        if row[4] == '44':
                                x1=row[0]
                                x_stu_44.append(x1)
                                x2=row[1]
                                x_stu_44.append(x2)
                                x3= row[2]
                                x_stu_44.append(x3)
                                y_stu_44.append(mvts.index(row[3]))
                                p_stu_44.append(id.index(row[4]))
                        if row[4] == '45':
                                x1=row[0]
                                x_stu_45.append(x1)
                                x2=row[1]
                                x_stu_45.append(x2)
                                x3= row[2]
                                x_stu_45.append(x3)
                                y_stu_45.append(mvts.index(row[3]))
                                p_stu_45.append(id.index(row[4]))
                        if row[4] == '46':
                                x1=row[0]
                                x_stu_46.append(x1)
                                x2=row[1]
                                x_stu_46.append(x2)
                                x3= row[2]
                                x_stu_46.append(x3)
                                y_stu_46.append(mvts.index(row[3]))
                                p_stu_46.append(id.index(row[4]))
                        if row[4] == '47':
                                x1=row[0]
                                x_stu_47.append(x1)
                                x2=row[1]
                                x_stu_47.append(x2)
                                x3= row[2]
                                x_stu_47.append(x3)
                                y_stu_47.append(mvts.index(row[3]))
                                p_stu_47.append(id.index(row[4]))
                        if row[4] == '48':
                                x1=row[0]
                                x_stu_48.append(x1)
                                x2=row[1]
                                x_stu_48.append(x2)
                                x3= row[2]
                                x_stu_48.append(x3)
                                y_stu_48.append(mvts.index(row[3]))
                                p_stu_48.append(id.index(row[4]))
                        if row[4] == '49':
                                x1=row[0]
                                x_stu_49.append(x1)
                                x2=row[1]
                                x_stu_49.append(x2)
                                x3= row[2]
                                x_stu_49.append(x3)
                                y_stu_49.append(mvts.index(row[3]))
                                p_stu_49.append(id.index(row[4]))
                        if row[4] == '50':
                                x1=row[0]
                                x_stu_50.append(x1)
                                x2=row[1]
                                x_stu_50.append(x2)
                                x3= row[2]
                                x_stu_50.append(x3)
                                y_stu_50.append(mvts.index(row[3]))
                                p_stu_50.append(id.index(row[4]))
                        if row[4] == '51':
                                x1=row[0]
                                x_stu_51.append(x1)
                                x2=row[1]
                                x_stu_51.append(x2)
                                x3= row[2]
                                x_stu_51.append(x3)
                                y_stu_51.append(mvts.index(row[3]))
                                p_stu_51.append(id.index(row[4]))
                        if row[4] == '52':
                                x1=row[0]
                                x_stu_52.append(x1)
                                x2=row[1]
                                x_stu_52.append(x2)
                                x3= row[2]
                                x_stu_52.append(x3)
                                y_stu_52.append(mvts.index(row[3]))
                                p_stu_52.append(id.index(row[4]))
                        if row[4] == '53':
                                x1=row[0]
                                x_stu_53.append(x1)
                                x2=row[1]
                                x_stu_53.append(x2)
                                x3= row[2]
                                x_stu_53.append(x3)
                                y_stu_53.append(mvts.index(row[3]))
                                p_stu_53.append(id.index(row[4]))
                        if row[4] == '54':
                                x1=row[0]
                                x_stu_54.append(x1)
                                x2=row[1]
                                x_stu_54.append(x2)
                                x3= row[2]
                                x_stu_54.append(x3)
                                y_stu_54.append(mvts.index(row[3]))
                                p_stu_54.append(id.index(row[4]))
                        if row[4] == '55':
                                x1=row[0]
                                x_stu_55.append(x1)
                                x2=row[1]
                                x_stu_55.append(x2)
                                x3= row[2]
                                x_stu_55.append(x3)
                                y_stu_55.append(mvts.index(row[3]))
                                p_stu_55.append(id.index(row[4]))
                        if row[4] == '56':
                                x1=row[0]
                                x_stu_56.append(x1)
                                x2=row[1]
                                x_stu_56.append(x2)
                                x3= row[2]
                                x_stu_56.append(x3)
                                y_stu_56.append(mvts.index(row[3]))
                                p_stu_56.append(id.index(row[4]))
                        if row[4] == '57':
                                x1=row[0]
                                x_stu_57.append(x1)
                                x2=row[1]
                                x_stu_57.append(x2)
                                x3= row[2]
                                x_stu_57.append(x3)
                                y_stu_57.append(mvts.index(row[3]))
                                p_stu_57.append(id.index(row[4]))
                        if row[4] == '58':
                                x1=row[0]
                                x_stu_58.append(x1)
                                x2=row[1]
                                x_stu_58.append(x2)
                                x3= row[2]
                                x_stu_58.append(x3)
                                y_stu_58.append(mvts.index(row[3]))
                                p_stu_58.append(id.index(row[4]))
                        if row[4] == '59':
                                x1=row[0]
                                x_stu_59.append(x1)
                                x2=row[1]
                                x_stu_59.append(x2)
                                x3= row[2]
                                x_stu_59.append(x3)
                                y_stu_59.append(mvts.index(row[3]))
                                p_stu_59.append(id.index(row[4]))
                        if row[4] == '60':
                                x1=row[0]
                                x_stu_60.append(x1)
                                x2=row[1]
                                x_stu_60.append(x2)
                                x3= row[2]
                                x_stu_60.append(x3)
                                y_stu_60.append(mvts.index(row[3]))
                                p_stu_60.append(id.index(row[4]))
                        if row[4] == '61':
                                x1=row[0]
                                x_stu_61.append(x1)
                                x2=row[1]
                                x_stu_61.append(x2)
                                x3= row[2]
                                x_stu_61.append(x3)
                                y_stu_61.append(mvts.index(row[3]))
                                p_stu_61.append(id.index(row[4]))
                        if row[4] == '62':
                                x1=row[0]
                                x_stu_62.append(x1)
                                x2=row[1]
                                x_stu_62.append(x2)
                                x3= row[2]
                                x_stu_62.append(x3)
                                y_stu_62.append(mvts.index(row[3]))
                                p_stu_62.append(id.index(row[4]))
                        if row[4] == '63':
                                x1=row[0]
                                x_stu_63.append(x1)
                                x2=row[1]
                                x_stu_63.append(x2)
                                x3= row[2]
                                x_stu_63.append(x3)
                                y_stu_63.append(mvts.index(row[3]))
                                p_stu_63.append(id.index(row[4]))
                        if row[4] == '64':
                                x1=row[0]
                                x_stu_64.append(x1)
                                x2=row[1]
                                x_stu_64.append(x2)
                                x3= row[2]
                                x_stu_64.append(x3)
                                y_stu_64.append(mvts.index(row[3]))
                                p_stu_64.append(id.index(row[4]))
                        if row[4] == '65':
                                x1=row[0]
                                x_stu_65.append(x1)
                                x2=row[1]
                                x_stu_65.append(x2)
                                x3= row[2]
                                x_stu_65.append(x3)
                                y_stu_65.append(mvts.index(row[3]))
                                p_stu_65.append(id.index(row[4]))
                        if row[4] == '66':
                                x1=row[0]
                                x_stu_66.append(x1)
                                x2=row[1]
                                x_stu_66.append(x2)
                                x3= row[2]
                                x_stu_66.append(x3)
                                y_stu_66.append(mvts.index(row[3]))
                                p_stu_66.append(id.index(row[4]))
                        if row[4] == '67':
                                x1=row[0]
                                x_stu_67.append(x1)
                                x2=row[1]
                                x_stu_67.append(x2)
                                x3= row[2]
                                x_stu_67.append(x3)
                                y_stu_67.append(mvts.index(row[3]))
                                p_stu_67.append(id.index(row[4]))
                        if row[4] == '68':
                                x1=row[0]
                                x_stu_68.append(x1)
                                x2=row[1]
                                x_stu_68.append(x2)
                                x3= row[2]
                                x_stu_68.append(x3)
                                y_stu_68.append(mvts.index(row[3]))
                                p_stu_68.append(id.index(row[4]))
                        if row[4] == '69':
                                x1=row[0]
                                x_stu_69.append(x1)
                                x2=row[1]
                                x_stu_69.append(x2)
                                x3= row[2]
                                x_stu_69.append(x3)
                                y_stu_69.append(mvts.index(row[3]))
                                p_stu_69.append(id.index(row[4]))
                        if row[4] == '70':
                                x1=row[0]
                                x_stu_70.append(x1)
                                x2=row[1]
                                x_stu_70.append(x2)
                                x3= row[2]
                                x_stu_70.append(x3)
                                y_stu_70.append(mvts.index(row[3]))
                                p_stu_70.append(id.index(row[4]))
                        if row[4] == '71':
                                x1=row[0]
                                x_stu_71.append(x1)
                                x2=row[1]
                                x_stu_71.append(x2)
                                x3= row[2]
                                x_stu_71.append(x3)
                                y_stu_71.append(mvts.index(row[3]))
                                p_stu_71.append(id.index(row[4]))
                        if row[4] == '72':
                                x1=row[0]
                                x_stu_72.append(x1)
                                x2=row[1]
                                x_stu_72.append(x2)
                                x3= row[2]
                                x_stu_72.append(x3)
                                y_stu_72.append(mvts.index(row[3]))
                                p_stu_72.append(id.index(row[4]))
                        if row[4] == '73':
                                x1=row[0]
                                x_stu_73.append(x1)
                                x2=row[1]
                                x_stu_73.append(x2)
                                x3= row[2]
                                x_stu_73.append(x3)
                                y_stu_73.append(mvts.index(row[3]))
                                p_stu_73.append(id.index(row[4]))
                        if row[4] == '74':
                                x1=row[0]
                                x_stu_74.append(x1)
                                x2=row[1]
                                x_stu_74.append(x2)
                                x3= row[2]
                                x_stu_74.append(x3)
                                y_stu_74.append(mvts.index(row[3]))
                                p_stu_74.append(id.index(row[4]))
                        if row[4] == '75':
                                x1=row[0]
                                x_stu_75.append(x1)
                                x2=row[1]
                                x_stu_75.append(x2)
                                x3= row[2]
                                x_stu_75.append(x3)
                                y_stu_75.append(mvts.index(row[3]))
                                p_stu_75.append(id.index(row[4]))
                        if row[4] == '76':
                                x1=row[0]
                                x_stu_76.append(x1)
                                x2=row[1]
                                x_stu_76.append(x2)
                                x3= row[2]
                                x_stu_76.append(x3)
                                y_stu_76.append(mvts.index(row[3]))
                                p_stu_76.append(id.index(row[4]))
                        if row[4] == '77':
                                x1=row[0]
                                x_stu_77.append(x1)
                                x2=row[1]
                                x_stu_77.append(x2)
                                x3= row[2]
                                x_stu_77.append(x3)
                                y_stu_77.append(mvts.index(row[3]))
                                p_stu_77.append(id.index(row[4]))
                        if row[4] == '78':
                                x1=row[0]
                                x_stu_78.append(x1)
                                x2=row[1]
                                x_stu_78.append(x2)
                                x3= row[2]
                                x_stu_78.append(x3)
                                y_stu_78.append(mvts.index(row[3]))
                                p_stu_78.append(id.index(row[4]))
                        if row[4] == '79':
                                x1=row[0]
                                x_stu_79.append(x1)
                                x2=row[1]
                                x_stu_79.append(x2)
                                x3= row[2]
                                x_stu_79.append(x3)
                                y_stu_79.append(mvts.index(row[3]))
                                p_stu_79.append(id.index(row[4]))
                        if row[4] == '80':
                                x1=row[0]
                                x_stu_80.append(x1)
                                x2=row[1]
                                x_stu_80.append(x2)
                                x3= row[2]
                                x_stu_80.append(x3)
                                y_stu_80.append(mvts.index(row[3]))
                                p_stu_80.append(id.index(row[4]))
                        if row[4] == '81':
                                x1=row[0]
                                x_stu_81.append(x1)
                                x2=row[1]
                                x_stu_81.append(x2)
                                x3= row[2]
                                x_stu_81.append(x3)
                                y_stu_81.append(mvts.index(row[3]))
                                p_stu_81.append(id.index(row[4]))
                        if row[4] == '82':
                                x1=row[0]
                                x_stu_82.append(x1)
                                x2=row[1]
                                x_stu_82.append(x2)
                                x3= row[2]
                                x_stu_82.append(x3)
                                y_stu_82.append(mvts.index(row[3]))
                                p_stu_82.append(id.index(row[4]))
                        if row[4] == '83':
                                x1=row[0]
                                x_stu_83.append(x1)
                                x2=row[1]
                                x_stu_83.append(x2)
                                x3= row[2]
                                x_stu_83.append(x3)
                                y_stu_83.append(mvts.index(row[3]))
                                p_stu_83.append(id.index(row[4]))
                        if row[4] == '84':
                                x1=row[0]
                                x_stu_84.append(x1)
                                x2=row[1]
                                x_stu_84.append(x2)
                                x3= row[2]
                                x_stu_84.append(x3)
                                y_stu_84.append(mvts.index(row[3]))
                                p_stu_84.append(id.index(row[4]))
                        if row[4] == '85':
                                x1=row[0]
                                x_stu_85.append(x1)
                                x2=row[1]
                                x_stu_85.append(x2)
                                x3= row[2]
                                x_stu_85.append(x3)
                                y_stu_85.append(mvts.index(row[3]))
                                p_stu_85.append(id.index(row[4]))
                        if row[4] == '86':
                                x1=row[0]
                                x_stu_86.append(x1)
                                x2=row[1]
                                x_stu_86.append(x2)
                                x3= row[2]
                                x_stu_86.append(x3)
                                y_stu_86.append(mvts.index(row[3]))
                                p_stu_86.append(id.index(row[4]))
                        if row[4] == '87':
                                x1=row[0]
                                x_stu_87.append(x1)
                                x2=row[1]
                                x_stu_87.append(x2)
                                x3= row[2]
                                x_stu_87.append(x3)
                                y_stu_87.append(mvts.index(row[3]))
                                p_stu_87.append(id.index(row[4]))
                        if row[4] == '88':
                                x1=row[0]
                                x_stu_88.append(x1)
                                x2=row[1]
                                x_stu_88.append(x2)
                                x3= row[2]
                                x_stu_88.append(x3)
                                y_stu_88.append(mvts.index(row[3]))
                                p_stu_88.append(id.index(row[4]))
                        if row[4] == '89':
                                x1=row[0]
                                x_stu_89.append(x1)
                                x2=row[1]
                                x_stu_89.append(x2)
                                x3= row[2]
                                x_stu_89.append(x3)
                                y_stu_89.append(mvts.index(row[3]))
                                p_stu_89.append(id.index(row[4]))
                        if row[4] == '90':
                                x1=row[0]
                                x_stu_90.append(x1)
                                x2=row[1]
                                x_stu_90.append(x2)
                                x3= row[2]
                                x_stu_90.append(x3)
                                y_stu_90.append(mvts.index(row[3]))
                                p_stu_90.append(id.index(row[4]))
                        if row[4] == '91':
                                x1=row[0]
                                x_stu_91.append(x1)
                                x2=row[1]
                                x_stu_91.append(x2)
                                x3= row[2]
                                x_stu_91.append(x3)
                                y_stu_91.append(mvts.index(row[3]))
                                p_stu_91.append(id.index(row[4]))
                        if row[4] == '92':
                                x1=row[0]
                                x_stu_92.append(x1)
                                x2=row[1]
                                x_stu_92.append(x2)
                                x3= row[2]
                                x_stu_92.append(x3)
                                y_stu_92.append(mvts.index(row[3]))
                                p_stu_92.append(id.index(row[4]))
                        if row[4] == '93':
                                x1=row[0]
                                x_stu_93.append(x1)
                                x2=row[1]
                                x_stu_93.append(x2)
                                x3= row[2]
                                x_stu_93.append(x3)
                                y_stu_93.append(mvts.index(row[3]))
                                p_stu_93.append(id.index(row[4]))
                        if row[4] == '94':
                                x1=row[0]
                                x_stu_94.append(x1)
                                x2=row[1]
                                x_stu_94.append(x2)
                                x3= row[2]
                                x_stu_94.append(x3)
                                y_stu_94.append(mvts.index(row[3]))
                                p_stu_94.append(id.index(row[4]))
                        if row[4] == '95':
                                x1=row[0]
                                x_stu_95.append(x1)
                                x2=row[1]
                                x_stu_95.append(x2)
                                x3= row[2]
                                x_stu_95.append(x3)
                                y_stu_95.append(mvts.index(row[3]))
                                p_stu_95.append(id.index(row[4]))
                        if row[4] == '96':
                                x1=row[0]
                                x_stu_96.append(x1)
                                x2=row[1]
                                x_stu_96.append(x2)
                                x3= row[2]
                                x_stu_96.append(x3)
                                y_stu_96.append(mvts.index(row[3]))
                                p_stu_96.append(id.index(row[4]))
                        if row[4] == '97':
                                x1=row[0]
                                x_stu_97.append(x1)
                                x2=row[1]
                                x_stu_97.append(x2)
                                x3= row[2]
                                x_stu_97.append(x3)
                                y_stu_97.append(mvts.index(row[3]))
                                p_stu_97.append(id.index(row[4]))
                        if row[4] == '98':
                                x1=row[0]
                                x_stu_98.append(x1)
                                x2=row[1]
                                x_stu_98.append(x2)
                                x3= row[2]
                                x_stu_98.append(x3)
                                y_stu_98.append(mvts.index(row[3]))
                                p_stu_98.append(id.index(row[4]))
                        if row[4] == '99':
                                x1=row[0]
                                x_stu_99.append(x1)
                                x2=row[1]
                                x_stu_99.append(x2)
                                x3= row[2]
                                x_stu_99.append(x3)
                                y_stu_99.append(mvts.index(row[3]))
                                p_stu_99.append(id.index(row[4]))
                if row[3]==' stDown':
                        if row[4] == '00':
                                x1=row[0]
                                x_std_0.append(x1)
                                x2=row[1]
                                x_std_0.append(x2)
                                x3= row[2]
                                x_std_0.append(x3)
                                y_std_0.append(mvts.index(row[3]))
                                p_std_0.append(id.index(row[4]))
                        if row[4] == '01':
                                x1=row[0]
                                x_std_1.append(x1)
                                x2=row[1]
                                x_std_1.append(x2)
                                x3= row[2]
                                x_std_1.append(x3)
                                y_std_1.append(mvts.index(row[3]))
                                p_std_1.append(id.index(row[4]))
                        if row[4] == '02':
                                x1=row[0]
                                x_std_2.append(x1)
                                x2=row[1]
                                x_std_2.append(x2)
                                x3= row[2]
                                x_std_2.append(x3)
                                y_std_2.append(mvts.index(row[3]))
                                p_std_2.append(id.index(row[4]))
                        if row[4] == '03':
                                x1=row[0]
                                x_std_3.append(x1)
                                x2=row[1]
                                x_std_3.append(x2)
                                x3= row[2]
                                x_std_3.append(x3)
                                y_std_3.append(mvts.index(row[3]))
                                p_std_3.append(id.index(row[4]))
                        if row[4] == '04':
                                x1=row[0]
                                x_std_4.append(x1)
                                x2=row[1]
                                x_std_4.append(x2)
                                x3= row[2]
                                x_std_4.append(x3)
                                y_std_4.append(mvts.index(row[3]))
                                p_std_4.append(id.index(row[4]))
                        if row[4] == '05':
                                x1=row[0]
                                x_std_5.append(x1)
                                x2=row[1]
                                x_std_5.append(x2)
                                x3= row[2]
                                x_std_5.append(x3)
                                y_std_5.append(mvts.index(row[3]))
                                p_std_5.append(id.index(row[4]))
                        if row[4] == '06':
                                x1=row[0]
                                x_std_6.append(x1)
                                x2=row[1]
                                x_std_6.append(x2)
                                x3= row[2]
                                x_std_6.append(x3)
                                y_std_6.append(mvts.index(row[3]))
                                p_std_6.append(id.index(row[4]))
                        if row[4] == '07':
                                x1=row[0]
                                x_std_7.append(x1)
                                x2=row[1]
                                x_std_7.append(x2)
                                x3= row[2]
                                x_std_7.append(x3)
                                y_std_7.append(mvts.index(row[3]))
                                p_std_7.append(id.index(row[4]))
                        if row[4] == '08':
                                x1=row[0]
                                x_std_8.append(x1)
                                x2=row[1]
                                x_std_8.append(x2)
                                x3= row[2]
                                x_std_8.append(x3)
                                y_std_8.append(mvts.index(row[3]))
                                p_std_8.append(id.index(row[4]))
                        if row[4] == '09':
                                x1=row[0]
                                x_std_9.append(x1)
                                x2=row[1]
                                x_std_9.append(x2)
                                x3= row[2]
                                x_std_9.append(x3)
                                y_std_9.append(mvts.index(row[3]))
                                p_std_9.append(id.index(row[4]))
                        if row[4] == '10':
                                x1=row[0]
                                x_std_10.append(x1)
                                x2=row[1]
                                x_std_10.append(x2)
                                x3= row[2]
                                x_std_10.append(x3)
                                y_std_10.append(mvts.index(row[3]))
                                p_std_10.append(id.index(row[4]))
                        if row[4] == '11':
                                x1=row[0]
                                x_std_11.append(x1)
                                x2=row[1]
                                x_std_11.append(x2)
                                x3= row[2]
                                x_std_11.append(x3)
                                y_std_11.append(mvts.index(row[3]))
                                p_std_11.append(id.index(row[4]))
                        if row[4] == '12':
                                x1=row[0]
                                x_std_12.append(x1)
                                x2=row[1]
                                x_std_12.append(x2)
                                x3= row[2]
                                x_std_12.append(x3)
                                y_std_12.append(mvts.index(row[3]))
                                p_std_12.append(id.index(row[4]))
                        if row[4] == '13':
                                x1=row[0]
                                x_std_13.append(x1)
                                x2=row[1]
                                x_std_13.append(x2)
                                x3= row[2]
                                x_std_13.append(x3)
                                y_std_13.append(mvts.index(row[3]))
                                p_std_13.append(id.index(row[4]))
                        if row[4] == '14':
                                x1=row[0]
                                x_std_14.append(x1)
                                x2=row[1]
                                x_std_14.append(x2)
                                x3= row[2]
                                x_std_14.append(x3)
                                y_std_14.append(mvts.index(row[3]))
                                p_std_14.append(id.index(row[4]))
                        if row[4] == '15':
                                x1=row[0]
                                x_std_15.append(x1)
                                x2=row[1]
                                x_std_15.append(x2)
                                x3= row[2]
                                x_std_15.append(x3)
                                y_std_15.append(mvts.index(row[3]))
                                p_std_15.append(id.index(row[4]))
                        if row[4] == '16':
                                x1=row[0]
                                x_std_16.append(x1)
                                x2=row[1]
                                x_std_16.append(x2)
                                x3= row[2]
                                x_std_16.append(x3)
                                y_std_16.append(mvts.index(row[3]))
                                p_std_16.append(id.index(row[4]))
                        if row[4] == '17':
                                x1=row[0]
                                x_std_17.append(x1)
                                x2=row[1]
                                x_std_17.append(x2)
                                x3= row[2]
                                x_std_17.append(x3)
                                y_std_17.append(mvts.index(row[3]))
                                p_std_17.append(id.index(row[4]))
                        if row[4] == '18':
                                x1=row[0]
                                x_std_18.append(x1)
                                x2=row[1]
                                x_std_18.append(x2)
                                x3= row[2]
                                x_std_18.append(x3)
                                y_std_18.append(mvts.index(row[3]))
                                p_std_18.append(id.index(row[4]))
                        if row[4] == '19':
                                x1=row[0]
                                x_std_19.append(x1)
                                x2=row[1]
                                x_std_19.append(x2)
                                x3= row[2]
                                x_std_19.append(x3)
                                y_std_19.append(mvts.index(row[3]))
                                p_std_19.append(id.index(row[4]))
                        if row[4] == '20':
                                x1=row[0]
                                x_std_20.append(x1)
                                x2=row[1]
                                x_std_20.append(x2)
                                x3= row[2]
                                x_std_20.append(x3)
                                y_std_20.append(mvts.index(row[3]))
                                p_std_20.append(id.index(row[4]))
                        if row[4] == '21':
                                x1=row[0]
                                x_std_21.append(x1)
                                x2=row[1]
                                x_std_21.append(x2)
                                x3= row[2]
                                x_std_21.append(x3)
                                y_std_21.append(mvts.index(row[3]))
                                p_std_21.append(id.index(row[4]))
                        if row[4] == '22':
                                x1=row[0]
                                x_std_22.append(x1)
                                x2=row[1]
                                x_std_22.append(x2)
                                x3= row[2]
                                x_std_22.append(x3)
                                y_std_22.append(mvts.index(row[3]))
                                p_std_22.append(id.index(row[4]))
                        if row[4] == '23':
                                x1=row[0]
                                x_std_23.append(x1)
                                x2=row[1]
                                x_std_23.append(x2)
                                x3= row[2]
                                x_std_23.append(x3)
                                y_std_23.append(mvts.index(row[3]))
                                p_std_23.append(id.index(row[4]))
                        if row[4] == '24':
                                x1=row[0]
                                x_std_24.append(x1)
                                x2=row[1]
                                x_std_24.append(x2)
                                x3= row[2]
                                x_std_24.append(x3)
                                y_std_24.append(mvts.index(row[3]))
                                p_std_24.append(id.index(row[4]))
                        if row[4] == '25':
                                x1=row[0]
                                x_std_25.append(x1)
                                x2=row[1]
                                x_std_25.append(x2)
                                x3= row[2]
                                x_std_25.append(x3)
                                y_std_25.append(mvts.index(row[3]))
                                p_std_25.append(id.index(row[4]))
                        if row[4] == '26':
                                x1=row[0]
                                x_std_26.append(x1)
                                x2=row[1]
                                x_std_26.append(x2)
                                x3= row[2]
                                x_std_26.append(x3)
                                y_std_26.append(mvts.index(row[3]))
                                p_std_26.append(id.index(row[4]))
                        if row[4] == '27':
                                x1=row[0]
                                x_std_27.append(x1)
                                x2=row[1]
                                x_std_27.append(x2)
                                x3= row[2]
                                x_std_27.append(x3)
                                y_std_27.append(mvts.index(row[3]))
                                p_std_27.append(id.index(row[4]))
                        if row[4] == '28':
                                x1=row[0]
                                x_std_28.append(x1)
                                x2=row[1]
                                x_std_28.append(x2)
                                x3= row[2]
                                x_std_28.append(x3)
                                y_std_28.append(mvts.index(row[3]))
                                p_std_28.append(id.index(row[4]))
                        if row[4] == '29':
                                x1=row[0]
                                x_std_29.append(x1)
                                x2=row[1]
                                x_std_29.append(x2)
                                x3= row[2]
                                x_std_29.append(x3)
                                y_std_29.append(mvts.index(row[3]))
                                p_std_29.append(id.index(row[4]))
                        if row[4] == '30':
                                x1=row[0]
                                x_std_30.append(x1)
                                x2=row[1]
                                x_std_30.append(x2)
                                x3= row[2]
                                x_std_30.append(x3)
                                y_std_30.append(mvts.index(row[3]))
                                p_std_30.append(id.index(row[4]))
                        if row[4] == '31':
                                x1=row[0]
                                x_std_31.append(x1)
                                x2=row[1]
                                x_std_31.append(x2)
                                x3= row[2]
                                x_std_31.append(x3)
                                y_std_31.append(mvts.index(row[3]))
                                p_std_31.append(id.index(row[4]))
                        if row[4] == '32':
                                x1=row[0]
                                x_std_32.append(x1)
                                x2=row[1]
                                x_std_32.append(x2)
                                x3= row[2]
                                x_std_32.append(x3)
                                y_std_32.append(mvts.index(row[3]))
                                p_std_32.append(id.index(row[4]))
                        if row[4] == '33':
                                x1=row[0]
                                x_std_33.append(x1)
                                x2=row[1]
                                x_std_33.append(x2)
                                x3= row[2]
                                x_std_33.append(x3)
                                y_std_33.append(mvts.index(row[3]))
                                p_std_33.append(id.index(row[4]))
                        if row[4] == '34':
                                x1=row[0]
                                x_std_34.append(x1)
                                x2=row[1]
                                x_std_34.append(x2)
                                x3= row[2]
                                x_std_34.append(x3)
                                y_std_34.append(mvts.index(row[3]))
                                p_std_34.append(id.index(row[4]))
                        if row[4] == '35':
                                x1=row[0]
                                x_std_35.append(x1)
                                x2=row[1]
                                x_std_35.append(x2)
                                x3= row[2]
                                x_std_35.append(x3)
                                y_std_35.append(mvts.index(row[3]))
                                p_std_35.append(id.index(row[4]))
                        if row[4] == '36':
                                x1=row[0]
                                x_std_36.append(x1)
                                x2=row[1]
                                x_std_36.append(x2)
                                x3= row[2]
                                x_std_36.append(x3)
                                y_std_36.append(mvts.index(row[3]))
                                p_std_36.append(id.index(row[4]))
                        if row[4] == '37':
                                x1=row[0]
                                x_std_37.append(x1)
                                x2=row[1]
                                x_std_37.append(x2)
                                x3= row[2]
                                x_std_37.append(x3)
                                y_std_37.append(mvts.index(row[3]))
                                p_std_37.append(id.index(row[4]))
                        if row[4] == '38':
                                x1=row[0]
                                x_std_38.append(x1)
                                x2=row[1]
                                x_std_38.append(x2)
                                x3= row[2]
                                x_std_38.append(x3)
                                y_std_38.append(mvts.index(row[3]))
                                p_std_38.append(id.index(row[4]))
                        if row[4] == '39':
                                x1=row[0]
                                x_std_39.append(x1)
                                x2=row[1]
                                x_std_39.append(x2)
                                x3= row[2]
                                x_std_39.append(x3)
                                y_std_39.append(mvts.index(row[3]))
                                p_std_39.append(id.index(row[4]))
                        if row[4] == '40':
                                x1=row[0]
                                x_std_40.append(x1)
                                x2=row[1]
                                x_std_40.append(x2)
                                x3= row[2]
                                x_std_40.append(x3)
                                y_std_40.append(mvts.index(row[3]))
                                p_std_40.append(id.index(row[4]))
                        if row[4] == '41':
                                x1=row[0]
                                x_std_41.append(x1)
                                x2=row[1]
                                x_std_41.append(x2)
                                x3= row[2]
                                x_std_41.append(x3)
                                y_std_41.append(mvts.index(row[3]))
                                p_std_41.append(id.index(row[4]))
                        if row[4] == '42':
                                x1=row[0]
                                x_std_42.append(x1)
                                x2=row[1]
                                x_std_42.append(x2)
                                x3= row[2]
                                x_std_42.append(x3)
                                y_std_42.append(mvts.index(row[3]))
                                p_std_42.append(id.index(row[4]))
                        if row[4] == '43':
                                x1=row[0]
                                x_std_43.append(x1)
                                x2=row[1]
                                x_std_43.append(x2)
                                x3= row[2]
                                x_std_43.append(x3)
                                y_std_43.append(mvts.index(row[3]))
                                p_std_43.append(id.index(row[4]))
                        if row[4] == '44':
                                x1=row[0]
                                x_std_44.append(x1)
                                x2=row[1]
                                x_std_44.append(x2)
                                x3= row[2]
                                x_std_44.append(x3)
                                y_std_44.append(mvts.index(row[3]))
                                p_std_44.append(id.index(row[4]))
                        if row[4] == '45':
                                x1=row[0]
                                x_std_45.append(x1)
                                x2=row[1]
                                x_std_45.append(x2)
                                x3= row[2]
                                x_std_45.append(x3)
                                y_std_45.append(mvts.index(row[3]))
                                p_std_45.append(id.index(row[4]))
                        if row[4] == '46':
                                x1=row[0]
                                x_std_46.append(x1)
                                x2=row[1]
                                x_std_46.append(x2)
                                x3= row[2]
                                x_std_46.append(x3)
                                y_std_46.append(mvts.index(row[3]))
                                p_std_46.append(id.index(row[4]))
                        if row[4] == '47':
                                x1=row[0]
                                x_std_47.append(x1)
                                x2=row[1]
                                x_std_47.append(x2)
                                x3= row[2]
                                x_std_47.append(x3)
                                y_std_47.append(mvts.index(row[3]))
                                p_std_47.append(id.index(row[4]))
                        if row[4] == '48':
                                x1=row[0]
                                x_std_48.append(x1)
                                x2=row[1]
                                x_std_48.append(x2)
                                x3= row[2]
                                x_std_48.append(x3)
                                y_std_48.append(mvts.index(row[3]))
                                p_std_48.append(id.index(row[4]))
                        if row[4] == '49':
                                x1=row[0]
                                x_std_49.append(x1)
                                x2=row[1]
                                x_std_49.append(x2)
                                x3= row[2]
                                x_std_49.append(x3)
                                y_std_49.append(mvts.index(row[3]))
                                p_std_49.append(id.index(row[4]))
                        if row[4] == '50':
                                x1=row[0]
                                x_std_50.append(x1)
                                x2=row[1]
                                x_std_50.append(x2)
                                x3= row[2]
                                x_std_50.append(x3)
                                y_std_50.append(mvts.index(row[3]))
                                p_std_50.append(id.index(row[4]))
                        if row[4] == '51':
                                x1=row[0]
                                x_std_51.append(x1)
                                x2=row[1]
                                x_std_51.append(x2)
                                x3= row[2]
                                x_std_51.append(x3)
                                y_std_51.append(mvts.index(row[3]))
                                p_std_51.append(id.index(row[4]))
                        if row[4] == '52':
                                x1=row[0]
                                x_std_52.append(x1)
                                x2=row[1]
                                x_std_52.append(x2)
                                x3= row[2]
                                x_std_52.append(x3)
                                y_std_52.append(mvts.index(row[3]))
                                p_std_52.append(id.index(row[4]))
                        if row[4] == '53':
                                x1=row[0]
                                x_std_53.append(x1)
                                x2=row[1]
                                x_std_53.append(x2)
                                x3= row[2]
                                x_std_53.append(x3)
                                y_std_53.append(mvts.index(row[3]))
                                p_std_53.append(id.index(row[4]))
                        if row[4] == '54':
                                x1=row[0]
                                x_std_54.append(x1)
                                x2=row[1]
                                x_std_54.append(x2)
                                x3= row[2]
                                x_std_54.append(x3)
                                y_std_54.append(mvts.index(row[3]))
                                p_std_54.append(id.index(row[4]))
                        if row[4] == '55':
                                x1=row[0]
                                x_std_55.append(x1)
                                x2=row[1]
                                x_std_55.append(x2)
                                x3= row[2]
                                x_std_55.append(x3)
                                y_std_55.append(mvts.index(row[3]))
                                p_std_55.append(id.index(row[4]))
                        if row[4] == '56':
                                x1=row[0]
                                x_std_56.append(x1)
                                x2=row[1]
                                x_std_56.append(x2)
                                x3= row[2]
                                x_std_56.append(x3)
                                y_std_56.append(mvts.index(row[3]))
                                p_std_56.append(id.index(row[4]))
                        if row[4] == '57':
                                x1=row[0]
                                x_std_57.append(x1)
                                x2=row[1]
                                x_std_57.append(x2)
                                x3= row[2]
                                x_std_57.append(x3)
                                y_std_57.append(mvts.index(row[3]))
                                p_std_57.append(id.index(row[4]))
                        if row[4] == '58':
                                x1=row[0]
                                x_std_58.append(x1)
                                x2=row[1]
                                x_std_58.append(x2)
                                x3= row[2]
                                x_std_58.append(x3)
                                y_std_58.append(mvts.index(row[3]))
                                p_std_58.append(id.index(row[4]))
                        if row[4] == '59':
                                x1=row[0]
                                x_std_59.append(x1)
                                x2=row[1]
                                x_std_59.append(x2)
                                x3= row[2]
                                x_std_59.append(x3)
                                y_std_59.append(mvts.index(row[3]))
                                p_std_59.append(id.index(row[4]))
                        if row[4] == '60':
                                x1=row[0]
                                x_std_60.append(x1)
                                x2=row[1]
                                x_std_60.append(x2)
                                x3= row[2]
                                x_std_60.append(x3)
                                y_std_60.append(mvts.index(row[3]))
                                p_std_60.append(id.index(row[4]))
                        if row[4] == '61':
                                x1=row[0]
                                x_std_61.append(x1)
                                x2=row[1]
                                x_std_61.append(x2)
                                x3= row[2]
                                x_std_61.append(x3)
                                y_std_61.append(mvts.index(row[3]))
                                p_std_61.append(id.index(row[4]))
                        if row[4] == '62':
                                x1=row[0]
                                x_std_62.append(x1)
                                x2=row[1]
                                x_std_62.append(x2)
                                x3= row[2]
                                x_std_62.append(x3)
                                y_std_62.append(mvts.index(row[3]))
                                p_std_62.append(id.index(row[4]))
                        if row[4] == '63':
                                x1=row[0]
                                x_std_63.append(x1)
                                x2=row[1]
                                x_std_63.append(x2)
                                x3= row[2]
                                x_std_63.append(x3)
                                y_std_63.append(mvts.index(row[3]))
                                p_std_63.append(id.index(row[4]))
                        if row[4] == '64':
                                x1=row[0]
                                x_std_64.append(x1)
                                x2=row[1]
                                x_std_64.append(x2)
                                x3= row[2]
                                x_std_64.append(x3)
                                y_std_64.append(mvts.index(row[3]))
                                p_std_64.append(id.index(row[4]))
                        if row[4] == '65':
                                x1=row[0]
                                x_std_65.append(x1)
                                x2=row[1]
                                x_std_65.append(x2)
                                x3= row[2]
                                x_std_65.append(x3)
                                y_std_65.append(mvts.index(row[3]))
                                p_std_65.append(id.index(row[4]))
                        if row[4] == '66':
                                x1=row[0]
                                x_std_66.append(x1)
                                x2=row[1]
                                x_std_66.append(x2)
                                x3= row[2]
                                x_std_66.append(x3)
                                y_std_66.append(mvts.index(row[3]))
                                p_std_66.append(id.index(row[4]))
                        if row[4] == '67':
                                x1=row[0]
                                x_std_67.append(x1)
                                x2=row[1]
                                x_std_67.append(x2)
                                x3= row[2]
                                x_std_67.append(x3)
                                y_std_67.append(mvts.index(row[3]))
                                p_std_67.append(id.index(row[4]))
                        if row[4] == '68':
                                x1=row[0]
                                x_std_68.append(x1)
                                x2=row[1]
                                x_std_68.append(x2)
                                x3= row[2]
                                x_std_68.append(x3)
                                y_std_68.append(mvts.index(row[3]))
                                p_std_68.append(id.index(row[4]))
                        if row[4] == '69':
                                x1=row[0]
                                x_std_69.append(x1)
                                x2=row[1]
                                x_std_69.append(x2)
                                x3= row[2]
                                x_std_69.append(x3)
                                y_std_69.append(mvts.index(row[3]))
                                p_std_69.append(id.index(row[4]))
                        if row[4] == '70':
                                x1=row[0]
                                x_std_70.append(x1)
                                x2=row[1]
                                x_std_70.append(x2)
                                x3= row[2]
                                x_std_70.append(x3)
                                y_std_70.append(mvts.index(row[3]))
                                p_std_70.append(id.index(row[4]))
                        if row[4] == '71':
                                x1=row[0]
                                x_std_71.append(x1)
                                x2=row[1]
                                x_std_71.append(x2)
                                x3= row[2]
                                x_std_71.append(x3)
                                y_std_71.append(mvts.index(row[3]))
                                p_std_71.append(id.index(row[4]))
                        if row[4] == '72':
                                x1=row[0]
                                x_std_72.append(x1)
                                x2=row[1]
                                x_std_72.append(x2)
                                x3= row[2]
                                x_std_72.append(x3)
                                y_std_72.append(mvts.index(row[3]))
                                p_std_72.append(id.index(row[4]))
                        if row[4] == '73':
                                x1=row[0]
                                x_std_73.append(x1)
                                x2=row[1]
                                x_std_73.append(x2)
                                x3= row[2]
                                x_std_73.append(x3)
                                y_std_73.append(mvts.index(row[3]))
                                p_std_73.append(id.index(row[4]))
                        if row[4] == '74':
                                x1=row[0]
                                x_std_74.append(x1)
                                x2=row[1]
                                x_std_74.append(x2)
                                x3= row[2]
                                x_std_74.append(x3)
                                y_std_74.append(mvts.index(row[3]))
                                p_std_74.append(id.index(row[4]))
                        if row[4] == '75':
                                x1=row[0]
                                x_std_75.append(x1)
                                x2=row[1]
                                x_std_75.append(x2)
                                x3= row[2]
                                x_std_75.append(x3)
                                y_std_75.append(mvts.index(row[3]))
                                p_std_75.append(id.index(row[4]))
                        if row[4] == '76':
                                x1=row[0]
                                x_std_76.append(x1)
                                x2=row[1]
                                x_std_76.append(x2)
                                x3= row[2]
                                x_std_76.append(x3)
                                y_std_76.append(mvts.index(row[3]))
                                p_std_76.append(id.index(row[4]))
                        if row[4] == '77':
                                x1=row[0]
                                x_std_77.append(x1)
                                x2=row[1]
                                x_std_77.append(x2)
                                x3= row[2]
                                x_std_77.append(x3)
                                y_std_77.append(mvts.index(row[3]))
                                p_std_77.append(id.index(row[4]))
                        if row[4] == '78':
                                x1=row[0]
                                x_std_78.append(x1)
                                x2=row[1]
                                x_std_78.append(x2)
                                x3= row[2]
                                x_std_78.append(x3)
                                y_std_78.append(mvts.index(row[3]))
                                p_std_78.append(id.index(row[4]))
                        if row[4] == '79':
                                x1=row[0]
                                x_std_79.append(x1)
                                x2=row[1]
                                x_std_79.append(x2)
                                x3= row[2]
                                x_std_79.append(x3)
                                y_std_79.append(mvts.index(row[3]))
                                p_std_79.append(id.index(row[4]))
                        if row[4] == '80':
                                x1=row[0]
                                x_std_80.append(x1)
                                x2=row[1]
                                x_std_80.append(x2)
                                x3= row[2]
                                x_std_80.append(x3)
                                y_std_80.append(mvts.index(row[3]))
                                p_std_80.append(id.index(row[4]))
                        if row[4] == '81':
                                x1=row[0]
                                x_std_81.append(x1)
                                x2=row[1]
                                x_std_81.append(x2)
                                x3= row[2]
                                x_std_81.append(x3)
                                y_std_81.append(mvts.index(row[3]))
                                p_std_81.append(id.index(row[4]))
                        if row[4] == '82':
                                x1=row[0]
                                x_std_82.append(x1)
                                x2=row[1]
                                x_std_82.append(x2)
                                x3= row[2]
                                x_std_82.append(x3)
                                y_std_82.append(mvts.index(row[3]))
                                p_std_82.append(id.index(row[4]))
                        if row[4] == '83':
                                x1=row[0]
                                x_std_83.append(x1)
                                x2=row[1]
                                x_std_83.append(x2)
                                x3= row[2]
                                x_std_83.append(x3)
                                y_std_83.append(mvts.index(row[3]))
                                p_std_83.append(id.index(row[4]))
                        if row[4] == '84':
                                x1=row[0]
                                x_std_84.append(x1)
                                x2=row[1]
                                x_std_84.append(x2)
                                x3= row[2]
                                x_std_84.append(x3)
                                y_std_84.append(mvts.index(row[3]))
                                p_std_84.append(id.index(row[4]))
                        if row[4] == '85':
                                x1=row[0]
                                x_std_85.append(x1)
                                x2=row[1]
                                x_std_85.append(x2)
                                x3= row[2]
                                x_std_85.append(x3)
                                y_std_85.append(mvts.index(row[3]))
                                p_std_85.append(id.index(row[4]))
                        if row[4] == '86':
                                x1=row[0]
                                x_std_86.append(x1)
                                x2=row[1]
                                x_std_86.append(x2)
                                x3= row[2]
                                x_std_86.append(x3)
                                y_std_86.append(mvts.index(row[3]))
                                p_std_86.append(id.index(row[4]))
                        if row[4] == '87':
                                x1=row[0]
                                x_std_87.append(x1)
                                x2=row[1]
                                x_std_87.append(x2)
                                x3= row[2]
                                x_std_87.append(x3)
                                y_std_87.append(mvts.index(row[3]))
                                p_std_87.append(id.index(row[4]))
                        if row[4] == '88':
                                x1=row[0]
                                x_std_88.append(x1)
                                x2=row[1]
                                x_std_88.append(x2)
                                x3= row[2]
                                x_std_88.append(x3)
                                y_std_88.append(mvts.index(row[3]))
                                p_std_88.append(id.index(row[4]))
                        if row[4] == '89':
                                x1=row[0]
                                x_std_89.append(x1)
                                x2=row[1]
                                x_std_89.append(x2)
                                x3= row[2]
                                x_std_89.append(x3)
                                y_std_89.append(mvts.index(row[3]))
                                p_std_89.append(id.index(row[4]))
                        if row[4] == '90':
                                x1=row[0]
                                x_std_90.append(x1)
                                x2=row[1]
                                x_std_90.append(x2)
                                x3= row[2]
                                x_std_90.append(x3)
                                y_std_90.append(mvts.index(row[3]))
                                p_std_90.append(id.index(row[4]))
                        if row[4] == '91':
                                x1=row[0]
                                x_std_91.append(x1)
                                x2=row[1]
                                x_std_91.append(x2)
                                x3= row[2]
                                x_std_91.append(x3)
                                y_std_91.append(mvts.index(row[3]))
                                p_std_91.append(id.index(row[4]))
                        if row[4] == '92':
                                x1=row[0]
                                x_std_92.append(x1)
                                x2=row[1]
                                x_std_92.append(x2)
                                x3= row[2]
                                x_std_92.append(x3)
                                y_std_92.append(mvts.index(row[3]))
                                p_std_92.append(id.index(row[4]))
                        if row[4] == '93':
                                x1=row[0]
                                x_std_93.append(x1)
                                x2=row[1]
                                x_std_93.append(x2)
                                x3= row[2]
                                x_std_93.append(x3)
                                y_std_93.append(mvts.index(row[3]))
                                p_std_93.append(id.index(row[4]))
                        if row[4] == '94':
                                x1=row[0]
                                x_std_94.append(x1)
                                x2=row[1]
                                x_std_94.append(x2)
                                x3= row[2]
                                x_std_94.append(x3)
                                y_std_94.append(mvts.index(row[3]))
                                p_std_94.append(id.index(row[4]))
                        if row[4] == '95':
                                x1=row[0]
                                x_std_95.append(x1)
                                x2=row[1]
                                x_std_95.append(x2)
                                x3= row[2]
                                x_std_95.append(x3)
                                y_std_95.append(mvts.index(row[3]))
                                p_std_95.append(id.index(row[4]))
                        if row[4] == '96':
                                x1=row[0]
                                x_std_96.append(x1)
                                x2=row[1]
                                x_std_96.append(x2)
                                x3= row[2]
                                x_std_96.append(x3)
                                y_std_96.append(mvts.index(row[3]))
                                p_std_96.append(id.index(row[4]))
                        if row[4] == '97':
                                x1=row[0]
                                x_std_97.append(x1)
                                x2=row[1]
                                x_std_97.append(x2)
                                x3= row[2]
                                x_std_97.append(x3)
                                y_std_97.append(mvts.index(row[3]))
                                p_std_97.append(id.index(row[4]))
                        if row[4] == '98':
                                x1=row[0]
                                x_std_98.append(x1)
                                x2=row[1]
                                x_std_98.append(x2)
                                x3= row[2]
                                x_std_98.append(x3)
                                y_std_98.append(mvts.index(row[3]))
                                p_std_98.append(id.index(row[4]))
                        if row[4] == '99':
                                x1=row[0]
                                x_std_99.append(x1)
                                x2=row[1]
                                x_std_99.append(x2)
                                x3= row[2]
                                x_std_99.append(x3)
                                y_std_99.append(mvts.index(row[3]))
                                p_std_99.append(id.index(row[4]))
                if row[3]==' stay':
                        if row[4] == '00':
                                x1=row[0]
                                x_st_0.append(x1)
                                x2=row[1]
                                x_st_0.append(x2)
                                x3= row[2]
                                x_st_0.append(x3)
                                y_st_0.append(mvts.index(row[3]))
                                p_st_0.append(id.index(row[4]))
                        if row[4] == '01':
                                x1=row[0]
                                x_st_1.append(x1)
                                x2=row[1]
                                x_st_1.append(x2)
                                x3= row[2]
                                x_st_1.append(x3)
                                y_st_1.append(mvts.index(row[3]))
                                p_st_1.append(id.index(row[4]))
                        if row[4] == '02':
                                x1=row[0]
                                x_st_2.append(x1)
                                x2=row[1]
                                x_st_2.append(x2)
                                x3= row[2]
                                x_st_2.append(x3)
                                y_st_2.append(mvts.index(row[3]))
                                p_st_2.append(id.index(row[4]))
                        if row[4] == '03':
                                x1=row[0]
                                x_st_3.append(x1)
                                x2=row[1]
                                x_st_3.append(x2)
                                x3= row[2]
                                x_st_3.append(x3)
                                y_st_3.append(mvts.index(row[3]))
                                p_st_3.append(id.index(row[4]))
                        if row[4] == '04':
                                x1=row[0]
                                x_st_4.append(x1)
                                x2=row[1]
                                x_st_4.append(x2)
                                x3= row[2]
                                x_st_4.append(x3)
                                y_st_4.append(mvts.index(row[3]))
                                p_st_4.append(id.index(row[4]))
                        if row[4] == '05':
                                x1=row[0]
                                x_st_5.append(x1)
                                x2=row[1]
                                x_st_5.append(x2)
                                x3= row[2]
                                x_st_5.append(x3)
                                y_st_5.append(mvts.index(row[3]))
                                p_st_5.append(id.index(row[4]))
                        if row[4] == '06':
                                x1=row[0]
                                x_st_6.append(x1)
                                x2=row[1]
                                x_st_6.append(x2)
                                x3= row[2]
                                x_st_6.append(x3)
                                y_st_6.append(mvts.index(row[3]))
                                p_st_6.append(id.index(row[4]))
                        if row[4] == '07':
                                x1=row[0]
                                x_st_7.append(x1)
                                x2=row[1]
                                x_st_7.append(x2)
                                x3= row[2]
                                x_st_7.append(x3)
                                y_st_7.append(mvts.index(row[3]))
                                p_st_7.append(id.index(row[4]))
                        if row[4] == '08':
                                x1=row[0]
                                x_st_8.append(x1)
                                x2=row[1]
                                x_st_8.append(x2)
                                x3= row[2]
                                x_st_8.append(x3)
                                y_st_8.append(mvts.index(row[3]))
                                p_st_8.append(id.index(row[4]))
                        if row[4] == '09':
                                x1=row[0]
                                x_st_9.append(x1)
                                x2=row[1]
                                x_st_9.append(x2)
                                x3= row[2]
                                x_st_9.append(x3)
                                y_st_9.append(mvts.index(row[3]))
                                p_st_9.append(id.index(row[4]))
                        if row[4] == '10':
                                x1=row[0]
                                x_st_10.append(x1)
                                x2=row[1]
                                x_st_10.append(x2)
                                x3= row[2]
                                x_st_10.append(x3)
                                y_st_10.append(mvts.index(row[3]))
                                p_st_10.append(id.index(row[4]))
                        if row[4] == '11':
                                x1=row[0]
                                x_st_11.append(x1)
                                x2=row[1]
                                x_st_11.append(x2)
                                x3= row[2]
                                x_st_11.append(x3)
                                y_st_11.append(mvts.index(row[3]))
                                p_st_11.append(id.index(row[4]))
                        if row[4] == '12':
                                x1=row[0]
                                x_st_12.append(x1)
                                x2=row[1]
                                x_st_12.append(x2)
                                x3= row[2]
                                x_st_12.append(x3)
                                y_st_12.append(mvts.index(row[3]))
                                p_st_12.append(id.index(row[4]))
                        if row[4] == '13':
                                x1=row[0]
                                x_st_13.append(x1)
                                x2=row[1]
                                x_st_13.append(x2)
                                x3= row[2]
                                x_st_13.append(x3)
                                y_st_13.append(mvts.index(row[3]))
                                p_st_13.append(id.index(row[4]))
                        if row[4] == '14':
                                x1=row[0]
                                x_st_14.append(x1)
                                x2=row[1]
                                x_st_14.append(x2)
                                x3= row[2]
                                x_st_14.append(x3)
                                y_st_14.append(mvts.index(row[3]))
                                p_st_14.append(id.index(row[4]))
                        if row[4] == '15':
                                x1=row[0]
                                x_st_15.append(x1)
                                x2=row[1]
                                x_st_15.append(x2)
                                x3= row[2]
                                x_st_15.append(x3)
                                y_st_15.append(mvts.index(row[3]))
                                p_st_15.append(id.index(row[4]))
                        if row[4] == '16':
                                x1=row[0]
                                x_st_16.append(x1)
                                x2=row[1]
                                x_st_16.append(x2)
                                x3= row[2]
                                x_st_16.append(x3)
                                y_st_16.append(mvts.index(row[3]))
                                p_st_16.append(id.index(row[4]))
                        if row[4] == '17':
                                x1=row[0]
                                x_st_17.append(x1)
                                x2=row[1]
                                x_st_17.append(x2)
                                x3= row[2]
                                x_st_17.append(x3)
                                y_st_17.append(mvts.index(row[3]))
                                p_st_17.append(id.index(row[4]))
                        if row[4] == '18':
                                x1=row[0]
                                x_st_18.append(x1)
                                x2=row[1]
                                x_st_18.append(x2)
                                x3= row[2]
                                x_st_18.append(x3)
                                y_st_18.append(mvts.index(row[3]))
                                p_st_18.append(id.index(row[4]))
                        if row[4] == '19':
                                x1=row[0]
                                x_st_19.append(x1)
                                x2=row[1]
                                x_st_19.append(x2)
                                x3= row[2]
                                x_st_19.append(x3)
                                y_st_19.append(mvts.index(row[3]))
                                p_st_19.append(id.index(row[4]))
                        if row[4] == '20':
                                x1=row[0]
                                x_st_20.append(x1)
                                x2=row[1]
                                x_st_20.append(x2)
                                x3= row[2]
                                x_st_20.append(x3)
                                y_st_20.append(mvts.index(row[3]))
                                p_st_20.append(id.index(row[4]))
                        if row[4] == '21':
                                x1=row[0]
                                x_st_21.append(x1)
                                x2=row[1]
                                x_st_21.append(x2)
                                x3= row[2]
                                x_st_21.append(x3)
                                y_st_21.append(mvts.index(row[3]))
                                p_st_21.append(id.index(row[4]))
                        if row[4] == '22':
                                x1=row[0]
                                x_st_22.append(x1)
                                x2=row[1]
                                x_st_22.append(x2)
                                x3= row[2]
                                x_st_22.append(x3)
                                y_st_22.append(mvts.index(row[3]))
                                p_st_22.append(id.index(row[4]))
                        if row[4] == '23':
                                x1=row[0]
                                x_st_23.append(x1)
                                x2=row[1]
                                x_st_23.append(x2)
                                x3= row[2]
                                x_st_23.append(x3)
                                y_st_23.append(mvts.index(row[3]))
                                p_st_23.append(id.index(row[4]))
                        if row[4] == '24':
                                x1=row[0]
                                x_st_24.append(x1)
                                x2=row[1]
                                x_st_24.append(x2)
                                x3= row[2]
                                x_st_24.append(x3)
                                y_st_24.append(mvts.index(row[3]))
                                p_st_24.append(id.index(row[4]))
                        if row[4] == '25':
                                x1=row[0]
                                x_st_25.append(x1)
                                x2=row[1]
                                x_st_25.append(x2)
                                x3= row[2]
                                x_st_25.append(x3)
                                y_st_25.append(mvts.index(row[3]))
                                p_st_25.append(id.index(row[4]))
                        if row[4] == '26':
                                x1=row[0]
                                x_st_26.append(x1)
                                x2=row[1]
                                x_st_26.append(x2)
                                x3= row[2]
                                x_st_26.append(x3)
                                y_st_26.append(mvts.index(row[3]))
                                p_st_26.append(id.index(row[4]))
                        if row[4] == '27':
                                x1=row[0]
                                x_st_27.append(x1)
                                x2=row[1]
                                x_st_27.append(x2)
                                x3= row[2]
                                x_st_27.append(x3)
                                y_st_27.append(mvts.index(row[3]))
                                p_st_27.append(id.index(row[4]))
                        if row[4] == '28':
                                x1=row[0]
                                x_st_28.append(x1)
                                x2=row[1]
                                x_st_28.append(x2)
                                x3= row[2]
                                x_st_28.append(x3)
                                y_st_28.append(mvts.index(row[3]))
                                p_st_28.append(id.index(row[4]))
                        if row[4] == '29':
                                x1=row[0]
                                x_st_29.append(x1)
                                x2=row[1]
                                x_st_29.append(x2)
                                x3= row[2]
                                x_st_29.append(x3)
                                y_st_29.append(mvts.index(row[3]))
                                p_st_29.append(id.index(row[4]))
                        if row[4] == '30':
                                x1=row[0]
                                x_st_30.append(x1)
                                x2=row[1]
                                x_st_30.append(x2)
                                x3= row[2]
                                x_st_30.append(x3)
                                y_st_30.append(mvts.index(row[3]))
                                p_st_30.append(id.index(row[4]))
                        if row[4] == '31':
                                x1=row[0]
                                x_st_31.append(x1)
                                x2=row[1]
                                x_st_31.append(x2)
                                x3= row[2]
                                x_st_31.append(x3)
                                y_st_31.append(mvts.index(row[3]))
                                p_st_31.append(id.index(row[4]))
                        if row[4] == '32':
                                x1=row[0]
                                x_st_32.append(x1)
                                x2=row[1]
                                x_st_32.append(x2)
                                x3= row[2]
                                x_st_32.append(x3)
                                y_st_32.append(mvts.index(row[3]))
                                p_st_32.append(id.index(row[4]))
                        if row[4] == '33':
                                x1=row[0]
                                x_st_33.append(x1)
                                x2=row[1]
                                x_st_33.append(x2)
                                x3= row[2]
                                x_st_33.append(x3)
                                y_st_33.append(mvts.index(row[3]))
                                p_st_33.append(id.index(row[4]))
                        if row[4] == '34':
                                x1=row[0]
                                x_st_34.append(x1)
                                x2=row[1]
                                x_st_34.append(x2)
                                x3= row[2]
                                x_st_34.append(x3)
                                y_st_34.append(mvts.index(row[3]))
                                p_st_34.append(id.index(row[4]))
                        if row[4] == '35':
                                x1=row[0]
                                x_st_35.append(x1)
                                x2=row[1]
                                x_st_35.append(x2)
                                x3= row[2]
                                x_st_35.append(x3)
                                y_st_35.append(mvts.index(row[3]))
                                p_st_35.append(id.index(row[4]))
                        if row[4] == '36':
                                x1=row[0]
                                x_st_36.append(x1)
                                x2=row[1]
                                x_st_36.append(x2)
                                x3= row[2]
                                x_st_36.append(x3)
                                y_st_36.append(mvts.index(row[3]))
                                p_st_36.append(id.index(row[4]))
                        if row[4] == '37':
                                x1=row[0]
                                x_st_37.append(x1)
                                x2=row[1]
                                x_st_37.append(x2)
                                x3= row[2]
                                x_st_37.append(x3)
                                y_st_37.append(mvts.index(row[3]))
                                p_st_37.append(id.index(row[4]))
                        if row[4] == '38':
                                x1=row[0]
                                x_st_38.append(x1)
                                x2=row[1]
                                x_st_38.append(x2)
                                x3= row[2]
                                x_st_38.append(x3)
                                y_st_38.append(mvts.index(row[3]))
                                p_st_38.append(id.index(row[4]))
                        if row[4] == '39':
                                x1=row[0]
                                x_st_39.append(x1)
                                x2=row[1]
                                x_st_39.append(x2)
                                x3= row[2]
                                x_st_39.append(x3)
                                y_st_39.append(mvts.index(row[3]))
                                p_st_39.append(id.index(row[4]))
                        if row[4] == '40':
                                x1=row[0]
                                x_st_40.append(x1)
                                x2=row[1]
                                x_st_40.append(x2)
                                x3= row[2]
                                x_st_40.append(x3)
                                y_st_40.append(mvts.index(row[3]))
                                p_st_40.append(id.index(row[4]))
                        if row[4] == '41':
                                x1=row[0]
                                x_st_41.append(x1)
                                x2=row[1]
                                x_st_41.append(x2)
                                x3= row[2]
                                x_st_41.append(x3)
                                y_st_41.append(mvts.index(row[3]))
                                p_st_41.append(id.index(row[4]))
                        if row[4] == '42':
                                x1=row[0]
                                x_st_42.append(x1)
                                x2=row[1]
                                x_st_42.append(x2)
                                x3= row[2]
                                x_st_42.append(x3)
                                y_st_42.append(mvts.index(row[3]))
                                p_st_42.append(id.index(row[4]))
                        if row[4] == '43':
                                x1=row[0]
                                x_st_43.append(x1)
                                x2=row[1]
                                x_st_43.append(x2)
                                x3= row[2]
                                x_st_43.append(x3)
                                y_st_43.append(mvts.index(row[3]))
                                p_st_43.append(id.index(row[4]))
                        if row[4] == '44':
                                x1=row[0]
                                x_st_44.append(x1)
                                x2=row[1]
                                x_st_44.append(x2)
                                x3= row[2]
                                x_st_44.append(x3)
                                y_st_44.append(mvts.index(row[3]))
                                p_st_44.append(id.index(row[4]))
                        if row[4] == '45':
                                x1=row[0]
                                x_st_45.append(x1)
                                x2=row[1]
                                x_st_45.append(x2)
                                x3= row[2]
                                x_st_45.append(x3)
                                y_st_45.append(mvts.index(row[3]))
                                p_st_45.append(id.index(row[4]))
                        if row[4] == '46':
                                x1=row[0]
                                x_st_46.append(x1)
                                x2=row[1]
                                x_st_46.append(x2)
                                x3= row[2]
                                x_st_46.append(x3)
                                y_st_46.append(mvts.index(row[3]))
                                p_st_46.append(id.index(row[4]))
                        if row[4] == '47':
                                x1=row[0]
                                x_st_47.append(x1)
                                x2=row[1]
                                x_st_47.append(x2)
                                x3= row[2]
                                x_st_47.append(x3)
                                y_st_47.append(mvts.index(row[3]))
                                p_st_47.append(id.index(row[4]))
                        if row[4] == '48':
                                x1=row[0]
                                x_st_48.append(x1)
                                x2=row[1]
                                x_st_48.append(x2)
                                x3= row[2]
                                x_st_48.append(x3)
                                y_st_48.append(mvts.index(row[3]))
                                p_st_48.append(id.index(row[4]))
                        if row[4] == '49':
                                x1=row[0]
                                x_st_49.append(x1)
                                x2=row[1]
                                x_st_49.append(x2)
                                x3= row[2]
                                x_st_49.append(x3)
                                y_st_49.append(mvts.index(row[3]))
                                p_st_49.append(id.index(row[4]))
                        if row[4] == '50':
                                x1=row[0]
                                x_st_50.append(x1)
                                x2=row[1]
                                x_st_50.append(x2)
                                x3= row[2]
                                x_st_50.append(x3)
                                y_st_50.append(mvts.index(row[3]))
                                p_st_50.append(id.index(row[4]))
                        if row[4] == '51':
                                x1=row[0]
                                x_st_51.append(x1)
                                x2=row[1]
                                x_st_51.append(x2)
                                x3= row[2]
                                x_st_51.append(x3)
                                y_st_51.append(mvts.index(row[3]))
                                p_st_51.append(id.index(row[4]))
                        if row[4] == '52':
                                x1=row[0]
                                x_st_52.append(x1)
                                x2=row[1]
                                x_st_52.append(x2)
                                x3= row[2]
                                x_st_52.append(x3)
                                y_st_52.append(mvts.index(row[3]))
                                p_st_52.append(id.index(row[4]))
                        if row[4] == '53':
                                x1=row[0]
                                x_st_53.append(x1)
                                x2=row[1]
                                x_st_53.append(x2)
                                x3= row[2]
                                x_st_53.append(x3)
                                y_st_53.append(mvts.index(row[3]))
                                p_st_53.append(id.index(row[4]))
                        if row[4] == '54':
                                x1=row[0]
                                x_st_54.append(x1)
                                x2=row[1]
                                x_st_54.append(x2)
                                x3= row[2]
                                x_st_54.append(x3)
                                y_st_54.append(mvts.index(row[3]))
                                p_st_54.append(id.index(row[4]))
                        if row[4] == '55':
                                x1=row[0]
                                x_st_55.append(x1)
                                x2=row[1]
                                x_st_55.append(x2)
                                x3= row[2]
                                x_st_55.append(x3)
                                y_st_55.append(mvts.index(row[3]))
                                p_st_55.append(id.index(row[4]))
                        if row[4] == '56':
                                x1=row[0]
                                x_st_56.append(x1)
                                x2=row[1]
                                x_st_56.append(x2)
                                x3= row[2]
                                x_st_56.append(x3)
                                y_st_56.append(mvts.index(row[3]))
                                p_st_56.append(id.index(row[4]))
                        if row[4] == '57':
                                x1=row[0]
                                x_st_57.append(x1)
                                x2=row[1]
                                x_st_57.append(x2)
                                x3= row[2]
                                x_st_57.append(x3)
                                y_st_57.append(mvts.index(row[3]))
                                p_st_57.append(id.index(row[4]))
                        if row[4] == '58':
                                x1=row[0]
                                x_st_58.append(x1)
                                x2=row[1]
                                x_st_58.append(x2)
                                x3= row[2]
                                x_st_58.append(x3)
                                y_st_58.append(mvts.index(row[3]))
                                p_st_58.append(id.index(row[4]))
                        if row[4] == '59':
                                x1=row[0]
                                x_st_59.append(x1)
                                x2=row[1]
                                x_st_59.append(x2)
                                x3= row[2]
                                x_st_59.append(x3)
                                y_st_59.append(mvts.index(row[3]))
                                p_st_59.append(id.index(row[4]))
                        if row[4] == '60':
                                x1=row[0]
                                x_st_60.append(x1)
                                x2=row[1]
                                x_st_60.append(x2)
                                x3= row[2]
                                x_st_60.append(x3)
                                y_st_60.append(mvts.index(row[3]))
                                p_st_60.append(id.index(row[4]))
                        if row[4] == '61':
                                x1=row[0]
                                x_st_61.append(x1)
                                x2=row[1]
                                x_st_61.append(x2)
                                x3= row[2]
                                x_st_61.append(x3)
                                y_st_61.append(mvts.index(row[3]))
                                p_st_61.append(id.index(row[4]))
                        if row[4] == '62':
                                x1=row[0]
                                x_st_62.append(x1)
                                x2=row[1]
                                x_st_62.append(x2)
                                x3= row[2]
                                x_st_62.append(x3)
                                y_st_62.append(mvts.index(row[3]))
                                p_st_62.append(id.index(row[4]))
                        if row[4] == '63':
                                x1=row[0]
                                x_st_63.append(x1)
                                x2=row[1]
                                x_st_63.append(x2)
                                x3= row[2]
                                x_st_63.append(x3)
                                y_st_63.append(mvts.index(row[3]))
                                p_st_63.append(id.index(row[4]))
                        if row[4] == '64':
                                x1=row[0]
                                x_st_64.append(x1)
                                x2=row[1]
                                x_st_64.append(x2)
                                x3= row[2]
                                x_st_64.append(x3)
                                y_st_64.append(mvts.index(row[3]))
                                p_st_64.append(id.index(row[4]))
                        if row[4] == '65':
                                x1=row[0]
                                x_st_65.append(x1)
                                x2=row[1]
                                x_st_65.append(x2)
                                x3= row[2]
                                x_st_65.append(x3)
                                y_st_65.append(mvts.index(row[3]))
                                p_st_65.append(id.index(row[4]))
                        if row[4] == '66':
                                x1=row[0]
                                x_st_66.append(x1)
                                x2=row[1]
                                x_st_66.append(x2)
                                x3= row[2]
                                x_st_66.append(x3)
                                y_st_66.append(mvts.index(row[3]))
                                p_st_66.append(id.index(row[4]))
                        if row[4] == '67':
                                x1=row[0]
                                x_st_67.append(x1)
                                x2=row[1]
                                x_st_67.append(x2)
                                x3= row[2]
                                x_st_67.append(x3)
                                y_st_67.append(mvts.index(row[3]))
                                p_st_67.append(id.index(row[4]))
                        if row[4] == '68':
                                x1=row[0]
                                x_st_68.append(x1)
                                x2=row[1]
                                x_st_68.append(x2)
                                x3= row[2]
                                x_st_68.append(x3)
                                y_st_68.append(mvts.index(row[3]))
                                p_st_68.append(id.index(row[4]))
                        if row[4] == '69':
                                x1=row[0]
                                x_st_69.append(x1)
                                x2=row[1]
                                x_st_69.append(x2)
                                x3= row[2]
                                x_st_69.append(x3)
                                y_st_69.append(mvts.index(row[3]))
                                p_st_69.append(id.index(row[4]))
                        if row[4] == '70':
                                x1=row[0]
                                x_st_70.append(x1)
                                x2=row[1]
                                x_st_70.append(x2)
                                x3= row[2]
                                x_st_70.append(x3)
                                y_st_70.append(mvts.index(row[3]))
                                p_st_70.append(id.index(row[4]))
                        if row[4] == '71':
                                x1=row[0]
                                x_st_71.append(x1)
                                x2=row[1]
                                x_st_71.append(x2)
                                x3= row[2]
                                x_st_71.append(x3)
                                y_st_71.append(mvts.index(row[3]))
                                p_st_71.append(id.index(row[4]))
                        if row[4] == '72':
                                x1=row[0]
                                x_st_72.append(x1)
                                x2=row[1]
                                x_st_72.append(x2)
                                x3= row[2]
                                x_st_72.append(x3)
                                y_st_72.append(mvts.index(row[3]))
                                p_st_72.append(id.index(row[4]))
                        if row[4] == '73':
                                x1=row[0]
                                x_st_73.append(x1)
                                x2=row[1]
                                x_st_73.append(x2)
                                x3= row[2]
                                x_st_73.append(x3)
                                y_st_73.append(mvts.index(row[3]))
                                p_st_73.append(id.index(row[4]))
                        if row[4] == '74':
                                x1=row[0]
                                x_st_74.append(x1)
                                x2=row[1]
                                x_st_74.append(x2)
                                x3= row[2]
                                x_st_74.append(x3)
                                y_st_74.append(mvts.index(row[3]))
                                p_st_74.append(id.index(row[4]))
                        if row[4] == '75':
                                x1=row[0]
                                x_st_75.append(x1)
                                x2=row[1]
                                x_st_75.append(x2)
                                x3= row[2]
                                x_st_75.append(x3)
                                y_st_75.append(mvts.index(row[3]))
                                p_st_75.append(id.index(row[4]))
                        if row[4] == '76':
                                x1=row[0]
                                x_st_76.append(x1)
                                x2=row[1]
                                x_st_76.append(x2)
                                x3= row[2]
                                x_st_76.append(x3)
                                y_st_76.append(mvts.index(row[3]))
                                p_st_76.append(id.index(row[4]))
                        if row[4] == '77':
                                x1=row[0]
                                x_st_77.append(x1)
                                x2=row[1]
                                x_st_77.append(x2)
                                x3= row[2]
                                x_st_77.append(x3)
                                y_st_77.append(mvts.index(row[3]))
                                p_st_77.append(id.index(row[4]))
                        if row[4] == '78':
                                x1=row[0]
                                x_st_78.append(x1)
                                x2=row[1]
                                x_st_78.append(x2)
                                x3= row[2]
                                x_st_78.append(x3)
                                y_st_78.append(mvts.index(row[3]))
                                p_st_78.append(id.index(row[4]))
                        if row[4] == '79':
                                x1=row[0]
                                x_st_79.append(x1)
                                x2=row[1]
                                x_st_79.append(x2)
                                x3= row[2]
                                x_st_79.append(x3)
                                y_st_79.append(mvts.index(row[3]))
                                p_st_79.append(id.index(row[4]))
                        if row[4] == '80':
                                x1=row[0]
                                x_st_80.append(x1)
                                x2=row[1]
                                x_st_80.append(x2)
                                x3= row[2]
                                x_st_80.append(x3)
                                y_st_80.append(mvts.index(row[3]))
                                p_st_80.append(id.index(row[4]))
                        if row[4] == '81':
                                x1=row[0]
                                x_st_81.append(x1)
                                x2=row[1]
                                x_st_81.append(x2)
                                x3= row[2]
                                x_st_81.append(x3)
                                y_st_81.append(mvts.index(row[3]))
                                p_st_81.append(id.index(row[4]))
                        if row[4] == '82':
                                x1=row[0]
                                x_st_82.append(x1)
                                x2=row[1]
                                x_st_82.append(x2)
                                x3= row[2]
                                x_st_82.append(x3)
                                y_st_82.append(mvts.index(row[3]))
                                p_st_82.append(id.index(row[4]))
                        if row[4] == '83':
                                x1=row[0]
                                x_st_83.append(x1)
                                x2=row[1]
                                x_st_83.append(x2)
                                x3= row[2]
                                x_st_83.append(x3)
                                y_st_83.append(mvts.index(row[3]))
                                p_st_83.append(id.index(row[4]))
                        if row[4] == '84':
                                x1=row[0]
                                x_st_84.append(x1)
                                x2=row[1]
                                x_st_84.append(x2)
                                x3= row[2]
                                x_st_84.append(x3)
                                y_st_84.append(mvts.index(row[3]))
                                p_st_84.append(id.index(row[4]))
                        if row[4] == '85':
                                x1=row[0]
                                x_st_85.append(x1)
                                x2=row[1]
                                x_st_85.append(x2)
                                x3= row[2]
                                x_st_85.append(x3)
                                y_st_85.append(mvts.index(row[3]))
                                p_st_85.append(id.index(row[4]))
                        if row[4] == '86':
                                x1=row[0]
                                x_st_86.append(x1)
                                x2=row[1]
                                x_st_86.append(x2)
                                x3= row[2]
                                x_st_86.append(x3)
                                y_st_86.append(mvts.index(row[3]))
                                p_st_86.append(id.index(row[4]))
                        if row[4] == '87':
                                x1=row[0]
                                x_st_87.append(x1)
                                x2=row[1]
                                x_st_87.append(x2)
                                x3= row[2]
                                x_st_87.append(x3)
                                y_st_87.append(mvts.index(row[3]))
                                p_st_87.append(id.index(row[4]))
                        if row[4] == '88':
                                x1=row[0]
                                x_st_88.append(x1)
                                x2=row[1]
                                x_st_88.append(x2)
                                x3= row[2]
                                x_st_88.append(x3)
                                y_st_88.append(mvts.index(row[3]))
                                p_st_88.append(id.index(row[4]))
                        if row[4] == '89':
                                x1=row[0]
                                x_st_89.append(x1)
                                x2=row[1]
                                x_st_89.append(x2)
                                x3= row[2]
                                x_st_89.append(x3)
                                y_st_89.append(mvts.index(row[3]))
                                p_st_89.append(id.index(row[4]))
                        if row[4] == '90':
                                x1=row[0]
                                x_st_90.append(x1)
                                x2=row[1]
                                x_st_90.append(x2)
                                x3= row[2]
                                x_st_90.append(x3)
                                y_st_90.append(mvts.index(row[3]))
                                p_st_90.append(id.index(row[4]))
                        if row[4] == '91':
                                x1=row[0]
                                x_st_91.append(x1)
                                x2=row[1]
                                x_st_91.append(x2)
                                x3= row[2]
                                x_st_91.append(x3)
                                y_st_91.append(mvts.index(row[3]))
                                p_st_91.append(id.index(row[4]))
                        if row[4] == '92':
                                x1=row[0]
                                x_st_92.append(x1)
                                x2=row[1]
                                x_st_92.append(x2)
                                x3= row[2]
                                x_st_92.append(x3)
                                y_st_92.append(mvts.index(row[3]))
                                p_st_92.append(id.index(row[4]))
                        if row[4] == '93':
                                x1=row[0]
                                x_st_93.append(x1)
                                x2=row[1]
                                x_st_93.append(x2)
                                x3= row[2]
                                x_st_93.append(x3)
                                y_st_93.append(mvts.index(row[3]))
                                p_st_93.append(id.index(row[4]))
                        if row[4] == '94':
                                x1=row[0]
                                x_st_94.append(x1)
                                x2=row[1]
                                x_st_94.append(x2)
                                x3= row[2]
                                x_st_94.append(x3)
                                y_st_94.append(mvts.index(row[3]))
                                p_st_94.append(id.index(row[4]))
                        if row[4] == '95':
                                x1=row[0]
                                x_st_95.append(x1)
                                x2=row[1]
                                x_st_95.append(x2)
                                x3= row[2]
                                x_st_95.append(x3)
                                y_st_95.append(mvts.index(row[3]))
                                p_st_95.append(id.index(row[4]))
                        if row[4] == '96':
                                x1=row[0]
                                x_st_96.append(x1)
                                x2=row[1]
                                x_st_96.append(x2)
                                x3= row[2]
                                x_st_96.append(x3)
                                y_st_96.append(mvts.index(row[3]))
                                p_st_96.append(id.index(row[4]))
                        if row[4] == '97':
                                x1=row[0]
                                x_st_97.append(x1)
                                x2=row[1]
                                x_st_97.append(x2)
                                x3= row[2]
                                x_st_97.append(x3)
                                y_st_97.append(mvts.index(row[3]))
                                p_st_97.append(id.index(row[4]))
                        if row[4] == '98':
                                x1=row[0]
                                x_st_98.append(x1)
                                x2=row[1]
                                x_st_98.append(x2)
                                x3= row[2]
                                x_st_98.append(x3)
                                y_st_98.append(mvts.index(row[3]))
                                p_st_98.append(id.index(row[4]))
                        if row[4] == '99':
                                x1=row[0]
                                x_st_99.append(x1)
                                x2=row[1]
                                x_st_99.append(x2)
                                x3= row[2]
                                x_st_99.append(x3)
                                y_st_99.append(mvts.index(row[3]))
                                p_st_99.append(id.index(row[4]))
                              

x_j_0= np.array(x_j_0)
y_j_0= np.array(y_j_0)
p_j_0= np.array(p_j_0)
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
x_j_37= np.array(x_j_37)
y_j_37= np.array(y_j_37)
p_j_37= np.array(p_j_37)
x_j_38= np.array(x_j_38)
y_j_38= np.array(y_j_38)
p_j_38= np.array(p_j_38)
x_j_39= np.array(x_j_39)
y_j_39= np.array(y_j_39)
p_j_39= np.array(p_j_39)
x_j_40= np.array(x_j_40)
y_j_40= np.array(y_j_40)
p_j_40= np.array(p_j_40)
x_j_41= np.array(x_j_41)
y_j_41= np.array(y_j_41)
p_j_41= np.array(p_j_41)
x_j_42= np.array(x_j_42)
y_j_42= np.array(y_j_42)
p_j_42= np.array(p_j_42)
x_j_43= np.array(x_j_43)
y_j_43= np.array(y_j_43)
p_j_43= np.array(p_j_43)
x_j_44= np.array(x_j_44)
y_j_44= np.array(y_j_44)
p_j_44= np.array(p_j_44)
x_j_45= np.array(x_j_45)
y_j_45= np.array(y_j_45)
p_j_45= np.array(p_j_45)
x_j_46= np.array(x_j_46)
y_j_46= np.array(y_j_46)
p_j_46= np.array(p_j_46)
x_j_47= np.array(x_j_47)
y_j_47= np.array(y_j_47)
p_j_47= np.array(p_j_47)
x_j_48= np.array(x_j_48)
y_j_48= np.array(y_j_48)
p_j_48= np.array(p_j_48)
x_j_49= np.array(x_j_49)
y_j_49= np.array(y_j_49)
p_j_49= np.array(p_j_49)
x_j_50= np.array(x_j_50)
y_j_50= np.array(y_j_50)
p_j_50= np.array(p_j_50)
x_j_51= np.array(x_j_51)
y_j_51= np.array(y_j_51)
p_j_51= np.array(p_j_51)
x_j_52= np.array(x_j_52)
y_j_52= np.array(y_j_52)
p_j_52= np.array(p_j_52)
x_j_53= np.array(x_j_53)
y_j_53= np.array(y_j_53)
p_j_53= np.array(p_j_53)
x_j_54= np.array(x_j_54)
y_j_54= np.array(y_j_54)
p_j_54= np.array(p_j_54)
x_j_55= np.array(x_j_55)
y_j_55= np.array(y_j_55)
p_j_55= np.array(p_j_55)
x_j_56= np.array(x_j_56)
y_j_56= np.array(y_j_56)
p_j_56= np.array(p_j_56)
x_j_57= np.array(x_j_57)
y_j_57= np.array(y_j_57)
p_j_57= np.array(p_j_57)
x_j_58= np.array(x_j_58)
y_j_58= np.array(y_j_58)
p_j_58= np.array(p_j_58)
x_j_59= np.array(x_j_59)
y_j_59= np.array(y_j_59)
p_j_59= np.array(p_j_59)
x_j_60= np.array(x_j_60)
y_j_60= np.array(y_j_60)
p_j_60= np.array(p_j_60)
x_j_61= np.array(x_j_61)
y_j_61= np.array(y_j_61)
p_j_61= np.array(p_j_61)
x_j_62= np.array(x_j_62)
y_j_62= np.array(y_j_62)
p_j_62= np.array(p_j_62)
x_j_63= np.array(x_j_63)
y_j_63= np.array(y_j_63)
p_j_63= np.array(p_j_63)
x_j_64= np.array(x_j_64)
y_j_64= np.array(y_j_64)
p_j_64= np.array(p_j_64)
x_j_65= np.array(x_j_65)
y_j_65= np.array(y_j_65)
p_j_65= np.array(p_j_65)
x_j_66= np.array(x_j_66)
y_j_66= np.array(y_j_66)
p_j_66= np.array(p_j_66)
x_j_67= np.array(x_j_67)
y_j_67= np.array(y_j_67)
p_j_67= np.array(p_j_67)
x_j_68= np.array(x_j_68)
y_j_68= np.array(y_j_68)
p_j_68= np.array(p_j_68)
x_j_69= np.array(x_j_69)
y_j_69= np.array(y_j_69)
p_j_69= np.array(p_j_69)
x_j_70= np.array(x_j_70)
y_j_70= np.array(y_j_70)
p_j_70= np.array(p_j_70)
x_j_71= np.array(x_j_71)
y_j_71= np.array(y_j_71)
p_j_71= np.array(p_j_71)
x_j_72= np.array(x_j_72)
y_j_72= np.array(y_j_72)
p_j_72= np.array(p_j_72)
x_j_73= np.array(x_j_73)
y_j_73= np.array(y_j_73)
p_j_73= np.array(p_j_73)
x_j_74= np.array(x_j_74)
y_j_74= np.array(y_j_74)
p_j_74= np.array(p_j_74)
x_j_75= np.array(x_j_75)
y_j_75= np.array(y_j_75)
p_j_75= np.array(p_j_75)
x_j_76= np.array(x_j_76)
y_j_76= np.array(y_j_76)
p_j_76= np.array(p_j_76)
x_j_77= np.array(x_j_77)
y_j_77= np.array(y_j_77)
p_j_77= np.array(p_j_77)
x_j_78= np.array(x_j_78)
y_j_78= np.array(y_j_78)
p_j_78= np.array(p_j_78)
x_j_79= np.array(x_j_79)
y_j_79= np.array(y_j_79)
p_j_79= np.array(p_j_79)
x_j_80= np.array(x_j_80)
y_j_80= np.array(y_j_80)
p_j_80= np.array(p_j_80)
x_j_81= np.array(x_j_81)
y_j_81= np.array(y_j_81)
p_j_81= np.array(p_j_81)
x_j_82= np.array(x_j_82)
y_j_82= np.array(y_j_82)
p_j_82= np.array(p_j_82)
x_j_83= np.array(x_j_83)
y_j_83= np.array(y_j_83)
p_j_83= np.array(p_j_83)
x_j_84= np.array(x_j_84)
y_j_84= np.array(y_j_84)
p_j_84= np.array(p_j_84)
x_j_85= np.array(x_j_85)
y_j_85= np.array(y_j_85)
p_j_85= np.array(p_j_85)
x_j_86= np.array(x_j_86)
y_j_86= np.array(y_j_86)
p_j_86= np.array(p_j_86)
x_j_87= np.array(x_j_87)
y_j_87= np.array(y_j_87)
p_j_87= np.array(p_j_87)
x_j_88= np.array(x_j_88)
y_j_88= np.array(y_j_88)
p_j_88= np.array(p_j_88)
x_j_89= np.array(x_j_89)
y_j_89= np.array(y_j_89)
p_j_89= np.array(p_j_89)
x_j_90= np.array(x_j_90)
y_j_90= np.array(y_j_90)
p_j_90= np.array(p_j_90)
x_j_91= np.array(x_j_91)
y_j_91= np.array(y_j_91)
p_j_91= np.array(p_j_91)
x_j_92= np.array(x_j_92)
y_j_92= np.array(y_j_92)
p_j_92= np.array(p_j_92)
x_j_93= np.array(x_j_93)
y_j_93= np.array(y_j_93)
p_j_93= np.array(p_j_93)
x_j_94= np.array(x_j_94)
y_j_94= np.array(y_j_94)
p_j_94= np.array(p_j_94)
x_j_95= np.array(x_j_95)
y_j_95= np.array(y_j_95)
p_j_95= np.array(p_j_95)
x_j_96= np.array(x_j_96)
y_j_96= np.array(y_j_96)
p_j_96= np.array(p_j_96)
x_j_97= np.array(x_j_97)
y_j_97= np.array(y_j_97)
p_j_97= np.array(p_j_97)
x_j_98= np.array(x_j_98)
y_j_98= np.array(y_j_98)
p_j_98= np.array(p_j_98)
x_j_99= np.array(x_j_99)
y_j_99= np.array(y_j_99)
p_j_99= np.array(p_j_99)
x_s_0= np.array(x_s_0)
y_s_0= np.array(y_s_0)
p_s_0= np.array(p_s_0)
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
x_s_37= np.array(x_s_37)
y_s_37= np.array(y_s_37)
p_s_37= np.array(p_s_37)
x_s_38= np.array(x_s_38)
y_s_38= np.array(y_s_38)
p_s_38= np.array(p_s_38)
x_s_39= np.array(x_s_39)
y_s_39= np.array(y_s_39)
p_s_39= np.array(p_s_39)
x_s_40= np.array(x_s_40)
y_s_40= np.array(y_s_40)
p_s_40= np.array(p_s_40)
x_s_41= np.array(x_s_41)
y_s_41= np.array(y_s_41)
p_s_41= np.array(p_s_41)
x_s_42= np.array(x_s_42)
y_s_42= np.array(y_s_42)
p_s_42= np.array(p_s_42)
x_s_43= np.array(x_s_43)
y_s_43= np.array(y_s_43)
p_s_43= np.array(p_s_43)
x_s_44= np.array(x_s_44)
y_s_44= np.array(y_s_44)
p_s_44= np.array(p_s_44)
x_s_45= np.array(x_s_45)
y_s_45= np.array(y_s_45)
p_s_45= np.array(p_s_45)
x_s_46= np.array(x_s_46)
y_s_46= np.array(y_s_46)
p_s_46= np.array(p_s_46)
x_s_47= np.array(x_s_47)
y_s_47= np.array(y_s_47)
p_s_47= np.array(p_s_47)
x_s_48= np.array(x_s_48)
y_s_48= np.array(y_s_48)
p_s_48= np.array(p_s_48)
x_s_49= np.array(x_s_49)
y_s_49= np.array(y_s_49)
p_s_49= np.array(p_s_49)
x_s_50= np.array(x_s_50)
y_s_50= np.array(y_s_50)
p_s_50= np.array(p_s_50)
x_s_51= np.array(x_s_51)
y_s_51= np.array(y_s_51)
p_s_51= np.array(p_s_51)
x_s_52= np.array(x_s_52)
y_s_52= np.array(y_s_52)
p_s_52= np.array(p_s_52)
x_s_53= np.array(x_s_53)
y_s_53= np.array(y_s_53)
p_s_53= np.array(p_s_53)
x_s_54= np.array(x_s_54)
y_s_54= np.array(y_s_54)
p_s_54= np.array(p_s_54)
x_s_55= np.array(x_s_55)
y_s_55= np.array(y_s_55)
p_s_55= np.array(p_s_55)
x_s_56= np.array(x_s_56)
y_s_56= np.array(y_s_56)
p_s_56= np.array(p_s_56)
x_s_57= np.array(x_s_57)
y_s_57= np.array(y_s_57)
p_s_57= np.array(p_s_57)
x_s_58= np.array(x_s_58)
y_s_58= np.array(y_s_58)
p_s_58= np.array(p_s_58)
x_s_59= np.array(x_s_59)
y_s_59= np.array(y_s_59)
p_s_59= np.array(p_s_59)
x_s_60= np.array(x_s_60)
y_s_60= np.array(y_s_60)
p_s_60= np.array(p_s_60)
x_s_61= np.array(x_s_61)
y_s_61= np.array(y_s_61)
p_s_61= np.array(p_s_61)
x_s_62= np.array(x_s_62)
y_s_62= np.array(y_s_62)
p_s_62= np.array(p_s_62)
x_s_63= np.array(x_s_63)
y_s_63= np.array(y_s_63)
p_s_63= np.array(p_s_63)
x_s_64= np.array(x_s_64)
y_s_64= np.array(y_s_64)
p_s_64= np.array(p_s_64)
x_s_65= np.array(x_s_65)
y_s_65= np.array(y_s_65)
p_s_65= np.array(p_s_65)
x_s_66= np.array(x_s_66)
y_s_66= np.array(y_s_66)
p_s_66= np.array(p_s_66)
x_s_67= np.array(x_s_67)
y_s_67= np.array(y_s_67)
p_s_67= np.array(p_s_67)
x_s_68= np.array(x_s_68)
y_s_68= np.array(y_s_68)
p_s_68= np.array(p_s_68)
x_s_69= np.array(x_s_69)
y_s_69= np.array(y_s_69)
p_s_69= np.array(p_s_69)
x_s_70= np.array(x_s_70)
y_s_70= np.array(y_s_70)
p_s_70= np.array(p_s_70)
x_s_71= np.array(x_s_71)
y_s_71= np.array(y_s_71)
p_s_71= np.array(p_s_71)
x_s_72= np.array(x_s_72)
y_s_72= np.array(y_s_72)
p_s_72= np.array(p_s_72)
x_s_73= np.array(x_s_73)
y_s_73= np.array(y_s_73)
p_s_73= np.array(p_s_73)
x_s_74= np.array(x_s_74)
y_s_74= np.array(y_s_74)
p_s_74= np.array(p_s_74)
x_s_75= np.array(x_s_75)
y_s_75= np.array(y_s_75)
p_s_75= np.array(p_s_75)
x_s_76= np.array(x_s_76)
y_s_76= np.array(y_s_76)
p_s_76= np.array(p_s_76)
x_s_77= np.array(x_s_77)
y_s_77= np.array(y_s_77)
p_s_77= np.array(p_s_77)
x_s_78= np.array(x_s_78)
y_s_78= np.array(y_s_78)
p_s_78= np.array(p_s_78)
x_s_79= np.array(x_s_79)
y_s_79= np.array(y_s_79)
p_s_79= np.array(p_s_79)
x_s_80= np.array(x_s_80)
y_s_80= np.array(y_s_80)
p_s_80= np.array(p_s_80)
x_s_81= np.array(x_s_81)
y_s_81= np.array(y_s_81)
p_s_81= np.array(p_s_81)
x_s_82= np.array(x_s_82)
y_s_82= np.array(y_s_82)
p_s_82= np.array(p_s_82)
x_s_83= np.array(x_s_83)
y_s_83= np.array(y_s_83)
p_s_83= np.array(p_s_83)
x_s_84= np.array(x_s_84)
y_s_84= np.array(y_s_84)
p_s_84= np.array(p_s_84)
x_s_85= np.array(x_s_85)
y_s_85= np.array(y_s_85)
p_s_85= np.array(p_s_85)
x_s_86= np.array(x_s_86)
y_s_86= np.array(y_s_86)
p_s_86= np.array(p_s_86)
x_s_87= np.array(x_s_87)
y_s_87= np.array(y_s_87)
p_s_87= np.array(p_s_87)
x_s_88= np.array(x_s_88)
y_s_88= np.array(y_s_88)
p_s_88= np.array(p_s_88)
x_s_89= np.array(x_s_89)
y_s_89= np.array(y_s_89)
p_s_89= np.array(p_s_89)
x_s_90= np.array(x_s_90)
y_s_90= np.array(y_s_90)
p_s_90= np.array(p_s_90)
x_s_91= np.array(x_s_91)
y_s_91= np.array(y_s_91)
p_s_91= np.array(p_s_91)
x_s_92= np.array(x_s_92)
y_s_92= np.array(y_s_92)
p_s_92= np.array(p_s_92)
x_s_93= np.array(x_s_93)
y_s_93= np.array(y_s_93)
p_s_93= np.array(p_s_93)
x_s_94= np.array(x_s_94)
y_s_94= np.array(y_s_94)
p_s_94= np.array(p_s_94)
x_s_95= np.array(x_s_95)
y_s_95= np.array(y_s_95)
p_s_95= np.array(p_s_95)
x_s_96= np.array(x_s_96)
y_s_96= np.array(y_s_96)
p_s_96= np.array(p_s_96)
x_s_97= np.array(x_s_97)
y_s_97= np.array(y_s_97)
p_s_97= np.array(p_s_97)
x_s_98= np.array(x_s_98)
y_s_98= np.array(y_s_98)
p_s_98= np.array(p_s_98)
x_s_99= np.array(x_s_99)
y_s_99= np.array(y_s_99)
p_s_99= np.array(p_s_99)
x_w_0= np.array(x_w_0)
y_w_0= np.array(y_w_0)
p_w_0= np.array(p_w_0)
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
x_w_37= np.array(x_w_37)
y_w_37= np.array(y_w_37)
p_w_37= np.array(p_w_37)
x_w_38= np.array(x_w_38)
y_w_38= np.array(y_w_38)
p_w_38= np.array(p_w_38)
x_w_39= np.array(x_w_39)
y_w_39= np.array(y_w_39)
p_w_39= np.array(p_w_39)
x_w_40= np.array(x_w_40)
y_w_40= np.array(y_w_40)
p_w_40= np.array(p_w_40)
x_w_41= np.array(x_w_41)
y_w_41= np.array(y_w_41)
p_w_41= np.array(p_w_41)
x_w_42= np.array(x_w_42)
y_w_42= np.array(y_w_42)
p_w_42= np.array(p_w_42)
x_w_43= np.array(x_w_43)
y_w_43= np.array(y_w_43)
p_w_43= np.array(p_w_43)
x_w_44= np.array(x_w_44)
y_w_44= np.array(y_w_44)
p_w_44= np.array(p_w_44)
x_w_45= np.array(x_w_45)
y_w_45= np.array(y_w_45)
p_w_45= np.array(p_w_45)
x_w_46= np.array(x_w_46)
y_w_46= np.array(y_w_46)
p_w_46= np.array(p_w_46)
x_w_47= np.array(x_w_47)
y_w_47= np.array(y_w_47)
p_w_47= np.array(p_w_47)
x_w_48= np.array(x_w_48)
y_w_48= np.array(y_w_48)
p_w_48= np.array(p_w_48)
x_w_49= np.array(x_w_49)
y_w_49= np.array(y_w_49)
p_w_49= np.array(p_w_49)
x_w_50= np.array(x_w_50)
y_w_50= np.array(y_w_50)
p_w_50= np.array(p_w_50)
x_w_51= np.array(x_w_51)
y_w_51= np.array(y_w_51)
p_w_51= np.array(p_w_51)
x_w_52= np.array(x_w_52)
y_w_52= np.array(y_w_52)
p_w_52= np.array(p_w_52)
x_w_53= np.array(x_w_53)
y_w_53= np.array(y_w_53)
p_w_53= np.array(p_w_53)
x_w_54= np.array(x_w_54)
y_w_54= np.array(y_w_54)
p_w_54= np.array(p_w_54)
x_w_55= np.array(x_w_55)
y_w_55= np.array(y_w_55)
p_w_55= np.array(p_w_55)
x_w_56= np.array(x_w_56)
y_w_56= np.array(y_w_56)
p_w_56= np.array(p_w_56)
x_w_57= np.array(x_w_57)
y_w_57= np.array(y_w_57)
p_w_57= np.array(p_w_57)
x_w_58= np.array(x_w_58)
y_w_58= np.array(y_w_58)
p_w_58= np.array(p_w_58)
x_w_59= np.array(x_w_59)
y_w_59= np.array(y_w_59)
p_w_59= np.array(p_w_59)
x_w_60= np.array(x_w_60)
y_w_60= np.array(y_w_60)
p_w_60= np.array(p_w_60)
x_w_61= np.array(x_w_61)
y_w_61= np.array(y_w_61)
p_w_61= np.array(p_w_61)
x_w_62= np.array(x_w_62)
y_w_62= np.array(y_w_62)
p_w_62= np.array(p_w_62)
x_w_63= np.array(x_w_63)
y_w_63= np.array(y_w_63)
p_w_63= np.array(p_w_63)
x_w_64= np.array(x_w_64)
y_w_64= np.array(y_w_64)
p_w_64= np.array(p_w_64)
x_w_65= np.array(x_w_65)
y_w_65= np.array(y_w_65)
p_w_65= np.array(p_w_65)
x_w_66= np.array(x_w_66)
y_w_66= np.array(y_w_66)
p_w_66= np.array(p_w_66)
x_w_67= np.array(x_w_67)
y_w_67= np.array(y_w_67)
p_w_67= np.array(p_w_67)
x_w_68= np.array(x_w_68)
y_w_68= np.array(y_w_68)
p_w_68= np.array(p_w_68)
x_w_69= np.array(x_w_69)
y_w_69= np.array(y_w_69)
p_w_69= np.array(p_w_69)
x_w_70= np.array(x_w_70)
y_w_70= np.array(y_w_70)
p_w_70= np.array(p_w_70)
x_w_71= np.array(x_w_71)
y_w_71= np.array(y_w_71)
p_w_71= np.array(p_w_71)
x_w_72= np.array(x_w_72)
y_w_72= np.array(y_w_72)
p_w_72= np.array(p_w_72)
x_w_73= np.array(x_w_73)
y_w_73= np.array(y_w_73)
p_w_73= np.array(p_w_73)
x_w_74= np.array(x_w_74)
y_w_74= np.array(y_w_74)
p_w_74= np.array(p_w_74)
x_w_75= np.array(x_w_75)
y_w_75= np.array(y_w_75)
p_w_75= np.array(p_w_75)
x_w_76= np.array(x_w_76)
y_w_76= np.array(y_w_76)
p_w_76= np.array(p_w_76)
x_w_77= np.array(x_w_77)
y_w_77= np.array(y_w_77)
p_w_77= np.array(p_w_77)
x_w_78= np.array(x_w_78)
y_w_78= np.array(y_w_78)
p_w_78= np.array(p_w_78)
x_w_79= np.array(x_w_79)
y_w_79= np.array(y_w_79)
p_w_79= np.array(p_w_79)
x_w_80= np.array(x_w_80)
y_w_80= np.array(y_w_80)
p_w_80= np.array(p_w_80)
x_w_81= np.array(x_w_81)
y_w_81= np.array(y_w_81)
p_w_81= np.array(p_w_81)
x_w_82= np.array(x_w_82)
y_w_82= np.array(y_w_82)
p_w_82= np.array(p_w_82)
x_w_83= np.array(x_w_83)
y_w_83= np.array(y_w_83)
p_w_83= np.array(p_w_83)
x_w_84= np.array(x_w_84)
y_w_84= np.array(y_w_84)
p_w_84= np.array(p_w_84)
x_w_85= np.array(x_w_85)
y_w_85= np.array(y_w_85)
p_w_85= np.array(p_w_85)
x_w_86= np.array(x_w_86)
y_w_86= np.array(y_w_86)
p_w_86= np.array(p_w_86)
x_w_87= np.array(x_w_87)
y_w_87= np.array(y_w_87)
p_w_87= np.array(p_w_87)
x_w_88= np.array(x_w_88)
y_w_88= np.array(y_w_88)
p_w_88= np.array(p_w_88)
x_w_89= np.array(x_w_89)
y_w_89= np.array(y_w_89)
p_w_89= np.array(p_w_89)
x_w_90= np.array(x_w_90)
y_w_90= np.array(y_w_90)
p_w_90= np.array(p_w_90)
x_w_91= np.array(x_w_91)
y_w_91= np.array(y_w_91)
p_w_91= np.array(p_w_91)
x_w_92= np.array(x_w_92)
y_w_92= np.array(y_w_92)
p_w_92= np.array(p_w_92)
x_w_93= np.array(x_w_93)
y_w_93= np.array(y_w_93)
p_w_93= np.array(p_w_93)
x_w_94= np.array(x_w_94)
y_w_94= np.array(y_w_94)
p_w_94= np.array(p_w_94)
x_w_95= np.array(x_w_95)
y_w_95= np.array(y_w_95)
p_w_95= np.array(p_w_95)
x_w_96= np.array(x_w_96)
y_w_96= np.array(y_w_96)
p_w_96= np.array(p_w_96)
x_w_97= np.array(x_w_97)
y_w_97= np.array(y_w_97)
p_w_97= np.array(p_w_97)
x_w_98= np.array(x_w_98)
y_w_98= np.array(y_w_98)
p_w_98= np.array(p_w_98)
x_w_99= np.array(x_w_99)
y_w_99= np.array(y_w_99)
p_w_99= np.array(p_w_99)
x_stu_0= np.array(x_stu_0)
y_stu_0= np.array(y_stu_0)
p_stu_0= np.array(p_stu_0)
x_stu_1= np.array(x_stu_1)
y_stu_1= np.array(y_stu_1)
p_stu_1= np.array(p_stu_1)
x_stu_2= np.array(x_stu_2)
y_stu_2= np.array(y_stu_2)
p_stu_2= np.array(p_stu_2)
x_stu_3= np.array(x_stu_3)
y_stu_3= np.array(y_stu_3)
p_stu_3= np.array(p_stu_3)
x_stu_4= np.array(x_stu_4)
y_stu_4= np.array(y_stu_4)
p_stu_4= np.array(p_stu_4)
x_stu_5= np.array(x_stu_5)
y_stu_5= np.array(y_stu_5)
p_stu_5= np.array(p_stu_5)
x_stu_6= np.array(x_stu_6)
y_stu_6= np.array(y_stu_6)
p_stu_6= np.array(p_stu_6)
x_stu_7= np.array(x_stu_7)
y_stu_7= np.array(y_stu_7)
p_stu_7= np.array(p_stu_7)
x_stu_8= np.array(x_stu_8)
y_stu_8= np.array(y_stu_8)
p_stu_8= np.array(p_stu_8)
x_stu_9= np.array(x_stu_9)
y_stu_9= np.array(y_stu_9)
p_stu_9= np.array(p_stu_9)
x_stu_10= np.array(x_stu_10)
y_stu_10= np.array(y_stu_10)
p_stu_10= np.array(p_stu_10)
x_stu_11= np.array(x_stu_11)
y_stu_11= np.array(y_stu_11)
p_stu_11= np.array(p_stu_11)
x_stu_12= np.array(x_stu_12)
y_stu_12= np.array(y_stu_12)
p_stu_12= np.array(p_stu_12)
x_stu_13= np.array(x_stu_13)
y_stu_13= np.array(y_stu_13)
p_stu_13= np.array(p_stu_13)
x_stu_14= np.array(x_stu_14)
y_stu_14= np.array(y_stu_14)
p_stu_14= np.array(p_stu_14)
x_stu_15= np.array(x_stu_15)
y_stu_15= np.array(y_stu_15)
p_stu_15= np.array(p_stu_15)
x_stu_16= np.array(x_stu_16)
y_stu_16= np.array(y_stu_16)
p_stu_16= np.array(p_stu_16)
x_stu_17= np.array(x_stu_17)
y_stu_17= np.array(y_stu_17)
p_stu_17= np.array(p_stu_17)
x_stu_18= np.array(x_stu_18)
y_stu_18= np.array(y_stu_18)
p_stu_18= np.array(p_stu_18)
x_stu_19= np.array(x_stu_19)
y_stu_19= np.array(y_stu_19)
p_stu_19= np.array(p_stu_19)
x_stu_20= np.array(x_stu_20)
y_stu_20= np.array(y_stu_20)
p_stu_20= np.array(p_stu_20)
x_stu_21= np.array(x_stu_21)
y_stu_21= np.array(y_stu_21)
p_stu_21= np.array(p_stu_21)
x_stu_22= np.array(x_stu_22)
y_stu_22= np.array(y_stu_22)
p_stu_22= np.array(p_stu_22)
x_stu_23= np.array(x_stu_23)
y_stu_23= np.array(y_stu_23)
p_stu_23= np.array(p_stu_23)
x_stu_24= np.array(x_stu_24)
y_stu_24= np.array(y_stu_24)
p_stu_24= np.array(p_stu_24)
x_stu_25= np.array(x_stu_25)
y_stu_25= np.array(y_stu_25)
p_stu_25= np.array(p_stu_25)
x_stu_26= np.array(x_stu_26)
y_stu_26= np.array(y_stu_26)
p_stu_26= np.array(p_stu_26)
x_stu_27= np.array(x_stu_27)
y_stu_27= np.array(y_stu_27)
p_stu_27= np.array(p_stu_27)
x_stu_28= np.array(x_stu_28)
y_stu_28= np.array(y_stu_28)
p_stu_28= np.array(p_stu_28)
x_stu_29= np.array(x_stu_29)
y_stu_29= np.array(y_stu_29)
p_stu_29= np.array(p_stu_29)
x_stu_30= np.array(x_stu_30)
y_stu_30= np.array(y_stu_30)
p_stu_30= np.array(p_stu_30)
x_stu_31= np.array(x_stu_31)
y_stu_31= np.array(y_stu_31)
p_stu_31= np.array(p_stu_31)
x_stu_32= np.array(x_stu_32)
y_stu_32= np.array(y_stu_32)
p_stu_32= np.array(p_stu_32)
x_stu_33= np.array(x_stu_33)
y_stu_33= np.array(y_stu_33)
p_stu_33= np.array(p_stu_33)
x_stu_34= np.array(x_stu_34)
y_stu_34= np.array(y_stu_34)
p_stu_34= np.array(p_stu_34)
x_stu_35= np.array(x_stu_35)
y_stu_35= np.array(y_stu_35)
p_stu_35= np.array(p_stu_35)
x_stu_36= np.array(x_stu_36)
y_stu_36= np.array(y_stu_36)
p_stu_36= np.array(p_stu_36)
x_stu_37= np.array(x_stu_37)
y_stu_37= np.array(y_stu_37)
p_stu_37= np.array(p_stu_37)
x_stu_38= np.array(x_stu_38)
y_stu_38= np.array(y_stu_38)
p_stu_38= np.array(p_stu_38)
x_stu_39= np.array(x_stu_39)
y_stu_39= np.array(y_stu_39)
p_stu_39= np.array(p_stu_39)
x_stu_40= np.array(x_stu_40)
y_stu_40= np.array(y_stu_40)
p_stu_40= np.array(p_stu_40)
x_stu_41= np.array(x_stu_41)
y_stu_41= np.array(y_stu_41)
p_stu_41= np.array(p_stu_41)
x_stu_42= np.array(x_stu_42)
y_stu_42= np.array(y_stu_42)
p_stu_42= np.array(p_stu_42)
x_stu_43= np.array(x_stu_43)
y_stu_43= np.array(y_stu_43)
p_stu_43= np.array(p_stu_43)
x_stu_44= np.array(x_stu_44)
y_stu_44= np.array(y_stu_44)
p_stu_44= np.array(p_stu_44)
x_stu_45= np.array(x_stu_45)
y_stu_45= np.array(y_stu_45)
p_stu_45= np.array(p_stu_45)
x_stu_46= np.array(x_stu_46)
y_stu_46= np.array(y_stu_46)
p_stu_46= np.array(p_stu_46)
x_stu_47= np.array(x_stu_47)
y_stu_47= np.array(y_stu_47)
p_stu_47= np.array(p_stu_47)
x_stu_48= np.array(x_stu_48)
y_stu_48= np.array(y_stu_48)
p_stu_48= np.array(p_stu_48)
x_stu_49= np.array(x_stu_49)
y_stu_49= np.array(y_stu_49)
p_stu_49= np.array(p_stu_49)
x_stu_50= np.array(x_stu_50)
y_stu_50= np.array(y_stu_50)
p_stu_50= np.array(p_stu_50)
x_stu_51= np.array(x_stu_51)
y_stu_51= np.array(y_stu_51)
p_stu_51= np.array(p_stu_51)
x_stu_52= np.array(x_stu_52)
y_stu_52= np.array(y_stu_52)
p_stu_52= np.array(p_stu_52)
x_stu_53= np.array(x_stu_53)
y_stu_53= np.array(y_stu_53)
p_stu_53= np.array(p_stu_53)
x_stu_54= np.array(x_stu_54)
y_stu_54= np.array(y_stu_54)
p_stu_54= np.array(p_stu_54)
x_stu_55= np.array(x_stu_55)
y_stu_55= np.array(y_stu_55)
p_stu_55= np.array(p_stu_55)
x_stu_56= np.array(x_stu_56)
y_stu_56= np.array(y_stu_56)
p_stu_56= np.array(p_stu_56)
x_stu_57= np.array(x_stu_57)
y_stu_57= np.array(y_stu_57)
p_stu_57= np.array(p_stu_57)
x_stu_58= np.array(x_stu_58)
y_stu_58= np.array(y_stu_58)
p_stu_58= np.array(p_stu_58)
x_stu_59= np.array(x_stu_59)
y_stu_59= np.array(y_stu_59)
p_stu_59= np.array(p_stu_59)
x_stu_60= np.array(x_stu_60)
y_stu_60= np.array(y_stu_60)
p_stu_60= np.array(p_stu_60)
x_stu_61= np.array(x_stu_61)
y_stu_61= np.array(y_stu_61)
p_stu_61= np.array(p_stu_61)
x_stu_62= np.array(x_stu_62)
y_stu_62= np.array(y_stu_62)
p_stu_62= np.array(p_stu_62)
x_stu_63= np.array(x_stu_63)
y_stu_63= np.array(y_stu_63)
p_stu_63= np.array(p_stu_63)
x_stu_64= np.array(x_stu_64)
y_stu_64= np.array(y_stu_64)
p_stu_64= np.array(p_stu_64)
x_stu_65= np.array(x_stu_65)
y_stu_65= np.array(y_stu_65)
p_stu_65= np.array(p_stu_65)
x_stu_66= np.array(x_stu_66)
y_stu_66= np.array(y_stu_66)
p_stu_66= np.array(p_stu_66)
x_stu_67= np.array(x_stu_67)
y_stu_67= np.array(y_stu_67)
p_stu_67= np.array(p_stu_67)
x_stu_68= np.array(x_stu_68)
y_stu_68= np.array(y_stu_68)
p_stu_68= np.array(p_stu_68)
x_stu_69= np.array(x_stu_69)
y_stu_69= np.array(y_stu_69)
p_stu_69= np.array(p_stu_69)
x_stu_70= np.array(x_stu_70)
y_stu_70= np.array(y_stu_70)
p_stu_70= np.array(p_stu_70)
x_stu_71= np.array(x_stu_71)
y_stu_71= np.array(y_stu_71)
p_stu_71= np.array(p_stu_71)
x_stu_72= np.array(x_stu_72)
y_stu_72= np.array(y_stu_72)
p_stu_72= np.array(p_stu_72)
x_stu_73= np.array(x_stu_73)
y_stu_73= np.array(y_stu_73)
p_stu_73= np.array(p_stu_73)
x_stu_74= np.array(x_stu_74)
y_stu_74= np.array(y_stu_74)
p_stu_74= np.array(p_stu_74)
x_stu_75= np.array(x_stu_75)
y_stu_75= np.array(y_stu_75)
p_stu_75= np.array(p_stu_75)
x_stu_76= np.array(x_stu_76)
y_stu_76= np.array(y_stu_76)
p_stu_76= np.array(p_stu_76)
x_stu_77= np.array(x_stu_77)
y_stu_77= np.array(y_stu_77)
p_stu_77= np.array(p_stu_77)
x_stu_78= np.array(x_stu_78)
y_stu_78= np.array(y_stu_78)
p_stu_78= np.array(p_stu_78)
x_stu_79= np.array(x_stu_79)
y_stu_79= np.array(y_stu_79)
p_stu_79= np.array(p_stu_79)
x_stu_80= np.array(x_stu_80)
y_stu_80= np.array(y_stu_80)
p_stu_80= np.array(p_stu_80)
x_stu_81= np.array(x_stu_81)
y_stu_81= np.array(y_stu_81)
p_stu_81= np.array(p_stu_81)
x_stu_82= np.array(x_stu_82)
y_stu_82= np.array(y_stu_82)
p_stu_82= np.array(p_stu_82)
x_stu_83= np.array(x_stu_83)
y_stu_83= np.array(y_stu_83)
p_stu_83= np.array(p_stu_83)
x_stu_84= np.array(x_stu_84)
y_stu_84= np.array(y_stu_84)
p_stu_84= np.array(p_stu_84)
x_stu_85= np.array(x_stu_85)
y_stu_85= np.array(y_stu_85)
p_stu_85= np.array(p_stu_85)
x_stu_86= np.array(x_stu_86)
y_stu_86= np.array(y_stu_86)
p_stu_86= np.array(p_stu_86)
x_stu_87= np.array(x_stu_87)
y_stu_87= np.array(y_stu_87)
p_stu_87= np.array(p_stu_87)
x_stu_88= np.array(x_stu_88)
y_stu_88= np.array(y_stu_88)
p_stu_88= np.array(p_stu_88)
x_stu_89= np.array(x_stu_89)
y_stu_89= np.array(y_stu_89)
p_stu_89= np.array(p_stu_89)
x_stu_90= np.array(x_stu_90)
y_stu_90= np.array(y_stu_90)
p_stu_90= np.array(p_stu_90)
x_stu_91= np.array(x_stu_91)
y_stu_91= np.array(y_stu_91)
p_stu_91= np.array(p_stu_91)
x_stu_92= np.array(x_stu_92)
y_stu_92= np.array(y_stu_92)
p_stu_92= np.array(p_stu_92)
x_stu_93= np.array(x_stu_93)
y_stu_93= np.array(y_stu_93)
p_stu_93= np.array(p_stu_93)
x_stu_94= np.array(x_stu_94)
y_stu_94= np.array(y_stu_94)
p_stu_94= np.array(p_stu_94)
x_stu_95= np.array(x_stu_95)
y_stu_95= np.array(y_stu_95)
p_stu_95= np.array(p_stu_95)
x_stu_96= np.array(x_stu_96)
y_stu_96= np.array(y_stu_96)
p_stu_96= np.array(p_stu_96)
x_stu_97= np.array(x_stu_97)
y_stu_97= np.array(y_stu_97)
p_stu_97= np.array(p_stu_97)
x_stu_98= np.array(x_stu_98)
y_stu_98= np.array(y_stu_98)
p_stu_98= np.array(p_stu_98)
x_stu_99= np.array(x_stu_99)
y_stu_99= np.array(y_stu_99)
p_stu_99= np.array(p_stu_99)
x_std_0= np.array(x_std_0)
y_std_0= np.array(y_std_0)
p_std_0= np.array(p_std_0)
x_std_1= np.array(x_std_1)
y_std_1= np.array(y_std_1)
p_std_1= np.array(p_std_1)
x_std_2= np.array(x_std_2)
y_std_2= np.array(y_std_2)
p_std_2= np.array(p_std_2)
x_std_3= np.array(x_std_3)
y_std_3= np.array(y_std_3)
p_std_3= np.array(p_std_3)
x_std_4= np.array(x_std_4)
y_std_4= np.array(y_std_4)
p_std_4= np.array(p_std_4)
x_std_5= np.array(x_std_5)
y_std_5= np.array(y_std_5)
p_std_5= np.array(p_std_5)
x_std_6= np.array(x_std_6)
y_std_6= np.array(y_std_6)
p_std_6= np.array(p_std_6)
x_std_7= np.array(x_std_7)
y_std_7= np.array(y_std_7)
p_std_7= np.array(p_std_7)
x_std_8= np.array(x_std_8)
y_std_8= np.array(y_std_8)
p_std_8= np.array(p_std_8)
x_std_9= np.array(x_std_9)
y_std_9= np.array(y_std_9)
p_std_9= np.array(p_std_9)
x_std_10= np.array(x_std_10)
y_std_10= np.array(y_std_10)
p_std_10= np.array(p_std_10)
x_std_11= np.array(x_std_11)
y_std_11= np.array(y_std_11)
p_std_11= np.array(p_std_11)
x_std_12= np.array(x_std_12)
y_std_12= np.array(y_std_12)
p_std_12= np.array(p_std_12)
x_std_13= np.array(x_std_13)
y_std_13= np.array(y_std_13)
p_std_13= np.array(p_std_13)
x_std_14= np.array(x_std_14)
y_std_14= np.array(y_std_14)
p_std_14= np.array(p_std_14)
x_std_15= np.array(x_std_15)
y_std_15= np.array(y_std_15)
p_std_15= np.array(p_std_15)
x_std_16= np.array(x_std_16)
y_std_16= np.array(y_std_16)
p_std_16= np.array(p_std_16)
x_std_17= np.array(x_std_17)
y_std_17= np.array(y_std_17)
p_std_17= np.array(p_std_17)
x_std_18= np.array(x_std_18)
y_std_18= np.array(y_std_18)
p_std_18= np.array(p_std_18)
x_std_19= np.array(x_std_19)
y_std_19= np.array(y_std_19)
p_std_19= np.array(p_std_19)
x_std_20= np.array(x_std_20)
y_std_20= np.array(y_std_20)
p_std_20= np.array(p_std_20)
x_std_21= np.array(x_std_21)
y_std_21= np.array(y_std_21)
p_std_21= np.array(p_std_21)
x_std_22= np.array(x_std_22)
y_std_22= np.array(y_std_22)
p_std_22= np.array(p_std_22)
x_std_23= np.array(x_std_23)
y_std_23= np.array(y_std_23)
p_std_23= np.array(p_std_23)
x_std_24= np.array(x_std_24)
y_std_24= np.array(y_std_24)
p_std_24= np.array(p_std_24)
x_std_25= np.array(x_std_25)
y_std_25= np.array(y_std_25)
p_std_25= np.array(p_std_25)
x_std_26= np.array(x_std_26)
y_std_26= np.array(y_std_26)
p_std_26= np.array(p_std_26)
x_std_27= np.array(x_std_27)
y_std_27= np.array(y_std_27)
p_std_27= np.array(p_std_27)
x_std_28= np.array(x_std_28)
y_std_28= np.array(y_std_28)
p_std_28= np.array(p_std_28)
x_std_29= np.array(x_std_29)
y_std_29= np.array(y_std_29)
p_std_29= np.array(p_std_29)
x_std_30= np.array(x_std_30)
y_std_30= np.array(y_std_30)
p_std_30= np.array(p_std_30)
x_std_31= np.array(x_std_31)
y_std_31= np.array(y_std_31)
p_std_31= np.array(p_std_31)
x_std_32= np.array(x_std_32)
y_std_32= np.array(y_std_32)
p_std_32= np.array(p_std_32)
x_std_33= np.array(x_std_33)
y_std_33= np.array(y_std_33)
p_std_33= np.array(p_std_33)
x_std_34= np.array(x_std_34)
y_std_34= np.array(y_std_34)
p_std_34= np.array(p_std_34)
x_std_35= np.array(x_std_35)
y_std_35= np.array(y_std_35)
p_std_35= np.array(p_std_35)
x_std_36= np.array(x_std_36)
y_std_36= np.array(y_std_36)
p_std_36= np.array(p_std_36)
x_std_37= np.array(x_std_37)
y_std_37= np.array(y_std_37)
p_std_37= np.array(p_std_37)
x_std_38= np.array(x_std_38)
y_std_38= np.array(y_std_38)
p_std_38= np.array(p_std_38)
x_std_39= np.array(x_std_39)
y_std_39= np.array(y_std_39)
p_std_39= np.array(p_std_39)
x_std_40= np.array(x_std_40)
y_std_40= np.array(y_std_40)
p_std_40= np.array(p_std_40)
x_std_41= np.array(x_std_41)
y_std_41= np.array(y_std_41)
p_std_41= np.array(p_std_41)
x_std_42= np.array(x_std_42)
y_std_42= np.array(y_std_42)
p_std_42= np.array(p_std_42)
x_std_43= np.array(x_std_43)
y_std_43= np.array(y_std_43)
p_std_43= np.array(p_std_43)
x_std_44= np.array(x_std_44)
y_std_44= np.array(y_std_44)
p_std_44= np.array(p_std_44)
x_std_45= np.array(x_std_45)
y_std_45= np.array(y_std_45)
p_std_45= np.array(p_std_45)
x_std_46= np.array(x_std_46)
y_std_46= np.array(y_std_46)
p_std_46= np.array(p_std_46)
x_std_47= np.array(x_std_47)
y_std_47= np.array(y_std_47)
p_std_47= np.array(p_std_47)
x_std_48= np.array(x_std_48)
y_std_48= np.array(y_std_48)
p_std_48= np.array(p_std_48)
x_std_49= np.array(x_std_49)
y_std_49= np.array(y_std_49)
p_std_49= np.array(p_std_49)
x_std_50= np.array(x_std_50)
y_std_50= np.array(y_std_50)
p_std_50= np.array(p_std_50)
x_std_51= np.array(x_std_51)
y_std_51= np.array(y_std_51)
p_std_51= np.array(p_std_51)
x_std_52= np.array(x_std_52)
y_std_52= np.array(y_std_52)
p_std_52= np.array(p_std_52)
x_std_53= np.array(x_std_53)
y_std_53= np.array(y_std_53)
p_std_53= np.array(p_std_53)
x_std_54= np.array(x_std_54)
y_std_54= np.array(y_std_54)
p_std_54= np.array(p_std_54)
x_std_55= np.array(x_std_55)
y_std_55= np.array(y_std_55)
p_std_55= np.array(p_std_55)
x_std_56= np.array(x_std_56)
y_std_56= np.array(y_std_56)
p_std_56= np.array(p_std_56)
x_std_57= np.array(x_std_57)
y_std_57= np.array(y_std_57)
p_std_57= np.array(p_std_57)
x_std_58= np.array(x_std_58)
y_std_58= np.array(y_std_58)
p_std_58= np.array(p_std_58)
x_std_59= np.array(x_std_59)
y_std_59= np.array(y_std_59)
p_std_59= np.array(p_std_59)
x_std_60= np.array(x_std_60)
y_std_60= np.array(y_std_60)
p_std_60= np.array(p_std_60)
x_std_61= np.array(x_std_61)
y_std_61= np.array(y_std_61)
p_std_61= np.array(p_std_61)
x_std_62= np.array(x_std_62)
y_std_62= np.array(y_std_62)
p_std_62= np.array(p_std_62)
x_std_63= np.array(x_std_63)
y_std_63= np.array(y_std_63)
p_std_63= np.array(p_std_63)
x_std_64= np.array(x_std_64)
y_std_64= np.array(y_std_64)
p_std_64= np.array(p_std_64)
x_std_65= np.array(x_std_65)
y_std_65= np.array(y_std_65)
p_std_65= np.array(p_std_65)
x_std_66= np.array(x_std_66)
y_std_66= np.array(y_std_66)
p_std_66= np.array(p_std_66)
x_std_67= np.array(x_std_67)
y_std_67= np.array(y_std_67)
p_std_67= np.array(p_std_67)
x_std_68= np.array(x_std_68)
y_std_68= np.array(y_std_68)
p_std_68= np.array(p_std_68)
x_std_69= np.array(x_std_69)
y_std_69= np.array(y_std_69)
p_std_69= np.array(p_std_69)
x_std_70= np.array(x_std_70)
y_std_70= np.array(y_std_70)
p_std_70= np.array(p_std_70)
x_std_71= np.array(x_std_71)
y_std_71= np.array(y_std_71)
p_std_71= np.array(p_std_71)
x_std_72= np.array(x_std_72)
y_std_72= np.array(y_std_72)
p_std_72= np.array(p_std_72)
x_std_73= np.array(x_std_73)
y_std_73= np.array(y_std_73)
p_std_73= np.array(p_std_73)
x_std_74= np.array(x_std_74)
y_std_74= np.array(y_std_74)
p_std_74= np.array(p_std_74)
x_std_75= np.array(x_std_75)
y_std_75= np.array(y_std_75)
p_std_75= np.array(p_std_75)
x_std_76= np.array(x_std_76)
y_std_76= np.array(y_std_76)
p_std_76= np.array(p_std_76)
x_std_77= np.array(x_std_77)
y_std_77= np.array(y_std_77)
p_std_77= np.array(p_std_77)
x_std_78= np.array(x_std_78)
y_std_78= np.array(y_std_78)
p_std_78= np.array(p_std_78)
x_std_79= np.array(x_std_79)
y_std_79= np.array(y_std_79)
p_std_79= np.array(p_std_79)
x_std_80= np.array(x_std_80)
y_std_80= np.array(y_std_80)
p_std_80= np.array(p_std_80)
x_std_81= np.array(x_std_81)
y_std_81= np.array(y_std_81)
p_std_81= np.array(p_std_81)
x_std_82= np.array(x_std_82)
y_std_82= np.array(y_std_82)
p_std_82= np.array(p_std_82)
x_std_83= np.array(x_std_83)
y_std_83= np.array(y_std_83)
p_std_83= np.array(p_std_83)
x_std_84= np.array(x_std_84)
y_std_84= np.array(y_std_84)
p_std_84= np.array(p_std_84)
x_std_85= np.array(x_std_85)
y_std_85= np.array(y_std_85)
p_std_85= np.array(p_std_85)
x_std_86= np.array(x_std_86)
y_std_86= np.array(y_std_86)
p_std_86= np.array(p_std_86)
x_std_87= np.array(x_std_87)
y_std_87= np.array(y_std_87)
p_std_87= np.array(p_std_87)
x_std_88= np.array(x_std_88)
y_std_88= np.array(y_std_88)
p_std_88= np.array(p_std_88)
x_std_89= np.array(x_std_89)
y_std_89= np.array(y_std_89)
p_std_89= np.array(p_std_89)
x_std_90= np.array(x_std_90)
y_std_90= np.array(y_std_90)
p_std_90= np.array(p_std_90)
x_std_91= np.array(x_std_91)
y_std_91= np.array(y_std_91)
p_std_91= np.array(p_std_91)
x_std_92= np.array(x_std_92)
y_std_92= np.array(y_std_92)
p_std_92= np.array(p_std_92)
x_std_93= np.array(x_std_93)
y_std_93= np.array(y_std_93)
p_std_93= np.array(p_std_93)
x_std_94= np.array(x_std_94)
y_std_94= np.array(y_std_94)
p_std_94= np.array(p_std_94)
x_std_95= np.array(x_std_95)
y_std_95= np.array(y_std_95)
p_std_95= np.array(p_std_95)
x_std_96= np.array(x_std_96)
y_std_96= np.array(y_std_96)
p_std_96= np.array(p_std_96)
x_std_97= np.array(x_std_97)
y_std_97= np.array(y_std_97)
p_std_97= np.array(p_std_97)
x_std_98= np.array(x_std_98)
y_std_98= np.array(y_std_98)
p_std_98= np.array(p_std_98)
x_std_99= np.array(x_std_99)
y_std_99= np.array(y_std_99)
p_std_99= np.array(p_std_99)
x_st_0= np.array(x_st_0)
y_st_0= np.array(y_st_0)
p_st_0= np.array(p_st_0)
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
x_st_37= np.array(x_st_37)
y_st_37= np.array(y_st_37)
p_st_37= np.array(p_st_37)
x_st_38= np.array(x_st_38)
y_st_38= np.array(y_st_38)
p_st_38= np.array(p_st_38)
x_st_39= np.array(x_st_39)
y_st_39= np.array(y_st_39)
p_st_39= np.array(p_st_39)
x_st_40= np.array(x_st_40)
y_st_40= np.array(y_st_40)
p_st_40= np.array(p_st_40)
x_st_41= np.array(x_st_41)
y_st_41= np.array(y_st_41)
p_st_41= np.array(p_st_41)
x_st_42= np.array(x_st_42)
y_st_42= np.array(y_st_42)
p_st_42= np.array(p_st_42)
x_st_43= np.array(x_st_43)
y_st_43= np.array(y_st_43)
p_st_43= np.array(p_st_43)
x_st_44= np.array(x_st_44)
y_st_44= np.array(y_st_44)
p_st_44= np.array(p_st_44)
x_st_45= np.array(x_st_45)
y_st_45= np.array(y_st_45)
p_st_45= np.array(p_st_45)
x_st_46= np.array(x_st_46)
y_st_46= np.array(y_st_46)
p_st_46= np.array(p_st_46)
x_st_47= np.array(x_st_47)
y_st_47= np.array(y_st_47)
p_st_47= np.array(p_st_47)
x_st_48= np.array(x_st_48)
y_st_48= np.array(y_st_48)
p_st_48= np.array(p_st_48)
x_st_49= np.array(x_st_49)
y_st_49= np.array(y_st_49)
p_st_49= np.array(p_st_49)
x_st_50= np.array(x_st_50)
y_st_50= np.array(y_st_50)
p_st_50= np.array(p_st_50)
x_st_51= np.array(x_st_51)
y_st_51= np.array(y_st_51)
p_st_51= np.array(p_st_51)
x_st_52= np.array(x_st_52)
y_st_52= np.array(y_st_52)
p_st_52= np.array(p_st_52)
x_st_53= np.array(x_st_53)
y_st_53= np.array(y_st_53)
p_st_53= np.array(p_st_53)
x_st_54= np.array(x_st_54)
y_st_54= np.array(y_st_54)
p_st_54= np.array(p_st_54)
x_st_55= np.array(x_st_55)
y_st_55= np.array(y_st_55)
p_st_55= np.array(p_st_55)
x_st_56= np.array(x_st_56)
y_st_56= np.array(y_st_56)
p_st_56= np.array(p_st_56)
x_st_57= np.array(x_st_57)
y_st_57= np.array(y_st_57)
p_st_57= np.array(p_st_57)
x_st_58= np.array(x_st_58)
y_st_58= np.array(y_st_58)
p_st_58= np.array(p_st_58)
x_st_59= np.array(x_st_59)
y_st_59= np.array(y_st_59)
p_st_59= np.array(p_st_59)
x_st_60= np.array(x_st_60)
y_st_60= np.array(y_st_60)
p_st_60= np.array(p_st_60)
x_st_61= np.array(x_st_61)
y_st_61= np.array(y_st_61)
p_st_61= np.array(p_st_61)
x_st_62= np.array(x_st_62)
y_st_62= np.array(y_st_62)
p_st_62= np.array(p_st_62)
x_st_63= np.array(x_st_63)
y_st_63= np.array(y_st_63)
p_st_63= np.array(p_st_63)
x_st_64= np.array(x_st_64)
y_st_64= np.array(y_st_64)
p_st_64= np.array(p_st_64)
x_st_65= np.array(x_st_65)
y_st_65= np.array(y_st_65)
p_st_65= np.array(p_st_65)
x_st_66= np.array(x_st_66)
y_st_66= np.array(y_st_66)
p_st_66= np.array(p_st_66)
x_st_67= np.array(x_st_67)
y_st_67= np.array(y_st_67)
p_st_67= np.array(p_st_67)
x_st_68= np.array(x_st_68)
y_st_68= np.array(y_st_68)
p_st_68= np.array(p_st_68)
x_st_69= np.array(x_st_69)
y_st_69= np.array(y_st_69)
p_st_69= np.array(p_st_69)
x_st_70= np.array(x_st_70)
y_st_70= np.array(y_st_70)
p_st_70= np.array(p_st_70)
x_st_71= np.array(x_st_71)
y_st_71= np.array(y_st_71)
p_st_71= np.array(p_st_71)
x_st_72= np.array(x_st_72)
y_st_72= np.array(y_st_72)
p_st_72= np.array(p_st_72)
x_st_73= np.array(x_st_73)
y_st_73= np.array(y_st_73)
p_st_73= np.array(p_st_73)
x_st_74= np.array(x_st_74)
y_st_74= np.array(y_st_74)
p_st_74= np.array(p_st_74)
x_st_75= np.array(x_st_75)
y_st_75= np.array(y_st_75)
p_st_75= np.array(p_st_75)
x_st_76= np.array(x_st_76)
y_st_76= np.array(y_st_76)
p_st_76= np.array(p_st_76)
x_st_77= np.array(x_st_77)
y_st_77= np.array(y_st_77)
p_st_77= np.array(p_st_77)
x_st_78= np.array(x_st_78)
y_st_78= np.array(y_st_78)
p_st_78= np.array(p_st_78)
x_st_79= np.array(x_st_79)
y_st_79= np.array(y_st_79)
p_st_79= np.array(p_st_79)
x_st_80= np.array(x_st_80)
y_st_80= np.array(y_st_80)
p_st_80= np.array(p_st_80)
x_st_81= np.array(x_st_81)
y_st_81= np.array(y_st_81)
p_st_81= np.array(p_st_81)
x_st_82= np.array(x_st_82)
y_st_82= np.array(y_st_82)
p_st_82= np.array(p_st_82)
x_st_83= np.array(x_st_83)
y_st_83= np.array(y_st_83)
p_st_83= np.array(p_st_83)
x_st_84= np.array(x_st_84)
y_st_84= np.array(y_st_84)
p_st_84= np.array(p_st_84)
x_st_85= np.array(x_st_85)
y_st_85= np.array(y_st_85)
p_st_85= np.array(p_st_85)
x_st_86= np.array(x_st_86)
y_st_86= np.array(y_st_86)
p_st_86= np.array(p_st_86)
x_st_87= np.array(x_st_87)
y_st_87= np.array(y_st_87)
p_st_87= np.array(p_st_87)
x_st_88= np.array(x_st_88)
y_st_88= np.array(y_st_88)
p_st_88= np.array(p_st_88)
x_st_89= np.array(x_st_89)
y_st_89= np.array(y_st_89)
p_st_89= np.array(p_st_89)
x_st_90= np.array(x_st_90)
y_st_90= np.array(y_st_90)
p_st_90= np.array(p_st_90)
x_st_91= np.array(x_st_91)
y_st_91= np.array(y_st_91)
p_st_91= np.array(p_st_91)
x_st_92= np.array(x_st_92)
y_st_92= np.array(y_st_92)
p_st_92= np.array(p_st_92)
x_st_93= np.array(x_st_93)
y_st_93= np.array(y_st_93)
p_st_93= np.array(p_st_93)
x_st_94= np.array(x_st_94)
y_st_94= np.array(y_st_94)
p_st_94= np.array(p_st_94)
x_st_95= np.array(x_st_95)
y_st_95= np.array(y_st_95)
p_st_95= np.array(p_st_95)
x_st_96= np.array(x_st_96)
y_st_96= np.array(y_st_96)
p_st_96= np.array(p_st_96)
x_st_97= np.array(x_st_97)
y_st_97= np.array(y_st_97)
p_st_97= np.array(p_st_97)
x_st_98= np.array(x_st_98)
y_st_98= np.array(y_st_98)
p_st_98= np.array(p_st_98)
x_st_99= np.array(x_st_99)
y_st_99= np.array(y_st_99)
p_st_99= np.array(p_st_99)

shp=(x_j_0.shape)[0]
shape_y = y_j_0.shape[0]
shape_p =  p_j_0.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_j_0 =  np.delete(x_j_0, [k for k in range(nece_shp,shp)], None)
x_j_0 = x_j_0.reshape(-1,sensors)
shape_x_j_0 = x_j_0.shape[0]
x_j_0 = preprocessing.normalize(x_j_0, axis=0)
x_j_0= np.reshape(x_j_0, (-1,segement_time_size, sensors))
y_j_0 = np.delete(y_j_0, [k for k in range(x_j_0.shape[0],shape_y)], None)
p_j_0 = np.delete(p_j_0,[k for k in range(x_j_0.shape[0],shape_p)], None)
y= np.concatenate((y,y_j_0), axis=0)
p= np.concatenate((p,p_j_0), axis=0)
print(y)
print(p)

shp=(x_j_1.shape)[0]
shape_y = y_j_1.shape[0]
shape_p =  p_j_1.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_j_1 =  np.delete(x_j_1, [k for k in range(nece_shp,shp)], None)
x_j_1 = x_j_1.reshape(-1,sensors)
shape_x_j_1 = x_j_1.shape[0]
x_j_1 = preprocessing.normalize(x_j_1, axis=0)
x_j_1= np.reshape(x_j_1, (-1,segement_time_size, sensors))
y_j_1= np.delete( y_j_1, [k for k in range(x_j_1.shape[0],shape_y)], None)
p_j_1 = np.delete(p_j_1,[k for k in range(x_j_1.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_j_1), axis=0)
y= np.concatenate((y,y_j_1), axis=0)
p= np.concatenate((p,p_j_1), axis=0)
shp=(x_j_2.shape)[0]
shape_y = y_j_2.shape[0]
shape_p =  p_j_2.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_j_2 =  np.delete(x_j_2, [k for k in range(nece_shp,shp)], None)
x_j_2 = x_j_2.reshape(-1,sensors)
shape_x_j_2 = x_j_2.shape[0]
x_j_2 = preprocessing.normalize(x_j_2, axis=0)
x_j_2= np.reshape(x_j_2, (-1,segement_time_size, sensors))
y_j_2= np.delete( y_j_2, [k for k in range(x_j_2.shape[0],shape_y)], None)
p_j_2 = np.delete(p_j_2,[k for k in range(x_j_2.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_j_2), axis=0)
y= np.concatenate((y,y_j_2), axis=0)
p= np.concatenate((p,p_j_2), axis=0)
shp=(x_j_3.shape)[0]
shape_y = y_j_3.shape[0]
shape_p =  p_j_3.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_j_3 =  np.delete(x_j_3, [k for k in range(nece_shp,shp)], None)
x_j_3 = x_j_3.reshape(-1,sensors)
shape_x_j_3 = x_j_3.shape[0]
x_j_3 = preprocessing.normalize(x_j_3, axis=0)
x_j_3= np.reshape(x_j_3, (-1,segement_time_size, sensors))
y_j_3= np.delete( y_j_3, [k for k in range(x_j_3.shape[0],shape_y)], None)
p_j_3 = np.delete(p_j_3,[k for k in range(x_j_3.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_j_3), axis=0)
y= np.concatenate((y,y_j_3), axis=0)
p= np.concatenate((p,p_j_3), axis=0)
shp=(x_j_4.shape)[0]
shape_y = y_j_4.shape[0]
shape_p =  p_j_4.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_j_4 =  np.delete(x_j_4, [k for k in range(nece_shp,shp)], None)
x_j_4 = x_j_4.reshape(-1,sensors)
shape_x_j_4 = x_j_4.shape[0]
x_j_4 = preprocessing.normalize(x_j_4, axis=0)
x_j_4= np.reshape(x_j_4, (-1,segement_time_size, sensors))
y_j_4= np.delete( y_j_4, [k for k in range(x_j_4.shape[0],shape_y)], None)
p_j_4 = np.delete(p_j_4,[k for k in range(x_j_4.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_j_4), axis=0)
y= np.concatenate((y,y_j_4), axis=0)
p= np.concatenate((p,p_j_4), axis=0)
shp=(x_j_5.shape)[0]
shape_y = y_j_5.shape[0]
shape_p =  p_j_5.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_j_5 =  np.delete(x_j_5, [k for k in range(nece_shp,shp)], None)
x_j_5 = x_j_5.reshape(-1,sensors)
shape_x_j_5 = x_j_5.shape[0]
x_j_5 = preprocessing.normalize(x_j_5, axis=0)
x_j_5= np.reshape(x_j_5, (-1,segement_time_size, sensors))
y_j_5= np.delete( y_j_5, [k for k in range(x_j_5.shape[0],shape_y)], None)
p_j_5 = np.delete(p_j_5,[k for k in range(x_j_5.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_j_5), axis=0)
y= np.concatenate((y,y_j_5), axis=0)
p= np.concatenate((p,p_j_5), axis=0)
shp=(x_j_6.shape)[0]
shape_y = y_j_6.shape[0]
shape_p =  p_j_6.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_j_6 =  np.delete(x_j_6, [k for k in range(nece_shp,shp)], None)
x_j_6 = x_j_6.reshape(-1,sensors)
shape_x_j_6 = x_j_6.shape[0]
x_j_6 = preprocessing.normalize(x_j_6, axis=0)
x_j_6= np.reshape(x_j_6, (-1,segement_time_size, sensors))
y_j_6= np.delete( y_j_6, [k for k in range(x_j_6.shape[0],shape_y)], None)
p_j_6 = np.delete(p_j_6,[k for k in range(x_j_6.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_j_6), axis=0)
y= np.concatenate((y,y_j_6), axis=0)
p= np.concatenate((p,p_j_6), axis=0)
shp=(x_j_7.shape)[0]
shape_y = y_j_7.shape[0]
shape_p =  p_j_7.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_j_7 =  np.delete(x_j_7, [k for k in range(nece_shp,shp)], None)
x_j_7 = x_j_7.reshape(-1,sensors)
shape_x_j_7 = x_j_7.shape[0]
x_j_7 = preprocessing.normalize(x_j_7, axis=0)
x_j_7= np.reshape(x_j_7, (-1,segement_time_size, sensors))
y_j_7= np.delete( y_j_7, [k for k in range(x_j_7.shape[0],shape_y)], None)
p_j_7 = np.delete(p_j_7,[k for k in range(x_j_7.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_j_7), axis=0)
y= np.concatenate((y,y_j_7), axis=0)
p= np.concatenate((p,p_j_7), axis=0)
shp=(x_j_8.shape)[0]
shape_y = y_j_8.shape[0]
shape_p =  p_j_8.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_j_8 =  np.delete(x_j_8, [k for k in range(nece_shp,shp)], None)
x_j_8 = x_j_8.reshape(-1,sensors)
shape_x_j_8 = x_j_8.shape[0]
x_j_8 = preprocessing.normalize(x_j_8, axis=0)
x_j_8= np.reshape(x_j_8, (-1,segement_time_size, sensors))
y_j_8= np.delete( y_j_8, [k for k in range(x_j_8.shape[0],shape_y)], None)
p_j_8 = np.delete(p_j_8,[k for k in range(x_j_8.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_j_8), axis=0)
y= np.concatenate((y,y_j_8), axis=0)
p= np.concatenate((p,p_j_8), axis=0)
shp=(x_j_9.shape)[0]
shape_y = y_j_9.shape[0]
shape_p =  p_j_9.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_j_9 =  np.delete(x_j_9, [k for k in range(nece_shp,shp)], None)
x_j_9 = x_j_9.reshape(-1,sensors)
shape_x_j_9 = x_j_9.shape[0]
x_j_9 = preprocessing.normalize(x_j_9, axis=0)
x_j_9= np.reshape(x_j_9, (-1,segement_time_size, sensors))
y_j_9= np.delete( y_j_9, [k for k in range(x_j_9.shape[0],shape_y)], None)
p_j_9 = np.delete(p_j_9,[k for k in range(x_j_9.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_j_9), axis=0)
y= np.concatenate((y,y_j_9), axis=0)
p= np.concatenate((p,p_j_9), axis=0)
shp=(x_j_10.shape)[0]
shape_y = y_j_10.shape[0]
shape_p =  p_j_10.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_j_10 =  np.delete(x_j_10, [k for k in range(nece_shp,shp)], None)
x_j_10 = x_j_10.reshape(-1,sensors)
shape_x_j_10 = x_j_10.shape[0]
x_j_10 = preprocessing.normalize(x_j_10, axis=0)
x_j_10= np.reshape(x_j_10, (-1,segement_time_size, sensors))
y_j_10= np.delete( y_j_10, [k for k in range(x_j_10.shape[0],shape_y)], None)
p_j_10 = np.delete(p_j_10,[k for k in range(x_j_10.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_j_10), axis=0)
y= np.concatenate((y,y_j_10), axis=0)
p= np.concatenate((p,p_j_10), axis=0)
shp=(x_j_11.shape)[0]
shape_y = y_j_11.shape[0]
shape_p =  p_j_11.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_j_11 =  np.delete(x_j_11, [k for k in range(nece_shp,shp)], None)
x_j_11 = x_j_11.reshape(-1,sensors)
shape_x_j_11 = x_j_11.shape[0]
x_j_11 = preprocessing.normalize(x_j_11, axis=0)
x_j_11= np.reshape(x_j_11, (-1,segement_time_size, sensors))
y_j_11= np.delete( y_j_11, [k for k in range(x_j_11.shape[0],shape_y)], None)
p_j_11 = np.delete(p_j_11,[k for k in range(x_j_11.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_j_11), axis=0)
y= np.concatenate((y,y_j_11), axis=0)
p= np.concatenate((p,p_j_11), axis=0)
shp=(x_j_12.shape)[0]
shape_y = y_j_12.shape[0]
shape_p =  p_j_12.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_j_12 =  np.delete(x_j_12, [k for k in range(nece_shp,shp)], None)
x_j_12 = x_j_12.reshape(-1,sensors)
shape_x_j_12 = x_j_12.shape[0]
x_j_12 = preprocessing.normalize(x_j_12, axis=0)
x_j_12= np.reshape(x_j_12, (-1,segement_time_size, sensors))
y_j_12= np.delete( y_j_12, [k for k in range(x_j_12.shape[0],shape_y)], None)
p_j_12 = np.delete(p_j_12,[k for k in range(x_j_12.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_j_12), axis=0)
y= np.concatenate((y,y_j_12), axis=0)
p= np.concatenate((p,p_j_12), axis=0)
shp=(x_j_13.shape)[0]
shape_y = y_j_13.shape[0]
shape_p =  p_j_13.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_j_13 =  np.delete(x_j_13, [k for k in range(nece_shp,shp)], None)
x_j_13 = x_j_13.reshape(-1,sensors)
shape_x_j_13 = x_j_13.shape[0]
x_j_13 = preprocessing.normalize(x_j_13, axis=0)
x_j_13= np.reshape(x_j_13, (-1,segement_time_size, sensors))
y_j_13= np.delete( y_j_13, [k for k in range(x_j_13.shape[0],shape_y)], None)
p_j_13 = np.delete(p_j_13,[k for k in range(x_j_13.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_j_13), axis=0)
y= np.concatenate((y,y_j_13), axis=0)
p= np.concatenate((p,p_j_13), axis=0)
shp=(x_j_14.shape)[0]
shape_y = y_j_14.shape[0]
shape_p =  p_j_14.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_j_14 =  np.delete(x_j_14, [k for k in range(nece_shp,shp)], None)
x_j_14 = x_j_14.reshape(-1,sensors)
shape_x_j_14 = x_j_14.shape[0]
x_j_14 = preprocessing.normalize(x_j_14, axis=0)
x_j_14= np.reshape(x_j_14, (-1,segement_time_size, sensors))
y_j_14= np.delete( y_j_14, [k for k in range(x_j_14.shape[0],shape_y)], None)
p_j_14 = np.delete(p_j_14,[k for k in range(x_j_14.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_j_14), axis=0)
y= np.concatenate((y,y_j_14), axis=0)
p= np.concatenate((p,p_j_14), axis=0)
shp=(x_j_15.shape)[0]
shape_y = y_j_15.shape[0]
shape_p =  p_j_15.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_j_15 =  np.delete(x_j_15, [k for k in range(nece_shp,shp)], None)
x_j_15 = x_j_15.reshape(-1,sensors)
shape_x_j_15 = x_j_15.shape[0]
x_j_15 = preprocessing.normalize(x_j_15, axis=0)
x_j_15= np.reshape(x_j_15, (-1,segement_time_size, sensors))
y_j_15= np.delete( y_j_15, [k for k in range(x_j_15.shape[0],shape_y)], None)
p_j_15 = np.delete(p_j_15,[k for k in range(x_j_15.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_j_15), axis=0)
y= np.concatenate((y,y_j_15), axis=0)
p= np.concatenate((p,p_j_15), axis=0)
shp=(x_j_16.shape)[0]
shape_y = y_j_16.shape[0]
shape_p =  p_j_16.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_j_16 =  np.delete(x_j_16, [k for k in range(nece_shp,shp)], None)
x_j_16 = x_j_16.reshape(-1,sensors)
shape_x_j_16 = x_j_16.shape[0]
x_j_16 = preprocessing.normalize(x_j_16, axis=0)
x_j_16= np.reshape(x_j_16, (-1,segement_time_size, sensors))
y_j_16= np.delete( y_j_16, [k for k in range(x_j_16.shape[0],shape_y)], None)
p_j_16 = np.delete(p_j_16,[k for k in range(x_j_16.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_j_16), axis=0)
y= np.concatenate((y,y_j_16), axis=0)
p= np.concatenate((p,p_j_16), axis=0)
shp=(x_j_17.shape)[0]
shape_y = y_j_17.shape[0]
shape_p =  p_j_17.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_j_17 =  np.delete(x_j_17, [k for k in range(nece_shp,shp)], None)
x_j_17 = x_j_17.reshape(-1,sensors)
shape_x_j_17 = x_j_17.shape[0]
x_j_17 = preprocessing.normalize(x_j_17, axis=0)
x_j_17= np.reshape(x_j_17, (-1,segement_time_size, sensors))
y_j_17= np.delete( y_j_17, [k for k in range(x_j_17.shape[0],shape_y)], None)
p_j_17 = np.delete(p_j_17,[k for k in range(x_j_17.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_j_17), axis=0)
y= np.concatenate((y,y_j_17), axis=0)
p= np.concatenate((p,p_j_17), axis=0)
shp=(x_j_18.shape)[0]
shape_y = y_j_18.shape[0]
shape_p =  p_j_18.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_j_18 =  np.delete(x_j_18, [k for k in range(nece_shp,shp)], None)
x_j_18 = x_j_18.reshape(-1,sensors)
shape_x_j_18 = x_j_18.shape[0]
x_j_18 = preprocessing.normalize(x_j_18, axis=0)
x_j_18= np.reshape(x_j_18, (-1,segement_time_size, sensors))
y_j_18= np.delete( y_j_18, [k for k in range(x_j_18.shape[0],shape_y)], None)
p_j_18 = np.delete(p_j_18,[k for k in range(x_j_18.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_j_18), axis=0)
y= np.concatenate((y,y_j_18), axis=0)
p= np.concatenate((p,p_j_18), axis=0)
shp=(x_j_19.shape)[0]
shape_y = y_j_19.shape[0]
shape_p =  p_j_19.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_j_19 =  np.delete(x_j_19, [k for k in range(nece_shp,shp)], None)
x_j_19 = x_j_19.reshape(-1,sensors)
shape_x_j_19 = x_j_19.shape[0]
x_j_19 = preprocessing.normalize(x_j_19, axis=0)
x_j_19= np.reshape(x_j_19, (-1,segement_time_size, sensors))
y_j_19= np.delete( y_j_19, [k for k in range(x_j_19.shape[0],shape_y)], None)
p_j_19 = np.delete(p_j_19,[k for k in range(x_j_19.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_j_19), axis=0)
y= np.concatenate((y,y_j_19), axis=0)
p= np.concatenate((p,p_j_19), axis=0)
shp=(x_j_20.shape)[0]
shape_y = y_j_20.shape[0]
shape_p =  p_j_20.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_j_20 =  np.delete(x_j_20, [k for k in range(nece_shp,shp)], None)
x_j_20 = x_j_20.reshape(-1,sensors)
shape_x_j_20 = x_j_20.shape[0]
x_j_20 = preprocessing.normalize(x_j_20, axis=0)
x_j_20= np.reshape(x_j_20, (-1,segement_time_size, sensors))
y_j_20= np.delete( y_j_20, [k for k in range(x_j_20.shape[0],shape_y)], None)
p_j_20 = np.delete(p_j_20,[k for k in range(x_j_20.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_j_20), axis=0)
y= np.concatenate((y,y_j_20), axis=0)
p= np.concatenate((p,p_j_20), axis=0)
shp=(x_j_21.shape)[0]
shape_y = y_j_21.shape[0]
shape_p =  p_j_21.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_j_21 =  np.delete(x_j_21, [k for k in range(nece_shp,shp)], None)
x_j_21 = x_j_21.reshape(-1,sensors)
shape_x_j_21 = x_j_21.shape[0]
x_j_21 = preprocessing.normalize(x_j_21, axis=0)
x_j_21= np.reshape(x_j_21, (-1,segement_time_size, sensors))
y_j_21= np.delete( y_j_21, [k for k in range(x_j_21.shape[0],shape_y)], None)
p_j_21 = np.delete(p_j_21,[k for k in range(x_j_21.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_j_21), axis=0)
y= np.concatenate((y,y_j_21), axis=0)
p= np.concatenate((p,p_j_21), axis=0)
shp=(x_j_22.shape)[0]
shape_y = y_j_22.shape[0]
shape_p =  p_j_22.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_j_22 =  np.delete(x_j_22, [k for k in range(nece_shp,shp)], None)
x_j_22 = x_j_22.reshape(-1,sensors)
shape_x_j_22 = x_j_22.shape[0]
x_j_22 = preprocessing.normalize(x_j_22, axis=0)
x_j_22= np.reshape(x_j_22, (-1,segement_time_size, sensors))
y_j_22= np.delete( y_j_22, [k for k in range(x_j_22.shape[0],shape_y)], None)
p_j_22 = np.delete(p_j_22,[k for k in range(x_j_22.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_j_22), axis=0)
y= np.concatenate((y,y_j_22), axis=0)
p= np.concatenate((p,p_j_22), axis=0)
shp=(x_j_23.shape)[0]
shape_y = y_j_23.shape[0]
shape_p =  p_j_23.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_j_23 =  np.delete(x_j_23, [k for k in range(nece_shp,shp)], None)
x_j_23 = x_j_23.reshape(-1,sensors)
shape_x_j_23 = x_j_23.shape[0]
x_j_23 = preprocessing.normalize(x_j_23, axis=0)
x_j_23= np.reshape(x_j_23, (-1,segement_time_size, sensors))
y_j_23= np.delete( y_j_23, [k for k in range(x_j_23.shape[0],shape_y)], None)
p_j_23 = np.delete(p_j_23,[k for k in range(x_j_23.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_j_23), axis=0)
y= np.concatenate((y,y_j_23), axis=0)
p= np.concatenate((p,p_j_23), axis=0)
shp=(x_j_24.shape)[0]
shape_y = y_j_24.shape[0]
shape_p =  p_j_24.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_j_24 =  np.delete(x_j_24, [k for k in range(nece_shp,shp)], None)
x_j_24 = x_j_24.reshape(-1,sensors)
shape_x_j_24 = x_j_24.shape[0]
x_j_24 = preprocessing.normalize(x_j_24, axis=0)
x_j_24= np.reshape(x_j_24, (-1,segement_time_size, sensors))
y_j_24= np.delete( y_j_24, [k for k in range(x_j_24.shape[0],shape_y)], None)
p_j_24 = np.delete(p_j_24,[k for k in range(x_j_24.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_j_24), axis=0)
y= np.concatenate((y,y_j_24), axis=0)
p= np.concatenate((p,p_j_24), axis=0)
shp=(x_j_25.shape)[0]
shape_y = y_j_25.shape[0]
shape_p =  p_j_25.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_j_25 =  np.delete(x_j_25, [k for k in range(nece_shp,shp)], None)
x_j_25 = x_j_25.reshape(-1,sensors)
shape_x_j_25 = x_j_25.shape[0]
x_j_25 = preprocessing.normalize(x_j_25, axis=0)
x_j_25= np.reshape(x_j_25, (-1,segement_time_size, sensors))
y_j_25= np.delete( y_j_25, [k for k in range(x_j_25.shape[0],shape_y)], None)
p_j_25 = np.delete(p_j_25,[k for k in range(x_j_25.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_j_25), axis=0)
y= np.concatenate((y,y_j_25), axis=0)
p= np.concatenate((p,p_j_25), axis=0)
shp=(x_j_26.shape)[0]
shape_y = y_j_26.shape[0]
shape_p =  p_j_26.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_j_26 =  np.delete(x_j_26, [k for k in range(nece_shp,shp)], None)
x_j_26 = x_j_26.reshape(-1,sensors)
shape_x_j_26 = x_j_26.shape[0]
x_j_26 = preprocessing.normalize(x_j_26, axis=0)
x_j_26= np.reshape(x_j_26, (-1,segement_time_size, sensors))
y_j_26= np.delete( y_j_26, [k for k in range(x_j_26.shape[0],shape_y)], None)
p_j_26 = np.delete(p_j_26,[k for k in range(x_j_26.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_j_26), axis=0)
y= np.concatenate((y,y_j_26), axis=0)
p= np.concatenate((p,p_j_26), axis=0)
shp=(x_j_27.shape)[0]
shape_y = y_j_27.shape[0]
shape_p =  p_j_27.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_j_27 =  np.delete(x_j_27, [k for k in range(nece_shp,shp)], None)
x_j_27 = x_j_27.reshape(-1,sensors)
shape_x_j_27 = x_j_27.shape[0]
x_j_27 = preprocessing.normalize(x_j_27, axis=0)
x_j_27= np.reshape(x_j_27, (-1,segement_time_size, sensors))
y_j_27= np.delete( y_j_27, [k for k in range(x_j_27.shape[0],shape_y)], None)
p_j_27 = np.delete(p_j_27,[k for k in range(x_j_27.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_j_27), axis=0)
y= np.concatenate((y,y_j_27), axis=0)
p= np.concatenate((p,p_j_27), axis=0)
shp=(x_j_28.shape)[0]
shape_y = y_j_28.shape[0]
shape_p =  p_j_28.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_j_28 =  np.delete(x_j_28, [k for k in range(nece_shp,shp)], None)
x_j_28 = x_j_28.reshape(-1,sensors)
shape_x_j_28 = x_j_28.shape[0]
x_j_28 = preprocessing.normalize(x_j_28, axis=0)
x_j_28= np.reshape(x_j_28, (-1,segement_time_size, sensors))
y_j_28= np.delete( y_j_28, [k for k in range(x_j_28.shape[0],shape_y)], None)
p_j_28 = np.delete(p_j_28,[k for k in range(x_j_28.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_j_28), axis=0)
y= np.concatenate((y,y_j_28), axis=0)
p= np.concatenate((p,p_j_28), axis=0)
shp=(x_j_29.shape)[0]
shape_y = y_j_29.shape[0]
shape_p =  p_j_29.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_j_29 =  np.delete(x_j_29, [k for k in range(nece_shp,shp)], None)
x_j_29 = x_j_29.reshape(-1,sensors)
shape_x_j_29 = x_j_29.shape[0]
x_j_29 = preprocessing.normalize(x_j_29, axis=0)
x_j_29= np.reshape(x_j_29, (-1,segement_time_size, sensors))
y_j_29= np.delete( y_j_29, [k for k in range(x_j_29.shape[0],shape_y)], None)
p_j_29 = np.delete(p_j_29,[k for k in range(x_j_29.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_j_29), axis=0)
y= np.concatenate((y,y_j_29), axis=0)
p= np.concatenate((p,p_j_29), axis=0)
shp=(x_j_30.shape)[0]
shape_y = y_j_30.shape[0]
shape_p =  p_j_30.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_j_30 =  np.delete(x_j_30, [k for k in range(nece_shp,shp)], None)
x_j_30 = x_j_30.reshape(-1,sensors)
shape_x_j_30 = x_j_30.shape[0]
x_j_30 = preprocessing.normalize(x_j_30, axis=0)
x_j_30= np.reshape(x_j_30, (-1,segement_time_size, sensors))
y_j_30= np.delete( y_j_30, [k for k in range(x_j_30.shape[0],shape_y)], None)
p_j_30 = np.delete(p_j_30,[k for k in range(x_j_30.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_j_30), axis=0)
y= np.concatenate((y,y_j_30), axis=0)
p= np.concatenate((p,p_j_30), axis=0)
shp=(x_j_31.shape)[0]
shape_y = y_j_31.shape[0]
shape_p =  p_j_31.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_j_31 =  np.delete(x_j_31, [k for k in range(nece_shp,shp)], None)
x_j_31 = x_j_31.reshape(-1,sensors)
shape_x_j_31 = x_j_31.shape[0]
x_j_31 = preprocessing.normalize(x_j_31, axis=0)
x_j_31= np.reshape(x_j_31, (-1,segement_time_size, sensors))
y_j_31= np.delete( y_j_31, [k for k in range(x_j_31.shape[0],shape_y)], None)
p_j_31 = np.delete(p_j_31,[k for k in range(x_j_31.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_j_31), axis=0)
y= np.concatenate((y,y_j_31), axis=0)
p= np.concatenate((p,p_j_31), axis=0)
shp=(x_j_32.shape)[0]
shape_y = y_j_32.shape[0]
shape_p =  p_j_32.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_j_32 =  np.delete(x_j_32, [k for k in range(nece_shp,shp)], None)
x_j_32 = x_j_32.reshape(-1,sensors)
shape_x_j_32 = x_j_32.shape[0]
x_j_32 = preprocessing.normalize(x_j_32, axis=0)
x_j_32= np.reshape(x_j_32, (-1,segement_time_size, sensors))
y_j_32= np.delete( y_j_32, [k for k in range(x_j_32.shape[0],shape_y)], None)
p_j_32 = np.delete(p_j_32,[k for k in range(x_j_32.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_j_32), axis=0)
y= np.concatenate((y,y_j_32), axis=0)
p= np.concatenate((p,p_j_32), axis=0)
shp=(x_j_33.shape)[0]
shape_y = y_j_33.shape[0]
shape_p =  p_j_33.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_j_33 =  np.delete(x_j_33, [k for k in range(nece_shp,shp)], None)
x_j_33 = x_j_33.reshape(-1,sensors)
shape_x_j_33 = x_j_33.shape[0]
x_j_33 = preprocessing.normalize(x_j_33, axis=0)
x_j_33= np.reshape(x_j_33, (-1,segement_time_size, sensors))
y_j_33= np.delete( y_j_33, [k for k in range(x_j_33.shape[0],shape_y)], None)
p_j_33 = np.delete(p_j_33,[k for k in range(x_j_33.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_j_33), axis=0)
y= np.concatenate((y,y_j_33), axis=0)
p= np.concatenate((p,p_j_33), axis=0)
shp=(x_j_34.shape)[0]
shape_y = y_j_34.shape[0]
shape_p =  p_j_34.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_j_34 =  np.delete(x_j_34, [k for k in range(nece_shp,shp)], None)
x_j_34 = x_j_34.reshape(-1,sensors)
shape_x_j_34 = x_j_34.shape[0]
x_j_34 = preprocessing.normalize(x_j_34, axis=0)
x_j_34= np.reshape(x_j_34, (-1,segement_time_size, sensors))
y_j_34= np.delete( y_j_34, [k for k in range(x_j_34.shape[0],shape_y)], None)
p_j_34 = np.delete(p_j_34,[k for k in range(x_j_34.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_j_34), axis=0)
y= np.concatenate((y,y_j_34), axis=0)
p= np.concatenate((p,p_j_34), axis=0)
shp=(x_j_35.shape)[0]
shape_y = y_j_35.shape[0]
shape_p =  p_j_35.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_j_35 =  np.delete(x_j_35, [k for k in range(nece_shp,shp)], None)
x_j_35 = x_j_35.reshape(-1,sensors)
shape_x_j_35 = x_j_35.shape[0]
x_j_35 = preprocessing.normalize(x_j_35, axis=0)
x_j_35= np.reshape(x_j_35, (-1,segement_time_size, sensors))
y_j_35= np.delete( y_j_35, [k for k in range(x_j_35.shape[0],shape_y)], None)
p_j_35 = np.delete(p_j_35,[k for k in range(x_j_35.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_j_35), axis=0)
y= np.concatenate((y,y_j_35), axis=0)
p= np.concatenate((p,p_j_35), axis=0)
shp=(x_j_36.shape)[0]
shape_y = y_j_36.shape[0]
shape_p =  p_j_36.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_j_36 =  np.delete(x_j_36, [k for k in range(nece_shp,shp)], None)
x_j_36 = x_j_36.reshape(-1,sensors)
shape_x_j_36 = x_j_36.shape[0]
x_j_36 = preprocessing.normalize(x_j_36, axis=0)
x_j_36= np.reshape(x_j_36, (-1,segement_time_size, sensors))
y_j_36= np.delete( y_j_36, [k for k in range(x_j_36.shape[0],shape_y)], None)
p_j_36 = np.delete(p_j_36,[k for k in range(x_j_36.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_j_36), axis=0)
y= np.concatenate((y,y_j_36), axis=0)
p= np.concatenate((p,p_j_36), axis=0)
shp=(x_j_37.shape)[0]
shape_y = y_j_37.shape[0]
shape_p =  p_j_37.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_j_37 =  np.delete(x_j_37, [k for k in range(nece_shp,shp)], None)
x_j_37 = x_j_37.reshape(-1,sensors)
shape_x_j_37 = x_j_37.shape[0]
x_j_37 = preprocessing.normalize(x_j_37, axis=0)
x_j_37= np.reshape(x_j_37, (-1,segement_time_size, sensors))
y_j_37= np.delete( y_j_37, [k for k in range(x_j_37.shape[0],shape_y)], None)
p_j_37 = np.delete(p_j_37,[k for k in range(x_j_37.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_j_37), axis=0)
y= np.concatenate((y,y_j_37), axis=0)
p= np.concatenate((p,p_j_37), axis=0)
shp=(x_j_38.shape)[0]
shape_y = y_j_38.shape[0]
shape_p =  p_j_38.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_j_38 =  np.delete(x_j_38, [k for k in range(nece_shp,shp)], None)
x_j_38 = x_j_38.reshape(-1,sensors)
shape_x_j_38 = x_j_38.shape[0]
x_j_38 = preprocessing.normalize(x_j_38, axis=0)
x_j_38= np.reshape(x_j_38, (-1,segement_time_size, sensors))
y_j_38= np.delete( y_j_38, [k for k in range(x_j_38.shape[0],shape_y)], None)
p_j_38 = np.delete(p_j_38,[k for k in range(x_j_38.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_j_38), axis=0)
y= np.concatenate((y,y_j_38), axis=0)
p= np.concatenate((p,p_j_38), axis=0)
shp=(x_j_39.shape)[0]
shape_y = y_j_39.shape[0]
shape_p =  p_j_39.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_j_39 =  np.delete(x_j_39, [k for k in range(nece_shp,shp)], None)
x_j_39 = x_j_39.reshape(-1,sensors)
shape_x_j_39 = x_j_39.shape[0]
x_j_39 = preprocessing.normalize(x_j_39, axis=0)
x_j_39= np.reshape(x_j_39, (-1,segement_time_size, sensors))
y_j_39= np.delete( y_j_39, [k for k in range(x_j_39.shape[0],shape_y)], None)
p_j_39 = np.delete(p_j_39,[k for k in range(x_j_39.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_j_39), axis=0)
y= np.concatenate((y,y_j_39), axis=0)
p= np.concatenate((p,p_j_39), axis=0)
shp=(x_j_40.shape)[0]
shape_y = y_j_40.shape[0]
shape_p =  p_j_40.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_j_40 =  np.delete(x_j_40, [k for k in range(nece_shp,shp)], None)
x_j_40 = x_j_40.reshape(-1,sensors)
shape_x_j_40 = x_j_40.shape[0]
x_j_40 = preprocessing.normalize(x_j_40, axis=0)
x_j_40= np.reshape(x_j_40, (-1,segement_time_size, sensors))
y_j_40= np.delete( y_j_40, [k for k in range(x_j_40.shape[0],shape_y)], None)
p_j_40 = np.delete(p_j_40,[k for k in range(x_j_40.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_j_40), axis=0)
y= np.concatenate((y,y_j_40), axis=0)
p= np.concatenate((p,p_j_40), axis=0)
shp=(x_j_41.shape)[0]
shape_y = y_j_41.shape[0]
shape_p =  p_j_41.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_j_41 =  np.delete(x_j_41, [k for k in range(nece_shp,shp)], None)
x_j_41 = x_j_41.reshape(-1,sensors)
shape_x_j_41 = x_j_41.shape[0]
x_j_41 = preprocessing.normalize(x_j_41, axis=0)
x_j_41= np.reshape(x_j_41, (-1,segement_time_size, sensors))
y_j_41= np.delete( y_j_41, [k for k in range(x_j_41.shape[0],shape_y)], None)
p_j_41 = np.delete(p_j_41,[k for k in range(x_j_41.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_j_41), axis=0)
y= np.concatenate((y,y_j_41), axis=0)
p= np.concatenate((p,p_j_41), axis=0)
shp=(x_j_42.shape)[0]
shape_y = y_j_42.shape[0]
shape_p =  p_j_42.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_j_42 =  np.delete(x_j_42, [k for k in range(nece_shp,shp)], None)
x_j_42 = x_j_42.reshape(-1,sensors)
shape_x_j_42 = x_j_42.shape[0]
x_j_42 = preprocessing.normalize(x_j_42, axis=0)
x_j_42= np.reshape(x_j_42, (-1,segement_time_size, sensors))
y_j_42= np.delete( y_j_42, [k for k in range(x_j_42.shape[0],shape_y)], None)
p_j_42 = np.delete(p_j_42,[k for k in range(x_j_42.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_j_42), axis=0)
y= np.concatenate((y,y_j_42), axis=0)
p= np.concatenate((p,p_j_42), axis=0)
shp=(x_j_43.shape)[0]
shape_y = y_j_43.shape[0]
shape_p =  p_j_43.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_j_43 =  np.delete(x_j_43, [k for k in range(nece_shp,shp)], None)
x_j_43 = x_j_43.reshape(-1,sensors)
shape_x_j_43 = x_j_43.shape[0]
x_j_43 = preprocessing.normalize(x_j_43, axis=0)
x_j_43= np.reshape(x_j_43, (-1,segement_time_size, sensors))
y_j_43= np.delete( y_j_43, [k for k in range(x_j_43.shape[0],shape_y)], None)
p_j_43 = np.delete(p_j_43,[k for k in range(x_j_43.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_j_43), axis=0)
y= np.concatenate((y,y_j_43), axis=0)
p= np.concatenate((p,p_j_43), axis=0)
shp=(x_j_44.shape)[0]
shape_y = y_j_44.shape[0]
shape_p =  p_j_44.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_j_44 =  np.delete(x_j_44, [k for k in range(nece_shp,shp)], None)
x_j_44 = x_j_44.reshape(-1,sensors)
shape_x_j_44 = x_j_44.shape[0]
x_j_44 = preprocessing.normalize(x_j_44, axis=0)
x_j_44= np.reshape(x_j_44, (-1,segement_time_size, sensors))
y_j_44= np.delete( y_j_44, [k for k in range(x_j_44.shape[0],shape_y)], None)
p_j_44 = np.delete(p_j_44,[k for k in range(x_j_44.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_j_44), axis=0)
y= np.concatenate((y,y_j_44), axis=0)
p= np.concatenate((p,p_j_44), axis=0)
shp=(x_j_45.shape)[0]
shape_y = y_j_45.shape[0]
shape_p =  p_j_45.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_j_45 =  np.delete(x_j_45, [k for k in range(nece_shp,shp)], None)
x_j_45 = x_j_45.reshape(-1,sensors)
shape_x_j_45 = x_j_45.shape[0]
x_j_45 = preprocessing.normalize(x_j_45, axis=0)
x_j_45= np.reshape(x_j_45, (-1,segement_time_size, sensors))
y_j_45= np.delete( y_j_45, [k for k in range(x_j_45.shape[0],shape_y)], None)
p_j_45 = np.delete(p_j_45,[k for k in range(x_j_45.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_j_45), axis=0)
y= np.concatenate((y,y_j_45), axis=0)
p= np.concatenate((p,p_j_45), axis=0)
shp=(x_j_46.shape)[0]
shape_y = y_j_46.shape[0]
shape_p =  p_j_46.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_j_46 =  np.delete(x_j_46, [k for k in range(nece_shp,shp)], None)
x_j_46 = x_j_46.reshape(-1,sensors)
shape_x_j_46 = x_j_46.shape[0]
x_j_46 = preprocessing.normalize(x_j_46, axis=0)
x_j_46= np.reshape(x_j_46, (-1,segement_time_size, sensors))
y_j_46= np.delete( y_j_46, [k for k in range(x_j_46.shape[0],shape_y)], None)
p_j_46 = np.delete(p_j_46,[k for k in range(x_j_46.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_j_46), axis=0)
y= np.concatenate((y,y_j_46), axis=0)
p= np.concatenate((p,p_j_46), axis=0)
shp=(x_j_47.shape)[0]
shape_y = y_j_47.shape[0]
shape_p =  p_j_47.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_j_47 =  np.delete(x_j_47, [k for k in range(nece_shp,shp)], None)
x_j_47 = x_j_47.reshape(-1,sensors)
shape_x_j_47 = x_j_47.shape[0]
x_j_47 = preprocessing.normalize(x_j_47, axis=0)
x_j_47= np.reshape(x_j_47, (-1,segement_time_size, sensors))
y_j_47= np.delete( y_j_47, [k for k in range(x_j_47.shape[0],shape_y)], None)
p_j_47 = np.delete(p_j_47,[k for k in range(x_j_47.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_j_47), axis=0)
y= np.concatenate((y,y_j_47), axis=0)
p= np.concatenate((p,p_j_47), axis=0)
shp=(x_j_48.shape)[0]
shape_y = y_j_48.shape[0]
shape_p =  p_j_48.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_j_48 =  np.delete(x_j_48, [k for k in range(nece_shp,shp)], None)
x_j_48 = x_j_48.reshape(-1,sensors)
shape_x_j_48 = x_j_48.shape[0]
x_j_48 = preprocessing.normalize(x_j_48, axis=0)
x_j_48= np.reshape(x_j_48, (-1,segement_time_size, sensors))
y_j_48= np.delete( y_j_48, [k for k in range(x_j_48.shape[0],shape_y)], None)
p_j_48 = np.delete(p_j_48,[k for k in range(x_j_48.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_j_48), axis=0)
y= np.concatenate((y,y_j_48), axis=0)
p= np.concatenate((p,p_j_48), axis=0)
shp=(x_j_49.shape)[0]
shape_y = y_j_49.shape[0]
shape_p =  p_j_49.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_j_49 =  np.delete(x_j_49, [k for k in range(nece_shp,shp)], None)
x_j_49 = x_j_49.reshape(-1,sensors)
shape_x_j_49 = x_j_49.shape[0]
x_j_49 = preprocessing.normalize(x_j_49, axis=0)
x_j_49= np.reshape(x_j_49, (-1,segement_time_size, sensors))
y_j_49= np.delete( y_j_49, [k for k in range(x_j_49.shape[0],shape_y)], None)
p_j_49 = np.delete(p_j_49,[k for k in range(x_j_49.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_j_49), axis=0)
y= np.concatenate((y,y_j_49), axis=0)
p= np.concatenate((p,p_j_49), axis=0)
shp=(x_j_50.shape)[0]
shape_y = y_j_50.shape[0]
shape_p =  p_j_50.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_j_50 =  np.delete(x_j_50, [k for k in range(nece_shp,shp)], None)
x_j_50 = x_j_50.reshape(-1,sensors)
shape_x_j_50 = x_j_50.shape[0]
x_j_50 = preprocessing.normalize(x_j_50, axis=0)
x_j_50= np.reshape(x_j_50, (-1,segement_time_size, sensors))
y_j_50= np.delete( y_j_50, [k for k in range(x_j_50.shape[0],shape_y)], None)
p_j_50 = np.delete(p_j_50,[k for k in range(x_j_50.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_j_50), axis=0)
y= np.concatenate((y,y_j_50), axis=0)
p= np.concatenate((p,p_j_50), axis=0)
shp=(x_j_51.shape)[0]
shape_y = y_j_51.shape[0]
shape_p =  p_j_51.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_j_51 =  np.delete(x_j_51, [k for k in range(nece_shp,shp)], None)
x_j_51 = x_j_51.reshape(-1,sensors)
shape_x_j_51 = x_j_51.shape[0]
x_j_51 = preprocessing.normalize(x_j_51, axis=0)
x_j_51= np.reshape(x_j_51, (-1,segement_time_size, sensors))
y_j_51= np.delete( y_j_51, [k for k in range(x_j_51.shape[0],shape_y)], None)
p_j_51 = np.delete(p_j_51,[k for k in range(x_j_51.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_j_51), axis=0)
y= np.concatenate((y,y_j_51), axis=0)
p= np.concatenate((p,p_j_51), axis=0)
shp=(x_j_52.shape)[0]
shape_y = y_j_52.shape[0]
shape_p =  p_j_52.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_j_52 =  np.delete(x_j_52, [k for k in range(nece_shp,shp)], None)
x_j_52 = x_j_52.reshape(-1,sensors)
shape_x_j_52 = x_j_52.shape[0]
x_j_52 = preprocessing.normalize(x_j_52, axis=0)
x_j_52= np.reshape(x_j_52, (-1,segement_time_size, sensors))
y_j_52= np.delete( y_j_52, [k for k in range(x_j_52.shape[0],shape_y)], None)
p_j_52 = np.delete(p_j_52,[k for k in range(x_j_52.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_j_52), axis=0)
y= np.concatenate((y,y_j_52), axis=0)
p= np.concatenate((p,p_j_52), axis=0)
shp=(x_j_53.shape)[0]
shape_y = y_j_53.shape[0]
shape_p =  p_j_53.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_j_53 =  np.delete(x_j_53, [k for k in range(nece_shp,shp)], None)
x_j_53 = x_j_53.reshape(-1,sensors)
shape_x_j_53 = x_j_53.shape[0]
x_j_53 = preprocessing.normalize(x_j_53, axis=0)
x_j_53= np.reshape(x_j_53, (-1,segement_time_size, sensors))
y_j_53= np.delete( y_j_53, [k for k in range(x_j_53.shape[0],shape_y)], None)
p_j_53 = np.delete(p_j_53,[k for k in range(x_j_53.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_j_53), axis=0)
y= np.concatenate((y,y_j_53), axis=0)
p= np.concatenate((p,p_j_53), axis=0)
shp=(x_j_54.shape)[0]
shape_y = y_j_54.shape[0]
shape_p =  p_j_54.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_j_54 =  np.delete(x_j_54, [k for k in range(nece_shp,shp)], None)
x_j_54 = x_j_54.reshape(-1,sensors)
shape_x_j_54 = x_j_54.shape[0]
x_j_54 = preprocessing.normalize(x_j_54, axis=0)
x_j_54= np.reshape(x_j_54, (-1,segement_time_size, sensors))
y_j_54= np.delete( y_j_54, [k for k in range(x_j_54.shape[0],shape_y)], None)
p_j_54 = np.delete(p_j_54,[k for k in range(x_j_54.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_j_54), axis=0)
y= np.concatenate((y,y_j_54), axis=0)
p= np.concatenate((p,p_j_54), axis=0)
shp=(x_j_55.shape)[0]
shape_y = y_j_55.shape[0]
shape_p =  p_j_55.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_j_55 =  np.delete(x_j_55, [k for k in range(nece_shp,shp)], None)
x_j_55 = x_j_55.reshape(-1,sensors)
shape_x_j_55 = x_j_55.shape[0]
x_j_55 = preprocessing.normalize(x_j_55, axis=0)
x_j_55= np.reshape(x_j_55, (-1,segement_time_size, sensors))
y_j_55= np.delete( y_j_55, [k for k in range(x_j_55.shape[0],shape_y)], None)
p_j_55 = np.delete(p_j_55,[k for k in range(x_j_55.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_j_55), axis=0)
y= np.concatenate((y,y_j_55), axis=0)
p= np.concatenate((p,p_j_55), axis=0)
shp=(x_j_56.shape)[0]
shape_y = y_j_56.shape[0]
shape_p =  p_j_56.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_j_56 =  np.delete(x_j_56, [k for k in range(nece_shp,shp)], None)
x_j_56 = x_j_56.reshape(-1,sensors)
shape_x_j_56 = x_j_56.shape[0]
x_j_56 = preprocessing.normalize(x_j_56, axis=0)
x_j_56= np.reshape(x_j_56, (-1,segement_time_size, sensors))
y_j_56= np.delete( y_j_56, [k for k in range(x_j_56.shape[0],shape_y)], None)
p_j_56 = np.delete(p_j_56,[k for k in range(x_j_56.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_j_56), axis=0)
y= np.concatenate((y,y_j_56), axis=0)
p= np.concatenate((p,p_j_56), axis=0)
shp=(x_j_57.shape)[0]
shape_y = y_j_57.shape[0]
shape_p =  p_j_57.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_j_57 =  np.delete(x_j_57, [k for k in range(nece_shp,shp)], None)
x_j_57 = x_j_57.reshape(-1,sensors)
shape_x_j_57 = x_j_57.shape[0]
x_j_57 = preprocessing.normalize(x_j_57, axis=0)
x_j_57= np.reshape(x_j_57, (-1,segement_time_size, sensors))
y_j_57= np.delete( y_j_57, [k for k in range(x_j_57.shape[0],shape_y)], None)
p_j_57 = np.delete(p_j_57,[k for k in range(x_j_57.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_j_57), axis=0)
y= np.concatenate((y,y_j_57), axis=0)
p= np.concatenate((p,p_j_57), axis=0)
shp=(x_j_58.shape)[0]
shape_y = y_j_58.shape[0]
shape_p =  p_j_58.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_j_58 =  np.delete(x_j_58, [k for k in range(nece_shp,shp)], None)
x_j_58 = x_j_58.reshape(-1,sensors)
shape_x_j_58 = x_j_58.shape[0]
x_j_58 = preprocessing.normalize(x_j_58, axis=0)
x_j_58= np.reshape(x_j_58, (-1,segement_time_size, sensors))
y_j_58= np.delete( y_j_58, [k for k in range(x_j_58.shape[0],shape_y)], None)
p_j_58 = np.delete(p_j_58,[k for k in range(x_j_58.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_j_58), axis=0)
y= np.concatenate((y,y_j_58), axis=0)
p= np.concatenate((p,p_j_58), axis=0)
shp=(x_j_59.shape)[0]
shape_y = y_j_59.shape[0]
shape_p =  p_j_59.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_j_59 =  np.delete(x_j_59, [k for k in range(nece_shp,shp)], None)
x_j_59 = x_j_59.reshape(-1,sensors)
shape_x_j_59 = x_j_59.shape[0]
x_j_59 = preprocessing.normalize(x_j_59, axis=0)
x_j_59= np.reshape(x_j_59, (-1,segement_time_size, sensors))
y_j_59= np.delete( y_j_59, [k for k in range(x_j_59.shape[0],shape_y)], None)
p_j_59 = np.delete(p_j_59,[k for k in range(x_j_59.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_j_59), axis=0)
y= np.concatenate((y,y_j_59), axis=0)
p= np.concatenate((p,p_j_59), axis=0)
shp=(x_j_60.shape)[0]
shape_y = y_j_60.shape[0]
shape_p =  p_j_60.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_j_60 =  np.delete(x_j_60, [k for k in range(nece_shp,shp)], None)
x_j_60 = x_j_60.reshape(-1,sensors)
shape_x_j_60 = x_j_60.shape[0]
x_j_60 = preprocessing.normalize(x_j_60, axis=0)
x_j_60= np.reshape(x_j_60, (-1,segement_time_size, sensors))
y_j_60= np.delete( y_j_60, [k for k in range(x_j_60.shape[0],shape_y)], None)
p_j_60 = np.delete(p_j_60,[k for k in range(x_j_60.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_j_60), axis=0)
y= np.concatenate((y,y_j_60), axis=0)
p= np.concatenate((p,p_j_60), axis=0)
shp=(x_j_61.shape)[0]
shape_y = y_j_61.shape[0]
shape_p =  p_j_61.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_j_61 =  np.delete(x_j_61, [k for k in range(nece_shp,shp)], None)
x_j_61 = x_j_61.reshape(-1,sensors)
shape_x_j_61 = x_j_61.shape[0]
x_j_61 = preprocessing.normalize(x_j_61, axis=0)
x_j_61= np.reshape(x_j_61, (-1,segement_time_size, sensors))
y_j_61= np.delete( y_j_61, [k for k in range(x_j_61.shape[0],shape_y)], None)
p_j_61 = np.delete(p_j_61,[k for k in range(x_j_61.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_j_61), axis=0)
y= np.concatenate((y,y_j_61), axis=0)
p= np.concatenate((p,p_j_61), axis=0)
shp=(x_j_62.shape)[0]
shape_y = y_j_62.shape[0]
shape_p =  p_j_62.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_j_62 =  np.delete(x_j_62, [k for k in range(nece_shp,shp)], None)
x_j_62 = x_j_62.reshape(-1,sensors)
shape_x_j_62 = x_j_62.shape[0]
x_j_62 = preprocessing.normalize(x_j_62, axis=0)
x_j_62= np.reshape(x_j_62, (-1,segement_time_size, sensors))
y_j_62= np.delete( y_j_62, [k for k in range(x_j_62.shape[0],shape_y)], None)
p_j_62 = np.delete(p_j_62,[k for k in range(x_j_62.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_j_62), axis=0)
y= np.concatenate((y,y_j_62), axis=0)
p= np.concatenate((p,p_j_62), axis=0)
shp=(x_j_63.shape)[0]
shape_y = y_j_63.shape[0]
shape_p =  p_j_63.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_j_63 =  np.delete(x_j_63, [k for k in range(nece_shp,shp)], None)
x_j_63 = x_j_63.reshape(-1,sensors)
shape_x_j_63 = x_j_63.shape[0]
x_j_63 = preprocessing.normalize(x_j_63, axis=0)
x_j_63= np.reshape(x_j_63, (-1,segement_time_size, sensors))
y_j_63= np.delete( y_j_63, [k for k in range(x_j_63.shape[0],shape_y)], None)
p_j_63 = np.delete(p_j_63,[k for k in range(x_j_63.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_j_63), axis=0)
y= np.concatenate((y,y_j_63), axis=0)
p= np.concatenate((p,p_j_63), axis=0)
shp=(x_j_64.shape)[0]
shape_y = y_j_64.shape[0]
shape_p =  p_j_64.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_j_64 =  np.delete(x_j_64, [k for k in range(nece_shp,shp)], None)
x_j_64 = x_j_64.reshape(-1,sensors)
shape_x_j_64 = x_j_64.shape[0]
x_j_64 = preprocessing.normalize(x_j_64, axis=0)
x_j_64= np.reshape(x_j_64, (-1,segement_time_size, sensors))
y_j_64= np.delete( y_j_64, [k for k in range(x_j_64.shape[0],shape_y)], None)
p_j_64 = np.delete(p_j_64,[k for k in range(x_j_64.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_j_64), axis=0)
y= np.concatenate((y,y_j_64), axis=0)
p= np.concatenate((p,p_j_64), axis=0)
shp=(x_j_65.shape)[0]
shape_y = y_j_65.shape[0]
shape_p =  p_j_65.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_j_65 =  np.delete(x_j_65, [k for k in range(nece_shp,shp)], None)
x_j_65 = x_j_65.reshape(-1,sensors)
shape_x_j_65 = x_j_65.shape[0]
x_j_65 = preprocessing.normalize(x_j_65, axis=0)
x_j_65= np.reshape(x_j_65, (-1,segement_time_size, sensors))
y_j_65= np.delete( y_j_65, [k for k in range(x_j_65.shape[0],shape_y)], None)
p_j_65 = np.delete(p_j_65,[k for k in range(x_j_65.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_j_65), axis=0)
y= np.concatenate((y,y_j_65), axis=0)
p= np.concatenate((p,p_j_65), axis=0)
shp=(x_j_66.shape)[0]
shape_y = y_j_66.shape[0]
shape_p =  p_j_66.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_j_66 =  np.delete(x_j_66, [k for k in range(nece_shp,shp)], None)
x_j_66 = x_j_66.reshape(-1,sensors)
shape_x_j_66 = x_j_66.shape[0]
x_j_66 = preprocessing.normalize(x_j_66, axis=0)
x_j_66= np.reshape(x_j_66, (-1,segement_time_size, sensors))
y_j_66= np.delete( y_j_66, [k for k in range(x_j_66.shape[0],shape_y)], None)
p_j_66 = np.delete(p_j_66,[k for k in range(x_j_66.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_j_66), axis=0)
y= np.concatenate((y,y_j_66), axis=0)
p= np.concatenate((p,p_j_66), axis=0)
shp=(x_j_67.shape)[0]
shape_y = y_j_67.shape[0]
shape_p =  p_j_67.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_j_67 =  np.delete(x_j_67, [k for k in range(nece_shp,shp)], None)
x_j_67 = x_j_67.reshape(-1,sensors)
shape_x_j_67 = x_j_67.shape[0]
x_j_67 = preprocessing.normalize(x_j_67, axis=0)
x_j_67= np.reshape(x_j_67, (-1,segement_time_size, sensors))
y_j_67= np.delete( y_j_67, [k for k in range(x_j_67.shape[0],shape_y)], None)
p_j_67 = np.delete(p_j_67,[k for k in range(x_j_67.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_j_67), axis=0)
y= np.concatenate((y,y_j_67), axis=0)
p= np.concatenate((p,p_j_67), axis=0)
shp=(x_j_68.shape)[0]
shape_y = y_j_68.shape[0]
shape_p =  p_j_68.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_j_68 =  np.delete(x_j_68, [k for k in range(nece_shp,shp)], None)
x_j_68 = x_j_68.reshape(-1,sensors)
shape_x_j_68 = x_j_68.shape[0]
x_j_68 = preprocessing.normalize(x_j_68, axis=0)
x_j_68= np.reshape(x_j_68, (-1,segement_time_size, sensors))
y_j_68= np.delete( y_j_68, [k for k in range(x_j_68.shape[0],shape_y)], None)
p_j_68 = np.delete(p_j_68,[k for k in range(x_j_68.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_j_68), axis=0)
y= np.concatenate((y,y_j_68), axis=0)
p= np.concatenate((p,p_j_68), axis=0)
shp=(x_j_69.shape)[0]
shape_y = y_j_69.shape[0]
shape_p =  p_j_69.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_j_69 =  np.delete(x_j_69, [k for k in range(nece_shp,shp)], None)
x_j_69 = x_j_69.reshape(-1,sensors)
shape_x_j_69 = x_j_69.shape[0]
x_j_69 = preprocessing.normalize(x_j_69, axis=0)
x_j_69= np.reshape(x_j_69, (-1,segement_time_size, sensors))
y_j_69= np.delete( y_j_69, [k for k in range(x_j_69.shape[0],shape_y)], None)
p_j_69 = np.delete(p_j_69,[k for k in range(x_j_69.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_j_69), axis=0)
y= np.concatenate((y,y_j_69), axis=0)
p= np.concatenate((p,p_j_69), axis=0)
shp=(x_j_70.shape)[0]
shape_y = y_j_70.shape[0]
shape_p =  p_j_70.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_j_70 =  np.delete(x_j_70, [k for k in range(nece_shp,shp)], None)
x_j_70 = x_j_70.reshape(-1,sensors)
shape_x_j_70 = x_j_70.shape[0]
x_j_70 = preprocessing.normalize(x_j_70, axis=0)
x_j_70= np.reshape(x_j_70, (-1,segement_time_size, sensors))
y_j_70= np.delete( y_j_70, [k for k in range(x_j_70.shape[0],shape_y)], None)
p_j_70 = np.delete(p_j_70,[k for k in range(x_j_70.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_j_70), axis=0)
y= np.concatenate((y,y_j_70), axis=0)
p= np.concatenate((p,p_j_70), axis=0)
shp=(x_j_71.shape)[0]
shape_y = y_j_71.shape[0]
shape_p =  p_j_71.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_j_71 =  np.delete(x_j_71, [k for k in range(nece_shp,shp)], None)
x_j_71 = x_j_71.reshape(-1,sensors)
shape_x_j_71 = x_j_71.shape[0]
x_j_71 = preprocessing.normalize(x_j_71, axis=0)
x_j_71= np.reshape(x_j_71, (-1,segement_time_size, sensors))
y_j_71= np.delete( y_j_71, [k for k in range(x_j_71.shape[0],shape_y)], None)
p_j_71 = np.delete(p_j_71,[k for k in range(x_j_71.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_j_71), axis=0)
y= np.concatenate((y,y_j_71), axis=0)
p= np.concatenate((p,p_j_71), axis=0)
shp=(x_j_72.shape)[0]
shape_y = y_j_72.shape[0]
shape_p =  p_j_72.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_j_72 =  np.delete(x_j_72, [k for k in range(nece_shp,shp)], None)
x_j_72 = x_j_72.reshape(-1,sensors)
shape_x_j_72 = x_j_72.shape[0]
x_j_72 = preprocessing.normalize(x_j_72, axis=0)
x_j_72= np.reshape(x_j_72, (-1,segement_time_size, sensors))
y_j_72= np.delete( y_j_72, [k for k in range(x_j_72.shape[0],shape_y)], None)
p_j_72 = np.delete(p_j_72,[k for k in range(x_j_72.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_j_72), axis=0)
y= np.concatenate((y,y_j_72), axis=0)
p= np.concatenate((p,p_j_72), axis=0)
shp=(x_j_73.shape)[0]
shape_y = y_j_73.shape[0]
shape_p =  p_j_73.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_j_73 =  np.delete(x_j_73, [k for k in range(nece_shp,shp)], None)
x_j_73 = x_j_73.reshape(-1,sensors)
shape_x_j_73 = x_j_73.shape[0]
x_j_73 = preprocessing.normalize(x_j_73, axis=0)
x_j_73= np.reshape(x_j_73, (-1,segement_time_size, sensors))
y_j_73= np.delete( y_j_73, [k for k in range(x_j_73.shape[0],shape_y)], None)
p_j_73 = np.delete(p_j_73,[k for k in range(x_j_73.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_j_73), axis=0)
y= np.concatenate((y,y_j_73), axis=0)
p= np.concatenate((p,p_j_73), axis=0)
shp=(x_j_74.shape)[0]
shape_y = y_j_74.shape[0]
shape_p =  p_j_74.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_j_74 =  np.delete(x_j_74, [k for k in range(nece_shp,shp)], None)
x_j_74 = x_j_74.reshape(-1,sensors)
shape_x_j_74 = x_j_74.shape[0]
x_j_74 = preprocessing.normalize(x_j_74, axis=0)
x_j_74= np.reshape(x_j_74, (-1,segement_time_size, sensors))
y_j_74= np.delete( y_j_74, [k for k in range(x_j_74.shape[0],shape_y)], None)
p_j_74 = np.delete(p_j_74,[k for k in range(x_j_74.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_j_74), axis=0)
y= np.concatenate((y,y_j_74), axis=0)
p= np.concatenate((p,p_j_74), axis=0)
shp=(x_j_75.shape)[0]
shape_y = y_j_75.shape[0]
shape_p =  p_j_75.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_j_75 =  np.delete(x_j_75, [k for k in range(nece_shp,shp)], None)
x_j_75 = x_j_75.reshape(-1,sensors)
shape_x_j_75 = x_j_75.shape[0]
x_j_75 = preprocessing.normalize(x_j_75, axis=0)
x_j_75= np.reshape(x_j_75, (-1,segement_time_size, sensors))
y_j_75= np.delete( y_j_75, [k for k in range(x_j_75.shape[0],shape_y)], None)
p_j_75 = np.delete(p_j_75,[k for k in range(x_j_75.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_j_75), axis=0)
y= np.concatenate((y,y_j_75), axis=0)
p= np.concatenate((p,p_j_75), axis=0)
shp=(x_j_76.shape)[0]
shape_y = y_j_76.shape[0]
shape_p =  p_j_76.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_j_76 =  np.delete(x_j_76, [k for k in range(nece_shp,shp)], None)
x_j_76 = x_j_76.reshape(-1,sensors)
shape_x_j_76 = x_j_76.shape[0]
x_j_76 = preprocessing.normalize(x_j_76, axis=0)
x_j_76= np.reshape(x_j_76, (-1,segement_time_size, sensors))
y_j_76= np.delete( y_j_76, [k for k in range(x_j_76.shape[0],shape_y)], None)
p_j_76 = np.delete(p_j_76,[k for k in range(x_j_76.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_j_76), axis=0)
y= np.concatenate((y,y_j_76), axis=0)
p= np.concatenate((p,p_j_76), axis=0)
shp=(x_j_77.shape)[0]
shape_y = y_j_77.shape[0]
shape_p =  p_j_77.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_j_77 =  np.delete(x_j_77, [k for k in range(nece_shp,shp)], None)
x_j_77 = x_j_77.reshape(-1,sensors)
shape_x_j_77 = x_j_77.shape[0]
x_j_77 = preprocessing.normalize(x_j_77, axis=0)
x_j_77= np.reshape(x_j_77, (-1,segement_time_size, sensors))
y_j_77= np.delete( y_j_77, [k for k in range(x_j_77.shape[0],shape_y)], None)
p_j_77 = np.delete(p_j_77,[k for k in range(x_j_77.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_j_77), axis=0)
y= np.concatenate((y,y_j_77), axis=0)
p= np.concatenate((p,p_j_77), axis=0)
shp=(x_j_78.shape)[0]
shape_y = y_j_78.shape[0]
shape_p =  p_j_78.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_j_78 =  np.delete(x_j_78, [k for k in range(nece_shp,shp)], None)
x_j_78 = x_j_78.reshape(-1,sensors)
shape_x_j_78 = x_j_78.shape[0]
x_j_78 = preprocessing.normalize(x_j_78, axis=0)
x_j_78= np.reshape(x_j_78, (-1,segement_time_size, sensors))
y_j_78= np.delete( y_j_78, [k for k in range(x_j_78.shape[0],shape_y)], None)
p_j_78 = np.delete(p_j_78,[k for k in range(x_j_78.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_j_78), axis=0)
y= np.concatenate((y,y_j_78), axis=0)
p= np.concatenate((p,p_j_78), axis=0)
shp=(x_j_79.shape)[0]
shape_y = y_j_79.shape[0]
shape_p =  p_j_79.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_j_79 =  np.delete(x_j_79, [k for k in range(nece_shp,shp)], None)
x_j_79 = x_j_79.reshape(-1,sensors)
shape_x_j_79 = x_j_79.shape[0]
x_j_79 = preprocessing.normalize(x_j_79, axis=0)
x_j_79= np.reshape(x_j_79, (-1,segement_time_size, sensors))
y_j_79= np.delete( y_j_79, [k for k in range(x_j_79.shape[0],shape_y)], None)
p_j_79 = np.delete(p_j_79,[k for k in range(x_j_79.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_j_79), axis=0)
y= np.concatenate((y,y_j_79), axis=0)
p= np.concatenate((p,p_j_79), axis=0)
shp=(x_j_80.shape)[0]
shape_y = y_j_80.shape[0]
shape_p =  p_j_80.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_j_80 =  np.delete(x_j_80, [k for k in range(nece_shp,shp)], None)
x_j_80 = x_j_80.reshape(-1,sensors)
shape_x_j_80 = x_j_80.shape[0]
x_j_80 = preprocessing.normalize(x_j_80, axis=0)
x_j_80= np.reshape(x_j_80, (-1,segement_time_size, sensors))
y_j_80= np.delete( y_j_80, [k for k in range(x_j_80.shape[0],shape_y)], None)
p_j_80 = np.delete(p_j_80,[k for k in range(x_j_80.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_j_80), axis=0)
y= np.concatenate((y,y_j_80), axis=0)
p= np.concatenate((p,p_j_80), axis=0)
shp=(x_j_81.shape)[0]
shape_y = y_j_81.shape[0]
shape_p =  p_j_81.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_j_81 =  np.delete(x_j_81, [k for k in range(nece_shp,shp)], None)
x_j_81 = x_j_81.reshape(-1,sensors)
shape_x_j_81 = x_j_81.shape[0]
x_j_81 = preprocessing.normalize(x_j_81, axis=0)
x_j_81= np.reshape(x_j_81, (-1,segement_time_size, sensors))
y_j_81= np.delete( y_j_81, [k for k in range(x_j_81.shape[0],shape_y)], None)
p_j_81 = np.delete(p_j_81,[k for k in range(x_j_81.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_j_81), axis=0)
y= np.concatenate((y,y_j_81), axis=0)
p= np.concatenate((p,p_j_81), axis=0)
shp=(x_j_82.shape)[0]
shape_y = y_j_82.shape[0]
shape_p =  p_j_82.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_j_82 =  np.delete(x_j_82, [k for k in range(nece_shp,shp)], None)
x_j_82 = x_j_82.reshape(-1,sensors)
shape_x_j_82 = x_j_82.shape[0]
x_j_82 = preprocessing.normalize(x_j_82, axis=0)
x_j_82= np.reshape(x_j_82, (-1,segement_time_size, sensors))
y_j_82= np.delete( y_j_82, [k for k in range(x_j_82.shape[0],shape_y)], None)
p_j_82 = np.delete(p_j_82,[k for k in range(x_j_82.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_j_82), axis=0)
y= np.concatenate((y,y_j_82), axis=0)
p= np.concatenate((p,p_j_82), axis=0)
shp=(x_j_83.shape)[0]
shape_y = y_j_83.shape[0]
shape_p =  p_j_83.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_j_83 =  np.delete(x_j_83, [k for k in range(nece_shp,shp)], None)
x_j_83 = x_j_83.reshape(-1,sensors)
shape_x_j_83 = x_j_83.shape[0]
x_j_83 = preprocessing.normalize(x_j_83, axis=0)
x_j_83= np.reshape(x_j_83, (-1,segement_time_size, sensors))
y_j_83= np.delete( y_j_83, [k for k in range(x_j_83.shape[0],shape_y)], None)
p_j_83 = np.delete(p_j_83,[k for k in range(x_j_83.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_j_83), axis=0)
y= np.concatenate((y,y_j_83), axis=0)
p= np.concatenate((p,p_j_83), axis=0)
shp=(x_j_84.shape)[0]
shape_y = y_j_84.shape[0]
shape_p =  p_j_84.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_j_84 =  np.delete(x_j_84, [k for k in range(nece_shp,shp)], None)
x_j_84 = x_j_84.reshape(-1,sensors)
shape_x_j_84 = x_j_84.shape[0]
x_j_84 = preprocessing.normalize(x_j_84, axis=0)
x_j_84= np.reshape(x_j_84, (-1,segement_time_size, sensors))
y_j_84= np.delete( y_j_84, [k for k in range(x_j_84.shape[0],shape_y)], None)
p_j_84 = np.delete(p_j_84,[k for k in range(x_j_84.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_j_84), axis=0)
y= np.concatenate((y,y_j_84), axis=0)
p= np.concatenate((p,p_j_84), axis=0)
shp=(x_j_85.shape)[0]
shape_y = y_j_85.shape[0]
shape_p =  p_j_85.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_j_85 =  np.delete(x_j_85, [k for k in range(nece_shp,shp)], None)
x_j_85 = x_j_85.reshape(-1,sensors)
shape_x_j_85 = x_j_85.shape[0]
x_j_85 = preprocessing.normalize(x_j_85, axis=0)
x_j_85= np.reshape(x_j_85, (-1,segement_time_size, sensors))
y_j_85= np.delete( y_j_85, [k for k in range(x_j_85.shape[0],shape_y)], None)
p_j_85 = np.delete(p_j_85,[k for k in range(x_j_85.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_j_85), axis=0)
y= np.concatenate((y,y_j_85), axis=0)
p= np.concatenate((p,p_j_85), axis=0)
shp=(x_j_86.shape)[0]
shape_y = y_j_86.shape[0]
shape_p =  p_j_86.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_j_86 =  np.delete(x_j_86, [k for k in range(nece_shp,shp)], None)
x_j_86 = x_j_86.reshape(-1,sensors)
shape_x_j_86 = x_j_86.shape[0]
x_j_86 = preprocessing.normalize(x_j_86, axis=0)
x_j_86= np.reshape(x_j_86, (-1,segement_time_size, sensors))
y_j_86= np.delete( y_j_86, [k for k in range(x_j_86.shape[0],shape_y)], None)
p_j_86 = np.delete(p_j_86,[k for k in range(x_j_86.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_j_86), axis=0)
y= np.concatenate((y,y_j_86), axis=0)
p= np.concatenate((p,p_j_86), axis=0)
shp=(x_j_87.shape)[0]
shape_y = y_j_87.shape[0]
shape_p =  p_j_87.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_j_87 =  np.delete(x_j_87, [k for k in range(nece_shp,shp)], None)
x_j_87 = x_j_87.reshape(-1,sensors)
shape_x_j_87 = x_j_87.shape[0]
x_j_87 = preprocessing.normalize(x_j_87, axis=0)
x_j_87= np.reshape(x_j_87, (-1,segement_time_size, sensors))
y_j_87= np.delete( y_j_87, [k for k in range(x_j_87.shape[0],shape_y)], None)
p_j_87 = np.delete(p_j_87,[k for k in range(x_j_87.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_j_87), axis=0)
y= np.concatenate((y,y_j_87), axis=0)
p= np.concatenate((p,p_j_87), axis=0)
shp=(x_j_88.shape)[0]
shape_y = y_j_88.shape[0]
shape_p =  p_j_88.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_j_88 =  np.delete(x_j_88, [k for k in range(nece_shp,shp)], None)
x_j_88 = x_j_88.reshape(-1,sensors)
shape_x_j_88 = x_j_88.shape[0]
x_j_88 = preprocessing.normalize(x_j_88, axis=0)
x_j_88= np.reshape(x_j_88, (-1,segement_time_size, sensors))
y_j_88= np.delete( y_j_88, [k for k in range(x_j_88.shape[0],shape_y)], None)
p_j_88 = np.delete(p_j_88,[k for k in range(x_j_88.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_j_88), axis=0)
y= np.concatenate((y,y_j_88), axis=0)
p= np.concatenate((p,p_j_88), axis=0)
shp=(x_j_89.shape)[0]
shape_y = y_j_89.shape[0]
shape_p =  p_j_89.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_j_89 =  np.delete(x_j_89, [k for k in range(nece_shp,shp)], None)
x_j_89 = x_j_89.reshape(-1,sensors)
shape_x_j_89 = x_j_89.shape[0]
x_j_89 = preprocessing.normalize(x_j_89, axis=0)
x_j_89= np.reshape(x_j_89, (-1,segement_time_size, sensors))
y_j_89= np.delete( y_j_89, [k for k in range(x_j_89.shape[0],shape_y)], None)
p_j_89 = np.delete(p_j_89,[k for k in range(x_j_89.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_j_89), axis=0)
y= np.concatenate((y,y_j_89), axis=0)
p= np.concatenate((p,p_j_89), axis=0)
shp=(x_j_90.shape)[0]
shape_y = y_j_90.shape[0]
shape_p =  p_j_90.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_j_90 =  np.delete(x_j_90, [k for k in range(nece_shp,shp)], None)
x_j_90 = x_j_90.reshape(-1,sensors)
shape_x_j_90 = x_j_90.shape[0]
x_j_90 = preprocessing.normalize(x_j_90, axis=0)
x_j_90= np.reshape(x_j_90, (-1,segement_time_size, sensors))
y_j_90= np.delete( y_j_90, [k for k in range(x_j_90.shape[0],shape_y)], None)
p_j_90 = np.delete(p_j_90,[k for k in range(x_j_90.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_j_90), axis=0)
y= np.concatenate((y,y_j_90), axis=0)
p= np.concatenate((p,p_j_90), axis=0)
shp=(x_j_91.shape)[0]
shape_y = y_j_91.shape[0]
shape_p =  p_j_91.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_j_91 =  np.delete(x_j_91, [k for k in range(nece_shp,shp)], None)
x_j_91 = x_j_91.reshape(-1,sensors)
shape_x_j_91 = x_j_91.shape[0]
x_j_91 = preprocessing.normalize(x_j_91, axis=0)
x_j_91= np.reshape(x_j_91, (-1,segement_time_size, sensors))
y_j_91= np.delete( y_j_91, [k for k in range(x_j_91.shape[0],shape_y)], None)
p_j_91 = np.delete(p_j_91,[k for k in range(x_j_91.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_j_91), axis=0)
y= np.concatenate((y,y_j_91), axis=0)
p= np.concatenate((p,p_j_91), axis=0)
shp=(x_j_92.shape)[0]
shape_y = y_j_92.shape[0]
shape_p =  p_j_92.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_j_92 =  np.delete(x_j_92, [k for k in range(nece_shp,shp)], None)
x_j_92 = x_j_92.reshape(-1,sensors)
shape_x_j_92 = x_j_92.shape[0]
x_j_92 = preprocessing.normalize(x_j_92, axis=0)
x_j_92= np.reshape(x_j_92, (-1,segement_time_size, sensors))
y_j_92= np.delete( y_j_92, [k for k in range(x_j_92.shape[0],shape_y)], None)
p_j_92 = np.delete(p_j_92,[k for k in range(x_j_92.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_j_92), axis=0)
y= np.concatenate((y,y_j_92), axis=0)
p= np.concatenate((p,p_j_92), axis=0)
shp=(x_j_93.shape)[0]
shape_y = y_j_93.shape[0]
shape_p =  p_j_93.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_j_93 =  np.delete(x_j_93, [k for k in range(nece_shp,shp)], None)
x_j_93 = x_j_93.reshape(-1,sensors)
shape_x_j_93 = x_j_93.shape[0]
x_j_93 = preprocessing.normalize(x_j_93, axis=0)
x_j_93= np.reshape(x_j_93, (-1,segement_time_size, sensors))
y_j_93= np.delete( y_j_93, [k for k in range(x_j_93.shape[0],shape_y)], None)
p_j_93 = np.delete(p_j_93,[k for k in range(x_j_93.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_j_93), axis=0)
y= np.concatenate((y,y_j_93), axis=0)
p= np.concatenate((p,p_j_93), axis=0)
shp=(x_j_94.shape)[0]
shape_y = y_j_94.shape[0]
shape_p =  p_j_94.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_j_94 =  np.delete(x_j_94, [k for k in range(nece_shp,shp)], None)
x_j_94 = x_j_94.reshape(-1,sensors)
shape_x_j_94 = x_j_94.shape[0]
x_j_94 = preprocessing.normalize(x_j_94, axis=0)
x_j_94= np.reshape(x_j_94, (-1,segement_time_size, sensors))
y_j_94= np.delete( y_j_94, [k for k in range(x_j_94.shape[0],shape_y)], None)
p_j_94 = np.delete(p_j_94,[k for k in range(x_j_94.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_j_94), axis=0)
y= np.concatenate((y,y_j_94), axis=0)
p= np.concatenate((p,p_j_94), axis=0)
shp=(x_j_95.shape)[0]
shape_y = y_j_95.shape[0]
shape_p =  p_j_95.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_j_95 =  np.delete(x_j_95, [k for k in range(nece_shp,shp)], None)
x_j_95 = x_j_95.reshape(-1,sensors)
shape_x_j_95 = x_j_95.shape[0]
x_j_95 = preprocessing.normalize(x_j_95, axis=0)
x_j_95= np.reshape(x_j_95, (-1,segement_time_size, sensors))
y_j_95= np.delete( y_j_95, [k for k in range(x_j_95.shape[0],shape_y)], None)
p_j_95 = np.delete(p_j_95,[k for k in range(x_j_95.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_j_95), axis=0)
y= np.concatenate((y,y_j_95), axis=0)
p= np.concatenate((p,p_j_95), axis=0)
shp=(x_j_96.shape)[0]
shape_y = y_j_96.shape[0]
shape_p =  p_j_96.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_j_96 =  np.delete(x_j_96, [k for k in range(nece_shp,shp)], None)
x_j_96 = x_j_96.reshape(-1,sensors)
shape_x_j_96 = x_j_96.shape[0]
x_j_96 = preprocessing.normalize(x_j_96, axis=0)
x_j_96= np.reshape(x_j_96, (-1,segement_time_size, sensors))
y_j_96= np.delete( y_j_96, [k for k in range(x_j_96.shape[0],shape_y)], None)
p_j_96 = np.delete(p_j_96,[k for k in range(x_j_96.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_j_96), axis=0)
y= np.concatenate((y,y_j_96), axis=0)
p= np.concatenate((p,p_j_96), axis=0)
shp=(x_j_97.shape)[0]
shape_y = y_j_97.shape[0]
shape_p =  p_j_97.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_j_97 =  np.delete(x_j_97, [k for k in range(nece_shp,shp)], None)
x_j_97 = x_j_97.reshape(-1,sensors)
shape_x_j_97 = x_j_97.shape[0]
x_j_97 = preprocessing.normalize(x_j_97, axis=0)
x_j_97= np.reshape(x_j_97, (-1,segement_time_size, sensors))
y_j_97= np.delete( y_j_97, [k for k in range(x_j_97.shape[0],shape_y)], None)
p_j_97 = np.delete(p_j_97,[k for k in range(x_j_97.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_j_97), axis=0)
y= np.concatenate((y,y_j_97), axis=0)
p= np.concatenate((p,p_j_97), axis=0)
shp=(x_j_98.shape)[0]
shape_y = y_j_98.shape[0]
shape_p =  p_j_98.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_j_98 =  np.delete(x_j_98, [k for k in range(nece_shp,shp)], None)
x_j_98 = x_j_98.reshape(-1,sensors)
shape_x_j_98 = x_j_98.shape[0]
x_j_98 = preprocessing.normalize(x_j_98, axis=0)
x_j_98= np.reshape(x_j_98, (-1,segement_time_size, sensors))
y_j_98= np.delete( y_j_98, [k for k in range(x_j_98.shape[0],shape_y)], None)
p_j_98 = np.delete(p_j_98,[k for k in range(x_j_98.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_j_98), axis=0)
y= np.concatenate((y,y_j_98), axis=0)
p= np.concatenate((p,p_j_98), axis=0)
shp=(x_j_99.shape)[0]
shape_y = y_j_99.shape[0]
shape_p =  p_j_99.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_j_99 =  np.delete(x_j_99, [k for k in range(nece_shp,shp)], None)
x_j_99 = x_j_99.reshape(-1,sensors)
shape_x_j_99 = x_j_99.shape[0]
x_j_99 = preprocessing.normalize(x_j_99, axis=0)
x_j_99= np.reshape(x_j_99, (-1,segement_time_size, sensors))
y_j_99= np.delete( y_j_99, [k for k in range(x_j_99.shape[0],shape_y)], None)
p_j_99 = np.delete(p_j_99,[k for k in range(x_j_99.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_j_99), axis=0)
y= np.concatenate((y,y_j_99), axis=0)
p= np.concatenate((p,p_j_99), axis=0)
shp=(x_s_0.shape)[0]
shape_y = y_s_0.shape[0]
shape_p =  p_s_0.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_s_0 =  np.delete(x_s_0, [k for k in range(nece_shp,shp)], None)
x_s_0 = x_s_0.reshape(-1,sensors)
shape_x_s_0 = x_s_0.shape[0]
x_s_0 = preprocessing.normalize(x_s_0, axis=0)
x_s_0= np.reshape(x_s_0, (-1,segement_time_size, sensors))
y_s_0= np.delete( y_s_0, [k for k in range(x_s_0.shape[0],shape_y)], None)
p_s_0 = np.delete(p_s_0,[k for k in range(x_s_0.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_s_0), axis=0)
y= np.concatenate((y,y_s_0), axis=0)
p= np.concatenate((p,p_s_0), axis=0)
shp=(x_s_1.shape)[0]
shape_y = y_s_1.shape[0]
shape_p =  p_s_1.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_s_1 =  np.delete(x_s_1, [k for k in range(nece_shp,shp)], None)
x_s_1 = x_s_1.reshape(-1,sensors)
shape_x_s_1 = x_s_1.shape[0]
x_s_1 = preprocessing.normalize(x_s_1, axis=0)
x_s_1= np.reshape(x_s_1, (-1,segement_time_size, sensors))
y_s_1= np.delete( y_s_1, [k for k in range(x_s_1.shape[0],shape_y)], None)
p_s_1 = np.delete(p_s_1,[k for k in range(x_s_1.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_s_1), axis=0)
y= np.concatenate((y,y_s_1), axis=0)
p= np.concatenate((p,p_s_1), axis=0)
shp=(x_s_2.shape)[0]
shape_y = y_s_2.shape[0]
shape_p =  p_s_2.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_s_2 =  np.delete(x_s_2, [k for k in range(nece_shp,shp)], None)
x_s_2 = x_s_2.reshape(-1,sensors)
shape_x_s_2 = x_s_2.shape[0]
x_s_2 = preprocessing.normalize(x_s_2, axis=0)
x_s_2= np.reshape(x_s_2, (-1,segement_time_size, sensors))
y_s_2= np.delete( y_s_2, [k for k in range(x_s_2.shape[0],shape_y)], None)
p_s_2 = np.delete(p_s_2,[k for k in range(x_s_2.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_s_2), axis=0)
y= np.concatenate((y,y_s_2), axis=0)
p= np.concatenate((p,p_s_2), axis=0)
shp=(x_s_3.shape)[0]
shape_y = y_s_3.shape[0]
shape_p =  p_s_3.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_s_3 =  np.delete(x_s_3, [k for k in range(nece_shp,shp)], None)
x_s_3 = x_s_3.reshape(-1,sensors)
shape_x_s_3 = x_s_3.shape[0]
x_s_3 = preprocessing.normalize(x_s_3, axis=0)
x_s_3= np.reshape(x_s_3, (-1,segement_time_size, sensors))
y_s_3= np.delete( y_s_3, [k for k in range(x_s_3.shape[0],shape_y)], None)
p_s_3 = np.delete(p_s_3,[k for k in range(x_s_3.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_s_3), axis=0)
y= np.concatenate((y,y_s_3), axis=0)
p= np.concatenate((p,p_s_3), axis=0)
shp=(x_s_4.shape)[0]
shape_y = y_s_4.shape[0]
shape_p =  p_s_4.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_s_4 =  np.delete(x_s_4, [k for k in range(nece_shp,shp)], None)
x_s_4 = x_s_4.reshape(-1,sensors)
shape_x_s_4 = x_s_4.shape[0]
x_s_4 = preprocessing.normalize(x_s_4, axis=0)
x_s_4= np.reshape(x_s_4, (-1,segement_time_size, sensors))
y_s_4= np.delete( y_s_4, [k for k in range(x_s_4.shape[0],shape_y)], None)
p_s_4 = np.delete(p_s_4,[k for k in range(x_s_4.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_s_4), axis=0)
y= np.concatenate((y,y_s_4), axis=0)
p= np.concatenate((p,p_s_4), axis=0)
shp=(x_s_5.shape)[0]
shape_y = y_s_5.shape[0]
shape_p =  p_s_5.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_s_5 =  np.delete(x_s_5, [k for k in range(nece_shp,shp)], None)
x_s_5 = x_s_5.reshape(-1,sensors)
shape_x_s_5 = x_s_5.shape[0]
x_s_5 = preprocessing.normalize(x_s_5, axis=0)
x_s_5= np.reshape(x_s_5, (-1,segement_time_size, sensors))
y_s_5= np.delete( y_s_5, [k for k in range(x_s_5.shape[0],shape_y)], None)
p_s_5 = np.delete(p_s_5,[k for k in range(x_s_5.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_s_5), axis=0)
y= np.concatenate((y,y_s_5), axis=0)
p= np.concatenate((p,p_s_5), axis=0)
shp=(x_s_6.shape)[0]
shape_y = y_s_6.shape[0]
shape_p =  p_s_6.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_s_6 =  np.delete(x_s_6, [k for k in range(nece_shp,shp)], None)
x_s_6 = x_s_6.reshape(-1,sensors)
shape_x_s_6 = x_s_6.shape[0]
x_s_6 = preprocessing.normalize(x_s_6, axis=0)
x_s_6= np.reshape(x_s_6, (-1,segement_time_size, sensors))
y_s_6= np.delete( y_s_6, [k for k in range(x_s_6.shape[0],shape_y)], None)
p_s_6 = np.delete(p_s_6,[k for k in range(x_s_6.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_s_6), axis=0)
y= np.concatenate((y,y_s_6), axis=0)
p= np.concatenate((p,p_s_6), axis=0)
shp=(x_s_7.shape)[0]
shape_y = y_s_7.shape[0]
shape_p =  p_s_7.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_s_7 =  np.delete(x_s_7, [k for k in range(nece_shp,shp)], None)
x_s_7 = x_s_7.reshape(-1,sensors)
shape_x_s_7 = x_s_7.shape[0]
x_s_7 = preprocessing.normalize(x_s_7, axis=0)
x_s_7= np.reshape(x_s_7, (-1,segement_time_size, sensors))
y_s_7= np.delete( y_s_7, [k for k in range(x_s_7.shape[0],shape_y)], None)
p_s_7 = np.delete(p_s_7,[k for k in range(x_s_7.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_s_7), axis=0)
y= np.concatenate((y,y_s_7), axis=0)
p= np.concatenate((p,p_s_7), axis=0)
shp=(x_s_8.shape)[0]
shape_y = y_s_8.shape[0]
shape_p =  p_s_8.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_s_8 =  np.delete(x_s_8, [k for k in range(nece_shp,shp)], None)
x_s_8 = x_s_8.reshape(-1,sensors)
shape_x_s_8 = x_s_8.shape[0]
x_s_8 = preprocessing.normalize(x_s_8, axis=0)
x_s_8= np.reshape(x_s_8, (-1,segement_time_size, sensors))
y_s_8= np.delete( y_s_8, [k for k in range(x_s_8.shape[0],shape_y)], None)
p_s_8 = np.delete(p_s_8,[k for k in range(x_s_8.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_s_8), axis=0)
y= np.concatenate((y,y_s_8), axis=0)
p= np.concatenate((p,p_s_8), axis=0)
shp=(x_s_9.shape)[0]
shape_y = y_s_9.shape[0]
shape_p =  p_s_9.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_s_9 =  np.delete(x_s_9, [k for k in range(nece_shp,shp)], None)
x_s_9 = x_s_9.reshape(-1,sensors)
shape_x_s_9 = x_s_9.shape[0]
x_s_9 = preprocessing.normalize(x_s_9, axis=0)
x_s_9= np.reshape(x_s_9, (-1,segement_time_size, sensors))
y_s_9= np.delete( y_s_9, [k for k in range(x_s_9.shape[0],shape_y)], None)
p_s_9 = np.delete(p_s_9,[k for k in range(x_s_9.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_s_9), axis=0)
y= np.concatenate((y,y_s_9), axis=0)
p= np.concatenate((p,p_s_9), axis=0)
shp=(x_s_10.shape)[0]
shape_y = y_s_10.shape[0]
shape_p =  p_s_10.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_s_10 =  np.delete(x_s_10, [k for k in range(nece_shp,shp)], None)
x_s_10 = x_s_10.reshape(-1,sensors)
shape_x_s_10 = x_s_10.shape[0]
x_s_10 = preprocessing.normalize(x_s_10, axis=0)
x_s_10= np.reshape(x_s_10, (-1,segement_time_size, sensors))
y_s_10= np.delete( y_s_10, [k for k in range(x_s_10.shape[0],shape_y)], None)
p_s_10 = np.delete(p_s_10,[k for k in range(x_s_10.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_s_10), axis=0)
y= np.concatenate((y,y_s_10), axis=0)
p= np.concatenate((p,p_s_10), axis=0)
shp=(x_s_11.shape)[0]
shape_y = y_s_11.shape[0]
shape_p =  p_s_11.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_s_11 =  np.delete(x_s_11, [k for k in range(nece_shp,shp)], None)
x_s_11 = x_s_11.reshape(-1,sensors)
shape_x_s_11 = x_s_11.shape[0]
x_s_11 = preprocessing.normalize(x_s_11, axis=0)
x_s_11= np.reshape(x_s_11, (-1,segement_time_size, sensors))
y_s_11= np.delete( y_s_11, [k for k in range(x_s_11.shape[0],shape_y)], None)
p_s_11 = np.delete(p_s_11,[k for k in range(x_s_11.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_s_11), axis=0)
y= np.concatenate((y,y_s_11), axis=0)
p= np.concatenate((p,p_s_11), axis=0)
shp=(x_s_12.shape)[0]
shape_y = y_s_12.shape[0]
shape_p =  p_s_12.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_s_12 =  np.delete(x_s_12, [k for k in range(nece_shp,shp)], None)
x_s_12 = x_s_12.reshape(-1,sensors)
shape_x_s_12 = x_s_12.shape[0]
x_s_12 = preprocessing.normalize(x_s_12, axis=0)
x_s_12= np.reshape(x_s_12, (-1,segement_time_size, sensors))
y_s_12= np.delete( y_s_12, [k for k in range(x_s_12.shape[0],shape_y)], None)
p_s_12 = np.delete(p_s_12,[k for k in range(x_s_12.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_s_12), axis=0)
y= np.concatenate((y,y_s_12), axis=0)
p= np.concatenate((p,p_s_12), axis=0)
shp=(x_s_13.shape)[0]
shape_y = y_s_13.shape[0]
shape_p =  p_s_13.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_s_13 =  np.delete(x_s_13, [k for k in range(nece_shp,shp)], None)
x_s_13 = x_s_13.reshape(-1,sensors)
shape_x_s_13 = x_s_13.shape[0]
x_s_13 = preprocessing.normalize(x_s_13, axis=0)
x_s_13= np.reshape(x_s_13, (-1,segement_time_size, sensors))
y_s_13= np.delete( y_s_13, [k for k in range(x_s_13.shape[0],shape_y)], None)
p_s_13 = np.delete(p_s_13,[k for k in range(x_s_13.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_s_13), axis=0)
y= np.concatenate((y,y_s_13), axis=0)
p= np.concatenate((p,p_s_13), axis=0)
shp=(x_s_14.shape)[0]
shape_y = y_s_14.shape[0]
shape_p =  p_s_14.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_s_14 =  np.delete(x_s_14, [k for k in range(nece_shp,shp)], None)
x_s_14 = x_s_14.reshape(-1,sensors)
shape_x_s_14 = x_s_14.shape[0]
x_s_14 = preprocessing.normalize(x_s_14, axis=0)
x_s_14= np.reshape(x_s_14, (-1,segement_time_size, sensors))
y_s_14= np.delete( y_s_14, [k for k in range(x_s_14.shape[0],shape_y)], None)
p_s_14 = np.delete(p_s_14,[k for k in range(x_s_14.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_s_14), axis=0)
y= np.concatenate((y,y_s_14), axis=0)
p= np.concatenate((p,p_s_14), axis=0)
shp=(x_s_15.shape)[0]
shape_y = y_s_15.shape[0]
shape_p =  p_s_15.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_s_15 =  np.delete(x_s_15, [k for k in range(nece_shp,shp)], None)
x_s_15 = x_s_15.reshape(-1,sensors)
shape_x_s_15 = x_s_15.shape[0]
x_s_15 = preprocessing.normalize(x_s_15, axis=0)
x_s_15= np.reshape(x_s_15, (-1,segement_time_size, sensors))
y_s_15= np.delete( y_s_15, [k for k in range(x_s_15.shape[0],shape_y)], None)
p_s_15 = np.delete(p_s_15,[k for k in range(x_s_15.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_s_15), axis=0)
y= np.concatenate((y,y_s_15), axis=0)
p= np.concatenate((p,p_s_15), axis=0)
shp=(x_s_16.shape)[0]
shape_y = y_s_16.shape[0]
shape_p =  p_s_16.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_s_16 =  np.delete(x_s_16, [k for k in range(nece_shp,shp)], None)
x_s_16 = x_s_16.reshape(-1,sensors)
shape_x_s_16 = x_s_16.shape[0]
x_s_16 = preprocessing.normalize(x_s_16, axis=0)
x_s_16= np.reshape(x_s_16, (-1,segement_time_size, sensors))
y_s_16= np.delete( y_s_16, [k for k in range(x_s_16.shape[0],shape_y)], None)
p_s_16 = np.delete(p_s_16,[k for k in range(x_s_16.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_s_16), axis=0)
y= np.concatenate((y,y_s_16), axis=0)
p= np.concatenate((p,p_s_16), axis=0)
shp=(x_s_17.shape)[0]
shape_y = y_s_17.shape[0]
shape_p =  p_s_17.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_s_17 =  np.delete(x_s_17, [k for k in range(nece_shp,shp)], None)
x_s_17 = x_s_17.reshape(-1,sensors)
shape_x_s_17 = x_s_17.shape[0]
x_s_17 = preprocessing.normalize(x_s_17, axis=0)
x_s_17= np.reshape(x_s_17, (-1,segement_time_size, sensors))
y_s_17= np.delete( y_s_17, [k for k in range(x_s_17.shape[0],shape_y)], None)
p_s_17 = np.delete(p_s_17,[k for k in range(x_s_17.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_s_17), axis=0)
y= np.concatenate((y,y_s_17), axis=0)
p= np.concatenate((p,p_s_17), axis=0)
shp=(x_s_18.shape)[0]
shape_y = y_s_18.shape[0]
shape_p =  p_s_18.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_s_18 =  np.delete(x_s_18, [k for k in range(nece_shp,shp)], None)
x_s_18 = x_s_18.reshape(-1,sensors)
shape_x_s_18 = x_s_18.shape[0]
x_s_18 = preprocessing.normalize(x_s_18, axis=0)
x_s_18= np.reshape(x_s_18, (-1,segement_time_size, sensors))
y_s_18= np.delete( y_s_18, [k for k in range(x_s_18.shape[0],shape_y)], None)
p_s_18 = np.delete(p_s_18,[k for k in range(x_s_18.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_s_18), axis=0)
y= np.concatenate((y,y_s_18), axis=0)
p= np.concatenate((p,p_s_18), axis=0)
shp=(x_s_19.shape)[0]
shape_y = y_s_19.shape[0]
shape_p =  p_s_19.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_s_19 =  np.delete(x_s_19, [k for k in range(nece_shp,shp)], None)
x_s_19 = x_s_19.reshape(-1,sensors)
shape_x_s_19 = x_s_19.shape[0]
x_s_19 = preprocessing.normalize(x_s_19, axis=0)
x_s_19= np.reshape(x_s_19, (-1,segement_time_size, sensors))
y_s_19= np.delete( y_s_19, [k for k in range(x_s_19.shape[0],shape_y)], None)
p_s_19 = np.delete(p_s_19,[k for k in range(x_s_19.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_s_19), axis=0)
y= np.concatenate((y,y_s_19), axis=0)
p= np.concatenate((p,p_s_19), axis=0)
shp=(x_s_20.shape)[0]
shape_y = y_s_20.shape[0]
shape_p =  p_s_20.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_s_20 =  np.delete(x_s_20, [k for k in range(nece_shp,shp)], None)
x_s_20 = x_s_20.reshape(-1,sensors)
shape_x_s_20 = x_s_20.shape[0]
x_s_20 = preprocessing.normalize(x_s_20, axis=0)
x_s_20= np.reshape(x_s_20, (-1,segement_time_size, sensors))
y_s_20= np.delete( y_s_20, [k for k in range(x_s_20.shape[0],shape_y)], None)
p_s_20 = np.delete(p_s_20,[k for k in range(x_s_20.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_s_20), axis=0)
y= np.concatenate((y,y_s_20), axis=0)
p= np.concatenate((p,p_s_20), axis=0)
shp=(x_s_21.shape)[0]
shape_y = y_s_21.shape[0]
shape_p =  p_s_21.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_s_21 =  np.delete(x_s_21, [k for k in range(nece_shp,shp)], None)
x_s_21 = x_s_21.reshape(-1,sensors)
shape_x_s_21 = x_s_21.shape[0]
x_s_21 = preprocessing.normalize(x_s_21, axis=0)
x_s_21= np.reshape(x_s_21, (-1,segement_time_size, sensors))
y_s_21= np.delete( y_s_21, [k for k in range(x_s_21.shape[0],shape_y)], None)
p_s_21 = np.delete(p_s_21,[k for k in range(x_s_21.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_s_21), axis=0)
y= np.concatenate((y,y_s_21), axis=0)
p= np.concatenate((p,p_s_21), axis=0)
shp=(x_s_22.shape)[0]
shape_y = y_s_22.shape[0]
shape_p =  p_s_22.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_s_22 =  np.delete(x_s_22, [k for k in range(nece_shp,shp)], None)
x_s_22 = x_s_22.reshape(-1,sensors)
shape_x_s_22 = x_s_22.shape[0]
x_s_22 = preprocessing.normalize(x_s_22, axis=0)
x_s_22= np.reshape(x_s_22, (-1,segement_time_size, sensors))
y_s_22= np.delete( y_s_22, [k for k in range(x_s_22.shape[0],shape_y)], None)
p_s_22 = np.delete(p_s_22,[k for k in range(x_s_22.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_s_22), axis=0)
y= np.concatenate((y,y_s_22), axis=0)
p= np.concatenate((p,p_s_22), axis=0)
shp=(x_s_23.shape)[0]
shape_y = y_s_23.shape[0]
shape_p =  p_s_23.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_s_23 =  np.delete(x_s_23, [k for k in range(nece_shp,shp)], None)
x_s_23 = x_s_23.reshape(-1,sensors)
shape_x_s_23 = x_s_23.shape[0]
x_s_23 = preprocessing.normalize(x_s_23, axis=0)
x_s_23= np.reshape(x_s_23, (-1,segement_time_size, sensors))
y_s_23= np.delete( y_s_23, [k for k in range(x_s_23.shape[0],shape_y)], None)
p_s_23 = np.delete(p_s_23,[k for k in range(x_s_23.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_s_23), axis=0)
y= np.concatenate((y,y_s_23), axis=0)
p= np.concatenate((p,p_s_23), axis=0)
shp=(x_s_24.shape)[0]
shape_y = y_s_24.shape[0]
shape_p =  p_s_24.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_s_24 =  np.delete(x_s_24, [k for k in range(nece_shp,shp)], None)
x_s_24 = x_s_24.reshape(-1,sensors)
shape_x_s_24 = x_s_24.shape[0]
x_s_24 = preprocessing.normalize(x_s_24, axis=0)
x_s_24= np.reshape(x_s_24, (-1,segement_time_size, sensors))
y_s_24= np.delete( y_s_24, [k for k in range(x_s_24.shape[0],shape_y)], None)
p_s_24 = np.delete(p_s_24,[k for k in range(x_s_24.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_s_24), axis=0)
y= np.concatenate((y,y_s_24), axis=0)
p= np.concatenate((p,p_s_24), axis=0)
shp=(x_s_25.shape)[0]
shape_y = y_s_25.shape[0]
shape_p =  p_s_25.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_s_25 =  np.delete(x_s_25, [k for k in range(nece_shp,shp)], None)
x_s_25 = x_s_25.reshape(-1,sensors)
shape_x_s_25 = x_s_25.shape[0]
x_s_25 = preprocessing.normalize(x_s_25, axis=0)
x_s_25= np.reshape(x_s_25, (-1,segement_time_size, sensors))
y_s_25= np.delete( y_s_25, [k for k in range(x_s_25.shape[0],shape_y)], None)
p_s_25 = np.delete(p_s_25,[k for k in range(x_s_25.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_s_25), axis=0)
y= np.concatenate((y,y_s_25), axis=0)
p= np.concatenate((p,p_s_25), axis=0)
shp=(x_s_26.shape)[0]
shape_y = y_s_26.shape[0]
shape_p =  p_s_26.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_s_26 =  np.delete(x_s_26, [k for k in range(nece_shp,shp)], None)
x_s_26 = x_s_26.reshape(-1,sensors)
shape_x_s_26 = x_s_26.shape[0]
x_s_26 = preprocessing.normalize(x_s_26, axis=0)
x_s_26= np.reshape(x_s_26, (-1,segement_time_size, sensors))
y_s_26= np.delete( y_s_26, [k for k in range(x_s_26.shape[0],shape_y)], None)
p_s_26 = np.delete(p_s_26,[k for k in range(x_s_26.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_s_26), axis=0)
y= np.concatenate((y,y_s_26), axis=0)
p= np.concatenate((p,p_s_26), axis=0)
shp=(x_s_27.shape)[0]
shape_y = y_s_27.shape[0]
shape_p =  p_s_27.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_s_27 =  np.delete(x_s_27, [k for k in range(nece_shp,shp)], None)
x_s_27 = x_s_27.reshape(-1,sensors)
shape_x_s_27 = x_s_27.shape[0]
x_s_27 = preprocessing.normalize(x_s_27, axis=0)
x_s_27= np.reshape(x_s_27, (-1,segement_time_size, sensors))
y_s_27= np.delete( y_s_27, [k for k in range(x_s_27.shape[0],shape_y)], None)
p_s_27 = np.delete(p_s_27,[k for k in range(x_s_27.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_s_27), axis=0)
y= np.concatenate((y,y_s_27), axis=0)
p= np.concatenate((p,p_s_27), axis=0)
shp=(x_s_28.shape)[0]
shape_y = y_s_28.shape[0]
shape_p =  p_s_28.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_s_28 =  np.delete(x_s_28, [k for k in range(nece_shp,shp)], None)
x_s_28 = x_s_28.reshape(-1,sensors)
shape_x_s_28 = x_s_28.shape[0]
x_s_28 = preprocessing.normalize(x_s_28, axis=0)
x_s_28= np.reshape(x_s_28, (-1,segement_time_size, sensors))
y_s_28= np.delete( y_s_28, [k for k in range(x_s_28.shape[0],shape_y)], None)
p_s_28 = np.delete(p_s_28,[k for k in range(x_s_28.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_s_28), axis=0)
y= np.concatenate((y,y_s_28), axis=0)
p= np.concatenate((p,p_s_28), axis=0)
shp=(x_s_29.shape)[0]
shape_y = y_s_29.shape[0]
shape_p =  p_s_29.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_s_29 =  np.delete(x_s_29, [k for k in range(nece_shp,shp)], None)
x_s_29 = x_s_29.reshape(-1,sensors)
shape_x_s_29 = x_s_29.shape[0]
x_s_29 = preprocessing.normalize(x_s_29, axis=0)
x_s_29= np.reshape(x_s_29, (-1,segement_time_size, sensors))
y_s_29= np.delete( y_s_29, [k for k in range(x_s_29.shape[0],shape_y)], None)
p_s_29 = np.delete(p_s_29,[k for k in range(x_s_29.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_s_29), axis=0)
y= np.concatenate((y,y_s_29), axis=0)
p= np.concatenate((p,p_s_29), axis=0)
shp=(x_s_30.shape)[0]
shape_y = y_s_30.shape[0]
shape_p =  p_s_30.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_s_30 =  np.delete(x_s_30, [k for k in range(nece_shp,shp)], None)
x_s_30 = x_s_30.reshape(-1,sensors)
shape_x_s_30 = x_s_30.shape[0]
x_s_30 = preprocessing.normalize(x_s_30, axis=0)
x_s_30= np.reshape(x_s_30, (-1,segement_time_size, sensors))
y_s_30= np.delete( y_s_30, [k for k in range(x_s_30.shape[0],shape_y)], None)
p_s_30 = np.delete(p_s_30,[k for k in range(x_s_30.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_s_30), axis=0)
y= np.concatenate((y,y_s_30), axis=0)
p= np.concatenate((p,p_s_30), axis=0)
shp=(x_s_31.shape)[0]
shape_y = y_s_31.shape[0]
shape_p =  p_s_31.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_s_31 =  np.delete(x_s_31, [k for k in range(nece_shp,shp)], None)
x_s_31 = x_s_31.reshape(-1,sensors)
shape_x_s_31 = x_s_31.shape[0]
x_s_31 = preprocessing.normalize(x_s_31, axis=0)
x_s_31= np.reshape(x_s_31, (-1,segement_time_size, sensors))
y_s_31= np.delete( y_s_31, [k for k in range(x_s_31.shape[0],shape_y)], None)
p_s_31 = np.delete(p_s_31,[k for k in range(x_s_31.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_s_31), axis=0)
y= np.concatenate((y,y_s_31), axis=0)
p= np.concatenate((p,p_s_31), axis=0)
shp=(x_s_32.shape)[0]
shape_y = y_s_32.shape[0]
shape_p =  p_s_32.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_s_32 =  np.delete(x_s_32, [k for k in range(nece_shp,shp)], None)
x_s_32 = x_s_32.reshape(-1,sensors)
shape_x_s_32 = x_s_32.shape[0]
x_s_32 = preprocessing.normalize(x_s_32, axis=0)
x_s_32= np.reshape(x_s_32, (-1,segement_time_size, sensors))
y_s_32= np.delete( y_s_32, [k for k in range(x_s_32.shape[0],shape_y)], None)
p_s_32 = np.delete(p_s_32,[k for k in range(x_s_32.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_s_32), axis=0)
y= np.concatenate((y,y_s_32), axis=0)
p= np.concatenate((p,p_s_32), axis=0)
shp=(x_s_33.shape)[0]
shape_y = y_s_33.shape[0]
shape_p =  p_s_33.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_s_33 =  np.delete(x_s_33, [k for k in range(nece_shp,shp)], None)
x_s_33 = x_s_33.reshape(-1,sensors)
shape_x_s_33 = x_s_33.shape[0]
x_s_33 = preprocessing.normalize(x_s_33, axis=0)
x_s_33= np.reshape(x_s_33, (-1,segement_time_size, sensors))
y_s_33= np.delete( y_s_33, [k for k in range(x_s_33.shape[0],shape_y)], None)
p_s_33 = np.delete(p_s_33,[k for k in range(x_s_33.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_s_33), axis=0)
y= np.concatenate((y,y_s_33), axis=0)
p= np.concatenate((p,p_s_33), axis=0)
shp=(x_s_34.shape)[0]
shape_y = y_s_34.shape[0]
shape_p =  p_s_34.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_s_34 =  np.delete(x_s_34, [k for k in range(nece_shp,shp)], None)
x_s_34 = x_s_34.reshape(-1,sensors)
shape_x_s_34 = x_s_34.shape[0]
x_s_34 = preprocessing.normalize(x_s_34, axis=0)
x_s_34= np.reshape(x_s_34, (-1,segement_time_size, sensors))
y_s_34= np.delete( y_s_34, [k for k in range(x_s_34.shape[0],shape_y)], None)
p_s_34 = np.delete(p_s_34,[k for k in range(x_s_34.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_s_34), axis=0)
y= np.concatenate((y,y_s_34), axis=0)
p= np.concatenate((p,p_s_34), axis=0)
shp=(x_s_35.shape)[0]
shape_y = y_s_35.shape[0]
shape_p =  p_s_35.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_s_35 =  np.delete(x_s_35, [k for k in range(nece_shp,shp)], None)
x_s_35 = x_s_35.reshape(-1,sensors)
shape_x_s_35 = x_s_35.shape[0]
x_s_35 = preprocessing.normalize(x_s_35, axis=0)
x_s_35= np.reshape(x_s_35, (-1,segement_time_size, sensors))
y_s_35= np.delete( y_s_35, [k for k in range(x_s_35.shape[0],shape_y)], None)
p_s_35 = np.delete(p_s_35,[k for k in range(x_s_35.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_s_35), axis=0)
y= np.concatenate((y,y_s_35), axis=0)
p= np.concatenate((p,p_s_35), axis=0)
shp=(x_s_36.shape)[0]
shape_y = y_s_36.shape[0]
shape_p =  p_s_36.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_s_36 =  np.delete(x_s_36, [k for k in range(nece_shp,shp)], None)
x_s_36 = x_s_36.reshape(-1,sensors)
shape_x_s_36 = x_s_36.shape[0]
x_s_36 = preprocessing.normalize(x_s_36, axis=0)
x_s_36= np.reshape(x_s_36, (-1,segement_time_size, sensors))
y_s_36= np.delete( y_s_36, [k for k in range(x_s_36.shape[0],shape_y)], None)
p_s_36 = np.delete(p_s_36,[k for k in range(x_s_36.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_s_36), axis=0)
y= np.concatenate((y,y_s_36), axis=0)
p= np.concatenate((p,p_s_36), axis=0)
shp=(x_s_37.shape)[0]
shape_y = y_s_37.shape[0]
shape_p =  p_s_37.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_s_37 =  np.delete(x_s_37, [k for k in range(nece_shp,shp)], None)
x_s_37 = x_s_37.reshape(-1,sensors)
shape_x_s_37 = x_s_37.shape[0]
x_s_37 = preprocessing.normalize(x_s_37, axis=0)
x_s_37= np.reshape(x_s_37, (-1,segement_time_size, sensors))
y_s_37= np.delete( y_s_37, [k for k in range(x_s_37.shape[0],shape_y)], None)
p_s_37 = np.delete(p_s_37,[k for k in range(x_s_37.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_s_37), axis=0)
y= np.concatenate((y,y_s_37), axis=0)
p= np.concatenate((p,p_s_37), axis=0)
shp=(x_s_38.shape)[0]
shape_y = y_s_38.shape[0]
shape_p =  p_s_38.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_s_38 =  np.delete(x_s_38, [k for k in range(nece_shp,shp)], None)
x_s_38 = x_s_38.reshape(-1,sensors)
shape_x_s_38 = x_s_38.shape[0]
x_s_38 = preprocessing.normalize(x_s_38, axis=0)
x_s_38= np.reshape(x_s_38, (-1,segement_time_size, sensors))
y_s_38= np.delete( y_s_38, [k for k in range(x_s_38.shape[0],shape_y)], None)
p_s_38 = np.delete(p_s_38,[k for k in range(x_s_38.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_s_38), axis=0)
y= np.concatenate((y,y_s_38), axis=0)
p= np.concatenate((p,p_s_38), axis=0)
shp=(x_s_39.shape)[0]
shape_y = y_s_39.shape[0]
shape_p =  p_s_39.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_s_39 =  np.delete(x_s_39, [k for k in range(nece_shp,shp)], None)
x_s_39 = x_s_39.reshape(-1,sensors)
shape_x_s_39 = x_s_39.shape[0]
x_s_39 = preprocessing.normalize(x_s_39, axis=0)
x_s_39= np.reshape(x_s_39, (-1,segement_time_size, sensors))
y_s_39= np.delete( y_s_39, [k for k in range(x_s_39.shape[0],shape_y)], None)
p_s_39 = np.delete(p_s_39,[k for k in range(x_s_39.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_s_39), axis=0)
y= np.concatenate((y,y_s_39), axis=0)
p= np.concatenate((p,p_s_39), axis=0)
shp=(x_s_40.shape)[0]
shape_y = y_s_40.shape[0]
shape_p =  p_s_40.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_s_40 =  np.delete(x_s_40, [k for k in range(nece_shp,shp)], None)
x_s_40 = x_s_40.reshape(-1,sensors)
shape_x_s_40 = x_s_40.shape[0]
x_s_40 = preprocessing.normalize(x_s_40, axis=0)
x_s_40= np.reshape(x_s_40, (-1,segement_time_size, sensors))
y_s_40= np.delete( y_s_40, [k for k in range(x_s_40.shape[0],shape_y)], None)
p_s_40 = np.delete(p_s_40,[k for k in range(x_s_40.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_s_40), axis=0)
y= np.concatenate((y,y_s_40), axis=0)
p= np.concatenate((p,p_s_40), axis=0)
shp=(x_s_41.shape)[0]
shape_y = y_s_41.shape[0]
shape_p =  p_s_41.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_s_41 =  np.delete(x_s_41, [k for k in range(nece_shp,shp)], None)
x_s_41 = x_s_41.reshape(-1,sensors)
shape_x_s_41 = x_s_41.shape[0]
x_s_41 = preprocessing.normalize(x_s_41, axis=0)
x_s_41= np.reshape(x_s_41, (-1,segement_time_size, sensors))
y_s_41= np.delete( y_s_41, [k for k in range(x_s_41.shape[0],shape_y)], None)
p_s_41 = np.delete(p_s_41,[k for k in range(x_s_41.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_s_41), axis=0)
y= np.concatenate((y,y_s_41), axis=0)
p= np.concatenate((p,p_s_41), axis=0)
shp=(x_s_42.shape)[0]
shape_y = y_s_42.shape[0]
shape_p =  p_s_42.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_s_42 =  np.delete(x_s_42, [k for k in range(nece_shp,shp)], None)
x_s_42 = x_s_42.reshape(-1,sensors)
shape_x_s_42 = x_s_42.shape[0]
x_s_42 = preprocessing.normalize(x_s_42, axis=0)
x_s_42= np.reshape(x_s_42, (-1,segement_time_size, sensors))
y_s_42= np.delete( y_s_42, [k for k in range(x_s_42.shape[0],shape_y)], None)
p_s_42 = np.delete(p_s_42,[k for k in range(x_s_42.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_s_42), axis=0)
y= np.concatenate((y,y_s_42), axis=0)
p= np.concatenate((p,p_s_42), axis=0)
shp=(x_s_43.shape)[0]
shape_y = y_s_43.shape[0]
shape_p =  p_s_43.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_s_43 =  np.delete(x_s_43, [k for k in range(nece_shp,shp)], None)
x_s_43 = x_s_43.reshape(-1,sensors)
shape_x_s_43 = x_s_43.shape[0]
x_s_43 = preprocessing.normalize(x_s_43, axis=0)
x_s_43= np.reshape(x_s_43, (-1,segement_time_size, sensors))
y_s_43= np.delete( y_s_43, [k for k in range(x_s_43.shape[0],shape_y)], None)
p_s_43 = np.delete(p_s_43,[k for k in range(x_s_43.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_s_43), axis=0)
y= np.concatenate((y,y_s_43), axis=0)
p= np.concatenate((p,p_s_43), axis=0)
shp=(x_s_44.shape)[0]
shape_y = y_s_44.shape[0]
shape_p =  p_s_44.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_s_44 =  np.delete(x_s_44, [k for k in range(nece_shp,shp)], None)
x_s_44 = x_s_44.reshape(-1,sensors)
shape_x_s_44 = x_s_44.shape[0]
x_s_44 = preprocessing.normalize(x_s_44, axis=0)
x_s_44= np.reshape(x_s_44, (-1,segement_time_size, sensors))
y_s_44= np.delete( y_s_44, [k for k in range(x_s_44.shape[0],shape_y)], None)
p_s_44 = np.delete(p_s_44,[k for k in range(x_s_44.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_s_44), axis=0)
y= np.concatenate((y,y_s_44), axis=0)
p= np.concatenate((p,p_s_44), axis=0)
shp=(x_s_45.shape)[0]
shape_y = y_s_45.shape[0]
shape_p =  p_s_45.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_s_45 =  np.delete(x_s_45, [k for k in range(nece_shp,shp)], None)
x_s_45 = x_s_45.reshape(-1,sensors)
shape_x_s_45 = x_s_45.shape[0]
x_s_45 = preprocessing.normalize(x_s_45, axis=0)
x_s_45= np.reshape(x_s_45, (-1,segement_time_size, sensors))
y_s_45= np.delete( y_s_45, [k for k in range(x_s_45.shape[0],shape_y)], None)
p_s_45 = np.delete(p_s_45,[k for k in range(x_s_45.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_s_45), axis=0)
y= np.concatenate((y,y_s_45), axis=0)
p= np.concatenate((p,p_s_45), axis=0)
shp=(x_s_46.shape)[0]
shape_y = y_s_46.shape[0]
shape_p =  p_s_46.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_s_46 =  np.delete(x_s_46, [k for k in range(nece_shp,shp)], None)
x_s_46 = x_s_46.reshape(-1,sensors)
shape_x_s_46 = x_s_46.shape[0]
x_s_46 = preprocessing.normalize(x_s_46, axis=0)
x_s_46= np.reshape(x_s_46, (-1,segement_time_size, sensors))
y_s_46= np.delete( y_s_46, [k for k in range(x_s_46.shape[0],shape_y)], None)
p_s_46 = np.delete(p_s_46,[k for k in range(x_s_46.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_s_46), axis=0)
y= np.concatenate((y,y_s_46), axis=0)
p= np.concatenate((p,p_s_46), axis=0)
shp=(x_s_47.shape)[0]
shape_y = y_s_47.shape[0]
shape_p =  p_s_47.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_s_47 =  np.delete(x_s_47, [k for k in range(nece_shp,shp)], None)
x_s_47 = x_s_47.reshape(-1,sensors)
shape_x_s_47 = x_s_47.shape[0]
x_s_47 = preprocessing.normalize(x_s_47, axis=0)
x_s_47= np.reshape(x_s_47, (-1,segement_time_size, sensors))
y_s_47= np.delete( y_s_47, [k for k in range(x_s_47.shape[0],shape_y)], None)
p_s_47 = np.delete(p_s_47,[k for k in range(x_s_47.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_s_47), axis=0)
y= np.concatenate((y,y_s_47), axis=0)
p= np.concatenate((p,p_s_47), axis=0)
shp=(x_s_48.shape)[0]
shape_y = y_s_48.shape[0]
shape_p =  p_s_48.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_s_48 =  np.delete(x_s_48, [k for k in range(nece_shp,shp)], None)
x_s_48 = x_s_48.reshape(-1,sensors)
shape_x_s_48 = x_s_48.shape[0]
x_s_48 = preprocessing.normalize(x_s_48, axis=0)
x_s_48= np.reshape(x_s_48, (-1,segement_time_size, sensors))
y_s_48= np.delete( y_s_48, [k for k in range(x_s_48.shape[0],shape_y)], None)
p_s_48 = np.delete(p_s_48,[k for k in range(x_s_48.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_s_48), axis=0)
y= np.concatenate((y,y_s_48), axis=0)
p= np.concatenate((p,p_s_48), axis=0)
shp=(x_s_49.shape)[0]
shape_y = y_s_49.shape[0]
shape_p =  p_s_49.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_s_49 =  np.delete(x_s_49, [k for k in range(nece_shp,shp)], None)
x_s_49 = x_s_49.reshape(-1,sensors)
shape_x_s_49 = x_s_49.shape[0]
x_s_49 = preprocessing.normalize(x_s_49, axis=0)
x_s_49= np.reshape(x_s_49, (-1,segement_time_size, sensors))
y_s_49= np.delete( y_s_49, [k for k in range(x_s_49.shape[0],shape_y)], None)
p_s_49 = np.delete(p_s_49,[k for k in range(x_s_49.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_s_49), axis=0)
y= np.concatenate((y,y_s_49), axis=0)
p= np.concatenate((p,p_s_49), axis=0)
shp=(x_s_50.shape)[0]
shape_y = y_s_50.shape[0]
shape_p =  p_s_50.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_s_50 =  np.delete(x_s_50, [k for k in range(nece_shp,shp)], None)
x_s_50 = x_s_50.reshape(-1,sensors)
shape_x_s_50 = x_s_50.shape[0]
x_s_50 = preprocessing.normalize(x_s_50, axis=0)
x_s_50= np.reshape(x_s_50, (-1,segement_time_size, sensors))
y_s_50= np.delete( y_s_50, [k for k in range(x_s_50.shape[0],shape_y)], None)
p_s_50 = np.delete(p_s_50,[k for k in range(x_s_50.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_s_50), axis=0)
y= np.concatenate((y,y_s_50), axis=0)
p= np.concatenate((p,p_s_50), axis=0)
shp=(x_s_51.shape)[0]
shape_y = y_s_51.shape[0]
shape_p =  p_s_51.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_s_51 =  np.delete(x_s_51, [k for k in range(nece_shp,shp)], None)
x_s_51 = x_s_51.reshape(-1,sensors)
shape_x_s_51 = x_s_51.shape[0]
x_s_51 = preprocessing.normalize(x_s_51, axis=0)
x_s_51= np.reshape(x_s_51, (-1,segement_time_size, sensors))
y_s_51= np.delete( y_s_51, [k for k in range(x_s_51.shape[0],shape_y)], None)
p_s_51 = np.delete(p_s_51,[k for k in range(x_s_51.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_s_51), axis=0)
y= np.concatenate((y,y_s_51), axis=0)
p= np.concatenate((p,p_s_51), axis=0)
shp=(x_s_52.shape)[0]
shape_y = y_s_52.shape[0]
shape_p =  p_s_52.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_s_52 =  np.delete(x_s_52, [k for k in range(nece_shp,shp)], None)
x_s_52 = x_s_52.reshape(-1,sensors)
shape_x_s_52 = x_s_52.shape[0]
x_s_52 = preprocessing.normalize(x_s_52, axis=0)
x_s_52= np.reshape(x_s_52, (-1,segement_time_size, sensors))
y_s_52= np.delete( y_s_52, [k for k in range(x_s_52.shape[0],shape_y)], None)
p_s_52 = np.delete(p_s_52,[k for k in range(x_s_52.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_s_52), axis=0)
y= np.concatenate((y,y_s_52), axis=0)
p= np.concatenate((p,p_s_52), axis=0)
shp=(x_s_53.shape)[0]
shape_y = y_s_53.shape[0]
shape_p =  p_s_53.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_s_53 =  np.delete(x_s_53, [k for k in range(nece_shp,shp)], None)
x_s_53 = x_s_53.reshape(-1,sensors)
shape_x_s_53 = x_s_53.shape[0]
x_s_53 = preprocessing.normalize(x_s_53, axis=0)
x_s_53= np.reshape(x_s_53, (-1,segement_time_size, sensors))
y_s_53= np.delete( y_s_53, [k for k in range(x_s_53.shape[0],shape_y)], None)
p_s_53 = np.delete(p_s_53,[k for k in range(x_s_53.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_s_53), axis=0)
y= np.concatenate((y,y_s_53), axis=0)
p= np.concatenate((p,p_s_53), axis=0)
shp=(x_s_54.shape)[0]
shape_y = y_s_54.shape[0]
shape_p =  p_s_54.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_s_54 =  np.delete(x_s_54, [k for k in range(nece_shp,shp)], None)
x_s_54 = x_s_54.reshape(-1,sensors)
shape_x_s_54 = x_s_54.shape[0]
x_s_54 = preprocessing.normalize(x_s_54, axis=0)
x_s_54= np.reshape(x_s_54, (-1,segement_time_size, sensors))
y_s_54= np.delete( y_s_54, [k for k in range(x_s_54.shape[0],shape_y)], None)
p_s_54 = np.delete(p_s_54,[k for k in range(x_s_54.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_s_54), axis=0)
y= np.concatenate((y,y_s_54), axis=0)
p= np.concatenate((p,p_s_54), axis=0)
shp=(x_s_55.shape)[0]
shape_y = y_s_55.shape[0]
shape_p =  p_s_55.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_s_55 =  np.delete(x_s_55, [k for k in range(nece_shp,shp)], None)
x_s_55 = x_s_55.reshape(-1,sensors)
shape_x_s_55 = x_s_55.shape[0]
x_s_55 = preprocessing.normalize(x_s_55, axis=0)
x_s_55= np.reshape(x_s_55, (-1,segement_time_size, sensors))
y_s_55= np.delete( y_s_55, [k for k in range(x_s_55.shape[0],shape_y)], None)
p_s_55 = np.delete(p_s_55,[k for k in range(x_s_55.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_s_55), axis=0)
y= np.concatenate((y,y_s_55), axis=0)
p= np.concatenate((p,p_s_55), axis=0)
shp=(x_s_56.shape)[0]
shape_y = y_s_56.shape[0]
shape_p =  p_s_56.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_s_56 =  np.delete(x_s_56, [k for k in range(nece_shp,shp)], None)
x_s_56 = x_s_56.reshape(-1,sensors)
shape_x_s_56 = x_s_56.shape[0]
x_s_56 = preprocessing.normalize(x_s_56, axis=0)
x_s_56= np.reshape(x_s_56, (-1,segement_time_size, sensors))
y_s_56= np.delete( y_s_56, [k for k in range(x_s_56.shape[0],shape_y)], None)
p_s_56 = np.delete(p_s_56,[k for k in range(x_s_56.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_s_56), axis=0)
y= np.concatenate((y,y_s_56), axis=0)
p= np.concatenate((p,p_s_56), axis=0)
shp=(x_s_57.shape)[0]
shape_y = y_s_57.shape[0]
shape_p =  p_s_57.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_s_57 =  np.delete(x_s_57, [k for k in range(nece_shp,shp)], None)
x_s_57 = x_s_57.reshape(-1,sensors)
shape_x_s_57 = x_s_57.shape[0]
x_s_57 = preprocessing.normalize(x_s_57, axis=0)
x_s_57= np.reshape(x_s_57, (-1,segement_time_size, sensors))
y_s_57= np.delete( y_s_57, [k for k in range(x_s_57.shape[0],shape_y)], None)
p_s_57 = np.delete(p_s_57,[k for k in range(x_s_57.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_s_57), axis=0)
y= np.concatenate((y,y_s_57), axis=0)
p= np.concatenate((p,p_s_57), axis=0)
shp=(x_s_58.shape)[0]
shape_y = y_s_58.shape[0]
shape_p =  p_s_58.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_s_58 =  np.delete(x_s_58, [k for k in range(nece_shp,shp)], None)
x_s_58 = x_s_58.reshape(-1,sensors)
shape_x_s_58 = x_s_58.shape[0]
x_s_58 = preprocessing.normalize(x_s_58, axis=0)
x_s_58= np.reshape(x_s_58, (-1,segement_time_size, sensors))
y_s_58= np.delete( y_s_58, [k for k in range(x_s_58.shape[0],shape_y)], None)
p_s_58 = np.delete(p_s_58,[k for k in range(x_s_58.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_s_58), axis=0)
y= np.concatenate((y,y_s_58), axis=0)
p= np.concatenate((p,p_s_58), axis=0)
shp=(x_s_59.shape)[0]
shape_y = y_s_59.shape[0]
shape_p =  p_s_59.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_s_59 =  np.delete(x_s_59, [k for k in range(nece_shp,shp)], None)
x_s_59 = x_s_59.reshape(-1,sensors)
shape_x_s_59 = x_s_59.shape[0]
x_s_59 = preprocessing.normalize(x_s_59, axis=0)
x_s_59= np.reshape(x_s_59, (-1,segement_time_size, sensors))
y_s_59= np.delete( y_s_59, [k for k in range(x_s_59.shape[0],shape_y)], None)
p_s_59 = np.delete(p_s_59,[k for k in range(x_s_59.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_s_59), axis=0)
y= np.concatenate((y,y_s_59), axis=0)
p= np.concatenate((p,p_s_59), axis=0)
shp=(x_s_60.shape)[0]
shape_y = y_s_60.shape[0]
shape_p =  p_s_60.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_s_60 =  np.delete(x_s_60, [k for k in range(nece_shp,shp)], None)
x_s_60 = x_s_60.reshape(-1,sensors)
shape_x_s_60 = x_s_60.shape[0]
x_s_60 = preprocessing.normalize(x_s_60, axis=0)
x_s_60= np.reshape(x_s_60, (-1,segement_time_size, sensors))
y_s_60= np.delete( y_s_60, [k for k in range(x_s_60.shape[0],shape_y)], None)
p_s_60 = np.delete(p_s_60,[k for k in range(x_s_60.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_s_60), axis=0)
y= np.concatenate((y,y_s_60), axis=0)
p= np.concatenate((p,p_s_60), axis=0)
shp=(x_s_61.shape)[0]
shape_y = y_s_61.shape[0]
shape_p =  p_s_61.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_s_61 =  np.delete(x_s_61, [k for k in range(nece_shp,shp)], None)
x_s_61 = x_s_61.reshape(-1,sensors)
shape_x_s_61 = x_s_61.shape[0]
x_s_61 = preprocessing.normalize(x_s_61, axis=0)
x_s_61= np.reshape(x_s_61, (-1,segement_time_size, sensors))
y_s_61= np.delete( y_s_61, [k for k in range(x_s_61.shape[0],shape_y)], None)
p_s_61 = np.delete(p_s_61,[k for k in range(x_s_61.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_s_61), axis=0)
y= np.concatenate((y,y_s_61), axis=0)
p= np.concatenate((p,p_s_61), axis=0)
shp=(x_s_62.shape)[0]
shape_y = y_s_62.shape[0]
shape_p =  p_s_62.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_s_62 =  np.delete(x_s_62, [k for k in range(nece_shp,shp)], None)
x_s_62 = x_s_62.reshape(-1,sensors)
shape_x_s_62 = x_s_62.shape[0]
x_s_62 = preprocessing.normalize(x_s_62, axis=0)
x_s_62= np.reshape(x_s_62, (-1,segement_time_size, sensors))
y_s_62= np.delete( y_s_62, [k for k in range(x_s_62.shape[0],shape_y)], None)
p_s_62 = np.delete(p_s_62,[k for k in range(x_s_62.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_s_62), axis=0)
y= np.concatenate((y,y_s_62), axis=0)
p= np.concatenate((p,p_s_62), axis=0)
shp=(x_s_63.shape)[0]
shape_y = y_s_63.shape[0]
shape_p =  p_s_63.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_s_63 =  np.delete(x_s_63, [k for k in range(nece_shp,shp)], None)
x_s_63 = x_s_63.reshape(-1,sensors)
shape_x_s_63 = x_s_63.shape[0]
x_s_63 = preprocessing.normalize(x_s_63, axis=0)
x_s_63= np.reshape(x_s_63, (-1,segement_time_size, sensors))
y_s_63= np.delete( y_s_63, [k for k in range(x_s_63.shape[0],shape_y)], None)
p_s_63 = np.delete(p_s_63,[k for k in range(x_s_63.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_s_63), axis=0)
y= np.concatenate((y,y_s_63), axis=0)
p= np.concatenate((p,p_s_63), axis=0)
shp=(x_s_64.shape)[0]
shape_y = y_s_64.shape[0]
shape_p =  p_s_64.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_s_64 =  np.delete(x_s_64, [k for k in range(nece_shp,shp)], None)
x_s_64 = x_s_64.reshape(-1,sensors)
shape_x_s_64 = x_s_64.shape[0]
x_s_64 = preprocessing.normalize(x_s_64, axis=0)
x_s_64= np.reshape(x_s_64, (-1,segement_time_size, sensors))
y_s_64= np.delete( y_s_64, [k for k in range(x_s_64.shape[0],shape_y)], None)
p_s_64 = np.delete(p_s_64,[k for k in range(x_s_64.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_s_64), axis=0)
y= np.concatenate((y,y_s_64), axis=0)
p= np.concatenate((p,p_s_64), axis=0)
shp=(x_s_65.shape)[0]
shape_y = y_s_65.shape[0]
shape_p =  p_s_65.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_s_65 =  np.delete(x_s_65, [k for k in range(nece_shp,shp)], None)
x_s_65 = x_s_65.reshape(-1,sensors)
shape_x_s_65 = x_s_65.shape[0]
x_s_65 = preprocessing.normalize(x_s_65, axis=0)
x_s_65= np.reshape(x_s_65, (-1,segement_time_size, sensors))
y_s_65= np.delete( y_s_65, [k for k in range(x_s_65.shape[0],shape_y)], None)
p_s_65 = np.delete(p_s_65,[k for k in range(x_s_65.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_s_65), axis=0)
y= np.concatenate((y,y_s_65), axis=0)
p= np.concatenate((p,p_s_65), axis=0)
shp=(x_s_66.shape)[0]
shape_y = y_s_66.shape[0]
shape_p =  p_s_66.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_s_66 =  np.delete(x_s_66, [k for k in range(nece_shp,shp)], None)
x_s_66 = x_s_66.reshape(-1,sensors)
shape_x_s_66 = x_s_66.shape[0]
x_s_66 = preprocessing.normalize(x_s_66, axis=0)
x_s_66= np.reshape(x_s_66, (-1,segement_time_size, sensors))
y_s_66= np.delete( y_s_66, [k for k in range(x_s_66.shape[0],shape_y)], None)
p_s_66 = np.delete(p_s_66,[k for k in range(x_s_66.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_s_66), axis=0)
y= np.concatenate((y,y_s_66), axis=0)
p= np.concatenate((p,p_s_66), axis=0)
shp=(x_s_67.shape)[0]
shape_y = y_s_67.shape[0]
shape_p =  p_s_67.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_s_67 =  np.delete(x_s_67, [k for k in range(nece_shp,shp)], None)
x_s_67 = x_s_67.reshape(-1,sensors)
shape_x_s_67 = x_s_67.shape[0]
x_s_67 = preprocessing.normalize(x_s_67, axis=0)
x_s_67= np.reshape(x_s_67, (-1,segement_time_size, sensors))
y_s_67= np.delete( y_s_67, [k for k in range(x_s_67.shape[0],shape_y)], None)
p_s_67 = np.delete(p_s_67,[k for k in range(x_s_67.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_s_67), axis=0)
y= np.concatenate((y,y_s_67), axis=0)
p= np.concatenate((p,p_s_67), axis=0)
shp=(x_s_68.shape)[0]
shape_y = y_s_68.shape[0]
shape_p =  p_s_68.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_s_68 =  np.delete(x_s_68, [k for k in range(nece_shp,shp)], None)
x_s_68 = x_s_68.reshape(-1,sensors)
shape_x_s_68 = x_s_68.shape[0]
x_s_68 = preprocessing.normalize(x_s_68, axis=0)
x_s_68= np.reshape(x_s_68, (-1,segement_time_size, sensors))
y_s_68= np.delete( y_s_68, [k for k in range(x_s_68.shape[0],shape_y)], None)
p_s_68 = np.delete(p_s_68,[k for k in range(x_s_68.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_s_68), axis=0)
y= np.concatenate((y,y_s_68), axis=0)
p= np.concatenate((p,p_s_68), axis=0)
shp=(x_s_69.shape)[0]
shape_y = y_s_69.shape[0]
shape_p =  p_s_69.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_s_69 =  np.delete(x_s_69, [k for k in range(nece_shp,shp)], None)
x_s_69 = x_s_69.reshape(-1,sensors)
shape_x_s_69 = x_s_69.shape[0]
x_s_69 = preprocessing.normalize(x_s_69, axis=0)
x_s_69= np.reshape(x_s_69, (-1,segement_time_size, sensors))
y_s_69= np.delete( y_s_69, [k for k in range(x_s_69.shape[0],shape_y)], None)
p_s_69 = np.delete(p_s_69,[k for k in range(x_s_69.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_s_69), axis=0)
y= np.concatenate((y,y_s_69), axis=0)
p= np.concatenate((p,p_s_69), axis=0)
shp=(x_s_70.shape)[0]
shape_y = y_s_70.shape[0]
shape_p =  p_s_70.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_s_70 =  np.delete(x_s_70, [k for k in range(nece_shp,shp)], None)
x_s_70 = x_s_70.reshape(-1,sensors)
shape_x_s_70 = x_s_70.shape[0]
x_s_70 = preprocessing.normalize(x_s_70, axis=0)
x_s_70= np.reshape(x_s_70, (-1,segement_time_size, sensors))
y_s_70= np.delete( y_s_70, [k for k in range(x_s_70.shape[0],shape_y)], None)
p_s_70 = np.delete(p_s_70,[k for k in range(x_s_70.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_s_70), axis=0)
y= np.concatenate((y,y_s_70), axis=0)
p= np.concatenate((p,p_s_70), axis=0)
shp=(x_s_71.shape)[0]
shape_y = y_s_71.shape[0]
shape_p =  p_s_71.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_s_71 =  np.delete(x_s_71, [k for k in range(nece_shp,shp)], None)
x_s_71 = x_s_71.reshape(-1,sensors)
shape_x_s_71 = x_s_71.shape[0]
x_s_71 = preprocessing.normalize(x_s_71, axis=0)
x_s_71= np.reshape(x_s_71, (-1,segement_time_size, sensors))
y_s_71= np.delete( y_s_71, [k for k in range(x_s_71.shape[0],shape_y)], None)
p_s_71 = np.delete(p_s_71,[k for k in range(x_s_71.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_s_71), axis=0)
y= np.concatenate((y,y_s_71), axis=0)
p= np.concatenate((p,p_s_71), axis=0)
shp=(x_s_72.shape)[0]
shape_y = y_s_72.shape[0]
shape_p =  p_s_72.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_s_72 =  np.delete(x_s_72, [k for k in range(nece_shp,shp)], None)
x_s_72 = x_s_72.reshape(-1,sensors)
shape_x_s_72 = x_s_72.shape[0]
x_s_72 = preprocessing.normalize(x_s_72, axis=0)
x_s_72= np.reshape(x_s_72, (-1,segement_time_size, sensors))
y_s_72= np.delete( y_s_72, [k for k in range(x_s_72.shape[0],shape_y)], None)
p_s_72 = np.delete(p_s_72,[k for k in range(x_s_72.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_s_72), axis=0)
y= np.concatenate((y,y_s_72), axis=0)
p= np.concatenate((p,p_s_72), axis=0)
shp=(x_s_73.shape)[0]
shape_y = y_s_73.shape[0]
shape_p =  p_s_73.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_s_73 =  np.delete(x_s_73, [k for k in range(nece_shp,shp)], None)
x_s_73 = x_s_73.reshape(-1,sensors)
shape_x_s_73 = x_s_73.shape[0]
x_s_73 = preprocessing.normalize(x_s_73, axis=0)
x_s_73= np.reshape(x_s_73, (-1,segement_time_size, sensors))
y_s_73= np.delete( y_s_73, [k for k in range(x_s_73.shape[0],shape_y)], None)
p_s_73 = np.delete(p_s_73,[k for k in range(x_s_73.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_s_73), axis=0)
y= np.concatenate((y,y_s_73), axis=0)
p= np.concatenate((p,p_s_73), axis=0)
shp=(x_s_74.shape)[0]
shape_y = y_s_74.shape[0]
shape_p =  p_s_74.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_s_74 =  np.delete(x_s_74, [k for k in range(nece_shp,shp)], None)
x_s_74 = x_s_74.reshape(-1,sensors)
shape_x_s_74 = x_s_74.shape[0]
x_s_74 = preprocessing.normalize(x_s_74, axis=0)
x_s_74= np.reshape(x_s_74, (-1,segement_time_size, sensors))
y_s_74= np.delete( y_s_74, [k for k in range(x_s_74.shape[0],shape_y)], None)
p_s_74 = np.delete(p_s_74,[k for k in range(x_s_74.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_s_74), axis=0)
y= np.concatenate((y,y_s_74), axis=0)
p= np.concatenate((p,p_s_74), axis=0)
shp=(x_s_75.shape)[0]
shape_y = y_s_75.shape[0]
shape_p =  p_s_75.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_s_75 =  np.delete(x_s_75, [k for k in range(nece_shp,shp)], None)
x_s_75 = x_s_75.reshape(-1,sensors)
shape_x_s_75 = x_s_75.shape[0]
x_s_75 = preprocessing.normalize(x_s_75, axis=0)
x_s_75= np.reshape(x_s_75, (-1,segement_time_size, sensors))
y_s_75= np.delete( y_s_75, [k for k in range(x_s_75.shape[0],shape_y)], None)
p_s_75 = np.delete(p_s_75,[k for k in range(x_s_75.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_s_75), axis=0)
y= np.concatenate((y,y_s_75), axis=0)
p= np.concatenate((p,p_s_75), axis=0)
shp=(x_s_76.shape)[0]
shape_y = y_s_76.shape[0]
shape_p =  p_s_76.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_s_76 =  np.delete(x_s_76, [k for k in range(nece_shp,shp)], None)
x_s_76 = x_s_76.reshape(-1,sensors)
shape_x_s_76 = x_s_76.shape[0]
x_s_76 = preprocessing.normalize(x_s_76, axis=0)
x_s_76= np.reshape(x_s_76, (-1,segement_time_size, sensors))
y_s_76= np.delete( y_s_76, [k for k in range(x_s_76.shape[0],shape_y)], None)
p_s_76 = np.delete(p_s_76,[k for k in range(x_s_76.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_s_76), axis=0)
y= np.concatenate((y,y_s_76), axis=0)
p= np.concatenate((p,p_s_76), axis=0)
shp=(x_s_77.shape)[0]
shape_y = y_s_77.shape[0]
shape_p =  p_s_77.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_s_77 =  np.delete(x_s_77, [k for k in range(nece_shp,shp)], None)
x_s_77 = x_s_77.reshape(-1,sensors)
shape_x_s_77 = x_s_77.shape[0]
x_s_77 = preprocessing.normalize(x_s_77, axis=0)
x_s_77= np.reshape(x_s_77, (-1,segement_time_size, sensors))
y_s_77= np.delete( y_s_77, [k for k in range(x_s_77.shape[0],shape_y)], None)
p_s_77 = np.delete(p_s_77,[k for k in range(x_s_77.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_s_77), axis=0)
y= np.concatenate((y,y_s_77), axis=0)
p= np.concatenate((p,p_s_77), axis=0)
shp=(x_s_78.shape)[0]
shape_y = y_s_78.shape[0]
shape_p =  p_s_78.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_s_78 =  np.delete(x_s_78, [k for k in range(nece_shp,shp)], None)
x_s_78 = x_s_78.reshape(-1,sensors)
shape_x_s_78 = x_s_78.shape[0]
x_s_78 = preprocessing.normalize(x_s_78, axis=0)
x_s_78= np.reshape(x_s_78, (-1,segement_time_size, sensors))
y_s_78= np.delete( y_s_78, [k for k in range(x_s_78.shape[0],shape_y)], None)
p_s_78 = np.delete(p_s_78,[k for k in range(x_s_78.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_s_78), axis=0)
y= np.concatenate((y,y_s_78), axis=0)
p= np.concatenate((p,p_s_78), axis=0)
shp=(x_s_79.shape)[0]
shape_y = y_s_79.shape[0]
shape_p =  p_s_79.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_s_79 =  np.delete(x_s_79, [k for k in range(nece_shp,shp)], None)
x_s_79 = x_s_79.reshape(-1,sensors)
shape_x_s_79 = x_s_79.shape[0]
x_s_79 = preprocessing.normalize(x_s_79, axis=0)
x_s_79= np.reshape(x_s_79, (-1,segement_time_size, sensors))
y_s_79= np.delete( y_s_79, [k for k in range(x_s_79.shape[0],shape_y)], None)
p_s_79 = np.delete(p_s_79,[k for k in range(x_s_79.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_s_79), axis=0)
y= np.concatenate((y,y_s_79), axis=0)
p= np.concatenate((p,p_s_79), axis=0)
shp=(x_s_80.shape)[0]
shape_y = y_s_80.shape[0]
shape_p =  p_s_80.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_s_80 =  np.delete(x_s_80, [k for k in range(nece_shp,shp)], None)
x_s_80 = x_s_80.reshape(-1,sensors)
shape_x_s_80 = x_s_80.shape[0]
x_s_80 = preprocessing.normalize(x_s_80, axis=0)
x_s_80= np.reshape(x_s_80, (-1,segement_time_size, sensors))
y_s_80= np.delete( y_s_80, [k for k in range(x_s_80.shape[0],shape_y)], None)
p_s_80 = np.delete(p_s_80,[k for k in range(x_s_80.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_s_80), axis=0)
y= np.concatenate((y,y_s_80), axis=0)
p= np.concatenate((p,p_s_80), axis=0)
shp=(x_s_81.shape)[0]
shape_y = y_s_81.shape[0]
shape_p =  p_s_81.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_s_81 =  np.delete(x_s_81, [k for k in range(nece_shp,shp)], None)
x_s_81 = x_s_81.reshape(-1,sensors)
shape_x_s_81 = x_s_81.shape[0]
x_s_81 = preprocessing.normalize(x_s_81, axis=0)
x_s_81= np.reshape(x_s_81, (-1,segement_time_size, sensors))
y_s_81= np.delete( y_s_81, [k for k in range(x_s_81.shape[0],shape_y)], None)
p_s_81 = np.delete(p_s_81,[k for k in range(x_s_81.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_s_81), axis=0)
y= np.concatenate((y,y_s_81), axis=0)
p= np.concatenate((p,p_s_81), axis=0)
shp=(x_s_82.shape)[0]
shape_y = y_s_82.shape[0]
shape_p =  p_s_82.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_s_82 =  np.delete(x_s_82, [k for k in range(nece_shp,shp)], None)
x_s_82 = x_s_82.reshape(-1,sensors)
shape_x_s_82 = x_s_82.shape[0]
x_s_82 = preprocessing.normalize(x_s_82, axis=0)
x_s_82= np.reshape(x_s_82, (-1,segement_time_size, sensors))
y_s_82= np.delete( y_s_82, [k for k in range(x_s_82.shape[0],shape_y)], None)
p_s_82 = np.delete(p_s_82,[k for k in range(x_s_82.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_s_82), axis=0)
y= np.concatenate((y,y_s_82), axis=0)
p= np.concatenate((p,p_s_82), axis=0)
shp=(x_s_83.shape)[0]
shape_y = y_s_83.shape[0]
shape_p =  p_s_83.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_s_83 =  np.delete(x_s_83, [k for k in range(nece_shp,shp)], None)
x_s_83 = x_s_83.reshape(-1,sensors)
shape_x_s_83 = x_s_83.shape[0]
x_s_83 = preprocessing.normalize(x_s_83, axis=0)
x_s_83= np.reshape(x_s_83, (-1,segement_time_size, sensors))
y_s_83= np.delete( y_s_83, [k for k in range(x_s_83.shape[0],shape_y)], None)
p_s_83 = np.delete(p_s_83,[k for k in range(x_s_83.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_s_83), axis=0)
y= np.concatenate((y,y_s_83), axis=0)
p= np.concatenate((p,p_s_83), axis=0)
shp=(x_s_84.shape)[0]
shape_y = y_s_84.shape[0]
shape_p =  p_s_84.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_s_84 =  np.delete(x_s_84, [k for k in range(nece_shp,shp)], None)
x_s_84 = x_s_84.reshape(-1,sensors)
shape_x_s_84 = x_s_84.shape[0]
x_s_84 = preprocessing.normalize(x_s_84, axis=0)
x_s_84= np.reshape(x_s_84, (-1,segement_time_size, sensors))
y_s_84= np.delete( y_s_84, [k for k in range(x_s_84.shape[0],shape_y)], None)
p_s_84 = np.delete(p_s_84,[k for k in range(x_s_84.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_s_84), axis=0)
y= np.concatenate((y,y_s_84), axis=0)
p= np.concatenate((p,p_s_84), axis=0)
shp=(x_s_85.shape)[0]
shape_y = y_s_85.shape[0]
shape_p =  p_s_85.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_s_85 =  np.delete(x_s_85, [k for k in range(nece_shp,shp)], None)
x_s_85 = x_s_85.reshape(-1,sensors)
shape_x_s_85 = x_s_85.shape[0]
x_s_85 = preprocessing.normalize(x_s_85, axis=0)
x_s_85= np.reshape(x_s_85, (-1,segement_time_size, sensors))
y_s_85= np.delete( y_s_85, [k for k in range(x_s_85.shape[0],shape_y)], None)
p_s_85 = np.delete(p_s_85,[k for k in range(x_s_85.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_s_85), axis=0)
y= np.concatenate((y,y_s_85), axis=0)
p= np.concatenate((p,p_s_85), axis=0)
shp=(x_s_86.shape)[0]
shape_y = y_s_86.shape[0]
shape_p =  p_s_86.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_s_86 =  np.delete(x_s_86, [k for k in range(nece_shp,shp)], None)
x_s_86 = x_s_86.reshape(-1,sensors)
shape_x_s_86 = x_s_86.shape[0]
x_s_86 = preprocessing.normalize(x_s_86, axis=0)
x_s_86= np.reshape(x_s_86, (-1,segement_time_size, sensors))
y_s_86= np.delete( y_s_86, [k for k in range(x_s_86.shape[0],shape_y)], None)
p_s_86 = np.delete(p_s_86,[k for k in range(x_s_86.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_s_86), axis=0)
y= np.concatenate((y,y_s_86), axis=0)
p= np.concatenate((p,p_s_86), axis=0)
shp=(x_s_87.shape)[0]
shape_y = y_s_87.shape[0]
shape_p =  p_s_87.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_s_87 =  np.delete(x_s_87, [k for k in range(nece_shp,shp)], None)
x_s_87 = x_s_87.reshape(-1,sensors)
shape_x_s_87 = x_s_87.shape[0]
x_s_87 = preprocessing.normalize(x_s_87, axis=0)
x_s_87= np.reshape(x_s_87, (-1,segement_time_size, sensors))
y_s_87= np.delete( y_s_87, [k for k in range(x_s_87.shape[0],shape_y)], None)
p_s_87 = np.delete(p_s_87,[k for k in range(x_s_87.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_s_87), axis=0)
y= np.concatenate((y,y_s_87), axis=0)
p= np.concatenate((p,p_s_87), axis=0)
shp=(x_s_88.shape)[0]
shape_y = y_s_88.shape[0]
shape_p =  p_s_88.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_s_88 =  np.delete(x_s_88, [k for k in range(nece_shp,shp)], None)
x_s_88 = x_s_88.reshape(-1,sensors)
shape_x_s_88 = x_s_88.shape[0]
x_s_88 = preprocessing.normalize(x_s_88, axis=0)
x_s_88= np.reshape(x_s_88, (-1,segement_time_size, sensors))
y_s_88= np.delete( y_s_88, [k for k in range(x_s_88.shape[0],shape_y)], None)
p_s_88 = np.delete(p_s_88,[k for k in range(x_s_88.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_s_88), axis=0)
y= np.concatenate((y,y_s_88), axis=0)
p= np.concatenate((p,p_s_88), axis=0)
shp=(x_s_89.shape)[0]
shape_y = y_s_89.shape[0]
shape_p =  p_s_89.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_s_89 =  np.delete(x_s_89, [k for k in range(nece_shp,shp)], None)
x_s_89 = x_s_89.reshape(-1,sensors)
shape_x_s_89 = x_s_89.shape[0]
x_s_89 = preprocessing.normalize(x_s_89, axis=0)
x_s_89= np.reshape(x_s_89, (-1,segement_time_size, sensors))
y_s_89= np.delete( y_s_89, [k for k in range(x_s_89.shape[0],shape_y)], None)
p_s_89 = np.delete(p_s_89,[k for k in range(x_s_89.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_s_89), axis=0)
y= np.concatenate((y,y_s_89), axis=0)
p= np.concatenate((p,p_s_89), axis=0)
shp=(x_s_90.shape)[0]
shape_y = y_s_90.shape[0]
shape_p =  p_s_90.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_s_90 =  np.delete(x_s_90, [k for k in range(nece_shp,shp)], None)
x_s_90 = x_s_90.reshape(-1,sensors)
shape_x_s_90 = x_s_90.shape[0]
x_s_90 = preprocessing.normalize(x_s_90, axis=0)
x_s_90= np.reshape(x_s_90, (-1,segement_time_size, sensors))
y_s_90= np.delete( y_s_90, [k for k in range(x_s_90.shape[0],shape_y)], None)
p_s_90 = np.delete(p_s_90,[k for k in range(x_s_90.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_s_90), axis=0)
y= np.concatenate((y,y_s_90), axis=0)
p= np.concatenate((p,p_s_90), axis=0)
shp=(x_s_91.shape)[0]
shape_y = y_s_91.shape[0]
shape_p =  p_s_91.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_s_91 =  np.delete(x_s_91, [k for k in range(nece_shp,shp)], None)
x_s_91 = x_s_91.reshape(-1,sensors)
shape_x_s_91 = x_s_91.shape[0]
x_s_91 = preprocessing.normalize(x_s_91, axis=0)
x_s_91= np.reshape(x_s_91, (-1,segement_time_size, sensors))
y_s_91= np.delete( y_s_91, [k for k in range(x_s_91.shape[0],shape_y)], None)
p_s_91 = np.delete(p_s_91,[k for k in range(x_s_91.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_s_91), axis=0)
y= np.concatenate((y,y_s_91), axis=0)
p= np.concatenate((p,p_s_91), axis=0)
shp=(x_s_92.shape)[0]
shape_y = y_s_92.shape[0]
shape_p =  p_s_92.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_s_92 =  np.delete(x_s_92, [k for k in range(nece_shp,shp)], None)
x_s_92 = x_s_92.reshape(-1,sensors)
shape_x_s_92 = x_s_92.shape[0]
x_s_92 = preprocessing.normalize(x_s_92, axis=0)
x_s_92= np.reshape(x_s_92, (-1,segement_time_size, sensors))
y_s_92= np.delete( y_s_92, [k for k in range(x_s_92.shape[0],shape_y)], None)
p_s_92 = np.delete(p_s_92,[k for k in range(x_s_92.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_s_92), axis=0)
y= np.concatenate((y,y_s_92), axis=0)
p= np.concatenate((p,p_s_92), axis=0)
shp=(x_s_93.shape)[0]
shape_y = y_s_93.shape[0]
shape_p =  p_s_93.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_s_93 =  np.delete(x_s_93, [k for k in range(nece_shp,shp)], None)
x_s_93 = x_s_93.reshape(-1,sensors)
shape_x_s_93 = x_s_93.shape[0]
x_s_93 = preprocessing.normalize(x_s_93, axis=0)
x_s_93= np.reshape(x_s_93, (-1,segement_time_size, sensors))
y_s_93= np.delete( y_s_93, [k for k in range(x_s_93.shape[0],shape_y)], None)
p_s_93 = np.delete(p_s_93,[k for k in range(x_s_93.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_s_93), axis=0)
y= np.concatenate((y,y_s_93), axis=0)
p= np.concatenate((p,p_s_93), axis=0)
shp=(x_s_94.shape)[0]
shape_y = y_s_94.shape[0]
shape_p =  p_s_94.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_s_94 =  np.delete(x_s_94, [k for k in range(nece_shp,shp)], None)
x_s_94 = x_s_94.reshape(-1,sensors)
shape_x_s_94 = x_s_94.shape[0]
x_s_94 = preprocessing.normalize(x_s_94, axis=0)
x_s_94= np.reshape(x_s_94, (-1,segement_time_size, sensors))
y_s_94= np.delete( y_s_94, [k for k in range(x_s_94.shape[0],shape_y)], None)
p_s_94 = np.delete(p_s_94,[k for k in range(x_s_94.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_s_94), axis=0)
y= np.concatenate((y,y_s_94), axis=0)
p= np.concatenate((p,p_s_94), axis=0)
shp=(x_s_95.shape)[0]
shape_y = y_s_95.shape[0]
shape_p =  p_s_95.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_s_95 =  np.delete(x_s_95, [k for k in range(nece_shp,shp)], None)
x_s_95 = x_s_95.reshape(-1,sensors)
shape_x_s_95 = x_s_95.shape[0]
x_s_95 = preprocessing.normalize(x_s_95, axis=0)
x_s_95= np.reshape(x_s_95, (-1,segement_time_size, sensors))
y_s_95= np.delete( y_s_95, [k for k in range(x_s_95.shape[0],shape_y)], None)
p_s_95 = np.delete(p_s_95,[k for k in range(x_s_95.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_s_95), axis=0)
y= np.concatenate((y,y_s_95), axis=0)
p= np.concatenate((p,p_s_95), axis=0)
shp=(x_s_96.shape)[0]
shape_y = y_s_96.shape[0]
shape_p =  p_s_96.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_s_96 =  np.delete(x_s_96, [k for k in range(nece_shp,shp)], None)
x_s_96 = x_s_96.reshape(-1,sensors)
shape_x_s_96 = x_s_96.shape[0]
x_s_96 = preprocessing.normalize(x_s_96, axis=0)
x_s_96= np.reshape(x_s_96, (-1,segement_time_size, sensors))
y_s_96= np.delete( y_s_96, [k for k in range(x_s_96.shape[0],shape_y)], None)
p_s_96 = np.delete(p_s_96,[k for k in range(x_s_96.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_s_96), axis=0)
y= np.concatenate((y,y_s_96), axis=0)
p= np.concatenate((p,p_s_96), axis=0)
shp=(x_s_97.shape)[0]
shape_y = y_s_97.shape[0]
shape_p =  p_s_97.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_s_97 =  np.delete(x_s_97, [k for k in range(nece_shp,shp)], None)
x_s_97 = x_s_97.reshape(-1,sensors)
shape_x_s_97 = x_s_97.shape[0]
x_s_97 = preprocessing.normalize(x_s_97, axis=0)
x_s_97= np.reshape(x_s_97, (-1,segement_time_size, sensors))
y_s_97= np.delete( y_s_97, [k for k in range(x_s_97.shape[0],shape_y)], None)
p_s_97 = np.delete(p_s_97,[k for k in range(x_s_97.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_s_97), axis=0)
y= np.concatenate((y,y_s_97), axis=0)
p= np.concatenate((p,p_s_97), axis=0)
shp=(x_s_98.shape)[0]
shape_y = y_s_98.shape[0]
shape_p =  p_s_98.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_s_98 =  np.delete(x_s_98, [k for k in range(nece_shp,shp)], None)
x_s_98 = x_s_98.reshape(-1,sensors)
shape_x_s_98 = x_s_98.shape[0]
x_s_98 = preprocessing.normalize(x_s_98, axis=0)
x_s_98= np.reshape(x_s_98, (-1,segement_time_size, sensors))
y_s_98= np.delete( y_s_98, [k for k in range(x_s_98.shape[0],shape_y)], None)
p_s_98 = np.delete(p_s_98,[k for k in range(x_s_98.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_s_98), axis=0)
y= np.concatenate((y,y_s_98), axis=0)
p= np.concatenate((p,p_s_98), axis=0)
shp=(x_s_99.shape)[0]
shape_y = y_s_99.shape[0]
shape_p =  p_s_99.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_s_99 =  np.delete(x_s_99, [k for k in range(nece_shp,shp)], None)
x_s_99 = x_s_99.reshape(-1,sensors)
shape_x_s_99 = x_s_99.shape[0]
x_s_99 = preprocessing.normalize(x_s_99, axis=0)
x_s_99= np.reshape(x_s_99, (-1,segement_time_size, sensors))
y_s_99= np.delete( y_s_99, [k for k in range(x_s_99.shape[0],shape_y)], None)
p_s_99 = np.delete(p_s_99,[k for k in range(x_s_99.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_s_99), axis=0)
y= np.concatenate((y,y_s_99), axis=0)
p= np.concatenate((p,p_s_99), axis=0)
shp=(x_w_0.shape)[0]
shape_y = y_w_0.shape[0]
shape_p =  p_w_0.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_w_0 =  np.delete(x_w_0, [k for k in range(nece_shp,shp)], None)
x_w_0 = x_w_0.reshape(-1,sensors)
shape_x_w_0 = x_w_0.shape[0]
x_w_0 = preprocessing.normalize(x_w_0, axis=0)
x_w_0= np.reshape(x_w_0, (-1,segement_time_size, sensors))
y_w_0= np.delete( y_w_0, [k for k in range(x_w_0.shape[0],shape_y)], None)
p_w_0 = np.delete(p_w_0,[k for k in range(x_w_0.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_w_0), axis=0)
y= np.concatenate((y,y_w_0), axis=0)
p= np.concatenate((p,p_w_0), axis=0)
shp=(x_w_1.shape)[0]
shape_y = y_w_1.shape[0]
shape_p =  p_w_1.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_w_1 =  np.delete(x_w_1, [k for k in range(nece_shp,shp)], None)
x_w_1 = x_w_1.reshape(-1,sensors)
shape_x_w_1 = x_w_1.shape[0]
x_w_1 = preprocessing.normalize(x_w_1, axis=0)
x_w_1= np.reshape(x_w_1, (-1,segement_time_size, sensors))
y_w_1= np.delete( y_w_1, [k for k in range(x_w_1.shape[0],shape_y)], None)
p_w_1 = np.delete(p_w_1,[k for k in range(x_w_1.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_w_1), axis=0)
y= np.concatenate((y,y_w_1), axis=0)
p= np.concatenate((p,p_w_1), axis=0)
shp=(x_w_2.shape)[0]
shape_y = y_w_2.shape[0]
shape_p =  p_w_2.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_w_2 =  np.delete(x_w_2, [k for k in range(nece_shp,shp)], None)
x_w_2 = x_w_2.reshape(-1,sensors)
shape_x_w_2 = x_w_2.shape[0]
x_w_2 = preprocessing.normalize(x_w_2, axis=0)
x_w_2= np.reshape(x_w_2, (-1,segement_time_size, sensors))
y_w_2= np.delete( y_w_2, [k for k in range(x_w_2.shape[0],shape_y)], None)
p_w_2 = np.delete(p_w_2,[k for k in range(x_w_2.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_w_2), axis=0)
y= np.concatenate((y,y_w_2), axis=0)
p= np.concatenate((p,p_w_2), axis=0)
shp=(x_w_3.shape)[0]
shape_y = y_w_3.shape[0]
shape_p =  p_w_3.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_w_3 =  np.delete(x_w_3, [k for k in range(nece_shp,shp)], None)
x_w_3 = x_w_3.reshape(-1,sensors)
shape_x_w_3 = x_w_3.shape[0]
x_w_3 = preprocessing.normalize(x_w_3, axis=0)
x_w_3= np.reshape(x_w_3, (-1,segement_time_size, sensors))
y_w_3= np.delete( y_w_3, [k for k in range(x_w_3.shape[0],shape_y)], None)
p_w_3 = np.delete(p_w_3,[k for k in range(x_w_3.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_w_3), axis=0)
y= np.concatenate((y,y_w_3), axis=0)
p= np.concatenate((p,p_w_3), axis=0)
shp=(x_w_4.shape)[0]
shape_y = y_w_4.shape[0]
shape_p =  p_w_4.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_w_4 =  np.delete(x_w_4, [k for k in range(nece_shp,shp)], None)
x_w_4 = x_w_4.reshape(-1,sensors)
shape_x_w_4 = x_w_4.shape[0]
x_w_4 = preprocessing.normalize(x_w_4, axis=0)
x_w_4= np.reshape(x_w_4, (-1,segement_time_size, sensors))
y_w_4= np.delete( y_w_4, [k for k in range(x_w_4.shape[0],shape_y)], None)
p_w_4 = np.delete(p_w_4,[k for k in range(x_w_4.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_w_4), axis=0)
y= np.concatenate((y,y_w_4), axis=0)
p= np.concatenate((p,p_w_4), axis=0)
shp=(x_w_5.shape)[0]
shape_y = y_w_5.shape[0]
shape_p =  p_w_5.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_w_5 =  np.delete(x_w_5, [k for k in range(nece_shp,shp)], None)
x_w_5 = x_w_5.reshape(-1,sensors)
shape_x_w_5 = x_w_5.shape[0]
x_w_5 = preprocessing.normalize(x_w_5, axis=0)
x_w_5= np.reshape(x_w_5, (-1,segement_time_size, sensors))
y_w_5= np.delete( y_w_5, [k for k in range(x_w_5.shape[0],shape_y)], None)
p_w_5 = np.delete(p_w_5,[k for k in range(x_w_5.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_w_5), axis=0)
y= np.concatenate((y,y_w_5), axis=0)
p= np.concatenate((p,p_w_5), axis=0)
shp=(x_w_6.shape)[0]
shape_y = y_w_6.shape[0]
shape_p =  p_w_6.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_w_6 =  np.delete(x_w_6, [k for k in range(nece_shp,shp)], None)
x_w_6 = x_w_6.reshape(-1,sensors)
shape_x_w_6 = x_w_6.shape[0]
x_w_6 = preprocessing.normalize(x_w_6, axis=0)
x_w_6= np.reshape(x_w_6, (-1,segement_time_size, sensors))
y_w_6= np.delete( y_w_6, [k for k in range(x_w_6.shape[0],shape_y)], None)
p_w_6 = np.delete(p_w_6,[k for k in range(x_w_6.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_w_6), axis=0)
y= np.concatenate((y,y_w_6), axis=0)
p= np.concatenate((p,p_w_6), axis=0)
shp=(x_w_7.shape)[0]
shape_y = y_w_7.shape[0]
shape_p =  p_w_7.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_w_7 =  np.delete(x_w_7, [k for k in range(nece_shp,shp)], None)
x_w_7 = x_w_7.reshape(-1,sensors)
shape_x_w_7 = x_w_7.shape[0]
x_w_7 = preprocessing.normalize(x_w_7, axis=0)
x_w_7= np.reshape(x_w_7, (-1,segement_time_size, sensors))
y_w_7= np.delete( y_w_7, [k for k in range(x_w_7.shape[0],shape_y)], None)
p_w_7 = np.delete(p_w_7,[k for k in range(x_w_7.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_w_7), axis=0)
y= np.concatenate((y,y_w_7), axis=0)
p= np.concatenate((p,p_w_7), axis=0)
shp=(x_w_8.shape)[0]
shape_y = y_w_8.shape[0]
shape_p =  p_w_8.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_w_8 =  np.delete(x_w_8, [k for k in range(nece_shp,shp)], None)
x_w_8 = x_w_8.reshape(-1,sensors)
shape_x_w_8 = x_w_8.shape[0]
x_w_8 = preprocessing.normalize(x_w_8, axis=0)
x_w_8= np.reshape(x_w_8, (-1,segement_time_size, sensors))
y_w_8= np.delete( y_w_8, [k for k in range(x_w_8.shape[0],shape_y)], None)
p_w_8 = np.delete(p_w_8,[k for k in range(x_w_8.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_w_8), axis=0)
y= np.concatenate((y,y_w_8), axis=0)
p= np.concatenate((p,p_w_8), axis=0)
shp=(x_w_9.shape)[0]
shape_y = y_w_9.shape[0]
shape_p =  p_w_9.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_w_9 =  np.delete(x_w_9, [k for k in range(nece_shp,shp)], None)
x_w_9 = x_w_9.reshape(-1,sensors)
shape_x_w_9 = x_w_9.shape[0]
x_w_9 = preprocessing.normalize(x_w_9, axis=0)
x_w_9= np.reshape(x_w_9, (-1,segement_time_size, sensors))
y_w_9= np.delete( y_w_9, [k for k in range(x_w_9.shape[0],shape_y)], None)
p_w_9 = np.delete(p_w_9,[k for k in range(x_w_9.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_w_9), axis=0)
y= np.concatenate((y,y_w_9), axis=0)
p= np.concatenate((p,p_w_9), axis=0)
shp=(x_w_10.shape)[0]
shape_y = y_w_10.shape[0]
shape_p =  p_w_10.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_w_10 =  np.delete(x_w_10, [k for k in range(nece_shp,shp)], None)
x_w_10 = x_w_10.reshape(-1,sensors)
shape_x_w_10 = x_w_10.shape[0]
x_w_10 = preprocessing.normalize(x_w_10, axis=0)
x_w_10= np.reshape(x_w_10, (-1,segement_time_size, sensors))
y_w_10= np.delete( y_w_10, [k for k in range(x_w_10.shape[0],shape_y)], None)
p_w_10 = np.delete(p_w_10,[k for k in range(x_w_10.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_w_10), axis=0)
y= np.concatenate((y,y_w_10), axis=0)
p= np.concatenate((p,p_w_10), axis=0)
shp=(x_w_11.shape)[0]
shape_y = y_w_11.shape[0]
shape_p =  p_w_11.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_w_11 =  np.delete(x_w_11, [k for k in range(nece_shp,shp)], None)
x_w_11 = x_w_11.reshape(-1,sensors)
shape_x_w_11 = x_w_11.shape[0]
x_w_11 = preprocessing.normalize(x_w_11, axis=0)
x_w_11= np.reshape(x_w_11, (-1,segement_time_size, sensors))
y_w_11= np.delete( y_w_11, [k for k in range(x_w_11.shape[0],shape_y)], None)
p_w_11 = np.delete(p_w_11,[k for k in range(x_w_11.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_w_11), axis=0)
y= np.concatenate((y,y_w_11), axis=0)
p= np.concatenate((p,p_w_11), axis=0)
shp=(x_w_12.shape)[0]
shape_y = y_w_12.shape[0]
shape_p =  p_w_12.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_w_12 =  np.delete(x_w_12, [k for k in range(nece_shp,shp)], None)
x_w_12 = x_w_12.reshape(-1,sensors)
shape_x_w_12 = x_w_12.shape[0]
x_w_12 = preprocessing.normalize(x_w_12, axis=0)
x_w_12= np.reshape(x_w_12, (-1,segement_time_size, sensors))
y_w_12= np.delete( y_w_12, [k for k in range(x_w_12.shape[0],shape_y)], None)
p_w_12 = np.delete(p_w_12,[k for k in range(x_w_12.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_w_12), axis=0)
y= np.concatenate((y,y_w_12), axis=0)
p= np.concatenate((p,p_w_12), axis=0)
shp=(x_w_13.shape)[0]
shape_y = y_w_13.shape[0]
shape_p =  p_w_13.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_w_13 =  np.delete(x_w_13, [k for k in range(nece_shp,shp)], None)
x_w_13 = x_w_13.reshape(-1,sensors)
shape_x_w_13 = x_w_13.shape[0]
x_w_13 = preprocessing.normalize(x_w_13, axis=0)
x_w_13= np.reshape(x_w_13, (-1,segement_time_size, sensors))
y_w_13= np.delete( y_w_13, [k for k in range(x_w_13.shape[0],shape_y)], None)
p_w_13 = np.delete(p_w_13,[k for k in range(x_w_13.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_w_13), axis=0)
y= np.concatenate((y,y_w_13), axis=0)
p= np.concatenate((p,p_w_13), axis=0)
shp=(x_w_14.shape)[0]
shape_y = y_w_14.shape[0]
shape_p =  p_w_14.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_w_14 =  np.delete(x_w_14, [k for k in range(nece_shp,shp)], None)
x_w_14 = x_w_14.reshape(-1,sensors)
shape_x_w_14 = x_w_14.shape[0]
x_w_14 = preprocessing.normalize(x_w_14, axis=0)
x_w_14= np.reshape(x_w_14, (-1,segement_time_size, sensors))
y_w_14= np.delete( y_w_14, [k for k in range(x_w_14.shape[0],shape_y)], None)
p_w_14 = np.delete(p_w_14,[k for k in range(x_w_14.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_w_14), axis=0)
y= np.concatenate((y,y_w_14), axis=0)
p= np.concatenate((p,p_w_14), axis=0)
shp=(x_w_15.shape)[0]
shape_y = y_w_15.shape[0]
shape_p =  p_w_15.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_w_15 =  np.delete(x_w_15, [k for k in range(nece_shp,shp)], None)
x_w_15 = x_w_15.reshape(-1,sensors)
shape_x_w_15 = x_w_15.shape[0]
x_w_15 = preprocessing.normalize(x_w_15, axis=0)
x_w_15= np.reshape(x_w_15, (-1,segement_time_size, sensors))
y_w_15= np.delete( y_w_15, [k for k in range(x_w_15.shape[0],shape_y)], None)
p_w_15 = np.delete(p_w_15,[k for k in range(x_w_15.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_w_15), axis=0)
y= np.concatenate((y,y_w_15), axis=0)
p= np.concatenate((p,p_w_15), axis=0)
shp=(x_w_16.shape)[0]
shape_y = y_w_16.shape[0]
shape_p =  p_w_16.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_w_16 =  np.delete(x_w_16, [k for k in range(nece_shp,shp)], None)
x_w_16 = x_w_16.reshape(-1,sensors)
shape_x_w_16 = x_w_16.shape[0]
x_w_16 = preprocessing.normalize(x_w_16, axis=0)
x_w_16= np.reshape(x_w_16, (-1,segement_time_size, sensors))
y_w_16= np.delete( y_w_16, [k for k in range(x_w_16.shape[0],shape_y)], None)
p_w_16 = np.delete(p_w_16,[k for k in range(x_w_16.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_w_16), axis=0)
y= np.concatenate((y,y_w_16), axis=0)
p= np.concatenate((p,p_w_16), axis=0)
shp=(x_w_17.shape)[0]
shape_y = y_w_17.shape[0]
shape_p =  p_w_17.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_w_17 =  np.delete(x_w_17, [k for k in range(nece_shp,shp)], None)
x_w_17 = x_w_17.reshape(-1,sensors)
shape_x_w_17 = x_w_17.shape[0]
x_w_17 = preprocessing.normalize(x_w_17, axis=0)
x_w_17= np.reshape(x_w_17, (-1,segement_time_size, sensors))
y_w_17= np.delete( y_w_17, [k for k in range(x_w_17.shape[0],shape_y)], None)
p_w_17 = np.delete(p_w_17,[k for k in range(x_w_17.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_w_17), axis=0)
y= np.concatenate((y,y_w_17), axis=0)
p= np.concatenate((p,p_w_17), axis=0)
shp=(x_w_18.shape)[0]
shape_y = y_w_18.shape[0]
shape_p =  p_w_18.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_w_18 =  np.delete(x_w_18, [k for k in range(nece_shp,shp)], None)
x_w_18 = x_w_18.reshape(-1,sensors)
shape_x_w_18 = x_w_18.shape[0]
x_w_18 = preprocessing.normalize(x_w_18, axis=0)
x_w_18= np.reshape(x_w_18, (-1,segement_time_size, sensors))
y_w_18= np.delete( y_w_18, [k for k in range(x_w_18.shape[0],shape_y)], None)
p_w_18 = np.delete(p_w_18,[k for k in range(x_w_18.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_w_18), axis=0)
y= np.concatenate((y,y_w_18), axis=0)
p= np.concatenate((p,p_w_18), axis=0)
shp=(x_w_19.shape)[0]
shape_y = y_w_19.shape[0]
shape_p =  p_w_19.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_w_19 =  np.delete(x_w_19, [k for k in range(nece_shp,shp)], None)
x_w_19 = x_w_19.reshape(-1,sensors)
shape_x_w_19 = x_w_19.shape[0]
x_w_19 = preprocessing.normalize(x_w_19, axis=0)
x_w_19= np.reshape(x_w_19, (-1,segement_time_size, sensors))
y_w_19= np.delete( y_w_19, [k for k in range(x_w_19.shape[0],shape_y)], None)
p_w_19 = np.delete(p_w_19,[k for k in range(x_w_19.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_w_19), axis=0)
y= np.concatenate((y,y_w_19), axis=0)
p= np.concatenate((p,p_w_19), axis=0)
shp=(x_w_20.shape)[0]
shape_y = y_w_20.shape[0]
shape_p =  p_w_20.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_w_20 =  np.delete(x_w_20, [k for k in range(nece_shp,shp)], None)
x_w_20 = x_w_20.reshape(-1,sensors)
shape_x_w_20 = x_w_20.shape[0]
x_w_20 = preprocessing.normalize(x_w_20, axis=0)
x_w_20= np.reshape(x_w_20, (-1,segement_time_size, sensors))
y_w_20= np.delete( y_w_20, [k for k in range(x_w_20.shape[0],shape_y)], None)
p_w_20 = np.delete(p_w_20,[k for k in range(x_w_20.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_w_20), axis=0)
y= np.concatenate((y,y_w_20), axis=0)
p= np.concatenate((p,p_w_20), axis=0)
shp=(x_w_21.shape)[0]
shape_y = y_w_21.shape[0]
shape_p =  p_w_21.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_w_21 =  np.delete(x_w_21, [k for k in range(nece_shp,shp)], None)
x_w_21 = x_w_21.reshape(-1,sensors)
shape_x_w_21 = x_w_21.shape[0]
x_w_21 = preprocessing.normalize(x_w_21, axis=0)
x_w_21= np.reshape(x_w_21, (-1,segement_time_size, sensors))
y_w_21= np.delete( y_w_21, [k for k in range(x_w_21.shape[0],shape_y)], None)
p_w_21 = np.delete(p_w_21,[k for k in range(x_w_21.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_w_21), axis=0)
y= np.concatenate((y,y_w_21), axis=0)
p= np.concatenate((p,p_w_21), axis=0)
shp=(x_w_22.shape)[0]
shape_y = y_w_22.shape[0]
shape_p =  p_w_22.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_w_22 =  np.delete(x_w_22, [k for k in range(nece_shp,shp)], None)
x_w_22 = x_w_22.reshape(-1,sensors)
shape_x_w_22 = x_w_22.shape[0]
x_w_22 = preprocessing.normalize(x_w_22, axis=0)
x_w_22= np.reshape(x_w_22, (-1,segement_time_size, sensors))
y_w_22= np.delete( y_w_22, [k for k in range(x_w_22.shape[0],shape_y)], None)
p_w_22 = np.delete(p_w_22,[k for k in range(x_w_22.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_w_22), axis=0)
y= np.concatenate((y,y_w_22), axis=0)
p= np.concatenate((p,p_w_22), axis=0)
shp=(x_w_23.shape)[0]
shape_y = y_w_23.shape[0]
shape_p =  p_w_23.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_w_23 =  np.delete(x_w_23, [k for k in range(nece_shp,shp)], None)
x_w_23 = x_w_23.reshape(-1,sensors)
shape_x_w_23 = x_w_23.shape[0]
x_w_23 = preprocessing.normalize(x_w_23, axis=0)
x_w_23= np.reshape(x_w_23, (-1,segement_time_size, sensors))
y_w_23= np.delete( y_w_23, [k for k in range(x_w_23.shape[0],shape_y)], None)
p_w_23 = np.delete(p_w_23,[k for k in range(x_w_23.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_w_23), axis=0)
y= np.concatenate((y,y_w_23), axis=0)
p= np.concatenate((p,p_w_23), axis=0)
shp=(x_w_24.shape)[0]
shape_y = y_w_24.shape[0]
shape_p =  p_w_24.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_w_24 =  np.delete(x_w_24, [k for k in range(nece_shp,shp)], None)
x_w_24 = x_w_24.reshape(-1,sensors)
shape_x_w_24 = x_w_24.shape[0]
x_w_24 = preprocessing.normalize(x_w_24, axis=0)
x_w_24= np.reshape(x_w_24, (-1,segement_time_size, sensors))
y_w_24= np.delete( y_w_24, [k for k in range(x_w_24.shape[0],shape_y)], None)
p_w_24 = np.delete(p_w_24,[k for k in range(x_w_24.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_w_24), axis=0)
y= np.concatenate((y,y_w_24), axis=0)
p= np.concatenate((p,p_w_24), axis=0)
shp=(x_w_25.shape)[0]
shape_y = y_w_25.shape[0]
shape_p =  p_w_25.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_w_25 =  np.delete(x_w_25, [k for k in range(nece_shp,shp)], None)
x_w_25 = x_w_25.reshape(-1,sensors)
shape_x_w_25 = x_w_25.shape[0]
x_w_25 = preprocessing.normalize(x_w_25, axis=0)
x_w_25= np.reshape(x_w_25, (-1,segement_time_size, sensors))
y_w_25= np.delete( y_w_25, [k for k in range(x_w_25.shape[0],shape_y)], None)
p_w_25 = np.delete(p_w_25,[k for k in range(x_w_25.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_w_25), axis=0)
y= np.concatenate((y,y_w_25), axis=0)
p= np.concatenate((p,p_w_25), axis=0)
shp=(x_w_26.shape)[0]
shape_y = y_w_26.shape[0]
shape_p =  p_w_26.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_w_26 =  np.delete(x_w_26, [k for k in range(nece_shp,shp)], None)
x_w_26 = x_w_26.reshape(-1,sensors)
shape_x_w_26 = x_w_26.shape[0]
x_w_26 = preprocessing.normalize(x_w_26, axis=0)
x_w_26= np.reshape(x_w_26, (-1,segement_time_size, sensors))
y_w_26= np.delete( y_w_26, [k for k in range(x_w_26.shape[0],shape_y)], None)
p_w_26 = np.delete(p_w_26,[k for k in range(x_w_26.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_w_26), axis=0)
y= np.concatenate((y,y_w_26), axis=0)
p= np.concatenate((p,p_w_26), axis=0)
shp=(x_w_27.shape)[0]
shape_y = y_w_27.shape[0]
shape_p =  p_w_27.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_w_27 =  np.delete(x_w_27, [k for k in range(nece_shp,shp)], None)
x_w_27 = x_w_27.reshape(-1,sensors)
shape_x_w_27 = x_w_27.shape[0]
x_w_27 = preprocessing.normalize(x_w_27, axis=0)
x_w_27= np.reshape(x_w_27, (-1,segement_time_size, sensors))
y_w_27= np.delete( y_w_27, [k for k in range(x_w_27.shape[0],shape_y)], None)
p_w_27 = np.delete(p_w_27,[k for k in range(x_w_27.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_w_27), axis=0)
y= np.concatenate((y,y_w_27), axis=0)
p= np.concatenate((p,p_w_27), axis=0)
shp=(x_w_28.shape)[0]
shape_y = y_w_28.shape[0]
shape_p =  p_w_28.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_w_28 =  np.delete(x_w_28, [k for k in range(nece_shp,shp)], None)
x_w_28 = x_w_28.reshape(-1,sensors)
shape_x_w_28 = x_w_28.shape[0]
x_w_28 = preprocessing.normalize(x_w_28, axis=0)
x_w_28= np.reshape(x_w_28, (-1,segement_time_size, sensors))
y_w_28= np.delete( y_w_28, [k for k in range(x_w_28.shape[0],shape_y)], None)
p_w_28 = np.delete(p_w_28,[k for k in range(x_w_28.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_w_28), axis=0)
y= np.concatenate((y,y_w_28), axis=0)
p= np.concatenate((p,p_w_28), axis=0)
shp=(x_w_29.shape)[0]
shape_y = y_w_29.shape[0]
shape_p =  p_w_29.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_w_29 =  np.delete(x_w_29, [k for k in range(nece_shp,shp)], None)
x_w_29 = x_w_29.reshape(-1,sensors)
shape_x_w_29 = x_w_29.shape[0]
x_w_29 = preprocessing.normalize(x_w_29, axis=0)
x_w_29= np.reshape(x_w_29, (-1,segement_time_size, sensors))
y_w_29= np.delete( y_w_29, [k for k in range(x_w_29.shape[0],shape_y)], None)
p_w_29 = np.delete(p_w_29,[k for k in range(x_w_29.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_w_29), axis=0)
y= np.concatenate((y,y_w_29), axis=0)
p= np.concatenate((p,p_w_29), axis=0)
shp=(x_w_30.shape)[0]
shape_y = y_w_30.shape[0]
shape_p =  p_w_30.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_w_30 =  np.delete(x_w_30, [k for k in range(nece_shp,shp)], None)
x_w_30 = x_w_30.reshape(-1,sensors)
shape_x_w_30 = x_w_30.shape[0]
x_w_30 = preprocessing.normalize(x_w_30, axis=0)
x_w_30= np.reshape(x_w_30, (-1,segement_time_size, sensors))
y_w_30= np.delete( y_w_30, [k for k in range(x_w_30.shape[0],shape_y)], None)
p_w_30 = np.delete(p_w_30,[k for k in range(x_w_30.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_w_30), axis=0)
y= np.concatenate((y,y_w_30), axis=0)
p= np.concatenate((p,p_w_30), axis=0)
shp=(x_w_31.shape)[0]
shape_y = y_w_31.shape[0]
shape_p =  p_w_31.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_w_31 =  np.delete(x_w_31, [k for k in range(nece_shp,shp)], None)
x_w_31 = x_w_31.reshape(-1,sensors)
shape_x_w_31 = x_w_31.shape[0]
x_w_31 = preprocessing.normalize(x_w_31, axis=0)
x_w_31= np.reshape(x_w_31, (-1,segement_time_size, sensors))
y_w_31= np.delete( y_w_31, [k for k in range(x_w_31.shape[0],shape_y)], None)
p_w_31 = np.delete(p_w_31,[k for k in range(x_w_31.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_w_31), axis=0)
y= np.concatenate((y,y_w_31), axis=0)
p= np.concatenate((p,p_w_31), axis=0)
shp=(x_w_32.shape)[0]
shape_y = y_w_32.shape[0]
shape_p =  p_w_32.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_w_32 =  np.delete(x_w_32, [k for k in range(nece_shp,shp)], None)
x_w_32 = x_w_32.reshape(-1,sensors)
shape_x_w_32 = x_w_32.shape[0]
x_w_32 = preprocessing.normalize(x_w_32, axis=0)
x_w_32= np.reshape(x_w_32, (-1,segement_time_size, sensors))
y_w_32= np.delete( y_w_32, [k for k in range(x_w_32.shape[0],shape_y)], None)
p_w_32 = np.delete(p_w_32,[k for k in range(x_w_32.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_w_32), axis=0)
y= np.concatenate((y,y_w_32), axis=0)
p= np.concatenate((p,p_w_32), axis=0)
shp=(x_w_33.shape)[0]
shape_y = y_w_33.shape[0]
shape_p =  p_w_33.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_w_33 =  np.delete(x_w_33, [k for k in range(nece_shp,shp)], None)
x_w_33 = x_w_33.reshape(-1,sensors)
shape_x_w_33 = x_w_33.shape[0]
x_w_33 = preprocessing.normalize(x_w_33, axis=0)
x_w_33= np.reshape(x_w_33, (-1,segement_time_size, sensors))
y_w_33= np.delete( y_w_33, [k for k in range(x_w_33.shape[0],shape_y)], None)
p_w_33 = np.delete(p_w_33,[k for k in range(x_w_33.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_w_33), axis=0)
y= np.concatenate((y,y_w_33), axis=0)
p= np.concatenate((p,p_w_33), axis=0)
shp=(x_w_34.shape)[0]
shape_y = y_w_34.shape[0]
shape_p =  p_w_34.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_w_34 =  np.delete(x_w_34, [k for k in range(nece_shp,shp)], None)
x_w_34 = x_w_34.reshape(-1,sensors)
shape_x_w_34 = x_w_34.shape[0]
x_w_34 = preprocessing.normalize(x_w_34, axis=0)
x_w_34= np.reshape(x_w_34, (-1,segement_time_size, sensors))
y_w_34= np.delete( y_w_34, [k for k in range(x_w_34.shape[0],shape_y)], None)
p_w_34 = np.delete(p_w_34,[k for k in range(x_w_34.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_w_34), axis=0)
y= np.concatenate((y,y_w_34), axis=0)
p= np.concatenate((p,p_w_34), axis=0)
shp=(x_w_35.shape)[0]
shape_y = y_w_35.shape[0]
shape_p =  p_w_35.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_w_35 =  np.delete(x_w_35, [k for k in range(nece_shp,shp)], None)
x_w_35 = x_w_35.reshape(-1,sensors)
shape_x_w_35 = x_w_35.shape[0]
x_w_35 = preprocessing.normalize(x_w_35, axis=0)
x_w_35= np.reshape(x_w_35, (-1,segement_time_size, sensors))
y_w_35= np.delete( y_w_35, [k for k in range(x_w_35.shape[0],shape_y)], None)
p_w_35 = np.delete(p_w_35,[k for k in range(x_w_35.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_w_35), axis=0)
y= np.concatenate((y,y_w_35), axis=0)
p= np.concatenate((p,p_w_35), axis=0)
shp=(x_w_36.shape)[0]
shape_y = y_w_36.shape[0]
shape_p =  p_w_36.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_w_36 =  np.delete(x_w_36, [k for k in range(nece_shp,shp)], None)
x_w_36 = x_w_36.reshape(-1,sensors)
shape_x_w_36 = x_w_36.shape[0]
x_w_36 = preprocessing.normalize(x_w_36, axis=0)
x_w_36= np.reshape(x_w_36, (-1,segement_time_size, sensors))
y_w_36= np.delete( y_w_36, [k for k in range(x_w_36.shape[0],shape_y)], None)
p_w_36 = np.delete(p_w_36,[k for k in range(x_w_36.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_w_36), axis=0)
y= np.concatenate((y,y_w_36), axis=0)
p= np.concatenate((p,p_w_36), axis=0)
shp=(x_w_37.shape)[0]
shape_y = y_w_37.shape[0]
shape_p =  p_w_37.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_w_37 =  np.delete(x_w_37, [k for k in range(nece_shp,shp)], None)
x_w_37 = x_w_37.reshape(-1,sensors)
shape_x_w_37 = x_w_37.shape[0]
x_w_37 = preprocessing.normalize(x_w_37, axis=0)
x_w_37= np.reshape(x_w_37, (-1,segement_time_size, sensors))
y_w_37= np.delete( y_w_37, [k for k in range(x_w_37.shape[0],shape_y)], None)
p_w_37 = np.delete(p_w_37,[k for k in range(x_w_37.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_w_37), axis=0)
y= np.concatenate((y,y_w_37), axis=0)
p= np.concatenate((p,p_w_37), axis=0)
shp=(x_w_38.shape)[0]
shape_y = y_w_38.shape[0]
shape_p =  p_w_38.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_w_38 =  np.delete(x_w_38, [k for k in range(nece_shp,shp)], None)
x_w_38 = x_w_38.reshape(-1,sensors)
shape_x_w_38 = x_w_38.shape[0]
x_w_38 = preprocessing.normalize(x_w_38, axis=0)
x_w_38= np.reshape(x_w_38, (-1,segement_time_size, sensors))
y_w_38= np.delete( y_w_38, [k for k in range(x_w_38.shape[0],shape_y)], None)
p_w_38 = np.delete(p_w_38,[k for k in range(x_w_38.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_w_38), axis=0)
y= np.concatenate((y,y_w_38), axis=0)
p= np.concatenate((p,p_w_38), axis=0)
shp=(x_w_39.shape)[0]
shape_y = y_w_39.shape[0]
shape_p =  p_w_39.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_w_39 =  np.delete(x_w_39, [k for k in range(nece_shp,shp)], None)
x_w_39 = x_w_39.reshape(-1,sensors)
shape_x_w_39 = x_w_39.shape[0]
x_w_39 = preprocessing.normalize(x_w_39, axis=0)
x_w_39= np.reshape(x_w_39, (-1,segement_time_size, sensors))
y_w_39= np.delete( y_w_39, [k for k in range(x_w_39.shape[0],shape_y)], None)
p_w_39 = np.delete(p_w_39,[k for k in range(x_w_39.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_w_39), axis=0)
y= np.concatenate((y,y_w_39), axis=0)
p= np.concatenate((p,p_w_39), axis=0)
shp=(x_w_40.shape)[0]
shape_y = y_w_40.shape[0]
shape_p =  p_w_40.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_w_40 =  np.delete(x_w_40, [k for k in range(nece_shp,shp)], None)
x_w_40 = x_w_40.reshape(-1,sensors)
shape_x_w_40 = x_w_40.shape[0]
x_w_40 = preprocessing.normalize(x_w_40, axis=0)
x_w_40= np.reshape(x_w_40, (-1,segement_time_size, sensors))
y_w_40= np.delete( y_w_40, [k for k in range(x_w_40.shape[0],shape_y)], None)
p_w_40 = np.delete(p_w_40,[k for k in range(x_w_40.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_w_40), axis=0)
y= np.concatenate((y,y_w_40), axis=0)
p= np.concatenate((p,p_w_40), axis=0)
shp=(x_w_41.shape)[0]
shape_y = y_w_41.shape[0]
shape_p =  p_w_41.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_w_41 =  np.delete(x_w_41, [k for k in range(nece_shp,shp)], None)
x_w_41 = x_w_41.reshape(-1,sensors)
shape_x_w_41 = x_w_41.shape[0]
x_w_41 = preprocessing.normalize(x_w_41, axis=0)
x_w_41= np.reshape(x_w_41, (-1,segement_time_size, sensors))
y_w_41= np.delete( y_w_41, [k for k in range(x_w_41.shape[0],shape_y)], None)
p_w_41 = np.delete(p_w_41,[k for k in range(x_w_41.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_w_41), axis=0)
y= np.concatenate((y,y_w_41), axis=0)
p= np.concatenate((p,p_w_41), axis=0)
shp=(x_w_42.shape)[0]
shape_y = y_w_42.shape[0]
shape_p =  p_w_42.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_w_42 =  np.delete(x_w_42, [k for k in range(nece_shp,shp)], None)
x_w_42 = x_w_42.reshape(-1,sensors)
shape_x_w_42 = x_w_42.shape[0]
x_w_42 = preprocessing.normalize(x_w_42, axis=0)
x_w_42= np.reshape(x_w_42, (-1,segement_time_size, sensors))
y_w_42= np.delete( y_w_42, [k for k in range(x_w_42.shape[0],shape_y)], None)
p_w_42 = np.delete(p_w_42,[k for k in range(x_w_42.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_w_42), axis=0)
y= np.concatenate((y,y_w_42), axis=0)
p= np.concatenate((p,p_w_42), axis=0)
shp=(x_w_43.shape)[0]
shape_y = y_w_43.shape[0]
shape_p =  p_w_43.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_w_43 =  np.delete(x_w_43, [k for k in range(nece_shp,shp)], None)
x_w_43 = x_w_43.reshape(-1,sensors)
shape_x_w_43 = x_w_43.shape[0]
x_w_43 = preprocessing.normalize(x_w_43, axis=0)
x_w_43= np.reshape(x_w_43, (-1,segement_time_size, sensors))
y_w_43= np.delete( y_w_43, [k for k in range(x_w_43.shape[0],shape_y)], None)
p_w_43 = np.delete(p_w_43,[k for k in range(x_w_43.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_w_43), axis=0)
y= np.concatenate((y,y_w_43), axis=0)
p= np.concatenate((p,p_w_43), axis=0)
shp=(x_w_44.shape)[0]
shape_y = y_w_44.shape[0]
shape_p =  p_w_44.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_w_44 =  np.delete(x_w_44, [k for k in range(nece_shp,shp)], None)
x_w_44 = x_w_44.reshape(-1,sensors)
shape_x_w_44 = x_w_44.shape[0]
x_w_44 = preprocessing.normalize(x_w_44, axis=0)
x_w_44= np.reshape(x_w_44, (-1,segement_time_size, sensors))
y_w_44= np.delete( y_w_44, [k for k in range(x_w_44.shape[0],shape_y)], None)
p_w_44 = np.delete(p_w_44,[k for k in range(x_w_44.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_w_44), axis=0)
y= np.concatenate((y,y_w_44), axis=0)
p= np.concatenate((p,p_w_44), axis=0)
shp=(x_w_45.shape)[0]
shape_y = y_w_45.shape[0]
shape_p =  p_w_45.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_w_45 =  np.delete(x_w_45, [k for k in range(nece_shp,shp)], None)
x_w_45 = x_w_45.reshape(-1,sensors)
shape_x_w_45 = x_w_45.shape[0]
x_w_45 = preprocessing.normalize(x_w_45, axis=0)
x_w_45= np.reshape(x_w_45, (-1,segement_time_size, sensors))
y_w_45= np.delete( y_w_45, [k for k in range(x_w_45.shape[0],shape_y)], None)
p_w_45 = np.delete(p_w_45,[k for k in range(x_w_45.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_w_45), axis=0)
y= np.concatenate((y,y_w_45), axis=0)
p= np.concatenate((p,p_w_45), axis=0)
shp=(x_w_46.shape)[0]
shape_y = y_w_46.shape[0]
shape_p =  p_w_46.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_w_46 =  np.delete(x_w_46, [k for k in range(nece_shp,shp)], None)
x_w_46 = x_w_46.reshape(-1,sensors)
shape_x_w_46 = x_w_46.shape[0]
x_w_46 = preprocessing.normalize(x_w_46, axis=0)
x_w_46= np.reshape(x_w_46, (-1,segement_time_size, sensors))
y_w_46= np.delete( y_w_46, [k for k in range(x_w_46.shape[0],shape_y)], None)
p_w_46 = np.delete(p_w_46,[k for k in range(x_w_46.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_w_46), axis=0)
y= np.concatenate((y,y_w_46), axis=0)
p= np.concatenate((p,p_w_46), axis=0)
shp=(x_w_47.shape)[0]
shape_y = y_w_47.shape[0]
shape_p =  p_w_47.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_w_47 =  np.delete(x_w_47, [k for k in range(nece_shp,shp)], None)
x_w_47 = x_w_47.reshape(-1,sensors)
shape_x_w_47 = x_w_47.shape[0]
x_w_47 = preprocessing.normalize(x_w_47, axis=0)
x_w_47= np.reshape(x_w_47, (-1,segement_time_size, sensors))
y_w_47= np.delete( y_w_47, [k for k in range(x_w_47.shape[0],shape_y)], None)
p_w_47 = np.delete(p_w_47,[k for k in range(x_w_47.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_w_47), axis=0)
y= np.concatenate((y,y_w_47), axis=0)
p= np.concatenate((p,p_w_47), axis=0)
shp=(x_w_48.shape)[0]
shape_y = y_w_48.shape[0]
shape_p =  p_w_48.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_w_48 =  np.delete(x_w_48, [k for k in range(nece_shp,shp)], None)
x_w_48 = x_w_48.reshape(-1,sensors)
shape_x_w_48 = x_w_48.shape[0]
x_w_48 = preprocessing.normalize(x_w_48, axis=0)
x_w_48= np.reshape(x_w_48, (-1,segement_time_size, sensors))
y_w_48= np.delete( y_w_48, [k for k in range(x_w_48.shape[0],shape_y)], None)
p_w_48 = np.delete(p_w_48,[k for k in range(x_w_48.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_w_48), axis=0)
y= np.concatenate((y,y_w_48), axis=0)
p= np.concatenate((p,p_w_48), axis=0)
shp=(x_w_49.shape)[0]
shape_y = y_w_49.shape[0]
shape_p =  p_w_49.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_w_49 =  np.delete(x_w_49, [k for k in range(nece_shp,shp)], None)
x_w_49 = x_w_49.reshape(-1,sensors)
shape_x_w_49 = x_w_49.shape[0]
x_w_49 = preprocessing.normalize(x_w_49, axis=0)
x_w_49= np.reshape(x_w_49, (-1,segement_time_size, sensors))
y_w_49= np.delete( y_w_49, [k for k in range(x_w_49.shape[0],shape_y)], None)
p_w_49 = np.delete(p_w_49,[k for k in range(x_w_49.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_w_49), axis=0)
y= np.concatenate((y,y_w_49), axis=0)
p= np.concatenate((p,p_w_49), axis=0)
shp=(x_w_50.shape)[0]
shape_y = y_w_50.shape[0]
shape_p =  p_w_50.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_w_50 =  np.delete(x_w_50, [k for k in range(nece_shp,shp)], None)
x_w_50 = x_w_50.reshape(-1,sensors)
shape_x_w_50 = x_w_50.shape[0]
x_w_50 = preprocessing.normalize(x_w_50, axis=0)
x_w_50= np.reshape(x_w_50, (-1,segement_time_size, sensors))
y_w_50= np.delete( y_w_50, [k for k in range(x_w_50.shape[0],shape_y)], None)
p_w_50 = np.delete(p_w_50,[k for k in range(x_w_50.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_w_50), axis=0)
y= np.concatenate((y,y_w_50), axis=0)
p= np.concatenate((p,p_w_50), axis=0)
shp=(x_w_51.shape)[0]
shape_y = y_w_51.shape[0]
shape_p =  p_w_51.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_w_51 =  np.delete(x_w_51, [k for k in range(nece_shp,shp)], None)
x_w_51 = x_w_51.reshape(-1,sensors)
shape_x_w_51 = x_w_51.shape[0]
x_w_51 = preprocessing.normalize(x_w_51, axis=0)
x_w_51= np.reshape(x_w_51, (-1,segement_time_size, sensors))
y_w_51= np.delete( y_w_51, [k for k in range(x_w_51.shape[0],shape_y)], None)
p_w_51 = np.delete(p_w_51,[k for k in range(x_w_51.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_w_51), axis=0)
y= np.concatenate((y,y_w_51), axis=0)
p= np.concatenate((p,p_w_51), axis=0)
shp=(x_w_52.shape)[0]
shape_y = y_w_52.shape[0]
shape_p =  p_w_52.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_w_52 =  np.delete(x_w_52, [k for k in range(nece_shp,shp)], None)
x_w_52 = x_w_52.reshape(-1,sensors)
shape_x_w_52 = x_w_52.shape[0]
x_w_52 = preprocessing.normalize(x_w_52, axis=0)
x_w_52= np.reshape(x_w_52, (-1,segement_time_size, sensors))
y_w_52= np.delete( y_w_52, [k for k in range(x_w_52.shape[0],shape_y)], None)
p_w_52 = np.delete(p_w_52,[k for k in range(x_w_52.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_w_52), axis=0)
y= np.concatenate((y,y_w_52), axis=0)
p= np.concatenate((p,p_w_52), axis=0)
shp=(x_w_53.shape)[0]
shape_y = y_w_53.shape[0]
shape_p =  p_w_53.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_w_53 =  np.delete(x_w_53, [k for k in range(nece_shp,shp)], None)
x_w_53 = x_w_53.reshape(-1,sensors)
shape_x_w_53 = x_w_53.shape[0]
x_w_53 = preprocessing.normalize(x_w_53, axis=0)
x_w_53= np.reshape(x_w_53, (-1,segement_time_size, sensors))
y_w_53= np.delete( y_w_53, [k for k in range(x_w_53.shape[0],shape_y)], None)
p_w_53 = np.delete(p_w_53,[k for k in range(x_w_53.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_w_53), axis=0)
y= np.concatenate((y,y_w_53), axis=0)
p= np.concatenate((p,p_w_53), axis=0)
shp=(x_w_54.shape)[0]
shape_y = y_w_54.shape[0]
shape_p =  p_w_54.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_w_54 =  np.delete(x_w_54, [k for k in range(nece_shp,shp)], None)
x_w_54 = x_w_54.reshape(-1,sensors)
shape_x_w_54 = x_w_54.shape[0]
x_w_54 = preprocessing.normalize(x_w_54, axis=0)
x_w_54= np.reshape(x_w_54, (-1,segement_time_size, sensors))
y_w_54= np.delete( y_w_54, [k for k in range(x_w_54.shape[0],shape_y)], None)
p_w_54 = np.delete(p_w_54,[k for k in range(x_w_54.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_w_54), axis=0)
y= np.concatenate((y,y_w_54), axis=0)
p= np.concatenate((p,p_w_54), axis=0)
shp=(x_w_55.shape)[0]
shape_y = y_w_55.shape[0]
shape_p =  p_w_55.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_w_55 =  np.delete(x_w_55, [k for k in range(nece_shp,shp)], None)
x_w_55 = x_w_55.reshape(-1,sensors)
shape_x_w_55 = x_w_55.shape[0]
x_w_55 = preprocessing.normalize(x_w_55, axis=0)
x_w_55= np.reshape(x_w_55, (-1,segement_time_size, sensors))
y_w_55= np.delete( y_w_55, [k for k in range(x_w_55.shape[0],shape_y)], None)
p_w_55 = np.delete(p_w_55,[k for k in range(x_w_55.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_w_55), axis=0)
y= np.concatenate((y,y_w_55), axis=0)
p= np.concatenate((p,p_w_55), axis=0)
shp=(x_w_56.shape)[0]
shape_y = y_w_56.shape[0]
shape_p =  p_w_56.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_w_56 =  np.delete(x_w_56, [k for k in range(nece_shp,shp)], None)
x_w_56 = x_w_56.reshape(-1,sensors)
shape_x_w_56 = x_w_56.shape[0]
x_w_56 = preprocessing.normalize(x_w_56, axis=0)
x_w_56= np.reshape(x_w_56, (-1,segement_time_size, sensors))
y_w_56= np.delete( y_w_56, [k for k in range(x_w_56.shape[0],shape_y)], None)
p_w_56 = np.delete(p_w_56,[k for k in range(x_w_56.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_w_56), axis=0)
y= np.concatenate((y,y_w_56), axis=0)
p= np.concatenate((p,p_w_56), axis=0)
shp=(x_w_57.shape)[0]
shape_y = y_w_57.shape[0]
shape_p =  p_w_57.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_w_57 =  np.delete(x_w_57, [k for k in range(nece_shp,shp)], None)
x_w_57 = x_w_57.reshape(-1,sensors)
shape_x_w_57 = x_w_57.shape[0]
x_w_57 = preprocessing.normalize(x_w_57, axis=0)
x_w_57= np.reshape(x_w_57, (-1,segement_time_size, sensors))
y_w_57= np.delete( y_w_57, [k for k in range(x_w_57.shape[0],shape_y)], None)
p_w_57 = np.delete(p_w_57,[k for k in range(x_w_57.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_w_57), axis=0)
y= np.concatenate((y,y_w_57), axis=0)
p= np.concatenate((p,p_w_57), axis=0)
shp=(x_w_58.shape)[0]
shape_y = y_w_58.shape[0]
shape_p =  p_w_58.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_w_58 =  np.delete(x_w_58, [k for k in range(nece_shp,shp)], None)
x_w_58 = x_w_58.reshape(-1,sensors)
shape_x_w_58 = x_w_58.shape[0]
x_w_58 = preprocessing.normalize(x_w_58, axis=0)
x_w_58= np.reshape(x_w_58, (-1,segement_time_size, sensors))
y_w_58= np.delete( y_w_58, [k for k in range(x_w_58.shape[0],shape_y)], None)
p_w_58 = np.delete(p_w_58,[k for k in range(x_w_58.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_w_58), axis=0)
y= np.concatenate((y,y_w_58), axis=0)
p= np.concatenate((p,p_w_58), axis=0)
shp=(x_w_59.shape)[0]
shape_y = y_w_59.shape[0]
shape_p =  p_w_59.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_w_59 =  np.delete(x_w_59, [k for k in range(nece_shp,shp)], None)
x_w_59 = x_w_59.reshape(-1,sensors)
shape_x_w_59 = x_w_59.shape[0]
x_w_59 = preprocessing.normalize(x_w_59, axis=0)
x_w_59= np.reshape(x_w_59, (-1,segement_time_size, sensors))
y_w_59= np.delete( y_w_59, [k for k in range(x_w_59.shape[0],shape_y)], None)
p_w_59 = np.delete(p_w_59,[k for k in range(x_w_59.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_w_59), axis=0)
y= np.concatenate((y,y_w_59), axis=0)
p= np.concatenate((p,p_w_59), axis=0)
shp=(x_w_60.shape)[0]
shape_y = y_w_60.shape[0]
shape_p =  p_w_60.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_w_60 =  np.delete(x_w_60, [k for k in range(nece_shp,shp)], None)
x_w_60 = x_w_60.reshape(-1,sensors)
shape_x_w_60 = x_w_60.shape[0]
x_w_60 = preprocessing.normalize(x_w_60, axis=0)
x_w_60= np.reshape(x_w_60, (-1,segement_time_size, sensors))
y_w_60= np.delete( y_w_60, [k for k in range(x_w_60.shape[0],shape_y)], None)
p_w_60 = np.delete(p_w_60,[k for k in range(x_w_60.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_w_60), axis=0)
y= np.concatenate((y,y_w_60), axis=0)
p= np.concatenate((p,p_w_60), axis=0)
shp=(x_w_61.shape)[0]
shape_y = y_w_61.shape[0]
shape_p =  p_w_61.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_w_61 =  np.delete(x_w_61, [k for k in range(nece_shp,shp)], None)
x_w_61 = x_w_61.reshape(-1,sensors)
shape_x_w_61 = x_w_61.shape[0]
x_w_61 = preprocessing.normalize(x_w_61, axis=0)
x_w_61= np.reshape(x_w_61, (-1,segement_time_size, sensors))
y_w_61= np.delete( y_w_61, [k for k in range(x_w_61.shape[0],shape_y)], None)
p_w_61 = np.delete(p_w_61,[k for k in range(x_w_61.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_w_61), axis=0)
y= np.concatenate((y,y_w_61), axis=0)
p= np.concatenate((p,p_w_61), axis=0)
shp=(x_w_62.shape)[0]
shape_y = y_w_62.shape[0]
shape_p =  p_w_62.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_w_62 =  np.delete(x_w_62, [k for k in range(nece_shp,shp)], None)
x_w_62 = x_w_62.reshape(-1,sensors)
shape_x_w_62 = x_w_62.shape[0]
x_w_62 = preprocessing.normalize(x_w_62, axis=0)
x_w_62= np.reshape(x_w_62, (-1,segement_time_size, sensors))
y_w_62= np.delete( y_w_62, [k for k in range(x_w_62.shape[0],shape_y)], None)
p_w_62 = np.delete(p_w_62,[k for k in range(x_w_62.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_w_62), axis=0)
y= np.concatenate((y,y_w_62), axis=0)
p= np.concatenate((p,p_w_62), axis=0)
shp=(x_w_63.shape)[0]
shape_y = y_w_63.shape[0]
shape_p =  p_w_63.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_w_63 =  np.delete(x_w_63, [k for k in range(nece_shp,shp)], None)
x_w_63 = x_w_63.reshape(-1,sensors)
shape_x_w_63 = x_w_63.shape[0]
x_w_63 = preprocessing.normalize(x_w_63, axis=0)
x_w_63= np.reshape(x_w_63, (-1,segement_time_size, sensors))
y_w_63= np.delete( y_w_63, [k for k in range(x_w_63.shape[0],shape_y)], None)
p_w_63 = np.delete(p_w_63,[k for k in range(x_w_63.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_w_63), axis=0)
y= np.concatenate((y,y_w_63), axis=0)
p= np.concatenate((p,p_w_63), axis=0)
shp=(x_w_64.shape)[0]
shape_y = y_w_64.shape[0]
shape_p =  p_w_64.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_w_64 =  np.delete(x_w_64, [k for k in range(nece_shp,shp)], None)
x_w_64 = x_w_64.reshape(-1,sensors)
shape_x_w_64 = x_w_64.shape[0]
x_w_64 = preprocessing.normalize(x_w_64, axis=0)
x_w_64= np.reshape(x_w_64, (-1,segement_time_size, sensors))
y_w_64= np.delete( y_w_64, [k for k in range(x_w_64.shape[0],shape_y)], None)
p_w_64 = np.delete(p_w_64,[k for k in range(x_w_64.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_w_64), axis=0)
y= np.concatenate((y,y_w_64), axis=0)
p= np.concatenate((p,p_w_64), axis=0)
shp=(x_w_65.shape)[0]
shape_y = y_w_65.shape[0]
shape_p =  p_w_65.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_w_65 =  np.delete(x_w_65, [k for k in range(nece_shp,shp)], None)
x_w_65 = x_w_65.reshape(-1,sensors)
shape_x_w_65 = x_w_65.shape[0]
x_w_65 = preprocessing.normalize(x_w_65, axis=0)
x_w_65= np.reshape(x_w_65, (-1,segement_time_size, sensors))
y_w_65= np.delete( y_w_65, [k for k in range(x_w_65.shape[0],shape_y)], None)
p_w_65 = np.delete(p_w_65,[k for k in range(x_w_65.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_w_65), axis=0)
y= np.concatenate((y,y_w_65), axis=0)
p= np.concatenate((p,p_w_65), axis=0)
shp=(x_w_66.shape)[0]
shape_y = y_w_66.shape[0]
shape_p =  p_w_66.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_w_66 =  np.delete(x_w_66, [k for k in range(nece_shp,shp)], None)
x_w_66 = x_w_66.reshape(-1,sensors)
shape_x_w_66 = x_w_66.shape[0]
x_w_66 = preprocessing.normalize(x_w_66, axis=0)
x_w_66= np.reshape(x_w_66, (-1,segement_time_size, sensors))
y_w_66= np.delete( y_w_66, [k for k in range(x_w_66.shape[0],shape_y)], None)
p_w_66 = np.delete(p_w_66,[k for k in range(x_w_66.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_w_66), axis=0)
y= np.concatenate((y,y_w_66), axis=0)
p= np.concatenate((p,p_w_66), axis=0)
shp=(x_w_67.shape)[0]
shape_y = y_w_67.shape[0]
shape_p =  p_w_67.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_w_67 =  np.delete(x_w_67, [k for k in range(nece_shp,shp)], None)
x_w_67 = x_w_67.reshape(-1,sensors)
shape_x_w_67 = x_w_67.shape[0]
x_w_67 = preprocessing.normalize(x_w_67, axis=0)
x_w_67= np.reshape(x_w_67, (-1,segement_time_size, sensors))
y_w_67= np.delete( y_w_67, [k for k in range(x_w_67.shape[0],shape_y)], None)
p_w_67 = np.delete(p_w_67,[k for k in range(x_w_67.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_w_67), axis=0)
y= np.concatenate((y,y_w_67), axis=0)
p= np.concatenate((p,p_w_67), axis=0)
shp=(x_w_68.shape)[0]
shape_y = y_w_68.shape[0]
shape_p =  p_w_68.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_w_68 =  np.delete(x_w_68, [k for k in range(nece_shp,shp)], None)
x_w_68 = x_w_68.reshape(-1,sensors)
shape_x_w_68 = x_w_68.shape[0]
x_w_68 = preprocessing.normalize(x_w_68, axis=0)
x_w_68= np.reshape(x_w_68, (-1,segement_time_size, sensors))
y_w_68= np.delete( y_w_68, [k for k in range(x_w_68.shape[0],shape_y)], None)
p_w_68 = np.delete(p_w_68,[k for k in range(x_w_68.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_w_68), axis=0)
y= np.concatenate((y,y_w_68), axis=0)
p= np.concatenate((p,p_w_68), axis=0)
shp=(x_w_69.shape)[0]
shape_y = y_w_69.shape[0]
shape_p =  p_w_69.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_w_69 =  np.delete(x_w_69, [k for k in range(nece_shp,shp)], None)
x_w_69 = x_w_69.reshape(-1,sensors)
shape_x_w_69 = x_w_69.shape[0]
x_w_69 = preprocessing.normalize(x_w_69, axis=0)
x_w_69= np.reshape(x_w_69, (-1,segement_time_size, sensors))
y_w_69= np.delete( y_w_69, [k for k in range(x_w_69.shape[0],shape_y)], None)
p_w_69 = np.delete(p_w_69,[k for k in range(x_w_69.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_w_69), axis=0)
y= np.concatenate((y,y_w_69), axis=0)
p= np.concatenate((p,p_w_69), axis=0)
shp=(x_w_70.shape)[0]
shape_y = y_w_70.shape[0]
shape_p =  p_w_70.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_w_70 =  np.delete(x_w_70, [k for k in range(nece_shp,shp)], None)
x_w_70 = x_w_70.reshape(-1,sensors)
shape_x_w_70 = x_w_70.shape[0]
x_w_70 = preprocessing.normalize(x_w_70, axis=0)
x_w_70= np.reshape(x_w_70, (-1,segement_time_size, sensors))
y_w_70= np.delete( y_w_70, [k for k in range(x_w_70.shape[0],shape_y)], None)
p_w_70 = np.delete(p_w_70,[k for k in range(x_w_70.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_w_70), axis=0)
y= np.concatenate((y,y_w_70), axis=0)
p= np.concatenate((p,p_w_70), axis=0)
shp=(x_w_71.shape)[0]
shape_y = y_w_71.shape[0]
shape_p =  p_w_71.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_w_71 =  np.delete(x_w_71, [k for k in range(nece_shp,shp)], None)
x_w_71 = x_w_71.reshape(-1,sensors)
shape_x_w_71 = x_w_71.shape[0]
x_w_71 = preprocessing.normalize(x_w_71, axis=0)
x_w_71= np.reshape(x_w_71, (-1,segement_time_size, sensors))
y_w_71= np.delete( y_w_71, [k for k in range(x_w_71.shape[0],shape_y)], None)
p_w_71 = np.delete(p_w_71,[k for k in range(x_w_71.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_w_71), axis=0)
y= np.concatenate((y,y_w_71), axis=0)
p= np.concatenate((p,p_w_71), axis=0)
shp=(x_w_72.shape)[0]
shape_y = y_w_72.shape[0]
shape_p =  p_w_72.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_w_72 =  np.delete(x_w_72, [k for k in range(nece_shp,shp)], None)
x_w_72 = x_w_72.reshape(-1,sensors)
shape_x_w_72 = x_w_72.shape[0]
x_w_72 = preprocessing.normalize(x_w_72, axis=0)
x_w_72= np.reshape(x_w_72, (-1,segement_time_size, sensors))
y_w_72= np.delete( y_w_72, [k for k in range(x_w_72.shape[0],shape_y)], None)
p_w_72 = np.delete(p_w_72,[k for k in range(x_w_72.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_w_72), axis=0)
y= np.concatenate((y,y_w_72), axis=0)
p= np.concatenate((p,p_w_72), axis=0)
shp=(x_w_73.shape)[0]
shape_y = y_w_73.shape[0]
shape_p =  p_w_73.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_w_73 =  np.delete(x_w_73, [k for k in range(nece_shp,shp)], None)
x_w_73 = x_w_73.reshape(-1,sensors)
shape_x_w_73 = x_w_73.shape[0]
x_w_73 = preprocessing.normalize(x_w_73, axis=0)
x_w_73= np.reshape(x_w_73, (-1,segement_time_size, sensors))
y_w_73= np.delete( y_w_73, [k for k in range(x_w_73.shape[0],shape_y)], None)
p_w_73 = np.delete(p_w_73,[k for k in range(x_w_73.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_w_73), axis=0)
y= np.concatenate((y,y_w_73), axis=0)
p= np.concatenate((p,p_w_73), axis=0)
shp=(x_w_74.shape)[0]
shape_y = y_w_74.shape[0]
shape_p =  p_w_74.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_w_74 =  np.delete(x_w_74, [k for k in range(nece_shp,shp)], None)
x_w_74 = x_w_74.reshape(-1,sensors)
shape_x_w_74 = x_w_74.shape[0]
x_w_74 = preprocessing.normalize(x_w_74, axis=0)
x_w_74= np.reshape(x_w_74, (-1,segement_time_size, sensors))
y_w_74= np.delete( y_w_74, [k for k in range(x_w_74.shape[0],shape_y)], None)
p_w_74 = np.delete(p_w_74,[k for k in range(x_w_74.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_w_74), axis=0)
y= np.concatenate((y,y_w_74), axis=0)
p= np.concatenate((p,p_w_74), axis=0)
shp=(x_w_75.shape)[0]
shape_y = y_w_75.shape[0]
shape_p =  p_w_75.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_w_75 =  np.delete(x_w_75, [k for k in range(nece_shp,shp)], None)
x_w_75 = x_w_75.reshape(-1,sensors)
shape_x_w_75 = x_w_75.shape[0]
x_w_75 = preprocessing.normalize(x_w_75, axis=0)
x_w_75= np.reshape(x_w_75, (-1,segement_time_size, sensors))
y_w_75= np.delete( y_w_75, [k for k in range(x_w_75.shape[0],shape_y)], None)
p_w_75 = np.delete(p_w_75,[k for k in range(x_w_75.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_w_75), axis=0)
y= np.concatenate((y,y_w_75), axis=0)
p= np.concatenate((p,p_w_75), axis=0)
shp=(x_w_76.shape)[0]
shape_y = y_w_76.shape[0]
shape_p =  p_w_76.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_w_76 =  np.delete(x_w_76, [k for k in range(nece_shp,shp)], None)
x_w_76 = x_w_76.reshape(-1,sensors)
shape_x_w_76 = x_w_76.shape[0]
x_w_76 = preprocessing.normalize(x_w_76, axis=0)
x_w_76= np.reshape(x_w_76, (-1,segement_time_size, sensors))
y_w_76= np.delete( y_w_76, [k for k in range(x_w_76.shape[0],shape_y)], None)
p_w_76 = np.delete(p_w_76,[k for k in range(x_w_76.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_w_76), axis=0)
y= np.concatenate((y,y_w_76), axis=0)
p= np.concatenate((p,p_w_76), axis=0)
shp=(x_w_77.shape)[0]
shape_y = y_w_77.shape[0]
shape_p =  p_w_77.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_w_77 =  np.delete(x_w_77, [k for k in range(nece_shp,shp)], None)
x_w_77 = x_w_77.reshape(-1,sensors)
shape_x_w_77 = x_w_77.shape[0]
x_w_77 = preprocessing.normalize(x_w_77, axis=0)
x_w_77= np.reshape(x_w_77, (-1,segement_time_size, sensors))
y_w_77= np.delete( y_w_77, [k for k in range(x_w_77.shape[0],shape_y)], None)
p_w_77 = np.delete(p_w_77,[k for k in range(x_w_77.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_w_77), axis=0)
y= np.concatenate((y,y_w_77), axis=0)
p= np.concatenate((p,p_w_77), axis=0)
shp=(x_w_78.shape)[0]
shape_y = y_w_78.shape[0]
shape_p =  p_w_78.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_w_78 =  np.delete(x_w_78, [k for k in range(nece_shp,shp)], None)
x_w_78 = x_w_78.reshape(-1,sensors)
shape_x_w_78 = x_w_78.shape[0]
x_w_78 = preprocessing.normalize(x_w_78, axis=0)
x_w_78= np.reshape(x_w_78, (-1,segement_time_size, sensors))
y_w_78= np.delete( y_w_78, [k for k in range(x_w_78.shape[0],shape_y)], None)
p_w_78 = np.delete(p_w_78,[k for k in range(x_w_78.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_w_78), axis=0)
y= np.concatenate((y,y_w_78), axis=0)
p= np.concatenate((p,p_w_78), axis=0)
shp=(x_w_79.shape)[0]
shape_y = y_w_79.shape[0]
shape_p =  p_w_79.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_w_79 =  np.delete(x_w_79, [k for k in range(nece_shp,shp)], None)
x_w_79 = x_w_79.reshape(-1,sensors)
shape_x_w_79 = x_w_79.shape[0]
x_w_79 = preprocessing.normalize(x_w_79, axis=0)
x_w_79= np.reshape(x_w_79, (-1,segement_time_size, sensors))
y_w_79= np.delete( y_w_79, [k for k in range(x_w_79.shape[0],shape_y)], None)
p_w_79 = np.delete(p_w_79,[k for k in range(x_w_79.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_w_79), axis=0)
y= np.concatenate((y,y_w_79), axis=0)
p= np.concatenate((p,p_w_79), axis=0)
shp=(x_w_80.shape)[0]
shape_y = y_w_80.shape[0]
shape_p =  p_w_80.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_w_80 =  np.delete(x_w_80, [k for k in range(nece_shp,shp)], None)
x_w_80 = x_w_80.reshape(-1,sensors)
shape_x_w_80 = x_w_80.shape[0]
x_w_80 = preprocessing.normalize(x_w_80, axis=0)
x_w_80= np.reshape(x_w_80, (-1,segement_time_size, sensors))
y_w_80= np.delete( y_w_80, [k for k in range(x_w_80.shape[0],shape_y)], None)
p_w_80 = np.delete(p_w_80,[k for k in range(x_w_80.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_w_80), axis=0)
y= np.concatenate((y,y_w_80), axis=0)
p= np.concatenate((p,p_w_80), axis=0)
shp=(x_w_81.shape)[0]
shape_y = y_w_81.shape[0]
shape_p =  p_w_81.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_w_81 =  np.delete(x_w_81, [k for k in range(nece_shp,shp)], None)
x_w_81 = x_w_81.reshape(-1,sensors)
shape_x_w_81 = x_w_81.shape[0]
x_w_81 = preprocessing.normalize(x_w_81, axis=0)
x_w_81= np.reshape(x_w_81, (-1,segement_time_size, sensors))
y_w_81= np.delete( y_w_81, [k for k in range(x_w_81.shape[0],shape_y)], None)
p_w_81 = np.delete(p_w_81,[k for k in range(x_w_81.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_w_81), axis=0)
y= np.concatenate((y,y_w_81), axis=0)
p= np.concatenate((p,p_w_81), axis=0)
shp=(x_w_82.shape)[0]
shape_y = y_w_82.shape[0]
shape_p =  p_w_82.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_w_82 =  np.delete(x_w_82, [k for k in range(nece_shp,shp)], None)
x_w_82 = x_w_82.reshape(-1,sensors)
shape_x_w_82 = x_w_82.shape[0]
x_w_82 = preprocessing.normalize(x_w_82, axis=0)
x_w_82= np.reshape(x_w_82, (-1,segement_time_size, sensors))
y_w_82= np.delete( y_w_82, [k for k in range(x_w_82.shape[0],shape_y)], None)
p_w_82 = np.delete(p_w_82,[k for k in range(x_w_82.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_w_82), axis=0)
y= np.concatenate((y,y_w_82), axis=0)
p= np.concatenate((p,p_w_82), axis=0)
shp=(x_w_83.shape)[0]
shape_y = y_w_83.shape[0]
shape_p =  p_w_83.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_w_83 =  np.delete(x_w_83, [k for k in range(nece_shp,shp)], None)
x_w_83 = x_w_83.reshape(-1,sensors)
shape_x_w_83 = x_w_83.shape[0]
x_w_83 = preprocessing.normalize(x_w_83, axis=0)
x_w_83= np.reshape(x_w_83, (-1,segement_time_size, sensors))
y_w_83= np.delete( y_w_83, [k for k in range(x_w_83.shape[0],shape_y)], None)
p_w_83 = np.delete(p_w_83,[k for k in range(x_w_83.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_w_83), axis=0)
y= np.concatenate((y,y_w_83), axis=0)
p= np.concatenate((p,p_w_83), axis=0)
shp=(x_w_84.shape)[0]
shape_y = y_w_84.shape[0]
shape_p =  p_w_84.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_w_84 =  np.delete(x_w_84, [k for k in range(nece_shp,shp)], None)
x_w_84 = x_w_84.reshape(-1,sensors)
shape_x_w_84 = x_w_84.shape[0]
x_w_84 = preprocessing.normalize(x_w_84, axis=0)
x_w_84= np.reshape(x_w_84, (-1,segement_time_size, sensors))
y_w_84= np.delete( y_w_84, [k for k in range(x_w_84.shape[0],shape_y)], None)
p_w_84 = np.delete(p_w_84,[k for k in range(x_w_84.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_w_84), axis=0)
y= np.concatenate((y,y_w_84), axis=0)
p= np.concatenate((p,p_w_84), axis=0)
shp=(x_w_85.shape)[0]
shape_y = y_w_85.shape[0]
shape_p =  p_w_85.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_w_85 =  np.delete(x_w_85, [k for k in range(nece_shp,shp)], None)
x_w_85 = x_w_85.reshape(-1,sensors)
shape_x_w_85 = x_w_85.shape[0]
x_w_85 = preprocessing.normalize(x_w_85, axis=0)
x_w_85= np.reshape(x_w_85, (-1,segement_time_size, sensors))
y_w_85= np.delete( y_w_85, [k for k in range(x_w_85.shape[0],shape_y)], None)
p_w_85 = np.delete(p_w_85,[k for k in range(x_w_85.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_w_85), axis=0)
y= np.concatenate((y,y_w_85), axis=0)
p= np.concatenate((p,p_w_85), axis=0)
shp=(x_w_86.shape)[0]
shape_y = y_w_86.shape[0]
shape_p =  p_w_86.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_w_86 =  np.delete(x_w_86, [k for k in range(nece_shp,shp)], None)
x_w_86 = x_w_86.reshape(-1,sensors)
shape_x_w_86 = x_w_86.shape[0]
x_w_86 = preprocessing.normalize(x_w_86, axis=0)
x_w_86= np.reshape(x_w_86, (-1,segement_time_size, sensors))
y_w_86= np.delete( y_w_86, [k for k in range(x_w_86.shape[0],shape_y)], None)
p_w_86 = np.delete(p_w_86,[k for k in range(x_w_86.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_w_86), axis=0)
y= np.concatenate((y,y_w_86), axis=0)
p= np.concatenate((p,p_w_86), axis=0)
shp=(x_w_87.shape)[0]
shape_y = y_w_87.shape[0]
shape_p =  p_w_87.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_w_87 =  np.delete(x_w_87, [k for k in range(nece_shp,shp)], None)
x_w_87 = x_w_87.reshape(-1,sensors)
shape_x_w_87 = x_w_87.shape[0]
x_w_87 = preprocessing.normalize(x_w_87, axis=0)
x_w_87= np.reshape(x_w_87, (-1,segement_time_size, sensors))
y_w_87= np.delete( y_w_87, [k for k in range(x_w_87.shape[0],shape_y)], None)
p_w_87 = np.delete(p_w_87,[k for k in range(x_w_87.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_w_87), axis=0)
y= np.concatenate((y,y_w_87), axis=0)
p= np.concatenate((p,p_w_87), axis=0)
shp=(x_w_88.shape)[0]
shape_y = y_w_88.shape[0]
shape_p =  p_w_88.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_w_88 =  np.delete(x_w_88, [k for k in range(nece_shp,shp)], None)
x_w_88 = x_w_88.reshape(-1,sensors)
shape_x_w_88 = x_w_88.shape[0]
x_w_88 = preprocessing.normalize(x_w_88, axis=0)
x_w_88= np.reshape(x_w_88, (-1,segement_time_size, sensors))
y_w_88= np.delete( y_w_88, [k for k in range(x_w_88.shape[0],shape_y)], None)
p_w_88 = np.delete(p_w_88,[k for k in range(x_w_88.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_w_88), axis=0)
y= np.concatenate((y,y_w_88), axis=0)
p= np.concatenate((p,p_w_88), axis=0)
shp=(x_w_89.shape)[0]
shape_y = y_w_89.shape[0]
shape_p =  p_w_89.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_w_89 =  np.delete(x_w_89, [k for k in range(nece_shp,shp)], None)
x_w_89 = x_w_89.reshape(-1,sensors)
shape_x_w_89 = x_w_89.shape[0]
x_w_89 = preprocessing.normalize(x_w_89, axis=0)
x_w_89= np.reshape(x_w_89, (-1,segement_time_size, sensors))
y_w_89= np.delete( y_w_89, [k for k in range(x_w_89.shape[0],shape_y)], None)
p_w_89 = np.delete(p_w_89,[k for k in range(x_w_89.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_w_89), axis=0)
y= np.concatenate((y,y_w_89), axis=0)
p= np.concatenate((p,p_w_89), axis=0)
shp=(x_w_90.shape)[0]
shape_y = y_w_90.shape[0]
shape_p =  p_w_90.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_w_90 =  np.delete(x_w_90, [k for k in range(nece_shp,shp)], None)
x_w_90 = x_w_90.reshape(-1,sensors)
shape_x_w_90 = x_w_90.shape[0]
x_w_90 = preprocessing.normalize(x_w_90, axis=0)
x_w_90= np.reshape(x_w_90, (-1,segement_time_size, sensors))
y_w_90= np.delete( y_w_90, [k for k in range(x_w_90.shape[0],shape_y)], None)
p_w_90 = np.delete(p_w_90,[k for k in range(x_w_90.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_w_90), axis=0)
y= np.concatenate((y,y_w_90), axis=0)
p= np.concatenate((p,p_w_90), axis=0)
shp=(x_w_91.shape)[0]
shape_y = y_w_91.shape[0]
shape_p =  p_w_91.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_w_91 =  np.delete(x_w_91, [k for k in range(nece_shp,shp)], None)
x_w_91 = x_w_91.reshape(-1,sensors)
shape_x_w_91 = x_w_91.shape[0]
x_w_91 = preprocessing.normalize(x_w_91, axis=0)
x_w_91= np.reshape(x_w_91, (-1,segement_time_size, sensors))
y_w_91= np.delete( y_w_91, [k for k in range(x_w_91.shape[0],shape_y)], None)
p_w_91 = np.delete(p_w_91,[k for k in range(x_w_91.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_w_91), axis=0)
y= np.concatenate((y,y_w_91), axis=0)
p= np.concatenate((p,p_w_91), axis=0)
shp=(x_w_92.shape)[0]
shape_y = y_w_92.shape[0]
shape_p =  p_w_92.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_w_92 =  np.delete(x_w_92, [k for k in range(nece_shp,shp)], None)
x_w_92 = x_w_92.reshape(-1,sensors)
shape_x_w_92 = x_w_92.shape[0]
x_w_92 = preprocessing.normalize(x_w_92, axis=0)
x_w_92= np.reshape(x_w_92, (-1,segement_time_size, sensors))
y_w_92= np.delete( y_w_92, [k for k in range(x_w_92.shape[0],shape_y)], None)
p_w_92 = np.delete(p_w_92,[k for k in range(x_w_92.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_w_92), axis=0)
y= np.concatenate((y,y_w_92), axis=0)
p= np.concatenate((p,p_w_92), axis=0)
shp=(x_w_93.shape)[0]
shape_y = y_w_93.shape[0]
shape_p =  p_w_93.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_w_93 =  np.delete(x_w_93, [k for k in range(nece_shp,shp)], None)
x_w_93 = x_w_93.reshape(-1,sensors)
shape_x_w_93 = x_w_93.shape[0]
x_w_93 = preprocessing.normalize(x_w_93, axis=0)
x_w_93= np.reshape(x_w_93, (-1,segement_time_size, sensors))
y_w_93= np.delete( y_w_93, [k for k in range(x_w_93.shape[0],shape_y)], None)
p_w_93 = np.delete(p_w_93,[k for k in range(x_w_93.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_w_93), axis=0)
y= np.concatenate((y,y_w_93), axis=0)
p= np.concatenate((p,p_w_93), axis=0)
shp=(x_w_94.shape)[0]
shape_y = y_w_94.shape[0]
shape_p =  p_w_94.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_w_94 =  np.delete(x_w_94, [k for k in range(nece_shp,shp)], None)
x_w_94 = x_w_94.reshape(-1,sensors)
shape_x_w_94 = x_w_94.shape[0]
x_w_94 = preprocessing.normalize(x_w_94, axis=0)
x_w_94= np.reshape(x_w_94, (-1,segement_time_size, sensors))
y_w_94= np.delete( y_w_94, [k for k in range(x_w_94.shape[0],shape_y)], None)
p_w_94 = np.delete(p_w_94,[k for k in range(x_w_94.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_w_94), axis=0)
y= np.concatenate((y,y_w_94), axis=0)
p= np.concatenate((p,p_w_94), axis=0)
shp=(x_w_95.shape)[0]
shape_y = y_w_95.shape[0]
shape_p =  p_w_95.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_w_95 =  np.delete(x_w_95, [k for k in range(nece_shp,shp)], None)
x_w_95 = x_w_95.reshape(-1,sensors)
shape_x_w_95 = x_w_95.shape[0]
x_w_95 = preprocessing.normalize(x_w_95, axis=0)
x_w_95= np.reshape(x_w_95, (-1,segement_time_size, sensors))
y_w_95= np.delete( y_w_95, [k for k in range(x_w_95.shape[0],shape_y)], None)
p_w_95 = np.delete(p_w_95,[k for k in range(x_w_95.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_w_95), axis=0)
y= np.concatenate((y,y_w_95), axis=0)
p= np.concatenate((p,p_w_95), axis=0)
shp=(x_w_96.shape)[0]
shape_y = y_w_96.shape[0]
shape_p =  p_w_96.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_w_96 =  np.delete(x_w_96, [k for k in range(nece_shp,shp)], None)
x_w_96 = x_w_96.reshape(-1,sensors)
shape_x_w_96 = x_w_96.shape[0]
x_w_96 = preprocessing.normalize(x_w_96, axis=0)
x_w_96= np.reshape(x_w_96, (-1,segement_time_size, sensors))
y_w_96= np.delete( y_w_96, [k for k in range(x_w_96.shape[0],shape_y)], None)
p_w_96 = np.delete(p_w_96,[k for k in range(x_w_96.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_w_96), axis=0)
y= np.concatenate((y,y_w_96), axis=0)
p= np.concatenate((p,p_w_96), axis=0)
shp=(x_w_97.shape)[0]
shape_y = y_w_97.shape[0]
shape_p =  p_w_97.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_w_97 =  np.delete(x_w_97, [k for k in range(nece_shp,shp)], None)
x_w_97 = x_w_97.reshape(-1,sensors)
shape_x_w_97 = x_w_97.shape[0]
x_w_97 = preprocessing.normalize(x_w_97, axis=0)
x_w_97= np.reshape(x_w_97, (-1,segement_time_size, sensors))
y_w_97= np.delete( y_w_97, [k for k in range(x_w_97.shape[0],shape_y)], None)
p_w_97 = np.delete(p_w_97,[k for k in range(x_w_97.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_w_97), axis=0)
y= np.concatenate((y,y_w_97), axis=0)
p= np.concatenate((p,p_w_97), axis=0)
shp=(x_w_98.shape)[0]
shape_y = y_w_98.shape[0]
shape_p =  p_w_98.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_w_98 =  np.delete(x_w_98, [k for k in range(nece_shp,shp)], None)
x_w_98 = x_w_98.reshape(-1,sensors)
shape_x_w_98 = x_w_98.shape[0]
x_w_98 = preprocessing.normalize(x_w_98, axis=0)
x_w_98= np.reshape(x_w_98, (-1,segement_time_size, sensors))
y_w_98= np.delete( y_w_98, [k for k in range(x_w_98.shape[0],shape_y)], None)
p_w_98 = np.delete(p_w_98,[k for k in range(x_w_98.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_w_98), axis=0)
y= np.concatenate((y,y_w_98), axis=0)
p= np.concatenate((p,p_w_98), axis=0)
shp=(x_w_99.shape)[0]
shape_y = y_w_99.shape[0]
shape_p =  p_w_99.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_w_99 =  np.delete(x_w_99, [k for k in range(nece_shp,shp)], None)
x_w_99 = x_w_99.reshape(-1,sensors)
shape_x_w_99 = x_w_99.shape[0]
x_w_99 = preprocessing.normalize(x_w_99, axis=0)
x_w_99= np.reshape(x_w_99, (-1,segement_time_size, sensors))
y_w_99= np.delete( y_w_99, [k for k in range(x_w_99.shape[0],shape_y)], None)
p_w_99 = np.delete(p_w_99,[k for k in range(x_w_99.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_w_99), axis=0)
y= np.concatenate((y,y_w_99), axis=0)
p= np.concatenate((p,p_w_99), axis=0)
shp=(x_stu_0.shape)[0]
shape_y = y_stu_0.shape[0]
shape_p =  p_stu_0.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_stu_0 =  np.delete(x_stu_0, [k for k in range(nece_shp,shp)], None)
x_stu_0 = x_stu_0.reshape(-1,sensors)
shape_x_stu_0 = x_stu_0.shape[0]
x_stu_0 = preprocessing.normalize(x_stu_0, axis=0)
x_stu_0= np.reshape(x_stu_0, (-1,segement_time_size, sensors))
y_stu_0= np.delete( y_stu_0, [k for k in range(x_stu_0.shape[0],shape_y)], None)
p_stu_0 = np.delete(p_stu_0,[k for k in range(x_stu_0.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_stu_0), axis=0)
y= np.concatenate((y,y_stu_0), axis=0)
p= np.concatenate((p,p_stu_0), axis=0)
shp=(x_stu_1.shape)[0]
shape_y = y_stu_1.shape[0]
shape_p =  p_stu_1.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_stu_1 =  np.delete(x_stu_1, [k for k in range(nece_shp,shp)], None)
x_stu_1 = x_stu_1.reshape(-1,sensors)
shape_x_stu_1 = x_stu_1.shape[0]
x_stu_1 = preprocessing.normalize(x_stu_1, axis=0)
x_stu_1= np.reshape(x_stu_1, (-1,segement_time_size, sensors))
y_stu_1= np.delete( y_stu_1, [k for k in range(x_stu_1.shape[0],shape_y)], None)
p_stu_1 = np.delete(p_stu_1,[k for k in range(x_stu_1.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_stu_1), axis=0)
y= np.concatenate((y,y_stu_1), axis=0)
p= np.concatenate((p,p_stu_1), axis=0)
shp=(x_stu_2.shape)[0]
shape_y = y_stu_2.shape[0]
shape_p =  p_stu_2.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_stu_2 =  np.delete(x_stu_2, [k for k in range(nece_shp,shp)], None)
x_stu_2 = x_stu_2.reshape(-1,sensors)
shape_x_stu_2 = x_stu_2.shape[0]
x_stu_2 = preprocessing.normalize(x_stu_2, axis=0)
x_stu_2= np.reshape(x_stu_2, (-1,segement_time_size, sensors))
y_stu_2= np.delete( y_stu_2, [k for k in range(x_stu_2.shape[0],shape_y)], None)
p_stu_2 = np.delete(p_stu_2,[k for k in range(x_stu_2.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_stu_2), axis=0)
y= np.concatenate((y,y_stu_2), axis=0)
p= np.concatenate((p,p_stu_2), axis=0)
shp=(x_stu_3.shape)[0]
shape_y = y_stu_3.shape[0]
shape_p =  p_stu_3.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_stu_3 =  np.delete(x_stu_3, [k for k in range(nece_shp,shp)], None)
x_stu_3 = x_stu_3.reshape(-1,sensors)
shape_x_stu_3 = x_stu_3.shape[0]
x_stu_3 = preprocessing.normalize(x_stu_3, axis=0)
x_stu_3= np.reshape(x_stu_3, (-1,segement_time_size, sensors))
y_stu_3= np.delete( y_stu_3, [k for k in range(x_stu_3.shape[0],shape_y)], None)
p_stu_3 = np.delete(p_stu_3,[k for k in range(x_stu_3.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_stu_3), axis=0)
y= np.concatenate((y,y_stu_3), axis=0)
p= np.concatenate((p,p_stu_3), axis=0)
shp=(x_stu_4.shape)[0]
shape_y = y_stu_4.shape[0]
shape_p =  p_stu_4.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_stu_4 =  np.delete(x_stu_4, [k for k in range(nece_shp,shp)], None)
x_stu_4 = x_stu_4.reshape(-1,sensors)
shape_x_stu_4 = x_stu_4.shape[0]
x_stu_4 = preprocessing.normalize(x_stu_4, axis=0)
x_stu_4= np.reshape(x_stu_4, (-1,segement_time_size, sensors))
y_stu_4= np.delete( y_stu_4, [k for k in range(x_stu_4.shape[0],shape_y)], None)
p_stu_4 = np.delete(p_stu_4,[k for k in range(x_stu_4.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_stu_4), axis=0)
y= np.concatenate((y,y_stu_4), axis=0)
p= np.concatenate((p,p_stu_4), axis=0)
shp=(x_stu_5.shape)[0]
shape_y = y_stu_5.shape[0]
shape_p =  p_stu_5.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_stu_5 =  np.delete(x_stu_5, [k for k in range(nece_shp,shp)], None)
x_stu_5 = x_stu_5.reshape(-1,sensors)
shape_x_stu_5 = x_stu_5.shape[0]
x_stu_5 = preprocessing.normalize(x_stu_5, axis=0)
x_stu_5= np.reshape(x_stu_5, (-1,segement_time_size, sensors))
y_stu_5= np.delete( y_stu_5, [k for k in range(x_stu_5.shape[0],shape_y)], None)
p_stu_5 = np.delete(p_stu_5,[k for k in range(x_stu_5.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_stu_5), axis=0)
y= np.concatenate((y,y_stu_5), axis=0)
p= np.concatenate((p,p_stu_5), axis=0)
shp=(x_stu_6.shape)[0]
shape_y = y_stu_6.shape[0]
shape_p =  p_stu_6.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_stu_6 =  np.delete(x_stu_6, [k for k in range(nece_shp,shp)], None)
x_stu_6 = x_stu_6.reshape(-1,sensors)
shape_x_stu_6 = x_stu_6.shape[0]
x_stu_6 = preprocessing.normalize(x_stu_6, axis=0)
x_stu_6= np.reshape(x_stu_6, (-1,segement_time_size, sensors))
y_stu_6= np.delete( y_stu_6, [k for k in range(x_stu_6.shape[0],shape_y)], None)
p_stu_6 = np.delete(p_stu_6,[k for k in range(x_stu_6.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_stu_6), axis=0)
y= np.concatenate((y,y_stu_6), axis=0)
p= np.concatenate((p,p_stu_6), axis=0)
shp=(x_stu_7.shape)[0]
shape_y = y_stu_7.shape[0]
shape_p =  p_stu_7.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_stu_7 =  np.delete(x_stu_7, [k for k in range(nece_shp,shp)], None)
x_stu_7 = x_stu_7.reshape(-1,sensors)
shape_x_stu_7 = x_stu_7.shape[0]
x_stu_7 = preprocessing.normalize(x_stu_7, axis=0)
x_stu_7= np.reshape(x_stu_7, (-1,segement_time_size, sensors))
y_stu_7= np.delete( y_stu_7, [k for k in range(x_stu_7.shape[0],shape_y)], None)
p_stu_7 = np.delete(p_stu_7,[k for k in range(x_stu_7.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_stu_7), axis=0)
y= np.concatenate((y,y_stu_7), axis=0)
p= np.concatenate((p,p_stu_7), axis=0)
shp=(x_stu_8.shape)[0]
shape_y = y_stu_8.shape[0]
shape_p =  p_stu_8.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_stu_8 =  np.delete(x_stu_8, [k for k in range(nece_shp,shp)], None)
x_stu_8 = x_stu_8.reshape(-1,sensors)
shape_x_stu_8 = x_stu_8.shape[0]
x_stu_8 = preprocessing.normalize(x_stu_8, axis=0)
x_stu_8= np.reshape(x_stu_8, (-1,segement_time_size, sensors))
y_stu_8= np.delete( y_stu_8, [k for k in range(x_stu_8.shape[0],shape_y)], None)
p_stu_8 = np.delete(p_stu_8,[k for k in range(x_stu_8.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_stu_8), axis=0)
y= np.concatenate((y,y_stu_8), axis=0)
p= np.concatenate((p,p_stu_8), axis=0)
shp=(x_stu_9.shape)[0]
shape_y = y_stu_9.shape[0]
shape_p =  p_stu_9.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_stu_9 =  np.delete(x_stu_9, [k for k in range(nece_shp,shp)], None)
x_stu_9 = x_stu_9.reshape(-1,sensors)
shape_x_stu_9 = x_stu_9.shape[0]
x_stu_9 = preprocessing.normalize(x_stu_9, axis=0)
x_stu_9= np.reshape(x_stu_9, (-1,segement_time_size, sensors))
y_stu_9= np.delete( y_stu_9, [k for k in range(x_stu_9.shape[0],shape_y)], None)
p_stu_9 = np.delete(p_stu_9,[k for k in range(x_stu_9.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_stu_9), axis=0)
y= np.concatenate((y,y_stu_9), axis=0)
p= np.concatenate((p,p_stu_9), axis=0)
shp=(x_stu_10.shape)[0]
shape_y = y_stu_10.shape[0]
shape_p =  p_stu_10.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_stu_10 =  np.delete(x_stu_10, [k for k in range(nece_shp,shp)], None)
x_stu_10 = x_stu_10.reshape(-1,sensors)
shape_x_stu_10 = x_stu_10.shape[0]
x_stu_10 = preprocessing.normalize(x_stu_10, axis=0)
x_stu_10= np.reshape(x_stu_10, (-1,segement_time_size, sensors))
y_stu_10= np.delete( y_stu_10, [k for k in range(x_stu_10.shape[0],shape_y)], None)
p_stu_10 = np.delete(p_stu_10,[k for k in range(x_stu_10.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_stu_10), axis=0)
y= np.concatenate((y,y_stu_10), axis=0)
p= np.concatenate((p,p_stu_10), axis=0)
shp=(x_stu_11.shape)[0]
shape_y = y_stu_11.shape[0]
shape_p =  p_stu_11.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_stu_11 =  np.delete(x_stu_11, [k for k in range(nece_shp,shp)], None)
x_stu_11 = x_stu_11.reshape(-1,sensors)
shape_x_stu_11 = x_stu_11.shape[0]
x_stu_11 = preprocessing.normalize(x_stu_11, axis=0)
x_stu_11= np.reshape(x_stu_11, (-1,segement_time_size, sensors))
y_stu_11= np.delete( y_stu_11, [k for k in range(x_stu_11.shape[0],shape_y)], None)
p_stu_11 = np.delete(p_stu_11,[k for k in range(x_stu_11.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_stu_11), axis=0)
y= np.concatenate((y,y_stu_11), axis=0)
p= np.concatenate((p,p_stu_11), axis=0)
shp=(x_stu_12.shape)[0]
shape_y = y_stu_12.shape[0]
shape_p =  p_stu_12.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_stu_12 =  np.delete(x_stu_12, [k for k in range(nece_shp,shp)], None)
x_stu_12 = x_stu_12.reshape(-1,sensors)
shape_x_stu_12 = x_stu_12.shape[0]
x_stu_12 = preprocessing.normalize(x_stu_12, axis=0)
x_stu_12= np.reshape(x_stu_12, (-1,segement_time_size, sensors))
y_stu_12= np.delete( y_stu_12, [k for k in range(x_stu_12.shape[0],shape_y)], None)
p_stu_12 = np.delete(p_stu_12,[k for k in range(x_stu_12.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_stu_12), axis=0)
y= np.concatenate((y,y_stu_12), axis=0)
p= np.concatenate((p,p_stu_12), axis=0)
shp=(x_stu_13.shape)[0]
shape_y = y_stu_13.shape[0]
shape_p =  p_stu_13.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_stu_13 =  np.delete(x_stu_13, [k for k in range(nece_shp,shp)], None)
x_stu_13 = x_stu_13.reshape(-1,sensors)
shape_x_stu_13 = x_stu_13.shape[0]
x_stu_13 = preprocessing.normalize(x_stu_13, axis=0)
x_stu_13= np.reshape(x_stu_13, (-1,segement_time_size, sensors))
y_stu_13= np.delete( y_stu_13, [k for k in range(x_stu_13.shape[0],shape_y)], None)
p_stu_13 = np.delete(p_stu_13,[k for k in range(x_stu_13.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_stu_13), axis=0)
y= np.concatenate((y,y_stu_13), axis=0)
p= np.concatenate((p,p_stu_13), axis=0)
shp=(x_stu_14.shape)[0]
shape_y = y_stu_14.shape[0]
shape_p =  p_stu_14.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_stu_14 =  np.delete(x_stu_14, [k for k in range(nece_shp,shp)], None)
x_stu_14 = x_stu_14.reshape(-1,sensors)
shape_x_stu_14 = x_stu_14.shape[0]
x_stu_14 = preprocessing.normalize(x_stu_14, axis=0)
x_stu_14= np.reshape(x_stu_14, (-1,segement_time_size, sensors))
y_stu_14= np.delete( y_stu_14, [k for k in range(x_stu_14.shape[0],shape_y)], None)
p_stu_14 = np.delete(p_stu_14,[k for k in range(x_stu_14.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_stu_14), axis=0)
y= np.concatenate((y,y_stu_14), axis=0)
p= np.concatenate((p,p_stu_14), axis=0)
shp=(x_stu_15.shape)[0]
shape_y = y_stu_15.shape[0]
shape_p =  p_stu_15.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_stu_15 =  np.delete(x_stu_15, [k for k in range(nece_shp,shp)], None)
x_stu_15 = x_stu_15.reshape(-1,sensors)
shape_x_stu_15 = x_stu_15.shape[0]
x_stu_15 = preprocessing.normalize(x_stu_15, axis=0)
x_stu_15= np.reshape(x_stu_15, (-1,segement_time_size, sensors))
y_stu_15= np.delete( y_stu_15, [k for k in range(x_stu_15.shape[0],shape_y)], None)
p_stu_15 = np.delete(p_stu_15,[k for k in range(x_stu_15.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_stu_15), axis=0)
y= np.concatenate((y,y_stu_15), axis=0)
p= np.concatenate((p,p_stu_15), axis=0)
shp=(x_stu_16.shape)[0]
shape_y = y_stu_16.shape[0]
shape_p =  p_stu_16.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_stu_16 =  np.delete(x_stu_16, [k for k in range(nece_shp,shp)], None)
x_stu_16 = x_stu_16.reshape(-1,sensors)
shape_x_stu_16 = x_stu_16.shape[0]
x_stu_16 = preprocessing.normalize(x_stu_16, axis=0)
x_stu_16= np.reshape(x_stu_16, (-1,segement_time_size, sensors))
y_stu_16= np.delete( y_stu_16, [k for k in range(x_stu_16.shape[0],shape_y)], None)
p_stu_16 = np.delete(p_stu_16,[k for k in range(x_stu_16.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_stu_16), axis=0)
y= np.concatenate((y,y_stu_16), axis=0)
p= np.concatenate((p,p_stu_16), axis=0)
shp=(x_stu_17.shape)[0]
shape_y = y_stu_17.shape[0]
shape_p =  p_stu_17.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_stu_17 =  np.delete(x_stu_17, [k for k in range(nece_shp,shp)], None)
x_stu_17 = x_stu_17.reshape(-1,sensors)
shape_x_stu_17 = x_stu_17.shape[0]
x_stu_17 = preprocessing.normalize(x_stu_17, axis=0)
x_stu_17= np.reshape(x_stu_17, (-1,segement_time_size, sensors))
y_stu_17= np.delete( y_stu_17, [k for k in range(x_stu_17.shape[0],shape_y)], None)
p_stu_17 = np.delete(p_stu_17,[k for k in range(x_stu_17.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_stu_17), axis=0)
y= np.concatenate((y,y_stu_17), axis=0)
p= np.concatenate((p,p_stu_17), axis=0)
shp=(x_stu_18.shape)[0]
shape_y = y_stu_18.shape[0]
shape_p =  p_stu_18.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_stu_18 =  np.delete(x_stu_18, [k for k in range(nece_shp,shp)], None)
x_stu_18 = x_stu_18.reshape(-1,sensors)
shape_x_stu_18 = x_stu_18.shape[0]
x_stu_18 = preprocessing.normalize(x_stu_18, axis=0)
x_stu_18= np.reshape(x_stu_18, (-1,segement_time_size, sensors))
y_stu_18= np.delete( y_stu_18, [k for k in range(x_stu_18.shape[0],shape_y)], None)
p_stu_18 = np.delete(p_stu_18,[k for k in range(x_stu_18.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_stu_18), axis=0)
y= np.concatenate((y,y_stu_18), axis=0)
p= np.concatenate((p,p_stu_18), axis=0)
shp=(x_stu_19.shape)[0]
shape_y = y_stu_19.shape[0]
shape_p =  p_stu_19.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_stu_19 =  np.delete(x_stu_19, [k for k in range(nece_shp,shp)], None)
x_stu_19 = x_stu_19.reshape(-1,sensors)
shape_x_stu_19 = x_stu_19.shape[0]
x_stu_19 = preprocessing.normalize(x_stu_19, axis=0)
x_stu_19= np.reshape(x_stu_19, (-1,segement_time_size, sensors))
y_stu_19= np.delete( y_stu_19, [k for k in range(x_stu_19.shape[0],shape_y)], None)
p_stu_19 = np.delete(p_stu_19,[k for k in range(x_stu_19.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_stu_19), axis=0)
y= np.concatenate((y,y_stu_19), axis=0)
p= np.concatenate((p,p_stu_19), axis=0)
shp=(x_stu_20.shape)[0]
shape_y = y_stu_20.shape[0]
shape_p =  p_stu_20.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_stu_20 =  np.delete(x_stu_20, [k for k in range(nece_shp,shp)], None)
x_stu_20 = x_stu_20.reshape(-1,sensors)
shape_x_stu_20 = x_stu_20.shape[0]
x_stu_20 = preprocessing.normalize(x_stu_20, axis=0)
x_stu_20= np.reshape(x_stu_20, (-1,segement_time_size, sensors))
y_stu_20= np.delete( y_stu_20, [k for k in range(x_stu_20.shape[0],shape_y)], None)
p_stu_20 = np.delete(p_stu_20,[k for k in range(x_stu_20.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_stu_20), axis=0)
y= np.concatenate((y,y_stu_20), axis=0)
p= np.concatenate((p,p_stu_20), axis=0)
shp=(x_stu_21.shape)[0]
shape_y = y_stu_21.shape[0]
shape_p =  p_stu_21.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_stu_21 =  np.delete(x_stu_21, [k for k in range(nece_shp,shp)], None)
x_stu_21 = x_stu_21.reshape(-1,sensors)
shape_x_stu_21 = x_stu_21.shape[0]
x_stu_21 = preprocessing.normalize(x_stu_21, axis=0)
x_stu_21= np.reshape(x_stu_21, (-1,segement_time_size, sensors))
y_stu_21= np.delete( y_stu_21, [k for k in range(x_stu_21.shape[0],shape_y)], None)
p_stu_21 = np.delete(p_stu_21,[k for k in range(x_stu_21.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_stu_21), axis=0)
y= np.concatenate((y,y_stu_21), axis=0)
p= np.concatenate((p,p_stu_21), axis=0)
shp=(x_stu_22.shape)[0]
shape_y = y_stu_22.shape[0]
shape_p =  p_stu_22.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_stu_22 =  np.delete(x_stu_22, [k for k in range(nece_shp,shp)], None)
x_stu_22 = x_stu_22.reshape(-1,sensors)
shape_x_stu_22 = x_stu_22.shape[0]
x_stu_22 = preprocessing.normalize(x_stu_22, axis=0)
x_stu_22= np.reshape(x_stu_22, (-1,segement_time_size, sensors))
y_stu_22= np.delete( y_stu_22, [k for k in range(x_stu_22.shape[0],shape_y)], None)
p_stu_22 = np.delete(p_stu_22,[k for k in range(x_stu_22.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_stu_22), axis=0)
y= np.concatenate((y,y_stu_22), axis=0)
p= np.concatenate((p,p_stu_22), axis=0)
shp=(x_stu_23.shape)[0]
shape_y = y_stu_23.shape[0]
shape_p =  p_stu_23.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_stu_23 =  np.delete(x_stu_23, [k for k in range(nece_shp,shp)], None)
x_stu_23 = x_stu_23.reshape(-1,sensors)
shape_x_stu_23 = x_stu_23.shape[0]
x_stu_23 = preprocessing.normalize(x_stu_23, axis=0)
x_stu_23= np.reshape(x_stu_23, (-1,segement_time_size, sensors))
y_stu_23= np.delete( y_stu_23, [k for k in range(x_stu_23.shape[0],shape_y)], None)
p_stu_23 = np.delete(p_stu_23,[k for k in range(x_stu_23.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_stu_23), axis=0)
y= np.concatenate((y,y_stu_23), axis=0)
p= np.concatenate((p,p_stu_23), axis=0)
shp=(x_stu_24.shape)[0]
shape_y = y_stu_24.shape[0]
shape_p =  p_stu_24.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_stu_24 =  np.delete(x_stu_24, [k for k in range(nece_shp,shp)], None)
x_stu_24 = x_stu_24.reshape(-1,sensors)
shape_x_stu_24 = x_stu_24.shape[0]
x_stu_24 = preprocessing.normalize(x_stu_24, axis=0)
x_stu_24= np.reshape(x_stu_24, (-1,segement_time_size, sensors))
y_stu_24= np.delete( y_stu_24, [k for k in range(x_stu_24.shape[0],shape_y)], None)
p_stu_24 = np.delete(p_stu_24,[k for k in range(x_stu_24.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_stu_24), axis=0)
y= np.concatenate((y,y_stu_24), axis=0)
p= np.concatenate((p,p_stu_24), axis=0)
shp=(x_stu_25.shape)[0]
shape_y = y_stu_25.shape[0]
shape_p =  p_stu_25.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_stu_25 =  np.delete(x_stu_25, [k for k in range(nece_shp,shp)], None)
x_stu_25 = x_stu_25.reshape(-1,sensors)
shape_x_stu_25 = x_stu_25.shape[0]
x_stu_25 = preprocessing.normalize(x_stu_25, axis=0)
x_stu_25= np.reshape(x_stu_25, (-1,segement_time_size, sensors))
y_stu_25= np.delete( y_stu_25, [k for k in range(x_stu_25.shape[0],shape_y)], None)
p_stu_25 = np.delete(p_stu_25,[k for k in range(x_stu_25.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_stu_25), axis=0)
y= np.concatenate((y,y_stu_25), axis=0)
p= np.concatenate((p,p_stu_25), axis=0)
shp=(x_stu_26.shape)[0]
shape_y = y_stu_26.shape[0]
shape_p =  p_stu_26.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_stu_26 =  np.delete(x_stu_26, [k for k in range(nece_shp,shp)], None)
x_stu_26 = x_stu_26.reshape(-1,sensors)
shape_x_stu_26 = x_stu_26.shape[0]
x_stu_26 = preprocessing.normalize(x_stu_26, axis=0)
x_stu_26= np.reshape(x_stu_26, (-1,segement_time_size, sensors))
y_stu_26= np.delete( y_stu_26, [k for k in range(x_stu_26.shape[0],shape_y)], None)
p_stu_26 = np.delete(p_stu_26,[k for k in range(x_stu_26.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_stu_26), axis=0)
y= np.concatenate((y,y_stu_26), axis=0)
p= np.concatenate((p,p_stu_26), axis=0)
shp=(x_stu_27.shape)[0]
shape_y = y_stu_27.shape[0]
shape_p =  p_stu_27.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_stu_27 =  np.delete(x_stu_27, [k for k in range(nece_shp,shp)], None)
x_stu_27 = x_stu_27.reshape(-1,sensors)
shape_x_stu_27 = x_stu_27.shape[0]
x_stu_27 = preprocessing.normalize(x_stu_27, axis=0)
x_stu_27= np.reshape(x_stu_27, (-1,segement_time_size, sensors))
y_stu_27= np.delete( y_stu_27, [k for k in range(x_stu_27.shape[0],shape_y)], None)
p_stu_27 = np.delete(p_stu_27,[k for k in range(x_stu_27.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_stu_27), axis=0)
y= np.concatenate((y,y_stu_27), axis=0)
p= np.concatenate((p,p_stu_27), axis=0)
shp=(x_stu_28.shape)[0]
shape_y = y_stu_28.shape[0]
shape_p =  p_stu_28.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_stu_28 =  np.delete(x_stu_28, [k for k in range(nece_shp,shp)], None)
x_stu_28 = x_stu_28.reshape(-1,sensors)
shape_x_stu_28 = x_stu_28.shape[0]
x_stu_28 = preprocessing.normalize(x_stu_28, axis=0)
x_stu_28= np.reshape(x_stu_28, (-1,segement_time_size, sensors))
y_stu_28= np.delete( y_stu_28, [k for k in range(x_stu_28.shape[0],shape_y)], None)
p_stu_28 = np.delete(p_stu_28,[k for k in range(x_stu_28.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_stu_28), axis=0)
y= np.concatenate((y,y_stu_28), axis=0)
p= np.concatenate((p,p_stu_28), axis=0)
shp=(x_stu_29.shape)[0]
shape_y = y_stu_29.shape[0]
shape_p =  p_stu_29.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_stu_29 =  np.delete(x_stu_29, [k for k in range(nece_shp,shp)], None)
x_stu_29 = x_stu_29.reshape(-1,sensors)
shape_x_stu_29 = x_stu_29.shape[0]
x_stu_29 = preprocessing.normalize(x_stu_29, axis=0)
x_stu_29= np.reshape(x_stu_29, (-1,segement_time_size, sensors))
y_stu_29= np.delete( y_stu_29, [k for k in range(x_stu_29.shape[0],shape_y)], None)
p_stu_29 = np.delete(p_stu_29,[k for k in range(x_stu_29.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_stu_29), axis=0)
y= np.concatenate((y,y_stu_29), axis=0)
p= np.concatenate((p,p_stu_29), axis=0)
shp=(x_stu_30.shape)[0]
shape_y = y_stu_30.shape[0]
shape_p =  p_stu_30.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_stu_30 =  np.delete(x_stu_30, [k for k in range(nece_shp,shp)], None)
x_stu_30 = x_stu_30.reshape(-1,sensors)
shape_x_stu_30 = x_stu_30.shape[0]
x_stu_30 = preprocessing.normalize(x_stu_30, axis=0)
x_stu_30= np.reshape(x_stu_30, (-1,segement_time_size, sensors))
y_stu_30= np.delete( y_stu_30, [k for k in range(x_stu_30.shape[0],shape_y)], None)
p_stu_30 = np.delete(p_stu_30,[k for k in range(x_stu_30.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_stu_30), axis=0)
y= np.concatenate((y,y_stu_30), axis=0)
p= np.concatenate((p,p_stu_30), axis=0)
shp=(x_stu_31.shape)[0]
shape_y = y_stu_31.shape[0]
shape_p =  p_stu_31.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_stu_31 =  np.delete(x_stu_31, [k for k in range(nece_shp,shp)], None)
x_stu_31 = x_stu_31.reshape(-1,sensors)
shape_x_stu_31 = x_stu_31.shape[0]
x_stu_31 = preprocessing.normalize(x_stu_31, axis=0)
x_stu_31= np.reshape(x_stu_31, (-1,segement_time_size, sensors))
y_stu_31= np.delete( y_stu_31, [k for k in range(x_stu_31.shape[0],shape_y)], None)
p_stu_31 = np.delete(p_stu_31,[k for k in range(x_stu_31.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_stu_31), axis=0)
y= np.concatenate((y,y_stu_31), axis=0)
p= np.concatenate((p,p_stu_31), axis=0)
shp=(x_stu_32.shape)[0]
shape_y = y_stu_32.shape[0]
shape_p =  p_stu_32.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_stu_32 =  np.delete(x_stu_32, [k for k in range(nece_shp,shp)], None)
x_stu_32 = x_stu_32.reshape(-1,sensors)
shape_x_stu_32 = x_stu_32.shape[0]
x_stu_32 = preprocessing.normalize(x_stu_32, axis=0)
x_stu_32= np.reshape(x_stu_32, (-1,segement_time_size, sensors))
y_stu_32= np.delete( y_stu_32, [k for k in range(x_stu_32.shape[0],shape_y)], None)
p_stu_32 = np.delete(p_stu_32,[k for k in range(x_stu_32.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_stu_32), axis=0)
y= np.concatenate((y,y_stu_32), axis=0)
p= np.concatenate((p,p_stu_32), axis=0)
shp=(x_stu_33.shape)[0]
shape_y = y_stu_33.shape[0]
shape_p =  p_stu_33.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_stu_33 =  np.delete(x_stu_33, [k for k in range(nece_shp,shp)], None)
x_stu_33 = x_stu_33.reshape(-1,sensors)
shape_x_stu_33 = x_stu_33.shape[0]
x_stu_33 = preprocessing.normalize(x_stu_33, axis=0)
x_stu_33= np.reshape(x_stu_33, (-1,segement_time_size, sensors))
y_stu_33= np.delete( y_stu_33, [k for k in range(x_stu_33.shape[0],shape_y)], None)
p_stu_33 = np.delete(p_stu_33,[k for k in range(x_stu_33.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_stu_33), axis=0)
y= np.concatenate((y,y_stu_33), axis=0)
p= np.concatenate((p,p_stu_33), axis=0)
shp=(x_stu_34.shape)[0]
shape_y = y_stu_34.shape[0]
shape_p =  p_stu_34.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_stu_34 =  np.delete(x_stu_34, [k for k in range(nece_shp,shp)], None)
x_stu_34 = x_stu_34.reshape(-1,sensors)
shape_x_stu_34 = x_stu_34.shape[0]
x_stu_34 = preprocessing.normalize(x_stu_34, axis=0)
x_stu_34= np.reshape(x_stu_34, (-1,segement_time_size, sensors))
y_stu_34= np.delete( y_stu_34, [k for k in range(x_stu_34.shape[0],shape_y)], None)
p_stu_34 = np.delete(p_stu_34,[k for k in range(x_stu_34.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_stu_34), axis=0)
y= np.concatenate((y,y_stu_34), axis=0)
p= np.concatenate((p,p_stu_34), axis=0)
shp=(x_stu_35.shape)[0]
shape_y = y_stu_35.shape[0]
shape_p =  p_stu_35.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_stu_35 =  np.delete(x_stu_35, [k for k in range(nece_shp,shp)], None)
x_stu_35 = x_stu_35.reshape(-1,sensors)
shape_x_stu_35 = x_stu_35.shape[0]
x_stu_35 = preprocessing.normalize(x_stu_35, axis=0)
x_stu_35= np.reshape(x_stu_35, (-1,segement_time_size, sensors))
y_stu_35= np.delete( y_stu_35, [k for k in range(x_stu_35.shape[0],shape_y)], None)
p_stu_35 = np.delete(p_stu_35,[k for k in range(x_stu_35.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_stu_35), axis=0)
y= np.concatenate((y,y_stu_35), axis=0)
p= np.concatenate((p,p_stu_35), axis=0)
shp=(x_stu_36.shape)[0]
shape_y = y_stu_36.shape[0]
shape_p =  p_stu_36.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_stu_36 =  np.delete(x_stu_36, [k for k in range(nece_shp,shp)], None)
x_stu_36 = x_stu_36.reshape(-1,sensors)
shape_x_stu_36 = x_stu_36.shape[0]
x_stu_36 = preprocessing.normalize(x_stu_36, axis=0)
x_stu_36= np.reshape(x_stu_36, (-1,segement_time_size, sensors))
y_stu_36= np.delete( y_stu_36, [k for k in range(x_stu_36.shape[0],shape_y)], None)
p_stu_36 = np.delete(p_stu_36,[k for k in range(x_stu_36.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_stu_36), axis=0)
y= np.concatenate((y,y_stu_36), axis=0)
p= np.concatenate((p,p_stu_36), axis=0)
shp=(x_stu_37.shape)[0]
shape_y = y_stu_37.shape[0]
shape_p =  p_stu_37.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_stu_37 =  np.delete(x_stu_37, [k for k in range(nece_shp,shp)], None)
x_stu_37 = x_stu_37.reshape(-1,sensors)
shape_x_stu_37 = x_stu_37.shape[0]
x_stu_37 = preprocessing.normalize(x_stu_37, axis=0)
x_stu_37= np.reshape(x_stu_37, (-1,segement_time_size, sensors))
y_stu_37= np.delete( y_stu_37, [k for k in range(x_stu_37.shape[0],shape_y)], None)
p_stu_37 = np.delete(p_stu_37,[k for k in range(x_stu_37.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_stu_37), axis=0)
y= np.concatenate((y,y_stu_37), axis=0)
p= np.concatenate((p,p_stu_37), axis=0)
shp=(x_stu_38.shape)[0]
shape_y = y_stu_38.shape[0]
shape_p =  p_stu_38.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_stu_38 =  np.delete(x_stu_38, [k for k in range(nece_shp,shp)], None)
x_stu_38 = x_stu_38.reshape(-1,sensors)
shape_x_stu_38 = x_stu_38.shape[0]
x_stu_38 = preprocessing.normalize(x_stu_38, axis=0)
x_stu_38= np.reshape(x_stu_38, (-1,segement_time_size, sensors))
y_stu_38= np.delete( y_stu_38, [k for k in range(x_stu_38.shape[0],shape_y)], None)
p_stu_38 = np.delete(p_stu_38,[k for k in range(x_stu_38.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_stu_38), axis=0)
y= np.concatenate((y,y_stu_38), axis=0)
p= np.concatenate((p,p_stu_38), axis=0)
shp=(x_stu_39.shape)[0]
shape_y = y_stu_39.shape[0]
shape_p =  p_stu_39.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_stu_39 =  np.delete(x_stu_39, [k for k in range(nece_shp,shp)], None)
x_stu_39 = x_stu_39.reshape(-1,sensors)
shape_x_stu_39 = x_stu_39.shape[0]
x_stu_39 = preprocessing.normalize(x_stu_39, axis=0)
x_stu_39= np.reshape(x_stu_39, (-1,segement_time_size, sensors))
y_stu_39= np.delete( y_stu_39, [k for k in range(x_stu_39.shape[0],shape_y)], None)
p_stu_39 = np.delete(p_stu_39,[k for k in range(x_stu_39.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_stu_39), axis=0)
y= np.concatenate((y,y_stu_39), axis=0)
p= np.concatenate((p,p_stu_39), axis=0)
shp=(x_stu_40.shape)[0]
shape_y = y_stu_40.shape[0]
shape_p =  p_stu_40.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_stu_40 =  np.delete(x_stu_40, [k for k in range(nece_shp,shp)], None)
x_stu_40 = x_stu_40.reshape(-1,sensors)
shape_x_stu_40 = x_stu_40.shape[0]
x_stu_40 = preprocessing.normalize(x_stu_40, axis=0)
x_stu_40= np.reshape(x_stu_40, (-1,segement_time_size, sensors))
y_stu_40= np.delete( y_stu_40, [k for k in range(x_stu_40.shape[0],shape_y)], None)
p_stu_40 = np.delete(p_stu_40,[k for k in range(x_stu_40.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_stu_40), axis=0)
y= np.concatenate((y,y_stu_40), axis=0)
p= np.concatenate((p,p_stu_40), axis=0)
shp=(x_stu_41.shape)[0]
shape_y = y_stu_41.shape[0]
shape_p =  p_stu_41.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_stu_41 =  np.delete(x_stu_41, [k for k in range(nece_shp,shp)], None)
x_stu_41 = x_stu_41.reshape(-1,sensors)
shape_x_stu_41 = x_stu_41.shape[0]
x_stu_41 = preprocessing.normalize(x_stu_41, axis=0)
x_stu_41= np.reshape(x_stu_41, (-1,segement_time_size, sensors))
y_stu_41= np.delete( y_stu_41, [k for k in range(x_stu_41.shape[0],shape_y)], None)
p_stu_41 = np.delete(p_stu_41,[k for k in range(x_stu_41.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_stu_41), axis=0)
y= np.concatenate((y,y_stu_41), axis=0)
p= np.concatenate((p,p_stu_41), axis=0)
shp=(x_stu_42.shape)[0]
shape_y = y_stu_42.shape[0]
shape_p =  p_stu_42.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_stu_42 =  np.delete(x_stu_42, [k for k in range(nece_shp,shp)], None)
x_stu_42 = x_stu_42.reshape(-1,sensors)
shape_x_stu_42 = x_stu_42.shape[0]
x_stu_42 = preprocessing.normalize(x_stu_42, axis=0)
x_stu_42= np.reshape(x_stu_42, (-1,segement_time_size, sensors))
y_stu_42= np.delete( y_stu_42, [k for k in range(x_stu_42.shape[0],shape_y)], None)
p_stu_42 = np.delete(p_stu_42,[k for k in range(x_stu_42.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_stu_42), axis=0)
y= np.concatenate((y,y_stu_42), axis=0)
p= np.concatenate((p,p_stu_42), axis=0)
shp=(x_stu_43.shape)[0]
shape_y = y_stu_43.shape[0]
shape_p =  p_stu_43.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_stu_43 =  np.delete(x_stu_43, [k for k in range(nece_shp,shp)], None)
x_stu_43 = x_stu_43.reshape(-1,sensors)
shape_x_stu_43 = x_stu_43.shape[0]
x_stu_43 = preprocessing.normalize(x_stu_43, axis=0)
x_stu_43= np.reshape(x_stu_43, (-1,segement_time_size, sensors))
y_stu_43= np.delete( y_stu_43, [k for k in range(x_stu_43.shape[0],shape_y)], None)
p_stu_43 = np.delete(p_stu_43,[k for k in range(x_stu_43.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_stu_43), axis=0)
y= np.concatenate((y,y_stu_43), axis=0)
p= np.concatenate((p,p_stu_43), axis=0)
shp=(x_stu_44.shape)[0]
shape_y = y_stu_44.shape[0]
shape_p =  p_stu_44.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_stu_44 =  np.delete(x_stu_44, [k for k in range(nece_shp,shp)], None)
x_stu_44 = x_stu_44.reshape(-1,sensors)
shape_x_stu_44 = x_stu_44.shape[0]
x_stu_44 = preprocessing.normalize(x_stu_44, axis=0)
x_stu_44= np.reshape(x_stu_44, (-1,segement_time_size, sensors))
y_stu_44= np.delete( y_stu_44, [k for k in range(x_stu_44.shape[0],shape_y)], None)
p_stu_44 = np.delete(p_stu_44,[k for k in range(x_stu_44.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_stu_44), axis=0)
y= np.concatenate((y,y_stu_44), axis=0)
p= np.concatenate((p,p_stu_44), axis=0)
shp=(x_stu_45.shape)[0]
shape_y = y_stu_45.shape[0]
shape_p =  p_stu_45.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_stu_45 =  np.delete(x_stu_45, [k for k in range(nece_shp,shp)], None)
x_stu_45 = x_stu_45.reshape(-1,sensors)
shape_x_stu_45 = x_stu_45.shape[0]
x_stu_45 = preprocessing.normalize(x_stu_45, axis=0)
x_stu_45= np.reshape(x_stu_45, (-1,segement_time_size, sensors))
y_stu_45= np.delete( y_stu_45, [k for k in range(x_stu_45.shape[0],shape_y)], None)
p_stu_45 = np.delete(p_stu_45,[k for k in range(x_stu_45.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_stu_45), axis=0)
y= np.concatenate((y,y_stu_45), axis=0)
p= np.concatenate((p,p_stu_45), axis=0)
shp=(x_stu_46.shape)[0]
shape_y = y_stu_46.shape[0]
shape_p =  p_stu_46.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_stu_46 =  np.delete(x_stu_46, [k for k in range(nece_shp,shp)], None)
x_stu_46 = x_stu_46.reshape(-1,sensors)
shape_x_stu_46 = x_stu_46.shape[0]
x_stu_46 = preprocessing.normalize(x_stu_46, axis=0)
x_stu_46= np.reshape(x_stu_46, (-1,segement_time_size, sensors))
y_stu_46= np.delete( y_stu_46, [k for k in range(x_stu_46.shape[0],shape_y)], None)
p_stu_46 = np.delete(p_stu_46,[k for k in range(x_stu_46.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_stu_46), axis=0)
y= np.concatenate((y,y_stu_46), axis=0)
p= np.concatenate((p,p_stu_46), axis=0)
shp=(x_stu_47.shape)[0]
shape_y = y_stu_47.shape[0]
shape_p =  p_stu_47.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_stu_47 =  np.delete(x_stu_47, [k for k in range(nece_shp,shp)], None)
x_stu_47 = x_stu_47.reshape(-1,sensors)
shape_x_stu_47 = x_stu_47.shape[0]
x_stu_47 = preprocessing.normalize(x_stu_47, axis=0)
x_stu_47= np.reshape(x_stu_47, (-1,segement_time_size, sensors))
y_stu_47= np.delete( y_stu_47, [k for k in range(x_stu_47.shape[0],shape_y)], None)
p_stu_47 = np.delete(p_stu_47,[k for k in range(x_stu_47.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_stu_47), axis=0)
y= np.concatenate((y,y_stu_47), axis=0)
p= np.concatenate((p,p_stu_47), axis=0)
shp=(x_stu_48.shape)[0]
shape_y = y_stu_48.shape[0]
shape_p =  p_stu_48.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_stu_48 =  np.delete(x_stu_48, [k for k in range(nece_shp,shp)], None)
x_stu_48 = x_stu_48.reshape(-1,sensors)
shape_x_stu_48 = x_stu_48.shape[0]
x_stu_48 = preprocessing.normalize(x_stu_48, axis=0)
x_stu_48= np.reshape(x_stu_48, (-1,segement_time_size, sensors))
y_stu_48= np.delete( y_stu_48, [k for k in range(x_stu_48.shape[0],shape_y)], None)
p_stu_48 = np.delete(p_stu_48,[k for k in range(x_stu_48.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_stu_48), axis=0)
y= np.concatenate((y,y_stu_48), axis=0)
p= np.concatenate((p,p_stu_48), axis=0)
shp=(x_stu_49.shape)[0]
shape_y = y_stu_49.shape[0]
shape_p =  p_stu_49.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_stu_49 =  np.delete(x_stu_49, [k for k in range(nece_shp,shp)], None)
x_stu_49 = x_stu_49.reshape(-1,sensors)
shape_x_stu_49 = x_stu_49.shape[0]
x_stu_49 = preprocessing.normalize(x_stu_49, axis=0)
x_stu_49= np.reshape(x_stu_49, (-1,segement_time_size, sensors))
y_stu_49= np.delete( y_stu_49, [k for k in range(x_stu_49.shape[0],shape_y)], None)
p_stu_49 = np.delete(p_stu_49,[k for k in range(x_stu_49.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_stu_49), axis=0)
y= np.concatenate((y,y_stu_49), axis=0)
p= np.concatenate((p,p_stu_49), axis=0)
shp=(x_stu_50.shape)[0]
shape_y = y_stu_50.shape[0]
shape_p =  p_stu_50.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_stu_50 =  np.delete(x_stu_50, [k for k in range(nece_shp,shp)], None)
x_stu_50 = x_stu_50.reshape(-1,sensors)
shape_x_stu_50 = x_stu_50.shape[0]
x_stu_50 = preprocessing.normalize(x_stu_50, axis=0)
x_stu_50= np.reshape(x_stu_50, (-1,segement_time_size, sensors))
y_stu_50= np.delete( y_stu_50, [k for k in range(x_stu_50.shape[0],shape_y)], None)
p_stu_50 = np.delete(p_stu_50,[k for k in range(x_stu_50.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_stu_50), axis=0)
y= np.concatenate((y,y_stu_50), axis=0)
p= np.concatenate((p,p_stu_50), axis=0)
shp=(x_stu_51.shape)[0]
shape_y = y_stu_51.shape[0]
shape_p =  p_stu_51.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_stu_51 =  np.delete(x_stu_51, [k for k in range(nece_shp,shp)], None)
x_stu_51 = x_stu_51.reshape(-1,sensors)
shape_x_stu_51 = x_stu_51.shape[0]
x_stu_51 = preprocessing.normalize(x_stu_51, axis=0)
x_stu_51= np.reshape(x_stu_51, (-1,segement_time_size, sensors))
y_stu_51= np.delete( y_stu_51, [k for k in range(x_stu_51.shape[0],shape_y)], None)
p_stu_51 = np.delete(p_stu_51,[k for k in range(x_stu_51.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_stu_51), axis=0)
y= np.concatenate((y,y_stu_51), axis=0)
p= np.concatenate((p,p_stu_51), axis=0)
shp=(x_stu_52.shape)[0]
shape_y = y_stu_52.shape[0]
shape_p =  p_stu_52.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_stu_52 =  np.delete(x_stu_52, [k for k in range(nece_shp,shp)], None)
x_stu_52 = x_stu_52.reshape(-1,sensors)
shape_x_stu_52 = x_stu_52.shape[0]
x_stu_52 = preprocessing.normalize(x_stu_52, axis=0)
x_stu_52= np.reshape(x_stu_52, (-1,segement_time_size, sensors))
y_stu_52= np.delete( y_stu_52, [k for k in range(x_stu_52.shape[0],shape_y)], None)
p_stu_52 = np.delete(p_stu_52,[k for k in range(x_stu_52.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_stu_52), axis=0)
y= np.concatenate((y,y_stu_52), axis=0)
p= np.concatenate((p,p_stu_52), axis=0)
shp=(x_stu_53.shape)[0]
shape_y = y_stu_53.shape[0]
shape_p =  p_stu_53.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_stu_53 =  np.delete(x_stu_53, [k for k in range(nece_shp,shp)], None)
x_stu_53 = x_stu_53.reshape(-1,sensors)
shape_x_stu_53 = x_stu_53.shape[0]
x_stu_53 = preprocessing.normalize(x_stu_53, axis=0)
x_stu_53= np.reshape(x_stu_53, (-1,segement_time_size, sensors))
y_stu_53= np.delete( y_stu_53, [k for k in range(x_stu_53.shape[0],shape_y)], None)
p_stu_53 = np.delete(p_stu_53,[k for k in range(x_stu_53.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_stu_53), axis=0)
y= np.concatenate((y,y_stu_53), axis=0)
p= np.concatenate((p,p_stu_53), axis=0)
shp=(x_stu_54.shape)[0]
shape_y = y_stu_54.shape[0]
shape_p =  p_stu_54.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_stu_54 =  np.delete(x_stu_54, [k for k in range(nece_shp,shp)], None)
x_stu_54 = x_stu_54.reshape(-1,sensors)
shape_x_stu_54 = x_stu_54.shape[0]
x_stu_54 = preprocessing.normalize(x_stu_54, axis=0)
x_stu_54= np.reshape(x_stu_54, (-1,segement_time_size, sensors))
y_stu_54= np.delete( y_stu_54, [k for k in range(x_stu_54.shape[0],shape_y)], None)
p_stu_54 = np.delete(p_stu_54,[k for k in range(x_stu_54.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_stu_54), axis=0)
y= np.concatenate((y,y_stu_54), axis=0)
p= np.concatenate((p,p_stu_54), axis=0)
shp=(x_stu_55.shape)[0]
shape_y = y_stu_55.shape[0]
shape_p =  p_stu_55.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_stu_55 =  np.delete(x_stu_55, [k for k in range(nece_shp,shp)], None)
x_stu_55 = x_stu_55.reshape(-1,sensors)
shape_x_stu_55 = x_stu_55.shape[0]
x_stu_55 = preprocessing.normalize(x_stu_55, axis=0)
x_stu_55= np.reshape(x_stu_55, (-1,segement_time_size, sensors))
y_stu_55= np.delete( y_stu_55, [k for k in range(x_stu_55.shape[0],shape_y)], None)
p_stu_55 = np.delete(p_stu_55,[k for k in range(x_stu_55.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_stu_55), axis=0)
y= np.concatenate((y,y_stu_55), axis=0)
p= np.concatenate((p,p_stu_55), axis=0)
shp=(x_stu_56.shape)[0]
shape_y = y_stu_56.shape[0]
shape_p =  p_stu_56.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_stu_56 =  np.delete(x_stu_56, [k for k in range(nece_shp,shp)], None)
x_stu_56 = x_stu_56.reshape(-1,sensors)
shape_x_stu_56 = x_stu_56.shape[0]
x_stu_56 = preprocessing.normalize(x_stu_56, axis=0)
x_stu_56= np.reshape(x_stu_56, (-1,segement_time_size, sensors))
y_stu_56= np.delete( y_stu_56, [k for k in range(x_stu_56.shape[0],shape_y)], None)
p_stu_56 = np.delete(p_stu_56,[k for k in range(x_stu_56.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_stu_56), axis=0)
y= np.concatenate((y,y_stu_56), axis=0)
p= np.concatenate((p,p_stu_56), axis=0)
shp=(x_stu_57.shape)[0]
shape_y = y_stu_57.shape[0]
shape_p =  p_stu_57.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_stu_57 =  np.delete(x_stu_57, [k for k in range(nece_shp,shp)], None)
x_stu_57 = x_stu_57.reshape(-1,sensors)
shape_x_stu_57 = x_stu_57.shape[0]
x_stu_57 = preprocessing.normalize(x_stu_57, axis=0)
x_stu_57= np.reshape(x_stu_57, (-1,segement_time_size, sensors))
y_stu_57= np.delete( y_stu_57, [k for k in range(x_stu_57.shape[0],shape_y)], None)
p_stu_57 = np.delete(p_stu_57,[k for k in range(x_stu_57.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_stu_57), axis=0)
y= np.concatenate((y,y_stu_57), axis=0)
p= np.concatenate((p,p_stu_57), axis=0)
shp=(x_stu_58.shape)[0]
shape_y = y_stu_58.shape[0]
shape_p =  p_stu_58.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_stu_58 =  np.delete(x_stu_58, [k for k in range(nece_shp,shp)], None)
x_stu_58 = x_stu_58.reshape(-1,sensors)
shape_x_stu_58 = x_stu_58.shape[0]
x_stu_58 = preprocessing.normalize(x_stu_58, axis=0)
x_stu_58= np.reshape(x_stu_58, (-1,segement_time_size, sensors))
y_stu_58= np.delete( y_stu_58, [k for k in range(x_stu_58.shape[0],shape_y)], None)
p_stu_58 = np.delete(p_stu_58,[k for k in range(x_stu_58.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_stu_58), axis=0)
y= np.concatenate((y,y_stu_58), axis=0)
p= np.concatenate((p,p_stu_58), axis=0)
shp=(x_stu_59.shape)[0]
shape_y = y_stu_59.shape[0]
shape_p =  p_stu_59.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_stu_59 =  np.delete(x_stu_59, [k for k in range(nece_shp,shp)], None)
x_stu_59 = x_stu_59.reshape(-1,sensors)
shape_x_stu_59 = x_stu_59.shape[0]
x_stu_59 = preprocessing.normalize(x_stu_59, axis=0)
x_stu_59= np.reshape(x_stu_59, (-1,segement_time_size, sensors))
y_stu_59= np.delete( y_stu_59, [k for k in range(x_stu_59.shape[0],shape_y)], None)
p_stu_59 = np.delete(p_stu_59,[k for k in range(x_stu_59.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_stu_59), axis=0)
y= np.concatenate((y,y_stu_59), axis=0)
p= np.concatenate((p,p_stu_59), axis=0)
shp=(x_stu_60.shape)[0]
shape_y = y_stu_60.shape[0]
shape_p =  p_stu_60.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_stu_60 =  np.delete(x_stu_60, [k for k in range(nece_shp,shp)], None)
x_stu_60 = x_stu_60.reshape(-1,sensors)
shape_x_stu_60 = x_stu_60.shape[0]
x_stu_60 = preprocessing.normalize(x_stu_60, axis=0)
x_stu_60= np.reshape(x_stu_60, (-1,segement_time_size, sensors))
y_stu_60= np.delete( y_stu_60, [k for k in range(x_stu_60.shape[0],shape_y)], None)
p_stu_60 = np.delete(p_stu_60,[k for k in range(x_stu_60.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_stu_60), axis=0)
y= np.concatenate((y,y_stu_60), axis=0)
p= np.concatenate((p,p_stu_60), axis=0)
shp=(x_stu_61.shape)[0]
shape_y = y_stu_61.shape[0]
shape_p =  p_stu_61.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_stu_61 =  np.delete(x_stu_61, [k for k in range(nece_shp,shp)], None)
x_stu_61 = x_stu_61.reshape(-1,sensors)
shape_x_stu_61 = x_stu_61.shape[0]
x_stu_61 = preprocessing.normalize(x_stu_61, axis=0)
x_stu_61= np.reshape(x_stu_61, (-1,segement_time_size, sensors))
y_stu_61= np.delete( y_stu_61, [k for k in range(x_stu_61.shape[0],shape_y)], None)
p_stu_61 = np.delete(p_stu_61,[k for k in range(x_stu_61.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_stu_61), axis=0)
y= np.concatenate((y,y_stu_61), axis=0)
p= np.concatenate((p,p_stu_61), axis=0)
shp=(x_stu_62.shape)[0]
shape_y = y_stu_62.shape[0]
shape_p =  p_stu_62.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_stu_62 =  np.delete(x_stu_62, [k for k in range(nece_shp,shp)], None)
x_stu_62 = x_stu_62.reshape(-1,sensors)
shape_x_stu_62 = x_stu_62.shape[0]
x_stu_62 = preprocessing.normalize(x_stu_62, axis=0)
x_stu_62= np.reshape(x_stu_62, (-1,segement_time_size, sensors))
y_stu_62= np.delete( y_stu_62, [k for k in range(x_stu_62.shape[0],shape_y)], None)
p_stu_62 = np.delete(p_stu_62,[k for k in range(x_stu_62.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_stu_62), axis=0)
y= np.concatenate((y,y_stu_62), axis=0)
p= np.concatenate((p,p_stu_62), axis=0)
shp=(x_stu_63.shape)[0]
shape_y = y_stu_63.shape[0]
shape_p =  p_stu_63.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_stu_63 =  np.delete(x_stu_63, [k for k in range(nece_shp,shp)], None)
x_stu_63 = x_stu_63.reshape(-1,sensors)
shape_x_stu_63 = x_stu_63.shape[0]
x_stu_63 = preprocessing.normalize(x_stu_63, axis=0)
x_stu_63= np.reshape(x_stu_63, (-1,segement_time_size, sensors))
y_stu_63= np.delete( y_stu_63, [k for k in range(x_stu_63.shape[0],shape_y)], None)
p_stu_63 = np.delete(p_stu_63,[k for k in range(x_stu_63.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_stu_63), axis=0)
y= np.concatenate((y,y_stu_63), axis=0)
p= np.concatenate((p,p_stu_63), axis=0)
shp=(x_stu_64.shape)[0]
shape_y = y_stu_64.shape[0]
shape_p =  p_stu_64.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_stu_64 =  np.delete(x_stu_64, [k for k in range(nece_shp,shp)], None)
x_stu_64 = x_stu_64.reshape(-1,sensors)
shape_x_stu_64 = x_stu_64.shape[0]
x_stu_64 = preprocessing.normalize(x_stu_64, axis=0)
x_stu_64= np.reshape(x_stu_64, (-1,segement_time_size, sensors))
y_stu_64= np.delete( y_stu_64, [k for k in range(x_stu_64.shape[0],shape_y)], None)
p_stu_64 = np.delete(p_stu_64,[k for k in range(x_stu_64.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_stu_64), axis=0)
y= np.concatenate((y,y_stu_64), axis=0)
p= np.concatenate((p,p_stu_64), axis=0)
shp=(x_stu_65.shape)[0]
shape_y = y_stu_65.shape[0]
shape_p =  p_stu_65.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_stu_65 =  np.delete(x_stu_65, [k for k in range(nece_shp,shp)], None)
x_stu_65 = x_stu_65.reshape(-1,sensors)
shape_x_stu_65 = x_stu_65.shape[0]
x_stu_65 = preprocessing.normalize(x_stu_65, axis=0)
x_stu_65= np.reshape(x_stu_65, (-1,segement_time_size, sensors))
y_stu_65= np.delete( y_stu_65, [k for k in range(x_stu_65.shape[0],shape_y)], None)
p_stu_65 = np.delete(p_stu_65,[k for k in range(x_stu_65.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_stu_65), axis=0)
y= np.concatenate((y,y_stu_65), axis=0)
p= np.concatenate((p,p_stu_65), axis=0)
shp=(x_stu_66.shape)[0]
shape_y = y_stu_66.shape[0]
shape_p =  p_stu_66.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_stu_66 =  np.delete(x_stu_66, [k for k in range(nece_shp,shp)], None)
x_stu_66 = x_stu_66.reshape(-1,sensors)
shape_x_stu_66 = x_stu_66.shape[0]
x_stu_66 = preprocessing.normalize(x_stu_66, axis=0)
x_stu_66= np.reshape(x_stu_66, (-1,segement_time_size, sensors))
y_stu_66= np.delete( y_stu_66, [k for k in range(x_stu_66.shape[0],shape_y)], None)
p_stu_66 = np.delete(p_stu_66,[k for k in range(x_stu_66.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_stu_66), axis=0)
y= np.concatenate((y,y_stu_66), axis=0)
p= np.concatenate((p,p_stu_66), axis=0)
shp=(x_stu_67.shape)[0]
shape_y = y_stu_67.shape[0]
shape_p =  p_stu_67.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_stu_67 =  np.delete(x_stu_67, [k for k in range(nece_shp,shp)], None)
x_stu_67 = x_stu_67.reshape(-1,sensors)
shape_x_stu_67 = x_stu_67.shape[0]
x_stu_67 = preprocessing.normalize(x_stu_67, axis=0)
x_stu_67= np.reshape(x_stu_67, (-1,segement_time_size, sensors))
y_stu_67= np.delete( y_stu_67, [k for k in range(x_stu_67.shape[0],shape_y)], None)
p_stu_67 = np.delete(p_stu_67,[k for k in range(x_stu_67.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_stu_67), axis=0)
y= np.concatenate((y,y_stu_67), axis=0)
p= np.concatenate((p,p_stu_67), axis=0)
shp=(x_stu_68.shape)[0]
shape_y = y_stu_68.shape[0]
shape_p =  p_stu_68.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_stu_68 =  np.delete(x_stu_68, [k for k in range(nece_shp,shp)], None)
x_stu_68 = x_stu_68.reshape(-1,sensors)
shape_x_stu_68 = x_stu_68.shape[0]
x_stu_68 = preprocessing.normalize(x_stu_68, axis=0)
x_stu_68= np.reshape(x_stu_68, (-1,segement_time_size, sensors))
y_stu_68= np.delete( y_stu_68, [k for k in range(x_stu_68.shape[0],shape_y)], None)
p_stu_68 = np.delete(p_stu_68,[k for k in range(x_stu_68.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_stu_68), axis=0)
y= np.concatenate((y,y_stu_68), axis=0)
p= np.concatenate((p,p_stu_68), axis=0)
shp=(x_stu_69.shape)[0]
shape_y = y_stu_69.shape[0]
shape_p =  p_stu_69.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_stu_69 =  np.delete(x_stu_69, [k for k in range(nece_shp,shp)], None)
x_stu_69 = x_stu_69.reshape(-1,sensors)
shape_x_stu_69 = x_stu_69.shape[0]
x_stu_69 = preprocessing.normalize(x_stu_69, axis=0)
x_stu_69= np.reshape(x_stu_69, (-1,segement_time_size, sensors))
y_stu_69= np.delete( y_stu_69, [k for k in range(x_stu_69.shape[0],shape_y)], None)
p_stu_69 = np.delete(p_stu_69,[k for k in range(x_stu_69.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_stu_69), axis=0)
y= np.concatenate((y,y_stu_69), axis=0)
p= np.concatenate((p,p_stu_69), axis=0)
shp=(x_stu_70.shape)[0]
shape_y = y_stu_70.shape[0]
shape_p =  p_stu_70.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_stu_70 =  np.delete(x_stu_70, [k for k in range(nece_shp,shp)], None)
x_stu_70 = x_stu_70.reshape(-1,sensors)
shape_x_stu_70 = x_stu_70.shape[0]
x_stu_70 = preprocessing.normalize(x_stu_70, axis=0)
x_stu_70= np.reshape(x_stu_70, (-1,segement_time_size, sensors))
y_stu_70= np.delete( y_stu_70, [k for k in range(x_stu_70.shape[0],shape_y)], None)
p_stu_70 = np.delete(p_stu_70,[k for k in range(x_stu_70.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_stu_70), axis=0)
y= np.concatenate((y,y_stu_70), axis=0)
p= np.concatenate((p,p_stu_70), axis=0)
shp=(x_stu_71.shape)[0]
shape_y = y_stu_71.shape[0]
shape_p =  p_stu_71.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_stu_71 =  np.delete(x_stu_71, [k for k in range(nece_shp,shp)], None)
x_stu_71 = x_stu_71.reshape(-1,sensors)
shape_x_stu_71 = x_stu_71.shape[0]
x_stu_71 = preprocessing.normalize(x_stu_71, axis=0)
x_stu_71= np.reshape(x_stu_71, (-1,segement_time_size, sensors))
y_stu_71= np.delete( y_stu_71, [k for k in range(x_stu_71.shape[0],shape_y)], None)
p_stu_71 = np.delete(p_stu_71,[k for k in range(x_stu_71.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_stu_71), axis=0)
y= np.concatenate((y,y_stu_71), axis=0)
p= np.concatenate((p,p_stu_71), axis=0)
shp=(x_stu_72.shape)[0]
shape_y = y_stu_72.shape[0]
shape_p =  p_stu_72.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_stu_72 =  np.delete(x_stu_72, [k for k in range(nece_shp,shp)], None)
x_stu_72 = x_stu_72.reshape(-1,sensors)
shape_x_stu_72 = x_stu_72.shape[0]
x_stu_72 = preprocessing.normalize(x_stu_72, axis=0)
x_stu_72= np.reshape(x_stu_72, (-1,segement_time_size, sensors))
y_stu_72= np.delete( y_stu_72, [k for k in range(x_stu_72.shape[0],shape_y)], None)
p_stu_72 = np.delete(p_stu_72,[k for k in range(x_stu_72.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_stu_72), axis=0)
y= np.concatenate((y,y_stu_72), axis=0)
p= np.concatenate((p,p_stu_72), axis=0)
shp=(x_stu_73.shape)[0]
shape_y = y_stu_73.shape[0]
shape_p =  p_stu_73.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_stu_73 =  np.delete(x_stu_73, [k for k in range(nece_shp,shp)], None)
x_stu_73 = x_stu_73.reshape(-1,sensors)
shape_x_stu_73 = x_stu_73.shape[0]
x_stu_73 = preprocessing.normalize(x_stu_73, axis=0)
x_stu_73= np.reshape(x_stu_73, (-1,segement_time_size, sensors))
y_stu_73= np.delete( y_stu_73, [k for k in range(x_stu_73.shape[0],shape_y)], None)
p_stu_73 = np.delete(p_stu_73,[k for k in range(x_stu_73.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_stu_73), axis=0)
y= np.concatenate((y,y_stu_73), axis=0)
p= np.concatenate((p,p_stu_73), axis=0)
shp=(x_stu_74.shape)[0]
shape_y = y_stu_74.shape[0]
shape_p =  p_stu_74.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_stu_74 =  np.delete(x_stu_74, [k for k in range(nece_shp,shp)], None)
x_stu_74 = x_stu_74.reshape(-1,sensors)
shape_x_stu_74 = x_stu_74.shape[0]
x_stu_74 = preprocessing.normalize(x_stu_74, axis=0)
x_stu_74= np.reshape(x_stu_74, (-1,segement_time_size, sensors))
y_stu_74= np.delete( y_stu_74, [k for k in range(x_stu_74.shape[0],shape_y)], None)
p_stu_74 = np.delete(p_stu_74,[k for k in range(x_stu_74.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_stu_74), axis=0)
y= np.concatenate((y,y_stu_74), axis=0)
p= np.concatenate((p,p_stu_74), axis=0)
shp=(x_stu_75.shape)[0]
shape_y = y_stu_75.shape[0]
shape_p =  p_stu_75.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_stu_75 =  np.delete(x_stu_75, [k for k in range(nece_shp,shp)], None)
x_stu_75 = x_stu_75.reshape(-1,sensors)
shape_x_stu_75 = x_stu_75.shape[0]
x_stu_75 = preprocessing.normalize(x_stu_75, axis=0)
x_stu_75= np.reshape(x_stu_75, (-1,segement_time_size, sensors))
y_stu_75= np.delete( y_stu_75, [k for k in range(x_stu_75.shape[0],shape_y)], None)
p_stu_75 = np.delete(p_stu_75,[k for k in range(x_stu_75.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_stu_75), axis=0)
y= np.concatenate((y,y_stu_75), axis=0)
p= np.concatenate((p,p_stu_75), axis=0)
shp=(x_stu_76.shape)[0]
shape_y = y_stu_76.shape[0]
shape_p =  p_stu_76.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_stu_76 =  np.delete(x_stu_76, [k for k in range(nece_shp,shp)], None)
x_stu_76 = x_stu_76.reshape(-1,sensors)
shape_x_stu_76 = x_stu_76.shape[0]
x_stu_76 = preprocessing.normalize(x_stu_76, axis=0)
x_stu_76= np.reshape(x_stu_76, (-1,segement_time_size, sensors))
y_stu_76= np.delete( y_stu_76, [k for k in range(x_stu_76.shape[0],shape_y)], None)
p_stu_76 = np.delete(p_stu_76,[k for k in range(x_stu_76.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_stu_76), axis=0)
y= np.concatenate((y,y_stu_76), axis=0)
p= np.concatenate((p,p_stu_76), axis=0)
shp=(x_stu_77.shape)[0]
shape_y = y_stu_77.shape[0]
shape_p =  p_stu_77.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_stu_77 =  np.delete(x_stu_77, [k for k in range(nece_shp,shp)], None)
x_stu_77 = x_stu_77.reshape(-1,sensors)
shape_x_stu_77 = x_stu_77.shape[0]
x_stu_77 = preprocessing.normalize(x_stu_77, axis=0)
x_stu_77= np.reshape(x_stu_77, (-1,segement_time_size, sensors))
y_stu_77= np.delete( y_stu_77, [k for k in range(x_stu_77.shape[0],shape_y)], None)
p_stu_77 = np.delete(p_stu_77,[k for k in range(x_stu_77.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_stu_77), axis=0)
y= np.concatenate((y,y_stu_77), axis=0)
p= np.concatenate((p,p_stu_77), axis=0)
shp=(x_stu_78.shape)[0]
shape_y = y_stu_78.shape[0]
shape_p =  p_stu_78.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_stu_78 =  np.delete(x_stu_78, [k for k in range(nece_shp,shp)], None)
x_stu_78 = x_stu_78.reshape(-1,sensors)
shape_x_stu_78 = x_stu_78.shape[0]
x_stu_78 = preprocessing.normalize(x_stu_78, axis=0)
x_stu_78= np.reshape(x_stu_78, (-1,segement_time_size, sensors))
y_stu_78= np.delete( y_stu_78, [k for k in range(x_stu_78.shape[0],shape_y)], None)
p_stu_78 = np.delete(p_stu_78,[k for k in range(x_stu_78.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_stu_78), axis=0)
y= np.concatenate((y,y_stu_78), axis=0)
p= np.concatenate((p,p_stu_78), axis=0)
shp=(x_stu_79.shape)[0]
shape_y = y_stu_79.shape[0]
shape_p =  p_stu_79.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_stu_79 =  np.delete(x_stu_79, [k for k in range(nece_shp,shp)], None)
x_stu_79 = x_stu_79.reshape(-1,sensors)
shape_x_stu_79 = x_stu_79.shape[0]
x_stu_79 = preprocessing.normalize(x_stu_79, axis=0)
x_stu_79= np.reshape(x_stu_79, (-1,segement_time_size, sensors))
y_stu_79= np.delete( y_stu_79, [k for k in range(x_stu_79.shape[0],shape_y)], None)
p_stu_79 = np.delete(p_stu_79,[k for k in range(x_stu_79.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_stu_79), axis=0)
y= np.concatenate((y,y_stu_79), axis=0)
p= np.concatenate((p,p_stu_79), axis=0)
shp=(x_stu_80.shape)[0]
shape_y = y_stu_80.shape[0]
shape_p =  p_stu_80.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_stu_80 =  np.delete(x_stu_80, [k for k in range(nece_shp,shp)], None)
x_stu_80 = x_stu_80.reshape(-1,sensors)
shape_x_stu_80 = x_stu_80.shape[0]
x_stu_80 = preprocessing.normalize(x_stu_80, axis=0)
x_stu_80= np.reshape(x_stu_80, (-1,segement_time_size, sensors))
y_stu_80= np.delete( y_stu_80, [k for k in range(x_stu_80.shape[0],shape_y)], None)
p_stu_80 = np.delete(p_stu_80,[k for k in range(x_stu_80.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_stu_80), axis=0)
y= np.concatenate((y,y_stu_80), axis=0)
p= np.concatenate((p,p_stu_80), axis=0)
shp=(x_stu_81.shape)[0]
shape_y = y_stu_81.shape[0]
shape_p =  p_stu_81.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_stu_81 =  np.delete(x_stu_81, [k for k in range(nece_shp,shp)], None)
x_stu_81 = x_stu_81.reshape(-1,sensors)
shape_x_stu_81 = x_stu_81.shape[0]
x_stu_81 = preprocessing.normalize(x_stu_81, axis=0)
x_stu_81= np.reshape(x_stu_81, (-1,segement_time_size, sensors))
y_stu_81= np.delete( y_stu_81, [k for k in range(x_stu_81.shape[0],shape_y)], None)
p_stu_81 = np.delete(p_stu_81,[k for k in range(x_stu_81.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_stu_81), axis=0)
y= np.concatenate((y,y_stu_81), axis=0)
p= np.concatenate((p,p_stu_81), axis=0)
shp=(x_stu_82.shape)[0]
shape_y = y_stu_82.shape[0]
shape_p =  p_stu_82.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_stu_82 =  np.delete(x_stu_82, [k for k in range(nece_shp,shp)], None)
x_stu_82 = x_stu_82.reshape(-1,sensors)
shape_x_stu_82 = x_stu_82.shape[0]
x_stu_82 = preprocessing.normalize(x_stu_82, axis=0)
x_stu_82= np.reshape(x_stu_82, (-1,segement_time_size, sensors))
y_stu_82= np.delete( y_stu_82, [k for k in range(x_stu_82.shape[0],shape_y)], None)
p_stu_82 = np.delete(p_stu_82,[k for k in range(x_stu_82.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_stu_82), axis=0)
y= np.concatenate((y,y_stu_82), axis=0)
p= np.concatenate((p,p_stu_82), axis=0)
shp=(x_stu_83.shape)[0]
shape_y = y_stu_83.shape[0]
shape_p =  p_stu_83.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_stu_83 =  np.delete(x_stu_83, [k for k in range(nece_shp,shp)], None)
x_stu_83 = x_stu_83.reshape(-1,sensors)
shape_x_stu_83 = x_stu_83.shape[0]
x_stu_83 = preprocessing.normalize(x_stu_83, axis=0)
x_stu_83= np.reshape(x_stu_83, (-1,segement_time_size, sensors))
y_stu_83= np.delete( y_stu_83, [k for k in range(x_stu_83.shape[0],shape_y)], None)
p_stu_83 = np.delete(p_stu_83,[k for k in range(x_stu_83.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_stu_83), axis=0)
y= np.concatenate((y,y_stu_83), axis=0)
p= np.concatenate((p,p_stu_83), axis=0)
shp=(x_stu_84.shape)[0]
shape_y = y_stu_84.shape[0]
shape_p =  p_stu_84.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_stu_84 =  np.delete(x_stu_84, [k for k in range(nece_shp,shp)], None)
x_stu_84 = x_stu_84.reshape(-1,sensors)
shape_x_stu_84 = x_stu_84.shape[0]
x_stu_84 = preprocessing.normalize(x_stu_84, axis=0)
x_stu_84= np.reshape(x_stu_84, (-1,segement_time_size, sensors))
y_stu_84= np.delete( y_stu_84, [k for k in range(x_stu_84.shape[0],shape_y)], None)
p_stu_84 = np.delete(p_stu_84,[k for k in range(x_stu_84.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_stu_84), axis=0)
y= np.concatenate((y,y_stu_84), axis=0)
p= np.concatenate((p,p_stu_84), axis=0)
shp=(x_stu_85.shape)[0]
shape_y = y_stu_85.shape[0]
shape_p =  p_stu_85.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_stu_85 =  np.delete(x_stu_85, [k for k in range(nece_shp,shp)], None)
x_stu_85 = x_stu_85.reshape(-1,sensors)
shape_x_stu_85 = x_stu_85.shape[0]
x_stu_85 = preprocessing.normalize(x_stu_85, axis=0)
x_stu_85= np.reshape(x_stu_85, (-1,segement_time_size, sensors))
y_stu_85= np.delete( y_stu_85, [k for k in range(x_stu_85.shape[0],shape_y)], None)
p_stu_85 = np.delete(p_stu_85,[k for k in range(x_stu_85.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_stu_85), axis=0)
y= np.concatenate((y,y_stu_85), axis=0)
p= np.concatenate((p,p_stu_85), axis=0)
shp=(x_stu_86.shape)[0]
shape_y = y_stu_86.shape[0]
shape_p =  p_stu_86.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_stu_86 =  np.delete(x_stu_86, [k for k in range(nece_shp,shp)], None)
x_stu_86 = x_stu_86.reshape(-1,sensors)
shape_x_stu_86 = x_stu_86.shape[0]
x_stu_86 = preprocessing.normalize(x_stu_86, axis=0)
x_stu_86= np.reshape(x_stu_86, (-1,segement_time_size, sensors))
y_stu_86= np.delete( y_stu_86, [k for k in range(x_stu_86.shape[0],shape_y)], None)
p_stu_86 = np.delete(p_stu_86,[k for k in range(x_stu_86.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_stu_86), axis=0)
y= np.concatenate((y,y_stu_86), axis=0)
p= np.concatenate((p,p_stu_86), axis=0)
shp=(x_stu_87.shape)[0]
shape_y = y_stu_87.shape[0]
shape_p =  p_stu_87.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_stu_87 =  np.delete(x_stu_87, [k for k in range(nece_shp,shp)], None)
x_stu_87 = x_stu_87.reshape(-1,sensors)
shape_x_stu_87 = x_stu_87.shape[0]
x_stu_87 = preprocessing.normalize(x_stu_87, axis=0)
x_stu_87= np.reshape(x_stu_87, (-1,segement_time_size, sensors))
y_stu_87= np.delete( y_stu_87, [k for k in range(x_stu_87.shape[0],shape_y)], None)
p_stu_87 = np.delete(p_stu_87,[k for k in range(x_stu_87.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_stu_87), axis=0)
y= np.concatenate((y,y_stu_87), axis=0)
p= np.concatenate((p,p_stu_87), axis=0)
shp=(x_stu_88.shape)[0]
shape_y = y_stu_88.shape[0]
shape_p =  p_stu_88.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_stu_88 =  np.delete(x_stu_88, [k for k in range(nece_shp,shp)], None)
x_stu_88 = x_stu_88.reshape(-1,sensors)
shape_x_stu_88 = x_stu_88.shape[0]
x_stu_88 = preprocessing.normalize(x_stu_88, axis=0)
x_stu_88= np.reshape(x_stu_88, (-1,segement_time_size, sensors))
y_stu_88= np.delete( y_stu_88, [k for k in range(x_stu_88.shape[0],shape_y)], None)
p_stu_88 = np.delete(p_stu_88,[k for k in range(x_stu_88.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_stu_88), axis=0)
y= np.concatenate((y,y_stu_88), axis=0)
p= np.concatenate((p,p_stu_88), axis=0)
shp=(x_stu_89.shape)[0]
shape_y = y_stu_89.shape[0]
shape_p =  p_stu_89.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_stu_89 =  np.delete(x_stu_89, [k for k in range(nece_shp,shp)], None)
x_stu_89 = x_stu_89.reshape(-1,sensors)
shape_x_stu_89 = x_stu_89.shape[0]
x_stu_89 = preprocessing.normalize(x_stu_89, axis=0)
x_stu_89= np.reshape(x_stu_89, (-1,segement_time_size, sensors))
y_stu_89= np.delete( y_stu_89, [k for k in range(x_stu_89.shape[0],shape_y)], None)
p_stu_89 = np.delete(p_stu_89,[k for k in range(x_stu_89.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_stu_89), axis=0)
y= np.concatenate((y,y_stu_89), axis=0)
p= np.concatenate((p,p_stu_89), axis=0)
shp=(x_stu_90.shape)[0]
shape_y = y_stu_90.shape[0]
shape_p =  p_stu_90.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_stu_90 =  np.delete(x_stu_90, [k for k in range(nece_shp,shp)], None)
x_stu_90 = x_stu_90.reshape(-1,sensors)
shape_x_stu_90 = x_stu_90.shape[0]
x_stu_90 = preprocessing.normalize(x_stu_90, axis=0)
x_stu_90= np.reshape(x_stu_90, (-1,segement_time_size, sensors))
y_stu_90= np.delete( y_stu_90, [k for k in range(x_stu_90.shape[0],shape_y)], None)
p_stu_90 = np.delete(p_stu_90,[k for k in range(x_stu_90.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_stu_90), axis=0)
y= np.concatenate((y,y_stu_90), axis=0)
p= np.concatenate((p,p_stu_90), axis=0)
shp=(x_stu_91.shape)[0]
shape_y = y_stu_91.shape[0]
shape_p =  p_stu_91.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_stu_91 =  np.delete(x_stu_91, [k for k in range(nece_shp,shp)], None)
x_stu_91 = x_stu_91.reshape(-1,sensors)
shape_x_stu_91 = x_stu_91.shape[0]
x_stu_91 = preprocessing.normalize(x_stu_91, axis=0)
x_stu_91= np.reshape(x_stu_91, (-1,segement_time_size, sensors))
y_stu_91= np.delete( y_stu_91, [k for k in range(x_stu_91.shape[0],shape_y)], None)
p_stu_91 = np.delete(p_stu_91,[k for k in range(x_stu_91.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_stu_91), axis=0)
y= np.concatenate((y,y_stu_91), axis=0)
p= np.concatenate((p,p_stu_91), axis=0)
shp=(x_stu_92.shape)[0]
shape_y = y_stu_92.shape[0]
shape_p =  p_stu_92.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_stu_92 =  np.delete(x_stu_92, [k for k in range(nece_shp,shp)], None)
x_stu_92 = x_stu_92.reshape(-1,sensors)
shape_x_stu_92 = x_stu_92.shape[0]
x_stu_92 = preprocessing.normalize(x_stu_92, axis=0)
x_stu_92= np.reshape(x_stu_92, (-1,segement_time_size, sensors))
y_stu_92= np.delete( y_stu_92, [k for k in range(x_stu_92.shape[0],shape_y)], None)
p_stu_92 = np.delete(p_stu_92,[k for k in range(x_stu_92.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_stu_92), axis=0)
y= np.concatenate((y,y_stu_92), axis=0)
p= np.concatenate((p,p_stu_92), axis=0)
shp=(x_stu_93.shape)[0]
shape_y = y_stu_93.shape[0]
shape_p =  p_stu_93.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_stu_93 =  np.delete(x_stu_93, [k for k in range(nece_shp,shp)], None)
x_stu_93 = x_stu_93.reshape(-1,sensors)
shape_x_stu_93 = x_stu_93.shape[0]
x_stu_93 = preprocessing.normalize(x_stu_93, axis=0)
x_stu_93= np.reshape(x_stu_93, (-1,segement_time_size, sensors))
y_stu_93= np.delete( y_stu_93, [k for k in range(x_stu_93.shape[0],shape_y)], None)
p_stu_93 = np.delete(p_stu_93,[k for k in range(x_stu_93.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_stu_93), axis=0)
y= np.concatenate((y,y_stu_93), axis=0)
p= np.concatenate((p,p_stu_93), axis=0)
shp=(x_stu_94.shape)[0]
shape_y = y_stu_94.shape[0]
shape_p =  p_stu_94.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_stu_94 =  np.delete(x_stu_94, [k for k in range(nece_shp,shp)], None)
x_stu_94 = x_stu_94.reshape(-1,sensors)
shape_x_stu_94 = x_stu_94.shape[0]
x_stu_94 = preprocessing.normalize(x_stu_94, axis=0)
x_stu_94= np.reshape(x_stu_94, (-1,segement_time_size, sensors))
y_stu_94= np.delete( y_stu_94, [k for k in range(x_stu_94.shape[0],shape_y)], None)
p_stu_94 = np.delete(p_stu_94,[k for k in range(x_stu_94.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_stu_94), axis=0)
y= np.concatenate((y,y_stu_94), axis=0)
p= np.concatenate((p,p_stu_94), axis=0)
shp=(x_stu_95.shape)[0]
shape_y = y_stu_95.shape[0]
shape_p =  p_stu_95.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_stu_95 =  np.delete(x_stu_95, [k for k in range(nece_shp,shp)], None)
x_stu_95 = x_stu_95.reshape(-1,sensors)
shape_x_stu_95 = x_stu_95.shape[0]
x_stu_95 = preprocessing.normalize(x_stu_95, axis=0)
x_stu_95= np.reshape(x_stu_95, (-1,segement_time_size, sensors))
y_stu_95= np.delete( y_stu_95, [k for k in range(x_stu_95.shape[0],shape_y)], None)
p_stu_95 = np.delete(p_stu_95,[k for k in range(x_stu_95.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_stu_95), axis=0)
y= np.concatenate((y,y_stu_95), axis=0)
p= np.concatenate((p,p_stu_95), axis=0)
shp=(x_stu_96.shape)[0]
shape_y = y_stu_96.shape[0]
shape_p =  p_stu_96.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_stu_96 =  np.delete(x_stu_96, [k for k in range(nece_shp,shp)], None)
x_stu_96 = x_stu_96.reshape(-1,sensors)
shape_x_stu_96 = x_stu_96.shape[0]
x_stu_96 = preprocessing.normalize(x_stu_96, axis=0)
x_stu_96= np.reshape(x_stu_96, (-1,segement_time_size, sensors))
y_stu_96= np.delete( y_stu_96, [k for k in range(x_stu_96.shape[0],shape_y)], None)
p_stu_96 = np.delete(p_stu_96,[k for k in range(x_stu_96.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_stu_96), axis=0)
y= np.concatenate((y,y_stu_96), axis=0)
p= np.concatenate((p,p_stu_96), axis=0)
shp=(x_stu_97.shape)[0]
shape_y = y_stu_97.shape[0]
shape_p =  p_stu_97.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_stu_97 =  np.delete(x_stu_97, [k for k in range(nece_shp,shp)], None)
x_stu_97 = x_stu_97.reshape(-1,sensors)
shape_x_stu_97 = x_stu_97.shape[0]
x_stu_97 = preprocessing.normalize(x_stu_97, axis=0)
x_stu_97= np.reshape(x_stu_97, (-1,segement_time_size, sensors))
y_stu_97= np.delete( y_stu_97, [k for k in range(x_stu_97.shape[0],shape_y)], None)
p_stu_97 = np.delete(p_stu_97,[k for k in range(x_stu_97.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_stu_97), axis=0)
y= np.concatenate((y,y_stu_97), axis=0)
p= np.concatenate((p,p_stu_97), axis=0)
shp=(x_stu_98.shape)[0]
shape_y = y_stu_98.shape[0]
shape_p =  p_stu_98.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_stu_98 =  np.delete(x_stu_98, [k for k in range(nece_shp,shp)], None)
x_stu_98 = x_stu_98.reshape(-1,sensors)
shape_x_stu_98 = x_stu_98.shape[0]
x_stu_98 = preprocessing.normalize(x_stu_98, axis=0)
x_stu_98= np.reshape(x_stu_98, (-1,segement_time_size, sensors))
y_stu_98= np.delete( y_stu_98, [k for k in range(x_stu_98.shape[0],shape_y)], None)
p_stu_98 = np.delete(p_stu_98,[k for k in range(x_stu_98.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_stu_98), axis=0)
y= np.concatenate((y,y_stu_98), axis=0)
p= np.concatenate((p,p_stu_98), axis=0)
shp=(x_stu_99.shape)[0]
shape_y = y_stu_99.shape[0]
shape_p =  p_stu_99.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_stu_99 =  np.delete(x_stu_99, [k for k in range(nece_shp,shp)], None)
x_stu_99 = x_stu_99.reshape(-1,sensors)
shape_x_stu_99 = x_stu_99.shape[0]
x_stu_99 = preprocessing.normalize(x_stu_99, axis=0)
x_stu_99= np.reshape(x_stu_99, (-1,segement_time_size, sensors))
y_stu_99= np.delete( y_stu_99, [k for k in range(x_stu_99.shape[0],shape_y)], None)
p_stu_99 = np.delete(p_stu_99,[k for k in range(x_stu_99.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_stu_99), axis=0)
y= np.concatenate((y,y_stu_99), axis=0)
p= np.concatenate((p,p_stu_99), axis=0)
shp=(x_std_0.shape)[0]
shape_y = y_std_0.shape[0]
shape_p =  p_std_0.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_std_0 =  np.delete(x_std_0, [k for k in range(nece_shp,shp)], None)
x_std_0 = x_std_0.reshape(-1,sensors)
shape_x_std_0 = x_std_0.shape[0]
x_std_0 = preprocessing.normalize(x_std_0, axis=0)
x_std_0= np.reshape(x_std_0, (-1,segement_time_size, sensors))
y_std_0= np.delete( y_std_0, [k for k in range(x_std_0.shape[0],shape_y)], None)
p_std_0 = np.delete(p_std_0,[k for k in range(x_std_0.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_std_0), axis=0)
y= np.concatenate((y,y_std_0), axis=0)
p= np.concatenate((p,p_std_0), axis=0)
shp=(x_std_1.shape)[0]
shape_y = y_std_1.shape[0]
shape_p =  p_std_1.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_std_1 =  np.delete(x_std_1, [k for k in range(nece_shp,shp)], None)
x_std_1 = x_std_1.reshape(-1,sensors)
shape_x_std_1 = x_std_1.shape[0]
x_std_1 = preprocessing.normalize(x_std_1, axis=0)
x_std_1= np.reshape(x_std_1, (-1,segement_time_size, sensors))
y_std_1= np.delete( y_std_1, [k for k in range(x_std_1.shape[0],shape_y)], None)
p_std_1 = np.delete(p_std_1,[k for k in range(x_std_1.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_std_1), axis=0)
y= np.concatenate((y,y_std_1), axis=0)
p= np.concatenate((p,p_std_1), axis=0)
shp=(x_std_2.shape)[0]
shape_y = y_std_2.shape[0]
shape_p =  p_std_2.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_std_2 =  np.delete(x_std_2, [k for k in range(nece_shp,shp)], None)
x_std_2 = x_std_2.reshape(-1,sensors)
shape_x_std_2 = x_std_2.shape[0]
x_std_2 = preprocessing.normalize(x_std_2, axis=0)
x_std_2= np.reshape(x_std_2, (-1,segement_time_size, sensors))
y_std_2= np.delete( y_std_2, [k for k in range(x_std_2.shape[0],shape_y)], None)
p_std_2 = np.delete(p_std_2,[k for k in range(x_std_2.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_std_2), axis=0)
y= np.concatenate((y,y_std_2), axis=0)
p= np.concatenate((p,p_std_2), axis=0)
shp=(x_std_3.shape)[0]
shape_y = y_std_3.shape[0]
shape_p =  p_std_3.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_std_3 =  np.delete(x_std_3, [k for k in range(nece_shp,shp)], None)
x_std_3 = x_std_3.reshape(-1,sensors)
shape_x_std_3 = x_std_3.shape[0]
x_std_3 = preprocessing.normalize(x_std_3, axis=0)
x_std_3= np.reshape(x_std_3, (-1,segement_time_size, sensors))
y_std_3= np.delete( y_std_3, [k for k in range(x_std_3.shape[0],shape_y)], None)
p_std_3 = np.delete(p_std_3,[k for k in range(x_std_3.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_std_3), axis=0)
y= np.concatenate((y,y_std_3), axis=0)
p= np.concatenate((p,p_std_3), axis=0)
shp=(x_std_4.shape)[0]
shape_y = y_std_4.shape[0]
shape_p =  p_std_4.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_std_4 =  np.delete(x_std_4, [k for k in range(nece_shp,shp)], None)
x_std_4 = x_std_4.reshape(-1,sensors)
shape_x_std_4 = x_std_4.shape[0]
x_std_4 = preprocessing.normalize(x_std_4, axis=0)
x_std_4= np.reshape(x_std_4, (-1,segement_time_size, sensors))
y_std_4= np.delete( y_std_4, [k for k in range(x_std_4.shape[0],shape_y)], None)
p_std_4 = np.delete(p_std_4,[k for k in range(x_std_4.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_std_4), axis=0)
y= np.concatenate((y,y_std_4), axis=0)
p= np.concatenate((p,p_std_4), axis=0)
shp=(x_std_5.shape)[0]
shape_y = y_std_5.shape[0]
shape_p =  p_std_5.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_std_5 =  np.delete(x_std_5, [k for k in range(nece_shp,shp)], None)
x_std_5 = x_std_5.reshape(-1,sensors)
shape_x_std_5 = x_std_5.shape[0]
x_std_5 = preprocessing.normalize(x_std_5, axis=0)
x_std_5= np.reshape(x_std_5, (-1,segement_time_size, sensors))
y_std_5= np.delete( y_std_5, [k for k in range(x_std_5.shape[0],shape_y)], None)
p_std_5 = np.delete(p_std_5,[k for k in range(x_std_5.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_std_5), axis=0)
y= np.concatenate((y,y_std_5), axis=0)
p= np.concatenate((p,p_std_5), axis=0)
shp=(x_std_6.shape)[0]
shape_y = y_std_6.shape[0]
shape_p =  p_std_6.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_std_6 =  np.delete(x_std_6, [k for k in range(nece_shp,shp)], None)
x_std_6 = x_std_6.reshape(-1,sensors)
shape_x_std_6 = x_std_6.shape[0]
x_std_6 = preprocessing.normalize(x_std_6, axis=0)
x_std_6= np.reshape(x_std_6, (-1,segement_time_size, sensors))
y_std_6= np.delete( y_std_6, [k for k in range(x_std_6.shape[0],shape_y)], None)
p_std_6 = np.delete(p_std_6,[k for k in range(x_std_6.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_std_6), axis=0)
y= np.concatenate((y,y_std_6), axis=0)
p= np.concatenate((p,p_std_6), axis=0)
shp=(x_std_7.shape)[0]
shape_y = y_std_7.shape[0]
shape_p =  p_std_7.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_std_7 =  np.delete(x_std_7, [k for k in range(nece_shp,shp)], None)
x_std_7 = x_std_7.reshape(-1,sensors)
shape_x_std_7 = x_std_7.shape[0]
x_std_7 = preprocessing.normalize(x_std_7, axis=0)
x_std_7= np.reshape(x_std_7, (-1,segement_time_size, sensors))
y_std_7= np.delete( y_std_7, [k for k in range(x_std_7.shape[0],shape_y)], None)
p_std_7 = np.delete(p_std_7,[k for k in range(x_std_7.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_std_7), axis=0)
y= np.concatenate((y,y_std_7), axis=0)
p= np.concatenate((p,p_std_7), axis=0)
shp=(x_std_8.shape)[0]
shape_y = y_std_8.shape[0]
shape_p =  p_std_8.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_std_8 =  np.delete(x_std_8, [k for k in range(nece_shp,shp)], None)
x_std_8 = x_std_8.reshape(-1,sensors)
shape_x_std_8 = x_std_8.shape[0]
x_std_8 = preprocessing.normalize(x_std_8, axis=0)
x_std_8= np.reshape(x_std_8, (-1,segement_time_size, sensors))
y_std_8= np.delete( y_std_8, [k for k in range(x_std_8.shape[0],shape_y)], None)
p_std_8 = np.delete(p_std_8,[k for k in range(x_std_8.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_std_8), axis=0)
y= np.concatenate((y,y_std_8), axis=0)
p= np.concatenate((p,p_std_8), axis=0)
shp=(x_std_9.shape)[0]
shape_y = y_std_9.shape[0]
shape_p =  p_std_9.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_std_9 =  np.delete(x_std_9, [k for k in range(nece_shp,shp)], None)
x_std_9 = x_std_9.reshape(-1,sensors)
shape_x_std_9 = x_std_9.shape[0]
x_std_9 = preprocessing.normalize(x_std_9, axis=0)
x_std_9= np.reshape(x_std_9, (-1,segement_time_size, sensors))
y_std_9= np.delete( y_std_9, [k for k in range(x_std_9.shape[0],shape_y)], None)
p_std_9 = np.delete(p_std_9,[k for k in range(x_std_9.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_std_9), axis=0)
y= np.concatenate((y,y_std_9), axis=0)
p= np.concatenate((p,p_std_9), axis=0)
shp=(x_std_10.shape)[0]
shape_y = y_std_10.shape[0]
shape_p =  p_std_10.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_std_10 =  np.delete(x_std_10, [k for k in range(nece_shp,shp)], None)
x_std_10 = x_std_10.reshape(-1,sensors)
shape_x_std_10 = x_std_10.shape[0]
x_std_10 = preprocessing.normalize(x_std_10, axis=0)
x_std_10= np.reshape(x_std_10, (-1,segement_time_size, sensors))
y_std_10= np.delete( y_std_10, [k for k in range(x_std_10.shape[0],shape_y)], None)
p_std_10 = np.delete(p_std_10,[k for k in range(x_std_10.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_std_10), axis=0)
y= np.concatenate((y,y_std_10), axis=0)
p= np.concatenate((p,p_std_10), axis=0)
shp=(x_std_11.shape)[0]
shape_y = y_std_11.shape[0]
shape_p =  p_std_11.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_std_11 =  np.delete(x_std_11, [k for k in range(nece_shp,shp)], None)
x_std_11 = x_std_11.reshape(-1,sensors)
shape_x_std_11 = x_std_11.shape[0]
x_std_11 = preprocessing.normalize(x_std_11, axis=0)
x_std_11= np.reshape(x_std_11, (-1,segement_time_size, sensors))
y_std_11= np.delete( y_std_11, [k for k in range(x_std_11.shape[0],shape_y)], None)
p_std_11 = np.delete(p_std_11,[k for k in range(x_std_11.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_std_11), axis=0)
y= np.concatenate((y,y_std_11), axis=0)
p= np.concatenate((p,p_std_11), axis=0)
shp=(x_std_12.shape)[0]
shape_y = y_std_12.shape[0]
shape_p =  p_std_12.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_std_12 =  np.delete(x_std_12, [k for k in range(nece_shp,shp)], None)
x_std_12 = x_std_12.reshape(-1,sensors)
shape_x_std_12 = x_std_12.shape[0]
x_std_12 = preprocessing.normalize(x_std_12, axis=0)
x_std_12= np.reshape(x_std_12, (-1,segement_time_size, sensors))
y_std_12= np.delete( y_std_12, [k for k in range(x_std_12.shape[0],shape_y)], None)
p_std_12 = np.delete(p_std_12,[k for k in range(x_std_12.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_std_12), axis=0)
y= np.concatenate((y,y_std_12), axis=0)
p= np.concatenate((p,p_std_12), axis=0)
shp=(x_std_13.shape)[0]
shape_y = y_std_13.shape[0]
shape_p =  p_std_13.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_std_13 =  np.delete(x_std_13, [k for k in range(nece_shp,shp)], None)
x_std_13 = x_std_13.reshape(-1,sensors)
shape_x_std_13 = x_std_13.shape[0]
x_std_13 = preprocessing.normalize(x_std_13, axis=0)
x_std_13= np.reshape(x_std_13, (-1,segement_time_size, sensors))
y_std_13= np.delete( y_std_13, [k for k in range(x_std_13.shape[0],shape_y)], None)
p_std_13 = np.delete(p_std_13,[k for k in range(x_std_13.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_std_13), axis=0)
y= np.concatenate((y,y_std_13), axis=0)
p= np.concatenate((p,p_std_13), axis=0)
shp=(x_std_14.shape)[0]
shape_y = y_std_14.shape[0]
shape_p =  p_std_14.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_std_14 =  np.delete(x_std_14, [k for k in range(nece_shp,shp)], None)
x_std_14 = x_std_14.reshape(-1,sensors)
shape_x_std_14 = x_std_14.shape[0]
x_std_14 = preprocessing.normalize(x_std_14, axis=0)
x_std_14= np.reshape(x_std_14, (-1,segement_time_size, sensors))
y_std_14= np.delete( y_std_14, [k for k in range(x_std_14.shape[0],shape_y)], None)
p_std_14 = np.delete(p_std_14,[k for k in range(x_std_14.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_std_14), axis=0)
y= np.concatenate((y,y_std_14), axis=0)
p= np.concatenate((p,p_std_14), axis=0)
shp=(x_std_15.shape)[0]
shape_y = y_std_15.shape[0]
shape_p =  p_std_15.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_std_15 =  np.delete(x_std_15, [k for k in range(nece_shp,shp)], None)
x_std_15 = x_std_15.reshape(-1,sensors)
shape_x_std_15 = x_std_15.shape[0]
x_std_15 = preprocessing.normalize(x_std_15, axis=0)
x_std_15= np.reshape(x_std_15, (-1,segement_time_size, sensors))
y_std_15= np.delete( y_std_15, [k for k in range(x_std_15.shape[0],shape_y)], None)
p_std_15 = np.delete(p_std_15,[k for k in range(x_std_15.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_std_15), axis=0)
y= np.concatenate((y,y_std_15), axis=0)
p= np.concatenate((p,p_std_15), axis=0)
shp=(x_std_16.shape)[0]
shape_y = y_std_16.shape[0]
shape_p =  p_std_16.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_std_16 =  np.delete(x_std_16, [k for k in range(nece_shp,shp)], None)
x_std_16 = x_std_16.reshape(-1,sensors)
shape_x_std_16 = x_std_16.shape[0]
x_std_16 = preprocessing.normalize(x_std_16, axis=0)
x_std_16= np.reshape(x_std_16, (-1,segement_time_size, sensors))
y_std_16= np.delete( y_std_16, [k for k in range(x_std_16.shape[0],shape_y)], None)
p_std_16 = np.delete(p_std_16,[k for k in range(x_std_16.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_std_16), axis=0)
y= np.concatenate((y,y_std_16), axis=0)
p= np.concatenate((p,p_std_16), axis=0)
shp=(x_std_17.shape)[0]
shape_y = y_std_17.shape[0]
shape_p =  p_std_17.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_std_17 =  np.delete(x_std_17, [k for k in range(nece_shp,shp)], None)
x_std_17 = x_std_17.reshape(-1,sensors)
shape_x_std_17 = x_std_17.shape[0]
x_std_17 = preprocessing.normalize(x_std_17, axis=0)
x_std_17= np.reshape(x_std_17, (-1,segement_time_size, sensors))
y_std_17= np.delete( y_std_17, [k for k in range(x_std_17.shape[0],shape_y)], None)
p_std_17 = np.delete(p_std_17,[k for k in range(x_std_17.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_std_17), axis=0)
y= np.concatenate((y,y_std_17), axis=0)
p= np.concatenate((p,p_std_17), axis=0)
shp=(x_std_18.shape)[0]
shape_y = y_std_18.shape[0]
shape_p =  p_std_18.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_std_18 =  np.delete(x_std_18, [k for k in range(nece_shp,shp)], None)
x_std_18 = x_std_18.reshape(-1,sensors)
shape_x_std_18 = x_std_18.shape[0]
x_std_18 = preprocessing.normalize(x_std_18, axis=0)
x_std_18= np.reshape(x_std_18, (-1,segement_time_size, sensors))
y_std_18= np.delete( y_std_18, [k for k in range(x_std_18.shape[0],shape_y)], None)
p_std_18 = np.delete(p_std_18,[k for k in range(x_std_18.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_std_18), axis=0)
y= np.concatenate((y,y_std_18), axis=0)
p= np.concatenate((p,p_std_18), axis=0)
shp=(x_std_19.shape)[0]
shape_y = y_std_19.shape[0]
shape_p =  p_std_19.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_std_19 =  np.delete(x_std_19, [k for k in range(nece_shp,shp)], None)
x_std_19 = x_std_19.reshape(-1,sensors)
shape_x_std_19 = x_std_19.shape[0]
x_std_19 = preprocessing.normalize(x_std_19, axis=0)
x_std_19= np.reshape(x_std_19, (-1,segement_time_size, sensors))
y_std_19= np.delete( y_std_19, [k for k in range(x_std_19.shape[0],shape_y)], None)
p_std_19 = np.delete(p_std_19,[k for k in range(x_std_19.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_std_19), axis=0)
y= np.concatenate((y,y_std_19), axis=0)
p= np.concatenate((p,p_std_19), axis=0)
shp=(x_std_20.shape)[0]
shape_y = y_std_20.shape[0]
shape_p =  p_std_20.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_std_20 =  np.delete(x_std_20, [k for k in range(nece_shp,shp)], None)
x_std_20 = x_std_20.reshape(-1,sensors)
shape_x_std_20 = x_std_20.shape[0]
x_std_20 = preprocessing.normalize(x_std_20, axis=0)
x_std_20= np.reshape(x_std_20, (-1,segement_time_size, sensors))
y_std_20= np.delete( y_std_20, [k for k in range(x_std_20.shape[0],shape_y)], None)
p_std_20 = np.delete(p_std_20,[k for k in range(x_std_20.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_std_20), axis=0)
y= np.concatenate((y,y_std_20), axis=0)
p= np.concatenate((p,p_std_20), axis=0)
shp=(x_std_21.shape)[0]
shape_y = y_std_21.shape[0]
shape_p =  p_std_21.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_std_21 =  np.delete(x_std_21, [k for k in range(nece_shp,shp)], None)
x_std_21 = x_std_21.reshape(-1,sensors)
shape_x_std_21 = x_std_21.shape[0]
x_std_21 = preprocessing.normalize(x_std_21, axis=0)
x_std_21= np.reshape(x_std_21, (-1,segement_time_size, sensors))
y_std_21= np.delete( y_std_21, [k for k in range(x_std_21.shape[0],shape_y)], None)
p_std_21 = np.delete(p_std_21,[k for k in range(x_std_21.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_std_21), axis=0)
y= np.concatenate((y,y_std_21), axis=0)
p= np.concatenate((p,p_std_21), axis=0)
shp=(x_std_22.shape)[0]
shape_y = y_std_22.shape[0]
shape_p =  p_std_22.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_std_22 =  np.delete(x_std_22, [k for k in range(nece_shp,shp)], None)
x_std_22 = x_std_22.reshape(-1,sensors)
shape_x_std_22 = x_std_22.shape[0]
x_std_22 = preprocessing.normalize(x_std_22, axis=0)
x_std_22= np.reshape(x_std_22, (-1,segement_time_size, sensors))
y_std_22= np.delete( y_std_22, [k for k in range(x_std_22.shape[0],shape_y)], None)
p_std_22 = np.delete(p_std_22,[k for k in range(x_std_22.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_std_22), axis=0)
y= np.concatenate((y,y_std_22), axis=0)
p= np.concatenate((p,p_std_22), axis=0)
shp=(x_std_23.shape)[0]
shape_y = y_std_23.shape[0]
shape_p =  p_std_23.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_std_23 =  np.delete(x_std_23, [k for k in range(nece_shp,shp)], None)
x_std_23 = x_std_23.reshape(-1,sensors)
shape_x_std_23 = x_std_23.shape[0]
x_std_23 = preprocessing.normalize(x_std_23, axis=0)
x_std_23= np.reshape(x_std_23, (-1,segement_time_size, sensors))
y_std_23= np.delete( y_std_23, [k for k in range(x_std_23.shape[0],shape_y)], None)
p_std_23 = np.delete(p_std_23,[k for k in range(x_std_23.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_std_23), axis=0)
y= np.concatenate((y,y_std_23), axis=0)
p= np.concatenate((p,p_std_23), axis=0)
shp=(x_std_24.shape)[0]
shape_y = y_std_24.shape[0]
shape_p =  p_std_24.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_std_24 =  np.delete(x_std_24, [k for k in range(nece_shp,shp)], None)
x_std_24 = x_std_24.reshape(-1,sensors)
shape_x_std_24 = x_std_24.shape[0]
x_std_24 = preprocessing.normalize(x_std_24, axis=0)
x_std_24= np.reshape(x_std_24, (-1,segement_time_size, sensors))
y_std_24= np.delete( y_std_24, [k for k in range(x_std_24.shape[0],shape_y)], None)
p_std_24 = np.delete(p_std_24,[k for k in range(x_std_24.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_std_24), axis=0)
y= np.concatenate((y,y_std_24), axis=0)
p= np.concatenate((p,p_std_24), axis=0)
shp=(x_std_25.shape)[0]
shape_y = y_std_25.shape[0]
shape_p =  p_std_25.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_std_25 =  np.delete(x_std_25, [k for k in range(nece_shp,shp)], None)
x_std_25 = x_std_25.reshape(-1,sensors)
shape_x_std_25 = x_std_25.shape[0]
x_std_25 = preprocessing.normalize(x_std_25, axis=0)
x_std_25= np.reshape(x_std_25, (-1,segement_time_size, sensors))
y_std_25= np.delete( y_std_25, [k for k in range(x_std_25.shape[0],shape_y)], None)
p_std_25 = np.delete(p_std_25,[k for k in range(x_std_25.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_std_25), axis=0)
y= np.concatenate((y,y_std_25), axis=0)
p= np.concatenate((p,p_std_25), axis=0)
shp=(x_std_26.shape)[0]
shape_y = y_std_26.shape[0]
shape_p =  p_std_26.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_std_26 =  np.delete(x_std_26, [k for k in range(nece_shp,shp)], None)
x_std_26 = x_std_26.reshape(-1,sensors)
shape_x_std_26 = x_std_26.shape[0]
x_std_26 = preprocessing.normalize(x_std_26, axis=0)
x_std_26= np.reshape(x_std_26, (-1,segement_time_size, sensors))
y_std_26= np.delete( y_std_26, [k for k in range(x_std_26.shape[0],shape_y)], None)
p_std_26 = np.delete(p_std_26,[k for k in range(x_std_26.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_std_26), axis=0)
y= np.concatenate((y,y_std_26), axis=0)
p= np.concatenate((p,p_std_26), axis=0)
shp=(x_std_27.shape)[0]
shape_y = y_std_27.shape[0]
shape_p =  p_std_27.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_std_27 =  np.delete(x_std_27, [k for k in range(nece_shp,shp)], None)
x_std_27 = x_std_27.reshape(-1,sensors)
shape_x_std_27 = x_std_27.shape[0]
x_std_27 = preprocessing.normalize(x_std_27, axis=0)
x_std_27= np.reshape(x_std_27, (-1,segement_time_size, sensors))
y_std_27= np.delete( y_std_27, [k for k in range(x_std_27.shape[0],shape_y)], None)
p_std_27 = np.delete(p_std_27,[k for k in range(x_std_27.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_std_27), axis=0)
y= np.concatenate((y,y_std_27), axis=0)
p= np.concatenate((p,p_std_27), axis=0)
shp=(x_std_28.shape)[0]
shape_y = y_std_28.shape[0]
shape_p =  p_std_28.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_std_28 =  np.delete(x_std_28, [k for k in range(nece_shp,shp)], None)
x_std_28 = x_std_28.reshape(-1,sensors)
shape_x_std_28 = x_std_28.shape[0]
x_std_28 = preprocessing.normalize(x_std_28, axis=0)
x_std_28= np.reshape(x_std_28, (-1,segement_time_size, sensors))
y_std_28= np.delete( y_std_28, [k for k in range(x_std_28.shape[0],shape_y)], None)
p_std_28 = np.delete(p_std_28,[k for k in range(x_std_28.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_std_28), axis=0)
y= np.concatenate((y,y_std_28), axis=0)
p= np.concatenate((p,p_std_28), axis=0)
shp=(x_std_29.shape)[0]
shape_y = y_std_29.shape[0]
shape_p =  p_std_29.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_std_29 =  np.delete(x_std_29, [k for k in range(nece_shp,shp)], None)
x_std_29 = x_std_29.reshape(-1,sensors)
shape_x_std_29 = x_std_29.shape[0]
x_std_29 = preprocessing.normalize(x_std_29, axis=0)
x_std_29= np.reshape(x_std_29, (-1,segement_time_size, sensors))
y_std_29= np.delete( y_std_29, [k for k in range(x_std_29.shape[0],shape_y)], None)
p_std_29 = np.delete(p_std_29,[k for k in range(x_std_29.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_std_29), axis=0)
y= np.concatenate((y,y_std_29), axis=0)
p= np.concatenate((p,p_std_29), axis=0)
shp=(x_std_30.shape)[0]
shape_y = y_std_30.shape[0]
shape_p =  p_std_30.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_std_30 =  np.delete(x_std_30, [k for k in range(nece_shp,shp)], None)
x_std_30 = x_std_30.reshape(-1,sensors)
shape_x_std_30 = x_std_30.shape[0]
x_std_30 = preprocessing.normalize(x_std_30, axis=0)
x_std_30= np.reshape(x_std_30, (-1,segement_time_size, sensors))
y_std_30= np.delete( y_std_30, [k for k in range(x_std_30.shape[0],shape_y)], None)
p_std_30 = np.delete(p_std_30,[k for k in range(x_std_30.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_std_30), axis=0)
y= np.concatenate((y,y_std_30), axis=0)
p= np.concatenate((p,p_std_30), axis=0)
shp=(x_std_31.shape)[0]
shape_y = y_std_31.shape[0]
shape_p =  p_std_31.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_std_31 =  np.delete(x_std_31, [k for k in range(nece_shp,shp)], None)
x_std_31 = x_std_31.reshape(-1,sensors)
shape_x_std_31 = x_std_31.shape[0]
x_std_31 = preprocessing.normalize(x_std_31, axis=0)
x_std_31= np.reshape(x_std_31, (-1,segement_time_size, sensors))
y_std_31= np.delete( y_std_31, [k for k in range(x_std_31.shape[0],shape_y)], None)
p_std_31 = np.delete(p_std_31,[k for k in range(x_std_31.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_std_31), axis=0)
y= np.concatenate((y,y_std_31), axis=0)
p= np.concatenate((p,p_std_31), axis=0)
shp=(x_std_32.shape)[0]
shape_y = y_std_32.shape[0]
shape_p =  p_std_32.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_std_32 =  np.delete(x_std_32, [k for k in range(nece_shp,shp)], None)
x_std_32 = x_std_32.reshape(-1,sensors)
shape_x_std_32 = x_std_32.shape[0]
x_std_32 = preprocessing.normalize(x_std_32, axis=0)
x_std_32= np.reshape(x_std_32, (-1,segement_time_size, sensors))
y_std_32= np.delete( y_std_32, [k for k in range(x_std_32.shape[0],shape_y)], None)
p_std_32 = np.delete(p_std_32,[k for k in range(x_std_32.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_std_32), axis=0)
y= np.concatenate((y,y_std_32), axis=0)
p= np.concatenate((p,p_std_32), axis=0)
shp=(x_std_33.shape)[0]
shape_y = y_std_33.shape[0]
shape_p =  p_std_33.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_std_33 =  np.delete(x_std_33, [k for k in range(nece_shp,shp)], None)
x_std_33 = x_std_33.reshape(-1,sensors)
shape_x_std_33 = x_std_33.shape[0]
x_std_33 = preprocessing.normalize(x_std_33, axis=0)
x_std_33= np.reshape(x_std_33, (-1,segement_time_size, sensors))
y_std_33= np.delete( y_std_33, [k for k in range(x_std_33.shape[0],shape_y)], None)
p_std_33 = np.delete(p_std_33,[k for k in range(x_std_33.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_std_33), axis=0)
y= np.concatenate((y,y_std_33), axis=0)
p= np.concatenate((p,p_std_33), axis=0)
shp=(x_std_34.shape)[0]
shape_y = y_std_34.shape[0]
shape_p =  p_std_34.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_std_34 =  np.delete(x_std_34, [k for k in range(nece_shp,shp)], None)
x_std_34 = x_std_34.reshape(-1,sensors)
shape_x_std_34 = x_std_34.shape[0]
x_std_34 = preprocessing.normalize(x_std_34, axis=0)
x_std_34= np.reshape(x_std_34, (-1,segement_time_size, sensors))
y_std_34= np.delete( y_std_34, [k for k in range(x_std_34.shape[0],shape_y)], None)
p_std_34 = np.delete(p_std_34,[k for k in range(x_std_34.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_std_34), axis=0)
y= np.concatenate((y,y_std_34), axis=0)
p= np.concatenate((p,p_std_34), axis=0)
shp=(x_std_35.shape)[0]
shape_y = y_std_35.shape[0]
shape_p =  p_std_35.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_std_35 =  np.delete(x_std_35, [k for k in range(nece_shp,shp)], None)
x_std_35 = x_std_35.reshape(-1,sensors)
shape_x_std_35 = x_std_35.shape[0]
x_std_35 = preprocessing.normalize(x_std_35, axis=0)
x_std_35= np.reshape(x_std_35, (-1,segement_time_size, sensors))
y_std_35= np.delete( y_std_35, [k for k in range(x_std_35.shape[0],shape_y)], None)
p_std_35 = np.delete(p_std_35,[k for k in range(x_std_35.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_std_35), axis=0)
y= np.concatenate((y,y_std_35), axis=0)
p= np.concatenate((p,p_std_35), axis=0)
shp=(x_std_36.shape)[0]
shape_y = y_std_36.shape[0]
shape_p =  p_std_36.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_std_36 =  np.delete(x_std_36, [k for k in range(nece_shp,shp)], None)
x_std_36 = x_std_36.reshape(-1,sensors)
shape_x_std_36 = x_std_36.shape[0]
x_std_36 = preprocessing.normalize(x_std_36, axis=0)
x_std_36= np.reshape(x_std_36, (-1,segement_time_size, sensors))
y_std_36= np.delete( y_std_36, [k for k in range(x_std_36.shape[0],shape_y)], None)
p_std_36 = np.delete(p_std_36,[k for k in range(x_std_36.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_std_36), axis=0)
y= np.concatenate((y,y_std_36), axis=0)
p= np.concatenate((p,p_std_36), axis=0)
shp=(x_std_37.shape)[0]
shape_y = y_std_37.shape[0]
shape_p =  p_std_37.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_std_37 =  np.delete(x_std_37, [k for k in range(nece_shp,shp)], None)
x_std_37 = x_std_37.reshape(-1,sensors)
shape_x_std_37 = x_std_37.shape[0]
x_std_37 = preprocessing.normalize(x_std_37, axis=0)
x_std_37= np.reshape(x_std_37, (-1,segement_time_size, sensors))
y_std_37= np.delete( y_std_37, [k for k in range(x_std_37.shape[0],shape_y)], None)
p_std_37 = np.delete(p_std_37,[k for k in range(x_std_37.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_std_37), axis=0)
y= np.concatenate((y,y_std_37), axis=0)
p= np.concatenate((p,p_std_37), axis=0)
shp=(x_std_38.shape)[0]
shape_y = y_std_38.shape[0]
shape_p =  p_std_38.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_std_38 =  np.delete(x_std_38, [k for k in range(nece_shp,shp)], None)
x_std_38 = x_std_38.reshape(-1,sensors)
shape_x_std_38 = x_std_38.shape[0]
x_std_38 = preprocessing.normalize(x_std_38, axis=0)
x_std_38= np.reshape(x_std_38, (-1,segement_time_size, sensors))
y_std_38= np.delete( y_std_38, [k for k in range(x_std_38.shape[0],shape_y)], None)
p_std_38 = np.delete(p_std_38,[k for k in range(x_std_38.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_std_38), axis=0)
y= np.concatenate((y,y_std_38), axis=0)
p= np.concatenate((p,p_std_38), axis=0)
shp=(x_std_39.shape)[0]
shape_y = y_std_39.shape[0]
shape_p =  p_std_39.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_std_39 =  np.delete(x_std_39, [k for k in range(nece_shp,shp)], None)
x_std_39 = x_std_39.reshape(-1,sensors)
shape_x_std_39 = x_std_39.shape[0]
x_std_39 = preprocessing.normalize(x_std_39, axis=0)
x_std_39= np.reshape(x_std_39, (-1,segement_time_size, sensors))
y_std_39= np.delete( y_std_39, [k for k in range(x_std_39.shape[0],shape_y)], None)
p_std_39 = np.delete(p_std_39,[k for k in range(x_std_39.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_std_39), axis=0)
y= np.concatenate((y,y_std_39), axis=0)
p= np.concatenate((p,p_std_39), axis=0)
shp=(x_std_40.shape)[0]
shape_y = y_std_40.shape[0]
shape_p =  p_std_40.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_std_40 =  np.delete(x_std_40, [k for k in range(nece_shp,shp)], None)
x_std_40 = x_std_40.reshape(-1,sensors)
shape_x_std_40 = x_std_40.shape[0]
x_std_40 = preprocessing.normalize(x_std_40, axis=0)
x_std_40= np.reshape(x_std_40, (-1,segement_time_size, sensors))
y_std_40= np.delete( y_std_40, [k for k in range(x_std_40.shape[0],shape_y)], None)
p_std_40 = np.delete(p_std_40,[k for k in range(x_std_40.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_std_40), axis=0)
y= np.concatenate((y,y_std_40), axis=0)
p= np.concatenate((p,p_std_40), axis=0)
shp=(x_std_41.shape)[0]
shape_y = y_std_41.shape[0]
shape_p =  p_std_41.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_std_41 =  np.delete(x_std_41, [k for k in range(nece_shp,shp)], None)
x_std_41 = x_std_41.reshape(-1,sensors)
shape_x_std_41 = x_std_41.shape[0]
x_std_41 = preprocessing.normalize(x_std_41, axis=0)
x_std_41= np.reshape(x_std_41, (-1,segement_time_size, sensors))
y_std_41= np.delete( y_std_41, [k for k in range(x_std_41.shape[0],shape_y)], None)
p_std_41 = np.delete(p_std_41,[k for k in range(x_std_41.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_std_41), axis=0)
y= np.concatenate((y,y_std_41), axis=0)
p= np.concatenate((p,p_std_41), axis=0)
shp=(x_std_42.shape)[0]
shape_y = y_std_42.shape[0]
shape_p =  p_std_42.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_std_42 =  np.delete(x_std_42, [k for k in range(nece_shp,shp)], None)
x_std_42 = x_std_42.reshape(-1,sensors)
shape_x_std_42 = x_std_42.shape[0]
x_std_42 = preprocessing.normalize(x_std_42, axis=0)
x_std_42= np.reshape(x_std_42, (-1,segement_time_size, sensors))
y_std_42= np.delete( y_std_42, [k for k in range(x_std_42.shape[0],shape_y)], None)
p_std_42 = np.delete(p_std_42,[k for k in range(x_std_42.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_std_42), axis=0)
y= np.concatenate((y,y_std_42), axis=0)
p= np.concatenate((p,p_std_42), axis=0)
shp=(x_std_43.shape)[0]
shape_y = y_std_43.shape[0]
shape_p =  p_std_43.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_std_43 =  np.delete(x_std_43, [k for k in range(nece_shp,shp)], None)
x_std_43 = x_std_43.reshape(-1,sensors)
shape_x_std_43 = x_std_43.shape[0]
x_std_43 = preprocessing.normalize(x_std_43, axis=0)
x_std_43= np.reshape(x_std_43, (-1,segement_time_size, sensors))
y_std_43= np.delete( y_std_43, [k for k in range(x_std_43.shape[0],shape_y)], None)
p_std_43 = np.delete(p_std_43,[k for k in range(x_std_43.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_std_43), axis=0)
y= np.concatenate((y,y_std_43), axis=0)
p= np.concatenate((p,p_std_43), axis=0)
shp=(x_std_44.shape)[0]
shape_y = y_std_44.shape[0]
shape_p =  p_std_44.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_std_44 =  np.delete(x_std_44, [k for k in range(nece_shp,shp)], None)
x_std_44 = x_std_44.reshape(-1,sensors)
shape_x_std_44 = x_std_44.shape[0]
x_std_44 = preprocessing.normalize(x_std_44, axis=0)
x_std_44= np.reshape(x_std_44, (-1,segement_time_size, sensors))
y_std_44= np.delete( y_std_44, [k for k in range(x_std_44.shape[0],shape_y)], None)
p_std_44 = np.delete(p_std_44,[k for k in range(x_std_44.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_std_44), axis=0)
y= np.concatenate((y,y_std_44), axis=0)
p= np.concatenate((p,p_std_44), axis=0)
shp=(x_std_45.shape)[0]
shape_y = y_std_45.shape[0]
shape_p =  p_std_45.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_std_45 =  np.delete(x_std_45, [k for k in range(nece_shp,shp)], None)
x_std_45 = x_std_45.reshape(-1,sensors)
shape_x_std_45 = x_std_45.shape[0]
x_std_45 = preprocessing.normalize(x_std_45, axis=0)
x_std_45= np.reshape(x_std_45, (-1,segement_time_size, sensors))
y_std_45= np.delete( y_std_45, [k for k in range(x_std_45.shape[0],shape_y)], None)
p_std_45 = np.delete(p_std_45,[k for k in range(x_std_45.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_std_45), axis=0)
y= np.concatenate((y,y_std_45), axis=0)
p= np.concatenate((p,p_std_45), axis=0)
shp=(x_std_46.shape)[0]
shape_y = y_std_46.shape[0]
shape_p =  p_std_46.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_std_46 =  np.delete(x_std_46, [k for k in range(nece_shp,shp)], None)
x_std_46 = x_std_46.reshape(-1,sensors)
shape_x_std_46 = x_std_46.shape[0]
x_std_46 = preprocessing.normalize(x_std_46, axis=0)
x_std_46= np.reshape(x_std_46, (-1,segement_time_size, sensors))
y_std_46= np.delete( y_std_46, [k for k in range(x_std_46.shape[0],shape_y)], None)
p_std_46 = np.delete(p_std_46,[k for k in range(x_std_46.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_std_46), axis=0)
y= np.concatenate((y,y_std_46), axis=0)
p= np.concatenate((p,p_std_46), axis=0)
shp=(x_std_47.shape)[0]
shape_y = y_std_47.shape[0]
shape_p =  p_std_47.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_std_47 =  np.delete(x_std_47, [k for k in range(nece_shp,shp)], None)
x_std_47 = x_std_47.reshape(-1,sensors)
shape_x_std_47 = x_std_47.shape[0]
x_std_47 = preprocessing.normalize(x_std_47, axis=0)
x_std_47= np.reshape(x_std_47, (-1,segement_time_size, sensors))
y_std_47= np.delete( y_std_47, [k for k in range(x_std_47.shape[0],shape_y)], None)
p_std_47 = np.delete(p_std_47,[k for k in range(x_std_47.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_std_47), axis=0)
y= np.concatenate((y,y_std_47), axis=0)
p= np.concatenate((p,p_std_47), axis=0)
shp=(x_std_48.shape)[0]
shape_y = y_std_48.shape[0]
shape_p =  p_std_48.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_std_48 =  np.delete(x_std_48, [k for k in range(nece_shp,shp)], None)
x_std_48 = x_std_48.reshape(-1,sensors)
shape_x_std_48 = x_std_48.shape[0]
x_std_48 = preprocessing.normalize(x_std_48, axis=0)
x_std_48= np.reshape(x_std_48, (-1,segement_time_size, sensors))
y_std_48= np.delete( y_std_48, [k for k in range(x_std_48.shape[0],shape_y)], None)
p_std_48 = np.delete(p_std_48,[k for k in range(x_std_48.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_std_48), axis=0)
y= np.concatenate((y,y_std_48), axis=0)
p= np.concatenate((p,p_std_48), axis=0)
shp=(x_std_49.shape)[0]
shape_y = y_std_49.shape[0]
shape_p =  p_std_49.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_std_49 =  np.delete(x_std_49, [k for k in range(nece_shp,shp)], None)
x_std_49 = x_std_49.reshape(-1,sensors)
shape_x_std_49 = x_std_49.shape[0]
x_std_49 = preprocessing.normalize(x_std_49, axis=0)
x_std_49= np.reshape(x_std_49, (-1,segement_time_size, sensors))
y_std_49= np.delete( y_std_49, [k for k in range(x_std_49.shape[0],shape_y)], None)
p_std_49 = np.delete(p_std_49,[k for k in range(x_std_49.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_std_49), axis=0)
y= np.concatenate((y,y_std_49), axis=0)
p= np.concatenate((p,p_std_49), axis=0)
shp=(x_std_50.shape)[0]
shape_y = y_std_50.shape[0]
shape_p =  p_std_50.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_std_50 =  np.delete(x_std_50, [k for k in range(nece_shp,shp)], None)
x_std_50 = x_std_50.reshape(-1,sensors)
shape_x_std_50 = x_std_50.shape[0]
x_std_50 = preprocessing.normalize(x_std_50, axis=0)
x_std_50= np.reshape(x_std_50, (-1,segement_time_size, sensors))
y_std_50= np.delete( y_std_50, [k for k in range(x_std_50.shape[0],shape_y)], None)
p_std_50 = np.delete(p_std_50,[k for k in range(x_std_50.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_std_50), axis=0)
y= np.concatenate((y,y_std_50), axis=0)
p= np.concatenate((p,p_std_50), axis=0)
shp=(x_std_51.shape)[0]
shape_y = y_std_51.shape[0]
shape_p =  p_std_51.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_std_51 =  np.delete(x_std_51, [k for k in range(nece_shp,shp)], None)
x_std_51 = x_std_51.reshape(-1,sensors)
shape_x_std_51 = x_std_51.shape[0]
x_std_51 = preprocessing.normalize(x_std_51, axis=0)
x_std_51= np.reshape(x_std_51, (-1,segement_time_size, sensors))
y_std_51= np.delete( y_std_51, [k for k in range(x_std_51.shape[0],shape_y)], None)
p_std_51 = np.delete(p_std_51,[k for k in range(x_std_51.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_std_51), axis=0)
y= np.concatenate((y,y_std_51), axis=0)
p= np.concatenate((p,p_std_51), axis=0)
shp=(x_std_52.shape)[0]
shape_y = y_std_52.shape[0]
shape_p =  p_std_52.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_std_52 =  np.delete(x_std_52, [k for k in range(nece_shp,shp)], None)
x_std_52 = x_std_52.reshape(-1,sensors)
shape_x_std_52 = x_std_52.shape[0]
x_std_52 = preprocessing.normalize(x_std_52, axis=0)
x_std_52= np.reshape(x_std_52, (-1,segement_time_size, sensors))
y_std_52= np.delete( y_std_52, [k for k in range(x_std_52.shape[0],shape_y)], None)
p_std_52 = np.delete(p_std_52,[k for k in range(x_std_52.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_std_52), axis=0)
y= np.concatenate((y,y_std_52), axis=0)
p= np.concatenate((p,p_std_52), axis=0)
shp=(x_std_53.shape)[0]
shape_y = y_std_53.shape[0]
shape_p =  p_std_53.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_std_53 =  np.delete(x_std_53, [k for k in range(nece_shp,shp)], None)
x_std_53 = x_std_53.reshape(-1,sensors)
shape_x_std_53 = x_std_53.shape[0]
x_std_53 = preprocessing.normalize(x_std_53, axis=0)
x_std_53= np.reshape(x_std_53, (-1,segement_time_size, sensors))
y_std_53= np.delete( y_std_53, [k for k in range(x_std_53.shape[0],shape_y)], None)
p_std_53 = np.delete(p_std_53,[k for k in range(x_std_53.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_std_53), axis=0)
y= np.concatenate((y,y_std_53), axis=0)
p= np.concatenate((p,p_std_53), axis=0)
shp=(x_std_54.shape)[0]
shape_y = y_std_54.shape[0]
shape_p =  p_std_54.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_std_54 =  np.delete(x_std_54, [k for k in range(nece_shp,shp)], None)
x_std_54 = x_std_54.reshape(-1,sensors)
shape_x_std_54 = x_std_54.shape[0]
x_std_54 = preprocessing.normalize(x_std_54, axis=0)
x_std_54= np.reshape(x_std_54, (-1,segement_time_size, sensors))
y_std_54= np.delete( y_std_54, [k for k in range(x_std_54.shape[0],shape_y)], None)
p_std_54 = np.delete(p_std_54,[k for k in range(x_std_54.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_std_54), axis=0)
y= np.concatenate((y,y_std_54), axis=0)
p= np.concatenate((p,p_std_54), axis=0)
shp=(x_std_55.shape)[0]
shape_y = y_std_55.shape[0]
shape_p =  p_std_55.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_std_55 =  np.delete(x_std_55, [k for k in range(nece_shp,shp)], None)
x_std_55 = x_std_55.reshape(-1,sensors)
shape_x_std_55 = x_std_55.shape[0]
x_std_55 = preprocessing.normalize(x_std_55, axis=0)
x_std_55= np.reshape(x_std_55, (-1,segement_time_size, sensors))
y_std_55= np.delete( y_std_55, [k for k in range(x_std_55.shape[0],shape_y)], None)
p_std_55 = np.delete(p_std_55,[k for k in range(x_std_55.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_std_55), axis=0)
y= np.concatenate((y,y_std_55), axis=0)
p= np.concatenate((p,p_std_55), axis=0)
shp=(x_std_56.shape)[0]
shape_y = y_std_56.shape[0]
shape_p =  p_std_56.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_std_56 =  np.delete(x_std_56, [k for k in range(nece_shp,shp)], None)
x_std_56 = x_std_56.reshape(-1,sensors)
shape_x_std_56 = x_std_56.shape[0]
x_std_56 = preprocessing.normalize(x_std_56, axis=0)
x_std_56= np.reshape(x_std_56, (-1,segement_time_size, sensors))
y_std_56= np.delete( y_std_56, [k for k in range(x_std_56.shape[0],shape_y)], None)
p_std_56 = np.delete(p_std_56,[k for k in range(x_std_56.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_std_56), axis=0)
y= np.concatenate((y,y_std_56), axis=0)
p= np.concatenate((p,p_std_56), axis=0)
shp=(x_std_57.shape)[0]
shape_y = y_std_57.shape[0]
shape_p =  p_std_57.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_std_57 =  np.delete(x_std_57, [k for k in range(nece_shp,shp)], None)
x_std_57 = x_std_57.reshape(-1,sensors)
shape_x_std_57 = x_std_57.shape[0]
x_std_57 = preprocessing.normalize(x_std_57, axis=0)
x_std_57= np.reshape(x_std_57, (-1,segement_time_size, sensors))
y_std_57= np.delete( y_std_57, [k for k in range(x_std_57.shape[0],shape_y)], None)
p_std_57 = np.delete(p_std_57,[k for k in range(x_std_57.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_std_57), axis=0)
y= np.concatenate((y,y_std_57), axis=0)
p= np.concatenate((p,p_std_57), axis=0)
shp=(x_std_58.shape)[0]
shape_y = y_std_58.shape[0]
shape_p =  p_std_58.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_std_58 =  np.delete(x_std_58, [k for k in range(nece_shp,shp)], None)
x_std_58 = x_std_58.reshape(-1,sensors)
shape_x_std_58 = x_std_58.shape[0]
x_std_58 = preprocessing.normalize(x_std_58, axis=0)
x_std_58= np.reshape(x_std_58, (-1,segement_time_size, sensors))
y_std_58= np.delete( y_std_58, [k for k in range(x_std_58.shape[0],shape_y)], None)
p_std_58 = np.delete(p_std_58,[k for k in range(x_std_58.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_std_58), axis=0)
y= np.concatenate((y,y_std_58), axis=0)
p= np.concatenate((p,p_std_58), axis=0)
shp=(x_std_59.shape)[0]
shape_y = y_std_59.shape[0]
shape_p =  p_std_59.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_std_59 =  np.delete(x_std_59, [k for k in range(nece_shp,shp)], None)
x_std_59 = x_std_59.reshape(-1,sensors)
shape_x_std_59 = x_std_59.shape[0]
x_std_59 = preprocessing.normalize(x_std_59, axis=0)
x_std_59= np.reshape(x_std_59, (-1,segement_time_size, sensors))
y_std_59= np.delete( y_std_59, [k for k in range(x_std_59.shape[0],shape_y)], None)
p_std_59 = np.delete(p_std_59,[k for k in range(x_std_59.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_std_59), axis=0)
y= np.concatenate((y,y_std_59), axis=0)
p= np.concatenate((p,p_std_59), axis=0)
shp=(x_std_60.shape)[0]
shape_y = y_std_60.shape[0]
shape_p =  p_std_60.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_std_60 =  np.delete(x_std_60, [k for k in range(nece_shp,shp)], None)
x_std_60 = x_std_60.reshape(-1,sensors)
shape_x_std_60 = x_std_60.shape[0]
x_std_60 = preprocessing.normalize(x_std_60, axis=0)
x_std_60= np.reshape(x_std_60, (-1,segement_time_size, sensors))
y_std_60= np.delete( y_std_60, [k for k in range(x_std_60.shape[0],shape_y)], None)
p_std_60 = np.delete(p_std_60,[k for k in range(x_std_60.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_std_60), axis=0)
y= np.concatenate((y,y_std_60), axis=0)
p= np.concatenate((p,p_std_60), axis=0)
shp=(x_std_61.shape)[0]
shape_y = y_std_61.shape[0]
shape_p =  p_std_61.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_std_61 =  np.delete(x_std_61, [k for k in range(nece_shp,shp)], None)
x_std_61 = x_std_61.reshape(-1,sensors)
shape_x_std_61 = x_std_61.shape[0]
x_std_61 = preprocessing.normalize(x_std_61, axis=0)
x_std_61= np.reshape(x_std_61, (-1,segement_time_size, sensors))
y_std_61= np.delete( y_std_61, [k for k in range(x_std_61.shape[0],shape_y)], None)
p_std_61 = np.delete(p_std_61,[k for k in range(x_std_61.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_std_61), axis=0)
y= np.concatenate((y,y_std_61), axis=0)
p= np.concatenate((p,p_std_61), axis=0)
shp=(x_std_62.shape)[0]
shape_y = y_std_62.shape[0]
shape_p =  p_std_62.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_std_62 =  np.delete(x_std_62, [k for k in range(nece_shp,shp)], None)
x_std_62 = x_std_62.reshape(-1,sensors)
shape_x_std_62 = x_std_62.shape[0]
x_std_62 = preprocessing.normalize(x_std_62, axis=0)
x_std_62= np.reshape(x_std_62, (-1,segement_time_size, sensors))
y_std_62= np.delete( y_std_62, [k for k in range(x_std_62.shape[0],shape_y)], None)
p_std_62 = np.delete(p_std_62,[k for k in range(x_std_62.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_std_62), axis=0)
y= np.concatenate((y,y_std_62), axis=0)
p= np.concatenate((p,p_std_62), axis=0)
shp=(x_std_63.shape)[0]
shape_y = y_std_63.shape[0]
shape_p =  p_std_63.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_std_63 =  np.delete(x_std_63, [k for k in range(nece_shp,shp)], None)
x_std_63 = x_std_63.reshape(-1,sensors)
shape_x_std_63 = x_std_63.shape[0]
x_std_63 = preprocessing.normalize(x_std_63, axis=0)
x_std_63= np.reshape(x_std_63, (-1,segement_time_size, sensors))
y_std_63= np.delete( y_std_63, [k for k in range(x_std_63.shape[0],shape_y)], None)
p_std_63 = np.delete(p_std_63,[k for k in range(x_std_63.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_std_63), axis=0)
y= np.concatenate((y,y_std_63), axis=0)
p= np.concatenate((p,p_std_63), axis=0)
shp=(x_std_64.shape)[0]
shape_y = y_std_64.shape[0]
shape_p =  p_std_64.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_std_64 =  np.delete(x_std_64, [k for k in range(nece_shp,shp)], None)
x_std_64 = x_std_64.reshape(-1,sensors)
shape_x_std_64 = x_std_64.shape[0]
x_std_64 = preprocessing.normalize(x_std_64, axis=0)
x_std_64= np.reshape(x_std_64, (-1,segement_time_size, sensors))
y_std_64= np.delete( y_std_64, [k for k in range(x_std_64.shape[0],shape_y)], None)
p_std_64 = np.delete(p_std_64,[k for k in range(x_std_64.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_std_64), axis=0)
y= np.concatenate((y,y_std_64), axis=0)
p= np.concatenate((p,p_std_64), axis=0)
shp=(x_std_65.shape)[0]
shape_y = y_std_65.shape[0]
shape_p =  p_std_65.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_std_65 =  np.delete(x_std_65, [k for k in range(nece_shp,shp)], None)
x_std_65 = x_std_65.reshape(-1,sensors)
shape_x_std_65 = x_std_65.shape[0]
x_std_65 = preprocessing.normalize(x_std_65, axis=0)
x_std_65= np.reshape(x_std_65, (-1,segement_time_size, sensors))
y_std_65= np.delete( y_std_65, [k for k in range(x_std_65.shape[0],shape_y)], None)
p_std_65 = np.delete(p_std_65,[k for k in range(x_std_65.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_std_65), axis=0)
y= np.concatenate((y,y_std_65), axis=0)
p= np.concatenate((p,p_std_65), axis=0)
shp=(x_std_66.shape)[0]
shape_y = y_std_66.shape[0]
shape_p =  p_std_66.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_std_66 =  np.delete(x_std_66, [k for k in range(nece_shp,shp)], None)
x_std_66 = x_std_66.reshape(-1,sensors)
shape_x_std_66 = x_std_66.shape[0]
x_std_66 = preprocessing.normalize(x_std_66, axis=0)
x_std_66= np.reshape(x_std_66, (-1,segement_time_size, sensors))
y_std_66= np.delete( y_std_66, [k for k in range(x_std_66.shape[0],shape_y)], None)
p_std_66 = np.delete(p_std_66,[k for k in range(x_std_66.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_std_66), axis=0)
y= np.concatenate((y,y_std_66), axis=0)
p= np.concatenate((p,p_std_66), axis=0)
shp=(x_std_67.shape)[0]
shape_y = y_std_67.shape[0]
shape_p =  p_std_67.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_std_67 =  np.delete(x_std_67, [k for k in range(nece_shp,shp)], None)
x_std_67 = x_std_67.reshape(-1,sensors)
shape_x_std_67 = x_std_67.shape[0]
x_std_67 = preprocessing.normalize(x_std_67, axis=0)
x_std_67= np.reshape(x_std_67, (-1,segement_time_size, sensors))
y_std_67= np.delete( y_std_67, [k for k in range(x_std_67.shape[0],shape_y)], None)
p_std_67 = np.delete(p_std_67,[k for k in range(x_std_67.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_std_67), axis=0)
y= np.concatenate((y,y_std_67), axis=0)
p= np.concatenate((p,p_std_67), axis=0)
shp=(x_std_68.shape)[0]
shape_y = y_std_68.shape[0]
shape_p =  p_std_68.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_std_68 =  np.delete(x_std_68, [k for k in range(nece_shp,shp)], None)
x_std_68 = x_std_68.reshape(-1,sensors)
shape_x_std_68 = x_std_68.shape[0]
x_std_68 = preprocessing.normalize(x_std_68, axis=0)
x_std_68= np.reshape(x_std_68, (-1,segement_time_size, sensors))
y_std_68= np.delete( y_std_68, [k for k in range(x_std_68.shape[0],shape_y)], None)
p_std_68 = np.delete(p_std_68,[k for k in range(x_std_68.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_std_68), axis=0)
y= np.concatenate((y,y_std_68), axis=0)
p= np.concatenate((p,p_std_68), axis=0)
shp=(x_std_69.shape)[0]
shape_y = y_std_69.shape[0]
shape_p =  p_std_69.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_std_69 =  np.delete(x_std_69, [k for k in range(nece_shp,shp)], None)
x_std_69 = x_std_69.reshape(-1,sensors)
shape_x_std_69 = x_std_69.shape[0]
x_std_69 = preprocessing.normalize(x_std_69, axis=0)
x_std_69= np.reshape(x_std_69, (-1,segement_time_size, sensors))
y_std_69= np.delete( y_std_69, [k for k in range(x_std_69.shape[0],shape_y)], None)
p_std_69 = np.delete(p_std_69,[k for k in range(x_std_69.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_std_69), axis=0)
y= np.concatenate((y,y_std_69), axis=0)
p= np.concatenate((p,p_std_69), axis=0)
shp=(x_std_70.shape)[0]
shape_y = y_std_70.shape[0]
shape_p =  p_std_70.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_std_70 =  np.delete(x_std_70, [k for k in range(nece_shp,shp)], None)
x_std_70 = x_std_70.reshape(-1,sensors)
shape_x_std_70 = x_std_70.shape[0]
x_std_70 = preprocessing.normalize(x_std_70, axis=0)
x_std_70= np.reshape(x_std_70, (-1,segement_time_size, sensors))
y_std_70= np.delete( y_std_70, [k for k in range(x_std_70.shape[0],shape_y)], None)
p_std_70 = np.delete(p_std_70,[k for k in range(x_std_70.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_std_70), axis=0)
y= np.concatenate((y,y_std_70), axis=0)
p= np.concatenate((p,p_std_70), axis=0)
shp=(x_std_71.shape)[0]
shape_y = y_std_71.shape[0]
shape_p =  p_std_71.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_std_71 =  np.delete(x_std_71, [k for k in range(nece_shp,shp)], None)
x_std_71 = x_std_71.reshape(-1,sensors)
shape_x_std_71 = x_std_71.shape[0]
x_std_71 = preprocessing.normalize(x_std_71, axis=0)
x_std_71= np.reshape(x_std_71, (-1,segement_time_size, sensors))
y_std_71= np.delete( y_std_71, [k for k in range(x_std_71.shape[0],shape_y)], None)
p_std_71 = np.delete(p_std_71,[k for k in range(x_std_71.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_std_71), axis=0)
y= np.concatenate((y,y_std_71), axis=0)
p= np.concatenate((p,p_std_71), axis=0)
shp=(x_std_72.shape)[0]
shape_y = y_std_72.shape[0]
shape_p =  p_std_72.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_std_72 =  np.delete(x_std_72, [k for k in range(nece_shp,shp)], None)
x_std_72 = x_std_72.reshape(-1,sensors)
shape_x_std_72 = x_std_72.shape[0]
x_std_72 = preprocessing.normalize(x_std_72, axis=0)
x_std_72= np.reshape(x_std_72, (-1,segement_time_size, sensors))
y_std_72= np.delete( y_std_72, [k for k in range(x_std_72.shape[0],shape_y)], None)
p_std_72 = np.delete(p_std_72,[k for k in range(x_std_72.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_std_72), axis=0)
y= np.concatenate((y,y_std_72), axis=0)
p= np.concatenate((p,p_std_72), axis=0)
shp=(x_std_73.shape)[0]
shape_y = y_std_73.shape[0]
shape_p =  p_std_73.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_std_73 =  np.delete(x_std_73, [k for k in range(nece_shp,shp)], None)
x_std_73 = x_std_73.reshape(-1,sensors)
shape_x_std_73 = x_std_73.shape[0]
x_std_73 = preprocessing.normalize(x_std_73, axis=0)
x_std_73= np.reshape(x_std_73, (-1,segement_time_size, sensors))
y_std_73= np.delete( y_std_73, [k for k in range(x_std_73.shape[0],shape_y)], None)
p_std_73 = np.delete(p_std_73,[k for k in range(x_std_73.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_std_73), axis=0)
y= np.concatenate((y,y_std_73), axis=0)
p= np.concatenate((p,p_std_73), axis=0)
shp=(x_std_74.shape)[0]
shape_y = y_std_74.shape[0]
shape_p =  p_std_74.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_std_74 =  np.delete(x_std_74, [k for k in range(nece_shp,shp)], None)
x_std_74 = x_std_74.reshape(-1,sensors)
shape_x_std_74 = x_std_74.shape[0]
x_std_74 = preprocessing.normalize(x_std_74, axis=0)
x_std_74= np.reshape(x_std_74, (-1,segement_time_size, sensors))
y_std_74= np.delete( y_std_74, [k for k in range(x_std_74.shape[0],shape_y)], None)
p_std_74 = np.delete(p_std_74,[k for k in range(x_std_74.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_std_74), axis=0)
y= np.concatenate((y,y_std_74), axis=0)
p= np.concatenate((p,p_std_74), axis=0)
shp=(x_std_75.shape)[0]
shape_y = y_std_75.shape[0]
shape_p =  p_std_75.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_std_75 =  np.delete(x_std_75, [k for k in range(nece_shp,shp)], None)
x_std_75 = x_std_75.reshape(-1,sensors)
shape_x_std_75 = x_std_75.shape[0]
x_std_75 = preprocessing.normalize(x_std_75, axis=0)
x_std_75= np.reshape(x_std_75, (-1,segement_time_size, sensors))
y_std_75= np.delete( y_std_75, [k for k in range(x_std_75.shape[0],shape_y)], None)
p_std_75 = np.delete(p_std_75,[k for k in range(x_std_75.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_std_75), axis=0)
y= np.concatenate((y,y_std_75), axis=0)
p= np.concatenate((p,p_std_75), axis=0)
shp=(x_std_76.shape)[0]
shape_y = y_std_76.shape[0]
shape_p =  p_std_76.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_std_76 =  np.delete(x_std_76, [k for k in range(nece_shp,shp)], None)
x_std_76 = x_std_76.reshape(-1,sensors)
shape_x_std_76 = x_std_76.shape[0]
x_std_76 = preprocessing.normalize(x_std_76, axis=0)
x_std_76= np.reshape(x_std_76, (-1,segement_time_size, sensors))
y_std_76= np.delete( y_std_76, [k for k in range(x_std_76.shape[0],shape_y)], None)
p_std_76 = np.delete(p_std_76,[k for k in range(x_std_76.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_std_76), axis=0)
y= np.concatenate((y,y_std_76), axis=0)
p= np.concatenate((p,p_std_76), axis=0)
shp=(x_std_77.shape)[0]
shape_y = y_std_77.shape[0]
shape_p =  p_std_77.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_std_77 =  np.delete(x_std_77, [k for k in range(nece_shp,shp)], None)
x_std_77 = x_std_77.reshape(-1,sensors)
shape_x_std_77 = x_std_77.shape[0]
x_std_77 = preprocessing.normalize(x_std_77, axis=0)
x_std_77= np.reshape(x_std_77, (-1,segement_time_size, sensors))
y_std_77= np.delete( y_std_77, [k for k in range(x_std_77.shape[0],shape_y)], None)
p_std_77 = np.delete(p_std_77,[k for k in range(x_std_77.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_std_77), axis=0)
y= np.concatenate((y,y_std_77), axis=0)
p= np.concatenate((p,p_std_77), axis=0)
shp=(x_std_78.shape)[0]
shape_y = y_std_78.shape[0]
shape_p =  p_std_78.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_std_78 =  np.delete(x_std_78, [k for k in range(nece_shp,shp)], None)
x_std_78 = x_std_78.reshape(-1,sensors)
shape_x_std_78 = x_std_78.shape[0]
x_std_78 = preprocessing.normalize(x_std_78, axis=0)
x_std_78= np.reshape(x_std_78, (-1,segement_time_size, sensors))
y_std_78= np.delete( y_std_78, [k for k in range(x_std_78.shape[0],shape_y)], None)
p_std_78 = np.delete(p_std_78,[k for k in range(x_std_78.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_std_78), axis=0)
y= np.concatenate((y,y_std_78), axis=0)
p= np.concatenate((p,p_std_78), axis=0)
shp=(x_std_79.shape)[0]
shape_y = y_std_79.shape[0]
shape_p =  p_std_79.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_std_79 =  np.delete(x_std_79, [k for k in range(nece_shp,shp)], None)
x_std_79 = x_std_79.reshape(-1,sensors)
shape_x_std_79 = x_std_79.shape[0]
x_std_79 = preprocessing.normalize(x_std_79, axis=0)
x_std_79= np.reshape(x_std_79, (-1,segement_time_size, sensors))
y_std_79= np.delete( y_std_79, [k for k in range(x_std_79.shape[0],shape_y)], None)
p_std_79 = np.delete(p_std_79,[k for k in range(x_std_79.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_std_79), axis=0)
y= np.concatenate((y,y_std_79), axis=0)
p= np.concatenate((p,p_std_79), axis=0)
shp=(x_std_80.shape)[0]
shape_y = y_std_80.shape[0]
shape_p =  p_std_80.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_std_80 =  np.delete(x_std_80, [k for k in range(nece_shp,shp)], None)
x_std_80 = x_std_80.reshape(-1,sensors)
shape_x_std_80 = x_std_80.shape[0]
x_std_80 = preprocessing.normalize(x_std_80, axis=0)
x_std_80= np.reshape(x_std_80, (-1,segement_time_size, sensors))
y_std_80= np.delete( y_std_80, [k for k in range(x_std_80.shape[0],shape_y)], None)
p_std_80 = np.delete(p_std_80,[k for k in range(x_std_80.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_std_80), axis=0)
y= np.concatenate((y,y_std_80), axis=0)
p= np.concatenate((p,p_std_80), axis=0)
shp=(x_std_81.shape)[0]
shape_y = y_std_81.shape[0]
shape_p =  p_std_81.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_std_81 =  np.delete(x_std_81, [k for k in range(nece_shp,shp)], None)
x_std_81 = x_std_81.reshape(-1,sensors)
shape_x_std_81 = x_std_81.shape[0]
x_std_81 = preprocessing.normalize(x_std_81, axis=0)
x_std_81= np.reshape(x_std_81, (-1,segement_time_size, sensors))
y_std_81= np.delete( y_std_81, [k for k in range(x_std_81.shape[0],shape_y)], None)
p_std_81 = np.delete(p_std_81,[k for k in range(x_std_81.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_std_81), axis=0)
y= np.concatenate((y,y_std_81), axis=0)
p= np.concatenate((p,p_std_81), axis=0)
shp=(x_std_82.shape)[0]
shape_y = y_std_82.shape[0]
shape_p =  p_std_82.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_std_82 =  np.delete(x_std_82, [k for k in range(nece_shp,shp)], None)
x_std_82 = x_std_82.reshape(-1,sensors)
shape_x_std_82 = x_std_82.shape[0]
x_std_82 = preprocessing.normalize(x_std_82, axis=0)
x_std_82= np.reshape(x_std_82, (-1,segement_time_size, sensors))
y_std_82= np.delete( y_std_82, [k for k in range(x_std_82.shape[0],shape_y)], None)
p_std_82 = np.delete(p_std_82,[k for k in range(x_std_82.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_std_82), axis=0)
y= np.concatenate((y,y_std_82), axis=0)
p= np.concatenate((p,p_std_82), axis=0)
shp=(x_std_83.shape)[0]
shape_y = y_std_83.shape[0]
shape_p =  p_std_83.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_std_83 =  np.delete(x_std_83, [k for k in range(nece_shp,shp)], None)
x_std_83 = x_std_83.reshape(-1,sensors)
shape_x_std_83 = x_std_83.shape[0]
x_std_83 = preprocessing.normalize(x_std_83, axis=0)
x_std_83= np.reshape(x_std_83, (-1,segement_time_size, sensors))
y_std_83= np.delete( y_std_83, [k for k in range(x_std_83.shape[0],shape_y)], None)
p_std_83 = np.delete(p_std_83,[k for k in range(x_std_83.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_std_83), axis=0)
y= np.concatenate((y,y_std_83), axis=0)
p= np.concatenate((p,p_std_83), axis=0)
shp=(x_std_84.shape)[0]
shape_y = y_std_84.shape[0]
shape_p =  p_std_84.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_std_84 =  np.delete(x_std_84, [k for k in range(nece_shp,shp)], None)
x_std_84 = x_std_84.reshape(-1,sensors)
shape_x_std_84 = x_std_84.shape[0]
x_std_84 = preprocessing.normalize(x_std_84, axis=0)
x_std_84= np.reshape(x_std_84, (-1,segement_time_size, sensors))
y_std_84= np.delete( y_std_84, [k for k in range(x_std_84.shape[0],shape_y)], None)
p_std_84 = np.delete(p_std_84,[k for k in range(x_std_84.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_std_84), axis=0)
y= np.concatenate((y,y_std_84), axis=0)
p= np.concatenate((p,p_std_84), axis=0)
shp=(x_std_85.shape)[0]
shape_y = y_std_85.shape[0]
shape_p =  p_std_85.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_std_85 =  np.delete(x_std_85, [k for k in range(nece_shp,shp)], None)
x_std_85 = x_std_85.reshape(-1,sensors)
shape_x_std_85 = x_std_85.shape[0]
x_std_85 = preprocessing.normalize(x_std_85, axis=0)
x_std_85= np.reshape(x_std_85, (-1,segement_time_size, sensors))
y_std_85= np.delete( y_std_85, [k for k in range(x_std_85.shape[0],shape_y)], None)
p_std_85 = np.delete(p_std_85,[k for k in range(x_std_85.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_std_85), axis=0)
y= np.concatenate((y,y_std_85), axis=0)
p= np.concatenate((p,p_std_85), axis=0)
shp=(x_std_86.shape)[0]
shape_y = y_std_86.shape[0]
shape_p =  p_std_86.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_std_86 =  np.delete(x_std_86, [k for k in range(nece_shp,shp)], None)
x_std_86 = x_std_86.reshape(-1,sensors)
shape_x_std_86 = x_std_86.shape[0]
x_std_86 = preprocessing.normalize(x_std_86, axis=0)
x_std_86= np.reshape(x_std_86, (-1,segement_time_size, sensors))
y_std_86= np.delete( y_std_86, [k for k in range(x_std_86.shape[0],shape_y)], None)
p_std_86 = np.delete(p_std_86,[k for k in range(x_std_86.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_std_86), axis=0)
y= np.concatenate((y,y_std_86), axis=0)
p= np.concatenate((p,p_std_86), axis=0)
shp=(x_std_87.shape)[0]
shape_y = y_std_87.shape[0]
shape_p =  p_std_87.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_std_87 =  np.delete(x_std_87, [k for k in range(nece_shp,shp)], None)
x_std_87 = x_std_87.reshape(-1,sensors)
shape_x_std_87 = x_std_87.shape[0]
x_std_87 = preprocessing.normalize(x_std_87, axis=0)
x_std_87= np.reshape(x_std_87, (-1,segement_time_size, sensors))
y_std_87= np.delete( y_std_87, [k for k in range(x_std_87.shape[0],shape_y)], None)
p_std_87 = np.delete(p_std_87,[k for k in range(x_std_87.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_std_87), axis=0)
y= np.concatenate((y,y_std_87), axis=0)
p= np.concatenate((p,p_std_87), axis=0)
shp=(x_std_88.shape)[0]
shape_y = y_std_88.shape[0]
shape_p =  p_std_88.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_std_88 =  np.delete(x_std_88, [k for k in range(nece_shp,shp)], None)
x_std_88 = x_std_88.reshape(-1,sensors)
shape_x_std_88 = x_std_88.shape[0]
x_std_88 = preprocessing.normalize(x_std_88, axis=0)
x_std_88= np.reshape(x_std_88, (-1,segement_time_size, sensors))
y_std_88= np.delete( y_std_88, [k for k in range(x_std_88.shape[0],shape_y)], None)
p_std_88 = np.delete(p_std_88,[k for k in range(x_std_88.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_std_88), axis=0)
y= np.concatenate((y,y_std_88), axis=0)
p= np.concatenate((p,p_std_88), axis=0)
shp=(x_std_89.shape)[0]
shape_y = y_std_89.shape[0]
shape_p =  p_std_89.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_std_89 =  np.delete(x_std_89, [k for k in range(nece_shp,shp)], None)
x_std_89 = x_std_89.reshape(-1,sensors)
shape_x_std_89 = x_std_89.shape[0]
x_std_89 = preprocessing.normalize(x_std_89, axis=0)
x_std_89= np.reshape(x_std_89, (-1,segement_time_size, sensors))
y_std_89= np.delete( y_std_89, [k for k in range(x_std_89.shape[0],shape_y)], None)
p_std_89 = np.delete(p_std_89,[k for k in range(x_std_89.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_std_89), axis=0)
y= np.concatenate((y,y_std_89), axis=0)
p= np.concatenate((p,p_std_89), axis=0)
shp=(x_std_90.shape)[0]
shape_y = y_std_90.shape[0]
shape_p =  p_std_90.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_std_90 =  np.delete(x_std_90, [k for k in range(nece_shp,shp)], None)
x_std_90 = x_std_90.reshape(-1,sensors)
shape_x_std_90 = x_std_90.shape[0]
x_std_90 = preprocessing.normalize(x_std_90, axis=0)
x_std_90= np.reshape(x_std_90, (-1,segement_time_size, sensors))
y_std_90= np.delete( y_std_90, [k for k in range(x_std_90.shape[0],shape_y)], None)
p_std_90 = np.delete(p_std_90,[k for k in range(x_std_90.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_std_90), axis=0)
y= np.concatenate((y,y_std_90), axis=0)
p= np.concatenate((p,p_std_90), axis=0)
shp=(x_std_91.shape)[0]
shape_y = y_std_91.shape[0]
shape_p =  p_std_91.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_std_91 =  np.delete(x_std_91, [k for k in range(nece_shp,shp)], None)
x_std_91 = x_std_91.reshape(-1,sensors)
shape_x_std_91 = x_std_91.shape[0]
x_std_91 = preprocessing.normalize(x_std_91, axis=0)
x_std_91= np.reshape(x_std_91, (-1,segement_time_size, sensors))
y_std_91= np.delete( y_std_91, [k for k in range(x_std_91.shape[0],shape_y)], None)
p_std_91 = np.delete(p_std_91,[k for k in range(x_std_91.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_std_91), axis=0)
y= np.concatenate((y,y_std_91), axis=0)
p= np.concatenate((p,p_std_91), axis=0)
shp=(x_std_92.shape)[0]
shape_y = y_std_92.shape[0]
shape_p =  p_std_92.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_std_92 =  np.delete(x_std_92, [k for k in range(nece_shp,shp)], None)
x_std_92 = x_std_92.reshape(-1,sensors)
shape_x_std_92 = x_std_92.shape[0]
x_std_92 = preprocessing.normalize(x_std_92, axis=0)
x_std_92= np.reshape(x_std_92, (-1,segement_time_size, sensors))
y_std_92= np.delete( y_std_92, [k for k in range(x_std_92.shape[0],shape_y)], None)
p_std_92 = np.delete(p_std_92,[k for k in range(x_std_92.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_std_92), axis=0)
y= np.concatenate((y,y_std_92), axis=0)
p= np.concatenate((p,p_std_92), axis=0)
shp=(x_std_93.shape)[0]
shape_y = y_std_93.shape[0]
shape_p =  p_std_93.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_std_93 =  np.delete(x_std_93, [k for k in range(nece_shp,shp)], None)
x_std_93 = x_std_93.reshape(-1,sensors)
shape_x_std_93 = x_std_93.shape[0]
x_std_93 = preprocessing.normalize(x_std_93, axis=0)
x_std_93= np.reshape(x_std_93, (-1,segement_time_size, sensors))
y_std_93= np.delete( y_std_93, [k for k in range(x_std_93.shape[0],shape_y)], None)
p_std_93 = np.delete(p_std_93,[k for k in range(x_std_93.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_std_93), axis=0)
y= np.concatenate((y,y_std_93), axis=0)
p= np.concatenate((p,p_std_93), axis=0)
shp=(x_std_94.shape)[0]
shape_y = y_std_94.shape[0]
shape_p =  p_std_94.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_std_94 =  np.delete(x_std_94, [k for k in range(nece_shp,shp)], None)
x_std_94 = x_std_94.reshape(-1,sensors)
shape_x_std_94 = x_std_94.shape[0]
x_std_94 = preprocessing.normalize(x_std_94, axis=0)
x_std_94= np.reshape(x_std_94, (-1,segement_time_size, sensors))
y_std_94= np.delete( y_std_94, [k for k in range(x_std_94.shape[0],shape_y)], None)
p_std_94 = np.delete(p_std_94,[k for k in range(x_std_94.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_std_94), axis=0)
y= np.concatenate((y,y_std_94), axis=0)
p= np.concatenate((p,p_std_94), axis=0)
shp=(x_std_95.shape)[0]
shape_y = y_std_95.shape[0]
shape_p =  p_std_95.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_std_95 =  np.delete(x_std_95, [k for k in range(nece_shp,shp)], None)
x_std_95 = x_std_95.reshape(-1,sensors)
shape_x_std_95 = x_std_95.shape[0]
x_std_95 = preprocessing.normalize(x_std_95, axis=0)
x_std_95= np.reshape(x_std_95, (-1,segement_time_size, sensors))
y_std_95= np.delete( y_std_95, [k for k in range(x_std_95.shape[0],shape_y)], None)
p_std_95 = np.delete(p_std_95,[k for k in range(x_std_95.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_std_95), axis=0)
y= np.concatenate((y,y_std_95), axis=0)
p= np.concatenate((p,p_std_95), axis=0)
shp=(x_std_96.shape)[0]
shape_y = y_std_96.shape[0]
shape_p =  p_std_96.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_std_96 =  np.delete(x_std_96, [k for k in range(nece_shp,shp)], None)
x_std_96 = x_std_96.reshape(-1,sensors)
shape_x_std_96 = x_std_96.shape[0]
x_std_96 = preprocessing.normalize(x_std_96, axis=0)
x_std_96= np.reshape(x_std_96, (-1,segement_time_size, sensors))
y_std_96= np.delete( y_std_96, [k for k in range(x_std_96.shape[0],shape_y)], None)
p_std_96 = np.delete(p_std_96,[k for k in range(x_std_96.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_std_96), axis=0)
y= np.concatenate((y,y_std_96), axis=0)
p= np.concatenate((p,p_std_96), axis=0)
shp=(x_std_97.shape)[0]
shape_y = y_std_97.shape[0]
shape_p =  p_std_97.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_std_97 =  np.delete(x_std_97, [k for k in range(nece_shp,shp)], None)
x_std_97 = x_std_97.reshape(-1,sensors)
shape_x_std_97 = x_std_97.shape[0]
x_std_97 = preprocessing.normalize(x_std_97, axis=0)
x_std_97= np.reshape(x_std_97, (-1,segement_time_size, sensors))
y_std_97= np.delete( y_std_97, [k for k in range(x_std_97.shape[0],shape_y)], None)
p_std_97 = np.delete(p_std_97,[k for k in range(x_std_97.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_std_97), axis=0)
y= np.concatenate((y,y_std_97), axis=0)
p= np.concatenate((p,p_std_97), axis=0)
shp=(x_std_98.shape)[0]
shape_y = y_std_98.shape[0]
shape_p =  p_std_98.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_std_98 =  np.delete(x_std_98, [k for k in range(nece_shp,shp)], None)
x_std_98 = x_std_98.reshape(-1,sensors)
shape_x_std_98 = x_std_98.shape[0]
x_std_98 = preprocessing.normalize(x_std_98, axis=0)
x_std_98= np.reshape(x_std_98, (-1,segement_time_size, sensors))
y_std_98= np.delete( y_std_98, [k for k in range(x_std_98.shape[0],shape_y)], None)
p_std_98 = np.delete(p_std_98,[k for k in range(x_std_98.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_std_98), axis=0)
y= np.concatenate((y,y_std_98), axis=0)
p= np.concatenate((p,p_std_98), axis=0)
shp=(x_std_99.shape)[0]
shape_y = y_std_99.shape[0]
shape_p =  p_std_99.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_std_99 =  np.delete(x_std_99, [k for k in range(nece_shp,shp)], None)
x_std_99 = x_std_99.reshape(-1,sensors)
shape_x_std_99 = x_std_99.shape[0]
x_std_99 = preprocessing.normalize(x_std_99, axis=0)
x_std_99= np.reshape(x_std_99, (-1,segement_time_size, sensors))
y_std_99= np.delete( y_std_99, [k for k in range(x_std_99.shape[0],shape_y)], None)
p_std_99 = np.delete(p_std_99,[k for k in range(x_std_99.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_std_99), axis=0)
y= np.concatenate((y,y_std_99), axis=0)
p= np.concatenate((p,p_std_99), axis=0)
shp=(x_st_0.shape)[0]
shape_y = y_st_0.shape[0]
shape_p =  p_st_0.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_st_0 =  np.delete(x_st_0, [k for k in range(nece_shp,shp)], None)
x_st_0 = x_st_0.reshape(-1,sensors)
shape_x_st_0 = x_st_0.shape[0]
x_st_0 = preprocessing.normalize(x_st_0, axis=0)
x_st_0= np.reshape(x_st_0, (-1,segement_time_size, sensors))
y_st_0= np.delete( y_st_0, [k for k in range(x_st_0.shape[0],shape_y)], None)
p_st_0 = np.delete(p_st_0,[k for k in range(x_st_0.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_st_0), axis=0)
y= np.concatenate((y,y_st_0), axis=0)
p= np.concatenate((p,p_st_0), axis=0)
shp=(x_st_1.shape)[0]
shape_y = y_st_1.shape[0]
shape_p =  p_st_1.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_st_1 =  np.delete(x_st_1, [k for k in range(nece_shp,shp)], None)
x_st_1 = x_st_1.reshape(-1,sensors)
shape_x_st_1 = x_st_1.shape[0]
x_st_1 = preprocessing.normalize(x_st_1, axis=0)
x_st_1= np.reshape(x_st_1, (-1,segement_time_size, sensors))
y_st_1= np.delete( y_st_1, [k for k in range(x_st_1.shape[0],shape_y)], None)
p_st_1 = np.delete(p_st_1,[k for k in range(x_st_1.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_st_1), axis=0)
y= np.concatenate((y,y_st_1), axis=0)
p= np.concatenate((p,p_st_1), axis=0)
shp=(x_st_2.shape)[0]
shape_y = y_st_2.shape[0]
shape_p =  p_st_2.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_st_2 =  np.delete(x_st_2, [k for k in range(nece_shp,shp)], None)
x_st_2 = x_st_2.reshape(-1,sensors)
shape_x_st_2 = x_st_2.shape[0]
x_st_2 = preprocessing.normalize(x_st_2, axis=0)
x_st_2= np.reshape(x_st_2, (-1,segement_time_size, sensors))
y_st_2= np.delete( y_st_2, [k for k in range(x_st_2.shape[0],shape_y)], None)
p_st_2 = np.delete(p_st_2,[k for k in range(x_st_2.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_st_2), axis=0)
y= np.concatenate((y,y_st_2), axis=0)
p= np.concatenate((p,p_st_2), axis=0)
shp=(x_st_3.shape)[0]
shape_y = y_st_3.shape[0]
shape_p =  p_st_3.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_st_3 =  np.delete(x_st_3, [k for k in range(nece_shp,shp)], None)
x_st_3 = x_st_3.reshape(-1,sensors)
shape_x_st_3 = x_st_3.shape[0]
x_st_3 = preprocessing.normalize(x_st_3, axis=0)
x_st_3= np.reshape(x_st_3, (-1,segement_time_size, sensors))
y_st_3= np.delete( y_st_3, [k for k in range(x_st_3.shape[0],shape_y)], None)
p_st_3 = np.delete(p_st_3,[k for k in range(x_st_3.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_st_3), axis=0)
y= np.concatenate((y,y_st_3), axis=0)
p= np.concatenate((p,p_st_3), axis=0)
shp=(x_st_4.shape)[0]
shape_y = y_st_4.shape[0]
shape_p =  p_st_4.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_st_4 =  np.delete(x_st_4, [k for k in range(nece_shp,shp)], None)
x_st_4 = x_st_4.reshape(-1,sensors)
shape_x_st_4 = x_st_4.shape[0]
x_st_4 = preprocessing.normalize(x_st_4, axis=0)
x_st_4= np.reshape(x_st_4, (-1,segement_time_size, sensors))
y_st_4= np.delete( y_st_4, [k for k in range(x_st_4.shape[0],shape_y)], None)
p_st_4 = np.delete(p_st_4,[k for k in range(x_st_4.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_st_4), axis=0)
y= np.concatenate((y,y_st_4), axis=0)
p= np.concatenate((p,p_st_4), axis=0)
shp=(x_st_5.shape)[0]
shape_y = y_st_5.shape[0]
shape_p =  p_st_5.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_st_5 =  np.delete(x_st_5, [k for k in range(nece_shp,shp)], None)
x_st_5 = x_st_5.reshape(-1,sensors)
shape_x_st_5 = x_st_5.shape[0]
x_st_5 = preprocessing.normalize(x_st_5, axis=0)
x_st_5= np.reshape(x_st_5, (-1,segement_time_size, sensors))
y_st_5= np.delete( y_st_5, [k for k in range(x_st_5.shape[0],shape_y)], None)
p_st_5 = np.delete(p_st_5,[k for k in range(x_st_5.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_st_5), axis=0)
y= np.concatenate((y,y_st_5), axis=0)
p= np.concatenate((p,p_st_5), axis=0)
shp=(x_st_6.shape)[0]
shape_y = y_st_6.shape[0]
shape_p =  p_st_6.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_st_6 =  np.delete(x_st_6, [k for k in range(nece_shp,shp)], None)
x_st_6 = x_st_6.reshape(-1,sensors)
shape_x_st_6 = x_st_6.shape[0]
x_st_6 = preprocessing.normalize(x_st_6, axis=0)
x_st_6= np.reshape(x_st_6, (-1,segement_time_size, sensors))
y_st_6= np.delete( y_st_6, [k for k in range(x_st_6.shape[0],shape_y)], None)
p_st_6 = np.delete(p_st_6,[k for k in range(x_st_6.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_st_6), axis=0)
y= np.concatenate((y,y_st_6), axis=0)
p= np.concatenate((p,p_st_6), axis=0)
shp=(x_st_7.shape)[0]
shape_y = y_st_7.shape[0]
shape_p =  p_st_7.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_st_7 =  np.delete(x_st_7, [k for k in range(nece_shp,shp)], None)
x_st_7 = x_st_7.reshape(-1,sensors)
shape_x_st_7 = x_st_7.shape[0]
x_st_7 = preprocessing.normalize(x_st_7, axis=0)
x_st_7= np.reshape(x_st_7, (-1,segement_time_size, sensors))
y_st_7= np.delete( y_st_7, [k for k in range(x_st_7.shape[0],shape_y)], None)
p_st_7 = np.delete(p_st_7,[k for k in range(x_st_7.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_st_7), axis=0)
y= np.concatenate((y,y_st_7), axis=0)
p= np.concatenate((p,p_st_7), axis=0)
shp=(x_st_8.shape)[0]
shape_y = y_st_8.shape[0]
shape_p =  p_st_8.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_st_8 =  np.delete(x_st_8, [k for k in range(nece_shp,shp)], None)
x_st_8 = x_st_8.reshape(-1,sensors)
shape_x_st_8 = x_st_8.shape[0]
x_st_8 = preprocessing.normalize(x_st_8, axis=0)
x_st_8= np.reshape(x_st_8, (-1,segement_time_size, sensors))
y_st_8= np.delete( y_st_8, [k for k in range(x_st_8.shape[0],shape_y)], None)
p_st_8 = np.delete(p_st_8,[k for k in range(x_st_8.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_st_8), axis=0)
y= np.concatenate((y,y_st_8), axis=0)
p= np.concatenate((p,p_st_8), axis=0)
shp=(x_st_9.shape)[0]
shape_y = y_st_9.shape[0]
shape_p =  p_st_9.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_st_9 =  np.delete(x_st_9, [k for k in range(nece_shp,shp)], None)
x_st_9 = x_st_9.reshape(-1,sensors)
shape_x_st_9 = x_st_9.shape[0]
x_st_9 = preprocessing.normalize(x_st_9, axis=0)
x_st_9= np.reshape(x_st_9, (-1,segement_time_size, sensors))
y_st_9= np.delete( y_st_9, [k for k in range(x_st_9.shape[0],shape_y)], None)
p_st_9 = np.delete(p_st_9,[k for k in range(x_st_9.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_st_9), axis=0)
y= np.concatenate((y,y_st_9), axis=0)
p= np.concatenate((p,p_st_9), axis=0)
shp=(x_st_10.shape)[0]
shape_y = y_st_10.shape[0]
shape_p =  p_st_10.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_st_10 =  np.delete(x_st_10, [k for k in range(nece_shp,shp)], None)
x_st_10 = x_st_10.reshape(-1,sensors)
shape_x_st_10 = x_st_10.shape[0]
x_st_10 = preprocessing.normalize(x_st_10, axis=0)
x_st_10= np.reshape(x_st_10, (-1,segement_time_size, sensors))
y_st_10= np.delete( y_st_10, [k for k in range(x_st_10.shape[0],shape_y)], None)
p_st_10 = np.delete(p_st_10,[k for k in range(x_st_10.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_st_10), axis=0)
y= np.concatenate((y,y_st_10), axis=0)
p= np.concatenate((p,p_st_10), axis=0)
shp=(x_st_11.shape)[0]
shape_y = y_st_11.shape[0]
shape_p =  p_st_11.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_st_11 =  np.delete(x_st_11, [k for k in range(nece_shp,shp)], None)
x_st_11 = x_st_11.reshape(-1,sensors)
shape_x_st_11 = x_st_11.shape[0]
x_st_11 = preprocessing.normalize(x_st_11, axis=0)
x_st_11= np.reshape(x_st_11, (-1,segement_time_size, sensors))
y_st_11= np.delete( y_st_11, [k for k in range(x_st_11.shape[0],shape_y)], None)
p_st_11 = np.delete(p_st_11,[k for k in range(x_st_11.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_st_11), axis=0)
y= np.concatenate((y,y_st_11), axis=0)
p= np.concatenate((p,p_st_11), axis=0)
shp=(x_st_12.shape)[0]
shape_y = y_st_12.shape[0]
shape_p =  p_st_12.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_st_12 =  np.delete(x_st_12, [k for k in range(nece_shp,shp)], None)
x_st_12 = x_st_12.reshape(-1,sensors)
shape_x_st_12 = x_st_12.shape[0]
x_st_12 = preprocessing.normalize(x_st_12, axis=0)
x_st_12= np.reshape(x_st_12, (-1,segement_time_size, sensors))
y_st_12= np.delete( y_st_12, [k for k in range(x_st_12.shape[0],shape_y)], None)
p_st_12 = np.delete(p_st_12,[k for k in range(x_st_12.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_st_12), axis=0)
y= np.concatenate((y,y_st_12), axis=0)
p= np.concatenate((p,p_st_12), axis=0)
shp=(x_st_13.shape)[0]
shape_y = y_st_13.shape[0]
shape_p =  p_st_13.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_st_13 =  np.delete(x_st_13, [k for k in range(nece_shp,shp)], None)
x_st_13 = x_st_13.reshape(-1,sensors)
shape_x_st_13 = x_st_13.shape[0]
x_st_13 = preprocessing.normalize(x_st_13, axis=0)
x_st_13= np.reshape(x_st_13, (-1,segement_time_size, sensors))
y_st_13= np.delete( y_st_13, [k for k in range(x_st_13.shape[0],shape_y)], None)
p_st_13 = np.delete(p_st_13,[k for k in range(x_st_13.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_st_13), axis=0)
y= np.concatenate((y,y_st_13), axis=0)
p= np.concatenate((p,p_st_13), axis=0)
shp=(x_st_14.shape)[0]
shape_y = y_st_14.shape[0]
shape_p =  p_st_14.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_st_14 =  np.delete(x_st_14, [k for k in range(nece_shp,shp)], None)
x_st_14 = x_st_14.reshape(-1,sensors)
shape_x_st_14 = x_st_14.shape[0]
x_st_14 = preprocessing.normalize(x_st_14, axis=0)
x_st_14= np.reshape(x_st_14, (-1,segement_time_size, sensors))
y_st_14= np.delete( y_st_14, [k for k in range(x_st_14.shape[0],shape_y)], None)
p_st_14 = np.delete(p_st_14,[k for k in range(x_st_14.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_st_14), axis=0)
y= np.concatenate((y,y_st_14), axis=0)
p= np.concatenate((p,p_st_14), axis=0)
shp=(x_st_15.shape)[0]
shape_y = y_st_15.shape[0]
shape_p =  p_st_15.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_st_15 =  np.delete(x_st_15, [k for k in range(nece_shp,shp)], None)
x_st_15 = x_st_15.reshape(-1,sensors)
shape_x_st_15 = x_st_15.shape[0]
x_st_15 = preprocessing.normalize(x_st_15, axis=0)
x_st_15= np.reshape(x_st_15, (-1,segement_time_size, sensors))
y_st_15= np.delete( y_st_15, [k for k in range(x_st_15.shape[0],shape_y)], None)
p_st_15 = np.delete(p_st_15,[k for k in range(x_st_15.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_st_15), axis=0)
y= np.concatenate((y,y_st_15), axis=0)
p= np.concatenate((p,p_st_15), axis=0)
shp=(x_st_16.shape)[0]
shape_y = y_st_16.shape[0]
shape_p =  p_st_16.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_st_16 =  np.delete(x_st_16, [k for k in range(nece_shp,shp)], None)
x_st_16 = x_st_16.reshape(-1,sensors)
shape_x_st_16 = x_st_16.shape[0]
x_st_16 = preprocessing.normalize(x_st_16, axis=0)
x_st_16= np.reshape(x_st_16, (-1,segement_time_size, sensors))
y_st_16= np.delete( y_st_16, [k for k in range(x_st_16.shape[0],shape_y)], None)
p_st_16 = np.delete(p_st_16,[k for k in range(x_st_16.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_st_16), axis=0)
y= np.concatenate((y,y_st_16), axis=0)
p= np.concatenate((p,p_st_16), axis=0)
shp=(x_st_17.shape)[0]
shape_y = y_st_17.shape[0]
shape_p =  p_st_17.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_st_17 =  np.delete(x_st_17, [k for k in range(nece_shp,shp)], None)
x_st_17 = x_st_17.reshape(-1,sensors)
shape_x_st_17 = x_st_17.shape[0]
x_st_17 = preprocessing.normalize(x_st_17, axis=0)
x_st_17= np.reshape(x_st_17, (-1,segement_time_size, sensors))
y_st_17= np.delete( y_st_17, [k for k in range(x_st_17.shape[0],shape_y)], None)
p_st_17 = np.delete(p_st_17,[k for k in range(x_st_17.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_st_17), axis=0)
y= np.concatenate((y,y_st_17), axis=0)
p= np.concatenate((p,p_st_17), axis=0)
shp=(x_st_18.shape)[0]
shape_y = y_st_18.shape[0]
shape_p =  p_st_18.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_st_18 =  np.delete(x_st_18, [k for k in range(nece_shp,shp)], None)
x_st_18 = x_st_18.reshape(-1,sensors)
shape_x_st_18 = x_st_18.shape[0]
x_st_18 = preprocessing.normalize(x_st_18, axis=0)
x_st_18= np.reshape(x_st_18, (-1,segement_time_size, sensors))
y_st_18= np.delete( y_st_18, [k for k in range(x_st_18.shape[0],shape_y)], None)
p_st_18 = np.delete(p_st_18,[k for k in range(x_st_18.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_st_18), axis=0)
y= np.concatenate((y,y_st_18), axis=0)
p= np.concatenate((p,p_st_18), axis=0)
shp=(x_st_19.shape)[0]
shape_y = y_st_19.shape[0]
shape_p =  p_st_19.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_st_19 =  np.delete(x_st_19, [k for k in range(nece_shp,shp)], None)
x_st_19 = x_st_19.reshape(-1,sensors)
shape_x_st_19 = x_st_19.shape[0]
x_st_19 = preprocessing.normalize(x_st_19, axis=0)
x_st_19= np.reshape(x_st_19, (-1,segement_time_size, sensors))
y_st_19= np.delete( y_st_19, [k for k in range(x_st_19.shape[0],shape_y)], None)
p_st_19 = np.delete(p_st_19,[k for k in range(x_st_19.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_st_19), axis=0)
y= np.concatenate((y,y_st_19), axis=0)
p= np.concatenate((p,p_st_19), axis=0)
shp=(x_st_20.shape)[0]
shape_y = y_st_20.shape[0]
shape_p =  p_st_20.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_st_20 =  np.delete(x_st_20, [k for k in range(nece_shp,shp)], None)
x_st_20 = x_st_20.reshape(-1,sensors)
shape_x_st_20 = x_st_20.shape[0]
x_st_20 = preprocessing.normalize(x_st_20, axis=0)
x_st_20= np.reshape(x_st_20, (-1,segement_time_size, sensors))
y_st_20= np.delete( y_st_20, [k for k in range(x_st_20.shape[0],shape_y)], None)
p_st_20 = np.delete(p_st_20,[k for k in range(x_st_20.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_st_20), axis=0)
y= np.concatenate((y,y_st_20), axis=0)
p= np.concatenate((p,p_st_20), axis=0)
shp=(x_st_21.shape)[0]
shape_y = y_st_21.shape[0]
shape_p =  p_st_21.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_st_21 =  np.delete(x_st_21, [k for k in range(nece_shp,shp)], None)
x_st_21 = x_st_21.reshape(-1,sensors)
shape_x_st_21 = x_st_21.shape[0]
x_st_21 = preprocessing.normalize(x_st_21, axis=0)
x_st_21= np.reshape(x_st_21, (-1,segement_time_size, sensors))
y_st_21= np.delete( y_st_21, [k for k in range(x_st_21.shape[0],shape_y)], None)
p_st_21 = np.delete(p_st_21,[k for k in range(x_st_21.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_st_21), axis=0)
y= np.concatenate((y,y_st_21), axis=0)
p= np.concatenate((p,p_st_21), axis=0)
shp=(x_st_22.shape)[0]
shape_y = y_st_22.shape[0]
shape_p =  p_st_22.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_st_22 =  np.delete(x_st_22, [k for k in range(nece_shp,shp)], None)
x_st_22 = x_st_22.reshape(-1,sensors)
shape_x_st_22 = x_st_22.shape[0]
x_st_22 = preprocessing.normalize(x_st_22, axis=0)
x_st_22= np.reshape(x_st_22, (-1,segement_time_size, sensors))
y_st_22= np.delete( y_st_22, [k for k in range(x_st_22.shape[0],shape_y)], None)
p_st_22 = np.delete(p_st_22,[k for k in range(x_st_22.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_st_22), axis=0)
y= np.concatenate((y,y_st_22), axis=0)
p= np.concatenate((p,p_st_22), axis=0)
shp=(x_st_23.shape)[0]
shape_y = y_st_23.shape[0]
shape_p =  p_st_23.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_st_23 =  np.delete(x_st_23, [k for k in range(nece_shp,shp)], None)
x_st_23 = x_st_23.reshape(-1,sensors)
shape_x_st_23 = x_st_23.shape[0]
x_st_23 = preprocessing.normalize(x_st_23, axis=0)
x_st_23= np.reshape(x_st_23, (-1,segement_time_size, sensors))
y_st_23= np.delete( y_st_23, [k for k in range(x_st_23.shape[0],shape_y)], None)
p_st_23 = np.delete(p_st_23,[k for k in range(x_st_23.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_st_23), axis=0)
y= np.concatenate((y,y_st_23), axis=0)
p= np.concatenate((p,p_st_23), axis=0)
shp=(x_st_24.shape)[0]
shape_y = y_st_24.shape[0]
shape_p =  p_st_24.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_st_24 =  np.delete(x_st_24, [k for k in range(nece_shp,shp)], None)
x_st_24 = x_st_24.reshape(-1,sensors)
shape_x_st_24 = x_st_24.shape[0]
x_st_24 = preprocessing.normalize(x_st_24, axis=0)
x_st_24= np.reshape(x_st_24, (-1,segement_time_size, sensors))
y_st_24= np.delete( y_st_24, [k for k in range(x_st_24.shape[0],shape_y)], None)
p_st_24 = np.delete(p_st_24,[k for k in range(x_st_24.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_st_24), axis=0)
y= np.concatenate((y,y_st_24), axis=0)
p= np.concatenate((p,p_st_24), axis=0)
shp=(x_st_25.shape)[0]
shape_y = y_st_25.shape[0]
shape_p =  p_st_25.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_st_25 =  np.delete(x_st_25, [k for k in range(nece_shp,shp)], None)
x_st_25 = x_st_25.reshape(-1,sensors)
shape_x_st_25 = x_st_25.shape[0]
x_st_25 = preprocessing.normalize(x_st_25, axis=0)
x_st_25= np.reshape(x_st_25, (-1,segement_time_size, sensors))
y_st_25= np.delete( y_st_25, [k for k in range(x_st_25.shape[0],shape_y)], None)
p_st_25 = np.delete(p_st_25,[k for k in range(x_st_25.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_st_25), axis=0)
y= np.concatenate((y,y_st_25), axis=0)
p= np.concatenate((p,p_st_25), axis=0)
shp=(x_st_26.shape)[0]
shape_y = y_st_26.shape[0]
shape_p =  p_st_26.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_st_26 =  np.delete(x_st_26, [k for k in range(nece_shp,shp)], None)
x_st_26 = x_st_26.reshape(-1,sensors)
shape_x_st_26 = x_st_26.shape[0]
x_st_26 = preprocessing.normalize(x_st_26, axis=0)
x_st_26= np.reshape(x_st_26, (-1,segement_time_size, sensors))
y_st_26= np.delete( y_st_26, [k for k in range(x_st_26.shape[0],shape_y)], None)
p_st_26 = np.delete(p_st_26,[k for k in range(x_st_26.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_st_26), axis=0)
y= np.concatenate((y,y_st_26), axis=0)
p= np.concatenate((p,p_st_26), axis=0)
shp=(x_st_27.shape)[0]
shape_y = y_st_27.shape[0]
shape_p =  p_st_27.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_st_27 =  np.delete(x_st_27, [k for k in range(nece_shp,shp)], None)
x_st_27 = x_st_27.reshape(-1,sensors)
shape_x_st_27 = x_st_27.shape[0]
x_st_27 = preprocessing.normalize(x_st_27, axis=0)
x_st_27= np.reshape(x_st_27, (-1,segement_time_size, sensors))
y_st_27= np.delete( y_st_27, [k for k in range(x_st_27.shape[0],shape_y)], None)
p_st_27 = np.delete(p_st_27,[k for k in range(x_st_27.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_st_27), axis=0)
y= np.concatenate((y,y_st_27), axis=0)
p= np.concatenate((p,p_st_27), axis=0)
shp=(x_st_28.shape)[0]
shape_y = y_st_28.shape[0]
shape_p =  p_st_28.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_st_28 =  np.delete(x_st_28, [k for k in range(nece_shp,shp)], None)
x_st_28 = x_st_28.reshape(-1,sensors)
shape_x_st_28 = x_st_28.shape[0]
x_st_28 = preprocessing.normalize(x_st_28, axis=0)
x_st_28= np.reshape(x_st_28, (-1,segement_time_size, sensors))
y_st_28= np.delete( y_st_28, [k for k in range(x_st_28.shape[0],shape_y)], None)
p_st_28 = np.delete(p_st_28,[k for k in range(x_st_28.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_st_28), axis=0)
y= np.concatenate((y,y_st_28), axis=0)
p= np.concatenate((p,p_st_28), axis=0)
shp=(x_st_29.shape)[0]
shape_y = y_st_29.shape[0]
shape_p =  p_st_29.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_st_29 =  np.delete(x_st_29, [k for k in range(nece_shp,shp)], None)
x_st_29 = x_st_29.reshape(-1,sensors)
shape_x_st_29 = x_st_29.shape[0]
x_st_29 = preprocessing.normalize(x_st_29, axis=0)
x_st_29= np.reshape(x_st_29, (-1,segement_time_size, sensors))
y_st_29= np.delete( y_st_29, [k for k in range(x_st_29.shape[0],shape_y)], None)
p_st_29 = np.delete(p_st_29,[k for k in range(x_st_29.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_st_29), axis=0)
y= np.concatenate((y,y_st_29), axis=0)
p= np.concatenate((p,p_st_29), axis=0)
shp=(x_st_30.shape)[0]
shape_y = y_st_30.shape[0]
shape_p =  p_st_30.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_st_30 =  np.delete(x_st_30, [k for k in range(nece_shp,shp)], None)
x_st_30 = x_st_30.reshape(-1,sensors)
shape_x_st_30 = x_st_30.shape[0]
x_st_30 = preprocessing.normalize(x_st_30, axis=0)
x_st_30= np.reshape(x_st_30, (-1,segement_time_size, sensors))
y_st_30= np.delete( y_st_30, [k for k in range(x_st_30.shape[0],shape_y)], None)
p_st_30 = np.delete(p_st_30,[k for k in range(x_st_30.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_st_30), axis=0)
y= np.concatenate((y,y_st_30), axis=0)
p= np.concatenate((p,p_st_30), axis=0)
shp=(x_st_31.shape)[0]
shape_y = y_st_31.shape[0]
shape_p =  p_st_31.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_st_31 =  np.delete(x_st_31, [k for k in range(nece_shp,shp)], None)
x_st_31 = x_st_31.reshape(-1,sensors)
shape_x_st_31 = x_st_31.shape[0]
x_st_31 = preprocessing.normalize(x_st_31, axis=0)
x_st_31= np.reshape(x_st_31, (-1,segement_time_size, sensors))
y_st_31= np.delete( y_st_31, [k for k in range(x_st_31.shape[0],shape_y)], None)
p_st_31 = np.delete(p_st_31,[k for k in range(x_st_31.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_st_31), axis=0)
y= np.concatenate((y,y_st_31), axis=0)
p= np.concatenate((p,p_st_31), axis=0)
shp=(x_st_32.shape)[0]
shape_y = y_st_32.shape[0]
shape_p =  p_st_32.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_st_32 =  np.delete(x_st_32, [k for k in range(nece_shp,shp)], None)
x_st_32 = x_st_32.reshape(-1,sensors)
shape_x_st_32 = x_st_32.shape[0]
x_st_32 = preprocessing.normalize(x_st_32, axis=0)
x_st_32= np.reshape(x_st_32, (-1,segement_time_size, sensors))
y_st_32= np.delete( y_st_32, [k for k in range(x_st_32.shape[0],shape_y)], None)
p_st_32 = np.delete(p_st_32,[k for k in range(x_st_32.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_st_32), axis=0)
y= np.concatenate((y,y_st_32), axis=0)
p= np.concatenate((p,p_st_32), axis=0)
shp=(x_st_33.shape)[0]
shape_y = y_st_33.shape[0]
shape_p =  p_st_33.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_st_33 =  np.delete(x_st_33, [k for k in range(nece_shp,shp)], None)
x_st_33 = x_st_33.reshape(-1,sensors)
shape_x_st_33 = x_st_33.shape[0]
x_st_33 = preprocessing.normalize(x_st_33, axis=0)
x_st_33= np.reshape(x_st_33, (-1,segement_time_size, sensors))
y_st_33= np.delete( y_st_33, [k for k in range(x_st_33.shape[0],shape_y)], None)
p_st_33 = np.delete(p_st_33,[k for k in range(x_st_33.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_st_33), axis=0)
y= np.concatenate((y,y_st_33), axis=0)
p= np.concatenate((p,p_st_33), axis=0)
shp=(x_st_34.shape)[0]
shape_y = y_st_34.shape[0]
shape_p =  p_st_34.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_st_34 =  np.delete(x_st_34, [k for k in range(nece_shp,shp)], None)
x_st_34 = x_st_34.reshape(-1,sensors)
shape_x_st_34 = x_st_34.shape[0]
x_st_34 = preprocessing.normalize(x_st_34, axis=0)
x_st_34= np.reshape(x_st_34, (-1,segement_time_size, sensors))
y_st_34= np.delete( y_st_34, [k for k in range(x_st_34.shape[0],shape_y)], None)
p_st_34 = np.delete(p_st_34,[k for k in range(x_st_34.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_st_34), axis=0)
y= np.concatenate((y,y_st_34), axis=0)
p= np.concatenate((p,p_st_34), axis=0)
shp=(x_st_35.shape)[0]
shape_y = y_st_35.shape[0]
shape_p =  p_st_35.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_st_35 =  np.delete(x_st_35, [k for k in range(nece_shp,shp)], None)
x_st_35 = x_st_35.reshape(-1,sensors)
shape_x_st_35 = x_st_35.shape[0]
x_st_35 = preprocessing.normalize(x_st_35, axis=0)
x_st_35= np.reshape(x_st_35, (-1,segement_time_size, sensors))
y_st_35= np.delete( y_st_35, [k for k in range(x_st_35.shape[0],shape_y)], None)
p_st_35 = np.delete(p_st_35,[k for k in range(x_st_35.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_st_35), axis=0)
y= np.concatenate((y,y_st_35), axis=0)
p= np.concatenate((p,p_st_35), axis=0)
shp=(x_st_36.shape)[0]
shape_y = y_st_36.shape[0]
shape_p =  p_st_36.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_st_36 =  np.delete(x_st_36, [k for k in range(nece_shp,shp)], None)
x_st_36 = x_st_36.reshape(-1,sensors)
shape_x_st_36 = x_st_36.shape[0]
x_st_36 = preprocessing.normalize(x_st_36, axis=0)
x_st_36= np.reshape(x_st_36, (-1,segement_time_size, sensors))
y_st_36= np.delete( y_st_36, [k for k in range(x_st_36.shape[0],shape_y)], None)
p_st_36 = np.delete(p_st_36,[k for k in range(x_st_36.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_st_36), axis=0)
y= np.concatenate((y,y_st_36), axis=0)
p= np.concatenate((p,p_st_36), axis=0)
shp=(x_st_37.shape)[0]
shape_y = y_st_37.shape[0]
shape_p =  p_st_37.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_st_37 =  np.delete(x_st_37, [k for k in range(nece_shp,shp)], None)
x_st_37 = x_st_37.reshape(-1,sensors)
shape_x_st_37 = x_st_37.shape[0]
x_st_37 = preprocessing.normalize(x_st_37, axis=0)
x_st_37= np.reshape(x_st_37, (-1,segement_time_size, sensors))
y_st_37= np.delete( y_st_37, [k for k in range(x_st_37.shape[0],shape_y)], None)
p_st_37 = np.delete(p_st_37,[k for k in range(x_st_37.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_st_37), axis=0)
y= np.concatenate((y,y_st_37), axis=0)
p= np.concatenate((p,p_st_37), axis=0)
shp=(x_st_38.shape)[0]
shape_y = y_st_38.shape[0]
shape_p =  p_st_38.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_st_38 =  np.delete(x_st_38, [k for k in range(nece_shp,shp)], None)
x_st_38 = x_st_38.reshape(-1,sensors)
shape_x_st_38 = x_st_38.shape[0]
x_st_38 = preprocessing.normalize(x_st_38, axis=0)
x_st_38= np.reshape(x_st_38, (-1,segement_time_size, sensors))
y_st_38= np.delete( y_st_38, [k for k in range(x_st_38.shape[0],shape_y)], None)
p_st_38 = np.delete(p_st_38,[k for k in range(x_st_38.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_st_38), axis=0)
y= np.concatenate((y,y_st_38), axis=0)
p= np.concatenate((p,p_st_38), axis=0)
shp=(x_st_39.shape)[0]
shape_y = y_st_39.shape[0]
shape_p =  p_st_39.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_st_39 =  np.delete(x_st_39, [k for k in range(nece_shp,shp)], None)
x_st_39 = x_st_39.reshape(-1,sensors)
shape_x_st_39 = x_st_39.shape[0]
x_st_39 = preprocessing.normalize(x_st_39, axis=0)
x_st_39= np.reshape(x_st_39, (-1,segement_time_size, sensors))
y_st_39= np.delete( y_st_39, [k for k in range(x_st_39.shape[0],shape_y)], None)
p_st_39 = np.delete(p_st_39,[k for k in range(x_st_39.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_st_39), axis=0)
y= np.concatenate((y,y_st_39), axis=0)
p= np.concatenate((p,p_st_39), axis=0)
shp=(x_st_40.shape)[0]
shape_y = y_st_40.shape[0]
shape_p =  p_st_40.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_st_40 =  np.delete(x_st_40, [k for k in range(nece_shp,shp)], None)
x_st_40 = x_st_40.reshape(-1,sensors)
shape_x_st_40 = x_st_40.shape[0]
x_st_40 = preprocessing.normalize(x_st_40, axis=0)
x_st_40= np.reshape(x_st_40, (-1,segement_time_size, sensors))
y_st_40= np.delete( y_st_40, [k for k in range(x_st_40.shape[0],shape_y)], None)
p_st_40 = np.delete(p_st_40,[k for k in range(x_st_40.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_st_40), axis=0)
y= np.concatenate((y,y_st_40), axis=0)
p= np.concatenate((p,p_st_40), axis=0)
shp=(x_st_41.shape)[0]
shape_y = y_st_41.shape[0]
shape_p =  p_st_41.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_st_41 =  np.delete(x_st_41, [k for k in range(nece_shp,shp)], None)
x_st_41 = x_st_41.reshape(-1,sensors)
shape_x_st_41 = x_st_41.shape[0]
x_st_41 = preprocessing.normalize(x_st_41, axis=0)
x_st_41= np.reshape(x_st_41, (-1,segement_time_size, sensors))
y_st_41= np.delete( y_st_41, [k for k in range(x_st_41.shape[0],shape_y)], None)
p_st_41 = np.delete(p_st_41,[k for k in range(x_st_41.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_st_41), axis=0)
y= np.concatenate((y,y_st_41), axis=0)
p= np.concatenate((p,p_st_41), axis=0)
shp=(x_st_42.shape)[0]
shape_y = y_st_42.shape[0]
shape_p =  p_st_42.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_st_42 =  np.delete(x_st_42, [k for k in range(nece_shp,shp)], None)
x_st_42 = x_st_42.reshape(-1,sensors)
shape_x_st_42 = x_st_42.shape[0]
x_st_42 = preprocessing.normalize(x_st_42, axis=0)
x_st_42= np.reshape(x_st_42, (-1,segement_time_size, sensors))
y_st_42= np.delete( y_st_42, [k for k in range(x_st_42.shape[0],shape_y)], None)
p_st_42 = np.delete(p_st_42,[k for k in range(x_st_42.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_st_42), axis=0)
y= np.concatenate((y,y_st_42), axis=0)
p= np.concatenate((p,p_st_42), axis=0)
shp=(x_st_43.shape)[0]
shape_y = y_st_43.shape[0]
shape_p =  p_st_43.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_st_43 =  np.delete(x_st_43, [k for k in range(nece_shp,shp)], None)
x_st_43 = x_st_43.reshape(-1,sensors)
shape_x_st_43 = x_st_43.shape[0]
x_st_43 = preprocessing.normalize(x_st_43, axis=0)
x_st_43= np.reshape(x_st_43, (-1,segement_time_size, sensors))
y_st_43= np.delete( y_st_43, [k for k in range(x_st_43.shape[0],shape_y)], None)
p_st_43 = np.delete(p_st_43,[k for k in range(x_st_43.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_st_43), axis=0)
y= np.concatenate((y,y_st_43), axis=0)
p= np.concatenate((p,p_st_43), axis=0)
shp=(x_st_44.shape)[0]
shape_y = y_st_44.shape[0]
shape_p =  p_st_44.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_st_44 =  np.delete(x_st_44, [k for k in range(nece_shp,shp)], None)
x_st_44 = x_st_44.reshape(-1,sensors)
shape_x_st_44 = x_st_44.shape[0]
x_st_44 = preprocessing.normalize(x_st_44, axis=0)
x_st_44= np.reshape(x_st_44, (-1,segement_time_size, sensors))
y_st_44= np.delete( y_st_44, [k for k in range(x_st_44.shape[0],shape_y)], None)
p_st_44 = np.delete(p_st_44,[k for k in range(x_st_44.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_st_44), axis=0)
y= np.concatenate((y,y_st_44), axis=0)
p= np.concatenate((p,p_st_44), axis=0)
shp=(x_st_45.shape)[0]
shape_y = y_st_45.shape[0]
shape_p =  p_st_45.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_st_45 =  np.delete(x_st_45, [k for k in range(nece_shp,shp)], None)
x_st_45 = x_st_45.reshape(-1,sensors)
shape_x_st_45 = x_st_45.shape[0]
x_st_45 = preprocessing.normalize(x_st_45, axis=0)
x_st_45= np.reshape(x_st_45, (-1,segement_time_size, sensors))
y_st_45= np.delete( y_st_45, [k for k in range(x_st_45.shape[0],shape_y)], None)
p_st_45 = np.delete(p_st_45,[k for k in range(x_st_45.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_st_45), axis=0)
y= np.concatenate((y,y_st_45), axis=0)
p= np.concatenate((p,p_st_45), axis=0)
shp=(x_st_46.shape)[0]
shape_y = y_st_46.shape[0]
shape_p =  p_st_46.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_st_46 =  np.delete(x_st_46, [k for k in range(nece_shp,shp)], None)
x_st_46 = x_st_46.reshape(-1,sensors)
shape_x_st_46 = x_st_46.shape[0]
x_st_46 = preprocessing.normalize(x_st_46, axis=0)
x_st_46= np.reshape(x_st_46, (-1,segement_time_size, sensors))
y_st_46= np.delete( y_st_46, [k for k in range(x_st_46.shape[0],shape_y)], None)
p_st_46 = np.delete(p_st_46,[k for k in range(x_st_46.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_st_46), axis=0)
y= np.concatenate((y,y_st_46), axis=0)
p= np.concatenate((p,p_st_46), axis=0)
shp=(x_st_47.shape)[0]
shape_y = y_st_47.shape[0]
shape_p =  p_st_47.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_st_47 =  np.delete(x_st_47, [k for k in range(nece_shp,shp)], None)
x_st_47 = x_st_47.reshape(-1,sensors)
shape_x_st_47 = x_st_47.shape[0]
x_st_47 = preprocessing.normalize(x_st_47, axis=0)
x_st_47= np.reshape(x_st_47, (-1,segement_time_size, sensors))
y_st_47= np.delete( y_st_47, [k for k in range(x_st_47.shape[0],shape_y)], None)
p_st_47 = np.delete(p_st_47,[k for k in range(x_st_47.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_st_47), axis=0)
y= np.concatenate((y,y_st_47), axis=0)
p= np.concatenate((p,p_st_47), axis=0)
shp=(x_st_48.shape)[0]
shape_y = y_st_48.shape[0]
shape_p =  p_st_48.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_st_48 =  np.delete(x_st_48, [k for k in range(nece_shp,shp)], None)
x_st_48 = x_st_48.reshape(-1,sensors)
shape_x_st_48 = x_st_48.shape[0]
x_st_48 = preprocessing.normalize(x_st_48, axis=0)
x_st_48= np.reshape(x_st_48, (-1,segement_time_size, sensors))
y_st_48= np.delete( y_st_48, [k for k in range(x_st_48.shape[0],shape_y)], None)
p_st_48 = np.delete(p_st_48,[k for k in range(x_st_48.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_st_48), axis=0)
y= np.concatenate((y,y_st_48), axis=0)
p= np.concatenate((p,p_st_48), axis=0)
shp=(x_st_49.shape)[0]
shape_y = y_st_49.shape[0]
shape_p =  p_st_49.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_st_49 =  np.delete(x_st_49, [k for k in range(nece_shp,shp)], None)
x_st_49 = x_st_49.reshape(-1,sensors)
shape_x_st_49 = x_st_49.shape[0]
x_st_49 = preprocessing.normalize(x_st_49, axis=0)
x_st_49= np.reshape(x_st_49, (-1,segement_time_size, sensors))
y_st_49= np.delete( y_st_49, [k for k in range(x_st_49.shape[0],shape_y)], None)
p_st_49 = np.delete(p_st_49,[k for k in range(x_st_49.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_st_49), axis=0)
y= np.concatenate((y,y_st_49), axis=0)
p= np.concatenate((p,p_st_49), axis=0)
shp=(x_st_50.shape)[0]
shape_y = y_st_50.shape[0]
shape_p =  p_st_50.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_st_50 =  np.delete(x_st_50, [k for k in range(nece_shp,shp)], None)
x_st_50 = x_st_50.reshape(-1,sensors)
shape_x_st_50 = x_st_50.shape[0]
x_st_50 = preprocessing.normalize(x_st_50, axis=0)
x_st_50= np.reshape(x_st_50, (-1,segement_time_size, sensors))
y_st_50= np.delete( y_st_50, [k for k in range(x_st_50.shape[0],shape_y)], None)
p_st_50 = np.delete(p_st_50,[k for k in range(x_st_50.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_st_50), axis=0)
y= np.concatenate((y,y_st_50), axis=0)
p= np.concatenate((p,p_st_50), axis=0)
shp=(x_st_51.shape)[0]
shape_y = y_st_51.shape[0]
shape_p =  p_st_51.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_st_51 =  np.delete(x_st_51, [k for k in range(nece_shp,shp)], None)
x_st_51 = x_st_51.reshape(-1,sensors)
shape_x_st_51 = x_st_51.shape[0]
x_st_51 = preprocessing.normalize(x_st_51, axis=0)
x_st_51= np.reshape(x_st_51, (-1,segement_time_size, sensors))
y_st_51= np.delete( y_st_51, [k for k in range(x_st_51.shape[0],shape_y)], None)
p_st_51 = np.delete(p_st_51,[k for k in range(x_st_51.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_st_51), axis=0)
y= np.concatenate((y,y_st_51), axis=0)
p= np.concatenate((p,p_st_51), axis=0)
shp=(x_st_52.shape)[0]
shape_y = y_st_52.shape[0]
shape_p =  p_st_52.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_st_52 =  np.delete(x_st_52, [k for k in range(nece_shp,shp)], None)
x_st_52 = x_st_52.reshape(-1,sensors)
shape_x_st_52 = x_st_52.shape[0]
x_st_52 = preprocessing.normalize(x_st_52, axis=0)
x_st_52= np.reshape(x_st_52, (-1,segement_time_size, sensors))
y_st_52= np.delete( y_st_52, [k for k in range(x_st_52.shape[0],shape_y)], None)
p_st_52 = np.delete(p_st_52,[k for k in range(x_st_52.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_st_52), axis=0)
y= np.concatenate((y,y_st_52), axis=0)
p= np.concatenate((p,p_st_52), axis=0)
shp=(x_st_53.shape)[0]
shape_y = y_st_53.shape[0]
shape_p =  p_st_53.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_st_53 =  np.delete(x_st_53, [k for k in range(nece_shp,shp)], None)
x_st_53 = x_st_53.reshape(-1,sensors)
shape_x_st_53 = x_st_53.shape[0]
x_st_53 = preprocessing.normalize(x_st_53, axis=0)
x_st_53= np.reshape(x_st_53, (-1,segement_time_size, sensors))
y_st_53= np.delete( y_st_53, [k for k in range(x_st_53.shape[0],shape_y)], None)
p_st_53 = np.delete(p_st_53,[k for k in range(x_st_53.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_st_53), axis=0)
y= np.concatenate((y,y_st_53), axis=0)
p= np.concatenate((p,p_st_53), axis=0)
shp=(x_st_54.shape)[0]
shape_y = y_st_54.shape[0]
shape_p =  p_st_54.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_st_54 =  np.delete(x_st_54, [k for k in range(nece_shp,shp)], None)
x_st_54 = x_st_54.reshape(-1,sensors)
shape_x_st_54 = x_st_54.shape[0]
x_st_54 = preprocessing.normalize(x_st_54, axis=0)
x_st_54= np.reshape(x_st_54, (-1,segement_time_size, sensors))
y_st_54= np.delete( y_st_54, [k for k in range(x_st_54.shape[0],shape_y)], None)
p_st_54 = np.delete(p_st_54,[k for k in range(x_st_54.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_st_54), axis=0)
y= np.concatenate((y,y_st_54), axis=0)
p= np.concatenate((p,p_st_54), axis=0)
shp=(x_st_55.shape)[0]
shape_y = y_st_55.shape[0]
shape_p =  p_st_55.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_st_55 =  np.delete(x_st_55, [k for k in range(nece_shp,shp)], None)
x_st_55 = x_st_55.reshape(-1,sensors)
shape_x_st_55 = x_st_55.shape[0]
x_st_55 = preprocessing.normalize(x_st_55, axis=0)
x_st_55= np.reshape(x_st_55, (-1,segement_time_size, sensors))
y_st_55= np.delete( y_st_55, [k for k in range(x_st_55.shape[0],shape_y)], None)
p_st_55 = np.delete(p_st_55,[k for k in range(x_st_55.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_st_55), axis=0)
y= np.concatenate((y,y_st_55), axis=0)
p= np.concatenate((p,p_st_55), axis=0)
shp=(x_st_56.shape)[0]
shape_y = y_st_56.shape[0]
shape_p =  p_st_56.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_st_56 =  np.delete(x_st_56, [k for k in range(nece_shp,shp)], None)
x_st_56 = x_st_56.reshape(-1,sensors)
shape_x_st_56 = x_st_56.shape[0]
x_st_56 = preprocessing.normalize(x_st_56, axis=0)
x_st_56= np.reshape(x_st_56, (-1,segement_time_size, sensors))
y_st_56= np.delete( y_st_56, [k for k in range(x_st_56.shape[0],shape_y)], None)
p_st_56 = np.delete(p_st_56,[k for k in range(x_st_56.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_st_56), axis=0)
y= np.concatenate((y,y_st_56), axis=0)
p= np.concatenate((p,p_st_56), axis=0)
shp=(x_st_57.shape)[0]
shape_y = y_st_57.shape[0]
shape_p =  p_st_57.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_st_57 =  np.delete(x_st_57, [k for k in range(nece_shp,shp)], None)
x_st_57 = x_st_57.reshape(-1,sensors)
shape_x_st_57 = x_st_57.shape[0]
x_st_57 = preprocessing.normalize(x_st_57, axis=0)
x_st_57= np.reshape(x_st_57, (-1,segement_time_size, sensors))
y_st_57= np.delete( y_st_57, [k for k in range(x_st_57.shape[0],shape_y)], None)
p_st_57 = np.delete(p_st_57,[k for k in range(x_st_57.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_st_57), axis=0)
y= np.concatenate((y,y_st_57), axis=0)
p= np.concatenate((p,p_st_57), axis=0)
shp=(x_st_58.shape)[0]
shape_y = y_st_58.shape[0]
shape_p =  p_st_58.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_st_58 =  np.delete(x_st_58, [k for k in range(nece_shp,shp)], None)
x_st_58 = x_st_58.reshape(-1,sensors)
shape_x_st_58 = x_st_58.shape[0]
x_st_58 = preprocessing.normalize(x_st_58, axis=0)
x_st_58= np.reshape(x_st_58, (-1,segement_time_size, sensors))
y_st_58= np.delete( y_st_58, [k for k in range(x_st_58.shape[0],shape_y)], None)
p_st_58 = np.delete(p_st_58,[k for k in range(x_st_58.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_st_58), axis=0)
y= np.concatenate((y,y_st_58), axis=0)
p= np.concatenate((p,p_st_58), axis=0)
shp=(x_st_59.shape)[0]
shape_y = y_st_59.shape[0]
shape_p =  p_st_59.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_st_59 =  np.delete(x_st_59, [k for k in range(nece_shp,shp)], None)
x_st_59 = x_st_59.reshape(-1,sensors)
shape_x_st_59 = x_st_59.shape[0]
x_st_59 = preprocessing.normalize(x_st_59, axis=0)
x_st_59= np.reshape(x_st_59, (-1,segement_time_size, sensors))
y_st_59= np.delete( y_st_59, [k for k in range(x_st_59.shape[0],shape_y)], None)
p_st_59 = np.delete(p_st_59,[k for k in range(x_st_59.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_st_59), axis=0)
y= np.concatenate((y,y_st_59), axis=0)
p= np.concatenate((p,p_st_59), axis=0)
shp=(x_st_60.shape)[0]
shape_y = y_st_60.shape[0]
shape_p =  p_st_60.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_st_60 =  np.delete(x_st_60, [k for k in range(nece_shp,shp)], None)
x_st_60 = x_st_60.reshape(-1,sensors)
shape_x_st_60 = x_st_60.shape[0]
x_st_60 = preprocessing.normalize(x_st_60, axis=0)
x_st_60= np.reshape(x_st_60, (-1,segement_time_size, sensors))
y_st_60= np.delete( y_st_60, [k for k in range(x_st_60.shape[0],shape_y)], None)
p_st_60 = np.delete(p_st_60,[k for k in range(x_st_60.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_st_60), axis=0)
y= np.concatenate((y,y_st_60), axis=0)
p= np.concatenate((p,p_st_60), axis=0)
shp=(x_st_61.shape)[0]
shape_y = y_st_61.shape[0]
shape_p =  p_st_61.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_st_61 =  np.delete(x_st_61, [k for k in range(nece_shp,shp)], None)
x_st_61 = x_st_61.reshape(-1,sensors)
shape_x_st_61 = x_st_61.shape[0]
x_st_61 = preprocessing.normalize(x_st_61, axis=0)
x_st_61= np.reshape(x_st_61, (-1,segement_time_size, sensors))
y_st_61= np.delete( y_st_61, [k for k in range(x_st_61.shape[0],shape_y)], None)
p_st_61 = np.delete(p_st_61,[k for k in range(x_st_61.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_st_61), axis=0)
y= np.concatenate((y,y_st_61), axis=0)
p= np.concatenate((p,p_st_61), axis=0)
shp=(x_st_62.shape)[0]
shape_y = y_st_62.shape[0]
shape_p =  p_st_62.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_st_62 =  np.delete(x_st_62, [k for k in range(nece_shp,shp)], None)
x_st_62 = x_st_62.reshape(-1,sensors)
shape_x_st_62 = x_st_62.shape[0]
x_st_62 = preprocessing.normalize(x_st_62, axis=0)
x_st_62= np.reshape(x_st_62, (-1,segement_time_size, sensors))
y_st_62= np.delete( y_st_62, [k for k in range(x_st_62.shape[0],shape_y)], None)
p_st_62 = np.delete(p_st_62,[k for k in range(x_st_62.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_st_62), axis=0)
y= np.concatenate((y,y_st_62), axis=0)
p= np.concatenate((p,p_st_62), axis=0)
shp=(x_st_63.shape)[0]
shape_y = y_st_63.shape[0]
shape_p =  p_st_63.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_st_63 =  np.delete(x_st_63, [k for k in range(nece_shp,shp)], None)
x_st_63 = x_st_63.reshape(-1,sensors)
shape_x_st_63 = x_st_63.shape[0]
x_st_63 = preprocessing.normalize(x_st_63, axis=0)
x_st_63= np.reshape(x_st_63, (-1,segement_time_size, sensors))
y_st_63= np.delete( y_st_63, [k for k in range(x_st_63.shape[0],shape_y)], None)
p_st_63 = np.delete(p_st_63,[k for k in range(x_st_63.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_st_63), axis=0)
y= np.concatenate((y,y_st_63), axis=0)
p= np.concatenate((p,p_st_63), axis=0)
shp=(x_st_64.shape)[0]
shape_y = y_st_64.shape[0]
shape_p =  p_st_64.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_st_64 =  np.delete(x_st_64, [k for k in range(nece_shp,shp)], None)
x_st_64 = x_st_64.reshape(-1,sensors)
shape_x_st_64 = x_st_64.shape[0]
x_st_64 = preprocessing.normalize(x_st_64, axis=0)
x_st_64= np.reshape(x_st_64, (-1,segement_time_size, sensors))
y_st_64= np.delete( y_st_64, [k for k in range(x_st_64.shape[0],shape_y)], None)
p_st_64 = np.delete(p_st_64,[k for k in range(x_st_64.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_st_64), axis=0)
y= np.concatenate((y,y_st_64), axis=0)
p= np.concatenate((p,p_st_64), axis=0)
shp=(x_st_65.shape)[0]
shape_y = y_st_65.shape[0]
shape_p =  p_st_65.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_st_65 =  np.delete(x_st_65, [k for k in range(nece_shp,shp)], None)
x_st_65 = x_st_65.reshape(-1,sensors)
shape_x_st_65 = x_st_65.shape[0]
x_st_65 = preprocessing.normalize(x_st_65, axis=0)
x_st_65= np.reshape(x_st_65, (-1,segement_time_size, sensors))
y_st_65= np.delete( y_st_65, [k for k in range(x_st_65.shape[0],shape_y)], None)
p_st_65 = np.delete(p_st_65,[k for k in range(x_st_65.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_st_65), axis=0)
y= np.concatenate((y,y_st_65), axis=0)
p= np.concatenate((p,p_st_65), axis=0)
shp=(x_st_66.shape)[0]
shape_y = y_st_66.shape[0]
shape_p =  p_st_66.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_st_66 =  np.delete(x_st_66, [k for k in range(nece_shp,shp)], None)
x_st_66 = x_st_66.reshape(-1,sensors)
shape_x_st_66 = x_st_66.shape[0]
x_st_66 = preprocessing.normalize(x_st_66, axis=0)
x_st_66= np.reshape(x_st_66, (-1,segement_time_size, sensors))
y_st_66= np.delete( y_st_66, [k for k in range(x_st_66.shape[0],shape_y)], None)
p_st_66 = np.delete(p_st_66,[k for k in range(x_st_66.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_st_66), axis=0)
y= np.concatenate((y,y_st_66), axis=0)
p= np.concatenate((p,p_st_66), axis=0)
shp=(x_st_67.shape)[0]
shape_y = y_st_67.shape[0]
shape_p =  p_st_67.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_st_67 =  np.delete(x_st_67, [k for k in range(nece_shp,shp)], None)
x_st_67 = x_st_67.reshape(-1,sensors)
shape_x_st_67 = x_st_67.shape[0]
x_st_67 = preprocessing.normalize(x_st_67, axis=0)
x_st_67= np.reshape(x_st_67, (-1,segement_time_size, sensors))
y_st_67= np.delete( y_st_67, [k for k in range(x_st_67.shape[0],shape_y)], None)
p_st_67 = np.delete(p_st_67,[k for k in range(x_st_67.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_st_67), axis=0)
y= np.concatenate((y,y_st_67), axis=0)
p= np.concatenate((p,p_st_67), axis=0)
shp=(x_st_68.shape)[0]
shape_y = y_st_68.shape[0]
shape_p =  p_st_68.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_st_68 =  np.delete(x_st_68, [k for k in range(nece_shp,shp)], None)
x_st_68 = x_st_68.reshape(-1,sensors)
shape_x_st_68 = x_st_68.shape[0]
x_st_68 = preprocessing.normalize(x_st_68, axis=0)
x_st_68= np.reshape(x_st_68, (-1,segement_time_size, sensors))
y_st_68= np.delete( y_st_68, [k for k in range(x_st_68.shape[0],shape_y)], None)
p_st_68 = np.delete(p_st_68,[k for k in range(x_st_68.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_st_68), axis=0)
y= np.concatenate((y,y_st_68), axis=0)
p= np.concatenate((p,p_st_68), axis=0)
shp=(x_st_69.shape)[0]
shape_y = y_st_69.shape[0]
shape_p =  p_st_69.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_st_69 =  np.delete(x_st_69, [k for k in range(nece_shp,shp)], None)
x_st_69 = x_st_69.reshape(-1,sensors)
shape_x_st_69 = x_st_69.shape[0]
x_st_69 = preprocessing.normalize(x_st_69, axis=0)
x_st_69= np.reshape(x_st_69, (-1,segement_time_size, sensors))
y_st_69= np.delete( y_st_69, [k for k in range(x_st_69.shape[0],shape_y)], None)
p_st_69 = np.delete(p_st_69,[k for k in range(x_st_69.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_st_69), axis=0)
y= np.concatenate((y,y_st_69), axis=0)
p= np.concatenate((p,p_st_69), axis=0)
shp=(x_st_70.shape)[0]
shape_y = y_st_70.shape[0]
shape_p =  p_st_70.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_st_70 =  np.delete(x_st_70, [k for k in range(nece_shp,shp)], None)
x_st_70 = x_st_70.reshape(-1,sensors)
shape_x_st_70 = x_st_70.shape[0]
x_st_70 = preprocessing.normalize(x_st_70, axis=0)
x_st_70= np.reshape(x_st_70, (-1,segement_time_size, sensors))
y_st_70= np.delete( y_st_70, [k for k in range(x_st_70.shape[0],shape_y)], None)
p_st_70 = np.delete(p_st_70,[k for k in range(x_st_70.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_st_70), axis=0)
y= np.concatenate((y,y_st_70), axis=0)
p= np.concatenate((p,p_st_70), axis=0)
shp=(x_st_71.shape)[0]
shape_y = y_st_71.shape[0]
shape_p =  p_st_71.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_st_71 =  np.delete(x_st_71, [k for k in range(nece_shp,shp)], None)
x_st_71 = x_st_71.reshape(-1,sensors)
shape_x_st_71 = x_st_71.shape[0]
x_st_71 = preprocessing.normalize(x_st_71, axis=0)
x_st_71= np.reshape(x_st_71, (-1,segement_time_size, sensors))
y_st_71= np.delete( y_st_71, [k for k in range(x_st_71.shape[0],shape_y)], None)
p_st_71 = np.delete(p_st_71,[k for k in range(x_st_71.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_st_71), axis=0)
y= np.concatenate((y,y_st_71), axis=0)
p= np.concatenate((p,p_st_71), axis=0)
shp=(x_st_72.shape)[0]
shape_y = y_st_72.shape[0]
shape_p =  p_st_72.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_st_72 =  np.delete(x_st_72, [k for k in range(nece_shp,shp)], None)
x_st_72 = x_st_72.reshape(-1,sensors)
shape_x_st_72 = x_st_72.shape[0]
x_st_72 = preprocessing.normalize(x_st_72, axis=0)
x_st_72= np.reshape(x_st_72, (-1,segement_time_size, sensors))
y_st_72= np.delete( y_st_72, [k for k in range(x_st_72.shape[0],shape_y)], None)
p_st_72 = np.delete(p_st_72,[k for k in range(x_st_72.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_st_72), axis=0)
y= np.concatenate((y,y_st_72), axis=0)
p= np.concatenate((p,p_st_72), axis=0)
shp=(x_st_73.shape)[0]
shape_y = y_st_73.shape[0]
shape_p =  p_st_73.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_st_73 =  np.delete(x_st_73, [k for k in range(nece_shp,shp)], None)
x_st_73 = x_st_73.reshape(-1,sensors)
shape_x_st_73 = x_st_73.shape[0]
x_st_73 = preprocessing.normalize(x_st_73, axis=0)
x_st_73= np.reshape(x_st_73, (-1,segement_time_size, sensors))
y_st_73= np.delete( y_st_73, [k for k in range(x_st_73.shape[0],shape_y)], None)
p_st_73 = np.delete(p_st_73,[k for k in range(x_st_73.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_st_73), axis=0)
y= np.concatenate((y,y_st_73), axis=0)
p= np.concatenate((p,p_st_73), axis=0)
shp=(x_st_74.shape)[0]
shape_y = y_st_74.shape[0]
shape_p =  p_st_74.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_st_74 =  np.delete(x_st_74, [k for k in range(nece_shp,shp)], None)
x_st_74 = x_st_74.reshape(-1,sensors)
shape_x_st_74 = x_st_74.shape[0]
x_st_74 = preprocessing.normalize(x_st_74, axis=0)
x_st_74= np.reshape(x_st_74, (-1,segement_time_size, sensors))
y_st_74= np.delete( y_st_74, [k for k in range(x_st_74.shape[0],shape_y)], None)
p_st_74 = np.delete(p_st_74,[k for k in range(x_st_74.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_st_74), axis=0)
y= np.concatenate((y,y_st_74), axis=0)
p= np.concatenate((p,p_st_74), axis=0)
shp=(x_st_75.shape)[0]
shape_y = y_st_75.shape[0]
shape_p =  p_st_75.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_st_75 =  np.delete(x_st_75, [k for k in range(nece_shp,shp)], None)
x_st_75 = x_st_75.reshape(-1,sensors)
shape_x_st_75 = x_st_75.shape[0]
x_st_75 = preprocessing.normalize(x_st_75, axis=0)
x_st_75= np.reshape(x_st_75, (-1,segement_time_size, sensors))
y_st_75= np.delete( y_st_75, [k for k in range(x_st_75.shape[0],shape_y)], None)
p_st_75 = np.delete(p_st_75,[k for k in range(x_st_75.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_st_75), axis=0)
y= np.concatenate((y,y_st_75), axis=0)
p= np.concatenate((p,p_st_75), axis=0)
shp=(x_st_76.shape)[0]
shape_y = y_st_76.shape[0]
shape_p =  p_st_76.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_st_76 =  np.delete(x_st_76, [k for k in range(nece_shp,shp)], None)
x_st_76 = x_st_76.reshape(-1,sensors)
shape_x_st_76 = x_st_76.shape[0]
x_st_76 = preprocessing.normalize(x_st_76, axis=0)
x_st_76= np.reshape(x_st_76, (-1,segement_time_size, sensors))
y_st_76= np.delete( y_st_76, [k for k in range(x_st_76.shape[0],shape_y)], None)
p_st_76 = np.delete(p_st_76,[k for k in range(x_st_76.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_st_76), axis=0)
y= np.concatenate((y,y_st_76), axis=0)
p= np.concatenate((p,p_st_76), axis=0)
shp=(x_st_77.shape)[0]
shape_y = y_st_77.shape[0]
shape_p =  p_st_77.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_st_77 =  np.delete(x_st_77, [k for k in range(nece_shp,shp)], None)
x_st_77 = x_st_77.reshape(-1,sensors)
shape_x_st_77 = x_st_77.shape[0]
x_st_77 = preprocessing.normalize(x_st_77, axis=0)
x_st_77= np.reshape(x_st_77, (-1,segement_time_size, sensors))
y_st_77= np.delete( y_st_77, [k for k in range(x_st_77.shape[0],shape_y)], None)
p_st_77 = np.delete(p_st_77,[k for k in range(x_st_77.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_st_77), axis=0)
y= np.concatenate((y,y_st_77), axis=0)
p= np.concatenate((p,p_st_77), axis=0)
shp=(x_st_78.shape)[0]
shape_y = y_st_78.shape[0]
shape_p =  p_st_78.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_st_78 =  np.delete(x_st_78, [k for k in range(nece_shp,shp)], None)
x_st_78 = x_st_78.reshape(-1,sensors)
shape_x_st_78 = x_st_78.shape[0]
x_st_78 = preprocessing.normalize(x_st_78, axis=0)
x_st_78= np.reshape(x_st_78, (-1,segement_time_size, sensors))
y_st_78= np.delete( y_st_78, [k for k in range(x_st_78.shape[0],shape_y)], None)
p_st_78 = np.delete(p_st_78,[k for k in range(x_st_78.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_st_78), axis=0)
y= np.concatenate((y,y_st_78), axis=0)
p= np.concatenate((p,p_st_78), axis=0)
shp=(x_st_79.shape)[0]
shape_y = y_st_79.shape[0]
shape_p =  p_st_79.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_st_79 =  np.delete(x_st_79, [k for k in range(nece_shp,shp)], None)
x_st_79 = x_st_79.reshape(-1,sensors)
shape_x_st_79 = x_st_79.shape[0]
x_st_79 = preprocessing.normalize(x_st_79, axis=0)
x_st_79= np.reshape(x_st_79, (-1,segement_time_size, sensors))
y_st_79= np.delete( y_st_79, [k for k in range(x_st_79.shape[0],shape_y)], None)
p_st_79 = np.delete(p_st_79,[k for k in range(x_st_79.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_st_79), axis=0)
y= np.concatenate((y,y_st_79), axis=0)
p= np.concatenate((p,p_st_79), axis=0)
shp=(x_st_80.shape)[0]
shape_y = y_st_80.shape[0]
shape_p =  p_st_80.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_st_80 =  np.delete(x_st_80, [k for k in range(nece_shp,shp)], None)
x_st_80 = x_st_80.reshape(-1,sensors)
shape_x_st_80 = x_st_80.shape[0]
x_st_80 = preprocessing.normalize(x_st_80, axis=0)
x_st_80= np.reshape(x_st_80, (-1,segement_time_size, sensors))
y_st_80= np.delete( y_st_80, [k for k in range(x_st_80.shape[0],shape_y)], None)
p_st_80 = np.delete(p_st_80,[k for k in range(x_st_80.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_st_80), axis=0)
y= np.concatenate((y,y_st_80), axis=0)
p= np.concatenate((p,p_st_80), axis=0)
shp=(x_st_81.shape)[0]
shape_y = y_st_81.shape[0]
shape_p =  p_st_81.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_st_81 =  np.delete(x_st_81, [k for k in range(nece_shp,shp)], None)
x_st_81 = x_st_81.reshape(-1,sensors)
shape_x_st_81 = x_st_81.shape[0]
x_st_81 = preprocessing.normalize(x_st_81, axis=0)
x_st_81= np.reshape(x_st_81, (-1,segement_time_size, sensors))
y_st_81= np.delete( y_st_81, [k for k in range(x_st_81.shape[0],shape_y)], None)
p_st_81 = np.delete(p_st_81,[k for k in range(x_st_81.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_st_81), axis=0)
y= np.concatenate((y,y_st_81), axis=0)
p= np.concatenate((p,p_st_81), axis=0)
shp=(x_st_82.shape)[0]
shape_y = y_st_82.shape[0]
shape_p =  p_st_82.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_st_82 =  np.delete(x_st_82, [k for k in range(nece_shp,shp)], None)
x_st_82 = x_st_82.reshape(-1,sensors)
shape_x_st_82 = x_st_82.shape[0]
x_st_82 = preprocessing.normalize(x_st_82, axis=0)
x_st_82= np.reshape(x_st_82, (-1,segement_time_size, sensors))
y_st_82= np.delete( y_st_82, [k for k in range(x_st_82.shape[0],shape_y)], None)
p_st_82 = np.delete(p_st_82,[k for k in range(x_st_82.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_st_82), axis=0)
y= np.concatenate((y,y_st_82), axis=0)
p= np.concatenate((p,p_st_82), axis=0)
shp=(x_st_83.shape)[0]
shape_y = y_st_83.shape[0]
shape_p =  p_st_83.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_st_83 =  np.delete(x_st_83, [k for k in range(nece_shp,shp)], None)
x_st_83 = x_st_83.reshape(-1,sensors)
shape_x_st_83 = x_st_83.shape[0]
x_st_83 = preprocessing.normalize(x_st_83, axis=0)
x_st_83= np.reshape(x_st_83, (-1,segement_time_size, sensors))
y_st_83= np.delete( y_st_83, [k for k in range(x_st_83.shape[0],shape_y)], None)
p_st_83 = np.delete(p_st_83,[k for k in range(x_st_83.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_st_83), axis=0)
y= np.concatenate((y,y_st_83), axis=0)
p= np.concatenate((p,p_st_83), axis=0)
shp=(x_st_84.shape)[0]
shape_y = y_st_84.shape[0]
shape_p =  p_st_84.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_st_84 =  np.delete(x_st_84, [k for k in range(nece_shp,shp)], None)
x_st_84 = x_st_84.reshape(-1,sensors)
shape_x_st_84 = x_st_84.shape[0]
x_st_84 = preprocessing.normalize(x_st_84, axis=0)
x_st_84= np.reshape(x_st_84, (-1,segement_time_size, sensors))
y_st_84= np.delete( y_st_84, [k for k in range(x_st_84.shape[0],shape_y)], None)
p_st_84 = np.delete(p_st_84,[k for k in range(x_st_84.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_st_84), axis=0)
y= np.concatenate((y,y_st_84), axis=0)
p= np.concatenate((p,p_st_84), axis=0)
shp=(x_st_85.shape)[0]
shape_y = y_st_85.shape[0]
shape_p =  p_st_85.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_st_85 =  np.delete(x_st_85, [k for k in range(nece_shp,shp)], None)
x_st_85 = x_st_85.reshape(-1,sensors)
shape_x_st_85 = x_st_85.shape[0]
x_st_85 = preprocessing.normalize(x_st_85, axis=0)
x_st_85= np.reshape(x_st_85, (-1,segement_time_size, sensors))
y_st_85= np.delete( y_st_85, [k for k in range(x_st_85.shape[0],shape_y)], None)
p_st_85 = np.delete(p_st_85,[k for k in range(x_st_85.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_st_85), axis=0)
y= np.concatenate((y,y_st_85), axis=0)
p= np.concatenate((p,p_st_85), axis=0)
shp=(x_st_86.shape)[0]
shape_y = y_st_86.shape[0]
shape_p =  p_st_86.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_st_86 =  np.delete(x_st_86, [k for k in range(nece_shp,shp)], None)
x_st_86 = x_st_86.reshape(-1,sensors)
shape_x_st_86 = x_st_86.shape[0]
x_st_86 = preprocessing.normalize(x_st_86, axis=0)
x_st_86= np.reshape(x_st_86, (-1,segement_time_size, sensors))
y_st_86= np.delete( y_st_86, [k for k in range(x_st_86.shape[0],shape_y)], None)
p_st_86 = np.delete(p_st_86,[k for k in range(x_st_86.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_st_86), axis=0)
y= np.concatenate((y,y_st_86), axis=0)
p= np.concatenate((p,p_st_86), axis=0)
shp=(x_st_87.shape)[0]
shape_y = y_st_87.shape[0]
shape_p =  p_st_87.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_st_87 =  np.delete(x_st_87, [k for k in range(nece_shp,shp)], None)
x_st_87 = x_st_87.reshape(-1,sensors)
shape_x_st_87 = x_st_87.shape[0]
x_st_87 = preprocessing.normalize(x_st_87, axis=0)
x_st_87= np.reshape(x_st_87, (-1,segement_time_size, sensors))
y_st_87= np.delete( y_st_87, [k for k in range(x_st_87.shape[0],shape_y)], None)
p_st_87 = np.delete(p_st_87,[k for k in range(x_st_87.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_st_87), axis=0)
y= np.concatenate((y,y_st_87), axis=0)
p= np.concatenate((p,p_st_87), axis=0)
shp=(x_st_88.shape)[0]
shape_y = y_st_88.shape[0]
shape_p =  p_st_88.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_st_88 =  np.delete(x_st_88, [k for k in range(nece_shp,shp)], None)
x_st_88 = x_st_88.reshape(-1,sensors)
shape_x_st_88 = x_st_88.shape[0]
x_st_88 = preprocessing.normalize(x_st_88, axis=0)
x_st_88= np.reshape(x_st_88, (-1,segement_time_size, sensors))
y_st_88= np.delete( y_st_88, [k for k in range(x_st_88.shape[0],shape_y)], None)
p_st_88 = np.delete(p_st_88,[k for k in range(x_st_88.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_st_88), axis=0)
y= np.concatenate((y,y_st_88), axis=0)
p= np.concatenate((p,p_st_88), axis=0)
shp=(x_st_89.shape)[0]
shape_y = y_st_89.shape[0]
shape_p =  p_st_89.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_st_89 =  np.delete(x_st_89, [k for k in range(nece_shp,shp)], None)
x_st_89 = x_st_89.reshape(-1,sensors)
shape_x_st_89 = x_st_89.shape[0]
x_st_89 = preprocessing.normalize(x_st_89, axis=0)
x_st_89= np.reshape(x_st_89, (-1,segement_time_size, sensors))
y_st_89= np.delete( y_st_89, [k for k in range(x_st_89.shape[0],shape_y)], None)
p_st_89 = np.delete(p_st_89,[k for k in range(x_st_89.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_st_89), axis=0)
y= np.concatenate((y,y_st_89), axis=0)
p= np.concatenate((p,p_st_89), axis=0)
shp=(x_st_90.shape)[0]
shape_y = y_st_90.shape[0]
shape_p =  p_st_90.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_st_90 =  np.delete(x_st_90, [k for k in range(nece_shp,shp)], None)
x_st_90 = x_st_90.reshape(-1,sensors)
shape_x_st_90 = x_st_90.shape[0]
x_st_90 = preprocessing.normalize(x_st_90, axis=0)
x_st_90= np.reshape(x_st_90, (-1,segement_time_size, sensors))
y_st_90= np.delete( y_st_90, [k for k in range(x_st_90.shape[0],shape_y)], None)
p_st_90 = np.delete(p_st_90,[k for k in range(x_st_90.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_st_90), axis=0)
y= np.concatenate((y,y_st_90), axis=0)
p= np.concatenate((p,p_st_90), axis=0)
shp=(x_st_91.shape)[0]
shape_y = y_st_91.shape[0]
shape_p =  p_st_91.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_st_91 =  np.delete(x_st_91, [k for k in range(nece_shp,shp)], None)
x_st_91 = x_st_91.reshape(-1,sensors)
shape_x_st_91 = x_st_91.shape[0]
x_st_91 = preprocessing.normalize(x_st_91, axis=0)
x_st_91= np.reshape(x_st_91, (-1,segement_time_size, sensors))
y_st_91= np.delete( y_st_91, [k for k in range(x_st_91.shape[0],shape_y)], None)
p_st_91 = np.delete(p_st_91,[k for k in range(x_st_91.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_st_91), axis=0)
y= np.concatenate((y,y_st_91), axis=0)
p= np.concatenate((p,p_st_91), axis=0)
shp=(x_st_92.shape)[0]
shape_y = y_st_92.shape[0]
shape_p =  p_st_92.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_st_92 =  np.delete(x_st_92, [k for k in range(nece_shp,shp)], None)
x_st_92 = x_st_92.reshape(-1,sensors)
shape_x_st_92 = x_st_92.shape[0]
x_st_92 = preprocessing.normalize(x_st_92, axis=0)
x_st_92= np.reshape(x_st_92, (-1,segement_time_size, sensors))
y_st_92= np.delete( y_st_92, [k for k in range(x_st_92.shape[0],shape_y)], None)
p_st_92 = np.delete(p_st_92,[k for k in range(x_st_92.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_st_92), axis=0)
y= np.concatenate((y,y_st_92), axis=0)
p= np.concatenate((p,p_st_92), axis=0)
shp=(x_st_93.shape)[0]
shape_y = y_st_93.shape[0]
shape_p =  p_st_93.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_st_93 =  np.delete(x_st_93, [k for k in range(nece_shp,shp)], None)
x_st_93 = x_st_93.reshape(-1,sensors)
shape_x_st_93 = x_st_93.shape[0]
x_st_93 = preprocessing.normalize(x_st_93, axis=0)
x_st_93= np.reshape(x_st_93, (-1,segement_time_size, sensors))
y_st_93= np.delete( y_st_93, [k for k in range(x_st_93.shape[0],shape_y)], None)
p_st_93 = np.delete(p_st_93,[k for k in range(x_st_93.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_st_93), axis=0)
y= np.concatenate((y,y_st_93), axis=0)
p= np.concatenate((p,p_st_93), axis=0)
shp=(x_st_94.shape)[0]
shape_y = y_st_94.shape[0]
shape_p =  p_st_94.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_st_94 =  np.delete(x_st_94, [k for k in range(nece_shp,shp)], None)
x_st_94 = x_st_94.reshape(-1,sensors)
shape_x_st_94 = x_st_94.shape[0]
x_st_94 = preprocessing.normalize(x_st_94, axis=0)
x_st_94= np.reshape(x_st_94, (-1,segement_time_size, sensors))
y_st_94= np.delete( y_st_94, [k for k in range(x_st_94.shape[0],shape_y)], None)
p_st_94 = np.delete(p_st_94,[k for k in range(x_st_94.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_st_94), axis=0)
y= np.concatenate((y,y_st_94), axis=0)
p= np.concatenate((p,p_st_94), axis=0)
shp=(x_st_95.shape)[0]
shape_y = y_st_95.shape[0]
shape_p =  p_st_95.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_st_95 =  np.delete(x_st_95, [k for k in range(nece_shp,shp)], None)
x_st_95 = x_st_95.reshape(-1,sensors)
shape_x_st_95 = x_st_95.shape[0]
x_st_95 = preprocessing.normalize(x_st_95, axis=0)
x_st_95= np.reshape(x_st_95, (-1,segement_time_size, sensors))
y_st_95= np.delete( y_st_95, [k for k in range(x_st_95.shape[0],shape_y)], None)
p_st_95 = np.delete(p_st_95,[k for k in range(x_st_95.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_st_95), axis=0)
y= np.concatenate((y,y_st_95), axis=0)
p= np.concatenate((p,p_st_95), axis=0)
shp=(x_st_96.shape)[0]
shape_y = y_st_96.shape[0]
shape_p =  p_st_96.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_st_96 =  np.delete(x_st_96, [k for k in range(nece_shp,shp)], None)
x_st_96 = x_st_96.reshape(-1,sensors)
shape_x_st_96 = x_st_96.shape[0]
x_st_96 = preprocessing.normalize(x_st_96, axis=0)
x_st_96= np.reshape(x_st_96, (-1,segement_time_size, sensors))
y_st_96= np.delete( y_st_96, [k for k in range(x_st_96.shape[0],shape_y)], None)
p_st_96 = np.delete(p_st_96,[k for k in range(x_st_96.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_st_96), axis=0)
y= np.concatenate((y,y_st_96), axis=0)
p= np.concatenate((p,p_st_96), axis=0)
shp=(x_st_97.shape)[0]
shape_y = y_st_97.shape[0]
shape_p =  p_st_97.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_st_97 =  np.delete(x_st_97, [k for k in range(nece_shp,shp)], None)
x_st_97 = x_st_97.reshape(-1,sensors)
shape_x_st_97 = x_st_97.shape[0]
x_st_97 = preprocessing.normalize(x_st_97, axis=0)
x_st_97= np.reshape(x_st_97, (-1,segement_time_size, sensors))
y_st_97= np.delete( y_st_97, [k for k in range(x_st_97.shape[0],shape_y)], None)
p_st_97 = np.delete(p_st_97,[k for k in range(x_st_97.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_st_97), axis=0)
y= np.concatenate((y,y_st_97), axis=0)
p= np.concatenate((p,p_st_97), axis=0)
shp=(x_st_98.shape)[0]
shape_y = y_st_98.shape[0]
shape_p =  p_st_98.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_st_98 =  np.delete(x_st_98, [k for k in range(nece_shp,shp)], None)
x_st_98 = x_st_98.reshape(-1,sensors)
shape_x_st_98 = x_st_98.shape[0]
x_st_98 = preprocessing.normalize(x_st_98, axis=0)
x_st_98= np.reshape(x_st_98, (-1,segement_time_size, sensors))
y_st_98= np.delete( y_st_98, [k for k in range(x_st_98.shape[0],shape_y)], None)
p_st_98 = np.delete(p_st_98,[k for k in range(x_st_98.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_st_98), axis=0)
y= np.concatenate((y,y_st_98), axis=0)
p= np.concatenate((p,p_st_98), axis=0)
shp=(x_st_99.shape)[0]
shape_y = y_st_99.shape[0]
shape_p =  p_st_99.shape[0]
nece_shp = biggest_multiple((segement_time_size*sensors),shp)
if shp >= nece_shp:
    x_st_99 =  np.delete(x_st_99, [k for k in range(nece_shp,shp)], None)
x_st_99 = x_st_99.reshape(-1,sensors)
shape_x_st_99 = x_st_99.shape[0]
x_st_99 = preprocessing.normalize(x_st_99, axis=0)
x_st_99= np.reshape(x_st_99, (-1,segement_time_size, sensors))
y_st_99= np.delete( y_st_99, [k for k in range(x_st_99.shape[0],shape_y)], None)
p_st_99 = np.delete(p_st_99,[k for k in range(x_st_99.shape[0],shape_p)], None)
x_j_0 = np.concatenate((x_j_0, x_st_99), axis=0)
y= np.concatenate((y,y_st_99), axis=0)
p= np.concatenate((p,p_st_99), axis=0)
print(y.shape, y)
y_onehot= to_categorical(np.array(y))
p_onehot = to_categorical(np.array(p))
h5f = h5py.File('RL_acc_hasc_data_a_p.h5', 'w')
h5f.create_dataset('X', data=x_j_0)
h5f.create_dataset('y', data=y)
h5f.create_dataset('p', data=p)
h5f.create_dataset('y_onehot', data=y_onehot)
h5f.create_dataset('p_onehot', data=p_onehot)
h5f.close()



"""

    
    x= np.array(x)
    y= np.array(y)
    p= np.array(p)
    shape_y = y.shape[0]
    shape_p = p.shape[0]
    shp=(x.shape)[0]
    nece_shp = biggest_multiple((segement_time_size*sensors),shp)
    if shp >= nece_shp:
        x =   np.delete(x, [k for k in range(nece_shp,shp)], None)
    x = x.reshape(-1,sensors)
    shape_x = x.shape[0]
    x = preprocessing.normalize(x, axis=0)
    y = np.delete(y, [k for k in range(shape_x,shape_y)], None)
    p = np.delete(p,[k for k in range(shape_x,shape_p)], None)
    x = np.reshape(x, (-1,segement_time_size, np.shape(x)[1]))
    if start==0:
        all_seqs = x
        start=1
    else:
        all_seqs = np.concatenate((all_seqs,x), axis=0)
    return all_seqs, np.asarray(y), to_categorical(np.asarray(y)),np.asarray(p), to_categorical(np.asarray(p))

x, y, y_onehot, p, p_onehot = create_segmented_data_a()

#once all of the data has been read in and the files have been appended, save
h5f = h5py.File('RL_acc_hasc_data.h5', 'w')
h5f.create_dataset('X', data=x)
h5f.create_dataset('y', data=y)
h5f.create_dataset('p', data=p)
h5f.create_dataset('y_onehot', data=y_onehot)
h5f.create_dataset('p_onehot', data=p_onehot)
h5f.close()

	
# WISDM
path =r'F:\PHD\GAN\code.12.2020\PGAN1\WISDM_data.csv'
segement_time_size = 180
sensors = 3
mvts = ["Downstairs","Jogging","Sitting","Standing","Upstairs","Walking"]
id = [i for i in range(1,37)]

def biggest_multiple(multiple_of, input_number):
    return input_number - input_number % multiple_of

def create_segmented_data_a():
    global all_seqs
    start=0
    x = []
    y = []
    p = []
    csvfile= open(path,'r')
    readf =  csv.reader(csvfile)
    for row in readf:
        if row:
            if row[2]!=0:
                x1=row[3]
                x.append(x1)
                x2= row[4]
                x.append(x2)
                x3= row[5]
                x.append(x3)
                y.append(mvts.index(row[1]))
                p.append(id.index(int(row[0])))
    x= np.array(x)
    y= np.array(y)
    p= np.array(p)
    shape_y = y.shape[0]
    shape_p = p.shape[0]
    shp=(x.shape)[0]
    nece_shp = biggest_multiple((segement_time_size*sensors),shp)
    if shp >= nece_shp:
        x =   np.delete(x, [k for k in range(nece_shp,shp)], None)
    x = x.reshape(-1,sensors)
    shape_x = x.shape[0]
    x = preprocessing.normalize(x, axis=0)
    y = np.delete(y, [k for k in range(shape_x,shape_y)], None)
    p = np.delete(p,[k for k in range(shape_x,shape_p)], None)
    x = np.reshape(x, (-1,segement_time_size, np.shape(x)[1]))
    if start==0:
        all_seqs = x
        start=1        
    else:
        all_seqs = np.concatenate((all_seqs,x), axis=0)
    return all_seqs, np.asarray(y), to_categorical(np.asarray(y)),np.asarray(p), to_categorical(np.asarray(p))

x, y, y_onehot, p, p_onehot = create_segmented_data_a()

#once all of the data has been read in and the files have been appended, save
h5f = h5py.File('RL_acc_wisdm_data.h5', 'w')
h5f.create_dataset('X', data=x)
h5f.create_dataset('y', data=y)
h5f.create_dataset('p', data=p)
h5f.create_dataset('y_onehot', data=y_onehot)
h5f.create_dataset('p_onehot', data=p_onehot)
h5f.close()


#UCI

# load dataset

# load a single file as a numpy array

def load_y(y_path):
    file = open(y_path, 'r')
    # Read dataset from disk, dealing with text file's syntax
    y_ = np.array(
        [elem for elem in [
            row.replace('  ', ' ').strip().split(' ') for row in file
        ]],
        dtype=np.int32
    )
    file.close()

    # Substract 1 to each output class for friendly 0-based indexing
    return y_ - 1

def load_file(filepath):
    dataframe = read_csv(filepath, header=None, delim_whitespace=True)
    return dataframe.values

# load a list of files, such as x, y, z data for a given variable
def load_group(filenames, prefix=''):
	loaded = list()
	for name in filenames:
		data = load_file(prefix + name)
		loaded.append(data)
	# stack group so that features are the 3rd dimension
	loaded = dstack(loaded)
	return loaded

# load a dataset group, such as train or test
def load_dataset(group, prefix=''):
	filepath = prefix + group + '/Inertial Signals/'
	# load all 9 files as a single array
	filenames = list()
	# total acceleration
	filenames += ['total_acc_x_'+group+'.txt', 'total_acc_y_'+group+'.txt', 'total_acc_z_'+group+'.txt']
	# body acceleration
	filenames += ['body_acc_x_'+group+'.txt', 'body_acc_y_'+group+'.txt', 'body_acc_z_'+group+'.txt']
	# body gyroscope
	filenames += ['body_gyro_x_'+group+'.txt', 'body_gyro_y_'+group+'.txt', 'body_gyro_z_'+group+'.txt']
	# load input data
	X = load_group(filenames, filepath)
	# load class output
	y = load_y(prefix + group + '/y_'+group+'.txt')
	p = load_y(prefix + group + '/subject_'+group+'.txt')
	return X, y, p

def one_hot_func(y):
	shape = (y.size, y.max()+1)
	one_hot = np.zeros(shape)
	rows = np.arange(y.size)
	one_hot[rows, y] = 1
	return one_hot


# load all train
trainX, trainy, trainp = load_dataset('train', 'UCI_HAR_Dataset/')
trainy = trainy.reshape(-1)
trainp = trainp.reshape(-1)

#once all of the data has been read in and the files have been appended, save

# load all test
testX, testy, testp = load_dataset('test', 'UCI_HAR_Dataset/')
testy = testy.reshape(-1)
testp = testp.reshape(-1)


X = np.concatenate([trainX,testX],0)
Y = np.concatenate([trainy, testy])
P = np.concatenate([trainp, testp])
y_onehot= one_hot_func(Y)
p_onehot= one_hot_func(P)

h5f = h5py.File('RL_acc_uci_data.h5', 'w')
h5f.create_dataset('X', data=X)
h5f.create_dataset('y', data=Y)
h5f.create_dataset('p', data=P)
h5f.create_dataset('y_onehot', data=y_onehot)
h5f.create_dataset('p_onehot', data=p_onehot)
h5f.close()
print(X.shape, Y.shape)
print(y_onehot.shape)
print(p_onehot.shape)

"""
