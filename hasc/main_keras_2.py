from sklearn.model_selection import train_test_split
import numpy as np
from keras.models import Sequential, Model
from keras.layers import Dense, LSTM, Dropout, Input, Conv1D, MaxPooling1D
import tensorflow as tf
import sys
import pandas as pd
import h5py

np.set_printoptions(threshold=sys.maxsize)

path = r'F:\PHD\GAN\code.12.2020\PGAN2\RL\RL_acc_uci_data.h5'
dataset = 'uci'

# - - - get data parameters and split - - - #

hf = h5py.File(path, 'r')
X_source = np.array(hf.get('X'))
y_source = np.array(hf.get('y'))
p_source = np.array(hf.get('p'))
y_onehot_source = np.array(hf.get('y_onehot'))
p_onehot_source = np.array(hf.get('p_onehot'))
print("X source shape: {}".format(X_source.shape))
print("Y source shape: {}".format(y_source.shape))
print("P source shape: {}".format(p_source.shape))

# VARIABLES REGARDING DATA SHAPE, TRAINING

seq_length = X_source.shape[1]
num_channels = X_source.shape[2]
input_shape = (seq_length, num_channels)
num_classes_a = y_onehot_source.shape[1]
num_classes_p = p_onehot_source.shape[1]

# SPLIT INTO TRAINING AND VALIDATION SETS

X_train, X_test, y_train, y_test = train_test_split(X_source, y_source, test_size=0.7, random_state=42)
p_train, p_test, p_train_onehot, p_test_onehot = train_test_split(p_source, p_onehot_source, test_size=0.7, random_state=42)
y_train_onehot, y_test_onehot = train_test_split(y_onehot_source, test_size=0.7, random_state=42)

# SPLIT INTO TRAINING AND VALIDATION SETS
#
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.35, random_state=42)
p_train, p_test, p_train_onehot, p_test_onehot = train_test_split(p_train, p_train_onehot, test_size=0.35, random_state=42)
y_train_onehot, y_test_onehot = train_test_split(y_train_onehot, test_size=0.35, random_state=42)

latent_dim = num_channels #length of random input fed to generator
batch_size_train = y_train.shape[0] #num instances generated for G/D training
batch_size_test = y_test.shape[0]
disc_lr = .01 #learning rate of discriminator
epoch = 0
vis_freq =10 #100
print(batch_size_train)
print(batch_size_test)
print(y_test_onehot.shape, y_train_onehot.shape, p_test_onehot.shape, p_train_onehot.shape)
#WEIGHTS FOR DIFFERENT TERMS IN THE LOSS FUNCTION

D_p_loss_weight = 1
D_a_loss_weight = 1

a_accuracy = 0.2
p_accuracy = 0.2

# - - - build the loss function - - - #

def categorical_crossentropy_f():
    
    return 

# - - - build the model - - - #
def loss2():
    
    return 

def generator(seq_length, latent_dim):
    G_input_shape = (seq_length, latent_dim)
    G_in = Input(shape=G_input_shape)
    G = Dropout(.5)(G_in)
    G = LSTM(128, return_sequences=True, activation="tanh")(G)#uci 128
    G = Dropout(.5)(G)
    G = Dense(latent_dim, activation="tanh")(G)
    G = Model(inputs=G_in, outputs=G)
    G.summary()
    return G

def descriminatorA(input_shape, num_classes):
    disc = Sequential(name="D_a")
    disc.add(Conv1D(filters=128, kernel_size=(5), padding="same", input_shape=input_shape))#uci 128
    disc.add(MaxPooling1D(pool_size=(1)))
    disc.add(Conv1D(filters=96, use_bias="true", kernel_size=(5), padding="same", activation="relu"))#uci 96
    disc.add(Conv1D(filters=64, kernel_size=(5), padding="same", activation="relu"))#uci 64
    disc.add(Conv1D(filters=48, kernel_size=(5), padding="same", activation="relu"))#uci 48
    disc.add(MaxPooling1D(pool_size=(1)))
    disc.add(Conv1D(filters=32, kernel_size=(5), padding="same", activation='relu'))#uci 32
    disc.add(MaxPooling1D(pool_size=(1)))
    disc.add(Dropout(0.5))
    disc.add(LSTM(128, return_sequences=True,activation="tanh"))#uci 128
    disc.add(Dropout(0.5))
    disc.add(LSTM(128, activation="tanh"))#uci 128
    disc.add(Dropout(0.5))
    disc.add(Dense(num_classes, activation="softmax"))

    # TRAINING PARAMETERS
    disc.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    disc.summary()
    return disc

def descriminatorP(input_shape, num_classes):
    classifier = Sequential(name="D_p")
    classifier.add(Conv1D(filters=128, kernel_size=(5), padding="same", input_shape=input_shape))#uci 128
    classifier.add(MaxPooling1D(pool_size=(1)))
    classifier.add(Conv1D(filters=96, use_bias="true", kernel_size=(5), padding="same", activation="relu"))#uci 96
    classifier.add(Conv1D(filters=64, kernel_size=(5), padding="same", activation="relu"))#uci 64
    classifier.add(Conv1D(filters=48, kernel_size=(5), padding="same", activation="relu"))#uci 48
    classifier.add(MaxPooling1D(pool_size=(1)))
    classifier.add(Conv1D(filters=32, kernel_size=(5), padding="same", activation='relu'))#uci 32
    classifier.add(MaxPooling1D(pool_size=(1)))
    classifier.add(Dropout(0.5))
    classifier.add(LSTM(128, return_sequences=True,activation="tanh"))
    classifier.add(Dropout(0.5))
    classifier.add(LSTM(128, activation="tanh"))
    classifier.add(Dropout(0.5))
    classifier.add(Dense(num_classes, activation="softmax"))

    # TRAINING PARAMETERS
    classifier.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    classifier.summary()
    return classifier


def create_gan(D_a, D_p, G):
    GDD = Model(inputs=G.input, outputs=[D_a(G.output), D_p(G.output)])
    GDD.compile(loss={"D_a":"categorical_crossentropy","D_p":"categorical_crossentropy"},
                optimizer='rmsprop', metrics={"D_a":"accuracy",'D_p':"accuracy"},
                loss_weights = { "D_a": D_a_loss_weight,"D_p": D_p_loss_weight}, lr= disc_lr)
    GDD.summary()
    return GDD

def create_generated_data(G):
    G_D= Model(inputs= G.input, outputs= G.output)
    G_D.summary()
    return G.output

#FUNCTION FOR GENERATING RANDOM INPUT BY SAMPLING FROM NORMAL DISTRIBUTION (INPUT VARIES AT EACH TIMESTEP)

def generate_input_noise(batch_size, latent_dim, time_steps):
    s=np.reshape(np.array(np.random.normal(0, 1, latent_dim * time_steps * batch_size)),(batch_size, time_steps, latent_dim))
    return s

#FUNCTION FOR GENERATING A SYNTHETIC DATA SET

def generate_synthetic_data(size, generator, latent_dim, time_steps):
    noise= generate_input_noise(size, latent_dim, time_steps)
    synthetic_data = generator.predict(noise)
    return synthetic_data

#FUNCTION FOR TRAINING GENERATOR FROM DBOTH DISCRIMINATOR AND CLASSIFIER OUTPUT

def train_G(batch_size, x, y_onehot, p_onehot, model, latent_dim):
    #noise = generate_input_noise(batch_size, latent_dim, x.shape[1])
    #concatenate real and noise for one input
    #gan_input = np.concatenate((np.array(x), np.array(noise)))
    #y_onehot_input = np.concatenate((np.array(y_onehot), np.array(np.zeros_like(y_onehot))))
    #p_onehot_input = np.concatenate((np.array(np.zeros_like(p_onehot)), np.array(p_onehot)))
    #loss = model.train_on_batch(gan_input, [y_onehot_input, p_onehot_input])#`train_on_batch` trains the model and updates the weights.
    loss = model.train_on_batch(x, [y_onehot,np.zeros_like(p_onehot)])#`train_on_batch` trains the model and updates the weights.loss = model.train_on_batch(x, [y_onehot, np.zeros_like(p_onehot)])
    return loss

# Loss definition

def mse(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_true-y_pred))

#CREATE GENERATOR AND DISCRIMINATOR

generator_model= generator(seq_length, latent_dim)
discriminator_a_model = descriminatorA(input_shape, num_classes_a)
discriminator_p_model = descriminatorP(input_shape, num_classes_p)
generated_data_model = create_generated_data(generator_model)

#CREATE FULL ARCHITECTURE WHERE OUPUT OF GENERATOR IS FED TO DISCRIMINATOR AND CLASSIFIER

for layer in discriminator_a_model.layers:
        layer.trainable = False
for layer in discriminator_p_model.layers:
        layer.trainable = False

gan = create_gan(discriminator_a_model, discriminator_p_model, generator_model)
test_loss_list = []
train_loss_list = []

gan.load_weights("F:\PHD\GAN\code.12.2020\PGAN1\experiments_keras\\rms_opt_wisdm_data\\best_generated\gan_ckpt_wisdm/variables/variables")
generator_model.load_weights("F:\PHD\GAN\code.12.2020\PGAN1\experiments_keras\\rms_opt_wisdm_data\\best_generated\\generator_ckpt_wisdm/variables/variables")
discriminator_p_model.load_weights("F:\PHD\GAN\code.12.2020\PGAN1\experiments_keras\\rms_opt_wisdm_data\\best_generated\discriminator_p_ckpt_wisdm/variables/variables")
discriminator_a_model.load_weights("F:\PHD\GAN\code.12.2020\PGAN1\experiments_keras\\rms_opt_wisdm_data\\best_generated\discriminaror_a_ckpt_wisdm/variables/variables")


while epoch <= 1000000:

    train_loss = train_G(batch_size_train, X_train, y_train_onehot, p_train_onehot, gan, latent_dim)
    train_loss_list.append((epoch,train_loss[0],train_loss[1],train_loss[2],train_loss[3],train_loss[4] ))
    print(epoch, 'train', gan.metrics_names, train_loss)
    
    #noise = generate_input_noise(batch_size_test, latent_dim, X_test.shape[1])
    
    test_loss = gan.test_on_batch(X_test, [y_test_onehot, p_test_onehot])
    test_loss_list.append((epoch,test_loss[0],test_loss[1],test_loss[2],test_loss[3],test_loss[4] ))
    print(epoch, 'test', gan.metrics_names, test_loss)
    if test_loss[3] > a_accuracy and test_loss[4] <=p_accuracy:
            gen_out = generator_model.predict(X_test)
            
            # save generated data
            
            h5f = h5py.File('vrms_GN_'+ dataset+'_data'+str(epoch)+'_a_'+str("{:.2f}".format(test_loss[3]) )+'_p_'+str("{:.2f}".format(test_loss[4]))+'.h5', 'w')
            h5f.create_dataset('X', data=gen_out)
            h5f.create_dataset('y', data=y_test)
            h5f.create_dataset('y_onehot', data=y_test_onehot)
            h5f.create_dataset('p', data=p_test)
            h5f.create_dataset('p_onehot', data=p_test_onehot)
            h5f.close()

            a_accuracy = test_loss[3]
            gan.save("gan_ckpt_"+dataset)
            #gan.save_weights(dataset+"vrms_best_e{0}_gan.h5".format(epoch))
            discriminator_a_model.save("discriminaror_a_ckpt_"+dataset)
            #discriminator_a_model.save_weights(dataset+"_rms_best_e{0}_discriminator_a.h5".format(epoch))
            #discriminator_p_model.save_weights(dataset+"_rms_best_e{0}_discriminator_p.h5".format(epoch))
            discriminator_p_model.save("discriminator_p_ckpt_"+dataset)
            generator_model.save("generator_ckpt_"+dataset)
            #generator_model.save_weights(dataset+"_rms_best_e{0}_generator.h5".format(epoch))
    
    if epoch % vis_freq == 0:
        pd.DataFrame(train_loss_list).to_csv("F:\PHD\GAN\code.12.2020\PGAN1/train_loss_rms_"+dataset+".csv")
        pd.DataFrame(test_loss_list).to_csv("F:\PHD\GAN\code.12.2020\PGAN1/test_loss_rms_"+dataset+".csv")
    
    epoch+=1

    """
    # Print loss before training
    y_pred, p_pred = gan.predict_on_batch(X_train)
    print("Before: " + str(mse(y_train_onehot,y_pred).numpy()))
    print("Before: " + str(mse(p_train_onehot,p_pred).numpy()))

    # Print loss output from train_on_batch
    print("Train output: " + str(gan.train_on_batch(X_train,[y_train_onehot, p_train_onehot])))

    print(gan.test_on_batch(X_test, [y_test_onehot, p_test_onehot]))

    # Print loss after training
    y_pred, p_pred = gan.predict_on_batch(X_test)
    print("After: " + str(mse(y_test_onehot,y_pred).numpy()))
    print("After: " + str(mse(p_test_onehot,p_pred).numpy()))"""



