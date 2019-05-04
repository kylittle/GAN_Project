import os
import numpy as np
import datetime
import configparser
from tqdm import tqdm
from keras.layers import Input
from keras.models import Model, Sequential
from keras.layers.core import Dense, Dropout
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import Adam
from keras import initializers
from mido import MidiFile
from mido import MidiTrack, Message

os.environ["KERAS_BACKEND"] = "tensorflow"
#np.random.seed(10)
random_dim = 2400
file_counter = 1
prev_note = 0
min_time = 99999999
max_time = -99999999
min_channel = 9999999
max_channel = -9999999
min_note = -127
max_note = 127
min_attack = 0
max_attack = 100
epoch_count = 0
output_frequency = 0
mode = 0

def main():
    #np.set_printoptions(threshold=np.nan, suppress=True)
    read_ini_config()
    print(output_frequency)
    train(1000, 128)

def read_ini_config():
    global epoch_count, output_frequency, mode #Let the setting variables be Global
    config = configparser.ConfigParser()
    config.read("./config.ini")
    epoch_count = int(config["DEFAULT"]["EpochCount"])
    output_frequency = int(config["DEFAULT"]["EpochsBeforeOutput"])
    mode = int(config["DEFAULT"]["TrainingType"])
    mode = 0 if mode != 0 and mode != 1 else mode

#Input: MidiFile Object, Output: Ordered Matrix with each part representing a note
def make_midi_matrix(mid_in, data_out):
    global min_time, max_time, min_channel, max_channel, min_note, max_note, min_attack, max_attack
    first_loop = True
    for track in mid_in.tracks:
        for msg in track:
            if not first_loop:
                if not msg.is_meta and msg.type == "note_on":
                    command = (msg.channel, msg.note-prev_note, msg.velocity, msg.time)
                    min_time = min_time if min_time <= msg.time else msg.time
                    max_time = max_time if max_time >= msg.time else msg.time
                    min_channel = min_channel if min_channel <= msg.channel else msg.channel
                    max_channel = max_channel if max_channel >= msg.channel else msg.channel
                    data_out.append(command)
            else:
                if not msg.is_meta and msg.type == "note_on":
                    command = (msg.channel, 0, msg.velocity, msg.time)
                    first_loop = False
                    min_time = min_time if min_time <= msg.time else msg.time
                    max_time = max_time if max_time >= msg.time else msg.time
                    min_channel = min_channel if min_channel <= msg.channel else msg.channel
                    max_channel = max_channel if max_channel >= msg.channel else msg.channel
                    prev_note = msg.note
    return data_out

def unscale_midi(data_set):
    unscaled_data=[]
    first_loop = True
    for unscale_data_line in data_set:
        if first_loop:
            unscale_data_line = (round(unscale(unscale_data_line[0], min_channel, max_channel)), 60, 
                round(unscale(unscale_data_line[2], min_attack, max_attack)), round(unscale(unscale_data_line[3], min_time, max_time)))
            prev_note = 60
            first_loop= False
        else:
            note = (prev_note + unscale(unscale_data_line[1], min_note, max_note)) % 127
            unscale_data_line = (round(unscale(unscale_data_line[0], min_channel, max_channel)), round(note), 
                round(unscale(unscale_data_line[2], min_attack, max_attack)), round(unscale(unscale_data_line[3], min_time, max_time)))
            prev_note = note
        unscaled_data.append(unscale_data_line)
    unscaled_data = np.array(unscaled_data)
    return unscaled_data

def scale_for_learning(data_set):
    return_data_set = []
    #First scale channel (this may not be something we need to do
    for data_line in data_set:
        scaled_data = (scale(data_line[0], min_channel, max_channel), scale(data_line[1], min_note, max_note), 
            scale(data_line[2], min_attack, max_attack), scale(data_line[3], min_time, max_time))
        return_data_set.append(scaled_data)
    return return_data_set

def scale(x, x_min, x_max):
    x_range = x_max - x_min
    scaled_value = (2/x_range)*x + ((2*x_min)/(x_min-x_max))-1
    #scaled_value = (2*scaled_value)-1
    return scaled_value

def unscale(x, x_min, x_max):
    x_range = x_max - x_min
    unscaled = ((-2*x_min + x_range * x + x_range)/2)
    if unscaled >= 0:
        unscaled%=127
    else:
        unscaled%=127
        unscaled/=-1
    return unscaled


def load_midi_data():
    if mode == 0:
        #Bach Training
        print("Training with various Bach Fugue")
        data = []
        mid = MidiFile('./input/bach-1.mid')
        data = make_midi_matrix(mid, data)
        mid = MidiFile('./input/bach-2.mid')
        data = make_midi_matrix(mid, data)
        mid = MidiFile('./input/bach-3.mid')
        data = make_midi_matrix(mid, data)
        mid = MidiFile('./input/bach-4.mid')
        data = make_midi_matrix(mid, data)
        mid = MidiFile('./input/bach-5.mid')
        data = make_midi_matrix(mid, data)
        mid = MidiFile('./input/bach-6.mid')
        data = make_midi_matrix(mid, data)
        mid = MidiFile('./input/bach-7.mid')
        data = make_midi_matrix(mid, data)
        mid = MidiFile('./input/bach-8.mid')
        data = make_midi_matrix(mid, data)
        mid = MidiFile('./input/bach-9.mid')
        data = make_midi_matrix(mid, data)
        mid = MidiFile('./input/bach-10.mid')
        data = make_midi_matrix(mid, data)
        mid = MidiFile('./input/bach-11.mid')
        data = make_midi_matrix(mid, data)
        mid = MidiFile('./input/bach-12.mid')
        data = make_midi_matrix(mid, data)
        mid = MidiFile('./input/bach-13.mid')
        data = make_midi_matrix(mid, data)
        mid = MidiFile('./input/bach-14.mid')
        data = make_midi_matrix(mid, data)
        mid = MidiFile('./input/bach-15.mid')
        data = make_midi_matrix(mid, data)
        mid = MidiFile('./input/bach-16.mid')
        data = make_midi_matrix(mid, data)
        mid = MidiFile('./input/bach-17.mid')
        data = make_midi_matrix(mid, data)
    
        data = scale_for_learning(data)

        data = np.array(data)
        return data

    elif mode == 1: #This is for future modal support
        #Captain Beefheart Training
        print("Training with Captain Beefheart - Trout Mask Replica")
        data = []
        mid = MidiFile('./input/brickbats.mid')
        data = make_midi_matrix(mid, data)
        mid = MidiFile('./input/batmiddle.mid')
        data = make_midi_matrix(mid, data)
        mid = MidiFile('./input/alice.mid')
        data = make_midi_matrix(mid, data)
        mid = MidiFile('./input/bigdummy.mid')
        data = make_midi_matrix(mid, data)
        mid = MidiFile('./input/myhuman.mid')
        data = make_midi_matrix(mid, data)
        mid = MidiFile('./input/peon.mid')
        data = make_midi_matrix(mid, data)
        mid = MidiFile('./input/golden.mid')
        data = make_midi_matrix(mid, data)
        mid = MidiFile('./input/fbl.mid')
        data = make_midi_matrix(mid, data)
        mid = MidiFile('./input/veterans.mid')
        data = make_midi_matrix(mid, data)

        data = scale_for_learning(data)

        data = np.array(data)
        return data

def get_optimizer():
    return Adam()

def get_generator(optimizer):
    generator = Sequential()
    generator.add(Dense(150, input_dim=random_dim, kernel_initializer=initializers.RandomNormal(stddev=0.02)))
    generator.add(LeakyReLU(0.2))
    generator.add(BatchNormalization(momentum=0.8))

    generator.add(Dense(300))
    generator.add(LeakyReLU(0.2))
    generator.add(BatchNormalization(momentum=0.8))

    generator.add(Dense(500))
    generator.add(LeakyReLU(0.2))
    generator.add(Dropout(0.3))

    generator.add(Dense(4, activation='tanh'))
    generator.compile(loss='binary_crossentropy', optimizer=optimizer)
    return generator

def get_discriminator(optimizer):
    discriminator = Sequential()
    discriminator.add(Dense(300, input_dim=4, kernel_initializer=initializers.RandomNormal(stddev=0.02)))
    discriminator.add(LeakyReLU(0.2))
    discriminator.add(Dropout(0.3))

    discriminator.add(Dense(150))
    discriminator.add(LeakyReLU(0.2))
    discriminator.add(Dropout(0.3))

    discriminator.add(Dense(75))
    discriminator.add(LeakyReLU(0.2))
    discriminator.add(Dropout(0.3))

    discriminator.add(Dense(1, activation='sigmoid'))
    discriminator.compile(loss='binary_crossentropy', optimizer=optimizer)
    return discriminator

def get_gan_network(discriminator, random_dim, generator, optimizer):
    discriminator.trainable = False
    gan_input = Input(shape=(random_dim,))
    x=generator(gan_input)
    gan_output = discriminator(x)
    gan=Model(inputs=gan_input, outputs=gan_output)
    gan.compile(loss='binary_crossentropy', optimizer=optimizer)
    return gan

def save_midi_output(epoch, generator, examples=2400):
    noise=np.random.normal(0,1,size=[examples,random_dim])
    
    generated = generator.predict(noise)
    generated = unscale_midi(generated)
    mid = MidiFile(type=2)
    track0=MidiTrack()
    track1=MidiTrack()
    track2=MidiTrack()
    track3=MidiTrack()
    for data_line in generated:
        if data_line[0] == 0:
            track0.append(Message('note_on', note=int(abs(data_line[1])), velocity=int(abs(data_line[2])), time=int(abs(data_line[3]))+200))
        if data_line[0] == 1:
            track1.append(Message('note_on', note=int(abs(data_line[1])), velocity=int(abs(data_line[2])), time=int(abs(data_line[3]))+200))
        if data_line[0] == 2:
            track2.append(Message('note_on', note=int(abs(data_line[1])), velocity=int(abs(data_line[2])), time=int(abs(data_line[3]))+200))
        if data_line[0] == 3:
            track3.append(Message('note_on', note=int(abs(data_line[1])), velocity=int(abs(data_line[2])), time=int(abs(data_line[3]))+200))
    print(generated)
    mid.tracks.append(track0)
    mid.tracks.append(track1)
    mid.tracks.append(track2)
    mid.tracks.append(track3)
    print('The generated track has been saved to output.mid')
    mid.save('./output/output' + str(epoch) + '.mid')

def train(epochs=1, batch_size=128):
    x_train = load_midi_data()
    batch_count = int(x_train.shape[0]/batch_size)
    
    adam = get_optimizer()
    generator = get_generator(adam)
    discriminator = get_discriminator(adam)
    gan = get_gan_network(discriminator, random_dim, generator, adam)

    for e in range(1, epochs+1):
        print('-'*15, 'Epoch %d' %e, '-'*15)
        for obj in tqdm(range(batch_count)):
            noise = np.random.normal(0,1,size=[batch_size, random_dim])
            mid_batch = x_train[np.random.randint(0,x_train.shape[0], size=batch_size)]

            generated_mid = generator.predict(noise)
            X = np.concatenate([mid_batch, generated_mid])

            y_dis = np.zeros(2*batch_size)
            y_dis[:batch_size] = 0.9

            discriminator.trainable = True
            discriminator.train_on_batch(X, y_dis)

            noise = np.random.normal(0, 1, size=[batch_size, random_dim])
            y_gen = np.ones(batch_size)
            discriminator.trainable = False
            gan.train_on_batch(noise, y_gen)

        if e == epoch_count or e % output_frequency == 0:
            #history = gan.fit()
            #plt.plot(history.history['loss'])
            #plt.plot(history.history['val_loss'])
            #plt.ylabel('loss')
            #plt.xlabel('epoch')
            save_midi_output(e, generator)

if __name__ == '__main__':
    main()
