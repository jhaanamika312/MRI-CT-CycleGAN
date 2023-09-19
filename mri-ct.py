# %%
from os import listdir
from numpy import asarray
from numpy import vstack
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from matplotlib import pyplot as plt
import numpy as np


def load_images(path, size=(256,256)):
	data_list = list()
	# enumerate filenames in directory, assume all are images
	for filename in listdir(path):
		# load and resize the image
		pixels = load_img(path + filename, target_size=size)
		# convert to numpy array
		pixels = img_to_array(pixels)
		# store
		data_list.append(pixels)
	return asarray(data_list)




# dataset path
path =r"/home/user/jha2a/cyclegan/Dataset/images/"

# load dataset A -ct scan
dataA_all = load_images(path + 'trainA/')
print('Loaded dataA: ', dataA_all.shape)


# load dataset B - Photos 
dataB_all = load_images(path + 'trainB/')
print('Loaded dataB: ', dataB_all.shape)


# Shuffle and align the datasets
num_samples = min(len(dataA_all), len(dataB_all))
indices = np.random.permutation(num_samples)

trainA_data = dataA_all[indices]
trainB_data = dataB_all[indices]


# Confirm dataset shapes
print("Train A shape:", trainA_data.shape)
print("Train B shape:", trainB_data.shape)


from sklearn.utils import resample
#To get a subset of all images, for faster training during demonstration
dataA = resample(trainA_data, 
                 replace=False,     
                 n_samples=100,    
                 random_state=42) 

dataB = resample(trainB_data, 
                 replace=False,     
                 n_samples=100,    
                 random_state=42) 


# plot source images
n_samples = 3
for i in range(n_samples):
	plt.subplot(2, n_samples, 1 + i)
	plt.axis('off')
	plt.imshow(dataA[i].astype('uint8'))
# plot target image
for i in range(n_samples):
	plt.subplot(2, n_samples, 1 + n_samples + i)
	plt.axis('off')
	plt.imshow(dataB[i].astype('uint8'))
plt.show()


# load image data
data = [dataA, dataB]

print('Loaded', data[0].shape, data[1].shape)



#Preprocess data to change input range to values between -1 and 1
# This is because the generator uses tanh activation in the output layer
#And tanh ranges between -1 and 1
def preprocess_data(data):
	# load compressed arrays
	# unpack arrays
	X1, X2 = data[0], data[1]
	# scale from [0,255] to [-1,1]
	X1 = (X1 - 127.5) / 127.5
	X2 = (X2 - 127.5) / 127.5
	return [X1, X2]

dataset = preprocess_data(data)

from cycleGAN_model import define_generator, define_discriminator, define_composite_model, train
# define input shape based on the loaded dataset
image_shape = dataset[0].shape[1:]
# generator: A -> B
g_model_AtoB = define_generator(image_shape)
# generator: B -> A
g_model_BtoA = define_generator(image_shape)
# discriminator: A -> [real/fake]
d_model_A = define_discriminator(image_shape)
# discriminator: B -> [real/fake]
d_model_B = define_discriminator(image_shape)
# composite: A -> B -> [real/fake, A]
c_model_AtoB = define_composite_model(g_model_AtoB, d_model_B, g_model_BtoA, image_shape)
# composite: B -> A -> [real/fake, B]
c_model_BtoA = define_composite_model(g_model_BtoA, d_model_A, g_model_AtoB, image_shape)

from datetime import datetime 
start1 = datetime.now() 
# train models
train(d_model_A, d_model_B, g_model_AtoB, g_model_BtoA, c_model_AtoB, c_model_BtoA, dataset, epochs=50)

stop1 = datetime.now()
#Execution time of the model 
execution_time = stop1-start1
print("Execution time is: ", execution_time)

#plot loss function
import pickle

# Specify the directory path where you saved the loss values file
directory_path = '/home/user/jha2a/cyclegan/'

# Combine the directory path with the file name
file_path = directory_path + 'loss_values.pkl'

# Load saved loss values from the file
with open(file_path, 'rb') as file:
    loss_values = pickle.load(file)

# Print the loaded loss values
print("Loaded loss values:")
print(loss_values)


# Extract the loss values from the loaded dictionary
dA_loss1_values = loss_values['dA_loss1_values']
dA_loss2_values = loss_values['dA_loss2_values']
dB_loss1_values = loss_values['dB_loss1_values']
dB_loss2_values = loss_values['dB_loss2_values']
g_loss1_values = loss_values['g_loss1_values']
g_loss2_values = loss_values['g_loss2_values']

# Print the extracted values to check their contents
print("dA_loss1_values:", dA_loss1_values)
print("dA_loss2_values:", dA_loss2_values)
print("dB_loss1_values:", dB_loss1_values)
print("dB_loss2_values:", dB_loss2_values)
print("g_loss1_values:", g_loss1_values)
print("g_loss2_values:", g_loss2_values)


# Plot the loss by training iteration for each loss component
import matplotlib.pyplot as plt

plt.plot(dA_loss1_values, label="Discriminator A Loss on Real")
plt.plot(dA_loss2_values, label="Discriminator A Loss on Fake")
plt.plot(dB_loss1_values, label="Discriminator B Loss on Real")
plt.plot(dB_loss2_values, label="Discriminator B Loss on Fake")
plt.plot(g_loss1_values, label="Generator AtoB Loss")
plt.plot(g_loss2_values, label="Generator BtoA Loss")

plt.title("Loss by Iteration")
plt.ylabel("Loss")
plt.xlabel("Iteration")
plt.legend()

# Display the plot
plt.show()


# Use the saved cyclegan models for image translation
 
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
from keras.models import load_model
from matplotlib import pyplot
from numpy.random import randint

# select a random sample of images from the dataset
def select_sample(dataset, n_samples):
	# choose random instances
	ix = randint(0, dataset.shape[0], n_samples)
	# retrieve selected images
	X = dataset[ix]
	return X

# plot the image, its translation, and the reconstruction
def show_plot(imagesX, imagesY1, imagesY2):
	images = vstack((imagesX, imagesY1, imagesY2))
	titles = ['Real', 'Generated', 'Reconstructed']
	# scale from [-1,1] to [0,1]
	images = (images + 1) / 2.0
	# plot images row by row
	for i in range(len(images)):
		# define subplot
		pyplot.subplot(1, len(images), 1 + i)
		# turn off axis
		pyplot.axis('off')
		# plot raw pixel data
		pyplot.imshow(images[i])
		# title
		pyplot.title(titles[i])
	pyplot.show()

# load dataset
A_data = resample(dataA_all, 
                 replace=False,     
                 n_samples=50,    
                 random_state=42) # reproducible results

B_data = resample(dataB_all, 
                 replace=False,     
                 n_samples=50,    
                 random_state=42) # reproducible results

A_data = (A_data - 127.5) / 127.5
B_data = (B_data - 127.5) / 127.5

# load the models
cust = {'InstanceNormalization': InstanceNormalization}
model_AtoB = load_model('/home/user/jha2a/cyclegan/g_model_AtoB_050000.h5', cust)
model_BtoA = load_model('/home/user/jha2a/cyclegan/g_model_BtoA_050000.h5', cust)

# plot A->B->A (CT to MRI to CT)
A_real = select_sample(A_data, 1)
B_generated  = model_AtoB.predict(A_real)
A_reconstructed = model_BtoA.predict(B_generated)
show_plot(A_real, B_generated, A_reconstructed)

# Save the generated output as files in the current directory
pyplot.savefig('AtoBtoA_output.png')
pyplot.close()

# plot B->A->B (MRI to CT to MRI)
B_real = select_sample(B_data, 1)
A_generated  = model_BtoA.predict(B_real)
B_reconstructed = model_AtoB.predict(A_generated)
show_plot(B_real, A_generated, B_reconstructed)

# Save the generated output as files in the current directory
pyplot.savefig('BtoAtoB_output.png')
pyplot.close()

# Import necessary modules for test datset
from numpy import asarray
from numpy import vstack
from keras_contrib.layers import InstanceNormalization
from keras.models import load_model
from matplotlib import pyplot as plt
from numpy.random import randint

# Load Test A (CT scan) and Test B (MRI) datasets
testA_data = load_images('/home/user/jha2a/cyclegan/Dataset/images/testA/')
testB_data = load_images('/home/user/jha2a/cyclegan/Dataset/images/testB/')

# Preprocess the test datasets
testA_data = (testA_data - 127.5) / 127.5
testB_data = (testB_data - 127.5) / 127.5

# Load the saved generator models
cust = {'InstanceNormalization': InstanceNormalization}



model_AtoB = load_model('/home/user/jha2a/cyclegan/g_model_AtoB_050000.h5', cust)
model_BtoA = load_model('/home/user/jha2a/cyclegan/g_model_AtoB_050000.h5', cust)



import os
import numpy as np
from matplotlib import pyplot as plt

# Select a random sample of images from the test dataset
def select_sample(dataset, n_samples):
    ix = np.random.randint(0, dataset.shape[0], n_samples)
    X = dataset[ix]
    return X

# Generate translated CT scan images from MRI images
batch_size = 100
num_batches = len(testB_data) // batch_size

for batch_idx in range(num_batches):
    start_idx = batch_idx * batch_size
    end_idx = (batch_idx + 1) * batch_size
    
    batch_testB_data = testB_data[start_idx:end_idx]
    translated_ct_scans = model_BtoA.predict(batch_testB_data)  # mri-ct

import os
output_dir = '/home/user/jha2a/cyclegan/test-images'
os.makedirs(output_dir, exist_ok=True)



for i in range(batch_size):
        plt.subplot(1, 2, 1)
        plt.axis('off')
        plt.imshow((batch_testB_data[i] + 1) / 2.0)
        plt.title('MRI')

        plt.subplot(1, 2, 2)
        plt.axis('off')
        plt.imshow((translated_ct_scans[i] + 1) / 2.0)
        plt.title('Translated CT')
        
        plt.savefig(os.path.join(output_dir, f'Batch_{batch_idx}_MRI_to_CT_{i}.png'))
        plt.close()

print('Visualizations saved to:', output_dir)

# Calculate PSNR for MRI to CT translation
#import numpy as np
#from skimage.metrics import peak_signal_noise_ratio
#psnr_mri_to_ct = np.mean([peak_signal_noise_ratio((testB_data[i] + 1) / 2.0, (translated_ct_scans[i] + 1) / 2.0) for i in range(len(testB_data))])
#print(f"PSNR for MRI to CT translation: {psnr_mri_to_ct:.4f}")