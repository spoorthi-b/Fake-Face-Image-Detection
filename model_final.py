from tkinter import *
from PIL import Image, ImageTk
from tkinter import filedialog
from tkinter.ttk import *
import tensorflow
from tensorflow.keras.layers import Input, BatchNormalization, ReLU, Conv2D, Conv1D, Flatten, Dense, MaxPool2D, AvgPool2D, GlobalAvgPool2D, GlobalAveragePooling1D, Concatenate, Reshape, Lambda, GlobalAveragePooling2D, Softmax
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K
import tensorflow as tf
import random
from tensorflow import keras
import time
import os
import numpy as np
import cv2

cffn = tf.keras.models.load_model(filepath = "./featureExtractor")
cffn.predict(np.array([cv2.resize(cv2.imread('./test/real/000001.jpg'), (64, 64))]))

def avg(arr):
	return (sum(arr)/len(arr))

def euclidean_distance(outputs):
    # unpack the vectors into separate lists
    (featsA, featsB) = outputs
    # compute the sum of squared distances between the vectors
    sumSquared = K.sum(K.square(featsA - featsB), axis=1, keepdims=True)
    # return the euclidean distance between the vectors
    return K.sqrt(K.maximum(sumSquared, K.epsilon()))

root = Tk()
image1 = Image.open("./13.jpg")
image1 = image1.resize((1700,900),Image.ANTIALIAS)
bg = ImageTk.PhotoImage(image1)

my_canvas = Canvas(root,width=1700,height=900)

my_canvas.pack(fill="both", expand=True)

my_canvas.create_image(0,0,image=bg,anchor="nw")

my_canvas.create_text(480,300,text="Fake Face Image Detection",font=("Comic Sans MS",50,"bold"),fill="white")
item = my_canvas.create_text(400,500,text = "", font=("Comic Sans MS",50,"bold"),fill="white")

def myClick():
	root.filename = filedialog.askopenfilename(initialdir="./",title="Select a file",filetypes=(("png","*.png"),("jpg","*.jpg")))
	
	real_path = random.choice(os.listdir("./data/test/test/real"))
	fake_path = random.choice(os.listdir("./data/test/test/fake"))
	real_imgs = []
	fake_imgs = []
	#real_img = cv2.imread(os.path.join("D:/project/real", real))
	#real_img = cv2.resize(real_img, (64, 64))
	#fake_img = cv2.imread(os.path.join("D:/project/fake", fake))
	#fake_img = cv2.resize(fake_img, (64, 64))
	for i in range(4):
		real_imgs.append(cv2.resize(cv2.imread(os.path.join('./data/test/test/real', np.random.choice(os.listdir('./data/test/test/real')))), (64, 64)))
		fake_imgs.append(cv2.resize(cv2.imread(os.path.join('./data/test/test/fake', np.random.choice(os.listdir('./data/test/test/fake')))), (64, 64)))
	test_image = cv2.resize(cv2.imread(root.filename), (64, 64))

	#real_dist = euclidean_distance([cffn.predict(np.array([real_img])), cffn.predict(np.array([test_image]))])
	#fake_dist = euclidean_distance([cffn.predict(np.array([fake_img])), cffn.predict(np.array([test_image]))])
	real_dist = []
	fake_dist = []
	for i in range(4):
		real_dist.append(euclidean_distance([cffn.predict(np.array([real_imgs[i]])), cffn.predict(np.array([test_image]))]))
		fake_dist.append(euclidean_distance([cffn.predict(np.array([fake_imgs[i]])), cffn.predict(np.array([test_image]))]))
	#print('real_dist = '+str(real_dist)+'fake_dist = '+str(fake_dist))
	if avg(real_dist)<avg(fake_dist):
		my_canvas.itemconfig(item, text = 'real')
		#my_canvas.create_text(500,500,text = "real",font=("Comic Sans MS",50,"bold"),fill="white")
		print('real')
	else:
		my_canvas.itemconfig(item, text = 'fake')
		print('fake')
		#my_canvas.create_text(500,500,text = "fake",font=("Comic Sans MS",50,"bold"),fill="white")
style = Style()
#photo = Image.open("./13c.jpg")
#image1 = image1.resize((10,10),Image.ANTIALIAS)
#bg1 = ImageTk.PhotoImage(photo)
style.configure('W.TButton',font=("calibri", 30,"bold"),foreground="#10518c")
button = Button(root,text="Upload Image",command=myClick)

button_window = my_canvas.create_window(350,400,anchor="nw",window=button)

#while True:
root.mainloop()