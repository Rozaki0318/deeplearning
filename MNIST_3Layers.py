
# coding: utf-8

# In[1]:


import tensorflow
from tensorflow import keras

# In[3]:


import matplotlib.pyplot as plt


# In[4]:


batch_size = 128
num_class = 10 # 出力
epochs = 20


# In[5]:


(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()


# In[6]:


len(x_train)


# In[7]:


x_train[0:10]


# In[8]:


len(x_test)


# In[9]:


len(y_test)


# In[10]:


# for i in range(10):
#     plt.subplot(2, 5, i+1)
#     plt.title("Label:" + str(i))
#     plt.imshow(x_train[i].reshape(28,28), cmap=None)


# In[11]:


y_train[0:10]


# In[12]:


x_train, x_test = x_train / 255.0, x_test / 255.0


# In[13]:


x_train[0]


# In[14]:


model = keras.models.Sequential([
    keras.layers.Flatten(),
    keras.layers.Dense(512, activation='relu'),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(10, activation='softmax')
])


# In[15]:


model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


# In[ ]:


model.fit(x_train, y_train, epochs=5)

