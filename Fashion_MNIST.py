import tensorflow
from tensorflow import keras

batch_size = 50
num_class = 10
epochs = 10

# データ読み込み
(x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()
# 正規化
x_train, x_test = x_train/255.0, x_test/255.0

model = keras.models.Sequential([
    keras.layers.Flatten(),
    keras.layers.Dense(512, activation='relu'),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=20)

final_accuracy = model.evaluate(x_test, y_test)
print(final_accuracy)

# results==================================
# optimizer: adam, keras.layers.Dense(512), epochs: 20, batch_size: 50 => 0.8935
# optimizer: adam, keras.layers.Dense(512), epochs: 20, batch_size: 100 => 0.8933
# optimizer: adam, keras.layers.Dense(512), epochs: 20, batch_size: 200 => 0.8917 
# optimizer: adam, keras.layers.Dense(512), epochs: 30, batch_size: 100 => 0.891
# optimizer: adam, keras.layers.Dense(1024), epochs: 20, batch_size: 100 => 0.8859
# optimizer: SGD , keras.layers.Dense(512), epochs: 20, batch_size: 100 => 0.8715
