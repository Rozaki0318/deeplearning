import tensorflow
from tensorflow import keras

# 変数定義
batch_size = 128
num_class = 10
epochs = 10
img_rows = 28
img_cols = 28

# データ読み込み
(x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()

# CNN向けにデータ変換(今回のデータはchannel_last)
x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
input_shape = (img_rows, img_cols, 1)

# 正規化
x_train, x_test = x_train/255.0, x_test/255.0

# CNNモデル定義
model = keras.models.Sequential([
    keras.layers.Conv2D(32, kernel_size=(3,3), activation='relu', input_shape = input_shape),
    keras.layers.Conv2D(64, (3,3), activation='relu'),
    keras.layers.MaxPooling2D(pool_size=(2,2)),
    keras.layers.Dropout(0.25),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(num_class, activation='softmax')
])

# コンパイル
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# 学習
model.fit(x_train, y_train, epochs=20)

# 精度確認
final_accuracy = model.evaluate(x_test, y_test)
print(final_accuracy)

# 試験的に一部データの推定
print(model.predict(x_test[99:100]))
