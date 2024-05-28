import tensorflow as tf
import tensorflow_datasets as tfds

# Carregando o dataset Cats vs Dogs
dataset_name = "cats_vs_dogs"
dataset, info = tfds.load(name=dataset_name, split=tfds.Split.TRAIN, with_info=True)

# Função de pré-processamento
def preprocess(features):
    image = tf.image.resize(features['image'], (224, 224))
    image = tf.cast(image, tf.float32) / 255.0
    label = tf.cast(features['label'], tf.float32)  # Garantir que o label seja float32
    return image, label

# Aplicando o pré-processamento e batching
batch_size = 32
dataset = dataset.map(preprocess).batch(batch_size)

# Carregando o modelo pré-treinado MobileNetV2
base_model = tf.keras.applications.MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Congelando todas as camadas do modelo base
for layer in base_model.layers:
    layer.trainable = False

# Adicionando camadas adicionais no topo do modelo base
x = base_model.output
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dense(1024, activation='relu')(x)
predictions = tf.keras.layers.Dense(1, activation='sigmoid')(x)

# Criando o modelo final
model = tf.keras.models.Model(inputs=base_model.input, outputs=predictions)

# Compilando o modelo
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Treinando o modelo
num_epochs = 10
model.fit(dataset, epochs=num_epochs)

# Avaliando o modelo no conjunto de teste
test_dataset = tfds.load(name=dataset_name, split=tfds.Split.TEST)
test_dataset = test_dataset.map(preprocess).batch(batch_size)

loss, accuracy = model.evaluate(test_dataset)
print(f'Accuracy on test set: {accuracy}')





