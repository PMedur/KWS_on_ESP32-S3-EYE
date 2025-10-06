import tensorflow as tf
from tensorflow.keras import models, layers


def compile_model(model, learning_rate=1e-3, metrics=None):
    """
    Compile model with standard configuration.
    Metrics should be passed from training module.
    """
    if metrics is None:
        metrics = ['accuracy']
    
    model.compile(
        loss='categorical_crossentropy',
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        metrics=metrics
    )
    return model


def build_model_1(input_shape=(13, 50), num_classes=36, metrics=None):
    """Lightweight CNN: 8 filters, 1 conv layer"""
    model = models.Sequential([
        layers.InputLayer(input_shape=input_shape),
        layers.Reshape(target_shape=(input_shape[0], input_shape[1], 1)),
        layers.Conv2D(8, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(32, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    return compile_model(model, metrics=metrics)


def build_model_2(input_shape=(13, 50), num_classes=36, metrics=None):
    """Lightweight CNN: 16 filters, 1 conv layer"""
    model = models.Sequential([
        layers.InputLayer(input_shape=input_shape),
        layers.Reshape(target_shape=(input_shape[0], input_shape[1], 1)),
        layers.Conv2D(16, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    return compile_model(model, metrics=metrics)


def build_model_3(input_shape=(13, 50), num_classes=36, metrics=None):
    """Medium CNN: 2 conv layers (16→32), dropout"""
    model = models.Sequential([
        layers.InputLayer(input_shape=input_shape),
        layers.Reshape(target_shape=(input_shape[0], input_shape[1], 1)),
        layers.Conv2D(16, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    return compile_model(model, metrics=metrics)


def build_model_4(input_shape=(13, 50), num_classes=36, metrics=None):
    """Medium CNN: 32 filters, 1 conv layer"""
    model = models.Sequential([
        layers.InputLayer(input_shape=input_shape),
        layers.Reshape(target_shape=(input_shape[0], input_shape[1], 1)),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    return compile_model(model, metrics=metrics)


def build_model_5(input_shape=(13, 50), num_classes=36, metrics=None):
    """Medium CNN: 2 conv layers (16→32), no dropout"""
    model = models.Sequential([
        layers.InputLayer(input_shape=input_shape),
        layers.Reshape(target_shape=(input_shape[0], input_shape[1], 1)),
        layers.Conv2D(16, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    return compile_model(model, metrics=metrics)


def build_model_6(input_shape=(13, 50), num_classes=36, metrics=None):
    """Medium CNN: 2 conv layers (16→32), 128 dense, dropout"""
    model = models.Sequential([
        layers.InputLayer(input_shape=input_shape),
        layers.Reshape(target_shape=(input_shape[0], input_shape[1], 1)),
        layers.Conv2D(16, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    return compile_model(model, metrics=metrics)


def build_model_7(input_shape=(13, 50), num_classes=36, metrics=None):
    """Deep CNN: 3 conv layers (32→64→128)"""
    model = models.Sequential([
        layers.InputLayer(input_shape=input_shape),
        layers.Reshape(target_shape=(input_shape[0], input_shape[1], 1)),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    return compile_model(model, metrics=metrics)


def build_model_8(input_shape=(13, 50), num_classes=36, metrics=None):
    """Large CNN: 2 conv layers (64→128), 256 dense"""
    model = models.Sequential([
        layers.InputLayer(input_shape=input_shape),
        layers.Reshape(target_shape=(input_shape[0], input_shape[1], 1)),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    return compile_model(model, metrics=metrics)


def build_model_9(input_shape=(13, 50), num_classes=36, metrics=None):
    """Very Deep CNN: 3 conv layers (64→128→256), 512 dense"""
    model = models.Sequential([
        layers.InputLayer(input_shape=input_shape),
        layers.Reshape(target_shape=(input_shape[0], input_shape[1], 1)),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    return compile_model(model, metrics=metrics)


def build_model_10(input_shape=(13, 50), num_classes=36, metrics=None):
    """Extra Large CNN: 3 conv layers (128→256→512), 1024 dense"""
    model = models.Sequential([
        layers.InputLayer(input_shape=input_shape),
        layers.Reshape(target_shape=(input_shape[0], input_shape[1], 1)),
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(512, (3, 3), activation='relu', padding='same'),
        layers.Flatten(),
        layers.Dense(1024, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    return compile_model(model, metrics=metrics)


MODEL_REGISTRY = {
    'model_1': build_model_1,
    'model_2': build_model_2,
    'model_3': build_model_3,
    'model_4': build_model_4,
    'model_5': build_model_5,
    'model_6': build_model_6,
    'model_7': build_model_7,
    'model_8': build_model_8,
    'model_9': build_model_9,
    'model_10': build_model_10,
}


def get_model_builder(model_name):
    """Get model builder function by name."""
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Model '{model_name}' not found. Available: {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[model_name]