from tensorflow.keras.models import load_model
import numpy as np

model = load_model("models/crowd_panic_cnn.h5")

def predict(frame):
    frame = np.expand_dims(frame, axis=0)
    return model.predict(frame)
