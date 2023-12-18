import numpy as np
import sys

sys.path.append('.')

from train import train
from PIL import Image
from face_emotion_recongnition.utils import trans_predict
from face_emotion_recongnition.utils import get_emotions

model = train(load=True)
image = Image.open('data/image/natural.png')

grey_image = image.convert('L')
grey_image = np.array(grey_image)

pre_image= trans_predict(grey_image)
ans = model.predict(pre_image)

print(get_emotions(ans))