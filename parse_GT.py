import os

imgs = list(map(lambda x:x[:-4],os.listdir('widerface/train/images')))
offsets = list(map(lambda x:x[:-4],os.listdir('widerface/train/offsets')))

outliers = filter(lambda x: x not in offsets,imgs)
print(list(outliers))


