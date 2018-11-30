import numpy as np

# utils to help iterate through data
class dataIterator(object):
    def __init__(self,images,label,batch_size):

        self.image = images
        self.label = label
        self.batch_size = batch_size
        self.data_size = len(self.image)

        self.start = 0

        self.randomize()


    def randomize(self):
        self.idx = np.random.permutation(self.data_size)
        self.start = 0

    def next_batch(self):
        end = self.start + self.batch_size
        idx = self.idx[self.start:end]

        batch_X = self.image[idx]
        batch_Y = self.label[idx]

        self.start += self.batch_size

        if self.start >= self.data_size:
            self.randomize()
        batch_X = np.reshape(batch_X,newshape=[self.batch_size,64,64,1])
        return batch_X,batch_Y

#load all the data
def dataloader():
    xTrain = np.load('train_val/train_X.npy')
    yTrain = np.load('train_val/train_Y.npy')

    train_bbox = np.load('train_val/train_bboxes.npy')

    xValid = np.load('train_val/valid_X.npy')
    yValid = np.load('train_val/valid_Y.npy')

    valid_bbox = np.load('train_val/train_bboxes.npy')


    yTrain = np.array([one_hot(m) for m in yTrain])
    yValid = np.array([one_hot(n) for n in yValid])

    xValid = np.reshape(xValid, newshape=[xValid.shape[0], 64, 64, 1])

    return xTrain,yTrain,xValid,yValid,train_bbox,valid_bbox

def one_hot(label):
    #print(len(label.reshape(-1)))
    return np.squeeze(np.eye(10)[label.reshape(-1)])

