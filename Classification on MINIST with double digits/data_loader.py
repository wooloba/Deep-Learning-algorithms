import numpy as np

# utils to help iterate through data
class dataIterator(object):
    def __init__(self,images,label,batch_size):

        self.image = images
        self.label = label
        self.batch_size = batch_size
        self.data_size = len(self.image)

        self.start = 0
        self.idx = self.randomize()


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

        return batch_X,batch_Y

#load all the data
def dataloader():
    xTrain = np.load('train_val/train_X.npy')
    yTrain = np.load('train_val/train_Y.npy')

    xValid = np.load('train_val/valid_X.npy')
    yValid = np.load('train_val/valid_Y.npy')


    yTrain = np.array([one_hot(m) for m in yTrain])
    yValid = np.array([one_hot(n) for n in yValid])

    return xTrain,yTrain,xValid,yValid

def one_hot(label):
    #print(len(label.reshape(-1)))
    return sum(np.squeeze(np.eye(10)[label.reshape(-1)]))


def data_spliter(image,label):
    test_data_size = 10000
    split_idx = len(image) - test_data_size


    Xtest = image[-test_data_size:]
    Ytest = label[-test_data_size:]

    Xtrain = image[:split_idx]
    Ytrain = label[:split_idx]

    return Xtrain,Ytrain,Xtest,Ytest
