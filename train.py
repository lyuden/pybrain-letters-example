from pybrain.datasets            import ClassificationDataSet
from pybrain.utilities           import percentError
from pybrain.tools.shortcuts     import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure.modules   import SoftmaxLayer

from itertools import izip

from dataset import sample_iter, label_iter

from dataset.show import get_random_sample_with_label, print_sample

import pickle
import time

#http://pybrain.org/docs/tutorial/fnn.html

train = sample_iter('dataset/train.gz')
train_labels = label_iter('dataset/train-labels.gz')

test = sample_iter('dataset/test.gz')
test_labels = label_iter('dataset/test-labels.gz')



def create_dataset(sample_iterator, label_iterator):

    #http://pybrain.org/docs/tutorial/datasets.html 1 is magic parameter that is not documented
    # see http://stackoverflow.com/questions/14917300/pybrain-feedforward-neural-network-training-error-completely-stuck
    dataset =  ClassificationDataSet(28 * 28,1, nb_classes = 10)

    for inp,out in izip(sample_iterator, label_iterator):
        #print inp, out 
        dataset.addSample(inp, [out])

    #Some Pybrain Magic
    dataset._convertToOneOfMany()
    return dataset

    
train_dataset = create_dataset(train, train_labels)
test_dataset = create_dataset(test, test_labels)


# 784=28*28 inputs 20 middle layer, 10 outputs, Softmax is recommended for classification
network = buildNetwork(784,100, 20,10, outclass=SoftmaxLayer)

trainer =  BackpropTrainer(network, dataset = train_dataset,  weightdecay=0.01)


def one_iteration():

    print "Training"
    trainer.trainEpochs(1)

    print "Testing on testing dataset"

    tstresult = percentError(trainer.testOnClassData(dataset = test_dataset),test_dataset['class'])
    print "Error %5.2f" % tstresult

    rsamples = [get_random_sample_with_label() for i in range(10)]
        
    rsample,  rlabel = zip(*rsamples)

    with open("nn" + str(time.time()), 'w') as nndump:

        pickle.dump(network, nndump)

    out = network.activateOnDataset(create_dataset(rsample, rlabel))

    print out,  rlabel
    for sample, label, prediction in zip(rsample, rlabel, out.argmax(axis =1)):
        
    
        print_sample(sample)
        
        print "Sample labeled as ",  label

        print "Neural network says it's a ", prediction


if __name__ == "__main__":
    
    for i in range(1):
        print "Starting iteration %d" % i

        one_iteration()

    

    




    