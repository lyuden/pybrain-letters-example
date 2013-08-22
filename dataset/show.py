import numpy
from itertools import islice
from dataset import sample_iter, label_iter

def nth(iterable, n, default=None):
    '''
    http://docs.python.org/2/library/itertools.html
    '''
    "Returns the nth item or a default value"
    return next(islice(iterable, n, None), default)

def print_sample(sample):
    r_sample = sample.reshape((28, 28))

    for line in r_sample:

        print "".join( "8" if x else "-" for x in line )


def get_random_sample_with_label():
    
    
    sample_number = numpy.random.randint(60000)

    sample = nth(sample_iter('dataset/train.gz'), sample_number)
    label = nth(label_iter('dataset/train-labels.gz'), sample_number)

    return sample,label
    
def show_random_sample(sample_number=-1):

    if sample_number == - 1:
        sample_number = numpy.random.randint(60000)

    sample = nth(sample_iter('dataset/train.gz'), sample_number)

    print_sample(sample)

    print "It is a %d" % nth(label_iter('dataset/train-labels.gz'), sample_number)


    
