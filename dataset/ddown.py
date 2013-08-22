import urllib

print "Starting download train images"
urllib.urlretrieve('http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz', 'train.gz')
print "Starting download train labels"
urllib.urlretrieve('http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz', 'train-labels.gz')
print "Starting download test images"
urllib.urlretrieve('http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz', 'test.gz')
print "Starting download test labels"
urllib.urlretrieve('http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz', 'test-labels.gz')