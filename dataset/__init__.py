import gzip
import numpy

# We know that images are 28 on 28
CHUNK_SIZE=28 * 28


def sample_iter(filename,  chunk_size = CHUNK_SIZE):

    # @filename Gzip filename

    # opening filename as gzip file. 
    with gzip.open(filename, "rb") as sample_file:

        # look at http://yann.lecun.com/exdb/mnist/
        # As we already know what's  in this file we just ignoring it
        sample_file.read(16)

        while True:

            # It would be faster to read in bigger chunks but that's just simple, and solves for now. 
            chunk =  sample_file.read(chunk_size)

            if chunk:

                # http://stackoverflow.com/questions/4090981/how-do-i-create-a-numpy-array-from-string

                yield numpy.fromstring(chunk, dtype="uint8")

            else:

                # If we reached end of file then go out of while cycle
                break


def label_iter(filename,chunksize=100):

    with gzip.open(filename, "rb") as label_file:

        label_file.read(8)
        while True:
            chunk = label_file.read(chunksize)
            if chunk:
                for b in chunk:
                    # Getting int value of that byte/character. That's complicated
                    yield ord(b)
            else:
                break
    
