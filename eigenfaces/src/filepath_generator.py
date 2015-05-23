from math import log10, floor, ceil



def count_zeros(number):
    '''
    Counts how many zeros will have the file name with particular number by file naming convention.
    :param number: file name number
    :return: number of zeros
    '''
    return 4 - int(floor(log10(number)))


def get_filepath_for(number):
    '''
    Creates actual relative file path by file naming convention.
    :param number: file name number
    :return: relative file path
    '''
    zeros = count_zeros(number)        
    
    return "train/face" + ("0" * zeros) + str(number) + ".pgm"     
