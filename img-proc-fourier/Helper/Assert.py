
class Assert(object):

    def __init__(self, params):
        pass
        '''
        Constructor
        '''


    @staticmethod
    def isTrue(b, message):
        """
         asserts, that a condition is true. If the assertion is violated, a RuntimeError is thrown. The error message must state,
         that the assertion is violated
         @param b: condition that should be checked
         @param message: that states that assertion is violated
        """
        if b != True:
            raise RuntimeError(message)  