from base import PMLM

def construct_PMLM(name, constants):
    """ Constructs new class from PMLM base by fixing some constructor parameters. """
    
    class PMLM_model(PMLM):

        def __init__(self, **kwargs):
            
            for constant in constants:
                if constant in kwargs:
                    raise Exception('{} defines {}={}. If you want to override this value'
                                    ', construct PMLM instead.'.format(name, constant, constants[constant]))
                kwargs[constant] = constants[constant]

            PMLM.__init__(self, **kwargs)
            self._str_fields = [field for field in self._str_fields if field not in constants]

    PMLM_model.__name__ = name
    return PMLM_model 

MLMM = construct_PMLM('MLMM', {'alpha': 0, 'pointwise_loss': 'mle', 'density_loss': None, 'density_averaging': None})
MEMM = construct_PMLM('MEMM', {'alpha': 0, 'pointwise_loss': 'me', 'density_loss': None, 'density_averaging': None})
EDMM = construct_PMLM('EDMM', {'alpha': 0, 'pointwise_loss': 'l2', 'density_loss': None, 'density_averaging': None})

DCSMM = construct_PMLM('DCSMM', {'alpha': 1, 'pointwise_loss': None, 'density_loss': 'dcs'})
DEDMM = construct_PMLM('DEDMM', {'alpha': 1, 'pointwise_loss': None, 'density_loss': 'ded'})
DKLMM = construct_PMLM('DKLMM', {'alpha': 1, 'pointwise_loss': None, 'density_loss': 'dkl'})
