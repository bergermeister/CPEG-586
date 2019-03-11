import numpy as voNP
from TcTypeActivation import TcTypeActivation

# Neuron Layer Class
class TcNeuronLayer( object ) :
    def __init__( aorSelf, aorShape, adLR, abLast = False, adDropOut = 0.2, aeActivation = TcTypeActivation.XeSigmoid ) :
        # Initialize matrix of weights and biases
        aorSelf.vdW = voNP.random.uniform( low=-0.1, high=0.1, size=aorShape )
        aorSelf.vdB = voNP.random.uniform( low=-1, high=1, size=( aorShape[ 0 ], 1 ) )

        # Store learning rate, last layer flag, drop out rate, and activation type
        aorSelf.vdLR = adLR;
        aorSelf.vbLast = abLast
        aorSelf.vdDropOut = adDropOut
        aorSelf.veAct = aeActivation

        # Initialize Result (A) and derivative to zero vector
        aorSelf.vdA = voNP.zeros( ( aorShape[ 0 ], 1 ) )
        aorSelf.vdAd = voNP.zeros( ( aorShape[ 0 ], 1 ) )

    def MForwardPass( aorSelf, adX ) :
        kdSum       = aorSelf.MSummation( adX )
        aorSelf.vdA = aorSelf.MActivation( kdSum )
        return( aorSelf.vdA )
    
    def MSummation( aorSelf, adX ) :
        return( voNP.dot( aorSelf.vdW, adX ) + aorSelf.vdB )

    def MActivation( aorSelf, adActual ) :
        # Tan Hyperbolic
        if aorSelf.veAct == TcTypeActivation.XeTanH :
            kdOut = aorSelf.MTanH( adActual )
        # Rectified Linear Unit
        elif aorSelf.veAct == TcTypeActivation.XeRELU :
            kdOut = aorSelf.MRELU( adActual )
        # SoftMax
        elif aorSelf.veAct == TcTypeActivation.XeSoftMax :
            kdOut = aorSelf.MSoftMax( adActual )
        # Sigmoid
        else : # aorSelf.veAct == TeTypeActivation.XeSigmoid
            kdOut = aorSelf.MSigmoid( adActual )

        return( kdOut )

    def MSigmoid( aorSelf, adActual ) :
        return( 1 / ( 1 + voNP.exp( -adActual ) ) )

    def MTanH( aorSelf, adActual ) :        
        return( voNP.tanh( adActual ) )

    def MRELU( aorSelf, adActual ) :
        return( voNP.maximum( 0, adActual ) )

    def MSoftMax( aorSelf, adActual ) :
        kdE = voNP.exp( adActual )
        return( kdE / kdE.sum( ) )

    def MBackpropagate( aorSelf, adGradW, adGradB ) :
        aorSelf.vdW = aorSelf.vdW - ( aorSelf.vdLR * adGradW )
        aorSelf.vdB = aorSelf.vdB - ( aorSelf.vdLR * adGradB )
