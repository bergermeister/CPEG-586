import sys
import numpy as voNP

# Neuron Layer Class
class TcNeuronLayer( ) :
    def __init__( aorSelf, aorShape, adLR, aiActivation ) :
        # Initialize matrix of weights and biases
        aorSelf.vdW = voNP.random.uniform( low=-0.1, high=0.1, size=aorShape )
        aorSelf.vdB = voNP.random.uniform( low=-1, high=1, size=( aorShape[ 0 ], 1 ) )
        aorSelf.vdLR = adLR;
        aorSelf.viAct = aiActivation

    def MForwardPass( aorSelf, adX ) :
        kdSum = aorSelf.MSummation( adX )
        kdOut = aorSelf.MActivation( kdSum )
        return( kdOut )
    
    def MSummation( aorSelf, adX ) :
        return( voNP.dot( aorSelf.vdW, adX ) + aorSelf.vdB )

    def MActivation( aorSelf, adActual ) :
        # Tan Hyperbolic
        if aorSelf.viAct == 1 :
            kdOut = aorSelf.MTanH( adActual )
        
        # Rectified Linear Unit
        elif aorSelf.viAct == 2 :
            kdOut = aorSelf.MRLU( adActual )

        # Sigmoid
        else :
            kdOut = aorSelf.MSigmoid( adActual )

        return( kdOut )

    def MSigmoid( aorSelf, adActual ) :
        return( 1 / ( 1 + voNP.exp( -1 * adActual ) ) )

    def MTanH( aorSelf, adActual ) :        
        kdResult = voNP.tanh( adActual )

        # Causes underflow when calculating kdE2
        # kdE1 = voNP.exp( adActual )
        # kdE2 = 1 / voNP.exp( adActual )
        # kdResult = ( kdE1 - kdE2 ) / ( kdE1 + kdE2 )

        return( kdResult )

    def MRLU( aorSelf, adActual ) :
        kdResult = adActual
        for kiIdx in range( len( kdResult ) ) :
            if kdResult[ kiIdx ] <= 0 :
                kdResult[ kiIdx ] = 0
            else :
                kdResult[ kiIdx ] = 1
        return( kdResult )

    def MBackpropagate( aorSelf, adGradW, adGradB ) :
        aorSelf.vdW = aorSelf.vdW - ( aorSelf.vdLR * adGradW )
        aorSelf.vdB = aorSelf.vdB - ( aorSelf.vdLR * adGradB )
