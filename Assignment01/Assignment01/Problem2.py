import sys
import numpy as voNP
import matplotlib.pyplot as voPlot

def GMProblem2( ) :
    print( "Problem 2" )    

    kdX = voNP.ndarray( ( 6, 1 ) )      # X Coordinates
    kdY = voNP.ndarray( ( 6, 1 ) )      # Y Coordinates
    kdC = voNP.ndarray( ( 3, 3 ) )      # Coefficient Matrix
    kdR = voNP.ndarray( ( 3, 1 ) )      # Result Matrix

    # Define X coordinates
    kdX[ 0 ] = [ 1 ]
    kdX[ 1 ] = [ 2 ]
    kdX[ 2 ] = [ 3 ]
    kdX[ 3 ] = [ 4 ]
    kdX[ 4 ] = [ 5 ]
    kdX[ 5 ] = [ 6 ]

    # Define Y coordinates
    kdY[ 0 ] = [  3.2 ]
    kdY[ 1 ] = [  6.4 ]
    kdY[ 2 ] = [ 10.5 ]
    kdY[ 3 ] = [ 17.7 ]
    kdY[ 4 ] = [ 28.1 ]
    kdY[ 5 ] = [ 38.5 ]

    # Initialize Coefficient and Result Matrices to 0
    for kiRow in range( 0, 3, 1 ) :
        kdR[ kiRow, 0 ] = 0.0
        for kiCol in range( 0, 3, 1 ) :
            kdC[ kiRow, kiCol ] = 0.0

    # Calculate Coefficient and Result from derivate of E with respect to coefficient a
    for kiIdx in range( 0, 6, 1 ) :
        kdC[ 0, 0 ] += 2 * ( kdX[ kiIdx ] * kdX[ kiIdx ] * kdX[ kiIdx ] * kdX[ kiIdx ] )    # 2ax^4
        kdC[ 0, 1 ] += 2 * ( kdX[ kiIdx ] * kdX[ kiIdx ] * kdX[ kiIdx ] )                   # 2bx^3
        kdC[ 0, 2 ] += 2 * ( kdX[ kiIdx ] * kdX[ kiIdx ] )                                  # 2cx^2                                                 
        kdR[ 0, 0 ] += 2 * ( kdX[ kiIdx ] * kdX[ kiIdx ] * kdY[ kiIdx ] )                   # 2yx^2
    
    # Calculate Coefficient and Result from derivate of E with respect to coefficient b  
    for kiIdx in range( 0, 6, 1 ) : 
        kdC[ 1, 0 ] += 2 * ( kdX[ kiIdx ] * kdX[ kiIdx ] * kdX[ kiIdx ] )                   # 2ax^3
        kdC[ 1, 1 ] += 2 * ( kdX[ kiIdx ] * kdX[ kiIdx ] )                                  # 2bx^2
        kdC[ 1, 2 ] += 2 * ( kdX[ kiIdx ] )                                                 # 2cx^1
        kdR[ 1, 0 ] += 2 * ( kdX[ kiIdx ] * kdY[ kiIdx ] )                                  # 2yx^1

    # Calculate Coefficient and Result from derivate of E with respect to coefficient c
    for kiIdx in range( 0, 6, 1 ) : 
        kdC[ 2, 0 ] += 2 * ( kdX[ kiIdx ] * kdX[ kiIdx ] )                                  # 2ax^2
        kdC[ 2, 1 ] += 2 * ( kdX[ kiIdx ] )                                                 # 2bx^1
        kdC[ 2, 2 ] += 2 * ( kdX[ kiIdx ] )                                                 # 2c
        kdR[ 2, 0 ] += 2 * ( kdY[ kiIdx ] )                                                 # 2y

    # Compute the inverse of the Coefficient Matrix:
    kdCi = voNP.linalg.inv( kdC )

    # Compute the Coefficient Vector by taking the dot product of the Coefficient Matrix with the Result Matrix
    kdM = voNP.dot( kdCi, kdR )

    # Print the Coefficients
    print( "a = ", kdM[ 0, 0 ], " b = ", kdM[ 1, 0 ], " c = ", kdM[ 2, 0 ] )

    # do a scatter plot of the data
    kiArea = 3
    koColors =['black']
    voPlot.scatter( kdX, kdY, s = kiArea, c = koColors, alpha = 0.5, linewidths = 8 )
    voPlot.title( 'Linear Least Squares Regression' )
    voPlot.xlabel( 'x' )
    voPlot.ylabel( 'y' )

    #plot the fitted line
    kdFitted = ( kdX * kdX ) * kdM[ 0, 0 ] + ( kdX ) * kdM[ 1, 0 ] + kdM[ 2, 0 ]
    koLine,=voPlot.plot( kdX, kdFitted, '--', linewidth = 2 ) #line plot
    koLine.set_color( 'red' )
    voPlot.show( )
