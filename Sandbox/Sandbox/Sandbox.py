from math import pi
import sys

# First Python Application

def computeAvg(a, b, c) :
    return( a + b + c ) / 3.0; # use // for integer division

def doComplexMath( ) :
    num1 = 3 + 4j
    num2 = 6 + 3.5j
    res = num1 * num2
    return( res );

def main( ):
    print( "Result = ", computeAvg( 5, 9, 23 ) )
    print( "Result Complex = ", doComplexMath( ) )

if __name__ == "__main__" :
    sys.exit( int( main( ) or 0 ) )
