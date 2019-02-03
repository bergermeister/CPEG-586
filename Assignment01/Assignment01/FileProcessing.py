import sys
from Student import Student

def displayFile( ) :
    fobj = open( "Data.txt", "r" )
    for line in fobj :
        print( line )

def copyFile( ) :
    fobj = open( "Data.txt", "r" )
    fobj2 = open( "Data2.txt", "w" )
    for line in fobj :
        print( line )
        fobj2.write( line )
    fobj2.write( "\n12344" + "\t" + "Gerard" + "\t" + "Way" + "\t" + "96" + "\t" + "89" )
    fobj2.close( )

def processGrades( ) :
    fobj = open( "Data.txt", "r" )
    fobj2 = open( "StudentGrades.txt", "w" )
    for line in fobj :
        parts = line.split( '\t' )
        s1 = Student( "", "", 0 )
        s1.id = parts[ 0 ]
        s1.firstName = parts[ 1 ]
        s1.lastName = parts[ 2 ]
        s1.addTestScore( parts[ 3 ] )
        s1.addTestScore( parts[ 4 ] )
        s1.grade = s1.computeGrade( )
        # now write id and grade to an output file
        fobj2.write( s1.id + "\t" + s1.lastName + "\t" + s1.grade + "\n" )
    fobj2.close( )
    print( "Done processing processGrades..." )

