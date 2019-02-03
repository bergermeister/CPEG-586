class Student :
    def __init__( self, firstName, lastName, id ) :
        self.firstName = firstName
        self.lastName = lastName
        self.id = id
        self.tests = [ ] #empty list
        self.grade = ""

    def addTestScore( self, score ) :
        self.tests.append( score )

    def computeGrade( self ) :
        sum = 0
        for testscore in self.tests :
            sum += int( testscore )
        avg = sum / len( self.tests )
        grade = ""
        if avg > 90 :
            grade = "A"
        elif avg > 85 :
            grade = "A-"
        elif avg > 80 :
            grade = "B"
        else :
            grade = "B"
        return( grade )

