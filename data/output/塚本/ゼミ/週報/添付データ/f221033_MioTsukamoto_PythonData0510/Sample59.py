class Person:

    def getName(self):
        return  self.name
    
    def getAge(self):
        return self.age
    
pr1 = Person()
pr1.name = "鈴木"
pr1.age = 23
n1 = pr1.getName()
a1 = pr1.getAge()

