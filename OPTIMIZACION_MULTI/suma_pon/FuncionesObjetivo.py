def ObtenerFuncion(numProblema, x, alfa):
    if numProblema == 1:
        Fx = (alfa * (x**2)) + ((1 - alfa) * (x - 2)**2)
    elif numProblema == 2:
        f1 =(4*x[0]**2) + (4*x[1]**2)
        f2 =(x[0]-5)**2 + (x[1]-5)**2
        Fx = (alfa * f1) + ((1 - alfa) * f2)
    
    elif numProblema == 3:
        f1 = (x[0]**2) + x[1]
        f2 = (x[1]**2) + x[0]
        Fx = (alfa * f1) + ((1 - alfa) * f2)
        
    elif numProblema == 4:
        f1 = -(0.4*x[0] + 0.3*x[1])
        f2 = -(x[0])
        Fx = (alfa * f1) + ((1 - alfa) * f2)
    return Fx