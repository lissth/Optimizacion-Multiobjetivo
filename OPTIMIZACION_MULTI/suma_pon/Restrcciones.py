def ObtenerRestriccion(numProblema, x):
    if numProblema == 1:
        ap = 0
    elif numProblema == 2:
        g1 = max(0, (x[0] - 5)**2 + x[1]**2 - 25)
        g2 = max(0, 7.7 - ((x[0] - 8)**3 + (x[1] + 3)**2))
        ap = g1 + g2

    elif numProblema == 3:
        ap = 0
        
    elif numProblema == 4:
        g1 = max(0, x[0] + x[1] - 400)
        g2 = max(0, 2*x[0] + x[1] - 500)
        ap = g1 + g2
    return ap