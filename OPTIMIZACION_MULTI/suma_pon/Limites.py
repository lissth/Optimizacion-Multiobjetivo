def ObtenerLimites(numProblema):
    if numProblema == 1:
        limite = [(-10**5, 10**5)]

    elif numProblema == 2:
        limite = [(0, 5), (0, 3)]

    elif numProblema == 3:
        limite = [(-5, 5), (-3, 3)]

    elif numProblema == 4:
        limite = [(0, 550), (0, 550)]

    return limite