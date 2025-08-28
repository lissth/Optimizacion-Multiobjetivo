import numpy as np
import Manejadores, FuncionesObjetivo, Restrcciones
import matplotlib.pyplot as plt
import csv


def Penalizacion(numProblema, ap, individuo, alfa):
    if ap == 0:
        z = FuncionesObjetivo.ObtenerFuncion(numProblema, individuo, alfa) 
    else:
        z = 10000
    return z

def Excedentes(v, limInf, limSup):
    V_ajustado = v.copy()

    ajuste = True
    while ajuste:
        ajuste = False
        for j in range(len(v)):
            valor = V_ajustado[j]
            if V_ajustado[j] < limInf[j]:
                V_ajustado[j] = 2*limInf[j] - V_ajustado[j] 
            elif V_ajustado[j] > limSup[j]:
                V_ajustado[j] = 2*limSup[j] - V_ajustado[j]
            if valor != V_ajustado[j]:
                ajuste = True
    return V_ajustado

def EvolucionDiferencial(numProblema, limites, tamPoblacion, CR, iteraciones, alfa):

    it = 0
    a = 1.0

    canVa = len(limites)
    poblacion = np.random.rand(tamPoblacion, canVa)
    limiteInferior, limiteSuperior = np.asarray(limites).T
    rangodimen=poblacion * (limiteSuperior - limiteInferior)
    poblacion = limiteInferior + rangodimen

    z = np.zeros(len(poblacion))  
    ap_valores = np.zeros(len(poblacion))

    for i, individuo in enumerate(poblacion):
        ap_valores[i] = Restrcciones.ObtenerRestriccion(numProblema, individuo)
        z[i]= Penalizacion(numProblema, ap_valores[i], individuo, alfa)

    while it < iteraciones and a > 1e-6: 

        for i in range(tamPoblacion):
            F = np.random.uniform(0.3, 0.7)
            candidatos = list(range(tamPoblacion))
            candidatos.pop(i)
            n, m, o = np.random.choice(candidatos, 3, replace=False)
            
            Vm = poblacion[n] + F * (poblacion[m] - poblacion[o])
            
            V = Excedentes(Vm, limiteInferior, limiteSuperior)

            pobh = poblacion[i].copy()
            for j in range(canVa):
                if np.random.rand() < CR:
                    pobh[j] = V[j]
                else:
                    pobh[j] = poblacion[i, j]

            aph = Restrcciones.ObtenerRestriccion(numProblema,pobh)
            zh = Penalizacion(numProblema, aph, pobh, alfa)

            ap_valores[i] =  Restrcciones.ObtenerRestriccion(numProblema,poblacion[i])
            z[i] = Penalizacion(numProblema, ap_valores[i], poblacion[i], alfa)

            poblacion = Manejadores.Deb(poblacion, pobh, ap_valores[i], aph, z[i], zh, i)

            if np.array_equal(poblacion[i], pobh):
                 z[i] = zh

        it += 1
        if max(z) == 10000 or min(z) == 10000: 
            a = 1.0
        else:
            a=abs(min(z)-max(z))

    mejorPosicion = np.argmin(z)  #Es el índice del mejor individuo (posición)
    mejorSolucionFinal = poblacion[mejorPosicion]  #Obtiene la mejor solución
    mejorValorFinal = z[mejorPosicion]  #Obtiene el mejor valor objetivo
    
    return mejorSolucionFinal, mejorValorFinal
    
def PuntosPareto(numProblema, limites, tamPoblacion, CR, iteraciones, alfas):
    puntosPareto = []
    
    for alfa in np.linspace(0, 1, alfas):
        mejorSolucionFinal, _ = EvolucionDiferencial(numProblema, limites, tamPoblacion, CR, iteraciones, alfa)
        
        if numProblema == 1:
            f1_valor = mejorSolucionFinal**2
            f2_valor = (mejorSolucionFinal - 2)**2
        elif numProblema == 2:
            # Evalúa las dos funciones objetivo originales en la mejor solución
            x1 = mejorSolucionFinal[0]
            x2 = mejorSolucionFinal[1]
        
            f1_valor =(4*x1**2) + (4*x2**2)
            f2_valor = (x1-5)**2 + (x2-5)**2
        elif numProblema == 3:
            # Evalúa las dos funciones objetivo originales en la mejor solución
            x1 = mejorSolucionFinal[0]
            x2 = mejorSolucionFinal[1]
        
            f1_valor =(x1**2) + x2
            f2_valor = (x2**2) + x1
        elif numProblema == 4:
            # Evalúa las dos funciones objetivo originales en la mejor solución
            x1 = mejorSolucionFinal[0]
            x2 = mejorSolucionFinal[1]
        
            f1_valor = -(0.4*x1 + 0.3*x2)
            f2_valor = -(x1)
            
        puntosPareto.append([f1_valor, f2_valor])

    puntosPareto = np.array(puntosPareto)
    
    # Guardar los puntos de Pareto en un archivo CSV
    nombreArchivo = f"suma_p{numProblema}.csv";
    with open(nombreArchivo, mode='w', newline='') as archivo_csv:
        escritor_csv = csv.writer(archivo_csv)
        escritor_csv.writerow(['f1(x)', 'f2(x)'])  # Escribir encabezados
        escritor_csv.writerows(puntosPareto)  # Escribir los datos
    print(f"Los puntos del frente de Pareto se han guardado en el archivo: {nombreArchivo}")

    # Grafica el frente de Pareto
    plt.figure(figsize=(8, 6))
    plt.scatter(puntosPareto[:, 0], puntosPareto[:, 1], marker='o', color='red', label='Frente de Pareto')
    plt.xlabel('f1(x)')
    plt.ylabel('f2(x)')
    plt.title('Frente de Pareto con Suma Ponderada')
    plt.grid(True)
    plt.legend()
    plt.show()

def Iniciar(limitesProblema, numProblema):
    limites = limitesProblema
    tamPoblacion = 40
    CR = .7 
    iteraciones = 4000
    alfas = 100

    PuntosPareto(numProblema, limites, tamPoblacion, CR, iteraciones, alfas)
