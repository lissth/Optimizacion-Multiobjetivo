import numpy as np
import matplotlib.pyplot as plt
import csv

# DATOS DEL EJERCICIO 2
def FuncionObjetvo1(x):
    F1 =(4*x[0]**2) + (4*x[1]**2)
    return F1

def FuncionObjetvo2(x):
    F2 =(x[0]-5)**2 + (x[1]-5)**2
    return F2

def Restricciones(x, e):
    g1 = max(0, (x[0] - 5)**2 + x[1]**2 - 25)
    g2 = max(0, 7.7 - ((x[0] - 8)**3 + (x[1] + 3)**2))
    ap = g1 + g2
    return ap

def Restricciones2(x, e2): 
    f2 = FuncionObjetvo2(x)
    F2 = max(0, f2 - e2)
    g1 = max(0, (x[0] - 5)**2 + x[1]**2 - 25)
    g2 = max(0, 7.7 - ((x[0] - 8)**3 + (x[1] + 3)**2))
    ap = g1 + g2 + F2
    
    return ap

def limP01():
    p1 = [(0, 5), (0, 3)]
    return p1

def Penalizacion(funcion, ap, individuo):
    if ap == 0:
        z = funcion(individuo) 
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

def Deb(pobp, pobh, app, aph, zp, zh, i ):

    if app == 0 and aph == 0:
        if zp <= zh:
            pobp[i] = pobp[i]  #te quedas con el individuo papá
        
        else:
            pobp[i] = pobh   #te quedas con el individuo hijo
    elif app == 0 and aph != 0:
        pobp[i] = pobp[i]

    elif app != 0 and aph == 0:
        pobp[i] = pobh

    elif aph <= app:
        pobp[i] = pobh

    else:
        pobp[i] = pobp[i]

    return pobp

def EvolucionDiferencial(funcion1, restricciones, limites, tamPoblacion, CR, iteraciones, e):

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
        ap_valores[i] = restricciones(individuo, e)
        z[i]= Penalizacion(funcion1, ap_valores[i], individuo)

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

            aph = restricciones(pobh, e)
            zh = Penalizacion(funcion1, aph, pobh)

            ap_valores[i] = restricciones(poblacion[i], e)
            z[i] = Penalizacion(funcion1, ap_valores[i], poblacion[i])
            
            poblacion = Deb(poblacion, pobh, ap_valores[i], aph, z[i], zh, i)

            if np.array_equal(poblacion[i], pobh):
                 z[i] = zh

        it += 1
        if max(z) == 10000 or min(z) == 10000: 
            a = 1.0
        else:
            a=abs(min(z)-max(z))
    
    mejorPosicion = np.argmin(z)  #Es el índice del mejor individuo (posición)
    peorPosicion = np.argmax(z)
    mejorSolucionFinal = poblacion[mejorPosicion]  #Obtiene la mejor solución
    mejorValorFinal = z[mejorPosicion]  #Obtiene el mejor valor objetivo
    peorValorFinal = z[peorPosicion]

    return mejorSolucionFinal, mejorValorFinal, peorValorFinal

def EConstraint(FuncionObjetvo1, FuncionObjetvo2, Restricciones,Restricciones2, limites, tamPoblacion, CR, iteraciones):
    puntosPareto = []
    
    _, e2, _ = EvolucionDiferencial(FuncionObjetvo2, Restricciones, limites, tamPoblacion, CR, iteraciones, 0)
    
    #emax = 100
    emax= 50

    for e in np.linspace(e2, emax, 200):
        mejorSolucionFinal, _, _ = EvolucionDiferencial(FuncionObjetvo1, Restricciones2, limites, tamPoblacion, CR, iteraciones, e)

        # Evalúa las dos funciones objetivo originales en la mejor solución
        f1_valor = FuncionObjetvo1(mejorSolucionFinal)
        f2_valor = FuncionObjetvo2(mejorSolucionFinal)
        puntosPareto.append([f1_valor, f2_valor])

    puntosPareto = np.array(puntosPareto)
    
    # Guardar los puntos de Pareto en un archivo CSV
    nombreArchivo = f"econs_p2.csv";
    with open(nombreArchivo, mode='w', newline='') as archivo_csv:
        escritor_csv = csv.writer(archivo_csv)
        escritor_csv.writerow(['f1(x)', 'f2(x)'])  # Escribir encabezados
        escritor_csv.writerows(puntosPareto)  # Escribir los datos
    print(f"Los puntos del frente de Pareto se han guardado en el archivo: {nombreArchivo}")

    # Grafica el frente de Pareto
    plt.figure(figsize=(8, 6))
    plt.scatter(puntosPareto[:, 0], puntosPareto[:, 1], marker='o', color='blue', label='Frente de Pareto')
    plt.xlabel('f1(x)')
    plt.ylabel('f2(x)')
    plt.title('Frente de Pareto con E-Constrain')
    plt.grid(True)
    plt.legend()
    plt.show()

"""----------------------------------DATOS----------------------------------"""
limites = limP01()
tamPoblacion = 40
CR = .7 
iteraciones = 4000

EConstraint(FuncionObjetvo1, FuncionObjetvo2, Restricciones, Restricciones2, limites, tamPoblacion, CR, iteraciones)