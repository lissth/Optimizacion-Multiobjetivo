import numpy as np
import matplotlib.pyplot as plt
import csv

# Funciones objetivo problema 4
def FuncionObjetivo1(x):
    return -(0.4*x[0] + 0.3*x[1])

def FuncionObjetivo2(x):
    return -(x[0])

# Restricciones f1
def Restricciones(x, f1, alfa):
    g1 = max(0, x[0] + x[1] - 400)
    g2 = max(0, 2*x[0] + x[1] - 500)
    ap = g1 + g2
    return ap

# Restricciones f2
def Restricciones2(x, f1, alfa):
    h1 = FuncionObjetivo1(x) - (f1 + alfa)
    if abs(h1) <= 0.001:
        h1 = 0
    
    g1 = max(0, x[0] + x[1] - 400)
    g2 = max(0, 2*x[0] + x[1] - 500)
    ap = g1 + g2 + abs(h1)
    return ap

def limP01():
    return [(0, 550), (0, 550)]

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

def EvolucionDiferencial(funcion, restricciones, limites, tamPoblacion, CR, iteraciones, f1, alfa):
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
        ap_valores[i] = restricciones(individuo, f1, alfa)
        z[i] = Penalizacion(funcion, ap_valores[i], individuo)
    
    while it < iteraciones and a > 1e-11: 
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
            
            aph = restricciones(pobh, f1, alfa)
            zh = Penalizacion(funcion, aph, pobh)
            
            app = restricciones(poblacion[i], f1, alfa)
            zp = Penalizacion(funcion, app, poblacion[i])

            poblacion = Deb(poblacion, pobh, app, aph, zp, zh, i)

            if np.array_equal(poblacion[i], pobh):
                z[i] = zh
                ap_valores[i] = aph

        it += 1
        if max(z) == 10000 or min(z) == 10000: 
            a = 1.0
        else:
            a=abs(min(z)-max(z))

    mejorPosicion = np.argmin(z)  #Es el índice del mejor individuo (posición)
    mejorSolucionFinal = poblacion[mejorPosicion]  #Obtiene la mejor solución
    mejorValorFinal = z[mejorPosicion]

    return mejorSolucionFinal, mejorValorFinal

def Lexico(FuncionObjetivo1, FuncionObjetivo2, Restricciones, Restricciones2, limites, tamPoblacion, CR, iteraciones, puntos):
    puntosPareto = []

    mejorSol1, _ = EvolucionDiferencial(FuncionObjetivo1, Restricciones, limites, tamPoblacion, CR, iteraciones, 0, 0)
    meta1 = FuncionObjetivo1(mejorSol1)
    
    for alfa in np.linspace(0, 30, puntos):
    
        mejorSol2, _ = EvolucionDiferencial(FuncionObjetivo2, Restricciones2, limites, tamPoblacion, CR, iteraciones, meta1, alfa)

        f1_val = FuncionObjetivo1(mejorSol2)
        f2_val = FuncionObjetivo2(mejorSol2)
        puntosPareto.append([f1_val, f2_val])
        
    puntosPareto = np.array(puntosPareto)
    
    # Guardar los puntos de Pareto en un CSV
    nombreArchivo = f"lexico_p4.csv"
    with open(nombreArchivo, mode='w', newline='') as archivo_csv:
        escritor_csv = csv.writer(archivo_csv)
        escritor_csv.writerow(['f1(x)', 'f2(x)'])  
        escritor_csv.writerows(puntosPareto)  

    plt.figure(figsize=(8, 6))
    plt.scatter(puntosPareto[:, 0], puntosPareto[:, 1], marker='o', color='green', label='Frente de Pareto')
    plt.xlabel('f1(x)')
    plt.ylabel('f2(x)')
    plt.title('Frente de Pareto con Metas Lexicográficas')
    plt.grid(True)
    plt.legend()
    plt.show()

"""----------------------------------DATOS----------------------------------"""
limites = limP01()
tamPoblacion = 40
CR = .7 
iteraciones = 4000
puntos = 100

Lexico(FuncionObjetivo1, FuncionObjetivo2, Restricciones, Restricciones2, limites, tamPoblacion, CR, iteraciones, puntos)