import numpy as np
import matplotlib.pyplot as plt
import csv

# Funciones objetivo problema 2
def FuncionObjetivo1(x):
    return (4*x[0]**2) + (4*x[1]**2)

def FuncionObjetivo2(x):
    return (x[0]-5)**2 + (x[1]-5)**2

def limP01():
    return [(0, 5), (0, 3)]

# Restricciones p2
def Restricciones(x, w1, w2, lam, meta1, meta2):
    g1 = max(0, (x[0] - 5)**2 + x[1]**2 - 25)
    g2 = max(0, 7.7 - ((x[0] - 8)**3 + (x[1] + 3)**2))
    ap = g1 + g2
    return ap

# Restricciones lambda
def RestriccionesLambda(x, w1, w2, lam, meta1, meta2):
    m1 = (4*x[0]**2) + (4*x[1]**2) - w1 * x[2] - meta1
    m2 = (x[0]-5)**2 + (x[1]-5)**2 - w2 * x[2] - meta2
    r1 = max(0, m1)
    r2 = max(0, m2)

    g1 = max(0, (x[0] - 5)**2 + x[1]**2 - 25)
    g2 = max(0, 7.7 - ((x[0] - 8)**3 + (x[1] + 3)**2))
    return r1 + r2 + g1 + g2

# Función objetivo 
def FunLamda(x):
    return x[2]

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
                V_ajustado[j] = 2 * limInf[j] - V_ajustado[j]
            elif V_ajustado[j] > limSup[j]:
                V_ajustado[j] = 2 * limSup[j] - V_ajustado[j]
            if valor != V_ajustado[j]:
                ajuste = True
    return V_ajustado

def Deb(pobp, pobh, app, aph, zp, zh, i):
    if app == 0 and aph == 0:
        pobp[i] = pobp[i] if zp <= zh else pobh
    elif app == 0 and aph != 0:
        pobp[i] = pobp[i]
    elif app != 0 and aph == 0:
        pobp[i] = pobh
    elif aph <= app:
        pobp[i] = pobh
    else:
        pobp[i] = pobp[i]
    return pobp

def EvolucionDiferencial(funcion, restricciones, limites, tamPoblacion, CR, iteraciones, w1, w2, meta1, meta2):
    it = 0
    a = 1.0

    canVa = len(limites)
    poblacion = np.random.rand(tamPoblacion, canVa)
    limiteInferior, limiteSuperior = np.asarray(limites).T
    poblacion = limiteInferior + poblacion * (limiteSuperior - limiteInferior)

    z = np.zeros(len(poblacion))
    ap_valores = np.zeros(len(poblacion))

    for i, individuo in enumerate(poblacion):
        ap_valores[i] = restricciones(individuo, w1, w2, 0, meta1, meta2)  
        z[i] = Penalizacion(funcion, ap_valores[i], individuo)

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

            aph = restricciones(pobh, w1, w2, 0, meta1, meta2)
            zh = Penalizacion(funcion, aph, pobh)

            app = restricciones(poblacion[i], w1, w2, 0, meta1, meta2)
            zp = Penalizacion(funcion, app, poblacion[i])

            poblacion = Deb(poblacion, pobh, app, aph, zp, zh, i)

            if np.array_equal(poblacion[i], pobh):
                z[i] = zh
                ap_valores[i] = aph

        it += 1
        if max(z) == 10000 or min(z) == 10000: 
            a = 1.0
        else:
            a = abs(min(z) - max(z))

    mejor_posicion_final = np.argmin(z)
    return poblacion[mejor_posicion_final], z[mejor_posicion_final]


def ObtenMetas(FuncionObjetivo1, FuncionObjetivo2, FunLambda, Restricciones, RestriccionesLambda, limites, tamPoblacion, CR, iteraciones, pesos):
    puntosPareto = []

    tk1, _ = EvolucionDiferencial(FuncionObjetivo1, Restricciones, limites, tamPoblacion, CR, iteraciones, 0, 0, 0, 0)
    tk2, _ = EvolucionDiferencial(FuncionObjetivo2, Restricciones, limites, tamPoblacion, CR, iteraciones, 0, 0, 0, 0)

    meta1 = FuncionObjetivo1(tk1)
    meta2 = FuncionObjetivo2(tk2)

    for w in np.linspace(0, 1, pesos):
        w1 = w
        w2 = 1 - w1

        limites_lambda = [limites[0], limites[1], (0, 150)]  
        mejorSol, _ = EvolucionDiferencial(FunLambda, RestriccionesLambda, limites_lambda, tamPoblacion, CR, iteraciones, w1, w2, meta1, meta2)

        sol = mejorSol[:2]
        f1_val = FuncionObjetivo1(sol)
        f2_val = FuncionObjetivo2(sol)
        puntosPareto.append([f1_val, f2_val])

    puntosPareto = np.array(puntosPareto)
    
    # Guardar los puntos de Pareto en un CSV
    nombreArchivo = f"obten_p2.csv"
    with open(nombreArchivo, mode='w', newline='') as archivo_csv:
        escritor_csv = csv.writer(archivo_csv)
        escritor_csv.writerow(['f1(x)', 'f2(x)'])  
        escritor_csv.writerows(puntosPareto)  
   
    plt.figure(figsize=(8, 6))
    plt.scatter(puntosPareto[:, 0], puntosPareto[:, 1], color='purple', label='Frente de Pareto')
    plt.xlabel('f1(x)')
    plt.ylabel('f2(x)')
    plt.title('Frente de Pareto: Obtención de Metas')
    plt.grid(True)
    plt.legend()
    plt.show()

"""----------------------------------DATOS----------------------------------"""
limites = limP01()
tamPoblacion = 40
CR = .7 
iteraciones = 4000
pesos = 100

ObtenMetas(FuncionObjetivo1, FuncionObjetivo2, FunLamda, Restricciones, RestriccionesLambda, limites, tamPoblacion, CR, iteraciones, pesos)