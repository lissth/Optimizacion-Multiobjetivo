import numpy as np
import matplotlib.pyplot as plt
import csv

# Funciones objetivo problema 1
def FuncionObjetivo1(x):
    return x[0]**2

def FuncionObjetivo2(x):
    return (x[0] - 2)**2

# Restricciones p1
def Restricciones(x):
    ap = 0  
    return ap

def limP01():
    return [(-10**5, 10**5)]

def FuncionObjetivoMeta(x, tk1, tk2, w11, w12, w21, w22):
    n1, n2, p1, p2 = x[1], x[2], x[3], x[4]
    return w11 * p1 + w12 * p2 + w21 * n1 + w22 * n2

#función para no mandar más variables a ED
def FxMeta(tk1, tk2, w11, w12, w21, w22):
  def fx(x):
    return FuncionObjetivoMeta(x, tk1, tk2, w11, w12, w21, w22)
  return fx

def RestriccionesMeta(x, Tk1, Tk2):
    n1, n2, p1, p2 = x[1], x[2], x[3], x[4]
    h1 = x[0]**2 + p1 - n1 - Tk1
    h2 = (x[0] - 2)**2 + p2 - n2 - Tk2

    if abs(h1) <= 0.001:
        h1 = 0
    else:
        h1 = abs(h1)

    if abs(h2) <= 0.001:
        h2 = 0
    else:
        h2 = abs(h2)

    return h1 + h2

#función para no mandar más variables a ED
def RMeta(tk1, tk2):
  def Rm(x):
    return RestriccionesMeta(x, tk1, tk2)
  return Rm

def Penalizacion(funcion, ap, individuo):
    if ap == 0:
        z = funcion(individuo) 
    else:
        z = 10000
    return z

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

def EvolucionDiferencial(funcion, restricciones, limites, tamPoblacion, CR, iteraciones):
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
        ap_valores[i] = restricciones(individuo)
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

            aph = restricciones(pobh)
            zh = Penalizacion(funcion, aph, pobh)
            
            app = restricciones(poblacion[i])
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

    mejorPosicion = np.argmin(z)
    mejorSolucionFinal = poblacion[mejorPosicion]
    mejorValorFinal = z[mejorPosicion]

    return mejorSolucionFinal, mejorValorFinal

def Metas(limites, tamPoblacion, CR, iteraciones, numPesos):
    puntosPareto = []

    x1, _ = EvolucionDiferencial(FuncionObjetivo1, Restricciones, limites, tamPoblacion, CR, iteraciones)
    x2, _ = EvolucionDiferencial(FuncionObjetivo2, Restricciones, limites, tamPoblacion, CR, iteraciones)

    #Metas
    tk1 = FuncionObjetivo1(x1)
    tk2 = FuncionObjetivo2(x2)

    for i in range(numPesos):
        # Pesos
        w11 = i / (numPesos - 1)
        w12 = 1 - w11
        w21 = 1 - w11
        w22 = w11
        
        funcionMeta = FxMeta(tk1, tk2, w11, w12, w21, w22)
        restMeta = RMeta(tk1, tk2)
        limitesMeta = [limites[0], (0, 10), (0, 10), (0, 10), (0, 10)] 
        mejorSolucion, _ = EvolucionDiferencial(funcionMeta, restMeta, limitesMeta, tamPoblacion, CR, iteraciones)

        sol = mejorSolucion[0]
        f1_val = FuncionObjetivo1([sol])
        f2_val = FuncionObjetivo2([sol])
        puntosPareto.append([f1_val, f2_val])

    puntosPareto = np.array(puntosPareto)
    
    # Guardar los puntos de Pareto en un archivo CSV
    nombreArchivo = f"metas_p1.csv"
    with open(nombreArchivo, mode='w', newline='') as archivo_csv:
        escritor_csv = csv.writer(archivo_csv)
        escritor_csv.writerow(['f1(x)', 'f2(x)'])  
        escritor_csv.writerows(puntosPareto)  

    # Graficar
    plt.figure(figsize=(8, 6))
    plt.scatter(puntosPareto[:, 0], puntosPareto[:, 1], marker='o', color='pink', label='Frente de Pareto')
    plt.xlabel('f1(x)')
    plt.ylabel('f2(x)')
    plt.title('Frente de Pareto: Programación por Metas')
    plt.grid(True)
    plt.legend()
    plt.show()

"""----------------------------------DATOS----------------------------------"""
limites = limP01()
tamPoblacion = 40
CR = 0.7
iteraciones = 4000
numPesos = 100

Metas(limites, tamPoblacion, CR, iteraciones, numPesos)