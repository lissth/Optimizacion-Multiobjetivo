import numpy as np
import matplotlib.pyplot as plt
import csv

# Funciones objetivo

def FuncionObjetivo1(x):
    return x[0]**2

def FuncionObjetivo2(x):
    return (x[0] - 2)**2

def Restricciones(x):
    return 0

def limP01():
    return [(-10**5, 10**5)]

def Penalizacion(funcion1, funcion2, ap, individuo):
    if ap == 0:
        return np.array([funcion1(individuo), funcion2(individuo)])
    else:
        return np.array([10000, 10000])

def Excedentes(v, limInf, limSup):
    V_ajustado = v.copy()
    ajuste = True
    while ajuste:
        ajuste = False
        for j in range(len(v)):
            if V_ajustado[j] < limInf[j]:
                V_ajustado[j] = 2 * limInf[j] - V_ajustado[j]
                ajuste = True
            elif V_ajustado[j] > limSup[j]:
                V_ajustado[j] = 2 * limSup[j] - V_ajustado[j]
                ajuste = True
    return V_ajustado

def Dominante(a, b):
    a = np.array(a)
    b = np.array(b)
    menorIgual = True
    menor = False
    for i in range(len(a)):
        if a[i] > b[i]:
            menorIgual = False
            break
        elif a[i] < b[i]:
            menor = True
    return menorIgual and menor

def Distancia(frentep):
    frentep = np.array(frentep)
    n = len(frentep)
    distancias = np.zeros(n)
    obj = frentep.shape[1]
    for m in range(obj):
        orden = np.argsort(frentep[:, m])
        f_min = frentep[orden[0], m]
        f_max = frentep[orden[-1], m]
        rango = f_max - f_min
        if rango == 0:
            continue
        distancias[orden[0]] += 2 * abs(frentep[orden[1], m] - frentep[orden[0], m]) / rango
        distancias[orden[-1]] += 2 * abs(frentep[orden[-1], m] - frentep[orden[-2], m]) / rango
        for i in range(1, n - 1):
            distancias[orden[i]] += abs(frentep[orden[i + 1], m] - frentep[orden[i - 1], m]) / rango
    return distancias

def ActualizarAlmacen(almacen, candidato):
    almacen_nuevo = []
    insertar = True
    existe = False
    for punto in almacen:
        if Dominante(punto, candidato):
            insertar = False
            almacen_nuevo.append(punto)
        elif not Dominante(candidato, punto):
            almacen_nuevo.append(punto)
    if insertar:
        for punto_almacen in almacen_nuevo:
            if np.array_equal(candidato, punto_almacen):
                existe = True
                break
        if not existe:
            almacen_nuevo.append(candidato)
    return almacen_nuevo

def Deb(pobp, pobh, app, aph, zp, zh, i, almacen):
    if app == 0 and aph == 0:
        if Dominante(zh, zp):
            pobp[i] = pobh
            zp = zh
        elif not Dominante(zp, zh) and not Dominante(zh, zp):
            temp = [zp, zh]
            distancias = Distancia(temp)
            if distancias[1] > distancias[0]:
                pobp[i] = pobh
                zp = zh
        almacen = ActualizarAlmacen(almacen, zp)
        almacen = ActualizarAlmacen(almacen, zh)
    elif app != 0 and aph == 0:
        pobp[i] = pobh
        zp = zh
        almacen = ActualizarAlmacen(almacen, zh)
    elif app == 0 and aph != 0:
        almacen = ActualizarAlmacen(almacen, zp)
    else:
        if aph < app:
            pobp[i] = pobh
            zp = zh
            if aph == 0:
                almacen = ActualizarAlmacen(almacen, zh)
        else:
            if app == 0:
                almacen = ActualizarAlmacen(almacen, zp)
    return pobp, zp, app, almacen

def EvolucionDiferencial(funcion1, funcion2, restricciones, limites, tamPoblacion, CR, iteraciones):
    canVa = len(limites)
    poblacion = np.random.rand(tamPoblacion, canVa)
    limiteInferior, limiteSuperior = np.asarray(limites).T
    poblacion = limiteInferior + poblacion * (limiteSuperior - limiteInferior)
    z = np.zeros((tamPoblacion, 2))
    ap_valores = np.zeros(tamPoblacion)
    almacen = []
    for i, individuo in enumerate(poblacion):
        ap_valores[i] = restricciones(individuo)
        z[i] = Penalizacion(funcion1, funcion2, ap_valores[i], individuo)
        if ap_valores[i] == 0:
            almacen = ActualizarAlmacen(almacen, z[i])
    for it in range(iteraciones):
        for i in range(tamPoblacion):
            F = np.random.uniform(0.3, 0.7)
            candidatos = list(range(tamPoblacion))
            candidatos.remove(i)
            n, m, o = np.random.choice(candidatos, 3, replace=False)
            Vm = poblacion[n] + F * (poblacion[m] - poblacion[o])
            V = Excedentes(Vm, limiteInferior, limiteSuperior)
            pobh = poblacion[i].copy()
            for j in range(canVa):
                if np.random.rand() < CR:
                    pobh[j] = V[j]
            aph = restricciones(pobh)
            zh = Penalizacion(funcion1, funcion2, aph, pobh)
            poblacion, z[i], ap_valores[i], almacen = Deb(poblacion, pobh, ap_valores[i], aph, z[i], zh, i, almacen)
    return almacen

def Crowding(funcion1, funcion2, restricciones, limites, tamPoblacion, CR, iteraciones):
    pareto = EvolucionDiferencial(funcion1, funcion2, restricciones, limites, tamPoblacion, CR, iteraciones)
    pareto = np.array(pareto)
    
    # Guardar los puntos de Pareto en un CSV
    nombreArchivo = f"crowding_p1.csv"
    with open(nombreArchivo, mode='w', newline='') as archivo_csv:
        escritor_csv = csv.writer(archivo_csv)
        escritor_csv.writerow(['f1(x)', 'f2(x)'])  
        escritor_csv.writerows(pareto)  
    
    plt.figure(figsize=(8, 6))
    plt.scatter(pareto[:, 0], pareto[:, 1], color='black', label='Frente de Pareto')
    plt.xlabel('f1(x)')
    plt.ylabel('f2(x)')
    plt.title('Frente de Pareto: Crowding')
    plt.grid(True)
    plt.legend()
    plt.show()

"""----------------------------------DATOS----------------------------------"""
limites = limP01()
tamPoblacion = 40
CR = 0.7
iteraciones = 500
Crowding(FuncionObjetivo1, FuncionObjetivo2, Restricciones, limites, tamPoblacion, CR, iteraciones)