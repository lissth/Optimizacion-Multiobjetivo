import numpy as np
import matplotlib.pyplot as plt
import csv

# Función Objetivo Problema 3
def FuncionObjetivo1(x):
    return (x[0]**2) + x[1]

def FuncionObjetivo2(x):
    return (x[1]**2) + x[0]

def Restricciones(x):
    return 0  

def limP01():
    return [(-5, 5), (-3, 3)]

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
            valor = V_ajustado[j]
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
            break  # Si un elemento de 'a' es mayor que su correspondiente en 'b', 'a' no domina.
        elif a[i] < b[i]:
            menor = True

    return menorIgual and menor

def ActualizarAlmacen(almacen, candidato):
    almacen_nuevo = []
    insertar = True  # se puede agregar al almacen 
    existe = False # existe o no en el almacen 
    for punto in almacen:
        if Dominante(punto, candidato):
            insertar = False
            almacen_nuevo.append(punto)
        elif not Dominante(candidato, punto):
            almacen_nuevo.append(punto)
            
    if insertar:
        # Verificar existencia en el nuevo almacén
        for punto_almacen in almacen_nuevo:
            if np.array_equal(candidato, punto_almacen):
                existe = True
                break

        if not existe:
            almacen_nuevo.append(candidato)
            
    return almacen_nuevo 

def Deb(pobp, pobh, app, aph, zp, zh, i, almacen):
    if app == 0 and aph == 0: 
        if Dominante(zh, zp) or not Dominante(zp, zh): 
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
            almacen = ActualizarAlmacen(almacen, z[i])  # Agrega al primer individuo con ap=0
    
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


def Dominancia(funcion1, funcion2, restricciones, limites, tamPoblacion, CR, iteraciones):
    pareto = EvolucionDiferencial(funcion1, funcion2, restricciones, limites, tamPoblacion, CR, iteraciones)
    pareto = np.array(pareto)
    
    # Guardar los puntos de Pareto en un CSV
    nombreArchivo = f"domin_p3.csv"
    with open(nombreArchivo, mode='w', newline='') as archivo_csv:
        escritor_csv = csv.writer(archivo_csv)
        escritor_csv.writerow(['f1(x)', 'f2(x)']) 
        escritor_csv.writerows(pareto)  

    plt.figure(figsize=(8, 6))
    plt.scatter(pareto[:, 0], pareto[:, 1], color='magenta', label='Frente de Pareto')
    plt.xlabel('f1(x)')
    plt.ylabel('f2(x)')
    plt.title('Frente de Pareto: Dominancia')
    plt.grid(True)
    plt.legend()
    plt.show()

"""----------------------------------DATOS----------------------------------"""
limites = limP01()
tamPoblacion = 40 
CR = 0.7 
iteraciones = 500

Dominancia(FuncionObjetivo1, FuncionObjetivo2, Restricciones, limites, tamPoblacion, CR, iteraciones)