import pandas as pd
import numpy as np
import glob
from sklearn.preprocessing import MinMaxScaler

# Función para normalizar los puntos entre 0 y 1
def normalizar(puntos):
    scaler = MinMaxScaler()
    return scaler.fit_transform(puntos)

# Aproximación al hipervolumen (básico)
def Hipervolumen(puntos):
    if puntos.shape[0] < 2:
        return 0
    ref = np.max(puntos, axis=0) + 0.1  # punto de referencia un poco peor
    volumen = 0
    ordenados = puntos[np.argsort(puntos[:, 0])]
    for i in range(len(ordenados) - 1):
        ancho = ref[0] - ordenados[i][0]
        alto = ref[1] - ordenados[i+1][1]
        volumen += ancho * alto
    return volumen

# Espaciamiento entre puntos consecutivos
def Espaciamiento(puntos):
    if len(puntos) < 2:
        return np.nan
    ordenados = puntos[np.argsort(puntos[:, 0])]
    distancia = np.linalg.norm(ordenados[1:] - ordenados[:-1], axis=1)
    distancia_prom = np.mean(distancia)
    return np.sum(np.abs(distancia - distancia_prom)) / len(distancia)

# Distancia promedio entre puntos vecinos
def DistanciaPromedio(puntos):
    if len(puntos) < 2:
        return np.nan
    ordenados = puntos[np.argsort(puntos[:, 0])]
    distancia = np.linalg.norm(ordenados[1:] - ordenados[:-1], axis=1)
    return np.mean(distancia)

# Rango de valores por objetivo
def Rango(puntos):
    if puntos.shape[0] == 0:
        return np.nan, np.nan
    return np.ptp(puntos[:, 0]), np.ptp(puntos[:, 1])

# Función principal
def AnalizarCSV():
    archivos = glob.glob("*.csv")
    if not archivos:
        print("No hay archivos CSV")
        return

    print("Archivos encontrados:")
    for a in archivos:
        print("-", a)

    resultados = {}

    for archivo in archivos:
        try:
            df = pd.read_csv(archivo)
            columnas = df.columns[:2]
            puntos = []

            for _, fila in df[columnas].iterrows():
                punto = []
                for valor in fila:
                    try:
                        punto.append(float(valor))
                    except:
                        # formato diferente
                        if isinstance(valor, str) and valor.startswith("[") and valor.endswith("]"):
                            try:
                                punto.append(float(valor[1:-1]))
                            except:
                                punto = None
                                break
                        else:
                            punto = None
                            break
                if punto and len(punto) == 2:
                    puntos.append(punto)

            puntos = np.array(puntos)
            if len(puntos) < 2:
                print(f"{archivo} tiene muy pocos puntos.")
                continue

            norm = normalizar(puntos)
            hv = Hipervolumen(norm)
            esp = Espaciamiento(norm)
            dist_prom = DistanciaPromedio(norm)
            r1, r2 = Rango(puntos)

            long_aprox = np.linalg.norm(norm[-1] - norm[0])
            densidad = len(puntos) / long_aprox if long_aprox > 0 else np.nan

            resultados[archivo] = {
                "n_puntos": len(puntos),
                "densidad": densidad,
                "dist_prom": dist_prom,
                "hipervolumen": hv,
                "espaciamiento": esp,
                "rango_obj1": r1,
                "rango_obj2": r2
            }

        except Exception as e:
            print(f"Error en {archivo}: {e}")

    print("\nResumen:")
    df_res = pd.DataFrame(resultados).T
    df_res = df_res.sort_values(by="hipervolumen", ascending=False)
    print(df_res)

    print("\nMejores frentes por métrica:")
    if df_res.empty:
        print("Nada que comparar.")
        return

    print("Hipervolumen:", df_res['hipervolumen'].idxmax())
    print("Espaciamiento:", df_res['espaciamiento'].idxmin())
    print("Densidad:", df_res['densidad'].idxmax())
    print("Distancia Promedio:", df_res['dist_prom'].idxmin())
    print("Rango Obj1:", df_res['rango_obj1'].idxmax())
    print("Rango Obj2:", df_res['rango_obj2'].idxmax())

# Ejecutar todo
AnalizarCSV()
