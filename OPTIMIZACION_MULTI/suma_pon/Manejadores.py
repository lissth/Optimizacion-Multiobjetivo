import numpy as np

def Deb(pobp, pobh, app, aph, zp, zh, i ):

    if app == 0 and aph == 0:
        if zp <= zh:
            pobp[i] = pobp[i]
        
        else:
            pobp[i] = pobh
    elif app == 0 and aph != 0:
        pobp[i] = pobp[i]

    elif app != 0 and aph == 0:
        pobp[i] = pobh

    elif aph <= app:
        pobp[i] = pobh

    else:
        pobp[i] = pobp[i]

    return pobp