# Ampliación de Inteligencia Artificial
# Modelos ocultos de Markov
# Dpto. de C. de la Computación e I.A. (Univ. de Sevilla)
# ===================================================================

# ********************************************************************
# Nombre: José Luis
# Apellidos: Mármol Romero
# ********************************************************************


# --------------------------------------------------------------------------- 
# Los siguientes apartados se proponen como ejercicio de programación que
# contará para la evaluación de la asignatura. Este entregable supone 1 PUNTO
# de la nota total de la asignatura.  Se deberá entregar a través de la página
# de la asignatura, en el formulario a tal efecto que estará disponible junto
# a la ficha de alumno.


# IMPORTANTE: No cambiar el nombre ni a este archivo ni a las funciones que se
# piden. Si se entregan con un nombre distinto, el entregable no será
# evaluado.
# --------------------------------------------------------------------



## ###################################################################
## HONESTIDAD ACADÉMICA Y COPIAS: la realización de los ejercicios es
## un trabajo personal, por lo que deben completarse por cada
## estudiante de manera individual.  La discusión y el intercambio de
## información de carácter general con los compañeros se permite (e
## incluso se recomienda), pero NO AL NIVEL DE CÓDIGO. Igualmente el
## remitir código de terceros, obtenido a través de la red o cualquier
## otro medio, se considerará plagio.

## Cualquier plagio o compartición de código que se detecte
## significará automáticamente la calificación de CERO EN LA
## ASIGNATURA para TODOS los alumnos involucrados. Por tanto a estos
## alumnos NO se les conservará, para futuras convocatorias, ninguna
## nota que hubiesen obtenido hasta el momento. Independientemente de
## OTRAS ACCIONES DE CARÁCTER DISCIPLINARIO que se pudieran tomar.
## ###################################################################

import random
# -------------------------------------------
# Representación de modelos ocultos de Markov
# -------------------------------------------

# Lo que sigue es la definición de una clase que con la que representar
# modelos ocultos de Markov:

class HMM(object):
    """Clase para definir un modelo oculto de Markov"""

    def __init__(self,estados,mat_ini,mat_trans,observables,mat_obs):
        
        self.eOcultos=estados
        self.eObservables=observables
        self.a={(si,sj):ptrans
                for (si,l) in zip(estados,mat_trans)
                for (sj,ptrans) in zip(estados,l)}
        self.b={(si,vj):pobs
                for (si,l) in zip(estados,mat_obs)
                for (vj,pobs) in zip(observables,l)}
        self.pi=dict(zip(estados,mat_ini))

# Descripción:
# ------------

# En esta clase hmm, el constructor recibe los siguientes argumentos:

# - estados: una lista con los estados ocultos.
# - mat_ini: una lista con las probabilidades iniciales, en el mismo orden en
#   el que se dan los estados en la lista anterior.
# - mat_trans: una lista de listas, con las probabilidades de transición, de
#   manera que mat_trans[i][j] es la probabilidad de pasar del estado i-ésimo
#   al estado j-ésimo.
# - observables: una lista con las posibles observaciones.
# - mat_obs: una lista de listas con las probabilidades de observación, de
#   manera que mat_obs[i][j] es la probabilidad de observar el observable
#   j-ésimo en el estado i-ésimo.     

# Lo que sigue son los objetos de esta clase HMM que respectivamente
# representan a los dos modelos ocultos de Markov que se usan como ejemplo en
# las diapositivas del tema 3:

ej1_hmm=HMM(["c","f"],
            [0.8,0.2],
            [[0.7,0.3],[0.4,0.6]],
            [1,2,3],   
            [[0.2,0.4,0.4],[0.5,0.4,0.1]])
            

ej2_hmm=HMM(["l","no l"],
            [0.5,0.5],
            [[0.7, 0.3], [0.3,0.7]],
            ["u","no u"],   
            [[0.9,0.1],[0.2,0.8]])


# Como se observa en la definición de la clase, una vez llamado al constructor
# los objetos de la clase HMM tienen los siguientes atributos:

# Atributos:
# * eOcultos: Una lista con los estados que definen la variable oculta
#             del modelo.
#             [s_1, ..., s_n]
# * eObservables: Una lista con los estados que definen la variable
#                 observable del modelo.
#                 [v_1, ..., v_m]
# * pi: Un diccionario, cuyas claves son los estados, y cuyos valores
#       son las probabilidades iniciales:
#                pi[s_i] = P(X_1 = s_i)
# * a: Un diccionario cuyas claves son parejas (tuplas) de estados y
#      cuyos valores son las correspondientes probabilidades de la
#      matriz de transición:
#      a[(s_i, s_j)] = P(X_t = s_j | X_{t-1} = s_i)
# * b: Un diccionario cuyas claves son parejas (tuplas) de estado y
#      observable, y cuyos valores son las correspondientes
#      probabilidades de la matriz de observación:
#      b[(s_i,v_j)] = P(E_t = v_j | X_{t-1} = s_i)


# Por ejemplo:

# >>> ej1_hmm.eOcultos
# ['c', 'f']
# >>> ej1_hmm.eObservables
# [1, 2, 3]
# >>> ej1_hmm.pi
# {'f': 0.2, 'c': 0.8}
# >>> ej1_hmm.a
# {('f', 'c'): 0.4, ('c', 'c'): 0.7, ('c', 'f'): 0.3, ('f', 'f'): 0.6}
# >>> ej1_hmm.b
# {('c', 2): 0.4, ('f', 1): 0.5, ('c', 3): 0.4, ('c', 1): 0.2,
#  ('f', 3): 0.1, ('f', 2): 0.4}





# *******
# PARTE I
# *******


#------------------------------------------------------------------------------
# EJERCICIO 1
#------------------------------------------------------------------------------


# Definir la función avance_norm(hmm,seq) que recibiendo un modelo oculto de
# Markov (un objeto de la clase HMM anterior) y una secuencia de observaciones
# [o_1,...,o_t], implementa el algoritmo de avance visto en clase pero en su
# versión modificada, en el que en cada iteración se normalizan los "alfa"
# calculados. Esta función debe devolver la lista con las probabilidades 
#      P(X_t = s_i | E_1 = o_1, ...,E_t = o_t), 1 <= i <= n.


# Ejemplos:

# >>> avance_norm(ej1_hmm,[3,1,3,2])
# {'f': 0.35290892476393537, 'c': 0.6470910752360646}
# >>> avance_norm(ej2_hmm,["u","u","no u"])
# {'l': 0.19066793972352525, 'no l': 0.8093320602764748}

def avance_norm(hmm,seq):
    #Paso 1
    alfa = {x : hmm.b[(x,seq[0])]*y for x,y in hmm.pi.items()}
    #Pasos recursivos
    i=1
    while i<len(seq):
        alfa = {x : hmm.b[(x,seq[i])]*sum(hmm.a[(y,x)]*alfa[y] for y in hmm.eOcultos) for x in hmm.eOcultos}
        i+=1
        #Normalizo
        suma = sum(y for x,y in alfa.items())
        alfa = {x: y/suma for x,y in alfa.items()}
    return alfa


#---------------------------------------------------------------------
# EJERCICIO 2
#---------------------------------------------------------------------

# Usar la función anterior para hacer el ejercicio 12 de la relación de
# problemas

####### Apartado a

# ej12_hmm.eOcultos
# ["D","no D"]
# ej12_hmm.eObservables
# [("DC","OR"),("DC","no OR"),("no DC","OR"),("no DC","no OR")]
# ej12_hmm.pi
# {"D": 0.6, "no D": 0.4}
# ej12_hmm.a
# {("D","D"): 0.8 ,("D","no D"): 0.2 ,("no D","no D"): 0.8 ,("no D","D"): 0.2}
# ej12_hmm.b
# {("D",("DC","OR")): 0.02, ("D",("DC","no OR")): 0.08, 
#   ("D",("no DC","OR")): 0.08, ("D",("no DC","no OR")): 0.72,
#  ("no D",("DC","OR")): 0.21, ("no D",("DC","no OR")): 0.49, 
#   ("no D",("no DC","OR")): 0.09, ("no D",("no DC","no OR")): 0.21}

# Modelado del problema 12
ej12_hmm = HMM(["D","no D"],
               [0.6,0.4],
               [[0.8,0.2],[0.2,0.8]],
               [("DC","OR"),("DC","no OR"),("no DC","OR"),("no DC","no OR")],
               [[0.02,0.08,0.08,0.72],[0.21, 0.49,0.09,0.21]])

####### Apartado b

# Obsevaciones: [("no DC","no OR"),("no DC","OR"),("DC","OR")]
# Es necesario programar forward-backward

def for_back_norm(hmm,seq,t):
    alfa = avance_norm(hmm,[seq[i] for i in range(t)])
    beta = {x : 1 for x in hmm.eOcultos}
    n=len(seq)-1
    while n>=t:
        beta = {x : sum(hmm.b[(y,seq[n])]*hmm.a[(x,y)]*beta[y] for y in hmm.eOcultos) for x in hmm.eOcultos}
        suma = sum(y for x,y in beta.items())
        alfa_beta = {x: y/suma for x,y in beta.items()}
        n-=1
    alfa_beta = {x : alfa[x]*beta[x] for x in hmm.eOcultos}
    suma = sum(y for x,y in alfa_beta.items())
    alfa_beta = {x: y/suma for x,y in alfa_beta.items()}
    return alfa_beta


# Probabilidad de que en el día 1 hubiese dormido lo suficiente sabiendo [("no DC","no OR"),("no DC","OR"),("DC","OR")]
# for_back_norm(ej12_hmm,[("no DC","no OR"),("no DC","OR"),("DC","OR")],1)
# {'D': 0.7245293728124114, 'no D': 0.2754706271875887}
# Solucion = 0.7245293728124114
#
# Probabilidad de que en el día 2 hubiese dormido lo suficiente sabiendo [("no DC","no OR"),("no DC","OR"),("DC","OR")]
# for_back_norm(ej12_hmm,[("no DC","no OR"),("no DC","OR"),("DC","OR")],2)
# {'D': 0.4142465235077097, 'no D': 0.5857534764922903}
# Solucion = 0.4142465235077097
#
# Probabilidad de que en el día 3 hubiese dormido lo suficiente sabiendo [("no DC","no OR"),("no DC","OR"),("DC","OR")]
# avance_norm(ej12_hmm,[("no DC","no OR"),("no DC","OR"),("DC","OR")])
# {'D': 0.1278970769085233, 'no D': 0.8721029230914766}
# Solucion = 0.1278970769085233
 
####### Apartado c

def convergencia_DC_OR():
    secuencia = [("DC","OR")]
    actual = avance_norm(ej12_hmm,secuencia)["D"]
    aux = 0
    iteraciones = 0
    while actual != aux:
        aux = actual 
        secuencia.append(("DC","OR"))
        actual = avance_norm(ej12_hmm,secuencia)["D"]
        iteraciones +=1
    return iteraciones,actual
    
# convergencia_DC_OR()
# Devuelve una tupla que representa el número de iteraciones en el que converge y la probabilidad con la que lo hace
# El resultado obtenido es (17, 0.02546192044223218)

# ********
# PARTE II
# ********

# PROBLEMA DEL CASINO TRAMPOSO
# ----------------------------


# En esta parte se pide aplicar las implementaciones anteriores a
# un problema simple, que llamaremos el "problema del casino tramposo".


#---------------------------------------------------------------------
# EJERCICIO 3
#---------------------------------------------------------------------

# Supongamos que tenemos un casino en el que uno de los juegos de azar
# consiste en apostar qué número saldrá al tirar un dado. Cada tirada,
# un jugador puede apostar 1 euro a un número entre 1 y 6. Si sale el
# número elegido, recibe 6 euros; si no, no recibe nada (es decir, cada vez
# que acierta gana 5v euros y cada vez que falla pierde un euro)

# Se pide implementar una función resultado_apuesta(l1,l2), que recibiendo dos
# secuencias l1 y l2 de la misma longitud, cuyos elementos son números entre 1
# y 6, calcula la ganancia (o pérdida) que tiene un jugador que haya apostado
# a los números que se indican en l1, y resultando que en realidad los números
# que salen son los que se indican en l2. Por ejemplo, si l1=[1,6,5,6,6,2,3,6]
# y l2=[1,4,5,6,6,2,2,6], entonces la ganancia es de 28 euros, ya que se
# acierta 6 veces y se falla 2 veces.

# Nota: las pérdidas se representan como ganancias negativas. 

# Ejemplos:

# >>> resultado_apuesta([1,6,5,6,6,2,3,6],[1,4,5,6,6,2,2,6])
# 28
# >>> resultado_apuesta([1,6,5,6,6,2,3,6],[2,1,3,5,6,2,4,1])
# 4
# >>> resultado_apuesta([1,6,5,6,6,2,3,6],[2,1,3,5,6,1,4,1])
# -2

def resultado_apuesta(l1,l2):
    total = 0
    for x,y in zip(l1,l2):
        if x==y:
            total+=5
        else:
            total-=1
    return total


#---------------------------------------------------------------------
# EJERCICIO 4
#---------------------------------------------------------------------

# Supongamos ahora que nos han dado un chivatazo, y sabemos que el
# casino hace trampas: en algunas tiradas, usa un dado trucado. En el
# dado trucado, el seis sale con probabilidad 0.5, y el resto de
# números salen con igual probabilidad. Por supuesto, no sabemos si en
# una jugada se está usando un dado trucado o no, y además esto se
# hace de manera aleatoria. Sin embargo, nuestro confidente nos ha
# dado cierta información probabilística: nos dice que en la primera
# tirada hay 1/3 de probabilidad de que el dado sea trucado; además,
# nos dice que usando un dado trucado en una jugada, la probabilidad
# de que cambien al dado normal en la siguiente es de 0.2; y que
# usando un dado normal en una jugada, la probabilidad que en la
# siguiente se siga con un dado normal es de 0.7.


# Se pide representar esta infomación usando un modelo oculto de
# Markov, generando el correspondiente objeto de la clase hmm, y asignádolo a
# una variable de nombre casino_hmm

casino_hmm = HMM(["normal","trucado"],
                 [2/3,1/3],
                 [[0.7,0.3],[0.2,0.8]],
                 [1,2,3,4,5,6],
                 [[1/6,1/6,1/6,1/6,1/6,1/6],[0.1,0.1,0.1,0.1,0.1,0.5]])




#---------------------------------------------------------------------
# EJERCICIO 5
#---------------------------------------------------------------------

# Definir una función muestreo_secuencias(hmm,n) tal que recibiendo un modelo
# oculto de Markov y un número n, genera aleatoriamente un par de secuencias
# de longitud n: un secuencia de estados ocultos, y un correspondiente
# secuencia de observaciones.

# Estas secuencias han de generarse aleatoriamente, pero SIGUIENDO las
# probabilidades del modelo oculto de Markov. Es decir:

# - Los estados ocultos iniciales se generan usando las probabilidades
#   iniciales.
# - Para pasar de un estado oculto al siguiente, se usan las probabilidades de
#   transición. 
# - Para dar la observación correspondiente a cada oculto, usar las
#   probabilidades de observación.  


# Ejemplo:

# Ejemplo:

# >>> muestreo_secuencias(casino_hmm,20)
#[['trucado','normal','trucado','trucado','normal','normal','trucado',
#  'normal','trucado','normal','normal','normal','normal','trucado',
#  'normal','trucado','trucado','trucado','trucado','trucado'],
# [6, 6, 1, 6, 2, 1, 1, 1, 6, 2, 5, 4, 3, 3, 1, 3, 2, 4, 6, 6]]




# >>> muestreo_secuencias(casino_hmm,20)
#[['normal','trucado','trucado','normal','trucado','trucado','trucado',
#  'trucado','trucado','trucado','trucado','normal','normal','normal',
#  'trucado','trucado','normal','trucado','normal','normal'],
# [1, 4, 6, 5, 6, 1, 6, 4, 6, 6, 5, 3, 2, 6, 3, 6, 4, 5, 2, 6]]

def muestreo(distr):
    r = random.random()
    acum = 0
    for e,p in distr.items():
        acum += p
        if acum > r:
            return e

def muestreo_secuencias(hmm,n):
    actualOculto = muestreo(hmm.pi)
    seqOcultos = [actualOculto]
    actualObservable = muestreo({x: hmm.b[(actualOculto,x)] for x in hmm.eObservables})
    seqObservables = [actualObservable]
    for _ in range(n-1):
        actualOculto = muestreo({x:hmm.a[(actualOculto,x)] for x in hmm.eOcultos})
        seqOcultos.append(actualOculto)
        actualObservable = muestreo({x: hmm.b[(actualOculto,x)] for x in hmm.eObservables})
        seqObservables.append(actualObservable)
    return seqOcultos,seqObservables





#---------------------------------------------------------------------
# EJERCICIO 6
#---------------------------------------------------------------------

# Ahora nosotros podríamos ir al casino y jugar de dos maneras
# distintas, en cada tirada:

# - Estrategia 1: independientemente de lo que haya salido en las
#   tiradas anteriores, usar nosotros un dado no trucado, y en cada jugada
#   tirar nuestro propio dado, y apostar al número que nos haya salido a
#   nosotros. 

# - Estrategia 2: usar nuestros conocimientos sobre modelos ocultos de
#   Markov, y hacer lo siguiente:

#   * En la primera tirada, suponer que el dado es no trucado y
#     apostar como en la estrategia 1.

#   * Para las siguientes tiradas:
#       + Calcular cuál es el tipo de dado más probable usado en la tirada
#       anterior, dada la secuencia completa de resultados obtenidos hasta esa
#       tirada anterior.
#       + Puesto que el modelo nos dice que lo más probable es no
#       cambiar de dado, suponer que en la siguiente tirada el dado
#       usado va a ser de ese mismo tipo que acabamos de calcular
#       + Apostar en consecuencia: si como resultado de nuestro cálculo es más
#       probable que el dado esté trucado, apostar al 6; si lo más probable es
#       que sea un dado normal, apostar siguiendo el método de la estrategia 1.
#       


# Se pide definir funciones estrategia_1(l) y estrategia_2(l), que
# recibiendo una secuencia l de resultados de tiradas, devuelva una
# lista de números a los que iríamos apostando, siguendo
# respectivamente las estrategias 1 y 2.

# Ejemplo 
# --------
# (nótese que no tiene porqué salir lo mismo, ya que hay aleatoriedad) :

# Generamos una secuencia de resultados de dados, con la simulación que se 
# ha implementado en el apartado anterior:
    
# >>> l=muestreo_secuencias(casino_hmm,20)[1] 
# >>> l
# [6, 6, 6, 6, 6, 6, 5, 2, 5, 5, 2, 6, 4, 4, 1, 4, 3, 3, 1, 6] 

# Con la estrategia 1, se genera la siguiente secuencia de 20 apuestas
# >>> l_ap1=estrategia_1(l)
# >>> l_ap1    
# [3, 2, 2, 1, 5, 4, 3, 2, 1, 6, 6, 2, 2, 3, 2, 3, 1, 5, 5, 2]

# En este caso, ¿cuánto hubieramos ganado con esta estrategia?
# >>> resultado_apuesta(l_ap1,l)    
# -14    
    
# Sin embargo, con la estrategia 2 se genera la siguiente secuencia 
# de apuestas:    
# >>> l_ap2=estrategia_2(l)
# >>> l_ap2    
# [1, 6, 6, 6, 6, 6, 6, 6, 2, 1, 1, 6, 6, 6, 1, 2, 1, 3, 1, 2]

# ¿Cuánto hubieramos ganado con esta estrategia?
# >>> resultado_apuesta(l_ap2,l)     
# 34
    
def estrategia_1(l):
    return [random.randrange(1,7) for _ in l]

def estrategia_2(l):
    actual = random.randrange(1,7)
    seq = [actual]
    for _ in range(len(l)-1):
        maxim = float("-inf")
        dado = []
        for x,y in avance_norm(casino_hmm,seq).items():
            if y > maxim :
                maxim = y
                dado = [x]
            elif y == maxim:
                dado.append(x)
        dado = random.choice(dado)
        if dado == "trucado":
            seq.append(6)
        else:
            seq.append(random.randrange(1,7))
    return seq 



            
#---------------------------------------------------------------------
# EJERCICIO 7
#---------------------------------------------------------------------

# Definir una función compara_estrategias(n), que implementa lo que 
#se ha hecho paso a paso en el ejemplo anterior    

# * En primer lugar, genera una secuencia de n tiradas, usando la
#   función de simulación que se pide en el ejercicio 5, y tomando la
#    secuencia de resultados que han salido (y desechando la secuencia de 
#    estados ocultos).
# * A continuación, imprime por pantalla una comparativa sobre lo que,
#   para esa secuencia generada, se hubiera ganado o perdido si se usa
#   la estrategia 2 frente a la estrategia 1.
# Una vez definida, ejecutarla varias veces para n=100, y comentar los
# resultados que se obtienen.
     
def compara_estrategias(n):
    l=muestreo_secuencias(casino_hmm,n)[1]
    l_ap2 = estrategia_2(l)
    res2 = resultado_apuesta(l_ap2,l)
    print("Estrategia 2 =",res2)
    l_ap1 = estrategia_1(l)
    res1 = resultado_apuesta(l_ap1,l)
    print("Estrategia 1 =",res1)
    print("Ganancias de la Estrategia2 - Ganancias de la Estrategia1 =",res2-res1)

# compara_estrategias(100)
# Estrategia 2 = 110
# Estrategia 1 = 20
# Ganancias de la Estrategia2 - Ganancias de la Estrategia1 = 90

# compara_estrategias(100)
# Estrategia 2 = 116
# Estrategia 1 = -22
# Ganancias de la Estrategia2 - Ganancias de la Estrategia1 = 138

# compara_estrategias(100)
# Estrategia 2 = 140
# Estrategia 1 = -34
# Ganancias de la Estrategia2 - Ganancias de la Estrategia1 = 174

# compara_estrategias(100)
# Estrategia 2 = 68
# Estrategia 1 = 8
# Ganancias de la Estrategia2 - Ganancias de la Estrategia1 = 60

# Observando los resultados se puede comprobar que la Estrategia2 es mucho más
# efectiva que la Estrategia1 debido al uso de los modelos ocultos de Markov