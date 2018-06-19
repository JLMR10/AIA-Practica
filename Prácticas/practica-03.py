# AIA: practica-03.py
# Procesos de Decisión de Markov
# Dpto. de C. de la Computación e I.A. (Univ. de Sevilla)
#==============================================================================

import random

# En esta práctica vamos a implementar algoritmos relacionados con Procesos de
# Decisión de Markov.

# Supondremos que un Procesos de Decisión de Markov (MDP, por sus siglas en
# inglés) va a ser un objeto de la siguiente clase (o mejor dicho, de una
# subclase de la siguiente clase).

class MDP(object):

    """La clase genérica MDP tiene como métodos la función de recompensa R,
    la función A que da la lista de acciones aplicables a un estado, y la
    función T que implementa el modelo de transición. Para cada estado y
    acción aplicable al estado, la función T devuelve una lista de pares
    (ei,pi) que describe los posibles estados ei que se pueden obtener al
    plical la acción al estado, junto con la probabilidad pi de que esto
    ocurra. El constructor de la clase recibe la lista de estados posibles y
    el factor de descuento.

    En esta clase genérica, las funciones R, A y T aparecen sin definir. Un
    MDP concreto va a ser un objeto de una subclase de esta clase MDP, en la
    que se definirán de manera concreta estas tres funciones"""

    def __init__(self,estados,descuento):

        self.estados=estados
        self.descuento=descuento

    def R(self,estado):
        pass

    def A(self,estado):
        pass

    def T(self,estado,accion):
        pass



#------------------------------------------------------------------------------
# Ejercicio 1
#------------------------------------------------------------------------------

# Consideramos el siguiente problema:

# A lo largo de su vida, una empresa pasa por situaciones muy distintas, que
# por simplificar resumiremos en que al inicio de cada campaña puede estar
# rica o pobre, y ser conocida o desconocida.  Para ello puede decidir en cada
# momento o bien invertir en publicidad, o bien optar por no hacer
# publicidad. Estas dos acciones no tienen siempre un resultado fijo, aunque
# podemos describirlo de manera probabilística:

# - Si la empresa es rica y conocida y no invierte en publicidad, seguirá
#   rica, pero existe un 50% de probabilidad de que se vuelva desconocida. Si
#   gasta en publicidad, con toda seguridad seguirá conocida pero pasará a ser
#   pobre.
# - Si la empresa es rica y desconocida y no gasta en publicidad, seguirá
#   desconocida, y existe un 50% de que se vuelva pobre. Si gasta en
#   publicidad, se volverá pobre, pero existe un 50% de probabilidades de que
#   se vuelva conocida.
# - Si la empresa es pobre y conocida y no invierte en publicidad, pasará a
#   ser pobre y desconocida con un 50% de probabilidad, y rica y conocida en
#   caso contrario. Si gasta en publicidad, con toda seguridad seguirá en la
#   misma situación.
# - Si la empresa es pobre y desconocida, y no invierte en publicidad,
#   seguirá en la misma situación con toda seguridad. Si gasta en publicidad,
#   seguirá pobre, pero con un 50% de posibilidades pasará aser conocida.
#

# Supondremos que la recompensa en una campaña en la que la empresa es rica
# es de 10, y de 0 en en las que sea pobre. El objetivo es conseguir la mayor
# recompesa acumulada a lo largo del tiempo, aunque penalizaremos las
# ganancias obtenidas en campañas muy lejanas en el tiempo, introduciendo un
# factor de descuento de 0.9.



# Se pide representar el problema como un proceso de decisión de Markov,
# definiendo una clase Rica_y_Conocida, subclase de la clase MDP genérica,
# cuyo constructor recibe como entrada únicamente el factor de descuento, y en
# la que se definen de manera concreta los métodos R, A y T, según lo
# descrito.

# ------------------------------------------------------------------

class Rica_y_Conocida(MDP):

    def __init__(self,descuento=0.9):
        self.Rdict = {"RC":10,"RD":10,"PC":0,"PD":0}
        self.Tdict = {("RC","No publicidad"):[("RC",0.5),("RD",0.5)],
                      ("RC","Gastar en publicidad"):[("PC",1)],
                      ("RD","No publicidad"):[("PD",0.5),("RD",0.5)],
                      ("RD","Gastar en publicidad"):[("PD",0.5),("PC",0.5)],
                      ("PC","No publicidad"):[("PD",0.5),("RC",0.5)],
                      ("PC","Gastar en publicidad"):[("PC",1)],
                      ("PD","No publicidad"):[("PD",1)],
                      ("PD","Gastar en publicidad"):[("PC",0.5),("PD",0.5)]}


        super().__init__(["RC","RD","PC","PD"],descuento)

    def R(self,estado):
        return self.Rdict[estado]

    def A(self,estado):
        return ["No publicidad", "Gastar en publicidad"]

    def T(self,estado,accion):
        return self.Tdict[(estado,accion)]



#------------------------------------------------------------------------------
# Ejercicio 2
#------------------------------------------------------------------------------

# En general, dado un MDP, representaremos una política para el mismo como un
# diccionario cuyas claves son los estados, y los valores las acciones. Una
# política representa una manera concreta de decidir en cada estado la
# acción (de entre las aplicables a ese estado) que ha de aplicarse.

# Dado un MDP, un estado de partida, y una política concreta, podemos generar
# (muestrear) una secuencia de estados por los que iríamos pasando si vamos
# aplicando las acciones que nos indica la política: dado un estado de la
# secuencia, aplicamos a ese estado la acción que indique la política, y
# obtenemos un estado siguiente de manera aleatoria, pero siguiendo la
# distribución de probabilidad que indica el modelo de transición dado por el
# método T.

# Se pide definir una función "genera_secuencia_estados(mdp,pi,e,n)" que
# devuelva una secuencia de estados de longitud n, obtenida siguiendo el
# método anterior. Aquí mdp es objeto de la clase MDP, pi es una política, e
# un estado de partida y n la longitud de la secuencia.

# Ejemplo:

# >>> mdp_ryc=Rica_y_Conocida()

# >>> pi_ryc_ahorra={"RC":"No publicidad","RD":"No publicidad","PC":"No publicidad","PD":"No publicidad"}
#                    
# >>> genera_secuencia_estados(mdp_ryc,pi_ryc_ahorra,"PC",20)
# ['PC', 'RC', 'RC', 'RC', 'RC', 'RC', 'RD', 'RD', 'RD', 'PD',
#  'PD', 'PD', 'PD', 'PD', 'PD', 'PD', 'PD', 'PD', 'PD', 'PD']

# >>> pi_ryc_2={"RC":"No publicidad","RD":"Gastar en publicidad","PC":"No publicidad","PD":"Gastar en publicidad"}
#               
# >>> genera_secuencia_estados(mdp_ryc,pi_ryc_2,"RD",16)
# ['RD', 'PC', 'PD', 'PC', 'RC', 'RC', 'RD', 'PD', 'PD', 'PC',
#  'RC', 'RC', 'RC', 'RC', 'RC', 'RC']

# -----------------------------------------------------------------------

def muestreo(distr):
    r = random.random()
    acum = 0
    for e,p in distr:
        acum += p
        if acum > r:
            return e

def genera_secuencia_estados(mdp,pi,estado,n):
    actual = estado
    seq=[actual]
    for _ in range(n-1):
        actual = muestreo(mdp.T(actual,pi[actual]))
        seq.append(actual)
    return seq


#------------------------------------------------------------------------------
# Ejercicio 3
#------------------------------------------------------------------------------

# Dado un MDP y una secuencia de estados, valoramos dicha secuencia como la
# suma de las recompensas de los estados de la secuencias (cada una con el
# correspondiente descuento). Se pide definir una función
# "valora_secuencia(seq,mdp)" que calcula esta valoración.

# Ejemplos:

# >>> valora_secuencia(['PC', 'RC', 'RC', 'RC', 'RC', 'RC','RD', 'RD', 'RD', 'PD', 'PD', 'PD','PD', 'PD', 'PD', 'PD', 'PD', 'PD','PD', 'PD'],mdp_ryc)
#                       
#                       
#                       
# 51.2579511

# >>> valora_secuencia(['RD', 'PC', 'PD', 'PC', 'RC', 'RC','RD', 'PD', 'PD', 'PC', 'RC', 'RC','RC', 'RC', 'RC', 'RC'],mdp_ryc)
#                       
#                       
# 44.11795212148159

# --------------------------------------------------------------------------
def valora_secuencia(secuencia,mdp):
    return sum(mdp.R(x)*mdp.descuento**n for n,x in enumerate(secuencia))



#------------------------------------------------------------------------------
# Ejercicio 4
#------------------------------------------------------------------------------

# Dada una política pi, la valoración de un estado e respecto de esa política,
# V^{pi}(e), se define como la media esperada de las valoraciones de las
# secuencias que se pueden generar teniendo dicho estado como estado de
# partida. Usando las funciones de los dos ejercicios anteriores, definir una
# función "estima_valor(e,pi,mdp,m,n)" que realiza una estimación aproximada
# del valor V^{pi}(e), para ello genera n secuencias de tamaño m, y calcula la
# media de sus valoraciones.

# Ejemplos:


# >>> estima_valor("PC",pi_ryc_ahorra,mdp_ryc,50,500)
# 14.745035878004751
# >>> estima_valor("PC",pi_ryc_2,mdp_ryc,50,500)
# 36.143411736436946
# >>> estima_valor("RC",pi_ryc_ahorra,mdp_ryc,60,700)
# 32.75657500158564
# >>> estima_valor("RC",pi_ryc_2,mdp_ryc,60,700)
# 50.6501047311598

# ---------------------------------------------------------------------
def estima_valor(estado,pi,mdp,n,m):
    return sum(valora_secuencia(genera_secuencia_estados(mdp,pi,estado,m),mdp) for _ in range(n))/n




#------------------------------------------------------------------------------
# Ejercicio 5
#------------------------------------------------------------------------------

# Usando la función anterior, estimar la valoración de cada estado del
# problema "Rica y conocida", con las dos siguientes políticas:

# * Aquella que sea cual sea el estado, siempre decide invertir en
#   publicidad.
# * Aquella que sea cual sea el estado, siempre decide ahorrar.

# ¿Cuál crees que es mejor? ¿Habrá alguna mejor que estas dos? ¿Cuál crees que
# sería la mejor política de todas?









# ------------------------------------------------------------------


#------------------------------------------------------------------------------
# Ejercicio 6
#------------------------------------------------------------------------------

# La función de valoración no se suele calcular mediante la técnica de
# muestreo vista en el ejercicio 4, sino como resultado de resolver un sistema
# de ecuaciones (ver diapositivas 71 y 72 del tema 3). Dicho sistema de
# ecuaciones se puede resolver de manera proximada de manera iterativa, tal y
# como se explica en el tema 3.

# Definir una función "valoración_respecto_política(pi,mdp, n)" que aplica
# dicho método iterativo (n iteraciones) para calcular la valoración
# V^{pi}. Dicha valoración debe devolverse como un diccionario que a cada
# estado e le asocia el valor "V^{pi}(e)" calculado.

# Aplicar la función para calcular la valoración asociada a las dos políticas
# que se dan en el ejercicio anterior, y comparara los valores obtenidos con
# los obtenidos mediante muestreo.


#----------------------------------------------------------------------------

def valoracion_respecto_politica(pi,mdp,k):
    R, T, gamma = mdp.R, mdp.T, mdp.descuento
    V = {e:0 for e in mdp.estados}
    for _ in range(k):
        V1 = V.copy()
        for s in mdp.estados:
            V[s] = R(s) + gamma * (sum([p * V1[s1] for (s1,p) in T(s,pi[s])]))
    return V







#------------------------------------------------------------------------------
# Ejercicio 7
#------------------------------------------------------------------------------


# En el tema 3 se ha visto que la valoración de un estado se define como la
# mejor valoración que pueda tener el estado, respecto a todas las políticas
# posibles. Y la política óptima es aquella que en cada estado realiza la
# acción con la mejor valoración esperada (entendiendo por valoración esperada
# la suma de las valoraciones de los estados que podrían resultar al aplicar
# dicha acción, ponderadas por la probabilidad de que ocurra eso). De esta
# manera, la valoración de un estado es la valoración que la política óptima
# asigna al estado.

# Para calcular tanto la valoración de los estados, como la política óptima,
# se han visto dos algoritmos: iteración de valores e iteración de
# políticas. En este ejercicio se pide implementar el algoritmo de iteración
# de políticas. En concreto, se pide definir una función
# "iteración_de_políticas(mdp,k)" que implementa el algoritmo de iteración de
# políticas, y devuelve dos diccionarios, uno con la valoración de los estados
# y otro con la política óptima.

# Comparar los resultados obtenidos con las políticas del ejercicio 5 y las
# valoraciones obtenidas.
# --------------------------------------------------------------------------

def iteracion_de_politicas(mdp,k):
    pi = {x:random.choice(mdp.A(x)) for x in mdp.estados}
    while True:
        Vpi = valoracion_respecto_politica(pi,mdp,k)
        actualizado = False
        for x in mdp.estados:
            accion = argmax(mdp.A(x),lambda a: valoracion_esperada(a,x,Vpi,mdp))
            if accion != pi[x]:
                actualizado = True
                pi[x] = accion
        if not actualizado:
            return pi,Vpi

def valoracion_esperada(acc,estado,V,mdp):
    return sum(p*V[x] for x,p in mdp.T(estado,acc))

def argmax(seq,f):
    max = float("-inf")
    amax = None
    for x in seq:
        fx = f(x)
        if fx>max:
            max=fx
            amax=x
    return amax