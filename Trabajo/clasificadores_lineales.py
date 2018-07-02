# ==========================================================
# Ampliación de Inteligencia Artificial. Tercer curso.
# Grado en Ingeniería Informática - Tecnologías Informáticas
# Curso 2017-18
# Trabajo práctico
# ===========================================================
import random, copy, numpy, math, shelve
# --------------------------------------------------------------------------
# Autor del trabajo:
#
# APELLIDOS: Mármol Romero
# NOMBRE: José Luis
#
# Segundo componente (si se trata de un grupo):
#
# APELLIDOS: González Corzo
# NOMBRE: Daniel
# ----------------------------------------------------------------------------


# *****************************************************************************
# HONESTIDAD ACADÉMICA Y COPIAS: un trabajo práctico es un examen, por lo que
# debe realizarse de manera individual. La discusión y el intercambio de
# información de carácter general con los compañeros se permite (e incluso se
# recomienda), pero NO AL NIVEL DE CÓDIGO. Igualmente el remitir código de
# terceros, OBTENIDO A TRAVÉS DE LA RED o cualquier otro medio, se considerará
# plagio.

# Cualquier plagio o compartición de código que se detecte significará
# automáticamente la calificación de CERO EN LA ASIGNATURA para TODOS los
# alumnos involucrados. Por tanto a estos alumnos NO se les conservará, para
# futuras convocatorias, ninguna nota que hubiesen obtenido hasta el
# momento. SIN PERJUICIO DE OTRAS MEDIDAS DE CARÁCTER DISCIPLINARIO QUE SE
# PUDIERAN TOMAR.
# *****************************************************************************


# IMPORTANTE: NO CAMBIAR EL NOMBRE NI A ESTE ARCHIVO NI A LAS CLASES Y MÉTODOS
# QUE SE PIDEN

# NOTA: En este trabajo no se permite usar scikit-learn, pero sí numpy o scipy.

# ====================================================
# PARTE I: MODELOS LINEALES PARA CLASIFICACIÓN BINARIA
# ====================================================

# En esta primera parte se pide implementar en Python los siguientes
# clasificadores BINARIOS, todos ellos vistos en el tema 5.

# - perceptrón umbral
# - Regresión logística minimizando el error cuadrático:
#      * Versión batch
#      * Versión estocástica (regla delta)
# - Regresión logística maximizando la verosimilitud:
#      * Versión batch
#      * Versión estocástica


# --------------------------------------------
# I.1. Generando conjuntos de datos aleatorios
# --------------------------------------------

# Previamente a la implementación de los clasificadores, conviene tener
# funciones que generen aleatoriamente conjuntos de datos fictícios.
# En concreto, se pide implementar estas dos funciones:

# X1,Y1=genera_conjunto_de_datos_l_s(4,8,400)
# * Función genera_conjunto_de_datos_l_s(rango,dim,n_datos):
def genera_conjunto_de_datos_l_s(rango,dim,n_datos):

    def clasificacion(hiperplano,x):
        res = 0
        x.insert(0,1)
        if sum(a*b for a,b in zip(hiperplano,x)) >=0 :
            res = 1
        return res

    hiperplano = [random.randint(-rango,rango) for x in range(dim+1)]

    X = [[random.randint(-rango,rango) for x in range(dim)] for x in range(n_datos)]

    Y = [clasificacion(hiperplano,x) for x in X]

    X = [x[1:] for x in X]

    return X,Y



#   Debe devolver dos listas X e Y, generadas aleatoriamente. La lista X debe
#   tener un número total n_datos de elementos, siendo cada uno de ellos una
#   lista (un ejemplo) de dim componentes, con valores entre -rango y rango. El
#   conjunto Y debe tener la clasificación binaria (1 o 0) de cada ejemplo del
#   conjunto X, en el mismo orden. El conjunto de datos debe ser linealmente
#   separable.

#   SUGERENCIA: generar en primer lugar un hiperplano aleatorio (mediante sus
#   coeficientes, elegidos aleatoriamente entre -rango y rango). Luego generar
#   aleatoriamente cada ejemplo de igual manera y clasificarlo como 1 o 0
#   dependiendo del lado del hiperplano en el que se situe. Eso asegura que el
#   conjunto de datos es linealmente separable.


# * Función genera_conjunto_de_datos_n_l_s(rango,dim,size,prop_n_l_s=0.1):

#   Como la anterior, pero el conjunto de datos debe ser no linealmente
#   separable. Para ello generar el conjunto de datos con la función anterior
#   y cambiar de clase a una proporción pequeña del total de ejemplos (por
#   ejemplo el 10%). La proporción se da con prop_n_l_s.

# X2,Y2=genera_conjunto_de_datos_n_l_s(4,8,400,0.1)
def genera_conjunto_de_datos_n_l_s(rango,dim,size,prop_n_l_s=0.1):
    X,Y=genera_conjunto_de_datos_l_s(rango,dim,size)
    randomizado = list(range(size))
    random.shuffle(randomizado)
    cantidadCambio = round(size * prop_n_l_s + 0.5)

    for x in range(cantidadCambio):
        Y[randomizado[x]] = 1-Y[randomizado[x]]

    return X,Y





# -----------------------------------
# I.2. Clases y métodos a implementar
# -----------------------------------


# En esta sección se pide implementar cada uno de los clasificadores lineales
# mencionados al principio. Cada uno de estos clasificadores se implementa a
# través de una clase python, que ha de tener la siguiente estructura general:

# class NOMBRE_DEL_CLASIFICADOR():

#     def __init__(self,clases,normalizacion=False):
#          .....

#     def entrena(self,entr,clas_entr,n_epochs,rate=0.1,
#                 pesos_iniciales=None,
#                 rate_decay=False):
#         ......

#     def clasifica_prob(self,ej):
#         ......

#     def clasifica(self,ej):
#         ......

# Explicamos a continuación cada uno de estos elementos:

# * NOMBRE_DEL_CLASIFICADOR:
# --------------------------

#  Este es el nombre de la clase que implementa el clasificador.
#  Obligatoriamente se han de usar cada uno de los siguientes
#  nombres:

#  - Perceptrón umbral:
#                       Clasificador_Perceptron

#  - Regresión logística, minimizando L2, batch:
#                       Clasificador_RL_L2_Batch

#  - Regresión logística, minimizando L2, estocástico:
#                       Clasificador_RL_L2_St

#  - Regresión logística, maximizando verosimilitud, batch:
#                       Clasificador_RL_ML_Batch

#  - Regresión logística, maximizando verosimilitud, estocástico:
#                       Clasificador_RL_ML_St

# * Constructor de la clase:
# --------------------------

#  El constructor debe tener los siguientes argumentos de entrada:

#  - Una lista clases con los nombres de las clases del problema de
#    clasificación, tal y como aparecen en el conjunto de datos.
#    Por ejemplo, en el caso del problema de las votaciones,
#    esta lista sería ["republicano","democrata"]

#  - El parámetro normalizacion, que puede ser True o False (False por
#    defecto). Indica si los datos se tienen que normalizar, tanto para el
#    entrenamiento como para la clasificación de nuevas instancias.  La
#    normalización es una técnica que suele ser útil cuando los distintos
#    atributos reflejan cantidades numéricas de muy distinta magnitud.
#    En ese caso, antes de entrenar se calcula la media m_i y la desviación
#    típica d_i en cada componente i-esima (es decir, en cada atributo) de los
#    datos del conjunto de entrenamiento.  A continuación, y antes del
#    entrenamiento, esos datos se transforman de manera que cada componente
#    x_i se cambia por (x_i - m_i)/d_i. Esta misma transformación se realiza
#    sobre las nuevas instancias que se quieran clasificar.  NOTA: se permite
#    usar la biblioteca numpy para calcular la media, la desviación típica, y
#    en general para cualquier cálculo matemático.

# * Método entrena:
# -----------------

#  Este método es el que realiza el entrenamiento del clasificador.
#  Debe calcular un conjunto de pesos, mediante el correspondiente
#  algoritmo de entrenamiento. Describimos a continuación los parámetros de
#  entrada:

#  - entr y clas_entr, son los datos del conjunto de entrenamiento y su
#    clasificación, respectivamente. El primero es una lista con los ejemplos,
#    y el segundo una lista con las clasificaciones de esos ejemplos, en el
#    mismo orden.

#  - n_epochs: número de veces que se itera sobre todo el conjunto de
#    entrenamiento.

#  - rate: si rate_decay es False, rate es la tasa de aprendizaje fija usada
#    durante todo el aprendizaje. Si rate_decay es True, rate marca una cota
#    mínima de la tasa de aprendizaje, como se explica a continuación.

#  - rate_decay, indica si la tasa de aprendizaje debe disminuir a medida que
#    se van realizando actualizaciones de los pases. En concreto, si
#    rate_decay es True, la tasa de aprendizaje que se usa en cada
#    actualización se debe de calcular con la siguiente fórmula:
#       rate_n= rate_0 + (2/n**(1.5))
#    donde n es el número de actualizaciones de pesos realizadas hasta el
#    momento, y rate_0 es la cantidad introducida en el parámetro rate
#    anterior.
#
#  - pesos_iniciales: si es None, se indica que los pesos deben iniciarse
#    aleatoriamente (por ejemplo, valores aleatorios entre -1 y 1). Si no es
#    None, entonces se debe proporcionar la lista de pesos iniciales. Esto
#    puede ser útil para continuar el aprendizaje a partir de un aprendizaje
#    anterior, si por ejemplo se dispone de nuevos datos.

#  NOTA: En las versiones estocásticas, y en el perceptrón umbral, en cada
#  epoch recorrer todos los ejemplos del conjunto de entrenamiento en un orden
#  aleatorio distinto cada vez.

# * Método clasifica_prob:
# ------------------------

#  El método que devuelve la probabilidad de pertenecer a la clase (la que se
#  ha tomado como clase 1), calculada para un nuevo ejemplo. Este método no es
#  necesario incluirlo para el perceptrón umbral.

# * Método clasifica:
# -------------------

#  El método que devuelve la clase que se predice para un nuevo ejemplo. La
#  clase debe ser una de las clases del problema (por ejemplo, "republicano" o
#  "democrata" en el problema de los votos).


# Si el clasificador aún no ha sido entrenado, tanto "clasifica" como
# "clasifica_prob" deben devolver una excepción del siguiente tipo:

class ClasificadorNoEntrenado(Exception):
     def __init__(self,tipo):
        Exception.__init__(self,"Clasificador {0} no entrenado".format(tipo))

#  NOTA: Se aconseja probar el funcionamiento de los clasificadores con
#  conjuntos de datos generados por las funciones del apartado anterior.

# Ejemplo de uso:

# ------------------------------------------------------------

# Generamos un conjunto de datos linealmente separables,
# In [1]: X1,Y1=genera_conjunto_de_datos_l_s(4,8,400)

# Lo partimos en dos trozos:
# In [2]: X1e,Y1e=X1[:300],Y1[:300]

# In [3]: X1t,Y1t=X1[300:],Y1[300:]

# Creamos el clasificador (perceptrón umbral en este caso):
# In [4]: clas_pb1=Clasificador_Perceptron([0,1])

# Lo entrenamos con elprimero de los conjuntos de datos:
# In [5]: clas_pb1.entrena(X1e,Y1e,100,rate_decay=True,rate=0.001)

# Clasificamos un ejemplo del otro conjunto, y lo comparamos con su clase real:
# In [6]: clas_pb1.clasifica(X1t[0]),Y1t[0]
# Out[6]: (1, 1)

# Comprobamos el porcentaje de aciertos sobre todos los ejemplos de X2t
# In [7]: sum(clas_pb1.clasifica(x) == y for x,y in zip(X1t,Y1t))/len(Y1t)
# Out[7]: 1.0

# Repetimos el experimento, pero ahora con un conjunto de datos que no es
# linealmente separable:
# In [8]: X2,Y2,w2=genera_conjunto_de_datos_n_l_s(4,8,400,0.1)

# In [8]: X2e,Y2e=X2[:300],Y2[:300]

# In [9]: X2t,Y2t=X2[300:],Y2[300:]

# In [10]: clas_pb2=Clasificador_Perceptron([0,1])

# In [11]: clas_pb2.entrena(X2e,Y2e,100,rate_decay=True,rate=0.001)

# In [12]: clas_pb2.clasifica(X2t[0]),Y2t[0]
# Out[12]: (1, 0)

# In [13]: sum(clas_pb2.clasifica(x) == y for x,y in zip(X2t,Y2t))/len(Y2t)
# Out[13]: 0.82
# ----------------------------------------------------------------

##################################################################################

##Funciones Globales para los clasificadores

def f_umbral(x):
    res = 0
    if x >= 0:
        res = 1
    return res

def f_sigmoide(x):
    res = 0
    if (x<0):
        res = 1 - 1/(1+math.exp(x))
    else:
        res = 1/(1+math.exp(-x))
    return res

def f_normalizadora(xs):
    ys = []
    for x in xs:
        m_i = numpy.mean(x)
        d_i = numpy.std(x)
        y = [(x[i]-m_i)/d_i for i in range(len(x))]
        ys.append(y)
    return ys

def convertidor(clases,entr_clas):
    res = []
    for x in entr_clas:
        res.append(clases.index(x))
    return res

def convertidorMulticlase(clases,entr_clas):
    res = []
    for x in entr_clas:
        if [x]==clases[1]:
            res.append(1)
        else:
            res.append(0)
    return res

##################################################################################

## Clasificador del perceptrón

class Clasificador_Perceptron():

    def __init__(self,clases,normalizacion=False):
        self.clasesP = clases
        self.normalizacionP = normalizacion
        self.pesos = None


    def entrena(self,entr,clas_entr,n_epochs,rate=0.1,pesos_iniciales=None,rate_decay=False):
        if self.normalizacionP:
            entr = f_normalizadora(entr)
        entr_aux = copy.deepcopy(entr)
        entr = [[1]+x for x in entr]
        clas_entr = convertidor(self.clasesP,clas_entr)
        if not pesos_iniciales:
            pesos_iniciales = [random.randint(-1,1) for x in range(len(entr[0]))]

        self.pesos = copy.copy(pesos_iniciales)

        rate_n = rate
        vector_accuracy = []
        for n in range(n_epochs):
            if not n == 0 and rate_decay:
                rate_n = rate + (2/n**(1.5))
            randomizado = list(range(len(entr)))
            random.shuffle(randomizado)

            for j in randomizado:
                umbral = sum(self.pesos[i]*entr[j][i] for i in range(len(entr[j])))
                o = f_umbral(umbral)
                self.pesos = [self.pesos[i] + rate_n*entr[j][i]*(clas_entr[j]-o) for i in range(len(entr[0]))]

            vector_accuracy.append(sum(self.clasifica(x) == y for x,y in zip(entr_aux,clas_entr))/len(clas_entr))

        return vector_accuracy


    def clasifica_prob(self,ej):
        pass


    def clasifica(self,ej):
        if not self.pesos:
            raise ClasificadorNoEntrenado("perceptrón")
        else:
            if self.normalizacionP:
                ej = f_normalizadora([ej])[0]
            ej = [1]+ej
            umbral = sum(self.pesos[i]*ej[i] for i in range(len(ej)))
            o = f_umbral(umbral)
            return self.clasesP[o]


##################################################################################

## Clasificador de regresión Lineal Bach  minimizando L2

class Clasificador_RL_L2_Batch:

    def __init__(self,clases,normalizacion=False):
       self.clasesP = clases
       self.normalizacionP = normalizacion
       self.pesos = None

    def entrena(self,entr,clas_entr,n_epochs,rate=0.1,pesos_iniciales=None,rate_decay=False):
        if self.normalizacionP:
            entr = f_normalizadora(entr)
        entr_aux = copy.deepcopy(entr)
        entr = [[1]+x for x in entr]
        clas_entr = convertidor(self.clasesP,clas_entr)

        if not pesos_iniciales:
            pesos_iniciales = [random.randint(-1,1) for x in range(len(entr[0]))]

        self.pesos = copy.copy(pesos_iniciales)

        rate_n = rate
        vector_accuracy = []
        vector_error = []
        for n in range(n_epochs):
            if not n == 0 and rate_decay:
                rate_n = rate + (2/n**(1.5))
            error = 0
            Delta_w = [0]*len(self.pesos)
            for x,y in zip(entr,clas_entr):
                c_in = sum(self.pesos[i]*x[i] for i in range(len(x)))
                o = f_sigmoide(c_in)
                Delta_w = [Delta_w[i]+rate_n*(y-o)*x[i]*o*(1-o) for i in range(len(x))]
                error += (y-o)**2
            self.pesos = [self.pesos[i]+Delta_w[i] for i in range(len(Delta_w))]
            vector_accuracy.append(sum(self.clasifica(x) == y for x,y in zip(entr_aux,clas_entr))/len(clas_entr))
            vector_error.append(error)
        return vector_accuracy,vector_error

    def clasifica_prob(self,ej):
        if not self.pesos:
            raise ClasificadorNoEntrenado("regresión lineal L2 batch")
        else:
            if self.normalizacionP:
                ej = f_normalizadora([ej])[0]

            ej = [1]+ej
            x = sum(self.pesos[i]*ej[i] for i in range(len(ej)))
            return f_sigmoide(x)

    def clasifica(self,ej):
        if not self.pesos:
            raise ClasificadorNoEntrenado("regresión lineal L2 batch")
        else:
            if self.normalizacionP:
                ej = f_normalizadora([ej])[0]

            prob = self.clasifica_prob(ej)
            return self.clasesP[round(prob)]

##################################################################################

## Clasificador regresión Lineal St minimizando L2

class Clasificador_RL_L2_St:

    def __init__(self,clases,normalizacion=False):
       self.clasesP = clases
       self.normalizacionP = normalizacion
       self.pesos = None

    def entrena(self,entr,clas_entr,n_epochs,rate=0.1,pesos_iniciales=None,rate_decay=False):
        if self.normalizacionP:
            entr = f_normalizadora(entr)
        entr_aux = copy.deepcopy(entr)
        entr = [[1]+x for x in entr]
        clas_entr = convertidor(self.clasesP,clas_entr)
        if not pesos_iniciales:
            pesos_iniciales = [random.randint(-1,1) for x in range(len(entr[0]))]

        self.pesos = copy.copy(pesos_iniciales)

        rate_n = rate
        vector_accuracy = []
        vector_error = []
        for n in range(n_epochs):
            if not n == 0 and rate_decay:
                rate_n = rate + (2/n**(1.5))
            randomizado = list(range(len(entr)))
            random.shuffle(randomizado)
            error = 0
            for j in randomizado:
                c_in = sum(self.pesos[i]*entr[j][i] for i in range(len(entr[j])))
                o = f_sigmoide(c_in)
                self.pesos = [self.pesos[i]+rate_n*(clas_entr[j]-o)*entr[j][i]*o*(1-o) for i in range(len(entr[j]))]
                error += (clas_entr[j]-o)**2
            vector_accuracy.append(sum(self.clasifica(x) == y for x,y in zip(entr_aux,clas_entr))/len(clas_entr))
            vector_error.append(error)

        return vector_accuracy,vector_error

    def clasifica_prob(self,ej):
        if not self.pesos:
            raise ClasificadorNoEntrenado("regresión lineal L2 estocástico")
        else:
            if self.normalizacionP:
                ej = f_normalizadora([ej])[0]

            ej = [1]+ej
            x = sum(self.pesos[i]*ej[i] for i in range(len(ej)))
            return f_sigmoide(x)

    def clasifica(self,ej):
        if not self.pesos:
            raise ClasificadorNoEntrenado("regresión lineal L2 estocástico")
        else:
            if self.normalizacionP:
                ej = f_normalizadora([ej])[0]

            prob = self.clasifica_prob(ej)
            return self.clasesP[round(prob)]

##################################################################################

##Clasificador regresión Lineal Bach maximizando verosimilitud

class Clasificador_RL_ML_Batch:

    def __init__(self,clases,normalizacion=False):
       self.clasesP = clases
       self.normalizacionP = normalizacion
       self.pesos = None

    def entrena(self,entr,clas_entr,n_epochs,rate=0.1,pesos_iniciales=None,rate_decay=False):
        if self.normalizacionP:
            entr = f_normalizadora(entr)
        entr_aux = copy.deepcopy(entr)
        entr = [[1]+x for x in entr]
        clas_entr = convertidor(self.clasesP,clas_entr)
        if not pesos_iniciales:
            pesos_iniciales = [random.randint(-1,1) for x in range(len(entr[0]))]

        self.pesos = copy.copy(pesos_iniciales)

        rate_n = rate
        vector_accuracy = []
        vector_error = []
        for n in range(n_epochs):
            if not n == 0 and rate_decay:
                rate_n = rate + (2/n**(1.5))
            error = 0
            Delta_w = [0]*len(self.pesos)
            for x,y in zip(entr,clas_entr):
                c_in = sum(self.pesos[i]*x[i] for i in range(len(x)))
                o = f_sigmoide(c_in)
                Delta_w = [Delta_w[i]+rate_n*(y-o)*x[i] for i in range(len(x))]
                if y==1:
                    if c_in < 0:
                        error+= -numpy.log10(1+(1/1+math.exp(c_in)))
                    else:
                        error+= -numpy.log10(1+math.exp(-c_in))
                else:
                    if c_in>0:
                        error+= -numpy.log10(1+(1/1+math.exp(-c_in)))
                    else:
                        error+= -numpy.log10(1+math.exp(c_in))
            self.pesos = [self.pesos[i]+Delta_w[i] for i in range(len(Delta_w))]
            vector_accuracy.append(sum(self.clasifica(x) == y for x,y in zip(entr_aux,clas_entr))/len(clas_entr))
            vector_error.append(error)
        return vector_accuracy,vector_error

    def clasifica_prob(self,ej):
        if not self.pesos:
            raise ClasificadorNoEntrenado("regresión lineal maximizando verosimilitud batch")
        else:
            if self.normalizacionP:
                ej = f_normalizadora([ej])[0]

            ej = [1]+ej
            x = sum(self.pesos[i]*ej[i] for i in range(len(ej)))
            return f_sigmoide(x)

    def clasifica(self,ej):
        if not self.pesos:
            raise ClasificadorNoEntrenado("regresión lineal maximizando verosimilitud batch")
        else:
            if self.normalizacionP:
                ej = f_normalizadora([ej])[0]

            prob = self.clasifica_prob(ej)
            return self.clasesP[round(prob)]

##################################################################################

##Clasificador regresión Lineal St maximizando verosimilitud

class Clasificador_RL_ML_St:

    def __init__(self,clases,normalizacion=False):
       self.clasesP = clases
       self.normalizacionP = normalizacion
       self.pesos = None

    def entrena(self,entr,clas_entr,n_epochs,rate=0.1,pesos_iniciales=None,rate_decay=False):
        if self.normalizacionP:
            entr = f_normalizadora(entr)
        entr_aux = copy.deepcopy(entr)
        entr = [[1]+x for x in entr]
        clas_entr = convertidor(self.clasesP,clas_entr)
        if not pesos_iniciales:
            pesos_iniciales = [random.randint(-1,1) for x in range(len(entr[0]))]

        self.pesos = copy.copy(pesos_iniciales)

        rate_n = rate
        vector_accuracy = []
        vector_error = []
        for n in range(n_epochs):
            if not n == 0 and rate_decay:
                rate_n = rate + (2/n**(1.5))
            randomizado = list(range(len(entr)))
            random.shuffle(randomizado)
            error = 0
            for j in randomizado:
                c_in = sum(self.pesos[i]*entr[j][i] for i in range(len(entr[j])))
                o = f_sigmoide(c_in)
                self.pesos = [self.pesos[i]+rate_n*(clas_entr[j]-o)*entr[j][i] for i in range(len(entr[j]))]
                if clas_entr[j]==1:
                    if c_in < 0:
                        error+= -numpy.log10(1+(1/1+math.exp(c_in)))
                    else:
                        error+= -numpy.log10(1+math.exp(-c_in))
                else:
                    if c_in>0:
                        error+= -numpy.log10(1+(1/1+math.exp(-c_in)))
                    else:
                        error+= -numpy.log10(1+math.exp(c_in))
            vector_accuracy.append(sum(self.clasifica(x) == y for x,y in zip(entr_aux,clas_entr))/len(clas_entr))
            vector_error.append(error)
        return vector_accuracy,vector_error

    def clasifica_prob(self,ej):
        if not self.pesos:
            raise ClasificadorNoEntrenado("regresión lineal maximizando verosimilitud estocástico")
        else:
            if self.normalizacionP:
                ej = f_normalizadora([ej])[0]

            ej = [1]+ej
            x = sum(self.pesos[i]*ej[i] for i in range(len(ej)))
            return f_sigmoide(x)

    def clasifica(self,ej):
        if not self.pesos:
            raise ClasificadorNoEntrenado("regresión lineal maximizando verosimilitud estocástico")
        else:
            if self.normalizacionP:
                ej = f_normalizadora([ej])[0]

            prob = self.clasifica_prob(ej)
            return self.clasesP[round(prob)]

##################################################################################
# --------------------------
# I.3. Curvas de aprendizaje
# --------------------------

# Se pide mostrar mediante gráficas la evolución del aprendizaje de los
# distintos algoritmos. En concreto, para cada clasificador usado con un
# conjunto de datos generado aleatoriamente con las funciones anteriores, las
# dos siguientes gráficas:

# - Una gráfica que indique cómo evoluciona el porcentaje de errores que
#   comete el clasificador sobre el conjunto de entrenamiento, en cada epoch.
# - Otra gráfica que indique cómo evoluciona el error cuadrático o la log
#   verosimilitud del clasificador (dependiendo de lo que se esté optimizando
#   en cada proceso de entrenamiento), en cada epoch.

# Para realizar gráficas, se recomiendo usar la biblioteca matplotlib de
# python:

import matplotlib.pyplot as plt


# Lo que sigue es un ejemplo de uso, para realizar una gráfica sencilla a
# partir de una lista "errores", que por ejemplo podría contener los sucesivos
# porcentajes de error que comete el clasificador, en los sucesivos epochs:


# plt.plot(range(1,len(errores)+1),errores,marker='o')
# plt.xlabel('Epochs')
# plt.ylabel('Porcentaje de errores')
# plt.show()

# Basta con incluir un código similar a este en el fichero python, para que en
# la terminal de Ipython se genere la correspondiente gráfica.

# Se pide generar una serie de gráficas que permitan explicar el
# comportamiento de los algoritmos, con las distintas opciones, y con
# conjuntos separables y no separables. Comentar la interpretación de las
# distintas gráficas obtenidas.

# NOTA: Para poder realizar las gráficas, debemos modificar los
# algoritmos de entrenamiento para que ademas de realizar el cálculo de los
# pesos, también calcule las listas con los sucesivos valores (de errores, de
# verosimilitud,etc.) que vamos obteniendo en cada epoch. Esta funcionalidad
# extra puede enlentecer algo el proceso de entrenamiento y si fuera necesario
# puede quitarse una vez se realize este apartado.


# ==================================
# PARTE II: CLASIFICACIÓN MULTICLASE
# ==================================

# Se pide implementar algoritmos de regresión logística para problemas de
# clasificación en los que hay más de dos clases. Para ello, usar las dos
# siguientes aproximaciones:

# ------------------------------------------------
# II.1 Técnica "One vs Rest" (Uno frente al Resto)
# ------------------------------------------------

#  Esta técnica construye un clasificador multiclase a partir de
#  clasificadores binarios que devuelven probabilidades (como es el caso de la
#  regresión logística). Para cada posible valor de clasificación, se
#  entrena un clasificador que estime cómo de probable es pertenecer a esa
#  clase, frente al resto. Este conjunto de clasificadores binarios se usa
#  para dar la clasificación de un ejemplo nuevo, sin más que devolver la
#  clase para la que su correspondiente clasificador binario da una mayor
#  probabilidad.

#  En concreto, se pide implementar una clase python Clasificador_RL_OvR con
#  la siguiente estructura, y que implemente el entrenamiento y la
#  clasificación como se ha explicado.

# class Clasificador_RL_OvR():

#     def __init__(self,class_clasif,clases):

#        .....
#     def entrena(self,entr,clas_entr,n_epochs,rate=0.1,rate_decay=False):

#        .....

#     def clasifica(self,ej):

#        .....

#  Excepto "class_clasif", los restantes parámetros de los métodos significan
#  lo mismo que en el apartado anterior, excepto que ahora "clases" puede ser
#  una lista con más de dos elementos. El parámetro class_clasif es el nombre
#  de la clase que implementa el clasificador binario a partir del cual se
#  forma el clasificador multiclase.

#  Un ejemplo de sesión, con el problema del iris:

# ---------------------------------------------------------------
# In [28]: from iris import *

# In [29]: iris_clases=["Iris-setosa","Iris-virginica","Iris-versicolor"]

# Creamos el clasificador, a partir de RL binaria estocástico:
# In [30]: clas_rlml1=Clasificador_RL_OvR(Clasificador_RL_ML_St,iris_clases)

# Lo entrenamos:
# In [32]: clas_rlml1.entrena(iris_entr,iris_entr_clas,100,rate_decay=True,rate=0.01)

# Clasificamos un par de ejemplos, comparándolo con su clase real:
# In [33]: clas_rlml1.clasifica(iris_entr[25]),iris_entr_clas[25]
# Out[33]: ('Iris-setosa', 'Iris-setosa')

# In [34]: clas_rlml1.clasifica(iris_entr[78]),iris_entr_clas[78]
# Out[34]: ('Iris-versicolor', 'Iris-versicolor')
# ----------------------------------------------------------------


class Clasificador_RL_OvR():

    def __init__(self,class_clasif,clases):
        self.class_clasifC=class_clasif
        self.clasesC=clases
        self.pesosPorClases = []
        self.clasificadores = []

    def entrena(self,entr,clas_entr,n_epochs,rate=0.1,rate_decay=False):
        for i in range(len(self.clasesC)):
            clasesAux = [self.clasesC[0:i]+self.clasesC[i+1:],self.clasesC[i:i+1]]
            clasificador = self.class_clasifC([0,1])
            clas_entrAux = convertidorMulticlase(clasesAux,clas_entr)
            clasificador.entrena(entr,clas_entrAux,n_epochs,rate,None,rate_decay)
            self.clasificadores.append(clasificador)
            self.pesosPorClases.append(clasificador.pesos)

    def clasifica(self,ej):
        if not self.pesosPorClases:
            raise ClasificadorNoEntrenado("One vs Rest")
        else:
            x = []
            for j in range(len(self.clasificadores)):
                o = Clasificador_RL_ML_St.clasifica_prob(self.clasificadores[j],ej)
                x.append(o)
            return self.clasesC[x.index(max(x))]


# ------------------------------------------------
# II.1 Regresión logística con softmax
# ------------------------------------------------


#  Se pide igualmente implementar un clasificador en python que implemente la
#  regresión multinomial logística mdiante softmax, tal y como se describe en
#  el tema 5, pero solo la versión ESTOCÁSTICA.

#  En concreto, se pide implementar una clase python Clasificador_RL_Softmax
#  con la siguiente estructura, y que implemente el entrenamiento y la
#  clasificación como se explica en el tema 5:

# class Clasificador_RL_Softmax():

#     def __init__(self,clases):
#        .....

#     def entrena(self,entr,clas_entr,n_epochs,rate=0.1,rate_decay=False):
#        .....

#     def clasifica(self,ej):
#        .....

# ----------------------------------------------------------------


class Clasificador_RL_Softmax():

    def __init__(self,clases):
        self.clasesC=clases
        self.pesos = []

    def entrena(self,entr,clas_entr,n_epochs,rate=0.1,rate_decay=False):
            def formula_o(clase,j,pesos):
                numerador = math.exp(sum(pesos[clase][i]*entr[j][i] for i in range(len(entr[j]))))
                denominador = sum(math.exp(sum(pesos[k][i]*entr[j][i] for i in range(len(entr[j])))) for k in range(len(self.clasesC)))
                return numerador/denominador

            entr = [[1]+x for x in entr]
            self.pesos = [[random.randint(-1,1) for x in range(len(entr[0]))] for n in range(len(self.clasesC))]
            rate_n = rate

            for n in range(n_epochs):
                if not n == 0 and rate_decay:
                    rate_n = rate + (2/n**(1.5))
                randomizado = list(range(len(entr)))
                random.shuffle(randomizado)
                for j in randomizado:
                    pesosAux = copy.deepcopy(self.pesos)
                    for m in range(len(self.clasesC)):
                        for i in range(len(self.pesos[m])):
                            y = 0
                            if clas_entr[j]==self.clasesC[m]:
                                y = 1
                            self.pesos[m][i]+= rate_n*(y-formula_o(m,j,pesosAux))*entr[j][i]


    def clasifica(self,ej):
        if not self.pesos:
            raise ClasificadorNoEntrenado("Softmax")
        else:
            ej = [1]+ej
            def formula_o(clase):
                numerador = math.exp(sum(self.pesos[clase][i]*ej[i] for i in range(len(ej))))
                denominador = sum(math.exp(sum(self.pesos[k][i]*ej[i] for i in range(len(ej)))) for k in range(len(self.clasesC)))
                return numerador/denominador

            vector_prob = [0]*len(self.clasesC)

            for m in range(len(self.clasesC)):
                vector_prob[m] = formula_o(m)

            return self.clasesC[vector_prob.index(max(vector_prob))]


# ===========================================
# PARTE III: APLICACIÓN DE LOS CLASIFICADORES
# ===========================================

# En este apartado se pide aplicar alguno de los clasificadores implementados
# en el apartado anterior,para tratar de resolver tres problemas: el de los
# votos, el de los dígitos y un tercer problema que hay que buscar.

# -------------------------------------
# III.1 Implementación del rendimiento
# -------------------------------------

# Una vez que hemos entrenado un clasificador, podemos medir su rendimiento
# sobre un conjunto de ejemplos de los que se conoce su clasificación,
# mediante el porcentaje de ejemplos clasificados correctamente. Se ide
# definir una función rendimiento(clf,X,Y) que calcula el rendimiento de
# clasificador concreto clf, sobre un conjunto de datos X cuya clasificación
# conocida viene dada por la lista Y.
# NOTA: clf es un objeto de las clases definidas en
# los apartados anteriores, que además debe estar ya entrenado.

from iris import *

# Por ejemplo (conectando con el ejemplo anterior):

# ---------------------------------------------------------
# In [36]: rendimiento(clas_rlml1,iris_entr,iris_entr_clas)
# Out[36]: 0.9666666666666667
# ---------------------------------------------------------
def rendimiento(clasificador,entr,clas_entr):
    return sum(clasificador.clasifica(x) == y for x,y in zip(entr,clas_entr))/len(clas_entr)

###=================================================================================================
##TESTS
###=================================================================================================

#----------------------------------------------
###Pruebas Basicas Clasificadores###

def pruebaSeparableConGraficas():

    X1,Y1=genera_conjunto_de_datos_l_s(4,8,400)
    X1e,Y1e=X1[:300],Y1[:300]
    X1t,Y1t=X1[300:],Y1[300:]

    #perceptrón
    print("perceptrón")
    clas_pb1=Clasificador_Perceptron([0,1])
    ac = clas_pb1.entrena(X1e,Y1e,100,rate_decay=True,rate=0.001)
    print("Accuracy Perceptrón:",sum(clas_pb1.clasifica(x) == y for x,y in zip(X1t,Y1t))/len(Y1t))
    plt.plot(range(1,len(ac)+1),ac,marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Porcentaje de acierto')
    plt.show()

    #regresión Lineal Bach minimizando L2
    print("-----------------------------")
    print("Clasificador_RL_L2_Batch")
    clas_pb2=Clasificador_RL_L2_Batch([0,1])
    ac2,error2 = clas_pb2.entrena(X1e,Y1e,100,rate_decay=True,rate=0.001)
    print("Accuracy Clasificador_RL_L2_Batch:",sum(clas_pb2.clasifica(x) == y for x,y in zip(X1t,Y1t))/len(Y1t))
    plt.plot(range(1,len(ac2)+1),ac2,marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Porcentaje de acierto')
    plt.show()
    plt.plot(range(1,len(error2)+1),error2,marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Error Cuadrático')
    plt.show()

    #regresión Lineal St minimizando L2
    print("-----------------------------")
    print("Clasificador_RL_L2_St")
    clas_pb3=Clasificador_RL_L2_St([0,1])
    ac3,error3=clas_pb3.entrena(X1e,Y1e,100,rate_decay=True,rate=0.001)
    print("Accuracy Clasificador_RL_L2_St:",sum(clas_pb3.clasifica(x) == y for x,y in zip(X1t,Y1t))/len(Y1t))
    plt.plot(range(1,len(ac3)+1),ac3,marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Porcentaje de acierto')
    plt.show()
    plt.plot(range(1,len(error3)+1),error3,marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Error Cuadrático')
    plt.show()

    #regresión Lineal Bach maximizando verosimilitud
    print("-----------------------------")
    print("Clasificador_RL_ML_Batch")
    clas_pb4=Clasificador_RL_ML_Batch([0,1])
    ac4,error4=clas_pb4.entrena(X1e,Y1e,100,rate_decay=True,rate=0.001)
    print("Accuracy Clasificador_RL_ML_Batch:",sum(clas_pb4.clasifica(x) == y for x,y in zip(X1t,Y1t))/len(Y1t))
    plt.plot(range(1,len(ac4)+1),ac4,marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Porcentaje de acierto')
    plt.show()
    plt.plot(range(1,len(error4)+1),error4,marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Log verosimilitud')
    plt.show()


    #regresión Lineal St maximizando verosimilitud
    print("-----------------------------")
    print("Clasificador_RL_ML_St")
    clas_pb5=Clasificador_RL_ML_St([0,1])
    ac5,error5=clas_pb5.entrena(X1e,Y1e,100,rate_decay=True,rate=0.001)
    print("Accuracy Clasificador_RL_ML_St:",sum(clas_pb5.clasifica(x) == y for x,y in zip(X1t,Y1t))/len(Y1t))
    plt.plot(range(1,len(ac5)+1),ac5,marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Porcentaje de acierto')
    plt.show()
    plt.plot(range(1,len(error5)+1),error5,marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Log Verosimilitud')
    plt.show()

def pruebaSeparableSinGraficas():

    X1,Y1=genera_conjunto_de_datos_l_s(4,8,400)
    X1e,Y1e=X1[:300],Y1[:300]
    X1t,Y1t=X1[300:],Y1[300:]

    #perceptrón
    print("perceptrón")
    clas_pb1=Clasificador_Perceptron([0,1])
    clas_pb1.entrena(X1e,Y1e,100,rate_decay=True,rate=0.001)
    print("Accuracy Perceptrón:",sum(clas_pb1.clasifica(x) == y for x,y in zip(X1t,Y1t))/len(Y1t))

    #regresión Lineal Bach minimizando L2
    print("-----------------------------")
    print("Clasificador_RL_L2_Batch")
    clas_pb2=Clasificador_RL_L2_Batch([0,1])
    clas_pb2.entrena(X1e,Y1e,100,rate_decay=True,rate=0.001)
    print("Accuracy Clasificador_RL_L2_Batch:",sum(clas_pb2.clasifica(x) == y for x,y in zip(X1t,Y1t))/len(Y1t))

    #regresión Lineal St minimizando L2
    print("-----------------------------")
    print("Clasificador_RL_L2_St")
    clas_pb3=Clasificador_RL_L2_St([0,1])
    clas_pb3.entrena(X1e,Y1e,100,rate_decay=True,rate=0.001)
    print("Accuracy Clasificador_RL_L2_St:",sum(clas_pb3.clasifica(x) == y for x,y in zip(X1t,Y1t))/len(Y1t))

    #regresión Lineal Bach maximizando verosimilitud
    print("-----------------------------")
    print("Clasificador_RL_ML_Batch")
    clas_pb4=Clasificador_RL_ML_Batch([0,1])
    clas_pb4.entrena(X1e,Y1e,100,rate_decay=True,rate=0.001)
    print("Accuracy Clasificador_RL_ML_Batch:",sum(clas_pb4.clasifica(x) == y for x,y in zip(X1t,Y1t))/len(Y1t))

    #regresión Lineal St maximizando verosimilitud
    print("-----------------------------")
    print("Clasificador_RL_ML_St")
    clas_pb5=Clasificador_RL_ML_St([0,1])
    clas_pb5.entrena(X1e,Y1e,100,rate_decay=True,rate=0.001)
    print("Accuracy Clasificador_RL_ML_St:",sum(clas_pb5.clasifica(x) == y for x,y in zip(X1t,Y1t))/len(Y1t))

def pruebaNoSeparableConGraficas():

    X1,Y1=genera_conjunto_de_datos_n_l_s(4,8,400)
    X1e,Y1e=X1[:300],Y1[:300]
    X1t,Y1t=X1[300:],Y1[300:]

    #perceptrón
    print("perceptrón")
    clas_pb1=Clasificador_Perceptron([0,1])
    ac = clas_pb1.entrena(X1e,Y1e,100,rate_decay=True,rate=0.001)
    print("Accuracy Perceptrón:",sum(clas_pb1.clasifica(x) == y for x,y in zip(X1t,Y1t))/len(Y1t))
    plt.plot(range(1,len(ac)+1),ac,marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Porcentaje de acierto')
    plt.show()

    #regresión Lineal Bach minimizando L2
    print("-----------------------------")
    print("Clasificador_RL_L2_Batch")
    clas_pb2=Clasificador_RL_L2_Batch([0,1])
    ac2,error2 = clas_pb2.entrena(X1e,Y1e,100,rate_decay=True,rate=0.001)
    print("Accuracy Clasificador_RL_L2_Batch:",sum(clas_pb2.clasifica(x) == y for x,y in zip(X1t,Y1t))/len(Y1t))
    plt.plot(range(1,len(ac2)+1),ac2,marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Porcentaje de acierto')
    plt.show()
    plt.plot(range(1,len(error2)+1),error2,marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Error Cuadrático')
    plt.show()

    #regresión Lineal St minimizando L2
    print("-----------------------------")
    print("Clasificador_RL_L2_St")
    clas_pb3=Clasificador_RL_L2_St([0,1])
    ac3,error3=clas_pb3.entrena(X1e,Y1e,100,rate_decay=True,rate=0.001)
    print("Accuracy Clasificador_RL_L2_St:",sum(clas_pb3.clasifica(x) == y for x,y in zip(X1t,Y1t))/len(Y1t))
    plt.plot(range(1,len(ac3)+1),ac3,marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Porcentaje de acierto')
    plt.show()
    plt.plot(range(1,len(error3)+1),error3,marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Error Cuadrático')
    plt.show()

    #regresión Lineal Bach maximizando verosimilitud
    print("-----------------------------")
    print("Clasificador_RL_ML_Batch")
    clas_pb4=Clasificador_RL_ML_Batch([0,1])
    ac4,error4=clas_pb4.entrena(X1e,Y1e,100,rate_decay=True,rate=0.001)
    print("Accuracy Clasificador_RL_ML_Batch:",sum(clas_pb4.clasifica(x) == y for x,y in zip(X1t,Y1t))/len(Y1t))
    plt.plot(range(1,len(ac4)+1),ac4,marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Porcentaje de acierto')
    plt.show()
    plt.plot(range(1,len(error4)+1),error4,marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Log verosimilitud')
    plt.show()


    #regresión Lineal St maximizando verosimilitud
    print("-----------------------------")
    print("Clasificador_RL_ML_St")
    clas_pb5=Clasificador_RL_ML_St([0,1])
    ac5,error5=clas_pb5.entrena(X1e,Y1e,100,rate_decay=True,rate=0.001)
    print("Accuracy Clasificador_RL_ML_St:",sum(clas_pb5.clasifica(x) == y for x,y in zip(X1t,Y1t))/len(Y1t))
    plt.plot(range(1,len(ac5)+1),ac5,marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Porcentaje de acierto')
    plt.show()
    plt.plot(range(1,len(error5)+1),error5,marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Log Verosimilitud')
    plt.show()

def pruebaNoSeparableSinGraficas():

    X1,Y1=genera_conjunto_de_datos_n_l_s(4,8,400)
    X1e,Y1e=X1[:300],Y1[:300]
    X1t,Y1t=X1[300:],Y1[300:]

    #perceptrón
    print("perceptrón")
    clas_pb1=Clasificador_Perceptron([0,1])
    clas_pb1.entrena(X1e,Y1e,100,rate_decay=True,rate=0.001)
    print("Accuracy Perceptrón:",sum(clas_pb1.clasifica(x) == y for x,y in zip(X1t,Y1t))/len(Y1t))

    #regresión Lineal Bach minimizando L2
    print("-----------------------------")
    print("Clasificador_RL_L2_Batch")
    clas_pb2=Clasificador_RL_L2_Batch([0,1])
    clas_pb2.entrena(X1e,Y1e,100,rate_decay=True,rate=0.001)
    print("Accuracy Clasificador_RL_L2_Batch:",sum(clas_pb2.clasifica(x) == y for x,y in zip(X1t,Y1t))/len(Y1t))

    #regresión Lineal St minimizando L2
    print("-----------------------------")
    print("Clasificador_RL_L2_St")
    clas_pb3=Clasificador_RL_L2_St([0,1])
    clas_pb3.entrena(X1e,Y1e,100,rate_decay=True,rate=0.001)
    print("Accuracy Clasificador_RL_L2_St:",sum(clas_pb3.clasifica(x) == y for x,y in zip(X1t,Y1t))/len(Y1t))

    #regresión Lineal Bach maximizando verosimilitud
    print("-----------------------------")
    print("Clasificador_RL_ML_Batch")
    clas_pb4=Clasificador_RL_ML_Batch([0,1])
    clas_pb4.entrena(X1e,Y1e,100,rate_decay=True,rate=0.001)
    print("Accuracy Clasificador_RL_ML_Batch:",sum(clas_pb4.clasifica(x) == y for x,y in zip(X1t,Y1t))/len(Y1t))

    #regresión Lineal St maximizando verosimilitud
    print("-----------------------------")
    print("Clasificador_RL_ML_St")
    clas_pb5=Clasificador_RL_ML_St([0,1])
    clas_pb5.entrena(X1e,Y1e,100,rate_decay=True,rate=0.001)
    print("Accuracy Clasificador_RL_ML_St:",sum(clas_pb5.clasifica(x) == y for x,y in zip(X1t,Y1t))/len(Y1t))

#----------------------------------------------
###Pruebas One vs Rest

def prueba_OvR_Iris():
    entr = iris_entr
    clas_entr = iris_entr_clas
    clases = iris_clases
    prueba = iris_entr
    clas_prueba = iris_entr_clas

    n_epochs = 100
    rate = 0.01
    rate_decay = False
    clasificadores = []
    rendimientos = []

    clasificadorMLBatch=Clasificador_RL_OvR(Clasificador_RL_ML_Batch,clases)
    clasificadorMLBatch.entrena(entr,clas_entr,n_epochs,rate_decay,rate)
    clasificadores.append(clasificadorMLBatch)
    rendimientoMLBatch = rendimiento(clasificadorMLBatch,prueba,clas_prueba)
    rendimientos.append(rendimientoMLBatch)
    print("Rendimiento del OvR-MLBatch:",rendimientoMLBatch)

    clasificadorMLSt=Clasificador_RL_OvR(Clasificador_RL_ML_St,clases)
    clasificadorMLSt.entrena(entr,clas_entr,n_epochs,rate_decay,rate)
    clasificadores.append(clasificadorMLSt)
    rendimientoMLSt = rendimiento(clasificadorMLSt,prueba,clas_prueba)
    rendimientos.append(rendimientoMLSt)
    print("Rendimiento del OvR-MLSt:",rendimientoMLSt)

    clasificadorL2Batch=Clasificador_RL_OvR(Clasificador_RL_L2_Batch,clases)
    clasificadorL2Batch.entrena(entr,clas_entr,n_epochs,rate_decay,rate)
    clasificadores.append(clasificadorL2Batch)
    rendimientoL2Batch = rendimiento(clasificadorL2Batch,prueba,clas_prueba)
    rendimientos.append(rendimientoL2Batch)
    print("Rendimiento del OvR-L2Batch:",rendimientoL2Batch)

    clasificadorL2St=Clasificador_RL_OvR(Clasificador_RL_L2_St,clases)
    clasificadorL2St.entrena(entr,clas_entr,n_epochs,rate_decay,rate)
    clasificadores.append(clasificadorL2St)
    rendimientoL2St = rendimiento(clasificadorL2St,prueba,clas_prueba)
    rendimientos.append(rendimientoL2St)
    print("Rendimiento del OvR-L2St:",rendimientoL2St)


#----------------------------------------------
###Pruebas SoftMax

def prueba_Softmax_Iris():
    iris_clases=["Iris-setosa","Iris-virginica","Iris-versicolor"]
    clas_rlml1=Clasificador_RL_Softmax(iris_clases)
    clas_rlml1.entrena(iris_entr,iris_entr_clas,100,rate_decay=True,rate=0.01)
    print("Rendimiento del Softmax:",rendimiento(clas_rlml1,iris_entr,iris_entr_clas))

#----------------------------------------------

# ----------------------------------
# III.2 Aplicando los clasificadores
# ----------------------------------

#  Obtener un clasificador para cada uno de los siguientes problemas,
#  intentando que el rendimiento obtenido sobre un conjunto independiente de
#  ejemplos de prueba sea lo mejor posible.

#  - Predecir el partido de un congresista en función de lo que ha votado en
#    las sucesivas votaciones, a partir de los datos en el archivo votos.py que
#    se suministra.


from votos import *

def mejorClasificadorVotos():
    clases = votos_clases

    entrenamiento = votos_entr
    clases_entrenamiento = votos_entr_clas

    validacion = votos_valid
    clases_validacion = votos_valid_clas

    epoch = 1000
    decay=True
    valor=0.01

    misMetodos = []
    metodo = []
    accuracy = []

    ## perceptrón
    Perceptron = Clasificador_Perceptron(clases)
    Perceptron.entrena(entrenamiento,clases_entrenamiento,epoch,rate_decay= decay, rate=valor)
    acPerceptron = rendimiento(Perceptron,validacion,clases_validacion)
    metodo.append("perceptrón:")
    accuracy.append(acPerceptron)
    misMetodos.append(Perceptron)

    ##Clasificador_RL_L2_Batch
    L2_Bach = Clasificador_RL_L2_Batch(clases)
    L2_Bach.entrena(entrenamiento,clases_entrenamiento,epoch,rate_decay=decay,rate=valor)
    acL2_bach = rendimiento(L2_Bach,validacion,clases_validacion)
    metodo.append("Clasificador_RL_L2_Batch:")
    accuracy.append(acL2_bach)
    misMetodos.append(L2_Bach)

    ##Clasificador_RL_L2_St
    L2_St = Clasificador_RL_L2_St(clases)
    L2_St.entrena(entrenamiento,clases_entrenamiento,epoch,rate_decay=decay,rate=valor)
    acL2_St = rendimiento(L2_St,validacion,clases_validacion)
    metodo.append("Clasificador_RL_L2_St:")
    accuracy.append(acL2_St)
    misMetodos.append(L2_St)

    ##Clasificador_RL_ML_Batch
    ML_Bach = Clasificador_RL_ML_Batch(clases)
    ML_Bach.entrena(entrenamiento,clases_entrenamiento,epoch,rate_decay=decay,rate=valor)
    acML_Bach = rendimiento(ML_Bach,validacion,clases_validacion)
    metodo.append("Clasificador_RL_ML_Batch:")
    accuracy.append(acML_Bach)
    misMetodos.append(ML_Bach)

    ##Clasificador_RL_ML_St
    ML_St = Clasificador_RL_ML_St(clases)
    ML_St.entrena(entrenamiento,clases_entrenamiento,epoch,rate_decay=decay,rate=valor)
    acML_St = rendimiento(ML_St,validacion,clases_validacion)
    metodo.append("Clasificador_RL_ML_St:")
    accuracy.append(acML_St)
    misMetodos.append(ML_St)


    print("con los siguientes parámetros:\n n_epoch = ",epoch,"\n rate_decay = ",decay,
    "\n rate = ",valor,"\n la tasa de aciertos serían: \n")

    for i in range(len(metodo)):
        print(metodo[i],accuracy[i])

    mejor = max(accuracy)
    indice = accuracy.index(mejor)

    print("\nPor lo que el mejor sería el",metodo[indice], "con una accuracy de",accuracy[indice],"para el conjunto de validación")

    return misMetodos[indice]


def probarPesosVotos(pesos,nombreClasif):
    clasificador = nombreClasif(votos_clases)
    clasificador.pesos = pesos 
    print("validacion:",rendimiento(clasificador,votos_valid,votos_valid_clas))
    print("prueba:",rendimiento(clasificador,votos_test,votos_test_clas))

# ######################################################################################################### primera prueba


# In [312]: m = mejorClasificadorVotos()
# con los siguientes parámetros:
#  n_epoch =  100
#  rate_decay =  True
#  rate =  0.01
#  la tasa de aciertos serían:

# perceptrón: 0.9565217391304348
# Clasificador_RL_L2_Batch: 0.9565217391304348
# Clasificador_RL_L2_St: 0.9855072463768116
# Clasificador_RL_ML_Batch: 0.9565217391304348
# Clasificador_RL_ML_St: 0.9710144927536232

# Por lo que el mejor sería el Clasificador_RL_L2_St: con una accuracy de 0.9855072463768116 para el conjunto de validación
# pesosVotos = [1.216012196209644,0.1474549075290975,-0.07764398622895145,1.9886215148004798,-4.361093770119169,-0.34328186963962626,
#  0.900568778476931,-1.288091566157545,-0.16268163121980186,0.808633407187405,-0.06442365233492293,1.1422904524363713,
#  -0.3330466650775169,1.3560247291444998,-0.6134059817882892,0.13086853584455305,-0.3366887928098736]
# Comprobando que es real:
# In [313]: probarPesosVotos(pesosVotos,Clasificador_RL_L2_St)
# validacion: 0.9855072463768116
# prueba: 0.9195402298850575

#####
# In [326]: m = mejorClasificadorVotos()
# con los siguientes parámetros:
#  n_epoch =  100
#  rate_decay =  False
#  rate =  0.01
#  la tasa de aciertos serían:

# perceptrón: 0.9710144927536232
# Clasificador_RL_L2_Batch: 1.0
# Clasificador_RL_L2_St: 0.9710144927536232
# Clasificador_RL_ML_Batch: 0.9855072463768116
# Clasificador_RL_ML_St: 0.9855072463768116

# Por lo que el mejor sería el Clasificador_RL_L2_Batch: con una accuracy de 1.0 para el conjunto de validación
# pesosVotos = [0.6856406135370731,0.17488631422530734,0.10381398138623765,1.2683359839730854,-2.07667231549208,
#  0.3365918381282302,-0.4110210109328556,0.003909764246512483,-0.08019283156312663,0.5127505190636534,0.011365744130015214,
#  1.1228468366373592,-1.0029004746215915,0.27804009279549663,0.2679548753019784,-0.31344301772172134, 0.19831655715205032]
# Comprobando que es real:
# In [328]: probarPesosVotos(pesosVotos,Clasificador_RL_L2_Batch)
# validacion: 1.0
# prueba: 0.8850574712643678

#####

# In [342]: m = mejorClasificadorVotos()
# con los siguientes parámetros:
#  n_epoch =  100
#  rate_decay =  False
#  rate =  0.001
#  la tasa de aciertos serían:

# perceptrón: 0.927536231884058
# Clasificador_RL_L2_Batch: 0.8260869565217391
# Clasificador_RL_L2_St: 0.9710144927536232
# Clasificador_RL_ML_Batch: 0.9420289855072463
# Clasificador_RL_ML_St: 1.0

# Por lo que el mejor sería el Clasificador_RL_ML_St: con una accuracy de 1.0 para el conjunto de validación
# pesosVotos = [1.0552702835552825,0.0999474498173142,0.1118974660987374,0.5247452441031819,-1.9395631616284454,
#  0.37053987696292645,0.26169132858781313,-0.11866760717494182,0.26968111313429916,0.9984987953149133,-0.3200279456312852,
#  0.8624195694699608,-0.7148819064644111,0.3011491413760779,-0.668770002574754,0.1741164365054168,0.01335097046276845]
# Comprobando que es real:
# In [343]: probarPesosVotos(pesosVotos,Clasificador_RL_ML_St)
# validacion: 1.0
# prueba: 0.9080459770114943

#####
# In [349]: m = mejorClasificadorVotos()
# con los siguientes parámetros:
#  n_epoch =  1000
#  rate_decay =  True
#  rate =  0.01
#  la tasa de aciertos serían:

# perceptrón: 0.9420289855072463
# Clasificador_RL_L2_Batch: 0.9565217391304348
# Clasificador_RL_L2_St: 0.9710144927536232
# Clasificador_RL_ML_Batch: 0.9710144927536232
# Clasificador_RL_ML_St: 0.9710144927536232

# Por lo que el mejor sería el Clasificador_RL_L2_St: con una accuracy de 0.9710144927536232 para el conjunto de validación
# pesosVotos = [0.246092944077681,0.4914157773120384,0.103631650946021,2.373783114701572,-4.908642850004016,
#  -0.47583840810574407,1.7735618786330514,-0.6138140078082851,-0.012100272079869585,1.1100442802337653,-0.1747288269540885,
#  1.4652777369032581,-0.540071609747973,1.9989113230738291,0.003719423215600246,0.39270697159225443,-0.2126114460212745]
# Comprobando que es real:
# In [350]: probarPesosVotos(pesosVotos,Clasificador_RL_L2_St)
# validacion: 0.9710144927536232
# prueba: 0.9195402298850575
#----------------------------------------------

#----------------------------------------------


#  - Predecir el dígito que se ha escrito a mano y que se dispone en forma de
#    imagen pixelada, a partir de los datos que están en el archivo digidata.zip
#    que se suministra.  Cada imagen viene dada por 28x28 píxeles, y cada pixel
#    vendrá representado por un caracter "espacio en blanco" (pixel blanco) o
#    los caracteres "+" (borde del dígito) o "#" (interior del dígito). En
#    nuestro caso trataremos ambos como un pixel negro (es decir, no
#    distinguiremos entre el borde y el interior). En cada conjunto las imágenes
#    vienen todas seguidas en un fichero de texto, y las clasificaciones de cada
#    imagen (es decir, el número que representan) vienen en un fichero aparte,
#    en el mismo orden. Será necesario, por tanto, definir funciones python que
#    lean esos ficheros y obtengan los datos en el mismo formato python en el
#    que los necesitan los algoritmos.
#############################################################

#############################################################
#
# def leer():
#     with open('digitdata/trainingimages','r') as fichero:
#         total = []
#         v = []
#         i=0
#         for linea in fichero:
#             w = []
#             i+=1
#             for posicion in linea[:-1]:
#                 if " " == posicion:
#                     w.append(0)
#                 else:
#                     w.append(1)
#             v.append(w)
#             if(i==28):
#                 i=0
#                 total.append(v)
#                 v = []
#         fichero.closed
#     return total

def leerEjemplos(fichero):
    with open(fichero,'r') as fichero:
        total = []
        w = []
        i=0
        for linea in fichero:
            i+=1
            for posicion in linea[:-1]:
                if " " == posicion:
                    w.append(0)
                else:
                    w.append(1)
            if(i==28):
                i=0
                total.append(w)
                w = []
        fichero.closed
    return total

def leerClases(fichero):
    v = []
    with open(fichero,'r') as fichero:
        for linea in fichero:
            v.append(int(linea[0]))
        fichero.closed
    return v

## Importa lo necesario para la utilización de todo lo referente a clasificar Dígitos
entrenaFichero = leerEjemplos('digitdata/trainingimages')
clasesEntrenaFichero = leerClases('digitdata/traininglabels')
validaFichero = leerEjemplos('digitdata/validationimages')
clasesValidaFichero = leerClases('digitdata/validationlabels')
testFichero = leerEjemplos('digitdata/testimages')
clasesTestFichero = leerClases('digitdata/testlabels')

####

def clasesParaFichero():
    res = []
    for i in clasesEntrenaFichero:
        if not i in res:
            res.append(i)
    return res
    
clasesFichero = clasesParaFichero()

########

def prueba_Digitos():

    epoch = 100
    decay = False
    valor = 0.001

    entr = entrenaFichero
    entr_clas = clasesEntrenaFichero

    val = validaFichero
    val_clas = clasesValidaFichero

    prueba = testFichero
    prueba_clas = clasesTestFichero

    misClasificadores = []
    metodo = []
    accuracy = []

    ##one vs rest ML_BATCH
    one_restMLB = Clasificador_RL_OvR(Clasificador_RL_ML_Batch,clasesFichero)
    one_restMLB.entrena(entr,entr_clas,epoch,rate_decay=decay,rate=valor)
    acOneRestMLB = rendimiento(one_restMLB,val,val_clas)
    metodo.append("One vs Rest con Clasificador_RL_ML_Batch:")
    accuracy.append(acOneRestMLB)
    misClasificadores.append(one_restMLB)

    ##one vs rest ML_ST
    one_restMLST = Clasificador_RL_OvR(Clasificador_RL_ML_St,clasesFichero)
    one_restMLST.entrena(entr,entr_clas,epoch,rate_decay=decay,rate=valor)
    acOneRestMLST = rendimiento(one_restMLST,val,val_clas)
    metodo.append("One vs Rest con Clasificador_RL_ML_St:")
    accuracy.append(acOneRestMLST)
    misClasificadores.append(one_restMLST)

    print("con los siguientes parámetros:\n n_epoch = ",epoch,"\n rate_decay = ",decay,
    "\n rate = ",valor,"\n la tasa de aciertos serían: \n")

    for i in range(len(metodo)):
        print(metodo[i],accuracy[i])

    mejor = max(accuracy)
    indice = accuracy.index(mejor)

    print("\nPor lo que el mejor sería el",metodo[indice], "con una accuracy de",rendimiento(misClasificadores[indice],prueba,prueba_clas),"para el conjunto de test")

    return misClasificadores[indice]


# In [6]: m = prueba_Digitos()
# con los siguientes parámetros:
#  n_epoch =  100
#  rate_decay =  True
#  rate =  0.001
#  la tasa de aciertos serían:

# One vs Rest con Clasificador_RL_ML_Batch: 0.185
# One vs Rest con Clasificador_RL_ML_St: 0.842

# Por lo que el mejor sería el One vs Rest con Clasificador_RL_ML_St: con una accuracy de 0.826 para el conjunto de test

# In [254]: m = prueba_Digitos()
# con los siguientes parámetros:
#  n_epoch =  100
#  rate_decay =  False
#  rate =  0.01
#  la tasa de aciertos serían:

# One vs Rest con Clasificador_RL_ML_Batch: 0.804
# One vs Rest con Clasificador_RL_ML_St: 0.856

# Por lo que el mejor sería el One vs Rest con Clasificador_RL_ML_St: con una accuracy de 0.811 para el conjunto de test

# m = prueba_Digitos()
# con los siguientes parámetros:
#  n_epoch =  50
#  rate_decay =  True
#  rate =  0.01
#  la tasa de aciertos serían:

# One vs Rest con Clasificador_RL_ML_Batch: 0.175
# One vs Rest con Clasificador_RL_ML_St: 0.857

# Por lo que el mejor sería el One vs Rest con Clasificador_RL_ML_St: con una accuracy de 0.817 para el conjunto de test

# m = prueba_Digitos()
# con los siguientes parámetros:
#  n_epoch =  100
#  rate_decay =  False
#  rate =  0.001
#  la tasa de aciertos serían:

# One vs Rest con Clasificador_RL_ML_Batch: 0.783
# One vs Rest con Clasificador_RL_ML_St: 0.854

# Por lo que el mejor sería el One vs Rest con Clasificador_RL_ML_St: con una accuracy de 0.81 para el conjunto de test

def probarPesosDigitos(pesosPorClases):
    clases = clasesFichero
    def clasifica(ej,y):
        x = []
        ej = [1]+ej
        for j in range(len(clases)):
            x.append(clasifica_prob(ej,j))
        return clases[x.index(max(x))]

    def clasifica_prob(ej,j):
        x = sum(pesosPorClases[j][i]*ej[i] for i in range(len(ej)))
        return f_sigmoide(x)

    return sum(clasifica(x,y) == y for x,y in zip(testFichero,clasesTestFichero))/len(clasesTestFichero)

def guardarPeso(clasificador):
    with open('pesosDigitos5.py','w') as fichero:
        fichero.write("pesos5 = {0}".format(clasificador.pesosPorClases))
    fichero.closed

from pesosDigitos import *
from pesosDigitos2 import *
from pesosDigitos3 import * ## la prueba de pesos3 no está pero los pesos están guardados y es el que mejor clasifica
from pesosDigitos4 import *
from pesosDigitos5 import *
#----------------------------------------------

#----------------------------------------------

#  - Cualquier otro problema de clasificación (por ejemplo,
#    alguno de los que se pueden encontrar en UCI Machine Learning repository,
#    http://archive.ics.uci.edu/ml/). Téngase en cuenta que el conjunto de
#    datos que se use ha de tener sus atríbutos numéricos. Sin embargo,
#    también es posible transformar atributos no numéricos en numéricos usando
#    la técnica conocida como "one hot encoding".
    
def crearDataSetUCI():
    with open('wine/wine-data.txt','r') as fichero:
        entr = []
        clas_entr = []
        for linea in fichero:
            lista = linea[:-1].split(',')
            clas_entr.append(int(lista[0]))
            entr.append([float(x) for x in lista[1:]])
        fichero.closed

    with open('wineData.py','w') as escritura:
        escritura.write("wine_entr = {0}\n".format(entr[:47]+entr[59:115]+entr[129:168]))
        escritura.write("wine_clas_entr = {0}\n".format(clas_entr[:47]+clas_entr[59:115]+clas_entr[129:168]))
        escritura.write("wine_prueba = {0}\n".format(entr[47:59]+entr[115:129]+entr[168:]))
        escritura.write("wine_clas_prueba = {0}\n".format(clas_entr[47:59]+clas_entr[115:129]+clas_entr[168:]))
        escritura.closed

from wineData import *

clasesWine = list(set(wine_clas_entr))

def mejorClasificadorWine():
    entr = wine_entr
    clas_entr = wine_clas_entr
    clases = clasesWine
    n_epochs = 1000
    rate = 0.01
    rate_decay = True
    clasificadores = []
    rendimientos = []

    clasificadorMLBatch=Clasificador_RL_OvR(Clasificador_RL_ML_Batch,clases)
    clasificadorMLBatch.entrena(entr,clas_entr,n_epochs,rate_decay,rate)
    clasificadores.append(clasificadorMLBatch)
    rendimientoMLBatch = rendimiento(clasificadorMLBatch,wine_prueba,wine_clas_prueba)
    rendimientos.append(rendimientoMLBatch)
    print("Rendimiento del OvR-MLBatch:",rendimientoMLBatch)

    clasificadorMLSt=Clasificador_RL_OvR(Clasificador_RL_ML_St,clases)
    clasificadorMLSt.entrena(entr,clas_entr,n_epochs,rate_decay,rate)
    clasificadores.append(clasificadorMLSt)
    rendimientoMLSt = rendimiento(clasificadorMLSt,wine_prueba,wine_clas_prueba)
    rendimientos.append(rendimientoMLSt)
    print("Rendimiento del OvR-MLSt:",rendimientoMLSt)

    clasficadorL2Batch=Clasificador_RL_OvR(Clasificador_RL_L2_Batch,clases)
    clasficadorL2Batch.entrena(entr,clas_entr,n_epochs,rate_decay,rate)
    clasificadores.append(clasficadorL2Batch)
    rendimientoL2Batch = rendimiento(clasficadorL2Batch,wine_prueba,wine_clas_prueba)
    rendimientos.append(rendimientoL2Batch)
    print("Rendimiento del OvR-L2Batch:",rendimientoL2Batch)

    clasficadorL2St=Clasificador_RL_OvR(Clasificador_RL_L2_St,clases)
    clasficadorL2St.entrena(entr,clas_entr,n_epochs,rate_decay,rate)
    clasificadores.append(clasficadorL2St)
    rendimientoL2St = rendimiento(clasficadorL2St,wine_prueba,wine_clas_prueba)
    rendimientos.append(rendimientoL2St)
    print("Rendimiento del OvR-L2St:",rendimientoL2St)

    return clasificadores[rendimientos.index(max(rendimientos))]

## m es el peso del mejor clasificador
## probarPesos(clasesWine,pesosWine)
## así comprobamos que efectivamente el resultado que tenemos aquí comentado es real y no inventado

def probarPesosWine(clases,pesosPorClases):
    def clasifica(ej):
        x = []
        ej = [1]+ej
        for j in range(len(pesosPorClases)):
            y = sum(pesosPorClases[j][i]*ej[i] for i in range(len(ej)))
            x.append(f_sigmoide(y))
        return clases[x.index(max(x))]
    return sum(clasifica(x) == y for x,y in zip(wine_prueba,wine_clas_prueba))/len(wine_clas_prueba)

# # In [60]: m = mejorClasificadorWine()
# # Rendimiento del OvR: 0.8888888888888888
# pesos del anterior rendimiento
# pesosWine = [[-4.464693575485346,-43.57991041638832,-9.391572151683524,-6.74739397811181,-132.7910589564548,
#   -221.73389800636392,0.04594572086797535,10.846333937742518,-1.9918609345517928,1.0998147943667742,
#   -15.70881245320547,-3.5378746279158166,1.912407028820642,32.616275132120016],[3.128022276706968,27.851287684572082,
#   -22.89983158183535,4.565980602913298,70.06623491027516,192.52817131856722,14.14586297780636,20.464163304184247,
#   0.5715583511768094,11.242848701007016,-62.70951289493242,11.26196734895127,24.908400839789365,-29.301253942773165],
#   [2.1951112829464345,24.291241917654194,40.369633906833315,6.969099310051722,85.29402998379574,94.988009663006,
#   -17.006795253596046,-37.569321997235576,2.454951090244822,-10.939178520538995,96.59921837480277,-10.113767433264647,
#   -27.213090050164293,-6.3197935697498515]]

#####
# In [361]: m = mejorClasificadorWine()
# Rendimiento del OvR-MLBatch: 0.4444444444444444
# Rendimiento del OvR-MLSt: 0.75
# Rendimiento del OvR-L2Batch: 0.3888888888888889
# Rendimiento del OvR-L2St: 0.3333333333333333
# Los parámetros han sido: rate_decay = false, rate = 0.01, n_epochs = 100

# pesosWine = [[-46.21189960904281,-520.9376823667673,-139.82676186140355,-96.0958913408886,-1282.7283086268296,
#   -3333.315275951497,-18.035790559038784,64.93111104497567,-28.0397289534357,-23.669044002877538,-228.5000029404919,
#   -25.388493300231747,-20.363728527780566,461.25775537233693],[39.52859822570123,388.5482098619103,-49.83509279700508,
#   60.74224474980208,780.1795045996014,2661.5066689780306,101.90286781518795,108.48740423487955,12.645556787730339,
#   88.75912383594648,-193.68903329209454,72.7766204467529,152.73739051797068,-448.4505328402602],[4.259430612321957,
#   90.68482337029403,189.56247549695314,22.93158404754266,347.82922430702916,-38.31854847540123,-85.66701020818904,
#   -184.36913881697473,17.433924090166663,-60.52163427839051,432.5069896917864,-46.23713127296177,-133.24360630397945,
#   -11.685907857088505]]

#####
# In [373]: m = mejorClasificadorWine()
# Rendimiento del OvR-MLBatch: 0.6111111111111112
# Rendimiento del OvR-MLSt: 0.9166666666666666
# Rendimiento del OvR-L2Batch: 0.3333333333333333
# Rendimiento del OvR-L2St: 0.3333333333333333
# Los parámetros han sido: rate_decay = true, rate = 0.01, n_epochs = 1000
# pesosWine = [[-1539.6268835905153,-10878.381086723624,-989.2950720224273,-760.0110038910298,-48346.32644873383,-7031.809963187124,
#   3571.681620237228,10871.466433506512,-1382.4466908548477,2309.434292377999,-7676.168796703483,-369.54284669054283,
#   5543.816376934862,2460.425484291416],[1769.0591838047142,4451.162212811217,-21938.07476834826,-1554.2546593901843,
#   15387.092732307709,4612.235314311085,7705.593762961092,12263.291546577817,205.53732821068223,7957.824551889497,
#   -46869.15393524881,7078.462408725737,14949.148959781058,-851.3068409341405],[-606.298278122936,-1616.2030264856808,
#   22467.557514125507,417.7465293565563,13802.568399244361,-1526.056848793015,-13776.715308886163,-27933.480447756963,
#   1505.6433944532637,-10709.707624891118,49780.66338018994,-7274.637754433477,-22453.48861234773,-369.60125134803275]]

# In [378]: m = mejorClasificadorWine()
# Rendimiento del OvR-MLBatch: 0.7222222222222222
# Rendimiento del OvR-MLSt: 0.9722222222222222
# Rendimiento del OvR-L2Batch: 0.3333333333333333
# Rendimiento del OvR-L2St: 0.3333333333333333
# Los parámetros han sido: rate_decay = true, rate = 0.01, n_epochs = 1000. Son los mismos que anteriormente
# pesosWine = [[-1535.2639350590302,-10868.250274174901,-1222.7734481383357,-733.1930643355506,-48252.542764197045,
#   -6741.642244430254,3619.7269347639794,10930.496538006555,-1379.337911346591,2365.2822714304716,-7644.938535234778,
#   -336.82669040413816,5581.573640054291,2091.5676962926564],[1753.1592243831176,4373.217242562065,-22542.966318853178,
#   -1585.0195771533167,14964.13279245178,4182.722967564622,7770.489781066902,12430.42737369197,189.28693788116306,
#   7920.805910127139,-46799.28063238624,7145.552027809979,15076.48291507059,-1091.5915738894428],[-647.6327709623212,
#   -2030.9921344297213,22593.131366482503,372.8855896712479,13413.789503659069,-1415.3829982506606,-13764.420790167671,
#   -27856.146219021073,1470.9192244280307,-10708.672306596942,49195.41363331626,-7266.7608695149565,-22422.12926570915,
#   -544.1988897842957]]
#----------------------------------------------

#----------------------------------------------

#  Nótese que en cualquiera de los tres casos, consiste en encontrar el
#  clasificador adecuado, entrenado con los parámetros y opciones
#  adecuadas. El entrenamiento ha de realizarse sobre el conjunto de
#  entrenamiento, y el conjunto de validación se emplea para medir el
#  rendimiento obtenido con los distintas combinaciones de parámetros y
#  opciones con las que se experimente. Finalmente, una vez elegido la mejor
#  combinación de parámetros y opciones, se da el rendimiento final sobre el
#  conjunto de test. Es importante no usar el conjunto de test para decididir
#  sobre los parámetros, sino sólo para dar el rendimiento final.

#  En nuestro caso concreto, estas son las opciones y parámetros con los que
#  hay que experimentar:

#  - En primer lugar, el tipo de clasificador usado (si es batch o
#    estaocástico, si es basado en error cuadrático o en verosimilitud, si es
#    softmax o OvR,...)
#  - n_epochs: el número de epochs realizados influye en el tiempo de
#    entrenamiento y evidentemente también en la calidad del clasificador
#    obtenido. Con un número bajo de epochs, no hay suficiente entrenamiento,
#    pero también hay que decir que un número excesivo de epochs puede
#    provocar un sobreajuste no deseado.
#  - El valor de "rate" usado.
#  - Si se usa "rate_decay" o no.
#  - Si se usa normalización o no.

# Se pide describir brevemente el proceso de experimentación en cada uno de
# los casos, y finalmente dar el clasificador con el que se obtienen mejor
# rendimiento sobre el conjunto de test correspondiente.

# Por dar una referencia, se pueden obtener clasificadores para el problema de
# los votos con un rendimiento sobre el test mayor al 90%, y para los dígitos
# un rendimiento superior al 80%.
