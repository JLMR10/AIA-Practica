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

# - Perceptron umbral
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

## Clasificador del Perceptron

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

## Clasificador de Regresion Lineal Bach  minimizando L2

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

## Clasificador Regresion Lineal St minimizando L2

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

##Clasificador Regresion Lineal Bach maximizando verosimilitud

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

##Clasificador Regresion Lineal St maximizando verosimilitud

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
            print(i)
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
    return sum(Clasificador_RL_OvR.clasifica(clasificador,x) == y for x,y in zip(entr,clas_entr))/len(clas_entr)

###=================================================================================================
##TESTS
###=================================================================================================

#----------------------------------------------
###Pruebas Basicas Clasificadores###

def pruebaSeparableConGraficas():

    X1,Y1=genera_conjunto_de_datos_l_s(4,8,400)
    X1e,Y1e=X1[:300],Y1[:300]
    X1t,Y1t=X1[300:],Y1[300:]

    #Perceptron
    print("Perceptron")
    clas_pb1=Clasificador_Perceptron([0,1])
    ac = clas_pb1.entrena(X1e,Y1e,100,rate_decay=True,rate=0.001)
    print("Accuracy Perceptrón:",sum(clas_pb1.clasifica(x) == y for x,y in zip(X1t,Y1t))/len(Y1t))
    plt.plot(range(1,len(ac)+1),ac,marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Porcentaje de acierto')
    plt.show()

    #Regresion Lineal Bach minimizando L2
    print("-----------------------------")
    print("Clasificador_RL_L2_Batch")
    clas_pb2=Clasificador_RL_L2_Batch([0,1])
    ac2,error2 = clas_pb2.entrena(X1e,Y1e,100,rate_decay=True,rate=0.001)
    print("Clasifica_prob de Clasificador_RL_L2_Batch:", clas_pb2.clasifica_prob(X1t[0]),Y1t[0])
    print("Accuracy Clasificador_RL_L2_Batch:",sum(clas_pb2.clasifica(x) == y for x,y in zip(X1t,Y1t))/len(Y1t))
    plt.plot(range(1,len(ac2)+1),ac2,marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Porcentaje de acierto')
    plt.show()
    plt.plot(range(1,len(error2)+1),error2,marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Error Cuadrático')
    plt.show()

    #Regresion Lineal St minimizando L2
    print("-----------------------------")
    print("Clasificador_RL_L2_St")
    clas_pb3=Clasificador_RL_L2_St([0,1])
    ac3,error3=clas_pb3.entrena(X1e,Y1e,100,rate_decay=True,rate=0.001)
    print("Clasifica_prob de Clasificador_RL_L2_St:", clas_pb3.clasifica_prob(X1t[0]),Y1t[0])
    print("Accuracy Clasificador_RL_L2_St:",sum(clas_pb3.clasifica(x) == y for x,y in zip(X1t,Y1t))/len(Y1t))
    plt.plot(range(1,len(ac3)+1),ac3,marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Porcentaje de acierto')
    plt.show()
    plt.plot(range(1,len(error3)+1),error3,marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Error Cuadrático')
    plt.show()

    #Regresion Lineal Bach maximizando verosimilitud
    print("-----------------------------")
    print("Clasificador_RL_ML_Batch")
    clas_pb4=Clasificador_RL_ML_Batch([0,1])
    ac4,error4=clas_pb4.entrena(X1e,Y1e,100,rate_decay=True,rate=0.001)
    print("Clasifica_prob de Clasificador_RL_ML_Batch:", clas_pb4.clasifica_prob(X1t[0]),Y1t[0])
    print("Accuracy Clasificador_RL_ML_Batch:",sum(clas_pb4.clasifica(x) == y for x,y in zip(X1t,Y1t))/len(Y1t))
    plt.plot(range(1,len(ac4)+1),ac4,marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Porcentaje de acierto')
    plt.show()
    plt.plot(range(1,len(error4)+1),error4,marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Log verosimilitud')
    plt.show()


    #Regresion Lineal St maximizando verosimilitud
    print("-----------------------------")
    print("Clasificador_RL_ML_St")
    clas_pb5=Clasificador_RL_ML_St([0,1])
    ac5,error5=clas_pb5.entrena(X1e,Y1e,100,rate_decay=True,rate=0.001)
    print("Clasifica_prob de Clasificador_RL_ML_St:", clas_pb5.clasifica_prob(X1t[0]),Y1t[0])
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

    #Perceptron
    print("Perceptron")
    clas_pb1=Clasificador_Perceptron([0,1])
    clas_pb1.entrena(X1e,Y1e,100,rate_decay=True,rate=0.001)
    print("Accuracy Perceptrón:",sum(clas_pb1.clasifica(x) == y for x,y in zip(X1t,Y1t))/len(Y1t))

    #Regresion Lineal Bach minimizando L2
    print("-----------------------------")
    print("Clasificador_RL_L2_Batch")
    clas_pb2=Clasificador_RL_L2_Batch([0,1])
    clas_pb2.entrena(X1e,Y1e,100,rate_decay=True,rate=0.001)
    print("Clasifica_prob de Clasificador_RL_L2_Batch:", clas_pb2.clasifica_prob(X1t[0]),Y1t[0])
    print("Accuracy Clasificador_RL_L2_Batch:",sum(clas_pb2.clasifica(x) == y for x,y in zip(X1t,Y1t))/len(Y1t))

    #Regresion Lineal St minimizando L2
    print("-----------------------------")
    print("Clasificador_RL_L2_St")
    clas_pb3=Clasificador_RL_L2_St([0,1])
    clas_pb3.entrena(X1e,Y1e,100,rate_decay=True,rate=0.001)
    print("Clasifica_prob de Clasificador_RL_L2_St:", clas_pb3.clasifica_prob(X1t[0]),Y1t[0])
    print("Accuracy Clasificador_RL_L2_St:",sum(clas_pb3.clasifica(x) == y for x,y in zip(X1t,Y1t))/len(Y1t))

    #Regresion Lineal Bach maximizando verosimilitud
    print("-----------------------------")
    print("Clasificador_RL_ML_Batch")
    clas_pb4=Clasificador_RL_ML_Batch([0,1])
    clas_pb4.entrena(X1e,Y1e,100,rate_decay=True,rate=0.001)
    print("Clasifica_prob de Clasificador_RL_ML_Batch:", clas_pb4.clasifica_prob(X1t[0]),Y1t[0])
    print("Accuracy Clasificador_RL_ML_Batch:",sum(clas_pb4.clasifica(x) == y for x,y in zip(X1t,Y1t))/len(Y1t))

    #Regresion Lineal St maximizando verosimilitud
    print("-----------------------------")
    print("Clasificador_RL_ML_St")
    clas_pb5=Clasificador_RL_ML_St([0,1])
    clas_pb5.entrena(X1e,Y1e,100,rate_decay=True,rate=0.001)
    print("Clasifica_prob de Clasificador_RL_ML_St:", clas_pb5.clasifica_prob(X1t[0]),Y1t[0])
    print("Accuracy Clasificador_RL_ML_St:",sum(clas_pb5.clasifica(x) == y for x,y in zip(X1t,Y1t))/len(Y1t))

def pruebaNoSeparableConGraficas():

    X1,Y1=genera_conjunto_de_datos_n_l_s(4,8,400)
    X1e,Y1e=X1[:300],Y1[:300]
    X1t,Y1t=X1[300:],Y1[300:]

    #Perceptron
    print("Perceptron")
    clas_pb1=Clasificador_Perceptron([0,1])
    ac = clas_pb1.entrena(X1e,Y1e,100,rate_decay=True,rate=0.001)
    print("Accuracy Perceptrón:",sum(clas_pb1.clasifica(x) == y for x,y in zip(X1t,Y1t))/len(Y1t))
    plt.plot(range(1,len(ac)+1),ac,marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Porcentaje de acierto')
    plt.show()

    #Regresion Lineal Bach minimizando L2
    print("-----------------------------")
    print("Clasificador_RL_L2_Batch")
    clas_pb2=Clasificador_RL_L2_Batch([0,1])
    ac2,error2 = clas_pb2.entrena(X1e,Y1e,100,rate_decay=True,rate=0.001)
    print("Clasifica_prob de Clasificador_RL_L2_Batch:", clas_pb2.clasifica_prob(X1t[0]),Y1t[0])
    print("Accuracy Clasificador_RL_L2_Batch:",sum(clas_pb2.clasifica(x) == y for x,y in zip(X1t,Y1t))/len(Y1t))
    plt.plot(range(1,len(ac2)+1),ac2,marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Porcentaje de acierto')
    plt.show()
    plt.plot(range(1,len(error2)+1),error2,marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Error Cuadrático')
    plt.show()

    #Regresion Lineal St minimizando L2
    print("-----------------------------")
    print("Clasificador_RL_L2_St")
    clas_pb3=Clasificador_RL_L2_St([0,1])
    ac3,error3=clas_pb3.entrena(X1e,Y1e,100,rate_decay=True,rate=0.001)
    print("Clasifica_prob de Clasificador_RL_L2_St:", clas_pb3.clasifica_prob(X1t[0]),Y1t[0])
    print("Accuracy Clasificador_RL_L2_St:",sum(clas_pb3.clasifica(x) == y for x,y in zip(X1t,Y1t))/len(Y1t))
    plt.plot(range(1,len(ac3)+1),ac3,marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Porcentaje de acierto')
    plt.show()
    plt.plot(range(1,len(error3)+1),error3,marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Error Cuadrático')
    plt.show()

    #Regresion Lineal Bach maximizando verosimilitud
    print("-----------------------------")
    print("Clasificador_RL_ML_Batch")
    clas_pb4=Clasificador_RL_ML_Batch([0,1])
    ac4,error4=clas_pb4.entrena(X1e,Y1e,100,rate_decay=True,rate=0.001)
    print("Clasifica_prob de Clasificador_RL_ML_Batch:", clas_pb4.clasifica_prob(X1t[0]),Y1t[0])
    print("Accuracy Clasificador_RL_ML_Batch:",sum(clas_pb4.clasifica(x) == y for x,y in zip(X1t,Y1t))/len(Y1t))
    plt.plot(range(1,len(ac4)+1),ac4,marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Porcentaje de acierto')
    plt.show()
    plt.plot(range(1,len(error4)+1),error4,marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Log verosimilitud')
    plt.show()


    #Regresion Lineal St maximizando verosimilitud
    print("-----------------------------")
    print("Clasificador_RL_ML_St")
    clas_pb5=Clasificador_RL_ML_St([0,1])
    ac5,error5=clas_pb5.entrena(X1e,Y1e,100,rate_decay=True,rate=0.001)
    print("Clasifica_prob de Clasificador_RL_ML_St:", clas_pb5.clasifica_prob(X1t[0]),Y1t[0])
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

    #Perceptron
    print("Perceptron")
    clas_pb1=Clasificador_Perceptron([0,1])
    clas_pb1.entrena(X1e,Y1e,100,rate_decay=True,rate=0.001)
    print("Accuracy Perceptrón:",sum(clas_pb1.clasifica(x) == y for x,y in zip(X1t,Y1t))/len(Y1t))

    #Regresion Lineal Bach minimizando L2
    print("-----------------------------")
    print("Clasificador_RL_L2_Batch")
    clas_pb2=Clasificador_RL_L2_Batch([0,1])
    clas_pb2.entrena(X1e,Y1e,100,rate_decay=True,rate=0.001)
    print("Clasifica_prob de Clasificador_RL_L2_Batch:", clas_pb2.clasifica_prob(X1t[0]),Y1t[0])
    print("Accuracy Clasificador_RL_L2_Batch:",sum(clas_pb2.clasifica(x) == y for x,y in zip(X1t,Y1t))/len(Y1t))

    #Regresion Lineal St minimizando L2
    print("-----------------------------")
    print("Clasificador_RL_L2_St")
    clas_pb3=Clasificador_RL_L2_St([0,1])
    clas_pb3.entrena(X1e,Y1e,100,rate_decay=True,rate=0.001)
    print("Clasifica_prob de Clasificador_RL_L2_St:", clas_pb3.clasifica_prob(X1t[0]),Y1t[0])
    print("Accuracy Clasificador_RL_L2_St:",sum(clas_pb3.clasifica(x) == y for x,y in zip(X1t,Y1t))/len(Y1t))

    #Regresion Lineal Bach maximizando verosimilitud
    print("-----------------------------")
    print("Clasificador_RL_ML_Batch")
    clas_pb4=Clasificador_RL_ML_Batch([0,1])
    clas_pb4.entrena(X1e,Y1e,100,rate_decay=True,rate=0.001)
    print("Clasifica_prob de Clasificador_RL_ML_Batch:", clas_pb4.clasifica_prob(X1t[0]),Y1t[0])
    print("Accuracy Clasificador_RL_ML_Batch:",sum(clas_pb4.clasifica(x) == y for x,y in zip(X1t,Y1t))/len(Y1t))

    #Regresion Lineal St maximizando verosimilitud
    print("-----------------------------")
    print("Clasificador_RL_ML_St")
    clas_pb5=Clasificador_RL_ML_St([0,1])
    clas_pb5.entrena(X1e,Y1e,100,rate_decay=True,rate=0.001)
    print("Clasifica_prob de Clasificador_RL_ML_St:", clas_pb5.clasifica_prob(X1t[0]),Y1t[0])
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

    return clasificadores[rendimientos.index(max(rendimientos))]

#----------------------------------------------
###Pruebas SoftMax

def prueba_Softmax_Iris():
    iris_clases=["Iris-setosa","Iris-virginica","Iris-versicolor"]
    clas_rlml1=Clasificador_RL_Softmax(iris_clases)
    clas_rlml1.entrena(iris_entr,iris_entr_clas,100,rate_decay=True,rate=0.01)
    print(rendimiento(clas_rlml1,iris_entr,iris_entr_clas))

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

    epoch = 100
    decay=False
    valor=0.001

    misMetodos = []
    metodo = []
    acuracy = []

    ## Perceptron
    Perceptron = Clasificador_Perceptron(clases)
    Perceptron.entrena(entrenamiento,clases_entrenamiento,epoch,rate_decay= decay, rate=valor)
    acPerceptron = rendimiento(Perceptron,validacion,clases_validacion)
    metodo.append("Perceptron:")
    acuracy.append(acPerceptron)
    misMetodos.append(Perceptron)

    ##Clasificador_RL_L2_Batch
    L2_Bach = Clasificador_RL_L2_Batch(clases)
    L2_Bach.entrena(entrenamiento,clases_entrenamiento,epoch,rate_decay=decay,rate=valor)
    acL2_bach = rendimiento(L2_Bach,validacion,clases_validacion)
    metodo.append("Clasificador_RL_L2_Batch:")
    acuracy.append(acL2_bach)
    misMetodos.append(L2_Bach)

    ##Clasificador_RL_L2_St
    L2_St = Clasificador_RL_L2_St(clases)
    L2_St.entrena(entrenamiento,clases_entrenamiento,epoch,rate_decay=decay,rate=valor)
    acL2_St = rendimiento(L2_St,validacion,clases_validacion)
    metodo.append("Clasificador_RL_L2_St:")
    acuracy.append(acL2_St)
    misMetodos.append(L2_St)

    ##Clasificador_RL_ML_Batch
    ML_Bach = Clasificador_RL_ML_Batch(clases)
    ML_Bach.entrena(entrenamiento,clases_entrenamiento,epoch,rate_decay=decay,rate=valor)
    acML_Bach = rendimiento(ML_Bach,validacion,clases_validacion)
    metodo.append("Clasificador_RL_ML_Batch:")
    acuracy.append(acML_Bach)
    misMetodos.append(ML_Bach)

    ##Clasificador_RL_ML_St
    ML_St = Clasificador_RL_ML_St(clases)
    ML_St.entrena(entrenamiento,clases_entrenamiento,epoch,rate_decay=decay,rate=valor)
    acML_St = rendimiento(ML_St,validacion,clases_validacion)
    metodo.append("Clasificador_RL_ML_St:")
    acuracy.append(acML_St)
    misMetodos.append(ML_St)

    ##One vs Rest L2_Batch
    one_restL2 = Clasificador_RL_OvR(Clasificador_RL_L2_Batch,clases)
    one_restL2.entrena(entrenamiento,clases_entrenamiento,epoch,rate_decay=decay,rate=valor)
    acOneRestL2 = rendimiento(one_restL2,validacion,clases_validacion)
    metodo.append("One vs Rest con Clasificador_RL_L2_Batch:")
    acuracy.append(acOneRestL2)
    misMetodos.append(one_restL2)

    print("con los siguietes parametros:\n n_epoch = ",epoch,"\n rate_decay = ",decay,
    "\n rate = ",valor,"\n la tasa de aciertos serian: \n")

    for i in range(len(metodo)):
        print(metodo[i],acuracy[i])

    mejor = max(acuracy)
    indice = acuracy.index(mejor)

    print("\nPor lo que el mejor seria el",metodo[indice], "con una acuracy de",acuracy[indice])

    return misMetodos[indice]

def clasificaMejor(m):
    print(m)
    print("validacion: ",rendimiento(m,votos_valid,votos_valid_clas))
    print(rendimiento(m,votos_test,votos_test_clas))

# ######################################################################################################### primera prueba

# # In [83]: mejorClasificadorVotos()
# # con los siguietes parametros:
# #  n_epoch =  50
# #  rate_decay =  True
# #  rate =  0.001
# #  la tasa de aciertos serian:

# # Perceptron: 0.9710144927536232
# # Clasificador_RL_L2_Batch: 0.9565217391304348
# # Clasificador_RL_L2_St: 0.9855072463768116
# # Clasificador_RL_ML_Batch: 0.9565217391304348
# # Clasificador_RL_ML_St: 0.9855072463768116

# # Por lo que el mejor seria el Clasificador_RL_L2_St: con una acuracy de 0.9855072463768116

# ######################################################################################################### otra prueba con los mismos parametros:

# # In [84]: mejorClasificadorVotos()
# # con los siguietes parametros:
# #  n_epoch =  50
# #  rate_decay =  True
# #  rate =  0.001
# #  la tasa de aciertos serian:

# # Perceptron: 0.9855072463768116
# # Clasificador_RL_L2_Batch: 0.9420289855072463
# # Clasificador_RL_L2_St: 0.9855072463768116
# # Clasificador_RL_ML_Batch: 0.9710144927536232
# # Clasificador_RL_ML_St: 0.9710144927536232



# # Por lo que el mejor seria el Perceptron: con una acuracy de 0.9855072463768116

# ######################################################################################################### otros parametros
# # In [96]: m = mejorClasificadorVotos()
# # con los siguietes parametros:
# #  n_epoch =  100
# #  rate_decay =  False
# #  rate =  0.001
# #  la tasa de aciertos serian:

# # Perceptron: 0.9565217391304348
# # Clasificador_RL_L2_Batch: 0.7971014492753623
# # Clasificador_RL_L2_St: 0.927536231884058
# # Clasificador_RL_ML_Batch: 1.0
# # Clasificador_RL_ML_St: 0.9565217391304348

# # Por lo que el mejor seria el Clasificador_RL_ML_Batch: con una acuracy de 1.0

# # In [97]: m.pesos
# # Out[97]:
# # [0.4229950881833548,
# #  0.27317538844627465,
# #  0.711027952150914,
# #  0.6994496767958582,
# #  -2.5717907044593353,
# #  -0.7196720449497429,
# #  0.1325039118550952,
# #  -0.9303798825074252,
# #  -0.1931287690701619,
# #  0.9872705683149539,
# #  -0.3206004413092406,
# #  0.18064387353039402,
# #  -0.050354214937230454,
# #  -0.5078123758537897,
# #  0.1271250545684751,
# #  -0.21250584478649107,
# #  0.443988026989932]

# # In [98]: clasificaMejor(m)
# # <__main__.Clasificador_RL_ML_Batch object at 0x0000029BD2F40400>
# # validacion:  1.0
# # rendimiento del conjunto de prueba -> 0.896551724137931

# ######################################################################################################### otros parametros

# # In [122]: m = mejorClasificadorVotos()
# # con los siguietes parametros:
# #  n_epoch =  100
# #  rate_decay =  True
# #  rate =  0.001
# #  la tasa de aciertos serian:

# # Perceptron: 0.9855072463768116
# # Clasificador_RL_L2_Batch: 0.9565217391304348
# # Clasificador_RL_L2_St: 0.9855072463768116
# # Clasificador_RL_ML_Batch: 0.9565217391304348
# # Clasificador_RL_ML_St: 0.9565217391304348

# # Por lo que el mejor seria el Perceptron: con una acuracy de 0.9855072463768116

# # In [123]: m.pesos
# # Out[123]:
# # [3.4671723097946203,
# #  0.7464109670914848,
# #  3.276934471391926,
# #  5.901662769684045,
# #  -16.477172309794625,
# #  2.242677540873024,
# #  5.709891103246117,
# #  -6.591827690205382,
# #  1.753322459126976,
# #  5.204156721980204,
# #  -2.2466775408730255,
# #  3.8496627696840537,
# #  -4.13029305517104,
# #  4.6298718653502915,
# #  -2.6870794927979476,
# #  3.252777422460347,
# #  2.5885619234229367]

# # In [124]: clasificaMejor(m)
# # <__main__.Clasificador_Perceptron object at 0x0000029BD2FF74A8>
# # validacion:  0.9855072463768116
# # rendimiento del conjunto de prueba -> 0.931034482758620


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

def leer():
    with open('digitdata/trainingimages','r') as fichero:
        total = []
        v = []
        i=0
        for linea in fichero:
            w = []
            i+=1
            for posicion in linea[:-1]:
                if " " == posicion:
                    w.append(0)
                else:
                    w.append(1)
            v.append(w)
            if(i==28):
                i=0
                total.append(v)
                v = []
        fichero.closed
    return total


entrenaFichero = []
def leerEntrena():
    with open('digitdata/trainingimages','r') as fichero:
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
entrenaFichero = leerEntrena()

clasesEntrenaFichero = []
def leerClasesEntrena():
    v = []
    with open('digitdata/traininglabels','r') as fichero:
        for linea in fichero:
            v.append(int(linea[0]))
        fichero.closed
    return v
clasesEntrenaFichero = leerClasesEntrena()

####

validaFichero = []
def leerValida():
    with open('digitdata/validationimages','r') as fichero:
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
validaFichero = leerValida()

clasesValidaFichero = []
def leerClasesValida():
    v = []
    with open('digitdata/validationlabels','r') as fichero:
        for linea in fichero:
            v.append(int(linea[0]))
        fichero.closed
    return v
clasesValidaFichero = leerClasesValida()

######

testFichero = []
def leerTest():
    with open('digitdata/testimages','r') as fichero:
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
testFichero = leerTest()

clasesTestFichero = []
def leerClasesTest():
    v = []
    with open('digitdata/testlabels','r') as fichero:
        for linea in fichero:
            v.append(int(linea[0]))
        fichero.closed
    return v
clasesTestFichero = leerClasesTest()

########

clasesFichero = []
def clasesParaFichero():
    res = []
    for i in clasesEntrenaFichero:
        if not i in res:
            res.append(i)
    return res
clasesFichero = clasesParaFichero()

######

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

    print("con los siguietes parametros:\n n_epoch = ",epoch,"\n rate_decay = ",decay,
    "\n rate = ",valor,"\n la tasa de aciertos serian: \n")

    for i in range(len(metodo)):
        print(metodo[i],accuracy[i])

    mejor = max(accuracy)
    indice = accuracy.index(mejor)

    print("\nPor lo que el mejor seria el",metodo[indice], "con una accuracy de",rendimiento(misClasificadores[indice],prueba,prueba_clas),"para el conjunto de test")

    return misClasificadores[indice]


# In [6]: m = prueba_Digitos()
# con los siguietes parametros:
#  n_epoch =  100
#  rate_decay =  True
#  rate =  0.001
#  la tasa de aciertos serian:

# One vs Rest con Clasificador_RL_ML_Batch: 0.185
# One vs Rest con Clasificador_RL_ML_St: 0.842

# Por lo que el mejor seria el One vs Rest con Clasificador_RL_ML_St: con una accuracy de 0.811 para el conjunto de test

# In [254]: m = prueba_Digitos()
# con los siguietes parametros:
#  n_epoch =  100
#  rate_decay =  False
#  rate =  0.01
#  la tasa de aciertos serian:

# One vs Rest con Clasificador_RL_ML_Batch: 0.804
# One vs Rest con Clasificador_RL_ML_St: 0.856

# Por lo que el mejor seria el One vs Rest con Clasificador_RL_ML_St: con una accuracy de 0.835 para el conjunto de test

# m = prueba_Digitos()
# con los siguietes parametros:
#  n_epoch =  50
#  rate_decay =  True
#  rate =  0.01
#  la tasa de aciertos serian:

# One vs Rest con Clasificador_RL_ML_Batch: 0.175
# One vs Rest con Clasificador_RL_ML_St: 0.857

# Por lo que el mejor seria el One vs Rest con Clasificador_RL_ML_St: con una accuracy de 0.817 para el conjunto de test

# m = prueba_Digitos()
# con los siguietes parametros:
#  n_epoch =  100
#  rate_decay =  False
#  rate =  0.001
#  la tasa de aciertos serian:

# One vs Rest con Clasificador_RL_ML_Batch: 0.783
# One vs Rest con Clasificador_RL_ML_St: 0.854

# Por lo que el mejor seria el One vs Rest con Clasificador_RL_ML_St: con una accuracy de 0.81 para el conjunto de test

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
from pesosDigitos3 import *
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

def mejorClasificadorWine():
    entr = wine_entr
    clas_entr = wine_clas_entr
    clases = list(set(clas_entr))
    n_epochs = 100
    rate = 0.01
    rate_decay = False
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
## probarPesos(list(set(wine_clas_entr)),m.pesosPorClases)
## así comprobamos que efectivamente el resultado que tenemos aquí comentado es real y no inventado
def probarPesosWine(clases,pesosPorClases):
    def clasifica(ej):
        print("??¿???")
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
# [[-4.464693575485346,
#   -43.57991041638832,
#   -9.391572151683524,
#   -6.74739397811181,
#   -132.7910589564548,
#   -221.73389800636392,
#   0.04594572086797535,
#   10.846333937742518,
#   -1.9918609345517928,
#   1.0998147943667742,
#   -15.70881245320547,
#   -3.5378746279158166,
#   1.912407028820642,
#   32.616275132120016],
#  [3.128022276706968,
#   27.851287684572082,
#   -22.89983158183535,
#   4.565980602913298,
#   70.06623491027516,
#   192.52817131856722,
#   14.14586297780636,
#   20.464163304184247,
#   0.5715583511768094,
#   11.242848701007016,
#   -62.70951289493242,
#   11.26196734895127,
#   24.908400839789365,
#   -29.301253942773165],
#  [2.1951112829464345,
#   24.291241917654194,
#   40.369633906833315,
#   6.969099310051722,
#   85.29402998379574,
#   94.988009663006,
#   -17.006795253596046,
#   -37.569321997235576,
#   2.454951090244822,
#   -10.939178520538995,
#   96.59921837480277,
#   -10.113767433264647,
#   -27.213090050164293,
#   -6.3197935697498515]]

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

# with open('digitdata/traininglabels','r') as fichero:
#     v = [int(l) for l in fichero]