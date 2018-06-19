## AIA
## Búsqueda con adversario
## Dpto. de C. de la Computación e I.A. (Univ. de Sevilla)
## ===================================================================

## En esta práctica vamos a implementar los algoritmos minimax y
## minimix con poda alfa beta para decidir el siguiente movimiento en
## un problema de búsqueda con adversario.

## ==================================================================
## Representación de problemas de búsqueda con adversario
## ==================================================================

## Recuérdese que según lo que se ha visto en clase, la implementación
## de la representación de un juego consiste en:

## * Representar estados y movimientos mediante alguna una estructura
##   de datos.
## * Definir: es_estado_final(_), es_estado_ganador(_,_,_),
##   movimientos(_), aplica(_,_), f_utilidad(_,_), y f_evaluacion(_,_)
## * Definir: estado_inicial, minimo_valor y maximo_valor

## ==================================================================
## Ejercicio 1
## ==================================================================

##   Definir en python una clase Juego que represente un problema de
## búsqueda con adversario. La clase debe tener los siguientes
## atributos:

## - estado_inicial: Estado inicial del juego.
## - estado_final: Estado final del juego (si es único).
## - maximo_valor: Cota superior de los valores de la función de
##   evaluación estática
## - minimo_valor: Cota inferior de los valores de la función de
##   evaluación estática

## y los siguientes métodos:

## - movimientos(estado): Lista de movimientos aplicables al 'estado'.
## - aplica(movimiento,estado): Estado resultado de aplicar el
##   'movimiento' al 'estado'.
## - es_estado_final(estado): Comprueba si el 'estado' es un estado
##   final del juego. Por defecto compara con el estado final.
## - es_estado_ganador(estado,turno,jugador): Comprueba si el
##   'jugador' gana el juego en el 'estado' cuando le toca al jugador
##   'turno'.
## - f_evaluacion(estado,turno): Devuelve el valor asociado al
##   'estado' cuando le toca jugar al jugador 'turno'. Por defecto
##   está definida como la función de utilidad para los estados
##   finales y 0 en caso cualquier otro caso.
## - str_estado(estado): Devuelve una repesentación en forma de cadena
##   de texto del 'estado'.
## - str_movimiento(movimiento): Devuelve una repesentación en forma
##   de cadena de texto del 'movimiento'.

##   El constructor de la clase recibe el estado inicial, el estado
## final, en caso de que éste sea único y los valores máximo y mínimo
## de la función de evaluación (por defecto, infinito y -infinito
## respectivamente).

class Juego:

    def __init__(self,estado_inicial,estado_final=None
                ,maximo_valor=float("inf"),minimo_valor=float("-inf")):

        self.estado_inicial = estado_inicial
        self.estado_final = estado_final
        self.maximo_valor = maximo_valor
        self.minimo_valor = minimo_valor

    def es_estado_final(self,estado):
        return estado==self.estado_final

    def movimientos(self,estado):
        pass
    
    def aplica(self,movimiento,estado):
        pass

    def es_estado_ganador(self,estado,turno,jugador):
        pass

    def f_evaluacion(self,estado,turno):
        if self.es_estado_ganador(estado,turno,"MAX"):
            return self.maximo_valor
        elif self.es_estado_ganador(estado,turno,"MIN"):
            return self.minimo_valor
        else: 
            return 0

    def str_estado(self,estado):
        return str(estado)

    def str_movimiento(self,movimiento):
        return str(movimiento)












## ==================================================================
## NIM
## ==================================================================

## Recordemos el juego del Nim visto en clase. Inicialmente se dispone
## de una pila de N fichas. En cada jugada, el jugador tiene que
## elegir 1, 2 ó 3 fichas. El jugador que coja la última pieza pierde.
        
## ==================================================================
## Ejercicio 2
## ==================================================================

##   Definir una función nim(n), que recibiendo como entrada un número
## natural n, devuelva la instancia de la clase Juego correspondiente
## al juego del Nim que inicia la partida con n piezas sobre la mesa.

##   Utilizar como función de evaluación estática la siguiente: Si el
## resto de dividir entre 4 el número de piezas del estado es igual a
## 1 entonces, si es el turno de 'MAX' devolver -1 y si es el turno de
## 'MIN', devolver 1. Si el resto de dividir entre 4 el número de
## piezas del estado es distinto de 1 entonces, si es el turno de
## 'MAX' devolver 1 y si es el turno de 'MIN', devolver -1.

## >>> juego_nim = nim(17)
## >>> juego_nim.estado_inicial
## 17
## >>> juego_nim.es_estado_final(3)
## False
## >>> juego_nim.movimientos(2)
## [2, 1]
## >>> juego_nim.movimientos(17)
## [3, 2, 1]
## >>> juego_nim.aplica(17, 3)
## 14
class Nim(Juego):
    def __init__(self,n):
        super().__init__(n,0,1,-1)
        self.movimientos_posibles=[1,2,3]

    def es_estado_ganador(self,estado,turno,jugador):
        return turno==jugador

    def movimientos(self,estado):
        return [m for m in self.movimientos_posibles if m<=estado]

    def aplica(self,movimiento,estado):
        return estado-movimiento

    def str_movimiento(self,mov):
        return "Quitar {}".format(mov) 

    def f_evaluacion(self,estado,turno):    ## turno=jugador
        if estado%4 == 1:
            if turno=="MAX":
                return self.minimo_valor
            else:
                return self.maximo_valor 
        else:
            if turno=="MAX":
                return self.maximo_valor
            else:
                return self.minimo_valor

def nim(n):
    return Nim(n)
    #return Juego(estado_inicial,estado_final,maximo_valor,minimo_valor)













## ===================================================================
## Algoritmo de decision minimax
## ===================================================================

##   En esta parte vamos a implementar el algoritmo de toma de
## decisiones minimax.

## ==================================================================
## Ejercicio 3
## ==================================================================

##   Implementar el procedimiento de decisión minimax visto en
## clase. Para ello definir las siguientes funciones:

## - minimax: Dado un juego, un estado del juego y un valor de
##   profundidad, devuelve el movimiento (aplicable a dicho estado en
##   el que tiene que jugar 'MAX', con mejor valor minimax de entre
##   todas las opciones disponibles) y el estado que resulta al
##   aplicar dicho movimiento.

## - valor_minimax: Dado un juego, un estado del juego, el jugador que
##   tiene el turno y un valor de profundidad, devuelve el valor
##   minimax obtenido como el valor de la función de evaluación
##   estática si se ha alcanzado la cota de profundidad, el estado es
##   final o no hay movimientos aplicables al estado; o el mejor de
##   los valores minimax de los estados sucesores (el máximo si juega
##   'MAX' o el mínimo si juega 'MIN').

## - maximizador: Dado un juego, un estado, una lista de movimientos
##   aplicables a dicho estado (sucesores) y un valor de profundidad,
##   devuelve el máximo de los valores minimax de los estados
##   obtenidos aplicando cada uno de los movimientos al estado
##   proporcionado.

## - minimizador: Dado un juego, un estado, una lista de movimientos
##   aplicables a dicho estado (sucesores) y un valor de profundidad,
##   devuelve el mínimo de los valores minimax de los estados
##   obtenidos aplicando cada uno de los movimientos al estado
##   proporcionado.

## ##################################################################

## >>> from juego import *
## >>> control(juego_nim, 'MAX', [minimax, 5])
## Estado  : 17
## Jugador : MAX
## Mi turno.
## Estado  : 16
## Jugador : MIN
## Los movimientos permitidos son:
##       Quitar 3 (0)
##       Quitar 2 (1)
##       Quitar 1 (2)
## Tu turno: 0
## Estado  : 13
## Jugador : MAX
## Mi turno.
## Estado  : 12
## Jugador : MIN
## Los movimientos permitidos son:
##       Quitar 3 (0)
##       Quitar 2 (1)
##       Quitar 1 (2)
## Tu turno: 0
## Estado  : 9
## Jugador : MAX
## Mi turno.
## Estado  : 8
## Jugador : MIN
## Los movimientos permitidos son:
##       Quitar 3 (0)
##       Quitar 2 (1)
##       Quitar 1 (2)
## Tu turno: 0
## Estado  : 5
## Jugador : MAX
## Mi turno.
## Estado  : 4
## Jugador : MIN
## Los movimientos permitidos son:
##       Quitar 3 (0)
##       Quitar 2 (1)
##       Quitar 1 (2)
## Tu turno: 0
## Estado  : 1
## Jugador : MAX
## Mi turno.
## Estado  : 0
## Jugador : MIN
## El humano ha ganado

def minimax(juego,estado,cota):
    max_val = float("-inf")
    mov_elegido = None
    nuevo_estado = None
    for m in juego.movimientos(estado):
        sucesor = juego.aplica(m,estado)
        valor_sucesor = valor_minimax(juego,sucesor,"MIN",cota-1)
        if max_val<valor_sucesor:
            max_val=valor_sucesor
            mov_elegido = m
            nuevo_estado = sucesor
    return (mov_elegido,nuevo_estado)



def valor_minimax(juego,estado,turno,cota):
    if(cota==0 or juego.es_estado_final(estado)):
        return juego.f_evaluacion(estado,turno)
    else:
        movs=juego.movimientos(estado)
        if turno=="MAX":
            return maximixador(juego,estado,movs,cota-1)
        else:
            return minimizador(juego,estado,movs,cota-1)

def maximixador(juego,estado,movs,cota):
    max_val=float("-inf")
    for m in movs:
        sucesor=juego.aplica(m,estado)
        valor_sucesor= valor_minimax(juego,sucesor,"MIN",cota)
        if max_val<valor_sucesor:
            max_val=valor_sucesor
    return max_val

def minimizador(juego,estado,movs,cota):
    min_val=float("inf")
    for m in movs:
        sucesor=juego.aplica(m,estado)
        valor_sucesor= valor_minimax(juego,sucesor,"MAX",cota)
        if min_val>valor_sucesor:
            min_val=valor_sucesor
    return min_val

## ------------------------------------------------------------------
## Ejercicio 4
## ------------------------------------------------------------------

##   Implementar el algoritmo de toma de decisiones minimax con poda
##   alfabeta.

## - alfa_beta: Dado un juego, un estado del juego y una cota de
##   profundidad, devuelve el movimiento (y el estado que resulta al
##   aplicarlo) del juego aplicable a dicho estado con el que tiene que
##   jugar 'MAX'. El movimiento con mejor valor minimax de entre todas
##   las opciones disponibles.


def alfa_beta(juego,estado,cota):
    alfa = float("-inf")
    beta = juego.maximo.valor
    mov_elegido = None
    nuevo_estado = None
    for m in juego.movimientos(estado):
        sucesor = juego.aplica(m,estado)
        valor_sucesor = valor_minimax_ab(juego,sucesor,"MIN",cota-1,alfa,beta)
        if alfa<valor_sucesor:
            alfa=valor_sucesor
            mov_elegido = m
            nuevo_estado = sucesor
            if alfa >= beta:
                break
    return (mov_elegido,nuevo_estado)



def valor_minimax_ab(juego,estado,turno,cota,alfa,beta):
    if(cota==0 or juego.es_estado_final(estado)):
        return juego.f_evaluacion(estado,turno)
    else:
        movs=juego.movimientos(estado)
        if turno=="MAX":
            return maximixador_ab(juego,estado,movs,cota-1,alfa,beta)
        else:
            return minimizador_ab(juego,estado,movs,cota-1,alfa,beta)

def maximixador_ab(juego,estado,movs,cota,alfa,beta):
    for m in movs:
        sucesor=juego.aplica(m,estado)
        valor_sucesor= valor_minimax_ab(juego,sucesor,"MIN",cota,alfa,beta)
        if alfa<valor_sucesor:
            alfa=valor_sucesor
            if alfa>=beta:
                break
    return alfa

def minimizador_ab(juego,estado,movs,cota,alfa,beta):
    for m in movs:
        sucesor=juego.aplica(m,estado)
        valor_sucesor= valor_minimax_ab(juego,sucesor,"MAX",cota,alfa,beta)
        if beta>valor_sucesor:
            beta=valor_sucesor
            if alfa>=beta:
                break
    return beta
