import random
from numpy import array
from numpy import argmin
from numpy import argmax
from math import cos
from math import pi
from math import ceil


# variaveis da Função Rastrigin
N = 2 #numero de dimensões
A = 10
range_x = [-5.12, 5.12]
# variaveis do AG
num_exec = 30
max_generations = 100
pop_size = 20
chance_mut=0.3


def createIndividuo(min=-5.12, max=5.12, size=2):
    return [random.uniform(min, max) for i in range(size)]


def createPopulation():
    return[createIndividuo(range_x[0], range_x[1], N) for i in range(pop_size)]


def rastrigin(dimension):
        return (A*N + (sum([(x)**2 - A*cos(2*pi*(x)) for x in dimension])))


def calcRastrigin(candidates):
    aux = []
    for c in candidates:
        z = rastrigin(c)
        aux.append(z)
    return aux


def inverte(fx): #os menores valores devem ter fitness maiores
    hx = min(fx)
    desl = 10**-0.2 - hx
    return 1/(fx + desl)


def calcFitness(fx): #inverte o valor do calculo de Rastrigin
    nFX=array(fx) #utiliza a biblioteca numpy
    return inverte(nFX)


def isSolucaoEncontrada(z):
    for i in z:
        if i == 0:
            return True
    return False


def selecaoTorneio(populacao, fitness):  #seleção por torneio 2 a 2
    aptos = []
    for i in range(pop_size): #loop para selecionar 20 individuos
        j = random.randint(0, pop_size-1)
        k = random.randint(0, pop_size-1)
        if fitness[j] > fitness[k]: #o individuo com o maior valor de fitness é selecionado
            aptos.append(populacao[j])
        else:
            aptos.append(populacao[k])
    return aptos


def mutacao(populacao, taxa=0.3):
    quant_indiv = ceil(len(populacao) * taxa) #quantidade de individuos que sofrerão a mutação
    indices = [random.randint(0, pop_size-1)for i in range(quant_indiv)] #soteia quais individuos sofrerão a mutação
    j = random.randint(0, 1)  #sorteia o indice da dimensão x1 ou x2
    for i in indices:
        mais_ou_menos = random.randint(0, 1)
        if mais_ou_menos == 0:
            populacao[i][j] += populacao[i][j]*0.05 #realiza a mutação de + ou - 5% do valor do individuo
        else:
            populacao[i][j] -= populacao[i][j]*0.05
    return populacao


'''
1. Escolher um conjunto de cromossomos iniciais
2. Repetir
2.1 Definir nota de cada cromossomo
2.2 Selecionar os cromossomos mais aptos
2.3 Aplicar operadores de reprodução sobre cromossomos selecionados
3. Até cromossomo adequado ser obtido ou serem realizadas N gerações
'''


'''
A sequência de passos esperado em AG é:

1 Gerar a população inicial
2 Avaliaçao dos indivíduos (calcular o fitness)
3 Seleção de quais indivíduos sofrem as operações genéticas
4 Cruzamento (não será realizado no primeiro experimento)
5 Mutação (de 30% dos indivíduos)
6 Atualização da população final (será considerado algum pai para a próxima geração?)
7 Finalização (Critério de parada)
'''

def runGeneration():
    pop_ini = createPopulation()  #gera a população inicial
    melhores=[] # vetor auxiliar para salvar os melhoes individuos das gerações

    for i in range(max_generations): #loop de 100 gerações
        z = calcRastrigin(pop_ini) #realiza o calculo da Função de Rastrigin

        if(isSolucaoEncontrada(z)):
            print("\n    Solucao Encontrada!")
            break
        
        fitness = calcFitness(z) #realiza a inversão dos valores da Função de Rastrigin

        ind_melhor_indiv = argmax(fitness) #o melhor individuo é aquele que tem o maior valor de fitness
        pop_sel = selecaoTorneio(pop_ini,fitness) #realiza seleção por torneio 2 a 2
        pop_mut = mutacao(pop_sel,chance_mut)

        z = calcRastrigin(pop_mut)
        fitness = calcFitness(z)
        ind_pior_fit = argmin(fitness) #o pior individuo é aquele que tem o menor valor de fitness
        pop_mut[ind_pior_fit] = pop_ini[ind_melhor_indiv]
        
        #recalcula o fitness para a população mutada
        z = calcRastrigin(pop_mut)
        fitness=calcFitness(z)
        ind_mg = argmax(fitness) #indice do melhor da geração
        melhores.append(pop_mut[ind_mg])
        print("GER %i:  i:%i Melhor: %s  Z: %f  Fit: %s" % (i+1,ind_mg,pop_mut[ind_mg],z[ind_mg],fitness[ind_mg]))

        pop_ini = pop_mut #a população mutada se torna a população inicial da proxima geração

    #recalcula o fitness dos melhores individos das 100 gerações
    z = calcRastrigin(melhores)
    fitness=calcFitness(z)
    ind_mm = argmax(fitness)
    #print("i: %i:  melhor: %s - Z:%s - FIT:%s"%(ind_mm,melhores[ind_mm],z[ind_mm],fitness[ind_mm]))
    return melhores[ind_mm]


def main():
    print("=====MAIN=====")
    melhores=[] # vetor auxiliar para salvar os melhoes individuos das execuções
    for i in range(num_exec): #loop de 30 execuções
        print("=====Execucao %i" % (i+1))
        indiv = runGeneration()
        melhores.append(indiv)

    z = calcRastrigin(melhores)
    fitness = calcFitness(z)
    ind_mm = argmax(fitness)

    print("\nOs %i melhores: "%(num_exec))
    i=0
    for m in melhores:
        print("%i: %s  z: %s  fit: %s"%(i+1,m,z[i],fitness[i]))
        i+=1

    print("\nO melhor dos melhores: ")
    print("posicao:%i %s  -  Z: %s  -  FIT: %s"%(ind_mm+1,melhores[ind_mm],z[ind_mm],fitness[ind_mm]))
    
#runGeneration()
main()