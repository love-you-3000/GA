import numpy as np
import random
import heapq
import matplotlib.pyplot as plt
from math import sin,cos


def function(x):
    return x + 10*sin(5*x) + 7*cos(4*x)


def getEncodeLength(decision_variables, delta):
    """
    :param decision_variables: 自变量的范围
    :param delta: 精度
    :return: 染色体长度
    """
    lengths = []
    for des in decision_variables:
        up = des[1]
        low = des[0]
        n = (up - low) * (1 / delta)
        length = 1
        while 2 ** length < n:
            length += 1
        lengths.append(length)
    return lengths


def get_initial_population(length, population_size):
    """
    :param length:染色体长度，由 getEncodeLength() 方法求得
    :param population_size: 生成初始化种群的数量
    :return: 返回初始化种群列表
    """
    chromsomes = np.zeros((population_size, length), dtype=np.int)
    for popusize in range(population_size):
        # np.random.randint()产生[0,2)之间的随机整数，第三个参数表示随机数的数量
        chromsomes[popusize, :] = np.random.randint(0, 2, length)
    return chromsomes


def getDecode(population, encode_length, decision_variables):
    population_size = population.shape[0]
    length = len(encode_length)
    # 初始化解码数组
    decode_variables = np.zeros((population_size,length), dtype=np.float)
    for i, population_child in enumerate(population):
        start = 0
        for j, length_child in enumerate(encode_length):
            power = length_child - 1
            decimal = 0
            for k in range(start,start+length_child):
                decimal += population_child[k]*(2**power)
                power -= 1
            start = length_child
            lower = decision_variables[j][0]
            upper = decision_variables[j][1]
            decode_value = lower+decimal*(upper-lower)/(2**length_child-1)
            decode_variables[i][j] = decode_value
    return decode_variables


def get_fitness_value(func,decode):
    # 获取种群的大小和决策变量的个数
    population_size, decision_val = decode.shape
    # 初始化适应度空间
    fitness_value = np.zeros((population_size, 1))
    for pop_num in range(population_size):
        fitness_value[pop_num][0] = func(decode[pop_num][0])
    probability = fitness_value / np.sum(fitness_value)
    cum_probability = np.cumsum(probability)
    return fitness_value, cum_probability


def selection_new_population(decode_population, cum_probability):
    m,n = decode_population.shape
    # 初始化新种群
    new_population = np.zeros((m,n))
    for i in range(m):
        random_number = np.random.random()
        for j in range(m):
            # 轮盘赌
            if random_number<cum_probability[j]:
                new_population[i] = decode_population[j]
                break
    return new_population


def cross_new_population(new_population, prob):
    """
    交叉操作
    :param new_population: =种群
    :param prob: 交叉概率
    :return: 交叉后的种群
    """
    m, n = new_population.shape
    # number表示有多少种群要进行交叉操作
    number = np.uint8(m * prob)
    if number % 2 != 0:
        number += 1
    update_population = np.zeros((m, n), dtype=np.uint8)
    index = random.sample(range(m), number)
    for i in range(m):
        # 不需要进行交叉操作的个体直接复制到新种群中
        if not index.__contains__(i):
            update_population[i] = new_population[i]
    # 交叉操作
    j = 0
    while j < number:
        # 随机生成交叉点
        cross_point = np.random.randint(0, n, 1)
        # 该函数返回一个列表，还需要将值取出
        cross_point = cross_point[0]
        update_population[index[j]][0:cross_point] = new_population[index[j]][0:cross_point]
        update_population[index[j]][cross_point:] = new_population[index[j+1]][cross_point:]
        update_population[index[j+1]][0:cross_point] = new_population[index[j+1]][0:cross_point]
        update_population[index[j+1]][cross_point:] = new_population[index[j]][cross_point:]
        j += 2
    return update_population


def mutation(cross_population, mutation_prob):
    # 初始化变异种群
    mutation_population = np.copy(cross_population)
    m, n = cross_population.shape
    mutation_nums = np.uint8(m * n * mutation_prob)
    mutation_index = random.sample(range(m*n), mutation_nums)
    for gene_index in mutation_index:
        row = np.uint8(np.floor(gene_index/n))
        column = gene_index % n
        if mutation_population[row][column] == 0:
            mutation_population[row][column] = 1
        else:
            mutation_population[row][column] = 0
    return mutation_population


def find_max_population(population, max_value_population, max_size):
    max_value = max_value_population.flatten()
    max_value_list = max_value.tolist()
    max_index = map(max_value_list.index,heapq.nlargest(100,max_value_list))
    index = list(max_index)
    column = population.shape[1]
    max_population = np.zeros((max_size,column))
    i = 0
    for ind in index:
        max_population[i] = population[ind]
        i += 1
    return max_population


def main():
    optimal_value = []
    optimal_variables= []

    limits = [[-100,100]]
    eps = 0.001
    encode_length = getEncodeLength(limits,eps)
    init_population_nums = 100
    populations = get_initial_population(sum(encode_length),init_population_nums)
    epochs = 100
    pc = 0.6
    pm = 0.01
    max_population_nums = 100
    for epoch in range(epochs):
        decode = getDecode(populations, encode_length, limits)
        evaluation, cum_prob = get_fitness_value(function, decode)
        new_population = selection_new_population(populations, cum_prob)
        cross_population = cross_new_population(new_population, pc)
        mutation_population = mutation(cross_population, pm)
        total_population = np.vstack((populations, mutation_population))
        final_decode = getDecode(total_population, encode_length, limits)
        final_evaluation, final_cum_prob = get_fitness_value(function, final_decode)
        populations = find_max_population(total_population, final_evaluation, max_population_nums)
        optimal_value.append(np.max(final_evaluation))
        index = np.where(final_evaluation == max(final_evaluation))
        optimal_variables.append(final_decode[index[0][0]])

    x = [i for i in range(epochs)]
    y = [optimal_value[i] for i in range(epochs)]
    plt.plot(x, y)
    plt.show()

    optimal_val = np.max(optimal_value)
    index = np.where(optimal_val == max(optimal_value))
    optimal_var = optimal_variables[index[0][0]]
    return optimal_val, optimal_var


if __name__ == '__main__':
    x = [i for i in np.arange(-100,100,0.01)]
    y = [function(i) for i in x]
    plt.plot(x,y)
    plt.show()
    opt_value, opt_var = main()
    print("x", opt_var[0])
    print("y", opt_value)



