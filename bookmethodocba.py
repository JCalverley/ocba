#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 21 15:09:51 2019

@author: JoeCalverley
"""

import numpy as np
import math
import scipy.stats as sk
import matplotlib.pyplot as plt

def choose_b(J_est):
    # Input is the array of current estimates of means
    b = J_est.index(min(J_est))
    #Function returns the index of the smallest element in J_est
    return b


def choose_b_np(J_est):
    # Need this function as opposed ot the simpler choose be when working with numpy arrays
    b = np.where(J_est == J_est.min())
    return b


def delta(k, b, mean):
    #Finds the deltas; difference in estimated mean of design b and each of the others 
    delta_ = np.zeros(k-1)
    j = 0
    for i in range(0,k):  
        # Delta for i = b does not exist
        if i == b[0] :
            continue
        else:
            delta_[j] = mean[i] - mean[b]
            # We use the j = j + 1 since there are k - 1 deltas for k designs
            j = j + 1
    return delta_


def i_ind(i,b):
    # Given that from the mean array to the delta array we lose an entry, the sunsequent indices of the elements changes
    # If i is before b then its index does not change
    if i < b[0]:
        index_i = i
    # But if i comes after b, its index is one less than previously
    else:
        index_i = i - 1
    # Rather than dealing with an array, this function works only with one value at a time, and hence outputs such
    return index_i
        

def N_i_N_j(delta_ij, var_i_j):
    # Finds the ratio between two designs (Using formula on page 50 in OCBA book)
    N_j = ((var_i_j[1] * (delta_ij[0] ** 2)) / (var_i_j[0] * (delta_ij[1] ** 2))) 
    return N_j


def N_b(k, b, N, var):
    # Finds ratio for the current best option (Using formula on page 50 in OCBA book)
    # Note also how this ratio differs from that with i =/ b
    sumNCC = 0
    for i in range(0,k):
        if i == b :
            continue
        sumNCC += (N[i] ** 2) / (var[i])
    N_b = math.sqrt(var[b]) * math.sqrt(sumNCC)
    return N_b
    
    
def initial_ratio(k, mean, var):
    # Calculates the fraction of simulations each design should be allocated
    # ie it is not dependent on how many sims are to be allocated
    b = choose_b_np(mean)
    # From before, this function calculates (small) deltas
    delta_ = delta(k, b, mean) 
    N = np.zeros(k)
    # Need to determine which is the first design to use, since this is the one whose ration we set as 1
    if b == 0:
        i = 1
    else:
        i = 0
    N[i] = 1
    for j in range(0,k): 
        if j == b:
            continue
        if j == i:
            continue
        # Using the index changing function
        # Here we are specifying the parameters to calculate the ratio for each design j
        delta_ij = [delta_[i_ind(i,b)], delta_[i_ind(j,b)]]
        var_i_j = [var[i], var[j]]
        # Given N[i] = 1, the we calculate N[j] using fucntion derived above
        N[j] = N_i_N_j(delta_ij, var_i_j)
    # Now we calculate the ratio of design b, given that we know the ratios of the other designs
    N[b] = N_b(k, b, N, var)
    # Now implement a simple calculation that gives the fraction required for each design i (incl b)
    ratio_sum = N.sum()
    prop = np.zeros(k) # Think about numpy arrays here    
    for i in range(0,k):
        prop[i] =  N[i] / ratio_sum
    # Returns an array of fractions, whose sum of course equals 1
    return prop


def rng_repeats_for_OCBA(option, mean=[1,2,3,4,5,6], var=[1,1,1,1,1,1], n=5):
    # This function generates n random numbers from the N(mean, var) distribution
    # It is used in OCBA to generate the (n_0) initial simulations for each design
    # Given that the option states is using normal indices, we need to subtract one to accommadate
    mean_option = mean[option - 1]
    var_option = var[option - 1]
    # np.random.randn(n) creates an array of random numbers from the standard normal dist. 
    # Therefore we need to use var and mean to scale it
    random_nos = (var_option ** 0.5) * np.random.randn(n) + mean_option
    # The variance function in Python divides by n rather than n-1, hence we need to edit the resulting var value
    correct_var = n * random_nos.var() / (n - 1)
    # Returns a tuple, containing that designs estimated mean and varainace for the generated sample
    return random_nos.mean(), correct_var


def rng_repeats_for_OCBA_2(option, mean, var, n):
    # This function differs from the one above in that it intakes an array containing how many numbers to generate, rather than a single value
    # It then takes, given the design chosen, the correct number to simulate
    # Again, note that we need to minus one to obtain the correct index
    mean_option = mean[option - 1]
    var_option = var[option - 1]
    # Need to change from float to integer in order for 'range' to work
    integer_n = int(n[option - 1])
    # Again, we need to scale the standard normal distribution
    random_nos_2 = (var_option ** 0.5) * np.random.randn(integer_n) + mean_option
    # Given that we are dividing by n-1, if n=1, as is standard, then we will end up dividing by 0 
    # One value has a variance of 0
    if n[option - 1] == 1:
        correct_var = 0
    else:
        # Agaian we need to scale using n/(n-1)
        correct_var = n[option - 1] * random_nos_2.var() / (n[option - 1] - 1)
    # Returns a tuple 
    return random_nos_2.mean(), correct_var


def new_var(mean_initial, var_initial, mean_new, var_new, n, allocation, pos):
    # Calculating the mean of a series of numbers, without knowing the previous number is easy enough
    # But calculating the variance requires work, hence this function is required
    # This function uses information on the webpage http://mathforum.org/library/drmath/view/52820.html
    # However, as with python, the formula given does not divide by n-1, hence we need to do as previously. at the appropriate points
    sum_squ_initial = n * ((n-1)*var_initial[pos-1]/(n) + (mean_initial[pos-1] ** 2))
    sum_squ_new = allocation * ((allocation-1)*var_new/(allocation) + (mean_new ** 2))
    sum_init = n * mean_initial[pos-1] 
    sum_new_ = allocation * mean_new
    variance_new = ((n+allocation) * ((n + allocation) * (sum_squ_new + sum_squ_initial) - ((sum_new_ + sum_init) ** 2))) / ((n+allocation-1) * ((n + allocation) ** 2))
    return variance_new


def prob_Jb_minus_Ji(N, J_est, var):
    # This function is used in the calulation of APSC
    # It calculates P(J_tilda_b - J_tilda_i < 0) by calculating P(P_tilda_i - J_tilda_b > 0)
    est = J_est[1]-J_est[0]
    sd = math.sqrt((var[0]/N[0])+(var[1]/N[1]))
    # Here we standardise the parameters
    z = est / sd
    # Here we use the imported skipy.stats module for the cumulative distribution
    prob = sk.norm.cdf(z)
    return prob


def APCS_B(k, N, J_est, var):
    # This function calculates the Approximate value of Correct Selection (using Bonferroni inequality) using the formula on page 37 of OCBA book
    b = choose_b_np(J_est)[0]
    probs = np.zeros(k)
    for i in range(0,k):
        if i == b:
            continue
        # For each design, excluding b, we create 3 1by2 arrays containing, respectively:
        # The number of replications the design i has completed
        N_prob = [N[b],N[i]]
        # The mean of design i
        J_est_prob = [J_est[b], J_est[i]]
        # The variance of design i
        var_prob = [var[b], var[i]]
        # Each also contains the corresponding parameter for design b, which does not change for each i
        probs[i] = 1 - prob_Jb_minus_Ji(N_prob, J_est_prob, var_prob)
    APSCB = round(1 - np.sum(probs),4)
    return APSCB


def allocation(prop, tri_del):
    # Given the triangle delta and the fraction of that allocation that each design should simulate,
    # This function uses the idea of creating a cumulative probability distribution using these fractions, which, usefully, sum to 1
    # Multiplying each fraction by triangle delta and rounding to the nearest integer creates the issue that these rounded numbers may not sum to triangle delta
    # This alternative method is therefore a solution
    k = len(prop)
    # This initial array, contains the allocation for this round, to which we iteratively add allocations
    # The sum of this array will equal triangle delta
    initial = np.zeros(k)
    for d in range(0,tri_del):
        # Generate a random number using U(0,1)
        random = rand.random()
        cdf = 0
        # If the random number is within the range corresponding to design i then we allocate the replication to i
        for i in range(0,k):
            cdf += prop[i]
            if cdf < random:
                continue
            else:
                initial[i] = initial[i] + 1
                break
    return initial


def OCBA_method(k, mean, var, T, n_0=5, tri_del=1):
    # This is the general function which calculates the best design, as well as the APCS using the OCBA method described on page 50 of the OCBA book
    
    # Define initial arrays which we will fill with the initial estimated mean and variance
    mean_initial = np.zeros(k)
    var_initial = np.zeros(k)
    # Generate the initial estimated mean and variance for each design
    for i in range(1,k+1):
        gen = rng_repeats_for_OCBA(i, mean, var, n_0)
        mean_initial[i-1] = gen[0]
        var_initial[i-1] = gen[1]
    #print('After', n_0, 'simulations, the estimated mean is:', mean_initial)
    #print('and the estimated variance is:',var_initial)
    # We have conducted n_0 simulations for each design so the current completed is k times n_0
    current_completed = k * n_0
    # This array will keep track of how many replication each design has done
    no_sims = np.full(k, n_0)
    # The integer 'multi' is the no of trideltas that fit in the remaining T after the initial k*n_0 simulations
    multi = int((T - current_completed) / tri_del)
    APCSB_current = APCS_B(k, no_sims, mean_initial, var_initial)
    APCS_graph = [APCSB_current]
    l=1
    for replications in range(0,multi):
        # Generate the initial ratio following the first n_o replications    
        props = initial_ratio(k, mean_initial, var_initial)
        # Use the random number generator to find the allocation
        allocation_sims = allocation(props, tri_del)
        mean_new = np.zeros(k)
        var_new = np.zeros(k)
        # We only need to include the designs for which we have allocated replications in the calculation of new mean and variance
        to_do = np.where(allocation_sims > 0)
        for j in to_do[0]:
            # For each design with at least one allocation, we generate that many new random numbers from the N(mean, var) dist
            # The fn below was designed for specifying the 'Option' as we would generally
            # However, since j is taking an index, we need to add 1 for the function to work as required
            new_gen = rng_repeats_for_OCBA_2(j+1, mean, var, allocation_sims)
            # Given these generated values, we now calculate the new mean and variance of design j 
            # As said before, the mean is relatively simple
            mean_new[j] = ((no_sims[j] * mean_initial[j]) + (allocation_sims[j] * new_gen[0])) / (no_sims[j] + allocation_sims[j])
            # But for the variance we need to use the function as define earlier
            var_new[j] = new_var(mean_initial, var_initial, new_gen[0], new_gen[1], no_sims[j], allocation_sims[j], j+1)
            # We can then update the number of replications design j has done
            no_sims[j] = no_sims[j] + allocation_sims[j]
        # For those designs where we have not allocated any replications, the means and variances remain the same
        to_not_do = np.where(allocation_sims == 0)
        for m in to_not_do[0]:
            mean_new[m] = mean_initial[m]
            var_new[m] = var_initial[m]
        # Having obtained mean_new and var_new which have been updated from mean_initial and var_initial, or not
        # Then we set them back as before for the next tri delta
        APCSB_current = APCS_B(k, no_sims, mean_initial, var_initial)
        current_completed = current_completed + tri_del
        APCS_graph.append(APCSB_current)
        mean_initial = mean_new
        var_initial = var_new
        l=l+1
    x=np.linspace(k*n_0, current_completed, num=l)
    #print(x)
    #print(APCS_graph)
    print(plt.plot(x, APCS_graph, 'r-'))
    #print('number of sims per design', no_sims)
    final_mean = mean_initial
    #print('Final mean:', final_mean)
    final_var = var_initial
    #print('Final var:', final_var)
    # Approximate probability of correct selection
    APCS_B_ = APCS_B(k, no_sims, final_mean, final_var)
    #print('The Approximate Probability of Correct Selection is:', APCS_B_)
    #print('The Final Chosen design is: Option', choose_b_np(final_mean)[0] + 1)
    return APCS_B_
    

# Example
example_mean = [1,2,3,4,5]
example_var = np.arange(5, 10)


OCBA_method(5, example_mean, example_var, 50, tri_del=1) 

k = 100


for t in range(0,100-25):
    avg_for_graph = np.zeros(100-25)
    n_0_equals_5 = np.zeros(k)
    for i in range(0,k):
        n_0_equals_5[i] = OCBA_method(5, example_mean, example_var, t+25)
        print(t, i)
    avg_for_graph[t] = n_0_equals_5.mean()
print(avg_for_graph)

x=np.linspace(25, 100, num=74)
print(plt.plot(x, avg_for_graph, 'r-'))




def OCBA_method_target_PCS(k, mean, var, n_0=5, tri_del=1, P=0.95):
    # This is the general function which calculates the total number of simulations required to obtain the desired PRobability of correct selection (and hence the best design) using the OCBA method described on page 50 of the OCBA book
    
    # Define initial arrays which we will fill with the initial estimated mean and variance
    mean_initial = np.zeros(k)
    var_initial = np.zeros(k)
    # Generate the initial estimated mean and variance for each design
    for i in range(1,k+1):
        gen = rng_repeats_for_OCBA(i, mean, var, n_0)
        mean_initial[i-1] = gen[0]
        var_initial[i-1] = gen[1]
    print('After', n_0, 'simulations, the estimated mean is:', mean_initial)
    print('and the estimated variance is:',var_initial)
    # We have conducted n_0 simulations for each design so the current completed is k times n_0
    # This parameter will be the returned value
    current_completed = k * n_0
    # This array will keep track of how many replication each design has done
    no_sims = np.full(k, n_0)
    # Given our n_0 and the inputs, it may be that the P(CS) is higher than our target without any further simulations
    APCSB_current = APCS_B(k, no_sims, mean_initial, var_initial)
    if APCSB_current > 0.95:
        return k*n_0
    else:
        print('after', current_completed, 'initial sims APCS = ', APCSB_current)
        APCSB_after_initial = APCSB_current
    # Since APCS is a lower bound of P(CS) then we can be certain that once the APCS is achieved, then so is P(CS)
    # Here we use a while loop, which stops when APCS reaches the desired level
    APCS_graph = [APCSB_after_initial]
    l=1
    while APCSB_current < P:
        # Generate the initial ratio following the first n_o replications    
        props = initial_ratio(k, mean_initial, var_initial)
        # Use the random number generator to find the allocation
        allocation_sims = allocation(props, tri_del)
        mean_new = np.zeros(k)
        var_new = np.zeros(k)
        # We only need to include the designs for which we have allocated replications in the calculation of new mean and variance
        to_do = np.where(allocation_sims > 0)
        for j in to_do[0]:
            # For each design with at least one allocation, we generate that many new random numbers from the N(mean, var) dist
            # The fn below was designed for specifying the 'Option' as we would generally
            # However, since j is taking an index, we need to add 1 for the function to work as required
            new_gen = rng_repeats_for_OCBA_2(j+1, mean, var, allocation_sims)
            # Given these generated values, we now calculate the new mean and variance of design j 
            # As said before, the mean is relatively simple
            mean_new[j] = ((no_sims[j] * mean_initial[j]) + (allocation_sims[j] * new_gen[0])) / (no_sims[j] + allocation_sims[j])
            # But for the variance we need to use the function as define earlier
            var_new[j] = new_var(mean_initial, var_initial, new_gen[0], new_gen[1], no_sims[j], allocation_sims[j], j+1)
            # We can then update the number of replications design j has done
            no_sims[j] = no_sims[j] + allocation_sims[j]
        # For those designs where we have not allocated any replications, the means and variances remain the same
        to_not_do = np.where(allocation_sims == 0)
        for m in to_not_do[0]:
            mean_new[m] = mean_initial[m]
            var_new[m] = var_initial[m]
        # Having obtained mean_new and var_new which have been updated from mean_initial and var_initial, or not
        # Then we set them back as before for the next tri delta
        mean_initial = mean_new
        var_initial = var_new
        APCSB_current = APCS_B(k, no_sims, mean_initial, var_initial)
        current_completed = current_completed + tri_del
        APCS_graph.append(APCSB_current)
        #print(APCSB_current)
        current_completed = current_completed + tri_del
        l=l+1
        if current_completed > 10000:
            break
    x=np.linspace(k*n_0, current_completed, num=l)
    print(plt.plot(x, APCS_graph, 'r-'))
    #print('number of sims per design', no_sims)
    final_mean = mean_initial
    #print('Final mean:', final_mean)
    final_var = var_initial
    #print('Final var:', final_var)
    #print('The Final Chosen design is: Option', choose_b_np(final_mean)[0] + 1)
    #print('Obtained after', current_completed, 'simulations')
    return current_completed

example_mean = np.arange(1,8)
example_var = np.full(7, 10)

OCBA_method_target_PCS(7, example_mean, example_var, tri_del=2)
            
k = 100

n_0_equals_5 = np.zeros(k)
for i in range(0,k):
    n_0_equals_5[i] = OCBA_method_target_PCS(10, example_mean, example_var)
    print('1st', i)
print(n_0_equals_5.mean())

n_0_equals_4 = np.zeros(k)
for i in range(0,k):
    n_0_equals_4[i] = OCBA_method_target_PCS(10, example_mean, example_var, n_0=4)
    print('2nd', i)
print(n_0_equals_4.mean())

n_0_equals_3 = np.zeros(k)
for i in range(0,k):
    n_0_equals_3[i] = OCBA_method_target_PCS(10, example_mean, example_var, n_0=3)
    print('3rd', i)
print(n_0_equals_3.mean())

n_0_equals_2 = np.zeros(k)
for i in range(0,k):
    n_0_equals_2[i] = OCBA_method_target_PCS(10, example_mean, example_var, n_0=2)
    print('4th', i)
print(n_0_equals_2.mean())
     
            
            
def OCBA_test(k, mean, mean_initial, var, var_initial, n_0=5, tri_del=1, P=0.95):
    # This is the general function which calculates the total number of simulations required to obtain the desired PRobability of correct selection (and hence the best design) using the OCBA method described on page 50 of the OCBA book
    
    current_completed = k * n_0
    # This array will keep track of how many replication each design has done
    no_sims = np.full(k, n_0)
    # Given our n_0 and the inputs, it may be that the P(CS) is higher than our target without any further simulations
    APCSB_current = APCS_B(k, no_sims, mean_initial, var_initial)
    if APCSB_current > 0.95:
        return k*n_0
    else:
        print('after', current_completed, 'initial sims APCS = ', APCSB_current)
        APCSB_after_initial = APCSB_current
    # Since APCS is a lower bound of P(CS) then we can be certain that once the APCS is achieved, then so is P(CS)
    # Here we use a while loop, which stops when APCS reaches the desired level
    APCS_graph = [APCSB_after_initial]
    l=1
    while APCSB_current < P:
        # Generate the initial ratio following the first n_o replications    
        props = initial_ratio(k, mean_initial, var_initial)
        # Use the random number generator to find the allocation
        allocation_sims = allocation(props, tri_del)
        mean_new = np.zeros(k)
        var_new = np.zeros(k)
        # We only need to include the designs for which we have allocated replications in the calculation of new mean and variance
        to_do = np.where(allocation_sims > 0)
        for j in to_do[0]:
            # For each design with at least one allocation, we generate that many new random numbers from the N(mean, var) dist
            # The fn below was designed for specifying the 'Option' as we would generally
            # However, since j is taking an index, we need to add 1 for the function to work as required
            new_gen = rng_repeats_for_OCBA_2(j+1, mean, var, allocation_sims)
            # Given these generated values, we now calculate the new mean and variance of design j 
            # As said before, the mean is relatively simple
            mean_new[j] = ((no_sims[j] * mean_initial[j]) + (allocation_sims[j] * new_gen[0])) / (no_sims[j] + allocation_sims[j])
            # But for the variance we need to use the function as define earlier
            var_new[j] = new_var(mean_initial, var_initial, new_gen[0], new_gen[1], no_sims[j], allocation_sims[j], j+1)
            # We can then update the number of replications design j has done
            no_sims[j] = no_sims[j] + allocation_sims[j]
        # For those designs where we have not allocated any replications, the means and variances remain the same
        to_not_do = np.where(allocation_sims == 0)
        for m in to_not_do[0]:
            mean_new[m] = mean_initial[m]
            var_new[m] = var_initial[m]
        # Having obtained mean_new and var_new which have been updated from mean_initial and var_initial, or not
        # Then we set them back as before for the next tri delta
        mean_initial = mean_new
        var_initial = var_new
        APCSB_current = APCS_B(k, no_sims, mean_initial, var_initial)
        current_completed = current_completed + tri_del
        APCS_graph.append(APCSB_current)
        #print(APCSB_current)
        current_completed = current_completed + tri_del
        l=l+1
        if current_completed > 10000:
            break
    x=np.linspace(k*n_0, current_completed, num=l)
    print(plt.plot(x, APCS_graph, 'g-'))
    #print('number of sims per design', no_sims)
    final_mean = mean_initial
    print('Final mean:', final_mean)
    final_var = var_initial
    print('Final var:', final_var)
    #print('The Final Chosen design is: Option', choose_b_np(final_mean)[0] + 1)
    #print('Obtained after', current_completed, 'simulations')
    return current_completed
















































