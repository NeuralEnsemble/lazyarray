# Support creating lazy arrays from SciPy sparse matrices 
#
# 1 program for the 7 sparse matrices classes : 
#
# csc_matrix(arg1[, shape, dtype, copy])	            Compressed Sparse Column matrix
# csr_matrix(arg1[, shape, dtype, copy])	            Compressed Sparse Row matrix
# bsr_matrix(arg1[, shape, dtype, copy, blocksize])	    Block Sparse Row matrix
# lil_matrix(arg1[, shape, dtype, copy])	            Row-based linked list sparse matrix
# dok_matrix(arg1[, shape, dtype, copy])	            Dictionary Of Keys based sparse matrix.
# coo_matrix(arg1[, shape, dtype, copy])	            A sparse matrix in COOrdinate format.
# dia_matrix(arg1[, shape, dtype, copy])	            Sparse matrix with DIAgonal storage
#


import numpy as np
from lazyarray import larray
from scipy import sparse 
import random


################
# Random numbers
################
i = random.randint(-100, 100)
j = random.randint(-100, 100)
k = random.randint(-100, 100)
l = random.randint(-100, 100)
m = random.randint(-100, 100)
n = random.randint(-100, 100)
p = random.randint(-100, 100)
q = random.randint(-100, 100)
r = random.randint(-100, 100)

################
# An example
################
#i = 1
#j = 2
#k = 0
#l = 0
#m = 0
#n = 3
#p = 1
#q = 0
#r = 4

#print "i =", i
#print "j =", j
#print "k =", k
#print "l =", l
#print "m =", m
#print "n =", n
#print "p =", p
#print "q  =", q
#print "r =", r    


##############################################################
# Definition of an array
##############################################################

def test_function_array_general():  
    A = np.array([[i, j, k], [l, m, n], [p, q, r]])
    #print "A ="
    #print A
    return A


##############################################################
# Definition of 7 sparse matrices
##############################################################

def sparse_csc_matrices():
    csc = sparse.csc_matrix([[i, j, k], [l, m, n], [p, q, r]])
    #print "csc matrices ="
    #print csc
    return csc

def sparse_csr_matrices():
    csr = sparse.csr_matrix([[i, j, k], [l, m, n], [p, q, r]])
    #print "csr matrices ="
    #print csr
    return csr

def sparse_bsr_matrices():
    bsr = sparse.bsr_matrix([[i, j, k], [l, m, n], [p, q, r]])
    #print "bsr matrices ="
    #print bsr
    return bsr

def sparse_lil_matrices():
    lil = sparse.lil_matrix([[i, j, k], [l, m, n], [p, q, r]])
    #print "lil matrices ="
    #print lil
    return lil

def sparse_dok_matrices():
    dok = sparse.dok_matrix([[i, j, k], [l, m, n], [p, q, r]])
    #print "dok matrices ="
    #print dok
    return dok

def sparse_coo_matrices():
    coo = sparse.coo_matrix([[i, j, k], [l, m, n], [p, q, r]])
    #print "coo matrices ="
    #print coo
    return coo

def sparse_dia_matrices():
    dia = sparse.dia_matrix([[i, j, k], [l, m, n], [p, q, r]])
    #print "dia matrices ="
    #print dia
    return dia

    

if __name__ == "__main__": 


##############################################################
# Call test_function_array_general
# Create a sparse matrix from array
# There are 7 sparse matrices
##############################################################

    #print "Array general ="
    test_function_array_general()    
    #print "Array ="
    #print test_function_array_general()
    
#    print "----"

#    print "Sparse array csc general ="
    sA_csc_general = sparse.csc_matrix(test_function_array_general())
    #print ("sparse csc matrices", sparse.csc_matrix(test_function_array_general()))
    #print "sparse csc matrices ="
    #print sA_csc_general
#    print "----"
#    print "Sparse array csr general ="
    sA_csr = sparse.csr_matrix(test_function_array_general())
    #print ("sparse csr matrices", sparse.csr_matrix(test_function_array_general()))
    #print "sparse csr matrices ="
    #print sA_csr
#    print "----"
#    print "Sparse array bsr general  ="
    sA_bsr = sparse.bsr_matrix(test_function_array_general())
#    print ("sparse bsr matrices", sparse.bsr_matrix(test_function_array_general()))
#    print "sparse bsr matrices ="
#    print sA_bsr
#    print "----"
#    print "Sparse array lil general ="
    sA_lil = sparse.lil_matrix(test_function_array_general())
#    print ("sparse lil matrices", sparse.lil_matrix(test_function_array_general()))
#    print "sparse lil matrices ="
#    print sA_lil
#    print "----"
#    print "Sparse array dok general ="
    sA_dok = sparse.dok_matrix(test_function_array_general())
#    print ("sparse dok matrices", sparse.dok_matrix(test_function_array_general()))
#    print "sparse dok matrices ="
#    print sA_dok
#    print "----"
#    print "Sparse array coo general ="
    sA_coo = sparse.coo_matrix(test_function_array_general())
#    print ("sparse coo matrices", sparse.coo_matrix(test_function_array_general()))
#    print "sparse coo matrices ="
#    print sA_coo
#    print "----"
#    print "Sparse array dia general ="
    sA_dia = sparse.dia_matrix(test_function_array_general())
#    print ("sparse dia matrices", sparse.dia_matrix(test_function_array_general()))
#    print "sparse dia matrices ="
#    print sA_dia


#print "----------------------------------------------------------------------"
 

 ##############################################################
 # Call the sparse matrices
 # Create a lazy array from sparse matrices
 ##############################################################
 
 
Array_csc_matrices = sparse_csc_matrices().toarray()
#print "Array csc matrices ="
#print Array_csc_matrices
  
Array_csr_matrices = sparse_csr_matrices().toarray()
#print "Array csr matrices ="
#print Array_csr_matrices

Array_bsr_matrices = sparse_bsr_matrices().toarray()
#print "Array bsr matrices ="
#print Array_bsr_matrices

Array_lil_matrices = sparse_lil_matrices().toarray()
#print "Array lil matrices ="
#print Array_lil_matrices
    
Array_dok_matrices = sparse_dok_matrices().toarray()
#print "Array dok matrices ="
#print Array_dok_matrices

Array_coo_matrices = sparse_coo_matrices().toarray()
#print "Array coo matrices ="
#print Array_coo_matrices

Array_dia_matrices = sparse_dia_matrices().toarray()
#print "Array dia matrices ="
#print Array_dia_matrices