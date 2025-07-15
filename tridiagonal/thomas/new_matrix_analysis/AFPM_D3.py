#!/usr/bin/env python
# coding: utf-8

# In[9]:


import random
import time
from tqdm import tqdm


# In[10]:


def int_to_bin(c) :
    binary = [0,0,0,0,0,0,0,0]
    
    for i in range(7):
        binary[i] = c % 2
        c = c // 2
        if i==6 :
            binary[7] = c
    
    return binary

def int_to_48bin(c) :
    binary = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    
    for i in range(47):
        binary[i] = c % 2
        c = c // 2
        if i==46 :
            binary[47] = c
    
    return binary

def Group(c,d):
    pp0 = [0,0,0,0,0,0,0,0]
    pp1 = [0,0,0,0,0,0,0,0]
    pp2 = [0,0,0,0,0,0,0,0]
    pp3 = [0,0,0,0,0,0,0,0]
    
    clm = [0,0,0,0,0,0,0,0,0,0,0]
    cry = [0,0,0,0,0,0,0,0,0,0,0]
    bit = [0,0,0,0,0,0,0,0,0,0,0]
    result = 0
    
    for i in range(8):
        pp0[i] = c[i] * d[0]
        pp1[i] = c[i] * d[1]
        pp2[i] = c[i] * d[2]
        pp3[i] = c[i] * d[3]
        
        
    clm[0]  = pp0[0]
    clm[1]  = pp0[1] + pp1[0]
    clm[2]  = pp0[2] + pp1[1] + pp2[0]
    clm[3]  = pp0[3] + pp1[2] + pp2[1] + pp3[0]
    clm[4]  = pp0[4] + pp1[3] + pp2[2] + pp3[1]
    clm[5]  = pp0[5] + pp1[4] + pp2[3] + pp3[2]
    clm[6]  = pp0[6] + pp1[5] + pp2[4] + pp3[3]
    clm[7]  = pp0[7] + pp1[6] + pp2[5] + pp3[4]
    clm[8]  =          pp1[7] + pp2[6] + pp3[5]
    clm[9]  =                   pp2[7] + pp3[6]
    clm[10] =                            pp3[7]
    
    for j in range(11):
        cry[j] = clm[j] // 2
        bit[j] = clm[j] %  2
        result = result + bit[j]*(2**j) + cry[j]*(2**(j+1))
    
    return result

def GroupX2(c,d):
    pp0 = [0,0,0,0,0,0,0,0]
    pp1 = [0,0,0,0,0,0,0,0]
    pp2 = [0,0,0,0,0,0,0,0]
    pp3 = [0,0,0,0,0,0,0,0]
    
    clm = [0,0,0,0,0,0,0,0,0,0,0]
    cry = [0,0,0,0,0,0,0,0,0,0,0]
    bit = [0,0,0,0,0,0,0,0,0,0,0]
    result = 0
    
    for i in range(8):
        pp0[i] = c[i] * d[0]
        pp1[i] = c[i] * d[1]
        pp2[i] = c[i] * d[2]
        pp3[i] = c[i] * d[3]
        
        
    clm[0]  = pp0[0]
    clm[1]  = pp0[1] + pp1[0]
    clm[2]  = pp0[2] + pp1[1] + pp2[0]
    clm[3]  = pp0[3] + pp1[2] + pp2[1] + pp3[0]
    clm[4]  = pp0[4] + pp1[3] + pp2[2] + pp3[1]
    clm[5]  = pp0[5] + pp1[4] + pp2[3] + pp3[2]
    clm[6]  = pp0[6] + pp1[5] + pp2[4] + pp3[3]
    clm[7]  = pp0[7] + pp1[6] + pp2[5] + pp3[4]
    clm[8]  =          pp1[7] + pp2[6] + pp3[5]
    clm[9]  =                   pp2[7] + pp3[6]
    clm[10] =                            pp3[7]
    
    for j in range(11):
        cry[j] = clm[j] // 2
        bit[j] = clm[j] %  2
        if j<=1 :
            result = result + bit[j]*(2**j)
        else : 
            result = result + bit[j]*(2**j) + cry[j]*(2**(j+1))
    
    return result

def GroupX3(c,d):
    pp0 = [0,0,0,0,0,0,0,0]
    pp1 = [0,0,0,0,0,0,0,0]
    pp2 = [0,0,0,0,0,0,0,0]
    pp3 = [0,0,0,0,0,0,0,0]
    
    clm = [0,0,0,0,0,0,0,0,0,0,0]
    cry = [0,0,0,0,0,0,0,0,0,0,0]
    bit = [0,0,0,0,0,0,0,0,0,0,0]
    result = 0
    
    for i in range(8):
        pp0[i] = c[i] * d[0]
        pp1[i] = c[i] * d[1]
        pp2[i] = c[i] * d[2]
        pp3[i] = c[i] * d[3]
        
        
    clm[0]  = pp0[0]
    clm[1]  = pp0[1] + pp1[0]
    clm[2]  = pp0[2] + pp1[1] + pp2[0]
    clm[3]  = pp0[3] + pp1[2] + pp2[1] + pp3[0]
    clm[4]  = pp0[4] + pp1[3] + pp2[2] + pp3[1]
    clm[5]  = pp0[5] + pp1[4] + pp2[3] + pp3[2]
    clm[6]  = pp0[6] + pp1[5] + pp2[4] + pp3[3]
    clm[7]  = pp0[7] + pp1[6] + pp2[5] + pp3[4]
    clm[8]  =          pp1[7] + pp2[6] + pp3[5]
    clm[9]  =                   pp2[7] + pp3[6]
    clm[10] =                            pp3[7]
    
    for j in range(11):
        cry[j] = clm[j] // 2
        bit[j] = clm[j] %  2
        if j<=2 :
            result = result + bit[j]*(2**j)
        else : 
            result = result + bit[j]*(2**j) + cry[j]*(2**(j+1))
    
    return result

def GroupX4(c,d):
    pp0 = [0,0,0,0,0,0,0,0]
    pp1 = [0,0,0,0,0,0,0,0]
    pp2 = [0,0,0,0,0,0,0,0]
    pp3 = [0,0,0,0,0,0,0,0]
    
    clm = [0,0,0,0,0,0,0,0,0,0,0]
    cry = [0,0,0,0,0,0,0,0,0,0,0]
    bit = [0,0,0,0,0,0,0,0,0,0,0]
    result = 0
    
    for i in range(8):
        pp0[i] = c[i] * d[0]
        pp1[i] = c[i] * d[1]
        pp2[i] = c[i] * d[2]
        pp3[i] = c[i] * d[3]
        
        
    clm[0]  = pp0[0]
    clm[1]  = pp0[1] + pp1[0]
    clm[2]  = pp0[2] + pp1[1] + pp2[0]
    clm[3]  = pp0[3] + pp1[2] + pp2[1] + pp3[0]
    clm[4]  = pp0[4] + pp1[3] + pp2[2] + pp3[1]
    clm[5]  = pp0[5] + pp1[4] + pp2[3] + pp3[2]
    clm[6]  = pp0[6] + pp1[5] + pp2[4] + pp3[3]
    clm[7]  = pp0[7] + pp1[6] + pp2[5] + pp3[4]
    clm[8]  =          pp1[7] + pp2[6] + pp3[5]
    clm[9]  =                   pp2[7] + pp3[6]
    clm[10] =                            pp3[7]
    
    for j in range(11):
        cry[j] = clm[j] // 2
        bit[j] = clm[j] %  2
        if j<=3 :
            result = result + bit[j]*(2**j)
        else : 
            result = result + bit[j]*(2**j) + cry[j]*(2**(j+1))
    
    return result

def GroupX5(c,d):
    pp0 = [0,0,0,0,0,0,0,0]
    pp1 = [0,0,0,0,0,0,0,0]
    pp2 = [0,0,0,0,0,0,0,0]
    pp3 = [0,0,0,0,0,0,0,0]
    
    clm = [0,0,0,0,0,0,0,0,0,0,0]
    cry = [0,0,0,0,0,0,0,0,0,0,0]
    bit = [0,0,0,0,0,0,0,0,0,0,0]
    result = 0
    
    for i in range(8):
        pp0[i] = c[i] * d[0]
        pp1[i] = c[i] * d[1]
        pp2[i] = c[i] * d[2]
        pp3[i] = c[i] * d[3]
        
        
    clm[0]  = pp0[0]
    clm[1]  = pp0[1] + pp1[0]
    clm[2]  = pp0[2] + pp1[1] + pp2[0]
    clm[3]  = pp0[3] + pp1[2] + pp2[1] + pp3[0]
    clm[4]  = pp0[4] + pp1[3] + pp2[2] + pp3[1]
    clm[5]  = pp0[5] + pp1[4] + pp2[3] + pp3[2]
    clm[6]  = pp0[6] + pp1[5] + pp2[4] + pp3[3]
    clm[7]  = pp0[7] + pp1[6] + pp2[5] + pp3[4]
    clm[8]  =          pp1[7] + pp2[6] + pp3[5]
    clm[9]  =                   pp2[7] + pp3[6]
    clm[10] =                            pp3[7]
    
    for j in range(11):
        cry[j] = clm[j] // 2
        bit[j] = clm[j] %  2
        if j<=4 :
            result = result + bit[j]*(2**j)
        else : 
            result = result + bit[j]*(2**j) + cry[j]*(2**(j+1))
    
    return result

def GroupX6(c,d):
    pp0 = [0,0,0,0,0,0,0,0]
    pp1 = [0,0,0,0,0,0,0,0]
    pp2 = [0,0,0,0,0,0,0,0]
    pp3 = [0,0,0,0,0,0,0,0]
    
    clm = [0,0,0,0,0,0,0,0,0,0,0]
    cry = [0,0,0,0,0,0,0,0,0,0,0]
    bit = [0,0,0,0,0,0,0,0,0,0,0]
    result = 0
    
    for i in range(8):
        pp0[i] = c[i] * d[0]
        pp1[i] = c[i] * d[1]
        pp2[i] = c[i] * d[2]
        pp3[i] = c[i] * d[3]
        
        
    clm[0]  = pp0[0]
    clm[1]  = pp0[1] + pp1[0]
    clm[2]  = pp0[2] + pp1[1] + pp2[0]
    clm[3]  = pp0[3] + pp1[2] + pp2[1] + pp3[0]
    clm[4]  = pp0[4] + pp1[3] + pp2[2] + pp3[1]
    clm[5]  = pp0[5] + pp1[4] + pp2[3] + pp3[2]
    clm[6]  = pp0[6] + pp1[5] + pp2[4] + pp3[3]
    clm[7]  = pp0[7] + pp1[6] + pp2[5] + pp3[4]
    clm[8]  =          pp1[7] + pp2[6] + pp3[5]
    clm[9]  =                   pp2[7] + pp3[6]
    clm[10] =                            pp3[7]
    
    for j in range(11):
        cry[j] = clm[j] // 2
        bit[j] = clm[j] %  2
        if j<=5 :
            result = result + bit[j]*(2**j)
        else : 
            result = result + bit[j]*(2**j) + cry[j]*(2**(j+1))
    
    return result

def GroupX7(c,d):
    pp0 = [0,0,0,0,0,0,0,0]
    pp1 = [0,0,0,0,0,0,0,0]
    pp2 = [0,0,0,0,0,0,0,0]
    pp3 = [0,0,0,0,0,0,0,0]
    
    clm = [0,0,0,0,0,0,0,0,0,0,0]
    cry = [0,0,0,0,0,0,0,0,0,0,0]
    bit = [0,0,0,0,0,0,0,0,0,0,0]
    result = 0
    
    for i in range(8):
        pp0[i] = c[i] * d[0]
        pp1[i] = c[i] * d[1]
        pp2[i] = c[i] * d[2]
        pp3[i] = c[i] * d[3]
        
        
    clm[0]  = pp0[0]
    clm[1]  = pp0[1] + pp1[0]
    clm[2]  = pp0[2] + pp1[1] + pp2[0]
    clm[3]  = pp0[3] + pp1[2] + pp2[1] + pp3[0]
    clm[4]  = pp0[4] + pp1[3] + pp2[2] + pp3[1]
    clm[5]  = pp0[5] + pp1[4] + pp2[3] + pp3[2]
    clm[6]  = pp0[6] + pp1[5] + pp2[4] + pp3[3]
    clm[7]  = pp0[7] + pp1[6] + pp2[5] + pp3[4]
    clm[8]  =          pp1[7] + pp2[6] + pp3[5]
    clm[9]  =                   pp2[7] + pp3[6]
    clm[10] =                            pp3[7]
    
    for j in range(11):
        cry[j] = clm[j] // 2
        bit[j] = clm[j] %  2
        if j<=6 :
            result = result + bit[j]*(2**j)
        else : 
            result = result + bit[j]*(2**j) + cry[j]*(2**(j+1))
    
    return result

def GroupX8(c,d):
    pp0 = [0,0,0,0,0,0,0,0]
    pp1 = [0,0,0,0,0,0,0,0]
    pp2 = [0,0,0,0,0,0,0,0]
    pp3 = [0,0,0,0,0,0,0,0]
    
    clm = [0,0,0,0,0,0,0,0,0,0,0]
    cry = [0,0,0,0,0,0,0,0,0,0,0]
    bit = [0,0,0,0,0,0,0,0,0,0,0]
    result = 0
    
    for i in range(8):
        pp0[i] = c[i] * d[0]
        pp1[i] = c[i] * d[1]
        pp2[i] = c[i] * d[2]
        pp3[i] = c[i] * d[3]
        
        
    clm[0]  = pp0[0]
    clm[1]  = pp0[1] + pp1[0]
    clm[2]  = pp0[2] + pp1[1] + pp2[0]
    clm[3]  = pp0[3] + pp1[2] + pp2[1] + pp3[0]
    clm[4]  = pp0[4] + pp1[3] + pp2[2] + pp3[1]
    clm[5]  = pp0[5] + pp1[4] + pp2[3] + pp3[2]
    clm[6]  = pp0[6] + pp1[5] + pp2[4] + pp3[3]
    clm[7]  = pp0[7] + pp1[6] + pp2[5] + pp3[4]
    clm[8]  =          pp1[7] + pp2[6] + pp3[5]
    clm[9]  =                   pp2[7] + pp3[6]
    clm[10] =                            pp3[7]
    
    for j in range(11):
        cry[j] = clm[j] // 2
        bit[j] = clm[j] %  2
        if j<=7 :
            result = result + bit[j]*(2**j)
        else : 
            result = result + bit[j]*(2**j) + cry[j]*(2**(j+1))
    
    return result

def GroupX9(c,d):
    pp0 = [0,0,0,0,0,0,0,0]
    pp1 = [0,0,0,0,0,0,0,0]
    pp2 = [0,0,0,0,0,0,0,0]
    pp3 = [0,0,0,0,0,0,0,0]
    
    clm = [0,0,0,0,0,0,0,0,0,0,0]
    cry = [0,0,0,0,0,0,0,0,0,0,0]
    bit = [0,0,0,0,0,0,0,0,0,0,0]
    result = 0
    
    for i in range(8):
        pp0[i] = c[i] * d[0]
        pp1[i] = c[i] * d[1]
        pp2[i] = c[i] * d[2]
        pp3[i] = c[i] * d[3]
        
        
    clm[0]  = pp0[0]
    clm[1]  = pp0[1] + pp1[0]
    clm[2]  = pp0[2] + pp1[1] + pp2[0]
    clm[3]  = pp0[3] + pp1[2] + pp2[1] + pp3[0]
    clm[4]  = pp0[4] + pp1[3] + pp2[2] + pp3[1]
    clm[5]  = pp0[5] + pp1[4] + pp2[3] + pp3[2]
    clm[6]  = pp0[6] + pp1[5] + pp2[4] + pp3[3]
    clm[7]  = pp0[7] + pp1[6] + pp2[5] + pp3[4]
    clm[8]  =          pp1[7] + pp2[6] + pp3[5]
    clm[9]  =                   pp2[7] + pp3[6]
    clm[10] =                            pp3[7]
    
    for j in range(11):
        cry[j] = clm[j] // 2
        bit[j] = clm[j] %  2
        if j<=8 :
            result = result + bit[j]*(2**j)
        else : 
            result = result + bit[j]*(2**j) + cry[j]*(2**(j+1))
    
    return result

def GroupX10(c,d):
    pp0 = [0,0,0,0,0,0,0,0]
    pp1 = [0,0,0,0,0,0,0,0]
    pp2 = [0,0,0,0,0,0,0,0]
    pp3 = [0,0,0,0,0,0,0,0]
    
    clm = [0,0,0,0,0,0,0,0,0,0,0]
    cry = [0,0,0,0,0,0,0,0,0,0,0]
    bit = [0,0,0,0,0,0,0,0,0,0,0]
    result = 0
    
    for i in range(8):
        pp0[i] = c[i] * d[0]
        pp1[i] = c[i] * d[1]
        pp2[i] = c[i] * d[2]
        pp3[i] = c[i] * d[3]
        
        
    clm[0]  = pp0[0]
    clm[1]  = pp0[1] + pp1[0]
    clm[2]  = pp0[2] + pp1[1] + pp2[0]
    clm[3]  = pp0[3] + pp1[2] + pp2[1] + pp3[0]
    clm[4]  = pp0[4] + pp1[3] + pp2[2] + pp3[1]
    clm[5]  = pp0[5] + pp1[4] + pp2[3] + pp3[2]
    clm[6]  = pp0[6] + pp1[5] + pp2[4] + pp3[3]
    clm[7]  = pp0[7] + pp1[6] + pp2[5] + pp3[4]
    clm[8]  =          pp1[7] + pp2[6] + pp3[5]
    clm[9]  =                   pp2[7] + pp3[6]
    clm[10] =                            pp3[7]
    
    for j in range(11):
        cry[j] = clm[j] // 2
        bit[j] = clm[j] %  2
        if j<=9 :
            result = result + bit[j]*(2**j)
        else : 
            result = result + bit[j]*(2**j) + cry[j]*(2**(j+1))
    
    return result


# In[11]:


########## CDM8_1x ##########

def CDM8_11(a,b) :
    c = int_to_bin(a)
    d = int_to_bin(b)
    
    resultA = 0
    resultB = 0
    
    resultA = Group(c,d[:4])
    resultB = Group(c,d[4:])
    
    return resultA + resultB*16

def CDM8_12(a,b) :
    c = int_to_bin(a)
    d = int_to_bin(b)
    
    resultA = 0
    resultB = 0
    
    resultA =   Group(c,d[:4])
    resultB = GroupX2(c,d[4:])
    
    return resultA + resultB*16

def CDM8_13(a,b) :
    c = int_to_bin(a)
    d = int_to_bin(b)
    
    resultA = 0
    resultB = 0
    
    resultA =   Group(c,d[:4])
    resultB = GroupX3(c,d[4:])
    
    return resultA + resultB*16

def CDM8_14(a,b) :
    c = int_to_bin(a)
    d = int_to_bin(b)
    
    resultA = 0
    resultB = 0
    
    resultA =   Group(c,d[:4])
    resultB = GroupX4(c,d[4:])
    
    return resultA + resultB*16

def CDM8_15(a,b) :
    c = int_to_bin(a)
    d = int_to_bin(b)
    
    resultA = 0
    resultB = 0
    
    resultA =   Group(c,d[:4])
    resultB = GroupX5(c,d[4:])
    
    return resultA + resultB*16

def CDM8_16(a,b) :
    c = int_to_bin(a)
    d = int_to_bin(b)
    
    resultA = 0
    resultB = 0
    
    resultA =   Group(c,d[:4])
    resultB = GroupX6(c,d[4:])
    
    return resultA + resultB*16

def CDM8_17(a,b) :
    c = int_to_bin(a)
    d = int_to_bin(b)
    
    resultA = 0
    resultB = 0
    
    resultA =   Group(c,d[:4])
    resultB = GroupX7(c,d[4:])
    
    return resultA + resultB*16

def CDM8_18(a,b) :
    c = int_to_bin(a)
    d = int_to_bin(b)
    
    resultA = 0
    resultB = 0
    
    resultA =   Group(c,d[:4])
    resultB = GroupX8(c,d[4:])
    
    return resultA + resultB*16

def CDM8_19(a,b) :
    c = int_to_bin(a)
    d = int_to_bin(b)
    
    resultA = 0
    resultB = 0
    
    resultA =   Group(c,d[:4])
    resultB = GroupX9(c,d[4:])
    
    return resultA + resultB*16

def CDM8_110(a,b) :
    c = int_to_bin(a)
    d = int_to_bin(b)
    
    resultA = 0
    resultB = 0
    
    resultA =   Group(c,d[:4])
    resultB = GroupX10(c,d[4:])
    
    return resultA + resultB*16

########## CDM8_2x ##########

def CDM8_21(a,b) :
    c = int_to_bin(a)
    d = int_to_bin(b)
    
    resultA = 0
    resultB = 0
    
    resultA = GroupX2(c,d[:4])
    resultB =   Group(c,d[4:])
    
    return resultA + resultB*16

def CDM8_22(a,b) :
    c = int_to_bin(a)
    d = int_to_bin(b)
    
    resultA = 0
    resultB = 0
    
    resultA = GroupX2(c,d[:4])
    resultB = GroupX2(c,d[4:])
    
    return resultA + resultB*16

def CDM8_23(a,b) :
    c = int_to_bin(a)
    d = int_to_bin(b)
    
    resultA = 0
    resultB = 0
    
    resultA = GroupX2(c,d[:4])
    resultB = GroupX3(c,d[4:])
    
    return resultA + resultB*16

def CDM8_24(a,b) :
    c = int_to_bin(a)
    d = int_to_bin(b)
    
    resultA = 0
    resultB = 0
    
    resultA = GroupX2(c,d[:4])
    resultB = GroupX4(c,d[4:])
    
    return resultA + resultB*16

def CDM8_25(a,b) :
    c = int_to_bin(a)
    d = int_to_bin(b)
    
    resultA = 0
    resultB = 0
    
    resultA = GroupX2(c,d[:4])
    resultB = GroupX5(c,d[4:])
    
    return resultA + resultB*16

def CDM8_26(a,b) :
    c = int_to_bin(a)
    d = int_to_bin(b)
    
    resultA = 0
    resultB = 0
    
    resultA = GroupX2(c,d[:4])
    resultB = GroupX6(c,d[4:])
    
    return resultA + resultB*16

def CDM8_27(a,b) :
    c = int_to_bin(a)
    d = int_to_bin(b)
    
    resultA = 0
    resultB = 0
    
    resultA = GroupX2(c,d[:4])
    resultB = GroupX7(c,d[4:])
    
    return resultA + resultB*16

def CDM8_28(a,b) :
    c = int_to_bin(a)
    d = int_to_bin(b)
    
    resultA = 0
    resultB = 0
    
    resultA = GroupX2(c,d[:4])
    resultB = GroupX8(c,d[4:])
    
    return resultA + resultB*16

def CDM8_29(a,b) :
    c = int_to_bin(a)
    d = int_to_bin(b)
    
    resultA = 0
    resultB = 0
    
    resultA = GroupX2(c,d[:4])
    resultB = GroupX9(c,d[4:])
    
    return resultA + resultB*16

def CDM8_210(a,b) :
    c = int_to_bin(a)
    d = int_to_bin(b)
    
    resultA = 0
    resultB = 0
    
    resultA =  GroupX2(c,d[:4])
    resultB = GroupX10(c,d[4:])
    
    return resultA + resultB*16


########## CDM8_3x ##########

def CDM8_31(a,b) :
    c = int_to_bin(a)
    d = int_to_bin(b)
    
    resultA = 0
    resultB = 0
    
    resultA = GroupX3(c,d[:4])
    resultB =   Group(c,d[4:])
    
    return resultA + resultB*16

def CDM8_32(a,b) :
    c = int_to_bin(a)
    d = int_to_bin(b)
    
    resultA = 0
    resultB = 0
    
    resultA = GroupX3(c,d[:4])
    resultB = GroupX2(c,d[4:])
    
    return resultA + resultB*16

def CDM8_33(a,b) :
    c = int_to_bin(a)
    d = int_to_bin(b)
    
    resultA = 0
    resultB = 0
    
    resultA = GroupX3(c,d[:4])
    resultB = GroupX3(c,d[4:])
    
    return resultA + resultB*16

def CDM8_34(a,b) :
    c = int_to_bin(a)
    d = int_to_bin(b)
    
    resultA = 0
    resultB = 0
    
    resultA = GroupX3(c,d[:4])
    resultB = GroupX4(c,d[4:])
    
    return resultA + resultB*16

def CDM8_35(a,b) :
    c = int_to_bin(a)
    d = int_to_bin(b)
    
    resultA = 0
    resultB = 0
    
    resultA = GroupX3(c,d[:4])
    resultB = GroupX5(c,d[4:])
    
    return resultA + resultB*16

def CDM8_36(a,b) :
    c = int_to_bin(a)
    d = int_to_bin(b)
    
    resultA = 0
    resultB = 0
    
    resultA = GroupX3(c,d[:4])
    resultB = GroupX6(c,d[4:])
    
    return resultA + resultB*16

def CDM8_37(a,b) :
    c = int_to_bin(a)
    d = int_to_bin(b)
    
    resultA = 0
    resultB = 0
    
    resultA = GroupX3(c,d[:4])
    resultB = GroupX7(c,d[4:])
    
    return resultA + resultB*16

def CDM8_38(a,b) :
    c = int_to_bin(a)
    d = int_to_bin(b)
    
    resultA = 0
    resultB = 0
    
    resultA = GroupX3(c,d[:4])
    resultB = GroupX8(c,d[4:])
    
    return resultA + resultB*16

def CDM8_39(a,b) :
    c = int_to_bin(a)
    d = int_to_bin(b)
    
    resultA = 0
    resultB = 0
    
    resultA = GroupX3(c,d[:4])
    resultB = GroupX9(c,d[4:])
    
    return resultA + resultB*16

def CDM8_310(a,b) :
    c = int_to_bin(a)
    d = int_to_bin(b)
    
    resultA = 0
    resultB = 0
    
    resultA = GroupX3(c,d[:4])
    resultB = GroupX10(c,d[4:])
    
    return resultA + resultB*16
    
########## CDM8_4x ##########

def CDM8_41(a,b) :
    c = int_to_bin(a)
    d = int_to_bin(b)

    
    resultA = 0
    resultB = 0
    
    resultA = GroupX4(c,d[:4])
    resultB =   Group(c,d[4:])
    
    return resultA + resultB*16
     
def CDM8_42(a,b) :
    c = int_to_bin(a)
    d = int_to_bin(b)

    
    resultA = 0
    resultB = 0
    
    resultA = GroupX4(c,d[:4])
    resultB = GroupX2(c,d[4:])
    
    return resultA + resultB*16
     
def CDM8_43(a,b) :
    c = int_to_bin(a)
    d = int_to_bin(b)

    
    resultA = 0
    resultB = 0
    
    resultA = GroupX4(c,d[:4])
    resultB = GroupX3(c,d[4:])
    
    return resultA + resultB*16
     
def CDM8_44(a,b) :
    c = int_to_bin(a)
    d = int_to_bin(b)

    
    resultA = 0
    resultB = 0
    
    resultA = GroupX4(c,d[:4])
    resultB = GroupX4(c,d[4:])
    
    return resultA + resultB*16
     
def CDM8_45(a,b) :
    c = int_to_bin(a)
    d = int_to_bin(b)

    
    resultA = 0
    resultB = 0
    
    resultA = GroupX4(c,d[:4])
    resultB = GroupX5(c,d[4:])
    
    return resultA + resultB*16
     
def CDM8_46(a,b) :
    c = int_to_bin(a)
    d = int_to_bin(b)

    
    resultA = 0
    resultB = 0
    
    resultA = GroupX4(c,d[:4])
    resultB = GroupX6(c,d[4:])
    
    return resultA + resultB*16
       
def CDM8_47(a,b) :
    c = int_to_bin(a)
    d = int_to_bin(b)

    
    resultA = 0
    resultB = 0
    
    resultA = GroupX4(c,d[:4])
    resultB = GroupX7(c,d[4:])
    
    return resultA + resultB*16
     
def CDM8_48(a,b) :
    c = int_to_bin(a)
    d = int_to_bin(b)

    
    resultA = 0
    resultB = 0
    
    resultA = GroupX4(c,d[:4])
    resultB = GroupX8(c,d[4:])
    
    return resultA + resultB*16
     
def CDM8_49(a,b) :
    c = int_to_bin(a)
    d = int_to_bin(b)

    
    resultA = 0
    resultB = 0
    
    resultA = GroupX4(c,d[:4])
    resultB = GroupX9(c,d[4:])
    
    return resultA + resultB*16
     
def CDM8_410(a,b) :
    c = int_to_bin(a)
    d = int_to_bin(b)

    
    resultA = 0
    resultB = 0
    
    resultA =  GroupX4(c,d[:4])
    resultB = GroupX10(c,d[4:])
    
    return resultA + resultB*16


########## CDM8_5x ##########

def CDM8_51(a,b) :
    c = int_to_bin(a)
    d = int_to_bin(b)
    
    resultA = 0
    resultB = 0
    
    resultA = GroupX5(c,d[:4])
    resultB = Group(c,d[4:])
    
    return resultA + resultB*16

def CDM8_52(a,b) :
    c = int_to_bin(a)
    d = int_to_bin(b)
    
    resultA = 0
    resultB = 0
    
    resultA = GroupX5(c,d[:4])
    resultB = GroupX2(c,d[4:])
    
    return resultA + resultB*16

def CDM8_53(a,b) :
    c = int_to_bin(a)
    d = int_to_bin(b)
    
    resultA = 0
    resultB = 0
    
    resultA = GroupX5(c,d[:4])
    resultB = GroupX3(c,d[4:])
    
    return resultA + resultB*16

def CDM8_54(a,b) :
    c = int_to_bin(a)
    d = int_to_bin(b)
    
    resultA = 0
    resultB = 0
    
    resultA = GroupX5(c,d[:4])
    resultB = GroupX4(c,d[4:])
    
    return resultA + resultB*16

def CDM8_55(a,b) :
    c = int_to_bin(a)
    d = int_to_bin(b)
    
    resultA = 0
    resultB = 0
    
    resultA = GroupX5(c,d[:4])
    resultB = GroupX5(c,d[4:])
    
    return resultA + resultB*16

def CDM8_56(a,b) :
    c = int_to_bin(a)
    d = int_to_bin(b)
    
    resultA = 0
    resultB = 0
    
    resultA = GroupX5(c,d[:4])
    resultB = GroupX6(c,d[4:])
    
    return resultA + resultB*16

def CDM8_57(a,b) :
    c = int_to_bin(a)
    d = int_to_bin(b)
    
    resultA = 0
    resultB = 0
    
    resultA = GroupX5(c,d[:4])
    resultB = GroupX7(c,d[4:])
    
    return resultA + resultB*16

def CDM8_58(a,b) :
    c = int_to_bin(a)
    d = int_to_bin(b)
    
    resultA = 0
    resultB = 0
    
    resultA = GroupX5(c,d[:4])
    resultB = GroupX8(c,d[4:])
    
    return resultA + resultB*16

def CDM8_59(a,b) :
    c = int_to_bin(a)
    d = int_to_bin(b)
    
    resultA = 0
    resultB = 0
    
    resultA = GroupX5(c,d[:4])
    resultB = GroupX9(c,d[4:])
    
    return resultA + resultB*16

def CDM8_510(a,b) :
    c = int_to_bin(a)
    d = int_to_bin(b)
    
    resultA = 0
    resultB = 0
    
    resultA =  GroupX5(c,d[:4])
    resultB = GroupX10(c,d[4:])
    
    return resultA + resultB*16


########## CDM8_6x ##########

def CDM8_61(a,b) :
    c = int_to_bin(a)
    d = int_to_bin(b)

    
    resultA = 0
    resultB = 0
    
    resultA = GroupX6(c,d[:4])
    resultB =   Group(c,d[4:])
    
    return resultA + resultB*16
     
def CDM8_62(a,b) :
    c = int_to_bin(a)
    d = int_to_bin(b)

    
    resultA = 0
    resultB = 0
    
    resultA = GroupX6(c,d[:4])
    resultB = GroupX2(c,d[4:])
    
    return resultA + resultB*16

def CDM8_63(a,b) :
    c = int_to_bin(a)
    d = int_to_bin(b)

    
    resultA = 0
    resultB = 0
    
    resultA = GroupX6(c,d[:4])
    resultB = GroupX3(c,d[4:])
    
    return resultA + resultB*16
    
def CDM8_64(a,b) :
    c = int_to_bin(a)
    d = int_to_bin(b)

    
    resultA = 0
    resultB = 0
    
    resultA = GroupX6(c,d[:4])
    resultB = GroupX4(c,d[4:])
    
    return resultA + resultB*16
     
def CDM8_65(a,b) :
    c = int_to_bin(a)
    d = int_to_bin(b)

    
    resultA = 0
    resultB = 0
    
    resultA = GroupX6(c,d[:4])
    resultB = GroupX5(c,d[4:])
    
    return resultA + resultB*16
     
def CDM8_66(a,b) :
    c = int_to_bin(a)
    d = int_to_bin(b)

    
    resultA = 0
    resultB = 0
    
    resultA = GroupX6(c,d[:4])
    resultB = GroupX6(c,d[4:])
    
    return resultA + resultB*16
     
def CDM8_67(a,b) :
    c = int_to_bin(a)
    d = int_to_bin(b)

    
    resultA = 0
    resultB = 0
    
    resultA = GroupX6(c,d[:4])
    resultB = GroupX7(c,d[4:])
    
    return resultA + resultB*16
     
def CDM8_68(a,b) :
    c = int_to_bin(a)
    d = int_to_bin(b)

    
    resultA = 0
    resultB = 0
    
    resultA = GroupX6(c,d[:4])
    resultB = GroupX8(c,d[4:])
    
    return resultA + resultB*16
     
def CDM8_69(a,b) :
    c = int_to_bin(a)
    d = int_to_bin(b)

    
    resultA = 0
    resultB = 0
    
    resultA = GroupX6(c,d[:4])
    resultB = GroupX9(c,d[4:])
    
    return resultA + resultB*16
     
def CDM8_610(a,b) :
    c = int_to_bin(a)
    d = int_to_bin(b)

    
    resultA = 0
    resultB = 0
    
    resultA = GroupX6(c,d[:4])
    resultB = GroupX10(c,d[4:])
    
    return resultA + resultB*16

########## CDM8_7x ##########

def CDM8_71(a,b) :
    c = int_to_bin(a)
    d = int_to_bin(b)

    
    resultA = 0
    resultB = 0
    
    resultA = GroupX7(c,d[:4])
    resultB =   Group(c,d[4:])
    
    return resultA + resultB*16
    
def CDM8_72(a,b) :
    c = int_to_bin(a)
    d = int_to_bin(b)

    
    resultA = 0
    resultB = 0
    
    resultA = GroupX7(c,d[:4])
    resultB = GroupX2(c,d[4:])
    
    return resultA + resultB*16
    
def CDM8_73(a,b) :
    c = int_to_bin(a)
    d = int_to_bin(b)

    
    resultA = 0
    resultB = 0
    
    resultA = GroupX7(c,d[:4])
    resultB = GroupX3(c,d[4:])
    
    return resultA + resultB*16
     
def CDM8_74(a,b) :
    c = int_to_bin(a)
    d = int_to_bin(b)

    
    resultA = 0
    resultB = 0
    
    resultA = GroupX7(c,d[:4])
    resultB = GroupX4(c,d[4:])
    
    return resultA + resultB*16

def CDM8_75(a,b) :
    c = int_to_bin(a)
    d = int_to_bin(b)

    
    resultA = 0
    resultB = 0
    
    resultA = GroupX7(c,d[:4])
    resultB = GroupX5(c,d[4:])
    
    return resultA + resultB*16
     
def CDM8_76(a,b) :
    c = int_to_bin(a)
    d = int_to_bin(b)

    
    resultA = 0
    resultB = 0
    
    resultA = GroupX7(c,d[:4])
    resultB = GroupX6(c,d[4:])
    
    return resultA + resultB*16
     
def CDM8_77(a,b) :
    c = int_to_bin(a)
    d = int_to_bin(b)

    
    resultA = 0
    resultB = 0
    
    resultA = GroupX7(c,d[:4])
    resultB = GroupX7(c,d[4:])
    
    return resultA + resultB*16
     
def CDM8_78(a,b) :
    c = int_to_bin(a)
    d = int_to_bin(b)

    
    resultA = 0
    resultB = 0
    
    resultA = GroupX7(c,d[:4])
    resultB = GroupX8(c,d[4:])
    
    return resultA + resultB*16
     
def CDM8_79(a,b) :
    c = int_to_bin(a)
    d = int_to_bin(b)

    
    resultA = 0
    resultB = 0
    
    resultA = GroupX7(c,d[:4])
    resultB = GroupX9(c,d[4:])
    
    return resultA + resultB*16
     
def CDM8_710(a,b) :
    c = int_to_bin(a)
    d = int_to_bin(b)

    
    resultA = 0
    resultB = 0
    
    resultA = GroupX7(c,d[:4])
    resultB = GroupX10(c,d[4:])
    
    return resultA + resultB*16

########## CDM8_8x ##########

def CDM8_81(a,b) :
    c = int_to_bin(a)
    d = int_to_bin(b)
    
    resultA = 0
    resultB = 0
    
    resultA = GroupX8(c,d[:4])
    resultB =   Group(c,d[4:])
    
    return resultA + resultB*16

def CDM8_82(a,b) :
    c = int_to_bin(a)
    d = int_to_bin(b)
    
    resultA = 0
    resultB = 0
    
    resultA = GroupX8(c,d[:4])
    resultB = GroupX2(c,d[4:])
    
    return resultA + resultB*16

def CDM8_83(a,b) :
    c = int_to_bin(a)
    d = int_to_bin(b)
    
    resultA = 0
    resultB = 0
    
    resultA = GroupX8(c,d[:4])
    resultB = GroupX3(c,d[4:])
    
    return resultA + resultB*16
 
def CDM8_84(a,b) :
    c = int_to_bin(a)
    d = int_to_bin(b)
    
    resultA = 0
    resultB = 0
    
    resultA = GroupX8(c,d[:4])
    resultB = GroupX4(c,d[4:])
    
    return resultA + resultB*16

def CDM8_85(a,b) :
    c = int_to_bin(a)
    d = int_to_bin(b)
    
    resultA = 0
    resultB = 0
    
    resultA = GroupX8(c,d[:4])
    resultB = GroupX5(c,d[4:])
    
    return resultA + resultB*16

def CDM8_86(a,b) :
    c = int_to_bin(a)
    d = int_to_bin(b)
    
    resultA = 0
    resultB = 0
    
    resultA = GroupX8(c,d[:4])
    resultB = GroupX6(c,d[4:])
    
    return resultA + resultB*16

def CDM8_87(a,b) :
    c = int_to_bin(a)
    d = int_to_bin(b)
    
    resultA = 0
    resultB = 0
    
    resultA = GroupX8(c,d[:4])
    resultB = GroupX7(c,d[4:])
    
    return resultA + resultB*16

def CDM8_88(a,b) :
    c = int_to_bin(a)
    d = int_to_bin(b)
    
    resultA = 0
    resultB = 0
    
    resultA = GroupX8(c,d[:4])
    resultB = GroupX8(c,d[4:])
    
    return resultA + resultB*16

def CDM8_89(a,b) :
    c = int_to_bin(a)
    d = int_to_bin(b)
    
    resultA = 0
    resultB = 0
    
    resultA = GroupX8(c,d[:4])
    resultB = GroupX9(c,d[4:])
    
    return resultA + resultB*16

def CDM8_810(a,b) :
    c = int_to_bin(a)
    d = int_to_bin(b)
    
    resultA = 0
    resultB = 0
    
    resultA = GroupX8(c,d[:4])
    resultB = GroupX10(c,d[4:])
    
    return resultA + resultB*16

########## CDM8_9x ##########

def CDM8_91(a,b) :
    c = int_to_bin(a)
    d = int_to_bin(b)
    
    resultA = 0
    resultB = 0
    
    resultA = GroupX9(c,d[:4])
    resultB =   Group(c,d[4:])
    
    return resultA + resultB*16

def CDM8_92(a,b) :
    c = int_to_bin(a)
    d = int_to_bin(b)
    
    resultA = 0
    resultB = 0
    
    resultA = GroupX9(c,d[:4])
    resultB = GroupX2(c,d[4:])
    
    return resultA + resultB*16

def CDM8_93(a,b) :
    c = int_to_bin(a)
    d = int_to_bin(b)
    
    resultA = 0
    resultB = 0
    
    resultA = GroupX9(c,d[:4])
    resultB = GroupX3(c,d[4:])
    
    return resultA + resultB*16

def CDM8_94(a,b) :
    c = int_to_bin(a)
    d = int_to_bin(b)
    
    resultA = 0
    resultB = 0
    
    resultA = GroupX9(c,d[:4])
    resultB = GroupX4(c,d[4:])
    
    return resultA + resultB*16

def CDM8_95(a,b) :
    c = int_to_bin(a)
    d = int_to_bin(b)
    
    resultA = 0
    resultB = 0
    
    resultA = GroupX9(c,d[:4])
    resultB = GroupX5(c,d[4:])
    
    return resultA + resultB*16

def CDM8_96(a,b) :
    c = int_to_bin(a)
    d = int_to_bin(b)
    
    resultA = 0
    resultB = 0
    
    resultA = GroupX9(c,d[:4])
    resultB = GroupX6(c,d[4:])
    
    return resultA + resultB*16

def CDM8_97(a,b) :
    c = int_to_bin(a)
    d = int_to_bin(b)
    
    resultA = 0
    resultB = 0
    
    resultA = GroupX9(c,d[:4])
    resultB = GroupX7(c,d[4:])
    
    return resultA + resultB*16

def CDM8_98(a,b) :
    c = int_to_bin(a)
    d = int_to_bin(b)
    
    resultA = 0
    resultB = 0
    
    resultA = GroupX9(c,d[:4])
    resultB = GroupX8(c,d[4:])
    
    return resultA + resultB*16

def CDM8_99(a,b) :
    c = int_to_bin(a)
    d = int_to_bin(b)
    
    resultA = 0
    resultB = 0
    
    resultA = GroupX9(c,d[:4])
    resultB = GroupX9(c,d[4:])
    
    return resultA + resultB*16

def CDM8_910(a,b) :
    c = int_to_bin(a)
    d = int_to_bin(b)
    
    resultA = 0
    resultB = 0
    
    resultA = GroupX9(c,d[:4])
    resultB = GroupX10(c,d[4:])
    
    return resultA + resultB*16

########## CDM8_10x ##########

def CDM8_101(a,b) :
    c = int_to_bin(a)
    d = int_to_bin(b)

    
    resultA = 0
    resultB = 0
    
    resultA = GroupX10(c,d[:4])
    resultB =    Group(c,d[4:])
    
    return resultA + resultB*16

def CDM8_102(a,b) :
    c = int_to_bin(a)
    d = int_to_bin(b)
    
    resultA = 0
    resultB = 0
    
    resultA = GroupX10(c,d[:4])
    resultB =  GroupX2(c,d[4:])
    
    return resultA + resultB*16

def CDM8_103(a,b) :
    c = int_to_bin(a)
    d = int_to_bin(b)
    
    resultA = 0
    resultB = 0
    
    resultA = GroupX10(c,d[:4])
    resultB =  GroupX3(c,d[4:])
    
    return resultA + resultB*16

def CDM8_104(a,b) :
    c = int_to_bin(a)
    d = int_to_bin(b)
    
    resultA = 0
    resultB = 0
    
    resultA = GroupX10(c,d[:4])
    resultB =  GroupX4(c,d[4:])
    
    return resultA + resultB*16

def CDM8_105(a,b) :
    c = int_to_bin(a)
    d = int_to_bin(b)
    
    resultA = 0
    resultB = 0
    
    resultA = GroupX10(c,d[:4])
    resultB =  GroupX5(c,d[4:])
    
    return resultA + resultB*16

def CDM8_106(a,b) :
    c = int_to_bin(a)
    d = int_to_bin(b)
    
    resultA = 0
    resultB = 0
    
    resultA = GroupX10(c,d[:4])
    resultB =  GroupX6(c,d[4:])
    
    return resultA + resultB*16

def CDM8_107(a,b) :
    c = int_to_bin(a)
    d = int_to_bin(b)
    
    resultA = 0
    resultB = 0
    
    resultA = GroupX10(c,d[:4])
    resultB =  GroupX7(c,d[4:])
    
    return resultA + resultB*16

def CDM8_108(a,b) :
    c = int_to_bin(a)
    d = int_to_bin(b)
    
    resultA = 0
    resultB = 0
    
    resultA = GroupX10(c,d[:4])
    resultB =  GroupX8(c,d[4:])
    
    return resultA + resultB*16

def CDM8_109(a,b) :
    c = int_to_bin(a)
    d = int_to_bin(b)
    
    resultA = 0
    resultB = 0
    
    resultA = GroupX10(c,d[:4])
    resultB =  GroupX9(c,d[4:])
    
    return resultA + resultB*16

def CDM8_1010(a,b) :
    c = int_to_bin(a)
    d = int_to_bin(b)
    
    resultA = 0
    resultB = 0
    
    resultA = GroupX10(c,d[:4])
    resultB = GroupX10(c,d[4:])
    
    return resultA + resultB*16


# In[12]:


def genes_list():
    gene_list = [CDM8_11,CDM8_12,CDM8_13,CDM8_14,CDM8_15,CDM8_16,CDM8_17,CDM8_18,CDM8_19,CDM8_21,CDM8_22,CDM8_23,CDM8_24,CDM8_25,CDM8_26,CDM8_27,CDM8_28,CDM8_29,CDM8_31,
CDM8_32,CDM8_33,CDM8_34,CDM8_35,CDM8_36,CDM8_37,CDM8_38,CDM8_39,CDM8_41,CDM8_42,CDM8_43,CDM8_44,CDM8_45,CDM8_46,CDM8_47,CDM8_48,CDM8_49,CDM8_51,CDM8_52,
CDM8_53,CDM8_54,CDM8_55,CDM8_56,CDM8_57,CDM8_58,CDM8_59,CDM8_61,CDM8_62,CDM8_63,CDM8_64,CDM8_65,CDM8_66,CDM8_67,CDM8_68,CDM8_69,CDM8_71,CDM8_72,CDM8_73,
CDM8_74,CDM8_75,CDM8_76,CDM8_77,CDM8_78,CDM8_79,CDM8_81,CDM8_82,CDM8_83,CDM8_84,CDM8_85,CDM8_86,CDM8_87,CDM8_88,CDM8_89,CDM8_91,CDM8_92,CDM8_93,CDM8_94
,CDM8_95,CDM8_96,CDM8_97,CDM8_98,CDM8_99,CDM8_101,CDM8_102,CDM8_103,CDM8_104,CDM8_105,CDM8_106,CDM8_107,CDM8_108,CDM8_109,CDM8_110,CDM8_210,CDM8_310,CDM8_410,
CDM8_510,CDM8_610,CDM8_710,CDM8_810,CDM8_910,CDM8_1010]
    #for i in range(0,28):
        #genes_list.append(genes_list[random.randint(0,99)])
    return gene_list


# In[13]:


#Approximate Integer Multiplier in which first 24 bits of product (output) are truncated. So, the product will have 24 bits.
def AIM_T23(a, b, chromosome, genes_list):
    AM=0
    AH=0
    BM=0
    BH=0
    for i in range(0,8):
        AM = AM + a[i+8]*(2**i)
        AH = AH + a[i+16]*(2**i)
        BM = BM + b[i+8]*(2**i)
        BH = BH + b[i+16]*(2**i)
        
    mul_7=genes_list[chromosome[0]](AH,BM)
    mul_8=genes_list[chromosome[1]](AM,BH)
    mul_9=genes_list[chromosome[2]](AH,BH)

    result = (mul_7 + mul_8) + (mul_9)*(2**8)
    return result


# In[14]:


def int_to_23bin(c) :
    binary = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    
    for i in range(22):
        binary[i] = c % 2
        c = c // 2
        if i==21 :
            binary[22] = c
    
    return binary
    
def int_to_24bin(c) :
    binary = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    
    for i in range(23):
        binary[i] = c % 2
        c = c // 2
        if i==22 :
            binary[23] = c
    
    return binary

def bin8_to_int(binary):
    integer=0
    for i in range(0,8): integer=integer+binary[i]*(2**i)
    return integer

def bin23_to_int(binary):
    integer=0
    for i in range(0,23): integer=integer+binary[i]*(2**i)
    return integer


# In[ ]:





# In[16]:


#Approximate Floating-Point Multiplier
def AFPM_D3(a,b,chromosome,genes_list):
    result=[]
    exponent=0
    a_expo = 0
    b_expo = 0
    if (a[31]==0 and b[31]==0) or (a[31]==1 and b[31]==1) : sign=0 
    else: sign=1
    
    result.append(sign)
    
    for i in range(0,8):
        a_expo = a_expo + a[i+23]
        b_expo = b_expo + b[i+23]
    
    if (a_expo==0 or b_expo==0):
        result = [0]*31 + result
        return result
    
    mul = AIM_T23(a[0:23]+[1],b[0:23]+[1],chromosome,genes_list)
    mul = int_to_24bin(mul)
    
    if mul[23]==0 :
        significand = [0] + mul[0:22]
    else:
        significand = mul[0:23]
        
    
    if (mul[23]==1): bias=126 
    else: bias=127
    
    for j in range(0,8): exponent = exponent + (a[j+23] + b[j+23])*(2**j) 

    exponent = int_to_bin(exponent - bias)

    result = significand + exponent + result
    return result


# In[15]:


def test_vector(N):
    Vector_A=[]
    Vector_B=[]
    for i in range(0,N):
        A=[]
        B=[]
        #Trailing Significand
        A.extend(int_to_23bin(random.randint(0,8388607))) #(2**23)-1=8388607
        B.extend(int_to_23bin(random.randint(0,8388607)))

        #Exponent:  0<Ex,Ey<255 & (127<Ex+Ey<382 or 126<Ex+Ey<381 (due to normalization) ---> to ensure everything is fine we selected 127<Ex+Ey<381)
        Ex = random.randint(1,254)
        if Ex>126: Ey = random.randint(1,380-Ex)
        else: Ey = random.randint(128-Ex,254)
        A.extend(int_to_bin(Ex))
        B.extend(int_to_bin(Ey))
        
        #Sign
        A.append(random.randint(0,1))
        B.append(random.randint(0,1))
        
        Vector_A.append(A)
        Vector_B.append(B)
    return Vector_A,Vector_B


# In[20]:


def fitness_D(chromosome,genes_list,test_vectorA,test_vectorB):
    
    SRED = 0
    MRED = 0
    for i in range(0,len(test_vectorA)):
        trailing_significandA = 0
        trailing_significandB = 0
        trailing_significandX = 0
        #Exact Multiplication Result
        #Decimal Representation of Trailing Significands
        for j in range(0,23):
            trailing_significandA = trailing_significandA + test_vectorA[i][j]*(2**(j-23))
            trailing_significandB = trailing_significandB + test_vectorB[i][j]*(2**(j-23))
            
        trailing_significandA = trailing_significandA + 1
        trailing_significandB = trailing_significandB + 1
        #Decimal Representation of Exponents
        exponentA = bin8_to_int(test_vectorA[i][23:31])
        exponentB = bin8_to_int(test_vectorB[i][23:31])

        E_mul = trailing_significandA*trailing_significandB*(2**(exponentA+exponentB-254))
        #Approximate Multipilcation Result
        A_mul_bin = AFPM_D3(test_vectorA[i],test_vectorB[i],chromosome,genes_list)
        #Decimal Representation of A_mul_Bin
        for k in range(0,23):
            trailing_significandX = trailing_significandX + A_mul_bin[k]*(2**(k-23))
        trailing_significandX = trailing_significandX + 1
        exponentX = bin8_to_int(A_mul_bin[23:31])
        A_mul = trailing_significandX*(2**(exponentX-127))

        #MRED
        SRED = SRED + abs(E_mul - A_mul)/(E_mul)
    MRED = SRED/len(test_vectorA)
    
    return MRED


# In[ ]:





# In[ ]:





# In[ ]:





# In[60]:


Fitness_list_N=[]
Fitness_list_D=[]
test_chorom = [59, 59, 0]
g=genes_list()
A,B=test_vector(2**16)
for i in tqdm(range(0,1)):
    #Fitness_list_N.append(fitness_N(test_chorom,g,A,B))
    Fitness_list_D.append(fitness_D(test_chorom,g,A,B))
#print(Fitness_list_N)
print(Fitness_list_D)


# In[58]:


g[99]

