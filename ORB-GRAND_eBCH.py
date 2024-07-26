# Constrained ORB-GRAND for eBCH ###########################################
#
# Copyright (c) 2022, Mohammad Rowshan and Jinhong Yuan
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without modification, 
# are permitted provided that:
# the source code retains the above copyright notice, and te redistribtuion condition.
# 
# Freely distributed for educational and research purposes
#######################################################################################

import numpy as np
import sympy as sp
import copy
from channel import channel
from rate_profile import rateprofile
import polar_coding_functions as pcf
#import GaloisField
from GaloisField import X, degree
import csv
import math

n = 7 
N = 2**n-1
t = 3 
q = 2**n
p = [1,0,0,0,1,0,0,1]#[1,0,0,0,0,1,1] #primitive # [1,0,0,0,0,1,1] for GF(2^6) # [1,0,0,0,1,0,0,1] for GF(2^7) #Note: order of coefficients: [coeff_highest_deg,...,coeff_deg1,coeff_deg0]
K = 106 
snrb_snr = 'SNRb'   # 'SNRb':Eb/N0 or 'SNR':Es/N0
modu = 'BPSK'      
# The order of coefficeints is reversed (unline for p which was the default of MATLAB and literature). We do it this way because the binary rep of the poy rep of elements in the H matrix is also revered accoding to Lin's book
poly = [1,1,0,0,0,1,1,1,1,0,0,1,1,0,1,1,0,1,1,0,0,1] #for (127,106) by default prim_poly = 'D^7+D^3+1', MATLAB: [genpoly,t] = bchgenpoly(127,106) >> t=3

b = [10**2, 10**3, 10**4, 10**5]#, 10**6, 10**7]

snr_range = np.arange(3,6,0.5)
err_cnt = 100

constraints = 'NoC' # '2C' # '1C'  # 'NoC'
# 'NoC' : This is the conventional ORBGRAND with no constraints
# '1C' : for a single constraint based on overall parity (if the all-one row exists in the H matrix)
# '2C' : for two constraints based on row 0 and row 1 (h_set = [0,1]) of manipulated H matrix (see section/cell "parity check matrix")

sim_comment = 'eBCH, Constraints: ' + constraints  # comment on the setting of simulation



# %% Generator matrix
    
G_poly = np.zeros((K,N), dtype=np.int8)
for i in range(0,K):
    if i+len(poly)-1 > N-1:  # In this case, because it is not a square matrix, it doesn't happen
        j = len(poly) - (i+len(poly)-1 - (K-1))
    else:
        j = len(poly)
    G_poly[i][i:i+len(poly)] = poly[0:j]


#G = np.matmul(G_poly,G_N)%2   #In MATLAB: inv_G = inv(gf(G,1))
G = G_poly
#G_inv = np.linalg.inv(G)%2

# Extension of the G matrix
sum_G_rows = np.sum(G,axis=1) # All the same
if sum_G_rows[0]%2 == 1:
    c1 = np.ones((K,1), dtype=np.int8)
else:
    c1 = np.zeros((K,1), dtype=np.int8)
    
G = np.concatenate((c1, G),1)



# %% Parity check matrix

INF = float('inf') # infinity variable
#GF2 = GaloisField(p) # global GF(2) field

def degree(p):
    """Returns degree of polynomial (highest exponent).
    (slide 3)
    """
    poly = np.poly1d(np.flipud(p))
    return poly.order

def X(i):
    """Create single coefficient polynomial with degree i: X^i
    """
    X = np.zeros(i + 1,dtype=np.int8) # including degree 0
    X[i] = 1
    return X#.astype(int)

def constructGF(p):
    """Construct GF(2^m) based on primitive polynomial p.
    The degree of pi(X) is used to determine m.
    (slide 12)
    Args:
        p: primitive polynomial p to construct the GF with.
        verbose: print information on how the GF is constructed.
    Returns:
        Elements of the GF in polynomial representation.
    """
    elements = []
    m = degree(p)  # Degree of the polynomial

    if m == 1: # special simple case: GF(2)
        elements = [np.array([0]), np.array([1])]
        return elements

    a_high = p[1:m+1] #Except the last element # The highest degree of alpha = the rest of the terms. Ex: a^4=1+a for p = a^4+a+1. We ignore signs and even coeff-elements in the GF(2)

    for i in range(0, 2**m):
        # create exponential representation
        if i == 0:
            exp = np.array([0]) #np.zeros(m)
        else:
            exp = X(i-1)

        poly = exp
        if degree(poly) >= m:
            quotient, remainder = divmod(degree(poly), m)   # modulo m of degree(poly)

            poly = X(remainder)
            for j in range(0, quotient):
                #poly = np.pad(poly, (len(a_high) - poly.size-1, 0), 'constant', constant_values = 0)
                poly = np.polymul(np.flipud(poly), a_high)
                poly = np.flipud(poly)%2

            while degree(poly) >= m:
                poly = np.polyadd(np.flipud(poly), np.flipud(elements[degree(poly) + 1]))%2
                poly = np.flipud(poly)
                poly = poly[:degree(poly)] # Discard the last element (with hiest degree/power)

        # format polynomial (size m)
        poly = poly[:degree(poly) + 1]
        poly = np.pad(poly, (0,m - poly.size), 'constant', constant_values = 0)

        # append to elements list for return
        elements.append(poly) #.astype(int))


    return elements

def element(a,q):
    """Return element that is the same as element a but with an
    exponent within 0 and q-1.
    """
    if a == 0: # zero element doesn't have an exponent
        return int(a)
    exp_a = a - 1 # convert from integer representation to exponent
    exp_a = exp_a % (q - 1) # get exponent within 0 and q-1
    a = exp_a + 1 # convert back to integer representation
    return int(a)
def elementFromExp(exp_a,q):
    """Returns element in integer representation from given exponent
    representation. For the zero element an exponent of +-infinity is
    expected by definition.
    """
    #if exp_a == INF or exp_a == -INF: # zero element is special case
        #return 0
    exp_a = exp_a % (q - 1) # element with exponent within 0 and q-1
    a = exp_a + 1 # convert to integer representation # Index in the GF table
    return int(a)

GF_table = constructGF(p)

H = np.zeros(((N-K),N), dtype=np.int8)
h_i = 0
m = degree(p)
for row in range(1, 2*t,2): # 1, 3, ..., 2^t-1
    for col in range(0,N): 
        GF_idx = elementFromExp(row*col,q) # (a^ti)^ni
        H[h_i*m:(h_i+1)*m, col] = GF_table[GF_idx]
    h_i += 1

# Extension of the H matrix ###############################
r1 = np.ones((1,N), dtype=np.int8)
cNp1 = np.zeros((N-K+1,1), dtype=np.int8)
cNp1[0,0] = 1

H = np.concatenate((r1, H),0)
H = np.concatenate((cNp1,H),1)
N += 1
###########################################################
Ht = H
H = np.transpose(H)

GHt = np.matmul(G, H)%2 #H is actuall Ht here

#%% Constraint Matters


def supp_row(h):
    #bnry = [int(x) for x in list(bin(n).replace("0b", ""))] #'{0:0b}'.format(n)
    #bnry = [x for x in list(bin(n).replace("0b", ""))]
    #bnry.reverse()
    indices_of_1s = set()
    for x in range(len(h)):    #indices_of_1s = np.where(bnry == 1)
        if h[x]==1:
            indices_of_1s |= {x}
    return indices_of_1s

row_weights = np.sum(H, axis=0)
min_wt,min_idx = np.min(row_weights), np.argmin(row_weights)
max_wt,max_idx = np.max(row_weights), np.argmax(row_weights)

H_col_set = [0,1] # Column indices are ordered from the largest to smallest in terms of weight
intrvl_cnt = len(H_col_set)

H2 = np.zeros((N,intrvl_cnt), dtype=np.int8)
indx_set = [set() for _ in range(len(H_col_set))]
i = 0
for c in H_col_set:
    indx_set[i] = supp_row(Ht[c,:])
    i += 1

indx_list = [[] for _ in range(len(H_col_set))]
piX = []
indx_intrvl = []
i = 0
end = start = accum = 0
for c in H_col_set:
    if c != H_col_set[-1]:
        indx_list[i] = list(indx_set[i] - indx_set[i+1])
        H2[:,i] = (H[:,i]+H[:,i+1])%2
    else:
        indx_list[i] = list(indx_set[i])
        H2[:,i] = H[:,i]

    start = accum
    end = start + len(indx_list[i]) - 1
    accum += len(indx_list[i])
    indx_intrvl += [[start,end]]
    
    indx_list[i].sort()
    piX += indx_list[i]
    i += 1
indx_intrvl = np.array(indx_intrvl)
pi2 = np.zeros(N, dtype=np.int8)
for x in piX:
    pi2[x] = piX.index(x)

# %% Error Pattern Generation, 
# This section inlcudes obsouloite functions as well. See the simulator section of the code


def int_part(N): #Integer Partitioning
    S = list()
    S.append([N])
    incr = 0
    x0 = 0
    y0 = N
    while x0 < y0:
        incr += 1
        x = x0 = incr
        y = y0 = N - incr
        last = []
        while x < y:
            S.append([y,x]+last)
            last = S[-1][1:]
            #print(S[-1])
            x += 1
            y = y - x
    return S


def int_part_odd(i,e_is_odd,N): #It doesn't extract all the partitions
    S = list()
    T = list()
    S.append([i])
    if e_is_odd == 1 and i < N:
        T.append(S[-1])
    incr = 0
    x0 = 0
    y0 = i
    while x0 < y0: #Finding alternatives for each last integer pair
        incr += 1
        x = x0 = incr
        y = y0 = i - incr
        last = []
        while x < y: # Spliting the last integer
            S.append([y,x]+last)
            if e_is_odd == 1 and len(last)%2 == 1 and y < N:
                T.append([y,x]+last)
            elif e_is_odd == 0 and len(last)%2 == 0 and y < N:
                T.append([y,x]+last)
            last1 = copy.deepcopy(last)
            last = S[-1][1:]
            
            x1 = x
            y1 = y
            #incr1 = 0
            # Finding aletrnatives for thre last two integers
            while x1+1 < y1-1 and ((e_is_odd == 1 and len(last1)%2 == 1) or (e_is_odd == 0 and len(last1)%2 == 0)) and last1 != []:
                #incr1 += 1
                x1 += 1 # x + incr1
                y1 -= 1 # y - incr1
                S.append([y1,x1]+last1)
                if y1 < N:
                    T.append([y1,x1]+last1)

            #print(S[-1])
            x += 1
            y = y - x

    return T

        
def split_int(s,N):
    s_minus = s[1:]
    x = s[1] if len(s)>1 else 0
    y = s[0]
    x += 1
    y -= x
    if y > x:
        return True, [y,x]+s[1:]
    else:
        return False, []

def alt_int_pair(s):
    S = list()
    x,y = s[1],s[0]
    incr = 1
    x += incr
    y -= incr
    while x < y:
        S.append([y,x]+s[2:])
        #incr += 1
        x += incr
        y -= incr
    return S


def int_part_rec(i,N,pi):
    S = list()
    T = list()
    S.append([i])
    if i <= N:
        T.append([pi[S[-1][0]-1]])
    splittable = True
    SS = list()
    SS = copy.deepcopy(S)
    #if e_is_odd == 1:
        #T += SS
    s_len = 1
    while splittable: #Finding alternatives for each last integer pair
        SSS = list()
        s_len +=1
        for ss in SS:
            status, s = split_int(ss,N)
            
            if status:
                SSS += [s] # The splitted one
                SSS += alt_int_pair(s)
        if len(SSS)>0:
            S += SSS
            SS = copy.deepcopy(SSS)
            
            for ss in SSS:
               if ss[0]<=N:
                   tt = [0 for t in range(len(ss))]
                   for t in range(len(ss)):
                       tt[t] = pi[ss[t]-1]
                   tt.sort(reverse=True)
                   T += [tt]
        else:
            splittable = False
    
    return T

def int_part_odd_rec(i,e_is_odd,N,symp,pi):
    S = list()
    T = list()
    S.append([i])
    if e_is_odd == 1 and i <= N and pi[i-1]>=symp:
        T.append([pi[S[-1][0]-1]])
    splittable = True
    SS = list()
    SS = copy.deepcopy(S)
    #if e_is_odd == 1:
        #T += SS
    s_len = 1
    while splittable: #Finding alternatives for each last integer pair
        SSS = list()
        s_len +=1
        min_col = [0,5,8,13,16,21,24,29,32,37,40,45,48,53,56,61]
        for ss in SS:
            status, s = split_int(ss,N)
            
            if status:
                SSS += [s] # The splitted one
                SSS += alt_int_pair(s)
        if len(SSS)>0:
            S += SSS
            SS = copy.deepcopy(SSS)
            
            if e_is_odd == 1 and s_len%2 == 1:
                for ss in SSS:
                    if ss[0]<=N:
                        tt = [0 for t in range(len(ss))]
                        set1_size = 0
                        for t in range(len(ss)):
                            tt[t] = pi[ss[t]-1]
                            cnt_in_list = min_col.count(tt[t])
                            #if tt[t]>=symp and symp>0:
                            if cnt_in_list>0 and symp>0:
                                set1_size += 1
                        tt.sort(reverse=True)
                        if set1_size%2==1 and symp>0:
                            T += [tt]
                        elif set1_size%2==0 and symp==0:
                            T += [tt]
                            
                        """tt.sort(reverse=True)
                        if tt[0]>=symp: #existence of element in set1
                            T += [tt]"""
                        #T += [tt]
            elif e_is_odd == 0 and s_len%2 == 0:
                for ss in SSS:
                    if ss[0]<=N:
                        tt = [0 for t in range(len(ss))]
                        set1_size = 0
                        for t in range(len(ss)):
                            tt[t] = pi[ss[t]-1]
                            cnt_in_list = min_col.count(tt[t])
                            #if tt[t]>=symp and symp>0:
                            if cnt_in_list>0 and symp>0:
                                 set1_size += 1
                        tt.sort(reverse=True)
                        if set1_size%2==1 and symp>0:
                            T += [tt]
                        elif set1_size%2==0 and symp==0:
                            T += [tt]
                            
                        """tt.sort(reverse=True)
                        if tt[0]>=symp: #existence of element in set1
                            T += [tt]"""
                        #T += [tt]

        else:
            splittable = False
    
    return T

def pass_synd2(syndrome2, set_size):
    cond = True
    for r in range(len(syndrome2)):
        if set_size[r]%2==syndrome2[r]:
            cond = cond and True
        else:
            cond = False
    return cond


def int_part2(i,N,syndrome,syndrome2,indx_intrvl,rows_sel,pi,pi2):
    S = list()
    T = list()
    S.append([i])
    if i <= N:
        tt0 = pi[i-1]
        ttt0 = pi2[tt0]
        set_size = [0 for _ in rows_sel]
        cond = True
        for r in range(len(rows_sel)):
            if ttt0 >= indx_intrvl[r,0] and ttt0 <= indx_intrvl[r,1]:
                set_size[r] += 1
                break #Because ttto cannot be in more than one interval
            """if set_size[r]%2==syndrome2[r]:
                cond = cond and True
            else:
                cond = False"""
        #if set_size[0]%2==syndrome2[0] and set_size[1]%2==syndrome2[1]:# and set_size[2]%2==syndrome2[2]:
        if pass_synd2(syndrome2, set_size):
            T.append([pi[S[-1][0]-1]])
    splittable = True
    SS = list()
    SS = copy.deepcopy(S)
    #if e_is_odd == 1:
        #T += SS
    s_len = 1
    while splittable: #Finding alternatives for each last integer pair
        SSS = list()
        s_len +=1
        #min_col = [0,5,8,13,16,21,24,29,32,37,40,45,48,53,56,61]
        for ss in SS:
            status, s = split_int(ss,N)
            
            if status:
                SSS += [s] # The splitted one
                SSS += alt_int_pair(s)
        if len(SSS)>0:
            S += SSS
            SS = copy.deepcopy(SSS)
##############################
            for ss in SSS:
                if ss[0]<=N:
                    tt = [0 for t in range(len(ss))]
                    set_size = [0 for _ in rows_sel]
                    for t in range(len(ss)):
                        tt[t] = pi[ss[t]-1]
                        ttt = pi2[tt[t]]
                        for r in range(len(rows_sel)):
                            if ttt >= indx_intrvl[r,0] and ttt <= indx_intrvl[r,1]:
                                set_size[r] += 1
                    tt.sort(reverse=True)
                    #if set_size[0]%2==syndrome2[0] and set_size[1]%2==syndrome2[1]:# and set_size[2]%2==syndrome2[2]:
                    if pass_synd2(syndrome2, set_size):
                        T += [tt]
#########################################
        else:
            splittable = False
    
    return T


def check_synd(syndrome,e1):
    e0 = np.zeros(N, dtype=np.int8)
    syndrome0 = copy.deepcopy(syndrome)
    for j in e1:
        e0[j] = 1
        #syndrome0 = (syndrome0 + H[pi[j-1]][:])%2
        syndrome0 = (syndrome0 + H[j][:])%2
    if sum(syndrome0) == 0:
        return True,e0
    else:
        return False,e0

def int_part2_check(i,N,syndrome,syndrome2,indx_intrvl,rows_sel,pi,pi2,g,gT,b):
    S = list()
    T = list()
    S.append([i])
    cnt_S = g
    cnt_T = gT
    e0 = np.zeros(N, dtype=np.int8)
    if i <= N:
        tt0 = pi[i-1]
        ttt0 = pi2[tt0]
        set_size = [0 for _ in rows_sel]
        cond = True
        for r in range(len(rows_sel)):
            if ttt0 >= indx_intrvl[r,0] and ttt0 <= indx_intrvl[r,1]:
                set_size[r] += 1
                break #Because ttto cannot be in more than one interval
            """if set_size[r]%2==syndrome2[r]:
                cond = cond and True
            else:
                cond = False"""
        #if set_size[0]%2==syndrome2[0] and set_size[1]%2==syndrome2[1]:# and set_size[2]%2==syndrome2[2]:
        cnt_S += 1
        if pass_synd2(syndrome2, set_size):
            T.append([pi[S[-1][0]-1]])
            cnt_T += 1
            out,e0 = check_synd(syndrome,[pi[S[-1][0]-1]])
            if out:
                return True, e0, cnt_S, cnt_T
    splittable = True
    SS = list()
    SS = copy.deepcopy(S)
    #if e_is_odd == 1:
        #T += SS
    s_len = 1
    while splittable: #Finding alternatives for each last integer pair
        SSS = list()
        s_len +=1
        #min_col = [0,5,8,13,16,21,24,29,32,37,40,45,48,53,56,61]
        for ss in SS:
            status, s = split_int(ss,N)
            
            if status:
                SSS += [s] # The splitted one
                SSS += alt_int_pair(s)
        if len(SSS)>0:
            S += SSS
            SS = copy.deepcopy(SSS)
##############################
            for ss in SSS:
                if ss[0]<=N:
                    cnt_S += 1
                    tt = [0 for t in range(len(ss))]
                    set_size = [0 for _ in rows_sel]
                    for t in range(len(ss)):
                        tt[t] = pi[ss[t]-1]
                        ttt = pi2[tt[t]]
                        for r in range(len(rows_sel)):
                            if ttt >= indx_intrvl[r,0] and ttt <= indx_intrvl[r,1]:
                                set_size[r] += 1
                    tt.sort(reverse=True)
                    #if set_size[0]%2==syndrome2[0] and set_size[1]%2==syndrome2[1]:# and set_size[2]%2==syndrome2[2]:
                    if pass_synd2(syndrome2, set_size):
                        cnt_T += 1
                        T += [tt]
                        out,e0 = check_synd(syndrome,tt)
                        if out:
                            return True, e0, cnt_S, cnt_T
                    if cnt_S >= b:
                        return False, e0, cnt_S, cnt_T
                    #if cnt_S == 101982:
                        #cnt_S = 100000
#########################################
        else:
            splittable = False
    
    return False, e0, cnt_S, cnt_T







def split_int2(s,set_size,syndrome2,pi,pi2,indx_intrvl,N):
    #s_minus = s[1:]
    x = s[1] if len(s)>1 else 0
    y = s[0]
    x += 1
    y -= x
    set_size0 = copy.deepcopy(set_size)
    set_size1 = copy.deepcopy(set_size)
    if y > x:
        for r in range(len(syndrome2)):
            tt = pi2[pi[x-1]]
            if tt >= indx_intrvl[r,0] and tt <= indx_intrvl[r,1]:
                set_size1[r] += 1
                set_size0[r] += 1
            tt = pi2[pi[y-1]]
            if tt >= indx_intrvl[r,0] and tt <= indx_intrvl[r,1]:
                set_size1[r] += 1
        if pass_synd2(syndrome2, set_size1):
            return [True,True], [y,x]+s[1:], set_size0
        else:
            return [True,False], [y,x]+s[1:], set_size0
    else:
        return [False,False], [], []

def alt_int_pair2(s,set_size,syndrome2,pi,pi2,indx_intrvl):
    S = list()
    Z = list()
    T = list()
    x,y = s[1],s[0]
    incr = 1
    x += incr
    y -= incr
    ttt = []
    for ss in s[2:]:
        ttt.append(pi[ss-1])
    while x < y:
        S.append([y,x]+s[2:])
        #incr += 1
        set_size0 = copy.deepcopy(set_size)
        set_size1 = copy.deepcopy(set_size)
        for r in range(len(syndrome2)):
            tt = pi2[pi[x-1]]
            if tt >= indx_intrvl[r,0] and tt <= indx_intrvl[r,1]:
                set_size1[r] += 1
                set_size0[r] += 1
            tt = pi2[pi[y-1]]
            if tt >= indx_intrvl[r,0] and tt <= indx_intrvl[r,1]:
                set_size1[r] += 1
        Z.append(set_size0)
        if pass_synd2(syndrome2, set_size1):
            T.append([pi[y-1],pi[x-1]]+ttt)#s[2:])
        x += incr
        y -= incr
    return S,Z,T

def int_part2_eff(i,N,syndrome2,indx_intrvl,pi,pi2):
    S = list()
    Z = list() #The size of the set resulting from s set intersection with supp(h_j) set
    T = list()
    """class seq:
         def __init__(self, s, size):
             self.s = s
             self.size = size"""
    cnt = 0
    S.append([i])
    z = [0 for _ in syndrome2]
    if i <= N:
        t0 = pi2[pi[i-1]]
        for r in range(len(syndrome2)):
            if t0 >= indx_intrvl[r,0] and t0 <= indx_intrvl[r,1]:
                z[r] += 1
                break #Because ttt0 cannot be in more than one interval
        if pass_synd2(syndrome2, z):
            T.append([pi[S[-1][0]-1]])
    splittable = True
    SS = list()
    SS = copy.deepcopy(S)
    ZZ = copy.deepcopy(Z)
    #if e_is_odd == 1:
        #T += SS
    cnt += 1
    s_len = 1
    ZZ.append([0 for _ in syndrome2])
    while splittable: #Finding alternatives for each last integer pair
        SSS = list()
        ZZZ = list()
        ##TTT = list()
        s_len += 1 #Because every sequence s is split once in each cycle of this loop
        #min_col = [0,5,8,13,16,21,24,29,32,37,40,45,48,53,56,61]
        for ss in range(len(SS)):
            #z0 = copy.deepcopy(Z[ss])
            status, s, z = split_int2(SS[ss],ZZ[ss],syndrome2,pi,pi2,indx_intrvl,N)
            cnt += 1
            if status[1]:
                ttt = []
                for ss1 in s:
                    ttt.append(pi[ss1-1])
                T += [ttt]
            if status[0]:
                SSS += [s] # The split one
                ZZZ.append(z)
                SSS1, ZZZ1, T1 = alt_int_pair2(s,ZZ[ss],syndrome2,pi,pi2,indx_intrvl)
                cnt += len(SSS1)
                SSS += SSS1
                ZZZ += ZZZ1
                T += T1
        if len(SSS)>0:
            S += SSS #we may not need this
            SS = copy.deepcopy(SSS)
            ZZ = copy.deepcopy(ZZZ)
        else:
            splittable = False
    
    return T, cnt



def int_part2_odd(i,N,is_odd,pi):
    S = list()
    T = list()
    S.append([i])
    if i <= N and is_odd == 1:
            T.append([pi[S[-1][0]-1]])
    splittable = True
    SS = list()
    SS = copy.deepcopy(S)
    s_len = 1
    while splittable: #Finding alternatives for each last integer pair
        SSS = list()
        s_len +=1
        for ss in SS:
            status, s = split_int(ss,N)
            
            if status:
                SSS += [s] # The splitted one
                SSS += alt_int_pair(s)
        if len(SSS)>0:
            S += SSS
            SS = copy.deepcopy(SSS) # For the next cycle of the while-loop
##############################
            for ss in SSS:
                if ss[0]<=N and len(ss)%2 == is_odd:
                    tt = [0 for t in range(len(ss))]
                    for t in range(len(ss)):
                        tt[t] = pi[ss[t]-1]
                    tt.sort(reverse=True)
                    T += [tt]
#########################################
        else:
            splittable = False
    
    return T



# %% Simulator
class BERFER():
    def __init__(self): # structure that keeps results of BER and FER tests
        self.snr = list()
        self.ber = list()
        self.fer = list()
        self.cplx = list()
        self.cplx_bit = list()
result = BERFER()


print("({},{}) b={}".format(N, K, b))
print(sim_comment)
print("BER & BLER & QUERIES evaluation is started\n")

        
for snr in snr_range:
    print("SNR={} dB".format(snr))
    t = -1
    fer = np.zeros(len(b), dtype=int)
    ber = np.zeros(len(b), dtype=int)
    cplx = np.zeros(len(b), dtype=int)
    ch = channel(modu, snr, snrb_snr, (K / N))
    
    np.random.seed(1000)
    while fer[len(b)-1] < err_cnt: 
        t += 1
        d = np.random.randint(0, 2, size=K, dtype=np.int8)
        
        x0 = np.matmul(d,G)%2
        
        modulated_x = ch.modulate(x0)
        y = ch.add_noise(modulated_x)
        
        sllr = ch.calc_llr(y)
        llr = abs(ch.calc_llr(y))
        
        pi = np.argsort(llr)
        
        teta_y = ch.demodulate(y)
        
        # Additional info on the error pattern
        e_actual = np.array(((teta_y + x0)%2).tolist())
        e_wt = sum(e_actual)
        e_pos = list()
        e_pos_inv = list()
        for ei in range(N):
            if e_actual[ei] == 1:
                e_pos.append(ei)
                for ej in range(N):
                    if ei == pi[ej]:
                        e_pos_inv.append(ej+1)
        ##print("\n")
        ##print(e_wt,e_pos,e_pos_inv,sum(e_pos_inv))
# %%
        is_odd = 0
        e_loc_min = np.zeros(N-K, dtype=np.int8)
        e_loc_max = np.zeros(N-K, dtype=np.int8)
        
        g = 0
        cnt_T = 0
        z = np.zeros(N, dtype=np.int8)
        
        
        metric = np.sum((np.subtract(y, ch.modulate(np.add(teta_y, z)%2)))**2)
        metric0 = metric
        for j in e_pos:
            metric0 = metric0 - (np.subtract(y[j-1], 1-2*(np.add(teta_y[j-1], 0)%2)))**2 + (np.subtract(y[j-1], 1-2*(np.add(teta_y[j-1], 1)%2)))**2
        ##print("correct_metri c=",metric0)
        
        syndrome = np.matmul(np.add(teta_y, z), H) % 2 #H is actually H_transpose
        syndrome2 = np.matmul(np.add(teta_y, z), H2) % 2 #H is actually H_transpose
        ##print("synd=",syndrome)
        if sum(syndrome) == 0:
            e0 = z
        else:
            if syndrome[0] == 1:
                is_odd = 1
                
            intgr = 1
            sum_synd = 1
            e0 = np.zeros(N, dtype=np.int8)
            while g < b[len(b)-1] and sum_synd > 0:
                #Generate error sequences based on int partitions
                if constraints == '2C': # For two constraint based rwows 0 and 1 of     modified H matrix.
                    out,e0,g,cnt_T = int_part2_check(intgr,N,syndrome,syndrome2,indx_intrvl,H_col_set,pi,pi2,g,cnt_T,b[len(b)-1]) 
                    if out == True:
                        sum_synd = 0
                    #S = int_part2(intgr,N,syndrome,syndrome2,indx_intrvl,H_col_set,pi,pi2)
                    #S,cnt = int_part2_eff(intgr,N,syndrome2,indx_intrvl,pi,pi2)
                else:
                    if constraints == '1C':   # For single constraint based on overall parity (if the all-one row in the H matrix exists)
                        S = int_part2_odd(intgr,N,is_odd,pi)    
                    else:    # For no constreaint (NoC)
                        S = int_part_rec(intgr,N,pi)   
                    for k in range(len(S)):
                        e0 = np.zeros(N, dtype=np.int8)
                        e1 = S[k] #S[e_idx[k]]
                        #print(g,e1)#,S_metric[k])
                        g += 1
                        syndrome0 = copy.deepcopy(syndrome)
                        for j in e1:
                            e0[j] = 1
                            syndrome0 = (syndrome0 + H[j][:])%2
                        if sum(syndrome0) == 0:
                            sum_synd = 0
                            break
                        if g == b[len(b)-1]: 
                            break

                if g >= b[len(b)-1]: 
                    break
                intgr += 1
# %% Statistics 
        #print(g,cnt_T)
        c = np.add(teta_y, e0)%2
        for i in range(len(b)):
            # For the sake of comparability, the reference for counting (i.e., g) upto the upperbound (b) for constrained scheme is identical with the non-constrained one.
            # Otherwise the FER would change (improve). This way, we can keep the FER constant but reduce the number of checking withing each upperbound (b).
            if g <= b[i]: 
                if not np.array_equal(x0, c):
                    for j in range(0, len(b)):
                        fer[j] += 1
                        #ber[j] += pcf.fails(x0, c)
                    print("Error # {0} t={1}, BLER={2:0.2e} *********************".format(fer[len(b)-1],t, fer[len(b)-1]/(t+1)))
                else:
                    for j in range(0, i):
                        fer[j] += 1
                        #ber[j] += pcf.fails(x0, c)  # incorrect for < max b
                if constraints == 'NoC':
                    for j in range(0, i):
                        cplx[j] += b[j]
                    for j in range(i, len(b)):
                        cplx[j] += g
                else:
                    for j in range(0, i):
                        cplx[j] += b[j]
                    for j in range(i, len(b)):
                        cplx[j] += cnt_T
                            
                break
        #print(g,cplx)
        if t%2000==0:
            print("@ t={} BLER = {}, avg_Queries={}".format(t, fer/(t+1), cplx / (t+1)))
        #fer += not np.array_equal(message, decoded)
    result.snr.append(snr)
    #result.ber.append(ber / ((t + 1) * K))
    result.fer.append(fer / (t + 1))
    result.cplx.append(cplx / (t + 1))
    result.cplx_bit.append(cplx / (t + 1)/N)

    print("\n####################################################\n")
    print("({},{}) b={} ".format(N, K, b),end='')
    print(sim_comment)
    print("SNR={}".format(result.snr))
    print("BLER={}".format(result.fer))
    #print("BER={}".format(result.ber))
    print("Queries={}".format(result.cplx))
    print("Queries_bit={}".format(result.cplx_bit))
    print("\n####################################################\n")



