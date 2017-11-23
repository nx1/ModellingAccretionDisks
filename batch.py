# -*- coding: utf-8 -*-





import itertools

N_l = [1,5,10,30]
const_l = [1.1,1.5]
Q_factor_l = [0.005,0.025,0.1]
tMax_factor_l = [1.1,10.0]
H_R_l = [0.1,1.0]
M_0_start_l = [1.0,10.0]


biglist = list(itertools.product(N_l, const_l, Q_factor_l, tMax_factor_l, H_R_l, M_0_start_l))

for i in range(77):
    print i
    print biglist[i]
    N = biglist[i][0]
    const = biglist[i][1]
    Q_factor = biglist[i][2]
    tMax_factor = biglist[i][3]
    H_R_= biglist[i][4]
    M_0_start = biglist[i][5]
    
    name0 = N_l.index(N)
    name1 = const_l.index(const)
    name2 = Q_factor_l.index(Q_factor)
    name3 = tMax_factor_l.index(tMax_factor)
    name4 = H_R_l.index(H_R_)
    name5 = M_0_start_l.index(M_0_start)
    
    
    
    #LC
    fig1.savefig('Plot1.eps', format='eps', dpi=1000, bbox_inches='tight', pad_inches = 0.0)
    fig1.savefig('Plot1.png', format='png', dpi=1000, bbox_inches='tight', pad_inches = 0.0)
    #RMS
    fig2.savefig('Plot1.eps', format='eps', dpi=1000, bbox_inches='tight', pad_inches = 0.0)
    fig2.savefig('Plot1.png', format='png', dpi=1000, bbox_inches='tight', pad_inches = 0.0)
    #PSD
    fig3.savefig('Plot1.eps', format='eps', dpi=1000, bbox_inches='tight', pad_inches = 0.0)
    fig3.savefig('Plot1.png', format='png', dpi=1000, bbox_inches='tight', pad_inches = 0.0)
    
    NAME = '%s%s%s%s%s%s.png' % (name0,name1,name2,name3,name4,name5)
    
    
    print NAME
    