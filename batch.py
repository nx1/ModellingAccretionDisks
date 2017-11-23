# -*- coding: utf-8 -*-





import itertools

N_l = [1,5,10,30]
const_l = [1.1,1.5]
Q_factor_l = [0.005,0.025,0.1]
tMax_factor_l = [1.1,1.4]
H_R_l = [0.1,1.0]
M_0_start_l = [1.0,10.0]


biglist = list(itertools.product(N_l, const_l, Q_factor_l, tMax_factor_l, H_R_l, M_0_start_l))

for i in range(178,191):
    print i
    print biglist[i]
    N = biglist[i][0]
    const = biglist[i][1]
    Q_factor = biglist[i][2]
    tMax_factor = biglist[i][3]
    H_R_= biglist[i][4]
    M_0_start = biglist[i][5]
    
    execfile('discfunc.py')
    
    name0 = N_l.index(N)
    name1 = const_l.index(const)
    name2 = Q_factor_l.index(Q_factor)
    name3 = tMax_factor_l.index(tMax_factor)
    name4 = H_R_l.index(H_R_)
    name5 = M_0_start_l.index(M_0_start)
    
    
    NAMEpng = '%s%s%s%s%s%s.png' % (name0,name1,name2,name3,name4,name5)
    NAMEeps = '%s%s%s%s%s%s.eps' % (name0,name1,name2,name3,name4,name5)
    #LC
    fig1.savefig('variable_graphs/LC/' + NAMEeps, format='eps', dpi=300, bbox_inches='tight', pad_inches = 0.0)
    fig1.savefig('variable_graphs/LC/' + NAMEpng, format='png', dpi=300, bbox_inches='tight', pad_inches = 0.0)
    #RMS
    fig2.savefig('variable_graphs/RMS/' + NAMEeps, format='eps', dpi=300, bbox_inches='tight', pad_inches = 0.0)
    fig2.savefig('variable_graphs/RMS/' + NAMEpng, format='png', dpi=300, bbox_inches='tight', pad_inches = 0.0)
    #PSD
    fig3.savefig('variable_graphs/PSD/' + NAMEeps, format='eps', dpi=300, bbox_inches='tight', pad_inches = 0.0)
    fig3.savefig('variable_graphs/PSD/' + NAMEpng, format='png', dpi=300, bbox_inches='tight', pad_inches = 0.0)
    print NAMEpng
    