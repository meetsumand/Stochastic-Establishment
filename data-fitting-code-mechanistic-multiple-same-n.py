import math
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 10})

import scipy.stats as ss
from scipy.stats import gamma 
from scipy.stats import gengamma

from scipy.integrate import quad
from scipy.integrate import fixed_quad
from scipy.special import gamma as fgamma
from scipy.stats import lognorm

from lmfit import minimize, Parameters, fit_report
import numpy as np

x_data_l0=np.array([0,0.0094999997,0.0142499991,0.0189999994,0.0237499997,0.0284999982,0.0332500003])
y_data_l0=np.array([0.967187643,0.996647418,0.982452631,0.949892342,0.42024222,0.0519746765,0.0025355187])
y_error_l0=np.array([0.0056553003,0.012330926,0.0008599359,0.0281999763,0.0423363,0.0097903274,0.0018090389])

x_data_l1=np.array([0,0.133000001,0.151999995,0.171000004,0.189999998,0.208999991,0.227999985])
y_data_l1=np.array([0.967922688,0.960352242,0.932593703,0.598617673,0.196154401,0.00995919481,0.0024743937])
y_error_l1=np.array([0.0263140108,0.0576331057,0.0791998953,0.0615205318,0.0246722251,0.0032367217,0.0015969626])


x_data_l2=np.array([0.0,0.911999941,1.06400001,1.21599996,1.36800003,1.51999998,1.67199993])
y_data_l2=np.array([0.989274204,0.971233368,0.904235661,0.16872558,0.0217255764,0.00164771825,0.0])
y_error_l2=np.array([0.0152943796,0.0341845341,0.0723776892,0.023543071,0.006790475,0.0023516114,0.0023516114]) # last error is 0


x_data_l3=np.array([0,1.13999999,1.36800003,1.59599996,1.82399988,2.05200005,2.27999997])
y_data_l3=np.array([0.972665429,0.951728344,0.941148341,0.924678087,0.634757638,0.321902901,0.0671594217])
y_error_l3=np.array([0.0263049826,0.0746038109,0.0768063441,0.0786067471,0.0557164066,0.0279576164,0.0150615787])

x_data_a0=np.array([0,0.0149999997,0.0199999996,0.0250000004,0.0299999993])
y_data_a0=np.array([1.0,0.846153855,0.761726081,0.353345841,0.071294561])
y_error_a0=np.array([.01,0.0439290293,0.0310991649,0.0488240272,0.0222661234])

x_data_a1=np.array([0,0.119999997,0.140000001,0.159999996,0.180000007,0.200000003,0.219999999,0.239999995])
y_data_a1=np.array([1,0.865641713,0.796122968,0.685160398,0.469251335,0.458556145,0.205213904,0.11831551])
y_error_a1=np.array([.01,0.0449974351,0.0425847545,0.0403186157,0.0349153019,0.0251294244,0.0181487836,0.0201263372])

x_data_a2=np.array([0,0.959999979,1.12,1.27999997,1.44000006,1.60000002,1.75999999])
y_data_a2=np.array([1,0.918523669,0.848885775,0.635793865,0.155292481,0.0111420611,0.000696378818])
y_error_a2=np.array([.01,0.0378336385,0.0500035547,0.0832411274,0.0230997559,0.0029601646,0.0006515035])

x_data_a3=np.array([0,1.44000006,1.60000002,1.75999999,1.91999996,2.07999992,2.24000001])
y_data_a3=np.array([1,0.684105933,0.560264885,0.327152312,0.123841062,0.0450331122,0.00463576149])
y_error_a3=np.array([.01,0.020180339,0.0457979776,0.040478982,0.0191284567,0.0118449656,0.0014652368])


xscl1=x_data_l0[-1]
xscl2=x_data_l1[-1]
xscl3=x_data_l2[-1]
xscl4=x_data_l3[-1]

x1=x_data_l0/xscl1
y1=y_data_l0
y1e=y_error_l0#/.02

x2=x_data_l1/xscl2
y2=y_data_l1
y2e=y_error_l1#/.02

x3=x_data_l2/xscl3
y3=y_data_l2
y3e=y_error_l2#/.02

x4=x_data_l3/xscl4
y4=y_data_l3
y4e=y_error_l3#/.02



xsca1=x_data_a0[-1]
xsca2=x_data_a1[-1]
xsca3=x_data_a2[-1]
xsca4=x_data_a3[-1]

xa1=x_data_a0/xsca1
ya1=y_data_a0
ya1e=y_error_a0#/.02

xa2=x_data_a1/xsca2
ya2=y_data_a1
ya2e=y_error_a1#/.02

xa3=x_data_a2/xsca3
ya3=y_data_a2
ya3e=y_error_a2#/.02

xa4=x_data_a3/xsca4
ya4=y_data_a3
ya4e=y_error_a3#/.02

#y1e=1.
#y2e=1.

def integrand(x0,x,par1,par2,par3):
	a=1./(par2**2)
	scale=par1*(par2**2)
	k=1.
	#pd=math.exp(-k*((x/x0)**par3))
	pd=1./(1.+(x/x0)**par3)
	if pd>0.5:
		y= (2.-1./pd)*gamma.pdf(x0,a,0.,scale)#(1/t**k)*(x0**(k-1))*exp(-x0/t)/math.gamma(k)
	else:
		y=0. 
#y=(1.-(x/x0)**par3)*math.exp(-x0/par2)/par2
	#print('y',y,'par3',par3,'par2',par2)
	return y

def modfunc(par1,par2,par3,x):
#        modval=(2-(1+(x/par1)**par2))*np.heaviside(par1-x,0.)
	modval=[0.0]*len(x)
	#print('y',y[0])
	for i in range(len(x)):
		#print(x[i])
		z=quad(integrand,x[i],np.inf,args=(x[i],par1,par2,par3),limit=1000)
		modval[i]=z[0]
	return modval



def resid(params):
	model1=modfunc(params['mean1'].value,params['covar1'].value,params['n'].value,x1)
	model2=modfunc(params['mean2'].value,params['covar2'].value,params['n'].value,x2)
	model3=modfunc(params['mean3'].value,params['covar3'].value,params['n'].value,x3)
	model4=modfunc(params['mean4'].value,params['covar4'].value,params['n'].value,x4)

	modela1=modfunc(params['meana1'].value,params['covara1'].value,params['n'].value,xa1)
	modela2=modfunc(params['meana2'].value,params['covara2'].value,params['n'].value,xa2)
	modela3=modfunc(params['meana3'].value,params['covara3'].value,params['n'].value,xa3)
	modela4=modfunc(params['meana4'].value,params['covara4'].value,params['n'].value,xa4)

	resid1=(y1-model1)/y1e
	resid2=(y2-model2)/y2e
	resid3=(y3-model3)/y3e
	resid4=(y4-model4)/y4e

	resida1=(ya1-modela1)/ya1e
	resida2=(ya2-modela2)/ya2e
	resida3=(ya3-modela3)/ya3e
	resida4=(ya4-modela4)/ya4e


	#return resid1
	return np.concatenate((resid1, resid2, resid3, resid4, resida1, resida2, resida3, resida4 ))
  

params = Parameters()
params.add('mean1', value=.8,min=0.)
params.add('covar1', value=.1, min=0.)
params.add('mean2', value=.8,min=0.)
params.add('covar2', value=.1,min=0.)
params.add('mean3', value=.8,min=0.)
params.add('covar3', value=.1,min=0.)
params.add('mean4', value=.8,min=0.)
params.add('covar4', value=.1,min=0.)
params.add('n', value=4., min=0.)


params.add('meana1', value=.8,min=0.)
params.add('covara1', value=.1, min=0.)
params.add('meana2', value=.8,min=0.)
params.add('covara2', value=.1,min=0.)
params.add('meana3', value=.8,min=0.)
params.add('covara3', value=.1,min=0.)
params.add('meana4', value=.8,min=0.)
params.add('covara4', value=.1,min=0.)
#params.add('na', value=4., min=0.)

#params.add('n2', value=1.5, min=0.)


out = minimize(resid, params)

#print('xx',out.params['par1'].value
print(fit_report(out))



x01=out.params['mean1'].value
x02=out.params['mean2'].value
x03=out.params['mean3'].value
x04=out.params['mean4'].value

#print(params)
xval1=np.linspace(0.25,x1[-1]*1.25,num=100)
xval2=np.linspace(0.25,x2[-1]*1.25,num=100)
xval3=np.linspace(0.25,x3[-1]*1.25,num=100)
xval4=np.linspace(0.25,x4[-1]*1.25,num=100)

#fnc1=[0.0]*len(xval1)
#fnc2=[0.0]*len(xval2)

#plpar=params['par1']
#print('xx',out.params['par1':value])

fnc1=(modfunc(out.params['mean1'].value,out.params['covar1'].value,out.params['n'].value,xval1))
fnc2=(modfunc(out.params['mean2'].value,out.params['covar2'].value,out.params['n'].value,xval2))
fnc3=(modfunc(out.params['mean3'].value,out.params['covar3'].value,out.params['n'].value,xval3))
fnc4=(modfunc(out.params['mean4'].value,out.params['covar4'].value,out.params['n'].value,xval4))

fnca1=(modfunc(out.params['meana1'].value,out.params['covara1'].value,out.params['n'].value,xval1))
fnca2=(modfunc(out.params['meana2'].value,out.params['covara2'].value,out.params['n'].value,xval2))
fnca3=(modfunc(out.params['meana3'].value,out.params['covara3'].value,out.params['n'].value,xval3))
fnca4=(modfunc(out.params['meana4'].value,out.params['covara4'].value,out.params['n'].value,xval4))
#fnac4=(modfunc(.78,out.params['covar4'].value,out.params['n'].value,xval4))
#fnc1=(modfunc(out.params['mean1'].value,out.params['covar1'].value,out.params['n'].value,xval))
#fnc2=(modfunc(out.params['mean1'].value,out.params['covar1'].value,out.params['n2'].value,xval))

plt.figure()
#plt.suptitle('all \n joint fit with common Hill power n')



nf=out.params['n'].value
mean1f=out.params['mean1'].value
covar1f=out.params['covar1'].value
mean2f=out.params['mean2'].value
covar2f=out.params['covar2'].value
mean3f=out.params['mean3'].value
covar3f=out.params['covar3'].value
mean4f=out.params['mean4'].value
covar4f=out.params['covar4'].value


naf=out.params['n'].value
mean1af=out.params['meana1'].value
covar1af=out.params['covara1'].value
mean2af=out.params['meana2'].value
covar2af=out.params['covara2'].value
mean3af=out.params['meana3'].value
covar3af=out.params['covara3'].value
mean4af=out.params['meana4'].value
covar4af=out.params['covara4'].value



print 'cov liquid', covar1f,covar2f,covar3f,covar4f
print 'cov agar', covar1af,covar2af,covar3af,covar4af




#plt.text(.001,.5,'n={:.2f}'.format(nf))
#plt.text(.0,.2,' n={:.3f}, \n liq:  x0 avg {:.4f}, covar {:.4f}, \n agar: x0 avg {:.4f}, covar {:.4f}'.format(nf,avg1f,std1f,avg2f,std2f))
#plt.text(.0,.2,' liq: n {:.2f}, \n x0 avg {:.4f}, covar {:.4f} \n agar: n {:.2f}, \n x0 avg {:.4f}, covar {:.4f}'.format(nf,mean1f,covar1f,n2f,mean2f,covar2f))
#plt.text(.1,.3,' chi-squared {:.3f}\n reduced chi-squared {:.3f}'.format(out.chisqr,out.redchi))
#plt.text(.0,.2,' x0 avg {:.4f}, covar {:.4f}, \n liq: n {:.2f}, agar: n {:.2f}'.format(avg1f,std1f,nf,n2f))
#plt.text(.1,.1,'liquid n={:.2f}'.format(out.params['n'].value),fontsize=12)
#plt.text(.1,.1,' n={:.2f}'.format(out.params['n'].value),fontsize=12)


mksz=8
y2e[0]=0.
plt.errorbar(x1[1:]/mean1f,y1[1:],yerr=y1e[1:],fmt='go',capsize=3.0,capthick=1.0,markersize=mksz,label='Liquid Anc')
plt.errorbar(x2[1:]/mean2f,y2[1:],yerr=y2e[1:],fmt='bo',capsize=3.0,capthick=1.0,markersize=mksz,label='Liquid Sm')
plt.errorbar(x3[1:]/mean3f,y3[1:],yerr=y3e[1:],fmt='ro',capsize=3.0,capthick=1.0,markersize=mksz,label='Liquid Dm')
plt.errorbar(x4[1:]/mean4f,y4[1:],yerr=y4e[1:],fmt='ko',capsize=3.0,capthick=1.0,markersize=mksz,label='Liquid Tm')


xshf=0.5

plt.errorbar(xshf+xa1[1:]/mean1af,ya1[1:],yerr=ya1e[1:],fmt='gs',capsize=3.0,capthick=1.0,mfc='white',markersize=mksz,label='Agar Anc')
plt.errorbar(xshf+xa2[1:]/mean2af,ya2[1:],yerr=ya2e[1:],fmt='bs',capsize=3.0,capthick=1.0,mfc='white',markersize=mksz,label='Agar Sm')
plt.errorbar(xshf+xa3[1:]/mean3af,ya3[1:],yerr=ya3e[1:],fmt='rs',capsize=3.0,capthick=1.0,mfc='white',markersize=mksz,label='Agar Dm')
plt.errorbar(xshf+xa4[1:]/mean4af,ya4[1:],yerr=ya4e[1:],fmt='ks',capsize=3.0,capthick=1.0,mfc='white',markersize=mksz,label='Agar Tm')

plt.plot(xval1/mean1f,fnc1,'g--')
plt.plot(xval2/mean2f,fnc2,'b--')
plt.plot(xval3/mean3f,fnc3,'r--')
plt.plot(xval4/mean4f,fnc4,'k--')



plt.plot(xshf+xval1/mean1af,fnca1,'g-')
plt.plot(xshf+xval2/mean2af,fnca2,'b-')
plt.plot(xshf+xval3/mean3af,fnca3,'r-')
plt.plot(xshf+xval4/mean4af,fnca4,'k-')

plt.xlabel('Rescaled (and shifted) $x$',fontsize=15)
plt.ylabel('Establishment probability $p_e$',fontsize=15)

plt.legend()

#plt.show()

plt.savefig("fig-mechanistic-Hill-all-same-n",format='eps')

print('liquid 0')

for i in range(len(xval1)):
	print xval1[i]*xscl1,'     ',fnc1[i]


print('liquid 1')

for i in range(len(xval1)):
	print xval2[i]*xscl2,'     ',fnc2[i] 


print('liquid 2')

for i in range(len(xval1)):
	print xval3[i]*xscl3,'     ',fnc3[i]


print('liquid 3')

for i in range(len(xval1)):
	print xval4[i]*xscl4,'     ',fnc4[i]


print(' ')

print(' ')

print('agar 0')

for i in range(len(xval1)):
	print xval1[i]*xsca1,'     ',fnca1[i]

print('agar 1')

for i in range(len(xval1)):
	print xval2[i]*xsca2,'     ',fnca2[i]

print('agar 2')

for i in range(len(xval1)):
	print xval3[i]*xsca3,'     ',fnca3[i]


print('agar 3')
for i in range(len(xval1)):
	print xval4[i]*xsca4,'     ',fnca4[i]



