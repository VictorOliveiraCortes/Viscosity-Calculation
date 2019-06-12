#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = "Victor Côrtes and Mariana Hoyer Moreira"
__credits__ = ["Victor Côrtes", "Mariana Hoyer Moreira"]
__version__ = "1.0.1"
__date__ = "2019-06-12"
__maintainer__ = "Victor Côrtes"
__email__ = "victoroliveiracortes@gmail.com"
__status__ = "Prototype"

if __name__ == '__main__':
    print('Calculating Viscosity...')
    print('Importing Packages...')



import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simps
from scipy.integrate import odeint
from scipy.optimize import curve_fit


def get_xs_ys_averages(coordinates,trajs,file_name_std):
    x_averages, y_averages = [], []
    n = 0

    print('Extracting data from trajectories...')
    for i in trajs:
        print('\nTrajectorie {}:'.format(i))
        xs, ys = [], []
        for coor in coordinates:
            print('\tcoordinates',coor)
            x, y = [], []
            try:
                if (type(i)==int or type(i)==float) and i <10:
                    i = '0{}'.format(i)
            except:
                pass

            

            with open(file_name_std.format(coor,i)) as f:
                n += 1
                for line in f:
                    cols = line.split()

                    if len(cols) == 2:
                        try:
                            x.append(float(cols[0]))
                            y.append(float(cols[1]))
                        except ValueError as e:
                            continue

                xs.append(np.array(x))
                ys.append(np.array(y))

        xs = np.array(xs)
        ys = np.array(ys)

        x_averages.append(xs[0])
        y_averages.append(np.average(ys,axis=0))

    x_averages = (np.array(x_averages))*1e-12
    y_averages = (np.array(y_averages))*1e+10
    
    return np.array([x_averages, y_averages])

def green_kubo(V,kb,T,xs,ys,n=1000):
    N = [] 
    for i in range(1,len(xs),n):
        I = simps(ys[:i],xs[:i])
        
        n = V/(kb*T) * I
        n = n*1e+3 #convertendo de Pa.s para cP
        N.append(n)
    
    return np.array(N)

def visc_list(x_averages,y_averages,V,kb,T,trajs,n=1000):
    Ns = [] # viscosity list
    n=1000

    if len(x_averages) != len(y_averages):
        raise Exception('len(x_averages) != len(y_averages)')        
    
    for i in range(len(trajs)):
        ni = green_kubo(V,kb,T,x_averages[i],y_averages[i],n)
        Ns.append(ni)

    return np.array(Ns)
    
def time_cut(Ns_average, st_des, tol_std_perc=0.4):
    foundcutoff = False
    time_cut = 1
    while not foundcutoff and time_cut<len(Ns_average):
        if st_des[time_cut] > tol_std_perc*Ns_average[time_cut]:
            foundcutoff = True
        else:
            time_cut += 1
            
    return time_cut
    
def fig_visc_by_eq_meth(ts,Ns,n=1000,title='Viscosities by the Equilibrium Method',figsize=(10,5),
                           label_i='Trajectorie {}',label_mean='Average of Trajectories'):

    plt.figure(figsize=figsize)
    
    for i in range(len(Ns)):
        plt.plot(ts*1e+9,Ns[i],label=label_i.format(i+1))

    
    Ns_average = np.average(Ns,axis=0)
    np.average(Ns,axis=0)
    plt.plot(ts*1e+9,Ns_average, 'black',label=label_mean)
    plt.title(title)
    plt.xlabel('Time(ns)')
    #plt.ylabel('\u03C1 (cP)')
    plt.ylabel('\u03B7 (cP)')
    plt.legend()
    title=clear_title(title)
    plt.savefig(title+'.png')
    
def visc_standard_deviation_percentage(Ns):
    st_des = np.std(Ns,axis=0)
    Ns_average = np.average(Ns,axis=0)
    return st_des[1:]/Ns_average[1:]*100 

def fig_standard_deviation(ts,Ns,title='Standard Deviation',figsize=(10,5)):
    plt.figure(figsize=figsize)
    st_des = np.std(Ns,axis=0)
    plt.plot(ts[1:]*1e+9,st_des[1:])
    plt.xlabel('Time(ns)')
    plt.ylabel('(%)')
    plt.title(title)

    title=clear_title(title)
    plt.savefig(title+'.png')


def fig_standard_deviation_percentage(ts,Ns,title='Percentage Standard Deviation',figsize=(10,5)):
    plt.figure(figsize=figsize)
    st_des_perc = visc_standard_deviation_percentage(Ns)
    plt.plot(ts[1:]*1e+9,st_des_perc)
    plt.xlabel('Time(ns)')
    plt.ylabel('(%)')
    plt.title(title)

    title=clear_title(title)
    plt.savefig(title+'.png')


def fig_visc_average_with_standard_deviation(ts,Ns,title='Average Viscosity with Standard Deviation',figsize=(10,5)):
    plt.figure(figsize=figsize)
    Ns_average = np.average(Ns,axis=0)
    st_des = np.std(Ns,axis=0)
    plt.plot(ts*1e+9,Ns_average,'black')
    plt.fill_between(ts*1e+9,Ns_average - st_des, Ns_average + st_des,color='0.85')
    plt.title(title)
    plt.xlabel('Time(ns)')
    plt.ylabel('\u03B7 (cP)')

    title=clear_title(title)
    plt.savefig(title+'.png')

def fit_desvio(t,A,b):
    return A*t**b

def fig_curve_fitting_standard_deviation(ts,st_des,A,b,label_real_std='Real Standard Deviation',
    label_fit_std='Fitted Standard Deviation',title='Curve Fitting of Standard Deviation',figsize=(10,5)):

    
    plt.figure(figsize=figsize)
    plt.plot(ts*1e+9,st_des,label=label_real_std)
    plt.plot(ts*1e+9,fit_desvio(ts*1e+9,A,b),label=label_fit_std)

    plt.title(title)
    plt.legend()
    plt.xlabel('Time(ns)')
    plt.ylabel('Standard Deviation (cP)')
    title=clear_title(title)
    plt.savefig(title+'.png')

def doubexp(x,A,alpha,tau1, tau2):
    return A*alpha*tau1*(1-np.exp(-x/tau1))+A*(1-alpha)*tau2*(1-np.exp(-x/tau2))

def visc_end(popt3):
    return popt3[0]*popt3[1]*popt3[2]+popt3[0]*(1-popt3[1])*popt3[3]

def fig_visc_fit_convergence(ts,visc,popt3,visc_final,cut,title='Convergence of Fitted Viscosity',label_fit_visc='Fitting Viscosity',figsize=(10,5)):
    plt.figure(figsize=figsize)
    plt.plot(ts[:cut]*1e+9,doubexp(ts[:cut]*1e+9,*popt3),label=label_fit_visc,color='orange')
    plt.plot(ts[:cut]*1e+9,visc_final*np.ones_like(visc[:cut]),label='Viscosity Final = {0:.5f}'.format(visc_final) + ' cP',
             color='g')
    plt.title(title)
    plt.legend()
    plt.xlabel('Time(ns)')
    plt.ylabel('\u03B7 (cP)')
    title=clear_title(title)
    plt.savefig(title+'.png')

def fig_visc_convergence(ts,visc,popt3,visc_final,cut,title='Convergence of Viscosity: NaC12C10 = 1.0',
    label_real='Real Viscosity',label_fit_visc='Fitting Viscosity',figsize=(10,5)):

    plt.figure(figsize=(10,5))
    plt.plot(ts[:cut]*1e+9,visc[:cut],'o',label=label_real)
    plt.plot(ts[:cut]*1e+9,doubexp(ts[:cut]*1e+9,*popt3),label=label_fit_visc,color='orange')
    plt.plot(ts[:cut]*1e+9,visc_final*np.ones_like(visc[:cut]),label='Viscosity Final = {0:.5f}'.format(visc_final) + ' cP',
             color='g')
    plt.title(title)
    plt.xlabel('Time(ns)')
    plt.ylabel('\u03B7 (cP)')
    plt.legend()
    title=clear_title(title)
    plt.savefig(title+'.png')

def fig_visc_convergence_with_trajs(ts,visc,Ns,visc_final,cut,popt3,trajs,title='Convergence of Viscosity with Trajectories: NaC12C10 = 1.0',
    figsize=(18,10)):

    plt.rcParams.update({'font.size': 18})
    plt.rcParams['xtick.labelsize'] = 20
    plt.rcParams['ytick.labelsize'] = 20
    plt.figure(figsize=figsize)
    plt.grid()
    

    Ns_average = np.average(Ns,axis=0)
    st_des=np.std(Ns,axis=0)
    
    #plt.plot(ts[:cut]*1e+9,visc[:cut],'o',label='Real Viscosity')
    plt.plot(ts[:cut]*1e+9,doubexp(ts[:cut]*1e+9,*popt3),label='Adjusted Viscosity')
    plt.plot(ts[:cut]*1e+9,visc_final*np.ones_like(visc[:cut]),label='Calculated Viscosity = {:.1f} cP'.format(visc_final))

    for i in range(len(Ns)):
        plt.plot(ts[:cut]*1e+9,Ns[i][:cut],label='Trajectorie {}'.format(trajs[i]))

    plt.plot(ts[:cut]*1e+9,Ns_average[:cut], 'black',label='Average of Trajectories')
    plt.fill_between(ts[:cut]*1e+9, Ns_average[:cut] - st_des[:cut], Ns_average[:cut] + st_des[:cut], color='0.85')

    plt.title(title)
    plt.legend(prop={'size':16})
    plt.xlabel('Time(ns)')
    plt.ylabel('\u03B7 (cP)')
    title=clear_title(title)
    plt.savefig(title+'.png')

    plt.rcParams.update(plt.rcParamsDefault)

def clear_title(title):
    prohibited_characters = ['\\','/',':','*','?','"','<','>','|']

    for pc in prohibited_characters:
        title = title.replace(pc,'_')

    return title

def main():

    coordinates = ['xx','xy','xz','yy','yz','zz']
    trajs = [1,3,5,6,7,8,9] #range(1,10)
    file_name_std = "DES_autocorr_p{}_{}.xvg"
    V,kb,T = 169*1e-27, 1.380648*1e-23, 313.15
    n=1000

    
    x_averages, y_averages = get_xs_ys_averages(coordinates,trajs,file_name_std)
    ts=x_averages[0][1::n]
    print('\nCalculating Viscositys of Trajectories')
    Ns=visc_list(x_averages,y_averages,V,kb,T,trajs,n)
    Ns_average = np.average(Ns,axis=0)
    st_des=np.std(Ns,axis=0)
    cut = time_cut(Ns_average, st_des, tol_std_perc=0.4)

    print('fig_visc_by_eq_meth(ts,Ns,n)')
    fig_visc_by_eq_meth(ts,Ns,n)
    print('fig_standard_deviation(ts,Ns)')
    fig_standard_deviation(ts,Ns)
    print('fig_standard_deviation_percentage(ts,Ns)')
    fig_standard_deviation_percentage(ts,Ns)
    print('fig_visc_average_with_standard_deviation(ts,Ns)')
    fig_visc_average_with_standard_deviation(ts,Ns)
    # Find the parameters of the standard displacement adjustment using the curve_fit
    res = curve_fit(fit_desvio,ts*1e+9,st_des)
    par = res[0]
    A, b = par
    print('fig_curve_fitting_standard_deviation(ts,st_des,A,b)')
    fig_curve_fitting_standard_deviation(ts,st_des,A,b)
    
    visc = Ns_average

    popt3,pcov2 = curve_fit(doubexp, ts[1:cut]*1e+9,visc[1:cut],
        sigma=st_des[1:cut],bounds=(0,[np.inf,1,np.inf,np.inf]),maxfev=1000000)

    visc_final=visc_end(popt3)
    print('\nviscosity: {0:.5f}'.format(visc_final) + ' cP\n')

    print('fig_visc_fit_convergence')
    fig_visc_fit_convergence(ts,visc,popt3,visc_final,cut)
    print('fig_visc_convergence')
    fig_visc_convergence(ts,visc,popt3,visc_final,cut)
    print('fig_visc_convergence_with_trajs')
    fig_visc_convergence_with_trajs(ts,visc,Ns,visc_final,cut,popt3,trajs)


if __name__ == "__main__":
    main()