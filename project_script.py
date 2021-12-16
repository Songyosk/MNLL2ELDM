# -*- coding: utf-8 -*-
"""
Created on Wed Nov 23 10:18:12 2016

@author: Son-Gyo Jung (sgj14@ic.ac.uk) CID: 00948246
@Project B1: A Log Likelihood fit for extracting the D^0 lifetime
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy import interp
from scipy import special
from scipy import integrate
import math
import pylab 
import matplotlib
import matplotlib.cm as cm
import matplotlib.mlab as mlab
import matplotlib.ticker as mtick
from numpy.linalg import inv

################################################################################ Task 3.1
def data(dataset=10000):
    '''
    Task 3.1.1: Access the data
    To read 'lifetime.txt' file.
    '''
    data.time, data.sigma = [], []
    input_file = open('lifetime.txt', 'r')
    for line in input_file:
        pair = line.rstrip('\n').split(' ',1)
        data.time.append(float(pair[0]))
        data.sigma.append(float(pair[1]))
    
    data.time, data.sigma = data.time[:int(dataset)], data.sigma[:int(dataset)]
    

def histogram(bins=500):
    '''
    Task 3.1.2: Access the data
    To plot histogram of decay times using data from 'lifetime.txt'.
    '''
    data()
    #plt.clf()
    plt.hist(data.time, bins, normed=1)
    plt.title('Histogram of decay times ('+str(bins)+' bins)', fontsize = 22)
    plt.ylabel('Frequency', fontsize = 22) 
    plt.xlabel('time (ps)', fontsize = 22)
    matplotlib.rcParams.update({'font.size': 22})
    plt.grid(True)

################################################################################ Task 3.2

def fit(tau, sigma, time):
    '''
    Task 3.2.1: Likelihood function
    The fit function.
    '''
    return((1 / (2*tau)) * np.exp((sigma**(2) / (2*tau**(2))) - (time/tau)) * special.erfc(2**(-0.5) * (sigma/tau - time/sigma)))


def fitfunction(tau=0.4, sigma=0.1):
    '''
    Task 3.2.2: Fit function
    To plot the fit function; that is, the convolution of the theoretical decay function with a Gaussian of width sigma.
    '''
    data()
    function = [] 
    for i in range(len(data.time)):
        f = fit(tau, sigma, data.time[i])
        function.append(f)
    
    f_intergrate = lambda time: fit(tau, sigma, time)
    print('(Integral of fit, error) = ' +str(integrate.quad(f_intergrate,-5.,10.))) #limits of the integral are [-5, 10]
    
    plt.plot(data.time, function, '.', label = '$\\tau$ = '+str(tau)+', $\sigma$ = '+str(sigma))
    plt.legend(fontsize = 25)
    plt.title('Normalised histogram of $t$ with the fit function depicting the expected distribution', fontsize = 28)
    plt.xlabel('Decay time $t$ (ps)', fontsize = 28)
    matplotlib.rcParams.update({'font.size': 26})
    plt.ylabel('$f^m$($t$)', fontsize = 28)
    plt.grid(True)

    
def plotmanyfits(tau=0.4, sigma=0.2):
    '''
    Task 3.2.3: Fit function
    To plot the fit function using different values of the parameters: tau and sigma.
    '''
    stepsize = np.arange(0.1, 0.7, 0.1)
    
    fig = plt.figure()
    fig.add_subplot(211)
    histogram(500)
    for i in stepsize:
        fitfunction(i, sigma)
    
    fig.add_subplot(212)
    histogram(500)  
    for i in stepsize:
        fitfunction(tau, i)
        
    plt.show()
      
################################################################################## Task 3.3

def cosh_func(x, dataset):
    '''
    Task 3.3.1: Likelihood function
    Cosh function to test NLL().
    '''
    return(np.cosh(x))
           

def NLL(tau, dataset): 
    '''
    Task 3.3.2: Likelihood function
    To plot a graph of the NLL (Negative Log Likelihood) as a function of tau; subsquently to approximate the minimum.
    '''   
    data(dataset)
    #Plot NLL with various tau values defined by np.arange. #(tau=None, dataset=10000)
    if tau == None:
        tau_values = np.arange(0.05,3.,0.05)
        final_NLL_values = []        
        for value in tau_values:
            tau = value
            fitfunction_values = []
            NLL_values = [] 
            for i in range(len(data.time)):
                fitfunction_values.append(fit(tau, data.sigma[i], data.time[i]))
                NLL_values.append(-1*np.log(fitfunction_values[i]))
    
            final_NLL_values.append(np.sum(NLL_values))    
        
        plt.clf()
        plt.plot(tau_values, final_NLL_values, '.-', label = '$\\tau$')
        plt.grid(True)
        plt.title('Graph of NLL($\\tau$)', fontsize = 22)
        plt.xlabel('$\\tau$ (ps)', fontsize = 22)
        plt.ylabel('NLL($\\tau$) (ln($\\frac{1}{ps}$))', fontsize = 22)    
        
    else:
        #return NLL with a given value of tau.
        fitfunction_values = []
        NLL_values = [] 
        for i in range(len(data.time)):
            fitfunction_values.append(fit(tau, data.sigma[i], data.time[i]))
            NLL_values.append(-1*np.log(fitfunction_values[i]))
    
        return(np.sum(NLL_values))    
        
################################################################################## Task 3.4

def minimumpoint(tau_0, tau_1, tau_2, function, dataset):
    '''
    Task 3.4.1: Minimise
    Calculating the new minimum of a'function' given three points.
    '''
    return(0.5 * ((tau_2**2 - tau_1**2) * function(tau_0, dataset) + (tau_0**2 - tau_2**2) * function(tau_1, dataset) + (tau_1**2 - tau_0**2) * function(tau_2, dataset) ) / ( (tau_2 - tau_1) * function(tau_0,dataset) + (tau_0 - tau_2) * function(tau_1,dataset) + (tau_1 - tau_0) * function(tau_2,dataset) ))
        
        
def minimise(tau_0, tau_1, tau_2, function, dataset, plot=False): 
    '''
    Task 3.4.2: Minimise
    Parabolic minimiser to find the value of tau that minimises the NLL, where 'function' should be NLL (or cosh_func to test the functionality).
    
    tau_0,1,2 = initial guesses
    function = NLL or cosh_func
    '''
    guess=0.1
    minimise.tau_3 = [guess, minimumpoint(tau_0, tau_1, tau_2, function, dataset)]
    count = 0 #Iteration counter     
    
    #Parabolic method for minimisation 
    while abs(minimise.tau_3[-1] - minimise.tau_3[-2]) >= 1E-6: 
        count += 1
        
        if max(function(tau_0, dataset), function(tau_1, dataset), function(tau_2, dataset)) == function(tau_0, dataset):
            tau_0 = minimise.tau_3[-1]
        elif max(function(tau_0, dataset), function(tau_1, dataset), function(tau_2, dataset)) == function(tau_1, dataset):
            tau_1 = minimise.tau_3[-1]
        elif max(function(tau_0, dataset), function(tau_1, dataset), function(tau_2, dataset)) == function(tau_2, dataset):
            tau_2 = minimise.tau_3[-1]
            
        minimise.tau_3.append(minimumpoint(tau_0, tau_1, tau_2, function, dataset))
        
          
    #Calculating the curvature of the final parabola
    d = (tau_1 - tau_0) * (tau_2 - tau_0) * (tau_2 - tau_1)
    y_0, y_1, y_2 = function(tau_0, dataset), function(tau_1, dataset), function(tau_2, dataset)

    minimise.curv = ((2*(tau_2 - tau_1) * y_0)/d + (2*(tau_0 - tau_2) * y_1)/d + (2*(tau_1 - tau_0) * y_2)/d)/2 #dPdx divided by 2
    
    print('Number of iteration = '+str(count))
    print('Estimated minimum = '+str(minimise.tau_3[-1]))
    print('Curvature = '+str(minimise.curv))
    
    
    #Plot the results of the minimising function
    if plot == True:
        func_data = [] 
        t_data = []
        
        if function == cosh_func:
            func = 'cosh($\\tau$)'
            t_data = np.arange(-4, +4, 0.01)
            func_data = np.cosh(t_data)
            
        elif function == NLL:
            func = 'NLL($\\tau$)'
            t_data = np.arange(0.02, 3.0, 0.05) #(0.02, 3.0, 0.01)
            for i in t_data:
                func_data.append(NLL(i, dataset))
        
        #http://matplotlib.org/users/annotations_guide.html
        #http://matplotlib.org/users/annotations_intro.html    
        plt.plot(t_data, func_data, label = str(func), linewidth = 4)
        plt.grid(True)
        #plt.legend(loc=1,prop={'size':25})
        plt.title('Minimisation using the parabolic method on '+ str(func), fontsize = 45)
        plt.scatter(minimise.tau_3[-1], function(minimise.tau_3[-1], dataset), s=500, c='r', label = 'Mininum')
        matplotlib.rcParams.update({'font.size': 45})
        if function == cosh_func:
            plt.annotate('Min. point: $\\tau_{min}$ = '+str(minimise.tau_3[-1]), xy=(minimise.tau_3[-1], function(minimise.tau_3[-1], dataset)), xytext = (-3, -3), arrowprops=dict(facecolor='black',shrink=0.05), fontsize = 45)
            plt.ylabel(str(func) , fontsize = 45)
        elif function == NLL:
            plt.annotate('Min. point: $\\tau_{min}$ = '+format(minimise.tau_3[-1], '.4f'), xy=(minimise.tau_3[-1], function(minimise.tau_3[-1], dataset)), xytext = (0.5, 20000), arrowprops=dict(facecolor='black',shrink=0.05), fontsize = 45)
            plt.ylabel(str(func) + ' (log($\\frac{1}{ps}$))' , fontsize = 45)
            plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        plt.xlabel('$\\tau$ (ps)', fontsize = 45) 
            
################################################################################## Task 3.5 (part 1)

def NLLerror(tau, tau_min, dataset):
    '''
    Task 3.5.1: Find accuracy of fit result
    Solving NLL(t) = NLL(t_min) + 0.5
    '''
    return(NLL(tau, dataset)-0.5-NLL(tau_min, dataset))


def paraerror(tau, tau_min, dataset):
    '''
    Task 3.5.1: Find accuracy of fit result
    Solving curv(t-t_min)^2 + NLL(t_min) = NLL(t_min) + 0.5
    
    curv = cuvature of the last parabolic estimate
    '''
    return(minimise.curv*(tau-tau_min)**(2) - 0.5)

      
def error(a, b):
    '''
    Task 3.5.1: Find accuracy of fit result
    Calculating the half of the difference between two given points; used in secant and bisection methods.
    '''
    return(abs((b-a)/2.))


def secant(tau_0, tau_1, tau_2, estimate, tau_plus, dataset, graph):  #(tau_0=0.2, tau_1=0.3, tau_2=0.5, estimate = 'nll', tau_plus = True, dataset=10000, graph = True)

    '''
    Task 3.5.1: Find accuracy of fit result
    Estimating the erorr in tau by examining the changed in NLL by 0.5 from the minimum. This utilises the _Secant_ method.
    
    estimate = 'nll' or 'parabolic'
    tau_plus = True or False #False to find tau_minus
    '''  
    f = None
    guess_1 = None
    guess_2 = None
    plot = graph
    
    if estimate == 'nll' or 'NLL':
        f = NLLerror
    elif estimate == 'parabolic':
        f = paraerror
    
    if tau_plus == True:
        guess_1 = 0.405
        guess_2 = 0.5
    elif tau_plus == False:
        guess_1 = 0.3
        guess_2 = 0.35
    
    minimise(tau_0, tau_1, tau_2, NLL, dataset, plot)
    tau_min = minimise.tau_3[-1]
    tolerance_level = 1E-6
    count = 0
    
    #plotting the parabolic estimate
    if plot == True and estimate == 'parabolic':
        NLL_minimum = NLL(tau_min, dataset)
        tau_range = np.arange(-0.18, 1., 0.01)
        plt.plot(tau_range, (minimise.curv*(tau_range - tau_min)**2 + NLL_minimum), linewidth = 4)
            
    while error(guess_2, guess_1) >= tolerance_level:
        if error(guess_2, guess_1) == 0:
            return(guess_2)
            break
        count += 1
        tmp = guess_2 - (f(guess_2, tau_min, dataset)*(guess_2-guess_1))/(f(guess_2, tau_min, dataset)-f(guess_1, tau_min, dataset))
        guess_1 = guess_2
        guess_2 = tmp
        
    print('final tau = ' + str(guess_2), 'difference from min. = ' + str(guess_2-minimise.tau_3[-1]))       
    return(guess_2-minimise.tau_3[-1])


def bisection(tau_0, tau_1, tau_2, estimate, tau_plus, dataset, graph): #(tau_0=0.2, tau_1=0.3, tau_2=0.5, estimate = 'nll', tau_plus = True, dataset=10000, graph = True)
    '''
    Task 3.5.1: Find accuracy of fit result
    Estimating the erorr in tau by examining the changed in NLL by 0.5 from the minimum. This utilises the _Bisection_ method.
    
    estimate = 'nll' or 'parabolic'
    tau_plus = True or False #False to find tau_minus
    '''
    f = None
    low = None
    high = None
    plot = graph
    
    if estimate == 'nll' or 'NLL':
        f = NLLerror
    elif estimate == 'parabolic':
        f = paraerror
    
    if tau_plus == True:
        low = 0.401
        high = 0.5
    elif tau_plus == False:
        low = 0.3
        high = 0.41
    
    minimise(tau_0, tau_1, tau_2, NLL, dataset, plot)
    tau_min = minimise.tau_3[-1]
    count = 0
    tolerance_level = 1E-6  
    avg = (high + low)/2.0
    
    #plotting the parabolic estimate
    if plot == True and estimate == 'parabolic':
        NLL_minimum = NLL(tau_min, dataset)
        tau_range = np.arange(-0.18, 1., 0.01)
        plt.plot(tau_range, (minimise.curv*(tau_range - tau_min)**2 + NLL_minimum), linewidth = 4)
    
    while error(low, high) >= tolerance_level:
        if f(avg, tau_min, dataset) == 0:
            return(avg)
        elif f(low, tau_min, dataset)*f(avg, tau_min, dataset) < 0:
            high = avg
        else:
            low = avg
        
        avg = (high + low)/2.0
        count += 1
        print('high = ' + str(high), 'low = ' + str(low), 'avg = ' + str(avg))
    
    print('final tau = ' + str(avg), 'difference from min. = ' + str(avg - minimise.tau_3[-1]))        
    return(avg-minimise.tau_3[-1])
    
################################################################################## Task 3.5 (part 2)

def sdev(method=secant, estimate='nll', uppersigma=True): 
    '''
    Task 3.5.2: Find accuracy of fit result
    Investigating the behaviour of the standard deviation depending on the number of measurements.
    
    method = secant or bisection
    estimate = 'nll' or 'parabolic'
    '''
    sdev.sigma_plus = []
    sdev.sigma_minus = []
    func = estimate
    no_of_measurements = np.arange(250.,10000.,250.)

    if uppersigma == True:
        for sample in no_of_measurements:
            print('Number of measurements = '+ str(sample))
            data(sample)
            error = method(tau_0=0.2, tau_1=0.3, tau_2=0.5, estimate = func, tau_plus = True, dataset=sample, graph = False) 
            sdev.sigma_plus.append(error)
        
        for i in range(len(sdev.sigma_plus)):
            sdev.sigma_plus[i]=np.log10(sdev.sigma_plus[i])
            no_of_measurements[i]=np.log10(no_of_measurements[i]) 
        
        y_axis = sdev.sigma_plus   
        
    elif uppersigma == False:    
        for sample in no_of_measurements:
            print('Number of measurements = '+ str(sample))
            data(sample)
            error = method(tau_0=0.2, tau_1=0.3, tau_2=0.5, estimate = func, tau_plus = False, dataset=sample, graph = False) 
            sdev.sigma_minus.append(error)
            
        for i in range(len(sdev.sigma_minus)):
            sdev.sigma_minus[i]=np.log10(abs(sdev.sigma_minus[i])) #positive deviation 
            no_of_measurements[i]=np.log10(no_of_measurements[i]) 
        
        y_axis = sdev.sigma_minus 
     
    #Graph for sigma_plus & _minus
    fig, ax = plt.subplots()
    ax.scatter(no_of_measurements, y_axis, label = 's.d.', s=300, c='r')
    plt.legend(fontsize = 45)
    plt.grid(True,which="both",ls="-")
    #ax.set_yscale('log')
    #ax.set_xscale('log')
    #pylab.ylim([1E-3,1E-1])
    #pylab.xlim([1E2,1E5])
    if uppersigma == True:
        plt.title('Log-log plot of $\sigma^{+}$ against number of measurements', fontsize = 45)
        plt.xlabel('log$_{10}$(Number of measurements)', fontsize = 45)
        plt.ylabel('log$_{10}$(Estimated error $\sigma^{+}$) (log$_{10}$(ps))', fontsize = 45)
    elif uppersigma == False:
        plt.title('Log-log plot of $\sigma^{-}$ against number of measurements', fontsize = 45)
        plt.xlabel('log$_{10}$(Number of measurements)', fontsize = 45)
        plt.ylabel('log$_{10}$(Estimated error $\sigma^{-}$) (log$_{10}$(ps))', fontsize = 45)
    
    #Linear fit and extrapolation of the graph
    x = no_of_measurements
    y = y_axis 
    z = np.polyfit(x,y,1)
    poly = np.poly1d(z)
    extrapol = np.linspace(x[0], 6., 100)
    plt.plot(extrapol,poly(extrapol), label = 'y = ' + format(poly[1], '.4f')+ 'x ' + format(poly[0], '.4f'), linewidth = 5)
    plt.annotate('n = '+format(10**((-3.-poly[0])/poly[1]), '.0f'), xy=((-3.-poly[0])/poly[1], -3.), xytext = (5, -2.5), arrowprops=dict(facecolor='black'), fontsize = 45)
    matplotlib.rcParams.update({'font.size': 45})
    plt.legend(fontsize = 45)

################################################################################# Task 4: Background signal (part 1)

def cosh_2d(x_0, x_1, h):
    '''
    Task 4.1: Background signal
    2D cosh function as the test function.
    '''
    return(np.cosh(x_0)+np.cosh(x_1))


def dcosh_2d(x_0, x_1, h):
    '''
    Task 4.1: Background signal
    2D sinh function which is the differential of  the cosh function.
    '''
    return([np.sinh(x_0), np.sinh(x_1)])

        
def contour_testing():
    '''
    Task 4.1: Background signal
    Create a contour plot of the 2D cosh function.
    '''
    i1 = np.arange(-4.0, 4.0, 0.1) 
    i2 = np.arange(-4.0, 4.0, 0.1) 
    x_mesh, y_mesh = np.meshgrid(i1, i2)
    x = [x_mesh, y_mesh]
    f_mesh = cosh_2d(x_mesh, y_mesh, h=1)
    
    contourlines = range(0,20,3)            
    CS = plt.contour(x_mesh, y_mesh, f_mesh, contourlines, colors = 'k')
    plt.clabel(CS, inline=2, fontsize=30)
    matplotlib.rcParams.update({'font.size': 40})
    plt.title('f($x, y$) = cosh($x$) + cosh($y$)', fontsize = 40)
    plt.xlabel('$x$', fontsize = 40)
    plt.ylabel('$y$', fontsize = 40)
    im = plt.imshow(f_mesh, interpolation='bilinear', origin='lower', cmap=cm.YlOrBr, extent=(-4, 4., -4, 4.))
    levels = np.arange(-1.2, 1.6, 0.1)
    CS = plt.contour(f_mesh, levels, origin='lower', linewidths=4, extent=(-4, 4., -4, 4.))
    
    #Change the colormap for the contour lines and colorbar
    plt.flag()
    
    #Thicken the zero contour.
    zc = CS.collections[6]
    plt.setp(zc, linewidth=4)
    plt.clabel(CS, levels[1::2], inline=1, fmt='%1.1f', fontsize=16)
    
    CBI = plt.colorbar(im, orientation='vertical', shrink=0.8)
    l, b, w, h = plt.gca().get_position().bounds

################################################################################# Task 4: Background signal (part 2)

def background(i):
    '''
    Task 4.2: Background signal
    The background signal (convolution of a delta function with a Gaussian)
    '''
    return(np.exp(-0.5*(data.time[i]**(2)/data.sigma[i]**(2)))/(data.sigma[i]*np.sqrt(2*np.pi)))


def new_fit(tau, a, i):
    '''
    Task 4.2: Background signal
    The fit function taking background noise into account.
    
    a = fraction of signal in the sample
    '''
    return(a*fit(tau, data.sigma[i], data.time[i]) + (1 - a)*background(i))


def new_NLL(tau, a): 
    '''
    Task 4.2: Background signal
    The new NLL (Negative Log Likelihood) as a function of tau and a. 
    '''
    dataset = 10000   
    data(dataset)
    
    fitfunction_values = []
    NLL_values = [] 
    
    for i in range(len(data.time)):
        fitfunction_values.append(new_fit(tau, a, i))
        NLL_values.append(-1*np.log(fitfunction_values[i]))
    
    return(np.sum(NLL_values))  


def FDSdnew_NLLdtau(tau,a, h):
    '''
    Task 4.2: Background signal
    Partial differentiate the new NLL wrt tau using the FDS
    '''
    return((new_NLL(tau+h,a)-new_NLL(tau,a))/h)
   
     
def FDSdnew_NLLda(tau,a, h):
    '''
    Task 4.2: Background signal
    Partial differentiate the new NLL wrt a using the FDS
    '''
    return((new_NLL(tau,a+h)-new_NLL(tau,a))/h)    
 
                       
def FDSdnew_NLL(tau,a, h):
    '''
    Task 4.2: Background signal
    Return the gradient of the new NLL using the FDS
    ''' 
    return[FDSdnew_NLLdtau(tau,a, h), FDSdnew_NLLda(tau,a,h)]  


def CDSdnew_NLLdtau(tau,a, h):
    '''
    Task 4.2: Background signal
    Partial differentiate the new NLL wrt tau using the CDS
    '''
    return(new_NLL(tau+h, a) - new_NLL(tau-h, a))/(2*h)
    
    
def CDSdnew_NLLda(tau,a, h):
    '''
    Task 4.2: Background signal
    Partial differentiate the new NLL wrt a using the CDS
    '''
    return(new_NLL(tau,a+h) - new_NLL(tau,a-h))/(2*h)
    
    
def CDSdnew_NLL(tau,a, h):
    '''
    Task 4.2: Background signal
    Return the gradient of the new NLL using the CDS
    ''' 
    return([CDSdnew_NLLdtau(tau,a, h), CDSdnew_NLLda(tau,a, h)])


def d2new_NLLda2(tau, a, h):
    '''
    Task 4.2: Background signal
    d2f/da2 of new_NLL
    '''
    return (new_NLL(tau, a+h) - 2*new_NLL(tau, a) + new_NLL(tau, a-h))/(h**2)


def d2new_NLLdtau2(tau, a, k):
    '''
    Task 4.2: Background signal
    d2f/dt2 of new_NLL
    '''
    return (new_NLL(tau+k, a) - 2*new_NLL(tau,a) + new_NLL(tau-k, a))/(k**2)
    
    
def d2new_NLLdadtau(tau, a, h):
    '''
    Task 4.2: Background signal
    d2f/dtda of new_NLL (partial differentiation)
    '''    
    k = h
    return (new_NLL(tau+k, a+h) - new_NLL(tau-k, a+h) - new_NLL(tau+k, a-h) + new_NLL(tau-k, a-h))/(4*h*k)  
    
#####################################################################################

def contour():
    '''
    Task 4.2: Background signal
    Contour of the new NLL as function of tau and a.
    '''
    delta = 0.05 #smoothness of the contour lines
    alist = np.arange(0.1, 1.+delta, delta)
    taulist = np.arange(0.1, 1.+delta, delta)
    a_mesh, tau_mesh = np.meshgrid(alist,taulist)
    nll_mesh = np.empty(a_mesh.shape)

    for row in range(len(a_mesh)):
        a_row = a_mesh[row]
        tau_row = tau_mesh[row]
        nll_row = nll_mesh[row]
        for element in range(len(a_row)):
            a_element = a_row[element]
            tau_element = tau_row[element]
            nll_row[element] = new_NLL(tau_element,a_element)
            
    lines = np.linspace(6400, 18400, 15)
    plt.figure()
    CS = plt.contour(a_mesh,tau_mesh,nll_mesh,lines)
    plt.clabel(CS, inline=1, fontsize=25)
    plt.xlabel('fraction of signal $a$', fontsize=35)
    plt.ylabel('Lifetime of $D^{0}$ meson $\\tau$ (ps)', fontsize=35)
    plt.title('Contour plot of NLL($\\tau$, a)', fontsize=35)
    matplotlib.rcParams.update({'font.size': 35}) #axes size
    im = plt.imshow(nll_mesh, interpolation='bilinear', origin='lower', cmap=cm.YlOrBr, extent=(0.1, 1., 0.1, 1.))
    levels = np.arange(-1.2, 1.6, 0.2)
    CS = plt.contour(nll_mesh, levels, origin='lower', linewidths=4, extent=(0.1, 1., 0.1, 1.))
    
    #Change the colormap for the contour lines and colorbar
    plt.flag()
    
    #Thicken the zero contour.
    zc = CS.collections[6]
    plt.setp(zc, linewidth=4)
    plt.clabel(CS, levels[1::2], inline=1, fmt='%1.1f', fontsize=16)
    
    CBI = plt.colorbar(im, orientation='vertical', shrink=0.8)
    l, b, w, h = plt.gca().get_position().bounds
    #ll, bb, ww, hh = CBI.ax.get_position().bounds
    #CBI.ax.set_position([ll, bb - 0.47*h, ww, h*0.8])
    
    
def gradient_method(n = 300, scheme = 'cds', contour_plot=True):
    '''
    Task 4.2: Background signal
    The gradient_method to find the minimum of the new NLL.
    
    n = number of iteration
    FDS = forward difference scheme
    CDS = central difference scheme
    (cosh = test the method using the 2D cosh function)
    '''
    #scheme = str.lower(raw_input('Which numerical differentiation scheme should I use? FDS or CDS: '))
    numer_diff = None
    alpha = None #step length
    h = 0.00001 #step-size for finite difference approxmiation
    X = np.empty((n+1,2))
    X[0] = None #tau, a
    
    if scheme == 'fds':
        numer_diff = FDSdnew_NLL
        if contour_plot == True:
            contour()
        alpha = 0.00001
        X[0] = [0.4,  0.2]
    elif scheme == 'cds':
        numer_diff = CDSdnew_NLL
        if contour_plot == True:
            contour()
        alpha = 0.00001
        X[0] = [0.4,  0.2]
    elif scheme == 'cosh':
        numer_diff = dcosh_2d
        if contour_plot == True:
            contour_testing()
        alpha = 0.05
        X[0] = [ 3.,  -2.5]
    else:
        print('Invalid input! Please try again.')
        scheme = str.lower(raw_input('Which numerical differentiation scheme should I use? FDS or CDS '))               
                                 
    
    for i in range(n):
        X[i+1] = X[i] - np.dot(alpha, numer_diff(X[i][0],X[i][1], h))
        
        if abs(X[i+1][0]-X[i][0]) < 1e-6 and abs(X[i+1][1]-X[i][1]) < 1E-6:
            print('Number of iteration = '+str(i))
            print('[final tau, final a] = ' +str(X[i])) 
            plt.plot(X[:i,1],X[:i,0],'-o', label = 'Gradient method', markersize=10, color = 'blue', linewidth=3)
            plt.legend(fontsize = 27, loc=1) 
            plt.annotate(' Start', xy=(X[0][1], X[0][0]), xytext = (X[0][1]-0.05, X[0][0]+0.05), fontsize = 28)
            return(X[i])
            break
            
            
def newton_method(n = 300, scheme = 'cds', contour_plot=True):
    '''
    Task 4.2: Background signal
    The Newton method to find the minimum of the new NLL.
    
    n = number of iteration
    FDS = forward difference scheme
    CDS = central difference scheme
    (cosh = test the method using the 2D cosh function)
    '''
    sch = scheme
    h = 0.00001
    numer_diff = None
    x = np.zeros((n+1,2))
    g = np.zeros((n+1,2))                       
    
    x[0] = None  #Initial values tau, a
    
    if scheme == 'fds':
        numer_diff = FDSdnew_NLL
        if contour_plot == True:
            contour()
        x[0] = [0.4,  0.2]
    elif scheme == 'cds':
        numer_diff = CDSdnew_NLL
        if contour_plot == True:
            contour()
        x[0] = [0.4,  0.2]
    elif scheme == 'cosh':
        numer_diff = dcosh_2d
        if contour_plot == True:
            contour_testing()
        x[0] = [ 3.,  -2.5]
    else:
        print('Invalid input! Please try again.')
        scheme = str.lower(raw_input('Which numerical differentiation scheme should I use? FDS or CDS '))               
    
                                                              
    g[0] = numer_diff(x[0][0],x[0][1], h) #Initial gradient                   
    inv_Hessian = np.zeros((n+1,2,2)) 

    
    for i in range(n):
        # inverse Hessian
        a = x[i][1]
        tau = x[i][0]
        if sch == 'fds' or sch == 'cds':
            A = d2new_NLLdtau2(tau, a, h)
            B = d2new_NLLdadtau(tau, a, h)
            C = d2new_NLLdadtau(tau, a, h)
            D = d2new_NLLda2(tau, a, h)
            
        elif sch == 'cosh':
            A = np.cosh(tau) #d2new_NLLdtau2(tau, a, h)
            B = 0
            C = 0
            D = np.cosh(a) 
            
        inv_Hessian[i] = np.array([[D, -1*B],[-1*C, A]]) /(A*D-B*C)
        
        # update the vector
        x[i+1] = x[i] - np.dot(inv_Hessian[i], g[i])
        
        # update the gradient
        g[i+1] = numer_diff(x[i+1][0],x[i+1][1], h)  
                
        if abs(x[i+1][0]-x[i][0]) < 1e-6 and abs(x[i+1][1]-x[i][1]) < 1e-6:
            print('Number of iteration = '+str(i))
            print('[final tau, final a] = ' +str(x[i])) 
            plt.plot(x[:i,1],x[:i,0],'-o', label = "Newton's method", markersize=10, color = 'red', linewidth=3)
            plt.legend(fontsize = 27, loc = 1)
            plt.annotate(' Start', xy=(x[0][1], x[0][0]), xytext = (x[0][1], x[0][0]), fontsize = 28)
            #plt.xlabel('a',fontsize=25)
            #plt.ylabel('$\\tau$ (ps)', fontsize=25)
            #plt.show()
            return(x[i])
            break

def q_newton(n = 300, scheme = 'cds', contour_plot=True):    
    '''
    Task 4.2: Background signal
    The quasi-Newton method to find the minimum of the new NLL.
    
    n = number of iteration
    FDS = forward difference scheme
    CDS = central difference scheme
    (cosh = test the method using the 2D cosh function)
    '''  
    #scheme = str.lower(raw_input('Which numerical differentiation scheme should I use? FDS or CDS: '))
    numer_diff = None
    alpha = None #step length
    h = 0.00001 #step-size for finite difference approxmiation
    x = np.zeros((n+1,2)) 
    g = np.zeros((n+1,2))  
    x[0] = None #Initial condition                

    
    if scheme == 'fds':
        if contour_plot == True:
            contour()
        numer_diff = FDSdnew_NLL
        alpha = 0.00001
        x[0] = [0.4,  0.2]
    elif scheme == 'cds':
        if contour_plot == True:
            contour()
        numer_diff = CDSdnew_NLL
        alpha = 0.00001
        x[0] = [0.4,  0.2]
    elif scheme == 'cosh':
        if contour_plot == True:
            contour_testing()
        numer_diff = dcosh_2d
        alpha = 0.05
        x[0] = [ 3.,  -2.5]
    else:
        print('Invalid input! Please try again.')
        scheme = str.lower(raw_input('Which numerical differentiation scheme should I use? FDS or CDS '))    
        
    g[0]= numer_diff(x[0][0],x[0][1], h)     #Initial condition   
    inv_Hessian = np.zeros((n+1,2,2))        #1st term of Davidon Fletcher Powell algorithm             
    inv_Hessian[0] = [[1, 0.0],[0.0, 1]]
    
    for i in range(n):
        #Update vector
        delta = -alpha * np.dot(inv_Hessian[i], g[i])
        x[i+1] = x[i] + delta
        
        #Update gradient for i+1
        g[i+1] = numer_diff(x[i+1][0], x[i+1][1], h)  
  
        #Update inverse Hessian for i+1
        gamma = g[i+1]-g[i]
        B1 = np.outer(delta,delta)
        B2 = np.dot(gamma,delta)
        term_B = np.dot(B1, 1./B2)          #2nd term of Davidon Fletcher Powell algorithm

        C1 = B1
        C2 = np.dot(C1, inv_Hessian[i])
        C3 = np.dot(inv_Hessian[i], C2)  
        C4 = np.dot(inv_Hessian[i],gamma)
        C5 = np.dot(gamma, C4)           
        term_C = np.dot(C3, 1./C5)          #3rd term of Davidon Fletcher Powell algorithm
        
        inv_Hessian[i+1] = inv_Hessian[i] + term_B - term_C
        
        if abs(x[i+1][0]-x[i][0]) < 1e-6 and abs(x[i+1][1]-x[i][1]) < 1e-6:
            print('Number of iteration = '+str(i))
            print('[final tau, final a] = ' +str(x[i])) 
            plt.plot(x[:i,1],x[:i,0],'-o', label = 'Quasi-Newton method', markersize=10, color = 'green',linewidth=3)
            plt.legend(fontsize = 27, loc = 1)
            plt.annotate(' Start', xy=(x[0][1], x[0][0]), xytext = (x[0][1], x[0][0]), fontsize = 28)
            #plt.xlabel('a',fontsize=25)
            #plt.ylabel('$\\tau$ (ps)', fontsize=25)
            #plt.show()
            return(x[i])
            break
        
    
def tau_error_NLL2D(tau, a_min, NLL_min):
    '''
    Task 4.2: Background signal
    Esitmating the error in tau_min using the new_NLL with a fixed a_min
    '''
    return(new_NLL(tau, a_min) - 0.5 - NLL_min)       
          
              
def a_error_NLL2D(tau_min, a, NLL_min):
    '''
    Task 4.2: Background signal
    Esitmating the error in a_min using the new_NLL with a fixed tau_min
    '''
    return(new_NLL(tau_min, a) - 0.5 - NLL_min)    
    
                                  
def tau_error_secant(minimiser = gradient_method, tau_plus = True):
    '''
    Task 4.2: Background signal
    Estimating the error in tau using the secant method.
    
    minimiser = gradient_method or q_newton
    tau_plus = True or False
    '''
    if tau_plus == True:
        guess_1 = 0.405
        guess_2 = 0.5
    elif tau_plus == False:
        guess_1 = 0.3
        guess_2 = 0.35 
    
    min_values = minimiser(300, 'cds', False)
    tau_min = min_values[0]
    a_min = min_values[1]
    NLL_min = new_NLL(tau_min, a_min)
    
    
    while error(guess_1, guess_2) > 1E-6:
        if error(guess_1, guess_2) == 0:
            return guess_2
        tmp = guess_2 - (tau_error_NLL2D(guess_2, a_min, NLL_min)*(guess_2-guess_1))/(tau_error_NLL2D(guess_2, a_min, NLL_min)-tau_error_NLL2D(guess_1, a_min, NLL_min))
        guess_1 = guess_2
        guess_2 = tmp
        
    if tau_plus == True:
        print('tau_min = ' +str(tau_min) + '+' + str(abs(guess_2-tau_min)))
    elif tau_plus == False:
        print('tau_min = ' +str(tau_min) + '-' + str(abs(guess_2-tau_min)))
    
    
def a_error_secant(minimiser = gradient_method, a_plus = True): 
    '''
    Task 4.2: Background signal
    Estimating the error in a_min using the secant method.
    
    minimiser = gradient_method or q_newton
    tau_plus = True or False
    '''
    if a_plus == True:
        guess_1 = 0.985
        guess_2 = 0.995
    elif a_plus == False:
        guess_1 = 0.6
        guess_2 = 0.7 
            
    min_values = minimiser(300, 'cds', False)
    tau_min = min_values[0]
    a_min = min_values[1]
    NLL_min = new_NLL(tau_min, a_min)
    

    while error(guess_1, guess_2) > 1E-6:
        if error(guess_1, guess_2) == 0:
            return guess_2
        tmp = guess_2 - (a_error_NLL2D(tau_min, guess_2, NLL_min)*(guess_2-guess_1))/(a_error_NLL2D(tau_min, guess_2, NLL_min)-a_error_NLL2D(tau_min, guess_1, NLL_min))
        guess_1 = guess_2
        guess_2 = tmp
    if a_plus == True:
        print('a_min = ' +str(a_min) + '+' + str(abs(guess_2-a_min)))
    elif a_plus == False:
        print('a_min = ' +str(a_min) + '-' + str(abs(guess_2-a_min)))



def tau_error_bisection(minimiser = gradient_method, tau_plus = True): 
    '''
    Task 4.2: Background signal
    Estimating the error in tau using the secant method.
    
    minimiser = gradient_method or q_newton
    tau_plus = True or False
    '''          
    count = 0
    min_values = minimiser(300, 'cds', False)
    tau_min = min_values[0]
    a_min = min_values[1]
    NLL_min = new_NLL(tau_min, a_min)
    tolerance_level = 1E-6   
    
    low = None
    high = None
    
    if tau_plus == True:
        low = 0.41
        high = 0.6
    elif tau_plus == False:
        low = 0.3
        high = 0.41
        
    avg = (high + low)/2.0
    print('high = ' + str(high), 'low = ' + str(low), 'avg = ' + str(avg))
    
    while error(low, high) >= tolerance_level:
        if tau_error_NLL2D(avg, a_min, NLL_min) == 0:
            return(avg)
        elif tau_error_NLL2D(low, a_min, NLL_min)*tau_error_NLL2D(avg, a_min, NLL_min) < 0:
            high = avg
        else:
            low = avg
        
        avg = (high + low)/2.0
        count += 1
        print('high = ' + str(high), 'low = ' + str(low), 'avg = ' + str(avg))
    
    print('final tau = ' + str(avg), 'difference from min. = ' + str(avg - tau_min))        
    return(avg-tau_min)
    
    
def a_error_bisection(minimiser = gradient_method, a_plus = True): 
    '''
    Task 4.2: Background signal
    Estimating the error in tau using the secant method.
    
    minimiser = gradient_method or q_newton
    tau_plus = True or False
    '''          
    count = 0
    min_values = minimiser(300, 'cds', False)
    tau_min = min_values[0]
    a_min = min_values[1]
    NLL_min = new_NLL(tau_min, a_min)
    tolerance_level = 1E-6   
    
    low = None
    high = None
    
    if a_plus == True:
        low = 0.98
        high = 0.995
    elif a_plus == False:
        low = 0.5
        high = 0.98
        
    avg = (high + low)/2.0
    print('high = ' + str(high), 'low = ' + str(low), 'avg = ' + str(avg))
    
    
    while error(low, high) >= tolerance_level:
        if a_error_NLL2D(tau_min, avg, NLL_min) == 0:
            return(avg)
        elif a_error_NLL2D(tau_min, low, NLL_min)*a_error_NLL2D(tau_min, avg, NLL_min) < 0:
            high = avg
        else:
            low = avg
        
        avg = (high + low)/2.0
        count += 1
        print('high = ' + str(high), 'low = ' + str(low), 'avg = ' + str(avg))
    
    print('final a = ' + str(avg), 'difference from min. = ' + str(avg - a_min))        
    return(avg-a_min)
    
######################################### Finding error using error matrix/covariance matrix    

def error_matrix(tau =  0.40968342, a=0.98368267):
    '''
    Calculating the error using the error matrix/covariance matrix in addition to the correlation coefficient.
    '''
    k = h = 0.0001 #step sizes

    element_11 = d2new_NLLdtau2(tau, a, k)
    element_12 = d2new_NLLdadtau(tau, a, h)
    element_21 = d2new_NLLdadtau(tau, a, h)
    element_22 = d2new_NLLda2(tau, a, h)
    
    weight_matrix = np.array([[element_11, element_12],[element_21, element_22]])
    Error_matrix = inv(weight_matrix)
    
    print('Error matrix = ' + str(Error_matrix))
    print('sigma_tau = ' +str(np.sqrt(Error_matrix[0][0])) + ', simga_a = ' +str(np.sqrt(Error_matrix[1][1])))
    print('correlation coefficient = ' + str(Error_matrix[1][0]/(np.sqrt(Error_matrix[0][0])*np.sqrt(Error_matrix[1][1]))))     
    
################################# Error ellipse

def contour_one_sigma():
    '''
    Task 4.2: Background signal
    Contour of the new NLL change change by 0.5.
    '''
    alist = np.linspace(0.97514578114854-0.001, 0.99218443885146+0.001, 10)
    taulist = np.linspace(0.40420466062081-0.001, 0.41517721937919+0.001, 10)
    a_mesh, tau_mesh = np.meshgrid(alist,taulist)
    nll_mesh = np.empty(a_mesh.shape)
    
    for row in range(len(a_mesh)):
        a_row = a_mesh[row]
        tau_row = tau_mesh[row]
        nll_row = nll_mesh[row]
        for element in range(len(a_row)):
            a_element = a_row[element]
            tau_element = tau_row[element]
            nll_row[element] = new_NLL(tau_element,a_element)
            
    lines = np.array([6218.3944, 6218.3944+0.5])
    #Contour plot
    plt.figure()
    plt.xlabel('fraction of signal $a$', fontsize=35)
    plt.ylabel('Lifetime of $D^{0}$ meson $\\tau$ (ps)', fontsize=35)
    plt.title('Error ellipse of NLL($\\tau$, a)', fontsize=35)
    matplotlib.rcParams.update({'font.size': 35}) #axes size
    CS = plt.contour(a_mesh,tau_mesh,nll_mesh,lines,colors='k',linestyles='dashed', linewidth = 4)
    plt.clabel(CS, inline=1, fontsize=25)
    plt.scatter(0.9837,  0.40969) #Minimum 
    #lines
    plt.plot(np.array([1,1])*0.9750,np.array([0.40420-0.01, 0.4152+0.01]),'r-')
    plt.plot(np.array([1,1])*0.9919,np.array([0.40420-0.01, 0.4151+0.01]),'r-')
    plt.plot(np.array([0.9751-0.01, 0.9921+0.01]),np.array([1,1])*0.41516,'r-')
    plt.plot(np.array([0.9751-0.01, 0.9921+0.01]),np.array([1,1])*0.40431,'r-')
    
    im =plt.imshow(nll_mesh, interpolation='bilinear', origin='lower', cmap=cm.YlOrBr,extent=(0.9651, 1.0022, 0.3942, 0.4252))
    
