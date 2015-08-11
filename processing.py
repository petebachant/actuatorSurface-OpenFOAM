#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Processing for OpenFOAM actuatorSurface simulation.

by Pete Bachant (petebachant@gmail.com)

"""
from __future__ import division, print_function
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import foampy
from subprocess import call
import pandas
from pxl.styleplot import set_sns

def styleplot():
    plt.tight_layout()
    
exp_path = "/media/pete/External 2/Research/Experiments/2014 Spring RVAT Re dep"

# Some constants
R = 0.5
U = 1.0
H = 0.05
D = 1.0
A = H*D
rho = 1000.0

ylabels = {"meanu" : r"$U/U_\infty$",
           "stdu" : r"$\sigma_u/U_\infty$",
           "meanv" : r"$V/U_\infty$",
           "meanw" : r"$W/U_\infty$",
           "meanuv" : r"$\overline{u'v'}/U_\infty^2$"}

def resample_wake(x=1.0):
    import gensampledict
    gensampledict.main(x)
    call(["sample", "-latestTime"])
    
def loadwake():
    """Loads wake data and returns y/R and statistics."""
    folder = os.listdir("postProcessing/sets")[0]
    flist = os.listdir("postProcessing/sets/"+folder)
    flist.remove("streamwise_U.xy")
    data = {}
    for fname in flist:
        fpath = "postProcessing/sets/"+folder+"/"+fname
        z_H = float(fname.split("_")[1])
        data_s = np.loadtxt(fpath, unpack=True)
        data[z_H] = data_s
    return data
    
def plotwake(plotlist=["meancontquiv"], save=False, savepath="figures", 
             savetype=".pdf", print_analysis=True):
    data = loadwake()
    y_R = data[0][0]/R
    z_H = np.asarray(sorted(data.keys()))
    # Assemble 2-D arrays
    u = np.zeros((len(z_H), len(y_R)))
    v = np.zeros((len(z_H), len(y_R)))
    w = np.zeros((len(z_H), len(y_R)))
    xvorticity = np.zeros((len(z_H), len(y_R)))
    for n in range(len(z_H)):
        u[n,:] = data[z_H[n]][1]
        v[n,:] = data[z_H[n]][2]
        w[n,:] = data[z_H[n]][3]
        try:
            xvorticity[n,:] = data[z_H[n]][4]
        except IndexError:
            pass
    def turb_lines():
        plt.hlines(0.5, -1, 1, linestyles='solid', linewidth=2)
        plt.vlines(-1, 0, 0.5, linestyles='solid', linewidth=2)
        plt.vlines(1, 0, 0.5, linestyles='solid', linewidth=2)
    if "meanu" in plotlist or "all" in plotlist:
        plt.figure(figsize=(10,5))
        cs = plt.contourf(y_R, z_H, u, 20, cmap=plt.cm.coolwarm)
        plt.xlabel(r'$y/R$')
        plt.ylabel(r'$z/H$')
        cb = plt.colorbar(cs, shrink=1, extend='both', 
                          orientation='horizontal', pad=0.2)
        cb.set_label(r'$U/U_{\infty}$')
        turb_lines()
        ax = plt.axes()
        ax.set_aspect(2)
        plt.grid(True)
        plt.yticks([0,0.13,0.25,0.38,0.5,0.63])
        styleplot()
    if "meanv" in plotlist or "all" in plotlist:
        plt.figure(figsize=(10,5))
        cs = plt.contourf(y/0.5, z, v, 20, cmap=plt.cm.coolwarm)
        plt.xlabel(r'$y/R$')
        plt.ylabel(r'$z/H$')
        styleplot()
        cb = plt.colorbar(cs, shrink=1, extend='both', 
                          orientation='horizontal', pad=0.3)
        cb.set_label(r'$V/U_{\infty}$')
        #turb_lines()
        ax = plt.axes()
        ax.set_aspect(2)
        plt.grid(True)
        plt.yticks([0,0.13,0.25,0.38,0.5,0.63])
    if "v-wquiver" in plotlist or "all" in plotlist:
        # Make quiver plot of v and w velocities
        plt.figure(figsize=(10,5))
        Q = plt.quiver(y_R, z_H, v, w, angles='xy')
        plt.xlabel(r'$y/R$')
        plt.ylabel(r'$z/H$')
        plt.ylim(-0.2, 0.78)
        plt.xlim(-3.2, 3.2)
        plt.quiverkey(Q, 0.75, 0.2, 0.1, r'$0.1$ m/s',
                   labelpos='E',
                   coordinates='figure',
                   fontproperties={'size': 'small'})
        plt.tight_layout()
        plt.hlines(0.5, -1, 1, linestyles='solid', colors='r',
                   linewidth=2)
        plt.vlines(-1, -0.2, 0.5, linestyles='solid', colors='r',
                   linewidth=2)
        plt.vlines(1, -0.2, 0.5, linestyles='solid', colors='r',
                   linewidth=2)
        ax = plt.axes()
        ax.set_aspect(2)
        plt.yticks([0,0.13,0.25,0.38,0.5,0.63])
        if save:
            plt.savefig(savepath+'v-wquiver'+savetype)
    if "xvorticity" in plotlist or "all" in plotlist:
        plt.figure(figsize=(10,5))
        cs = plt.contourf(y_R, z_H, xvorticity, 10, cmap=plt.cm.coolwarm)
        plt.xlabel(r'$y/R$')
        plt.ylabel(r'$z/H$')
        cb = plt.colorbar(cs, shrink=1, extend='both', 
                          orientation='horizontal', pad=0.26)
        cb.set_label(r"$\Omega_x$")
        turb_lines()
        ax = plt.axes()
        ax.set_aspect(2)
        plt.yticks([0,0.13,0.25,0.38,0.5,0.63])
        styleplot()
        if save:
            plt.savefig(savepath+'/xvorticity_AD'+savetype)
    if "meancontquiv" in plotlist or "all" in plotlist:
        plt.figure(figsize=(7.5, 6.66))
        # Add contours of mean velocity
        cs = plt.contourf(y_R, z_H, u, 20, cmap=plt.cm.coolwarm)
        cb = plt.colorbar(cs, shrink=1, extend='both', 
                          orientation='horizontal', pad=0.1)
                          #ticks=np.round(np.linspace(0.44, 1.12, 10), decimals=2))
        cb.set_label(r'$U/U_{\infty}$')
        plt.hold(True)
        # Make quiver plot of v and w velocities
        Q = plt.quiver(y_R, z_H, v, w, angles='xy', width=0.0022, scale=1)
        plt.xlabel(r'$y/R$')
        plt.ylabel(r'$z/H$')
        #plt.ylim(-0.2, 0.78)
        #plt.xlim(-3.2, 3.2)
        plt.xlim(-3.66, 3.66)
        plt.ylim(-1.22, 1.22)
        veckeyscale = 0.1
        plt.quiverkey(Q, 0.8, 0.21, veckeyscale, 
                      r'${} U_\infty$'.format(veckeyscale),
                      labelpos='E', coordinates='figure', 
                      fontproperties={'size': 'small'})
        plt.hlines(0.5, -1, 1, linestyles='solid', colors='gray',
                   linewidth=3)
        plt.hlines(-0.5, -1, 1, linestyles='solid', colors='gray',
                   linewidth=3)
        plt.vlines(-1, -0.5, 0.5, linestyles='solid', colors='gray',
                   linewidth=3)
        plt.vlines(1, -0.5, 0.5, linestyles='solid', colors='gray',
                   linewidth=3)
        ax = plt.axes()
        ax.set_aspect(2.0)
        styleplot()
        if save:
            plt.savefig(os.path.join(savepath, "meancontquiv_AD" + savetype))
    if print_analysis:
        print("Spatial average of U =", u.mean())
        
def plotexpwake(Re_D, quantity, z_H=0.0, save=False, savepath="", 
                savetype=".pdf", newfig=True, marker="--ok",
                fill="none", figsize=(10, 5)):
    """Plots the transverse wake profile of some quantity. These can be
      * meanu
      * meanv
      * meanw
      * stdu
    """
    U = Re_D/1e6
    label = "Exp."
    folder = exp_path + "/Wake/U_" + str(U) + "/Processed/"
    z_H_arr = np.load(folder + "z_H.npy")
    i = np.where(z_H_arr==z_H)
    q = np.load(folder + quantity + ".npy")[i]
    y_R = np.load(folder + "y_R.npy")[i]
    if newfig:
        plt.figure(figsize=figsize)
    plt.plot(y_R, q/U, marker, markerfacecolor=fill, label=label)
    plt.xlabel(r"$y/R$")
    plt.ylabel(ylabels[quantity])
    plt.grid(True)
    styleplot()

def set_funky_plane(x=1.0):
    foampy.dictionaries.replace_value("system/funkyDoCalcDict", "basePoint", 
                                      "({}".format(x))

def read_funky_log():
    with open("log.funkyDoCalc") as f:
        for line in f.readlines():
            try:
                line = line.replace("=", " ")
                line = line.split()
                if line[0] == "planeAverageAdvectionY":
                    y_adv = float(line[-1])
                elif line[0] == "weightedAverage":
                    z_adv = float(line[-1])
                elif line[0] == "planeAverageTurbTrans":
                    turb_trans = float(line[-1])
                elif line[0] == "planeAverageViscTrans":
                    visc_trans = float(line[-1])
                elif line[0] == "planeAveragePressureGradient":
                    pressure_trans = float(line[-1])
            except IndexError:
                pass
    return {"y_adv" : y_adv, "z_adv" : z_adv, "turb_trans" : turb_trans,
            "visc_trans" : visc_trans, "pressure_trans" : pressure_trans}

def run_funky_batch():
    xlist = [-1.99, -1.5, -1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 
             1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 7.99]
    df = pandas.DataFrame()
    for x in xlist:
        print("Setting measurement plane to x =", x)
        set_funky_plane(x)
        call(["./Allrun.post"])
        dfi = pandas.DataFrame(read_funky_log(), index=[x])
        df = df.append(dfi)
    if not os.path.isdir("processed"):
        os.mkdir("processed")
    df.index.name = "x"
    print(df)
    df.to_csv("processed/mom_transport.csv", index_label="x")

def make_momentum_trans_bargraph(print_analysis=True):
    data = read_funky_log()
    y_adv = data["y_adv"]
    z_adv = data["z_adv"]
    turb_trans = data["turb_trans"]
    visc_trans = data["visc_trans"]
    pressure_trans = data["pressure_trans"]
    plt.figure(figsize=(6,4))
    ax = plt.gca()
    ax.bar(range(5), [y_adv, z_adv, turb_trans, visc_trans, pressure_trans], 
           color="gray", edgecolor="black", hatch="//", width=0.5)
    ax.set_xticks(np.arange(5)+0.25)
    ax.set_xticklabels(["$y$-adv.", "$z$-adv.",
                        "Turb.", "Visc.", "Press."])
    plt.ylabel(r"$\frac{U \, \mathrm{ transport}}{UDU_\infty}$")
    plt.tight_layout()
    if print_analysis:
        sum = y_adv + z_adv + turb_trans + visc_trans + pressure_trans
        print("Momentum recovery = {:.3f}% per turbine diameter".format(sum))

def plot_mom_transport():
    df = pandas.read_csv("processed/mom_transport.csv")
    print(df)
    plt.plot(df.x, df.y_adv, "-o", label=r"$-V \partial U / \partial y$")
    plt.plot(df.x, df.z_adv, "-s", label=r"$-W \partial U / \partial z$")
    plt.plot(df.x, df.turb_trans, "-^", label=r"$\nu_t \nabla^2 U$")
    plt.plot(df.x, df.visc_trans, "->", label=r"$\nu \nabla^2 U$")
    plt.plot(df.x, df.pressure_trans/10, "-<", label=r"$-\partial P / \partial x$ ($\times 10^{-1}$)")
    plt.legend(loc="lower right", ncol=1)
    plt.xlabel("$x/D$")
    plt.ylabel(r"$\frac{U \, \mathrm{ transport}}{UU_\infty D^{-1}}$")
    plt.grid()
    plt.tight_layout()

def plot_U_streamwise():
    times = os.listdir("postProcessing/sets")
    times.sort()
    latest = times[-1]
    filepath = os.path.join("postProcessing", "sets", latest, 
                            "streamwise_U.xy")
    x, u, v, w = np.loadtxt(filepath, unpack=True)
    plt.plot(x, u, "k")
    plt.xlabel("$x/D$")
    plt.ylabel(r"$U/U_\infty$")
    plt.grid()
    plt.tight_layout()

def plot_streamwise(save=False, savepath="figures"):
    plt.figure(figsize=(7.5, 4))
    plt.subplot(121)
    plot_U_streamwise()
    plt.subplot(122)
    plot_mom_transport()
    plt.tight_layout()
    if save:
        plt.savefig(os.path.join(savepath, "AD_streamwise.pdf"))

def main():
    if not os.path.isdir("figures"):
        os.mkdir("figures")
    set_sns()
    #resample_wake(x=1.0)
    plotwake(plotlist=["meancontquiv"], save=True)
    #make_momentum_trans_bargraph()
    if not os.path.isfile("processed/mom_transport.csv"):
        run_funky_batch()
    #plot_mom_transport()
    plot_streamwise(save=True)
    plt.show()

if __name__ == "__main__":
    main()
