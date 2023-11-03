
import os
import re

import numpy           as np
import numpy.linalg    as npla
import scipy.linalg    as spla
import scipy.integrate as spi


def num_lines(filename):
    """ Counts the number of lines in a file """
    with open(filename, "r") as fd:
        count = 0
        for i in fd: count += 1
    return count

def parse_command_line_args(cmd_args, args=[]):
    """ Determines the files to process from the command line arguments """
    files, args = [], check_args(cmd_args)
    if( len(args) == 0 ):
        files = os.listdir("./data/")
    elif( len(args) > 1 ):
        for i in range(0, len(args)):
            files.append("H_STO-" + str(args[i]) + "G.dat")
    elif( args[0] == 12 ):
        files = ["./data/H_STO-12G.dat"]
    elif( args[0] == 20 ):
        files = ["./data/H_STO-20G.dat"]
    elif( args[0] == 23 ):
        files = ["./data/H_STO-23G.dat"]
    elif( args[0] > 6 or args[0] <= 0 ):
        print("Given value must be between 1 and 6 or it can be 20")
    elif( args[0] <= 6 and args[0] > 0 ):
        files = ["./data/H_STO-" + str(args[0]) + "G.dat"]
    else:
        files = ["./data/H_STO-6G.dat"]
    if( len(files) > 1 ):
        files = sort_files(files)
    return files

# Clearly there is a better way of doing this...
def sort_files(files):
    """ Ensures the files are in a certain order """
    new_files = []
    if( "H_STO-1G.dat" in files ):
        new_files.append("H_STO-1G.dat")
    if( "H_STO-2G.dat" in files ):
        new_files.append("H_STO-2G.dat")
    if( "H_STO-3G.dat" in files ):
        new_files.append("H_STO-3G.dat")
    if( "H_STO-4G.dat" in files ):
        new_files.append("H_STO-4G.dat")
    if( "H_STO-5G.dat" in files ):
        new_files.append("H_STO-5G.dat")
    if( "H_STO-6G.dat" in files ):
        new_files.append("H_STO-6G.dat")
    if( "H_STO-12G.dat" in files ):
        new_files.append("H_STO-12G.dat")
    if( "H_STO-20G.dat" in files ):
        new_files.append("H_STO-20G.dat")
    if( "H_STO-23G.dat" in files ):
        new_files.append("H_STO-23G.dat")
    return new_files

def check_args(cmd_args, args=[]):
    """ Checks to make sure the command line arguments are positive integers """
    for i in cmd_args:
        try:
            i = abs(int(i))
            args.append(i)
        except ValueError:
            pass
    return args

def get_alpha_coeff(filename):
    """ Reading in alpha coeff for gaussian functions """
    contents = []
    with open(filename, "r") as fd:
        for line in fd:
            if( "****" in line or "!" in line ):
                pass
            else:
                line = line.strip().split()
                if( len(line) > 0 and len(line) < 3 ):
                    contents.append(line)
    alpha_coeff = basis_set_parser(contents)
    return alpha_coeff

def basis_set_parser(contents):
    """ Takes the basis set file contents and extracts the alpha values """
    alpha_coeff = np.array([], dtype=np.float64)
    for items in contents:
        if( len(items[0]) <= 1 ):
            pass
        else:
            items[0] = items[0].replace("D", "E")
            alpha_coeff = np.append(alpha_coeff, np.float64(items[0]))
    return alpha_coeff

def eval_linear_combination(C, alpha, r_points):
    """ Evaluating linear combination of gaussian functions to plot """
    points = [0 for i in range(len(r_points))]
    for j in range(len(alpha)):
        for i, r in enumerate(r_points):
            points[i] += C[j] * ngf(r, alpha[j])
    return points

def ngf(r, alpha):
    """ Normalized Gaussian Function """
    norm = ( ((2 / np.pi)**(3/4)) * np.sqrt(alpha**(3/2)) )
    return norm * np.exp(-alpha * r**2)

def ngf_prod(r, ai, aj):
    """ Product of two normalized gaussians """
    temp_1 = (2 * np.sqrt(2) * ((ai * aj)**(3/4)))
    temp_2 = np.exp((r**2) * (-(ai + aj)))
    temp_3 = (np.pi)**(3/2)
    return (temp_1 * temp_2) / temp_3

def ngf_first_derivative(r, alpha):
    """ Derivative of a normalized gaussian """
    temp_1 = 2 * (2/np.pi)**(3/4) * alpha * np.sqrt(alpha**(3/2))
    temp_2 = r * np.exp(-alpha * (r**2))
    return -(temp_1 * temp_2)

def ngf_second_derivative(r, alpha):
    """ Derivative of a normalized gaussian """
    temp_1 = 2 * ((2/np.pi)**(3/4)) * alpha**(7/4) * (r**2)
    temp_2 = np.exp(alpha * (-r**2)) * ((2 * alpha * (r**2)) - 3)
    return (temp_1 * temp_2)

def overlap(r, ai, aj):
    """ Integrable overlap function """
    return 4 * np.pi * (r**2) * ngf_prod(r, ai, aj)

def calculate_s(func, ai, aj):
    """ Calculate overlap integral """
    return spi.quad(func, 0, np.inf, args=(ai, aj), epsabs=1E-13, epsrel=1E-13, limit=200)[0]

def calculate_s_alt(ai, aj):
    """ Analytical evaluation of  the overlap matrix """
    temp_1 = 2 * np.sqrt(2) * ((ai*aj)**(3/4))
    temp_2 = (ai + aj)**(3/2)
    return temp_1 / temp_2

def potential(r, ai, aj):
    """ Integrable potential function """
    return 4 * np.pi * (r**2) * (-ngf(r, ai) / r) * ngf(r, aj)

def calculate_v(func, ai, aj):
    """ Calculate potential energy """
    return spi.quad(func, 0, np.inf, args=(ai, aj), epsabs=1E-10, epsrel=1E-10, limit=200)[0]

def calculate_v_alt(ai, aj):
    """ Analytical evaluation of the PE matrix """
    temp_1 = 4 * np.power((2/np.pi), (1/2)) * (ai*aj)**(3/4)
    temp_2 = ai + aj
    return -(temp_1 / temp_2)

def kinetic_proxy(r, aj):
    """ Base kinetic function """
    temp_1 = -( 1 / (2 * (r**2)) )
    return temp_1 * ngf_second_derivative(r, aj)

def kinetic(r, ai, aj):
    """ Integrable kinetic function """
    temp_1 = ngf(r, ai)
    temp_2 = kinetic_proxy(r, aj)
    return 4 * np.pi * (r**2) * temp_1 * temp_2

def calculate_t(func, ai, aj):
    """ Calculate kinetic energy """
    return spi.quad(func, 0, np.inf, args=(ai, aj), epsabs=1E-10, epsrel=1E-10, limit=200)[0]

def calculate_t_alt(ai, aj):
    """ Analytical evaluation of the KE matrix """
    temp_1 = 6 * np.sqrt(2) * (ai*aj)**(7/4)
    temp_2 = (ai + aj)**(5/2)
    return temp_1 / temp_2

def diagonalize_s(S):
    """ Diagonalize S to get PDP """
    P     = npla.eigh(S)[1].T
    P_i   = npla.inv(P)
    A     = np.dot(P_i, np.dot(S, P))
    A_isr = spla.fractional_matrix_power(A, (-1/2))
    return P, A_isr, P_i

def form_density(C):
    """ Building the density matrix """
    D = np.zeros((len(C), len(C)), dtype=np.float64)
    for i in range(len(C)):
        for j in range(len(C)):
            D[i,j] += C[i] * C[j]
    return D

def overlap_total(D, S):
    """ Computing overlap for determining the total energy """
    return np.sum(D * S)

def potential_total(D, V, S):
    """ Computing the total potential energy """
    return np.sum(D * V) / S

def kinetic_total(D, T, S):
    """ Computing the total kinetic energy """
    return np.sum(D * T) / S

def total_energy(D, H, S):
    """ Computing the total energy """
    return np.sum(D * H) / S
