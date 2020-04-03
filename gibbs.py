import math
import copy
import random
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter, defaultdict

MAX_BURNS = 50
MAX_SAMPLES = 200
ETA = 1
BETA = 1

def energy(Y, X):
    N, M = Y.shape
    energy = -1*(np.sum(X*Y) + np.sum(Y[:N-1, :]*Y[1:, :])
                + np.sum(Y[:, :M-1]*Y[:, 1:]))
    return energy

def sample(i, j, Y, X):
    blanket = [Y[i-1][j], Y[i][j-1], Y[i][j+1], Y[i+1][j], X[i][j]]

    prob = np.exp(2*np.sum(blanket)) / (1 + np.exp(2*np.sum(blanket)))

    if random.random() < prob:
        return 1
    else:
        return -1

def sampling(filename, initialization='same', logfile=None):
  print("\n########## Working on '{}' ##########".format(filename))
  print("Initialization : ", initialization)
  X = read_txt_file(filename)
  Y = np.copy(X)
  N, M = Y.shape
  if initialization == 'neg':
    Y = -1 * Y
  if initialization == 'rand':
    Y[np.where(Y != 0)] = np.random.choice([1, -1], size=((N-2)*(M-2)))
  B = 50
  S = 200
  
  if logfile != None:
    with open(logfile, 'w') as log:
      ctr = 0
      for _ in range(B):
          for i in range(1, N-1):
              for j in range(1, M-1):
                  Y[i][j] = sample(i, j, Y, X)
          log.write("{}   {}  B\n".format(ctr, energy(Y, X)))

          ctr += 1
          if ctr % 10 == 0:
              print("Burn-in %s done!" % ctr)

      counts = np.zeros((N, M))
      for _ in range(S):
          sampled = np.zeros((N, M))
          for i in range(1, N-1):
              for j in range(1, M-1):
                  sampled[i][j] = sample(i, j, Y, X)
          log.write("{}   {}  S\n".format(ctr, energy(sampled, X)))
          is_one_in_sample = (sampled == 1)
          counts += (is_one_in_sample)
          ctr += 1
          if ctr % 10 == 0:
              print("Sample %s done!" % (ctr - (B)))

  else:
    ctr = 0
    for _ in range(B):
        for i in range(1, N-1):
            for j in range(1, M-1):
                Y[i][j] = sample(i, j, Y, X)

        ctr += 1
        if ctr % 10 == 0:
            print("Burn-in %s done!" % ctr)

    counts = np.zeros((N, M))
    for _ in range(S):
        sampled = np.zeros((N, M))
        for i in range(1, N-1):
            for j in range(1, M-1):
                sampled[i][j] = sample(i, j, Y, X)
        is_one_in_sample = (sampled == 1)
        counts += (is_one_in_sample)
        ctr += 1
        if ctr % 10 == 0:
            print("Sample %s done!" % (ctr - (B)))
      
  return counts / float(S)

def denoise_image(filename, initialization='rand', logfile=None):
  '''
   Doing Gibbs sampling and computing the energy of each assignment for the image
   specified in filename.
   It should run MAX_BURNS iterations of burn in and then
   MAX_SAMPLES iterations for collecting samples.

   filename: file name of image in txt
   initialization: 'same' or 'neg' or 'rand'
   logfile: the file name that stores the energy log (will use for plotting
       later) look at the explanation of plot_energy to see detail
  '''

  posterior = sampling(filename, initialization, logfile=logfile)
  denoised = np.zeros(posterior.shape)
  denoised[np.where(posterior < 0.5)] = 1
  return denoised



# ===========================================
# Helper functions
# ===========================================

def plot_energy(filename):
  '''
  filename: a file with energy log, each row should have three terms separated
    by a \t:
      iteration: iteration number
      energy: the energy at this iteration
      S or B: indicates whether it's burning in or a sample
  e.g.
      1   -202086.0   B
      2   -210446.0   S
      ...
  '''
  its_burn, energies_burn = [], []
  its_sample, energies_sample = [], []
  with open(filename, 'r') as f:
    for line in f:
      it, en, phase = line.strip().split()
      if phase == 'B':
        its_burn.append(float(it))
        energies_burn.append(float(en))
      elif phase == 'S':
        its_sample.append(float(it))
        energies_sample.append(float(en))
      else:
        print("bad phase: -%s-" % phase)

  p1, = plt.plot(its_burn, energies_burn, 'r')
  p2, = plt.plot(its_sample, energies_sample, 'b')
  plt.title(filename)
  plt.legend([p1, p2], ["burn in", "sampling"])
  plt.savefig(filename)
  plt.close()


def read_txt_file(filename):
  '''
  filename: image filename in txt
  return:   2-d array image
  '''
  f = open(filename, "r")
  lines = f.readlines()
  height = int(lines[0].split()[1].split("=")[1])
  width  = int(lines[0].split()[2].split("=")[1])
  Y = [[0]*(width+2) for i in range(height+2)]
  print("Shape : ",np.shape(Y))
  for line in lines[2:]:
    i, j, val = [int(entry) for entry in line.split()]
    Y[i+1][j+1] = val
  return np.array(Y)


def convert_to_png(denoised_image, title):
  '''
  save array as a png figure with given title.
  '''
  plt.imshow(denoised_image, cmap='gray')
  plt.title(title)
  plt.savefig(title + '.png')


def text_to_png(imagefile):
    imageData = read_txt_file(imagefile)
    imageData = .5 * (1 - imageData)
    convert_to_png(imageData,imagefile)

def get_error(img_a, img_b):
  '''
  compute the fraction of all pixels that differ between the two input images.
  '''
  N = len(img_b[0])*len(img_b)*1.0
  return sum([sum([1 if img_a[row][col] != img_b[row][col] else 0
                   for col in range(len(img_a[0]))])
              for row in range(len(img_a))]
             ) / N


def run_q2():
  '''
  Run denoise_image function with different initialization and plot out the
  energy functions.
  '''
  #Saving the denoised image for Q3
  global denoised_a

  denoise_image("inputs/a_noise10.png.txt", initialization='rand',
                logfile='outputs/log_rand')
  denoise_image("inputs/a_noise10.png.txt", initialization='neg',
                logfile='outputs/log_neg')
  denoised_a = denoise_image("inputs/a_noise10.png.txt",
                                             initialization='same',
                                               logfile='outputs/log_same')

  # plot out the energy functions
  plot_energy("outputs/log_rand")
  plot_energy("outputs/log_neg")
  plot_energy("outputs/log_same")


def run_q3():
  '''
  Run denoise_image function with two different pics, and
  report the errors between denoised images and original image
  '''
  global denoised_b
  denoised_b = denoise_image("inputs/b_noise10.png.txt",
                                             initialization='same',
                                             logfile=None)
  orig_img_a = read_txt_file("inputs/a.png.txt")
  orig_img_a = .5 * (1 - orig_img_a)
  orig_img_b = read_txt_file("inputs/b.png.txt")
  orig_img_b = .5 * (1 - orig_img_b)

  # save denoised images and original image to png figures
  convert_to_png(denoised_b, "outputs/denoised_b")
  convert_to_png(denoised_a, "outputs/denoised_a")
  convert_to_png(orig_img_b, "outputs/orig_img_b")
  convert_to_png(orig_img_a, "outputs/orig_img_a")

  N, M = orig_img_a.shape
  print("restoration error for image %s : %s" %
        ("a", np.sum((orig_img_a != denoised_a)[1:N-1, 1:M-1]) / float(
            (N-1) * (M-1))))
  N, M = orig_img_b.shape
  print("restoration error for image %s : %s" %
        ("b", np.sum((orig_img_b != denoised_b)[1:N-1, 1:M-1]) / float(
            (N-1) * (M-1))))



if __name__ == "__main__":
  
  run_q2()
  run_q3()
