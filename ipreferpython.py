import numpy as np
import matplotlib.pyplot as plt
import cmath
import imageio
import os


# for i in range(2):
#     strint = str(i).zfill(4)
#
#     data = np.genfromtxt('rawdata/rad1kick_20p_7E-4damp00{:}00.dat'.format(strint), skip_header=4, dtype=float, delimiter='\t')
#     cdata = np.array([x[0] + 1j*x[1] for x in data])
#     rdata = cdata.reshape((1024, 256)).transpose()
#
#     fig, ax = plt.subplots(2, 1, figsize=(10, 8))
#     ax[0].imshow(abs(rdata))
#     ax[1].imshow(np.angle(rdata))
#     plt.savefig("imgs/{:}.png".format(strint))
#     plt.close()


for i in range(100, 400):
    print(i)
    strint = str(i).zfill(4)
    data = np.genfromtxt('rawdata/rad1kick_20p_7E-4damp00{:}00.dat'.format(strint), skip_header=4, dtype=float, delimiter='\t')
    cdata = np.array([x[0] + 1j*x[1] for x in data])
    rdata = cdata.reshape((1024, 256)).transpose()
    for i in range(60):
        rdata = np.delete(rdata, 127 - 30, 0)
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    ax.imshow(abs(rdata), cmap="afmhot", vmin=-0.001, vmax=abs(rdata).max())
    # ax[1].imshow(np.angle(rdata))
    plt.savefig("imgs_short/{:}.png".format(strint))
    plt.close()


images = []
for filename in sorted(os.listdir("imgs_short/")):
    print("Moviefying ", filename)
    images.append(imageio.imread("imgs_short/" + filename))
imageio.mimsave('simulations_short.mov', images)


quit()


# Output: [(1, 4, 6, 8, 3) (4, 5, 6, 8, 9) (2, 3, 6, 8, 5)]