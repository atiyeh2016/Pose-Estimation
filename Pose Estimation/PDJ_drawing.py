import matplotlib.pyplot as plt
import matplotlib


#fig = plt.figure()
#ax = fig.gca()
#ax.plot(loss_valid_1_200, linewidth=5)
#plt.title('Test Loss')
#matplotlib.rc('font', size=14)

#ax1=plt.subplot(2, 2, 1)
#ax1.plot(PDJ_All[0,:])
#ax1.set_title('Right ankle')
##ax1.set_xlabel('epoch')
#ax1.set_ylabel('PDJ')
#
#ax1=plt.subplot(2, 2, 2)
#ax1.plot(PDJ_All[1,:])
#ax1.set_title('Right knee')
##ax1.set_xlabel('epoch')
##ax1.set_ylabel('PDJ')
##
##
#ax1=plt.subplot(2, 2, 3)
#ax1.plot(PDJ_All[2,:])
#ax1.set_title('Right hip')
#ax1.set_xlabel('epoch')
#ax1.set_ylabel('PDJ')
#
#ax1=plt.subplot(2, 2, 4)
#ax1.plot(PDJ_All[3,:])
#ax1.set_title('Left hip')
#ax1.set_xlabel('epoch')
##ax1.set_ylabel('PDJ')

#ax1=plt.subplot(2, 2, 1)
#ax1.plot(PDJ_All[4,:])
#ax1.set_title('Left knee')
##ax1.set_xlabel('epoch')
#ax1.set_ylabel('PDJ')
#
#ax1=plt.subplot(2, 2, 2)
#ax1.plot(PDJ_All[5,:])
#ax1.set_title('Left ankle')
##ax1.set_xlabel('epoch')
##ax1.set_ylabel('PDJ')
##
##
#ax1=plt.subplot(2, 2, 3)
#ax1.plot(PDJ_All[6,:])
#ax1.set_title('Right elbow')
#ax1.set_xlabel('epoch')
#ax1.set_ylabel('PDJ')
#
#ax1=plt.subplot(2, 2, 4)
#ax1.plot(PDJ_All[7,:])
#ax1.set_title('Left hip')
#ax1.set_xlabel('epoch')
#ax1.set_ylabel('PDJ')
#
#ax1=plt.subplot(2, 2, 1)
#ax1.plot(PDJ_All[8,:])
#ax1.set_title('Right shoulder')
##ax1.set_xlabel('epoch')
#ax1.set_ylabel('PDJ')
#
#ax1=plt.subplot(2, 2, 2)
#ax1.plot(PDJ_All[9,:])
#ax1.set_title('Left shoulder')
##ax1.set_xlabel('epoch')
##ax1.set_ylabel('PDJ')
##
##
#ax1=plt.subplot(2, 2, 3)
#ax1.plot(PDJ_All[10,:])
#ax1.set_title('Left elbow')
#ax1.set_xlabel('epoch')
#ax1.set_ylabel('PDJ')
#
#ax1=plt.subplot(2, 2, 4)
#ax1.plot(PDJ_All[11,:])
#ax1.set_title('Left wrist')
#ax1.set_xlabel('epoch')
#ax1.set_ylabel('PDJ')
##
#
#ax1=plt.subplot(1, 2, 1)
#ax1.plot(PDJ_All[12,:])
#ax1.set_title('Neck')
##ax1.set_xlabel('epoch')
#ax1.set_ylabel('PDJ')
#
#ax1=plt.subplot(1, 2, 2)
#ax1.plot(PDJ_All[13,:])
#ax1.set_title('Head')
#ax1.set_xlabel('epoch')
#ax1.set_ylabel('PDJ')


left = 0.125  # the left side of the subplots of the figure
right = 0.9   # the right side of the subplots of the figure
bottom = 0.1  # the bottom of the subplots of the figure
top = 0.9     # the top of the subplots of the figure
wspace = 0.3  # the amount of width reserved for space between subplots,
              # expressed as a fraction of the average axis width
hspace = 0.4  # the amount of height reserved for space between subplots,
              # expressed as a fraction of the average axis height

matplotlib.pyplot.subplots_adjust(left=left, bottom=bottom,
                                  right=right, top=top,
                                  wspace=wspace,
                                  hspace=hspace)
