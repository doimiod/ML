from cmath import sqrt
import numpy as np
import math
from PIL import Image

# make a function that takes an n × n array and a k × k kernel, convolves the kernel to the input array and returns the result
def convolve(array, kernel):
    if (array.size<kernel.size): # array should be larger than kernel
        return 0
    convolved = []               # array for convolved img
    sum = 0
    size = 0
    kSize = math.sqrt(kernel.size)  # kernel size. if the kernel is 3*3, the size is 3
    kSize = int(kSize)
    arrSize = math.sqrt(array.size) # priginal array size. if it is 3*3, the size is 3
    arrSize = int(arrSize)
    size = arrSize - kSize + 1
    print(size)
    for i in range(size):
        for j in range(size):
            for k in range(kSize):
                for l in range(kSize):
                    sum = sum + array[k+i,l+j]*kernel[k,l]
            convolved.append(sum)
            sum = 0
     
    convolved = np.array(convolved)
    convoSize = math.sqrt(convolved.size)
    convoSize = int(convoSize)
    convolved = convolved.reshape(-1, convoSize)         
    return convolved

# the comment below was used to check if the conv function works correctly

# array = [1,0,1,0,-1,1,0,1,0,-1,1,0,1,0,-1,1,0,1,0,-1,1,0,1,0,-1]
# kernel1 = [1,2,1,1,1,3,2,1,1]
# array = np.array(array)
# array = array.reshape(-1,5)
# kernel1 = np.array(kernel1)
# kernel1 = kernel1.reshape(-1,3)
# print(array)
# print(kernel1)
# newArr1 = convolve(array, kernel1)
# print(newArr1)

# kernel2 = [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,2]
# kernel2 = np.array(kernel2)
# kernel2 = kernel2.reshape(-1,4)
# print(kernel2)

# newArr2 = convolve(array, kernel2)
# print(newArr2)

im = Image.open("/Users/doimasanari/Documents/ML/week8/pics.jpg/wk8.jpg")
rgb = np.array(im.convert("RGB"))
r=rgb[:,:,0] # array of R pixels 

print(r)
print(r.size)

kernel1 = [-1, -1, -1, -1, 8, -1, -1, -1, -1]
kernel1 = np.array(kernel1)
kernel1 = kernel1.reshape(-1,3)
convolvedR = convolve(r, kernel1)
Image.fromarray(np.uint8(convolvedR)).show()

kernel2 = [0, -1, 0, -1, 8, -1, 0, -1, 0]
kernel2 = np.array(kernel2)
kernel2 = kernel2.reshape(-1,3)
convolvedR = convolve(r, kernel2)
Image.fromarray(np.uint8(convolvedR)).show()
    
# sum = sum + array[j+i,k]*kernel[j,k]
# sum = sum + array[j,k+1]*kernel[j,k]