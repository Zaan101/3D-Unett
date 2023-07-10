
  if i == 1:
      

      temp1= np.fft.rfft2(image_array[1][:,:,7]) #fourier transform to generate noise
      temp2= np.fft.rfft2(image_array[1][:,:,15])

      temp1 += np.random.normal(1,100**.1,(238,129)).reshape(238,129) 
      temp2 += np.random.normal(1,100**.1,(238,129)).reshape(238,129)



      image_array[1][:,:,7]= np.fft.irfft2(temp1) 
      image_array[1][:,:,15]= np.fft.irfft2(temp2)


      t = [7,15]
      image_array[1][:,:,7] += np.random.normal(1,1**.5,(238,256)).reshape(238,256)
      image_array[1][:,:,15] += np.random.normal(1,1**.5,(238,256)).reshape(238,256)
      for x in range(2):
        for y in range(256):
          for z in range(238):
            if image_array[1][z,y,t[x]] > 190: #reduing high pixel brightness

              image_array[1][z,y,t[x]] /= 1.9
