'''
HR_Encoder extends Alberto Rosas Garcias unofficial StyleGAN2-Tensorflow-2.x implementation (https://github.com/rosasalberto/StyleGAN2-TensorFlow-2.x) to encode your own images into StyleGAN2 *W* space.

Legal information

HR_Encoder.ipynb, HR_Encoder.py and w_utilities.py are Copyright © 2020 HANS ROETTGER (mailto:oss.roettger@posteo.org)
and distributed under the terms of GNU AGPLv3 (https://www.gnu.org/licenses/agpl-3.0.html)

Severability clause: If a provision of the [GNU AGPLv3](https://www.gnu.org/licenses/agpl-3.0.html) is or becomes illegal, invalid or unenforceable in any jurisdiction, that shall not affect: 
1. the validity or enforceability in that jurisdiction of any other provision of the GNU AGPLv3; or 
2. the validity or enforceability in other jurisdictions of that or any other provision of the GNU AGPLv3.

!!! WARNING !!!

HR_Encoder makes heavy use of your hardware via the tensorflow libraries. Pay attention to adequate ventilation/ cooling and take all necessary precautions to prevent overload, overheating or consequential damage according to the recommendations of your hardware provider.
'''

import numpy as np
import tensorflow as tf

class HR_Encoder:
    def __init__(self,generator_model,features_model,resolution=(1024,1024,3)):
        '''
        generator_model: styleGAN2.generator(latent vector w) -> image 
        features_model: intermediate results styleGAN2.discriminator(image) -> features
        resolution: according to trained model (height,width,channels)
        '''
        model_scales = {(1024,1024,3):18,(512,512,3):16,(256,256,3):14}
        try:
            self.model_scale=int(model_scales[resolution])
        except:
            raise SystemExit("Exit in HR_Encoder.init! Unsupported image resolution: {0}".format(resolution))
        
        print("HR_Encoder initializing...")
        
        self.height,self.width,self.channels=resolution
        self.img_size=self.height*self.width*self.channels
        
        # create the HR_Encoder model
            
        # input of HR_Encoder model is the target image
        i=tf.keras.layers.Input([self.channels,self.height,self.width],dtype=tf.float32)
        # Features for optimization are the image itself plus the output of the features_model
        x=features_model(tf.reshape(i,[1,self.channels,self.height,self.width]))
        out=tf.concat([tf.reshape(i,[-1]),tf.reshape(x,[-1])],axis=0)
        self.features_model=tf.keras.Model(inputs=[i],outputs=out)
                                                      
        # complete chain: latent vector w -> generator_model -> image -> features_model -> features
        self.chain=tf.keras.Sequential()
        self.chain.add(generator_model.synthesis_network)
        self.chain.add(self.features_model)
        self.chain.compile()
            
        # instantiate variables with dummy values  
        
        # latent vector w to optimize
        self.current_w=tf.Variable(np.ones((1, self.model_scale, 512)).astype('float32')*.5)
        # copy of start w used for truncation calculation
        self.start_w=tf.Variable(np.ones((1, self.model_scale, 512)).astype('float32')*.5)
        # target image and features of target image
        dummy_img=np.ones((1,self.channels,self.height,self.width),dtype=np.float32)
        self.target_result=tf.Variable(self.features_model(dummy_img).numpy())
        self.target_image=tf.Variable(self.target_result[0:self.img_size])
        self.target_features=tf.Variable(self.target_result[self.img_size::])
        # optimizer
        self.optimizer=tf.keras.optimizers.Adam(learning_rate=0.01)
        self.trainable_variables=[self.current_w]
        # loss weights
        self.weight_image=tf.Variable(1.0,trainable=False)
        self.weight_features=tf.Variable(1.0,trainable=False)
        self.weight_regularization=tf.Variable(1.0,trainable=False)

 
    # This is the core function for optimization of latent vector w to generate the target image
    # ------------------------------------------------------------------------------------------    
    @tf.function
    def runOptimization(self):
        with tf.GradientTape() as g:
            current_result=self.chain(self.current_w)
            # calculate loss
            loss=0.0
            loss+=self.weight_image*tf.reduce_sum(tf.math.squared_difference(current_result[0:self.img_size],self.target_image))
            loss+=self.weight_features*tf.reduce_sum(tf.math.squared_difference(current_result[self.img_size::],self.target_features))
            loss+=self.weight_regularization*tf.reduce_sum(tf.math.squared_difference(self.current_w,tf.reduce_mean(self.current_w,axis=1,keepdims=True)))
            # optimize latent vector w
            gradients=g.gradient(loss,self.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients,self.trainable_variables))
            return loss
    # -------------------------------------------------------------------------------------------    
        
    # Mapping f(0.0..1.0)->[1.0..0.0] for parameter ratio f(x)*parameter+(1-f(x))*dyn_rapameter in img2w()
    def parameterDynamics(self,x):
        return ((x-1.0)*(x-1.0))
        
    def encodeImage(self,img_path,w_tape=[],steps=400,parameters=[1.0,0.02,100.0,6.0,0.02],dyn_parameters=[],dyn_function=None):
        '''
        img path : path to target image (accepts .png & .jpg; use square images with high resolution)
        w_tape : list of latent vectors w; HR_Encoder starts with the last(!) w and appends optimized w at the end
        steps : number of optimization steps - the more the better (but also a while longer)
        parameters (There is a tradeoff between them! Keep [0] at 1.0 and adjust the others!):
          [0] : weight of image data in optimization process (moves w to fit target shape - can result in blurry images)
          [1] : weight of high level features (moves w to generate sharp images with natural textures - but images at will)
          [2] : weight for regularization of w (to keep w in the preferred form for the generator)
          [3] : truncation value for w (to prevent w from taking values too far from average w)
          [4] : learning rate
        dyn_parameters : same format as above; if given, parameters shift from parameters to dyn_parameters during optimization
        dyn_function: defines the ratio between parameters / dyn_parameters form first step (0.0) to last step (1.0)
        
        returns w_tape - a list of latent vectors w for each optimization step + the best fitting w as last element
        '''
        print("\rHR_Encoder preparing...",end="")
        # prepare target image to fit HR_Encoder model
        img=tf.image.decode_png(tf.io.read_file(img_path))
        img=tf.image.resize(img,[self.height,self.width], method='bilinear')
        target_img=img.numpy().astype(np.ubyte)
        img=tf.expand_dims(img,0)
        img=tf.cast(img,tf.float32)
        img/=128.0
        img-=1.0
        img=np.transpose(img,(0,3,1,2))
        
        # calculate target features to be achieve in optimization process
        self.target_result.assign(self.features_model(img).numpy())
        self.target_image.assign(self.target_result[0:self.img_size])
        self.target_features.assign(self.target_result[self.img_size::])

        # if w_tape is given start with last w, else start with random w
        if w_tape:
            self.current_w.assign(w_tape[-1].reshape((1, self.model_scale, 512)).astype('float32'))  
        else:
            self.current_w.assign(np.random.normal(1, self.model_scale, 512).astype('float32')) 
        self.start_w.assign(self.current_w)
        
        # if dynamic parameter are used
        if dyn_parameters:
            if dyn_function:
                # use given mapping function for changing parameters
                rat=map(dyn_function,np.linspace(0.0,1.0,steps))
            else:
                # or change linearly from parameters to dyn_parameters 
                rat=np.linspace(1.0,0.0,steps)
        else:
            # use same parameter for all steps
            rat=np.linspace(0.0,0.0,steps)
            current_parameters=np.array(parameters)
            self.weight_image.assign(current_parameters[0])
            # scale features to same size as image data
            self.weight_features.assign(current_parameters[1]*self.img_size/(self.target_result.shape[0]-self.img_size))            
            self.weight_regularization.assign(current_parameters[2])
            truncation=current_parameters[3]
            self.optimizer.learning_rate.assign(current_parameters[4])

        # remember lowest loss and corresponding latent vector w
        best_loss=float("inf") 
        first_loss=0.0
        best_w=self.current_w.numpy()[0]
            
        #optimize
        # ------------------------------------------------------------------------------------------------------
        for i,r in enumerate(rat):
            if dyn_parameters:
                # update current parameters for each step
                current_parameters=r*np.array(parameters)+(1.0-r)*np.array(dyn_parameters)
                self.weight_image.assign(current_parameters[0])
                self.weight_features.assign(current_parameters[1]*self.img_size/(self.target_result.shape[0]-self.img_size))            
                self.weight_regularization.assign(current_parameters[2])
                truncation=current_parameters[3]
                self.optimizer.learning_rate.assign(current_parameters[4])

            # w truncation
            last_w=self.current_w.numpy()[0]-self.start_w.numpy()[0]
            mask=(last_w>truncation)|(last_w<-truncation)
            last_w[mask]=np.random.normal(0,truncation,(self.model_scale, 512))[mask]
            self.current_w[0].assign(last_w+self.start_w.numpy()[0])
            
            # optimization step
            # -----------------
            current_loss=self.runOptimization()
            # -----------------
            
            loss=str(current_loss)
            loss=float(loss[10:loss.find(",",9)])
            
            if first_loss<=0.1:
                first_loss=loss
                
            rel_loss=(loss/first_loss)*100.0
           
            if loss<best_loss:
                best_loss=loss
                best_w=self.current_w.numpy()[0]
            w_tape.append(self.current_w.numpy()[0])  
            
            print("\rHR_Encoder {0:4d}/{1:4d}: [{2:6.3f},{3:6.3f},{4:6.3f},{5:6.3f},{6:6.3f}], loss≈ {7:2.1f} %".format(i,steps,*current_parameters,rel_loss),end='  ')
        # --------------------------------------------------------------------------------------------------------     
        print("\n  best loss≈ {0:2.1f} %".format((best_loss/first_loss)*100.0))
        w_tape.append(best_w)
        return w_tape,target_img
