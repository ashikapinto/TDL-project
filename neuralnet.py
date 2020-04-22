import numpy as np
import tensorflow as tf
from tqdm import tqdm
import midi_manipulation

def menu(songs):
    """
    #the index of the lowest note on the piano roll
    #the index of the highest note on the piano roll
    #the note range
    """
    l_note = midi_manipulation.lowerBound 
    h_note = midi_manipulation.upperBound
    music_note_range = h_note-l_note 

    """
    #the number of timesteps that we will create at a time
    #the size of the visible layer.
    #the size of the hidden layer
    """
    number_of_timesteps  = 10 
    number_of_visible      = 2*music_note_range*number_of_timesteps  
    number_of_hidden       = 100

    """
    #The number of training epochs that we are going to run. For each epoch we go through the entire data set.
    #The number of training examples that we are going to send through the RBM at a time.
    #The learning rate of our model
    """

    number_of_epochs = 200 
    batch_size = 100  
    learning_rate = tf.constant(0.005, tf.float32) 

    x  = tf.placeholder(tf.float32, [None, number_of_visible], name="x") #The placeholder variable that holds our data
    W  = tf.Variable(tf.random_normal([number_of_visible, number_of_hidden], 0.01), name="W") #The weight matrix that stores the edge weights
    bh = tf.Variable(tf.zeros([1, number_of_hidden],  tf.float32, name="bh")) #The bias vector for the hidden layer
    bv = tf.Variable(tf.zeros([1, number_of_visible],  tf.float32, name="bv")) #The bias vector for the visible layer


    
    def sample(vector_probs):
        """
        Sample function will return the uniform randomly sampled vector of zeroes and ones of the size of vector_probs
        """
        return tf.floor(vector_probs + tf.random_uniform(tf.shape(vector_probs), 0, 1))

    
    def single_gibbs_step(xk):
        """
        this function will be used two times
        1) while training to sample out the data from the network
        2) while generating the output music by sampling from the network by giving random input vectors
        #Runs a single gibbs step. The visible values are initialized to xk
        """
        hk = sample(tf.sigmoid(tf.matmul(xk, W) + bh)) #Propagate the visible values to sample the hidden values
        xk = sample(tf.sigmoid(tf.matmul(hk, tf.transpose(W)) + bv)) #Propagate the hidden values to sample the visible values
        return xk
    def get_gibbs_sample(k):
        #Runs a k-step gibbs chain to sample from the probability distribution of the RBM defined by W, bh, bv
        #Run gibbs steps for k iterations
        count = 0
        x_sample = x
        while(count < k):
            x_sample = single_gibbs_step(x_sample)
            count+=1
        x_sample = tf.stop_gradient(x_sample) 
        return x_sample

    
    ##########------------SPECIFYING THE UPDATE RULES FOR VARIOUS PARAMETERS OF THE MODEL----------------------############

    x_sample = get_gibbs_sample(1) 
   
    #Now, we update the values of W, bh, and bv, based on the difference between the samples that we drew and the original values
    size_bt = tf.cast(tf.shape(x)[0], tf.float32)
    """
            W = W + η[sigmoid(W X+BH)XT) – sigmoid(W Xsample + BH)XsampleT]
            BV = BV + η[X - Xsample]
            BH = BH + η[sigmoid(W X+BH)) – sigmoid(W Xsample + BH)]

    """
    W_delta  = tf.multiply(learning_rate/size_bt, tf.subtract(tf.matmul(tf.transpose(x), sample(tf.sigmoid(tf.matmul(x, W) + bh)) ), tf.matmul(tf.transpose(x_sample), sample(tf.sigmoid(tf.matmul(x_sample, W) + bh)) )))
    bv_delta = tf.multiply(learning_rate/size_bt, tf.reduce_sum(tf.subtract(x, x_sample), 0, True))
    bh_delta = tf.multiply(learning_rate/size_bt, tf.reduce_sum(tf.subtract(sample(tf.sigmoid(tf.matmul(x, W) + bh)) , sample(tf.sigmoid(tf.matmul(x_sample, W) + bh)) ), 0, True))
    updated_weights = [W.assign_add(W_delta), bv.assign_add(bv_delta), bh.assign_add(bh_delta)]


    ###########-----------TRAINING THE MODEL--------------##################
    with tf.Session() as sess:
        #First, we train the model
        #initialize the variables of the model
        init = tf.global_variables_initializer()
        sess.run(init)
        #Run through all of the training data number_of_epochs times
        for epoch in tqdm(range(number_of_epochs)):
            for song in songs:
                #The songs are stored in a time x notes format. The size of each song is timesteps_in_song x 2*note_range
                #Here we reshape the songs so that each training example is a vector with number_of_timesteps x 2*note_range elements
                song = np.array(song)
                song = song[:int(np.floor(song.shape[0]//number_of_timesteps)*number_of_timesteps)]
                song = np.reshape(song, [song.shape[0]//number_of_timesteps, song.shape[1]*number_of_timesteps])
                #Train the RBM on batch_size examples at a time
                for i in range(1, len(song), batch_size): 
                    tr_x = song[i:i+batch_size]
                    sess.run(updated_weights, feed_dict={x: tr_x})     
    # Return the learnt parameters for new music generation
    return W, bv, bh