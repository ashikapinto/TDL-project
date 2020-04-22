import numpy as np
import midi_manipulation
import tensorflow as tf

#Training of the model is done now.

def generate( W, bv, bh):

    """
    We initialize the variables l_note, h_note, music_note_range for creating new music chords
    from the distribution of the emperical data learnt
    """

    l_note = midi_manipulation.lowerBound 
    h_note = midi_manipulation.upperBound
    music_note_range = h_note-l_note

    # No of time steps 
    number_of_timesteps = 10 
    # No of neurons in visible layer(input layer)
    number_of_visible = 2*music_note_range*number_of_timesteps
    # No of music files to be generated
    number_of_chords = 50


    ########----------------Helper Functions-----------------########
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

    ##Declaring x tensor place holder to sample the vectors from the distribution learnt
    
    x  = tf.placeholder(tf.float32, [None, number_of_visible], name="x") #The placeholder variable that holds our data
    
    
    #We need to generate music now......Finally :)
    #To do that we need to sample 50 vectors from the gibbs sample(the whole intelligence is now stored in the weights and the biases of the network.)
        
    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        print(sess.run(W))
        samples = sample = get_gibbs_sample(1).eval(session=sess, feed_dict={x: np.zeros((number_of_chords, number_of_visible))})
        for i in range(samples.shape[0]):
            if not any(samples[i,:]):
                continue
            #Here we reshape the vector to be time x notes, and then save the vector as a midi file
            S = np.reshape(sample[i,:], (number_of_timesteps, 2*music_note_range))
            midi_manipulation.noteStateMatrixToMidi(S, "generated_new_music/generated_chord_{}".format(i))